"""E16 supervision checks and sealed recovery on the VM; never run on the coordinator."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

MAX_FILE = 64 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def complete(plan):
    import e16_quant as q
    from essaim import code, data
    from essaim.results import read_manifest, read_rows
    from run_solo import LOADERS
    root = Path("results")
    files = []

    def checked(path, expected, key, quant=None):
        raw = path.read_bytes()
        # read_rows permits an incomplete tail for checkpoints; final artifacts do not.
        for line in raw.splitlines():
            json.loads(line)
        rows = read_rows(path, key=key)
        if {tuple(r[k] for k in key) for r in rows} != expected:
            raise ValueError("incomplete artifact " + path.name)
        manifest = read_manifest(path)
        if not isinstance(manifest, dict):
            raise ValueError("missing manifest")
        if quant and manifest.get("weights", {}).get("sha256") != quant.sha256:
            raise ValueError("weights differ from pinned quantization")
        files.extend((path.name, path.name + ".meta.json"))
        return rows, manifest

    for quant in q.QUANT:
        for bench, loader in LOADERS.items():
            for split in data.SPLITS:
                items = loader(data.PART[bench], split=split)
                path = root / f"solo_{quant.model.replace('/', '__')}{quant.suffix}_{bench}_{split}.jsonl"
                rows, man = checked(path, {(it["id"],) for it in items}, ("id",), quant)
                if (man.get("n") != len(items) or man.get("data") != data.dataset_identity(bench)
                        or man.get("model") != quant.model or man.get("split") != split):
                    raise ValueError("generation identity differs")
        if plan != "e16-1" or (quant.model, quant.tag) not in q.CODE_BOUNDS:
            continue
        for bench, loader in data.CODE_LOADERS.items():
            for split in data.SPLITS:
                items = loader(data.PART[bench], split=split)
                rows, man = checked(code.gen_path(quant.model, quant.suffix, bench, split),
                                    {(it["id"], s) for it in items for s in range(5)}, ("id", "sample"), quant)
                if (man.get("samples") != 4 or man.get("n") != len(items)
                        or man.get("data") != data.dataset_identity(bench)):
                    raise ValueError("code generation identity differs")
                expected = {(it["id"], "reference") for it in items}
                entries = {it["id"]: code.entry_point(bench, it) for it in items}
                expected.update((r["id"], code.prog_id(code.extract_code(r["text"], entries[r["id"]])[0]))
                                for r in rows)
                _, execution = checked(code.exec_path(quant.exec_tag, quant.suffix, bench, split),
                                       expected, ("id", "prog"))
                if execution.get("models") != [quant.model] or execution.get("data") != man["data"]:
                    raise ValueError("execution identity differs")
    labels = {q.label(it.model, it.tag) for it in q.QUANT} | {q.label(m, "Q8_0") for m in q.SPEED_Q8}
    speed, man = checked(root / "solo_e16speed_colab_bench_all.jsonl", {(lab,) for lab in labels}, ("label",))
    if man.get("alone_on_gpu") is not True or "A100" not in str(man.get("gpu")):
        raise ValueError("speed GPU identity differs")
    pins = {q.label(it.model, it.tag): it.sha256 for it in q.QUANT}
    if any(r.get("weights", {}).get("sha256") != pins[r["label"]] for r in speed if r["label"] in pins):
        raise ValueError("speed weight identity differs")
    return files


def seal(request, final=False):
    os.chdir("/content/dllm/phase0")
    root = Path("results")
    boot = json.loads((root / "colab_bootstrap_status.json").read_text())
    if (boot.get("campaign_id") != request["campaign_id"]
            or ("plan" in boot and boot["plan"] != request["plan"])):
        raise PermissionError("foreign campaign")
    if "plan" not in boot:
        # Historical bootstrap failures omit plan; they do not revoke allocation ownership.
        if not str(boot.get("step", "")).startswith("ÉCHEC") or final:
            raise ValueError("bootstrap initialization incomplete")
    provenance = json.loads((root / "e16_sources.json").read_text())
    if provenance.get("campaign_id") != request["campaign_id"] or provenance.get("plan") != request["plan"]:
        raise ValueError("source provenance identity differs")
    for rel, hashed in provenance["sources"].items():
        path = Path("/content/dllm") / rel
        if path.is_symlink() or digest(path) != hashed:
            raise ValueError("active source differs from prepared archive")
    for name, hashed in provenance["references"].items():
        if digest(root / name) != hashed:
            raise ValueError("reference changed on VM")
    if final:
        expected = complete(request["plan"])
        with (root / "e16_analysis.log").open("w") as log:
            subprocess.run(["uv", "run", "--group", "paper", "python", "analyze_e16.py",
                            *(["--no-code"] if request["plan"] == "e16-mc" else [])],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        summary = json.loads((root / "e16_summary.json").read_text())
        if "choices" not in summary or (request["plan"] == "e16-1" and "code" not in summary):
            raise ValueError("analysis did not produce requested results")
        expected += ["e16_summary.json", "e16_report.md", "e16_analysis.log"]
    else:
        expected = [p.name for p in root.iterdir() if p.is_file() and
                    (p.name.startswith(("solo_", "code_", "codeexec_", "colab_", "e16_"))) and
                    p.suffix in (".jsonl", ".json", ".md", ".log") and
                    p.name not in provenance["references"]]
    expected += ["e16_sources.json", "colab_status.json", "colab_bootstrap_status.json", "colab_launcher.log"]
    dest = Path("/content") / ("e16-recovery-" + request["nonce"])
    dest.mkdir()
    files, total = {}, 0
    for name in sorted(set(expected)):
        src = root / name
        if src.is_symlink() or not src.is_file():
            if final:
                raise ValueError("missing final artifact")
            continue
        size = src.stat().st_size
        total += size
        if size > MAX_FILE or total > MAX_TOTAL:
            raise ValueError("recovery size exceeds bounds")
        shutil.copyfile(src, dest / name)
        files[name] = {"size": (dest / name).stat().st_size, "sha256": digest(dest / name)}
    return {**request, "schema": 1, "complete": final, "files": files}
