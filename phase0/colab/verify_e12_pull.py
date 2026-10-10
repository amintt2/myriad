"""Require every final E12 artifact from this pull, never adopt stale local results as recovery evidence."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_jobs import SOLO
from aa_timing import coverage, header, parse


def verify(stage, expected_provenance):
    root = Path(stage)
    provenance = json.loads((root / "e12_campaign_sources.json").read_text())
    if (provenance.get("schema") != 1 or provenance.get("plan") != "aa-1" or not provenance.get("sources")
            or provenance.get("benches") not in (["scicode"], ["gpqa", "scicode"])):
        raise ValueError("missing campaign source provenance")
    if provenance != json.loads(Path(expected_provenance).read_text()):
        raise ValueError("recovered sources do not match the uploaded source manifest")
    names = [("sciexec_oracle_colab_scicode_dev.jsonl", 50)]
    for model in SOLO:
        slug = model.replace("/", "__")
        for split, n in (("dev", 50), ("test", 288)):
            names.extend((f"{kind}_{slug}_colab_scicode_{split}.jsonl", n) for kind in ("aa", "sciexec"))
        gpqa = root / f"aa_{slug}_colab_gpqa_all.jsonl"
        if "gpqa" in provenance["benches"]:
            names.append((gpqa.name, 198))
    for name, n in names:
        raw = (root / name).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in raw.splitlines()]
        if not raw.endswith("\n") or len(rows) != n or len({r["id"] for r in rows}) != n:
            raise ValueError(f"incomplete final artifact: {name}")
        meta = json.loads((root / (name + ".meta.json")).read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or not meta:
            raise ValueError(f"invalid manifest: {name}")
        if name.startswith("aa_"):
            expected = header(name, meta) if meta.get("measurements") == "e12-wall-v2" else None
            timing = parse((root / (name + ".timing.jsonl")).read_bytes(), expected)
            observations = [r for r in timing if r.get("kind") != "timing_header"]
            if not observations:
                raise ValueError(f"invalid timing observations: {name}")
            if expected:
                coverage(timing, root / name)
                if not any(r.get("kind") == "batch" for r in observations):
                    raise ValueError(f"missing completed batch timing: {name}")


if __name__ == "__main__":
    verify(*sys.argv[1:])
