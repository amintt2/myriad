"""Uploadable autonomous VERIF-1 bootstrap. Explicit --run enables downloads and DEV inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

LLAMA_BASE = "https://github.com/ggml-org/llama.cpp/releases/download/b11505"
LLAMA_ARCHIVES = ("llama-b11505-bin-ubuntu-cuda-12.8-x64.tar.gz",
                  "cudart-llama-b11505-bin-ubuntu-cuda-12.8-x64.tar.gz")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def command(argv, **kwargs):
    subprocess.run(argv, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/content/verif-1"))
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"archive": str(args.archive), "sha256": digest(args.archive),
                          "status": "dry_bootstrap_no_network_no_extraction", "scope": args.scope}))
        return
    if not sys.platform.startswith("linux") or args.work.resolve() != Path("/content/verif-1"):
        raise ValueError("bootstrap is restricted to its dedicated Linux /content/verif-1 directory")
    work = args.work
    work.mkdir(exist_ok=True)
    lock = work / ".bootstrap.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        archive_hash = digest(args.archive)
        marker = work / "archive.sha256"
        if marker.exists():
            if marker.read_text().strip() != archive_hash:
                raise ValueError("existing dedicated campaign has different code/inputs")
        else:
            if any(p != lock for p in work.iterdir()):
                raise ValueError("refuse nonempty campaign directory without binding")
            with tarfile.open(args.archive) as archive:
                members = archive.getmembers()
                if any(not m.isfile() or m.name.startswith("/") or ".." in Path(m.name).parts for m in members):
                    raise ValueError("only regular relative files allowed in upload")
                archive.extractall(work, filter="data")
            marker.write_text(archive_hash + "\n")
        plan = json.loads((work / "verif1_dev_plan.json").read_text())
        for name, wanted in plan["sources"].items():
            if digest(work / name) != wanted:
                raise ValueError("uploaded source differs from frozen plan")
        if digest(Path(__file__)) != plan["sources"]["phase0/colab/verif1_bootstrap.py"]:
            raise ValueError("uploaded bootstrap differs from frozen plan")
        runtime = work / "runtime"
        runtime.mkdir(exist_ok=True)
        env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1", HF_HOME=str(runtime / "hf"),
                   UV_CACHE_DIR=str(runtime / "uv-cache"))
        uv = shutil.which("uv")
        if not uv:
            # Install into this campaign only; never alter another campaign's environment.
            command([sys.executable, "-m", "pip", "install", "--target", str(runtime / "uv"), "uv==0.11.21"])
            uv = str(runtime / "uv/bin/uv")
        command([uv, "sync", "--frozen"], cwd=work / "phase0", env=env)
        py = work / "phase0/.venv/bin/python"
        llama = runtime / "llama"
        llama.mkdir(exist_ok=True)
        if not (llama / "installed.json").exists():
            artifacts = {}
            for name in LLAMA_ARCHIVES:
                path = llama / name
                command(["curl", "--fail", "--location", "--silent", "--show-error", "--output", str(path),
                         f"{LLAMA_BASE}/{name}"])
                artifacts[name] = digest(path)
                command(["tar", "xzf", str(path), "-C", str(llama)])
            (llama / "installed.json").write_text(json.dumps(artifacts))
        servers = list(llama.rglob("llama-server"))
        if len(servers) != 1:
            raise ValueError("ambiguous llama-server binary")
        env["LLAMA_SERVER"] = str(servers[0])
        libs = sorted({str(servers[0].parent)} | {str(p.parent) for p in llama.rglob("*.so*")})
        env["LD_LIBRARY_PATH"] = ":".join(libs + ["/usr/lib64-nvidia", env.get("LD_LIBRARY_PATH", "")])
        devices = subprocess.run([str(servers[0]), "--list-devices"], env=env, capture_output=True,
                                 text=True, check=True)
        if "CUDA0" not in devices.stdout + devices.stderr:
            raise ValueError("no CUDA GPU visible")
        models = runtime / "models"
        models.mkdir(exist_ok=True)
        # A separate process uses the frozen phase0 dependency lock for Hugging Face metadata/downloads.
        command([str(py), str(Path(__file__).resolve()), "--fetch-models", str(work)], env=env)
        # The run is foreground, so logs and exit status are explicit. No cross-campaign supervisor touched.
        command([str(py), "verif1_run.py", "--bundle", str(work / "verif1_dev_bundle.json"),
                 "--plan", str(work / "verif1_dev_plan.json"), "--output", str(work / "results"),
                 "--models", str(models), "--scope", args.scope, "--run"], cwd=work / "phase0", env=env)
    finally:
        lock.unlink()


def fetch_models(work):
    """Resolve metadata to an immutable GGUF commit, requiring the already frozen E11 LFS SHA first."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    catalogue = json.loads((work / "verif1_dev_bundle.json").read_text())["catalogue"]
    records, models = {}, work / "runtime/models"
    for family, entry in catalogue.items():
        target = models / entry["gguf"]
        if not target.exists():
            info = HfApi().model_info(entry["gguf_repository"], files_metadata=True)
            matching = [f for f in info.siblings if f.rfilename == entry["gguf"]]
            if (len(matching) != 1 or not matching[0].lfs
                    or matching[0].lfs.sha256 != entry["weights"]["sha256"]):
                raise ValueError("catalogue GGUF no longer matches frozen E11 SHA; no floating replacement")
            downloaded = hf_hub_download(entry["gguf_repository"], entry["gguf"], revision=info.sha)
            shutil.copyfile(downloaded, target)
            records[family] = {"gguf_commit": info.sha, "gguf_sha256": digest(target)}
        if target.stat().st_size != entry["weights"]["size"] or digest(target) != entry["weights"]["sha256"]:
            raise ValueError("GGUF content hash/size mismatch")
        # Preserve pinned tokenizer/config provenance even though ChatServer uses the GGUF chat template.
        snapshot_download(entry["model"], revision=entry["revision"],
                          allow_patterns=["*.json", "*.jinja", "*.model", "README.md", "LICENSE*"])
    destination = work / "runtime/gguf_downloads.json"
    previous = json.loads(destination.read_text()) if destination.exists() else {}
    destination.write_text(json.dumps({**previous, **records}, sort_keys=True, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--fetch-models":
        fetch_models(Path(sys.argv[2]))
    else:
        main()
