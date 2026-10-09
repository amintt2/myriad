"""Convert a downloaded Hugging Face model to GGUF (Q8_0 by default) for llama.cpp.

    uv run python make_gguf.py Qwen/Qwen3-1.7B [--outtype q8_0]

Needs the llama.cpp source checkout in tools/llama.cpp-src (converter script).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HUB = Path.home() / ".cache" / "huggingface" / "hub"


def snapshot(model_id: str) -> Path:
    snaps = sorted((HUB / f"models--{model_id.replace('/', '--')}" / "snapshots").iterdir(), key=lambda p: p.stat().st_mtime)
    return snaps[-1]


def check_gguf(path: Path) -> str | None:
    """None if the file is a GGUF whose every tensor lies entirely inside the file."""
    sys.path.insert(0, str(ROOT / "tools" / "llama.cpp-src" / "gguf-py"))
    try:
        from gguf import GGUFReader
        r = GGUFReader(str(path))
        size = path.stat().st_size
        end = max((int(t.data_offset) + int(t.n_bytes) for t in r.tensors), default=0)
        if not r.tensors:
            return "no tensors"
        if end > size:
            return f"tensors end at {end} bytes but the file has {size}"
    except Exception as e:  # unreadable header or metadata
        return f"{type(e).__name__}: {e}"
    return None


def gguf_path(model_id: str, outtype: str = "q8_0") -> Path:
    return ROOT / "models" / f"{model_id.split('/')[-1]}-{outtype.upper()}.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--outtype", default="q8_0")
    a = ap.parse_args()
    out = gguf_path(a.model, a.outtype)
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        # New conversions are published by an atomic rename; older files are checked anyway.
        problem = check_gguf(out)
        if problem:
            raise SystemExit(f"{out} is not a complete GGUF ({problem}): remove it and convert again")
        print("exists:", out)
        return
    tmp = out.with_name(out.name + ".part")
    tmp.unlink(missing_ok=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "llama.cpp-src" / "convert_hf_to_gguf.py"),
                    str(snapshot(a.model)), "--outtype", a.outtype, "--outfile", str(tmp)], check=True)
    problem = check_gguf(tmp)
    if problem:
        raise SystemExit(f"{tmp} is not a complete GGUF ({problem})")
    os.replace(tmp, out)
    print("written:", out)


if __name__ == "__main__":
    main()
