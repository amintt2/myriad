"""E16: run the sandbox (exec_code.py) on the code generations of every quantized file, one result set per file.

exec_code.py takes one suffix and writes the list of its models in the manifest of its result files; it refuses to
resume if that list changes. The quantized runs have one suffix per quantization (e16_quant.Quant.suffix) and a later
plan (e16-2) adds models to a quantization an earlier plan already executed, so each (model, quantization) gets its own
files: tag e16_quant.Quant.exec_tag, suffix Quant.suffix. This launcher calls exec_code.py once per file that has
generations, one after the other (the workers are CPU processes with timeouts: two sandboxes at once would make the
timeouts depend on the neighbour).

    uv run python exec_e16.py --workers 8
Writes results/codeexec_<exec_tag>_<quant>_colab_<bench>_<split>.jsonl; resumable (exec_code.py keys: id, prog).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import e16_quant
from essaim import code, data
from essaim.results import read_manifest

HERE = Path(__file__).resolve().parent


def pending(only_tags: list[str] | None = None) -> list[e16_quant.Quant]:
    """The quantized files that have code generations (every benchmark and split present), in table order. A missing
    file among the others is an error for exec_code.py, not a reason to skip: an incomplete set fails loudly."""
    return [q for q in e16_quant.QUANT
            if (not only_tags or q.tag in only_tags)
            and all(read_manifest(code.gen_path(q.model, q.suffix, b, s)) is not None
                    for b in code.BENCHES for s in data.SPLITS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tags", nargs="*", default=None, help="only these quantization tags (default: all that have generations)")
    a = ap.parse_args()
    todo = pending(a.tags)
    if not todo:
        raise SystemExit("aucune génération de code E16 à exécuter")
    for q in todo:
        print(f"exec_code.py {q.exec_tag}{q.suffix} : {q.model}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / "exec_code.py"), "--suffix", q.suffix, "--tag", q.exec_tag,
                            "--models", q.model, "--workers", str(a.workers)], cwd=HERE)
        if r.returncode != 0:
            raise SystemExit(f"exec_code.py a échoué pour {q.model}{q.suffix} (code {r.returncode})")
    print("done", flush=True)


if __name__ == "__main__":
    main()
