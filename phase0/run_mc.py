"""Collect one model's answer-letter probabilities on ARC-Challenge and MMLU-Pro.

Runs the model in-process (no network): fusion happens offline in analyze_mc.py,
so each machine (PC, Mac) can run its own models and bring back the results file.
Each pass shows the options in a different, balanced order (essaim.data.balanced_perm).

    uv run python run_mc.py --model Qwen/Qwen3-1.7B --gguf ../models/Qwen3-1.7B-Q8_0.gguf --split dev --runs 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from essaim import data
from essaim.common import FIXED_DATE, PROMPT_VERSION, resolve_revision
from essaim.results import ResultsFile, file_identity

RESULTS = Path(__file__).resolve().parent / "results"


def results_path(model: str, suffix: str, split: str) -> Path:
    return RESULTS / f"mc_{model.replace('/', '__')}{suffix}_{split}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", choices=data.SPLITS, default="dev")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--gguf", default=None, help="run through llama.cpp (all layers on the GPU) with this GGUF file")
    ap.add_argument("--suffix", default="", help="appended to the results file name, e.g. _gpu")
    ap.add_argument("--runs", type=int, default=5, help="passes per question, each with a different balanced option order")
    a = ap.parse_args()

    benches = {"arc": data.arc(a.n, split=a.split), "mmlupro": data.mmlu_pro(a.n, split=a.split)}
    rev = resolve_revision(a.model)
    if a.gguf:
        from essaim.llamacpp import LlamaCppLM
        lm = LlamaCppLM(a.model, a.gguf, revision=rev)
    else:
        from essaim.lm import LM
        lm = LM(a.model, a.device, a.dtype, a.threads, revision=rev)
    manifest = {**lm.identity(), "weights": file_identity(a.gguf) if a.gguf else None,
                "data": {b: data.dataset_identity(b) for b in benches}, "split": a.split, "n": a.n,
                "runs": a.runs, "repetition": "balanced option order (balanced_perm v2, step coprime with m)",
                "prompt": PROMPT_VERSION, "template_date": str(FIXED_DATE)}
    out = ResultsFile(results_path(a.model, a.suffix, a.split), manifest, key=("bench", "id", "run"))
    print("loaded", lm.info(), flush=True)
    for run in range(a.runs):
        for bench, items in benches.items():
            t0, k = time.perf_counter(), 0
            for it in items:
                if (bench, it["id"], run) in out.done:
                    continue
                perm = data.balanced_perm(len(it["options"]), run, a.runs)
                shown = {**it, "options": [it["options"][i] for i in perm]}
                user, letters = data.mc_prompt(shown)
                p_shown = lm.letter_probs(user, letters)
                probs = [0.0] * len(perm)
                for pos, orig in enumerate(perm):  # back to the original option order
                    probs[orig] = p_shown[pos]
                out.write({"bench": bench, "id": it["id"], "run": run, "perm": perm, "model": a.model,
                           "probs": probs, "answer": it["answer"],
                           **({"category": it["category"]} if "category" in it else {})})
                k += 1
                if k % 100 == 0:
                    print(f"run {run} {bench} {k}/{len(items)}  {(time.perf_counter() - t0) / k:.2f}s/q", flush=True)
    print("done", lm.info(), flush=True)
    out.release()


if __name__ == "__main__":
    main()
