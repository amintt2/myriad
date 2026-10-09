"""Multiple choice WITH the model's thinking mode, under a token budget.

The model reasons (sampled, temperature 0.6, a different seed per pass); if it is still
thinking when the budget runs out, the reasoning is closed for it (s1-style budget
forcing). The answer is then read as the exact probability of each letter after
"Answer:", so analyze_mc.py fuses these files like the no-thinking ones. The option
order is the original one in every pass: here the passes differ by their sampling seed.

    uv run python run_think.py --model Qwen/Qwen3-1.7B --gguf ../models/Qwen3-1.7B-Q8_0.gguf --suffix _gpu --split dev

--parallel N sends N questions at once to one llama-server with N slots (continuous batching):
on a big GPU this is far faster than one question at a time. Each question keeps its own seed;
batched decoding is not bit-identical to one-at-a-time decoding, so each row records `slots`.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from essaim import data
from essaim.common import FIXED_DATE, PROMPT_VERSION, resolve_revision
from essaim.results import ResultsFile, file_identity

RESULTS = Path(__file__).resolve().parent / "results"
INSTR = "\n\nThink it through, then answer with the letter."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--split", choices=data.SPLITS, default="dev")
    ap.add_argument("--n", type=int, default=100, help="questions per benchmark (the first n of the split)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--budget", type=int, default=768, help="maximum thinking tokens")
    ap.add_argument("--parallel", type=int, default=1, help="questions in flight (llama.cpp only)")
    a = ap.parse_args()

    benches = {"arc": data.arc(a.n, split=a.split), "mmlupro": data.mmlu_pro(a.n, split=a.split)}
    rev = resolve_revision(a.model)
    if a.gguf:
        from essaim.llamacpp import LlamaCppLM
        lm = LlamaCppLM(a.model, a.gguf, ctx=a.budget + 1536, revision=rev, parallel=a.parallel)
    else:
        if a.parallel != 1:
            raise SystemExit("--parallel needs --gguf (llama.cpp)")
        from essaim.lm import LM
        lm = LM(a.model, a.device, revision=rev)
    manifest = {**lm.identity(), "weights": file_identity(a.gguf) if a.gguf else None,
                "data": {b: data.dataset_identity(b) for b in benches}, "split": a.split, "n": a.n,
                "runs": a.runs, "budget": a.budget, "temperature": 0.6,
                "repetition": "sampling seed (original option order)",
                "prompt": PROMPT_VERSION + "+think", "template_date": str(FIXED_DATE)}
    path = RESULTS / f"mc_{a.model.replace('/', '__')}{a.suffix}_think{a.budget}_{a.split}.jsonl"
    out = ResultsFile(path, manifest, key=("bench", "id", "run"))
    todo = [(run, bench, it) for run in range(a.runs) for bench, items in benches.items() for it in items
            if (bench, it["id"], run) not in out.done]

    def one(run, bench, it):
        user, letters = data.mc_prompt(it)
        return lm.think_then_letters(user + INSTR, letters, a.budget, seed=1000 * run + 7)

    # Requests run in worker threads; only this thread writes results (in completion order).
    t0, k, ntok, nforced = time.perf_counter(), 0, 0, 0
    ex = ThreadPoolExecutor(max_workers=a.parallel)
    try:
        futs = {ex.submit(one, *job): job for job in todo}
        for fut in as_completed(futs):
            run, bench, it = futs[fut]
            g = fut.result()  # a failed request stops the run; finished rows are kept for resumption
            out.write({"bench": bench, "id": it["id"], "run": run, "model": a.model, "probs": g["probs"],
                       "answer": it["answer"], "n_tokens": g["n_tokens"], "forced": g["forced"],
                       "stop": g.get("stop"), "slots": a.parallel})
            k += 1
            ntok += g["n_tokens"]
            nforced += g["forced"]
            if k % 20 == 0:
                el = time.perf_counter() - t0
                print(f"{k}/{len(todo)}  {el / k:.2f}s/q  {ntok / el:.0f} tok/s  budget atteint {nforced}/{k}",
                      flush=True)
    except BaseException:  # do not wait for the questions still in flight: stop now, resume later
        ex.shutdown(wait=False, cancel_futures=True)
        if hasattr(lm, "close"):
            lm.close()  # stops the server: requests in flight fail at once instead of finishing
        raise
    ex.shutdown()
    print("done", lm.info(), flush=True)
    out.release()


if __name__ == "__main__":
    main()
