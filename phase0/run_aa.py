"""E12 generation: every model answers GPQA Diamond and writes the SciCode sub-problems, alone, greedy.

Like E4 (run_solo.py): one model at a time on the GPU, served by llama-server (--jinja, OpenAI chat endpoint,
thinking off). The swarm decisions and the grading are done offline (analyze_e12.py, exec_scicode.py).

  gpqa     198 questions, 4 options in a fixed balanced order (essaim/gpqa.py), the E4 multiple-choice prompt;
           the result rows hold the model output and the letters, never the question text.
  scicode  the dev (15 problems) and test (65) splits; each problem is a chain of sub-problems written one
           after the other, each seeing the model's own earlier code (essaim/scicode.py). Problems run in
           parallel, the steps of a problem in order.

    uv run python run_aa.py --model Qwen/Qwen3.5-4B --gguf ../models/Qwen3.5-4B-Q8_0.gguf --suffix _colab
Writes results/aa_<model><suffix>_<bench>_<split>.jsonl (+ manifest), resumable. GPQA files must never be
published (tools/export_public.py leaves them out).
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from essaim import answers, data, gpqa, scicode, sot
from essaim.common import resolve_revision
from essaim.results import ResultsFile, file_identity, read_rows
from run_solo import ChatServer, PROMPT_VERSION as MC_PROMPT, prompt as mc_prompt

RESULTS = Path(__file__).resolve().parent / "results"
BENCHES = ("gpqa", "scicode")
CTX_PER_SLOT = 8192  # SciCode prompts carry the earlier steps; 6 slots of this size = 16 slots of E4's 3072
MAX_TOKENS = {"gpqa": 1024, "scicode": 2048}
SEED = 7


def path_for(model: str, suffix: str, bench: str, split: str) -> Path:
    return RESULTS / f"aa_{model.replace('/', '__')}{suffix}_{bench}_{split}.jsonl"


def base_manifest(a, rev, srv, weights, bench, split, n):
    return {"model": a.model, "revision": rev, "backend": "llama.cpp (--jinja, OpenAI chat)",
            "gguf": Path(a.gguf).name, "weights": weights, "engine": srv.version, "bench": bench, "split": split,
            "n": n, "max_tokens": MAX_TOKENS[bench], "temperature": 0.0, "seed": SEED, "thinking": False,
            "ctx_per_slot": CTX_PER_SLOT}


def run_gpqa(a, srv, rev, weights):
    items = gpqa.diamond()
    man = {**base_manifest(a, rev, srv, weights, "gpqa", "all", len(items)), "data": data.dataset_identity("gpqa"),
           "prompt": MC_PROMPT, "logprobs": False}
    out = ResultsFile(path_for(a.model, a.suffix, "gpqa", "all"), man, key=("id",))
    todo = [it for it in items[: a.n or None] if (it["id"],) not in out.done]
    t0, k = time.perf_counter(), 0
    ex = ThreadPoolExecutor(max_workers=a.parallel)
    try:
        futs = {ex.submit(srv.chat, mc_prompt("gpqa", it), MAX_TOKENS["gpqa"], SEED): it for it in todo}
        for fut in as_completed(futs):
            it, g = futs[fut], fut.result()
            out.write({"id": it["id"], "model": a.model, "bench": "gpqa", "text": g["text"],
                       "answer": answers.extract("gpqa", g["text"], it, g["finish"] == "stop"),
                       "gold": answers.gold("gpqa", it), "finish": g["finish"], "n_tokens": g["n_tokens"],
                       "ms": g["ms"], "reasoning": bool(g["reasoning"])})
            k += 1
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        out.release()
        raise
    ex.shutdown()
    out.release()
    print(f"gpqa: {k} nouvelles réponses en {time.perf_counter() - t0:.0f} s", flush=True)


def solve_problem(srv, a, problem, out, lock, have: dict) -> int:
    """All steps of one problem, in order, each prompted with the code kept for the earlier ones."""
    chain, new = [], 0
    for k, step in enumerate(problem["steps"]):
        rid = scicode.row_id(step)
        if scicode.is_skipped(problem, step):
            chain.append(scicode.chain_code(problem, k, None))
            continue
        if rid in have:  # resumed: the chain is rebuilt from the stored text with the current extractor
            chain.append(scicode.chain_code(problem, k, have[rid]["text"]))
            continue
        try:
            g = sot.chat(srv.http, [{"role": "user", "content": scicode.prompt(problem, k, chain, not a.no_background)}],
                         MAX_TOKENS["scicode"])
        except RuntimeError as e:  # e.g. the prompt does not fit the context: an empty answer, recorded as such
            g = {"text": "", "finish": "error", "n_tokens": 0, "prompt_tokens": None, "ms": 0, "reasoning": False,
                 "error": str(e)[:300]}
        name = scicode.def_name(step["header"])
        code_text, status = scicode.extract(g["text"], name)
        row = {"id": rid, "problem": problem["id"], "step": scicode.step_number(step), "model": a.model,
               "bench": "scicode", "text": g["text"], "extract": status, "finish": g["finish"],
               "n_tokens": g["n_tokens"], "prompt_tokens": g["prompt_tokens"], "ms": g["ms"],
               "reasoning": bool(g["reasoning"])}
        if g.get("error"):
            row["error"] = g["error"]
        with lock:
            out.write(row)
        chain.append(scicode.chain_code(problem, k, g["text"]))
        new += 1
    return new


def run_scicode(a, srv, rev, weights):
    for split in a.splits:
        probs = scicode.problems(split)[: a.n or None]
        man = {**base_manifest(a, rev, srv, weights, "scicode", split, len(probs)), "data": data.dataset_identity("scicode"),
               "n_steps": sum(len(p["steps"]) for p in probs), "prompt": scicode.PROMPT_VERSION,
               "background": not a.no_background, "official_commit": scicode.OFFICIAL_COMMIT,
               "skipped_code": scicode.SKIPPED_SHA256, "skipped_steps": sorted(f"{p}.{s}" for p, s in scicode.SKIPPED)}
        path = path_for(a.model, a.suffix, "scicode", split)
        out = ResultsFile(path, man, key=("id",))
        have = {r["id"]: r for r in read_rows(path, key=("id",))}
        lock, t0, k = threading.Lock(), time.perf_counter(), 0
        ex = ThreadPoolExecutor(max_workers=a.parallel)
        try:
            futs = [ex.submit(solve_problem, srv, a, p, out, lock, have) for p in probs]
            for fut in as_completed(futs):
                k += fut.result()
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            out.release()
            raise
        ex.shutdown()
        out.release()
        print(f"scicode {split}: {k} nouveaux sous-problèmes en {time.perf_counter() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--benches", nargs="+", default=list(BENCHES), choices=BENCHES)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"], help="scicode only")
    ap.add_argument("--n", type=int, default=None, help="first n questions (gpqa) or problems (scicode); smoke tests")
    ap.add_argument("--no-background", action="store_true", help="scicode: without the scientists' background text")
    ap.add_argument("--parallel", type=int, default=6)
    a = ap.parse_args()

    rev = resolve_revision(a.model)
    srv = ChatServer(a.gguf, a.parallel, CTX_PER_SLOT)
    try:
        weights = file_identity(a.gguf)
        for bench in a.benches:
            (run_gpqa if bench == "gpqa" else run_scicode)(a, srv, rev, weights)
    finally:
        srv.close()
    print("done", a.model, flush=True)


if __name__ == "__main__":
    main()
