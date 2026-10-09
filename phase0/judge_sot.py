"""E10: pairwise LLM-as-judge, parallel answer (skeleton + expansions by several peers) against
single-peer answers, on the same prompts.

For each prompt and each single model X of --baselines, the judge sees the two answers in BOTH display
orders (parallel answer first, then second) and gives a one-letter verdict, A, B or C (tie), forced by a
grammar; the raw probabilities of the three letters (top log-probabilities of that one token, before the
grammar) are recorded too. Identical answers (a skeleton fallback against the outline model itself) are
recorded as ties without asking the judge. Thinking is off: the verdict is the first token.

    uv run python judge_sot.py --judge Qwen/Qwen3.8-27B --gguf ../models/Qwen3.8-27B-Q8_0.gguf --tag sot1 \
        --outline-model Qwen/Qwen3.5-4B --peers <P peers> --baselines <P peers> --suffix _colab --parallel 4
Writes results/sot_judge_<tag>_<judge><suffix>_<split>.jsonl (+ manifest), resumable.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from essaim import data, sot
from essaim.common import resolve_revision
from essaim.results import ResultsFile, file_identity
from run_solo import ChatServer

ORDERS = {"sot_first": "A", "sot_second": "B"}  # display order -> letter of the parallel answer


def judge_one(srv: ChatServer, question: str, sot_text: str, other: str, order: str) -> dict:
    a, b = (sot_text, other) if order == "sot_first" else (other, sot_text)
    g = sot.chat(srv.http, [{"role": "user", "content": sot.judge_prompt(question, a, b)}], 1,
                 grammar=sot.JUDGE_GRAMMAR, logprobs=True, top_logprobs=sot.JUDGE_TOP)
    lp = g.pop("logprobs") or []
    verdict = g["text"].strip() or None
    return {"verdict": verdict if verdict in ("A", "B", "C") else None, "raw": g["text"][:20],
            "probs": sot.letter_probs(lp[0].get("top_logprobs") if lp else None),
            "prompt_tokens": g["prompt_tokens"], "ms": g["ms"], "finish": g["finish"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outline-model", required=True)
    ap.add_argument("--peers", nargs="+", required=True)
    ap.add_argument("--baselines", nargs="+", required=True, help="single models the parallel answer is compared with")
    ap.add_argument("--splits", nargs="+", default=list(data.MT_SPLITS), choices=list(data.MT_SPLITS))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--ctx-per-slot", type=int, default=6144)
    a = ap.parse_args()

    rev = resolve_revision(a.judge)
    srv = ChatServer(a.gguf, a.parallel, a.ctx_per_slot)
    try:
        weights = file_identity(a.gguf)
        for split in a.splits:
            items = data.mt_bench(a.n, split=split)
            ids = [it["id"] for it in items]
            par = sot.load_sot(a.tag, a.outline_model, a.peers, a.suffix, split, ids)
            singles = {x: {r["id"]: r["text"] for r in sot.complete(sot.base_path(x, a.suffix, split), ids, what="baseline")}
                       for x in a.baselines}
            inputs = {i: {"sot": par[i]["text"], **{x: singles[x][i] for x in a.baselines}} for i in ids}
            manifest = {"judge": a.judge, "revision": rev, "backend": "llama.cpp (--jinja, OpenAI chat)",
                        "gguf": Path(a.gguf).name, "weights": weights, "engine": srv.version,
                        "data": data.dataset_identity("mtbench"), "split": split, "n": len(items),
                        "judge_prompt": sot.JUDGE_VERSION, "prompt": sot.PROMPT_VERSION, "tag": a.tag,
                        "outline_model": a.outline_model, "peers": a.peers, "baselines": a.baselines,
                        "inputs_sha256": sot.canonical_sha(inputs), "grammar": sot.JUDGE_GRAMMAR,
                        "top_logprobs": sot.JUDGE_TOP, "temperature": 0.0, "thinking": False,
                        "ctx_per_slot": a.ctx_per_slot}
            out = ResultsFile(sot.judge_path(a.tag, a.judge, a.suffix, split), manifest, key=("id", "vs", "order"))
            t0, k = time.perf_counter(), 0
            ex = ThreadPoolExecutor(max_workers=a.parallel)
            try:
                futs = {}
                for it in items:
                    for x in a.baselines:
                        for order, sot_is in ORDERS.items():
                            row = {"id": it["id"], "category": it["category"], "vs": x, "order": order,
                                   "sot_is": sot_is, "fallback": par[it["id"]]["fallback"]}
                            if (it["id"], x, order) in out.done:
                                continue
                            if inputs[it["id"]]["sot"] == inputs[it["id"]][x]:
                                out.write({**row, "verdict": "C", "identical": True, "probs": None})
                                k += 1
                                continue
                            futs[ex.submit(judge_one, srv, it["question"], inputs[it["id"]]["sot"],
                                           inputs[it["id"]][x], order)] = row
                for fut in as_completed(futs):
                    g = fut.result()
                    if g["verdict"] is None:
                        raise RuntimeError(f"verdict illisible {g['raw']!r} pour {futs[fut]}")
                    out.write({**futs[fut], **g, "identical": False})
                    k += 1
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                out.release()
            ex.shutdown()
            print(f"juge {split}: {k} nouveaux verdicts en {time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        srv.close()
    print("done", a.judge, flush=True)


if __name__ == "__main__":
    main()
