"""E18 generation and checking: one prover writes Lean 4 proofs of miniF2F, Lean checks each one (docs/14_e18_lean.md).

Per problem and prover: `--samples` independent attempts (sample s, round 0, seed fixed by (problem, s)). The first
`--chains` samples are also revision chains: when the attempt fails, Lean's error messages are sent back to the
model with its code (essaim/lean.feedback), for at most `--revisions` more rounds; a chain stops at the first
verified proof. Every attempt is checked here, on this machine, by Lean (essaim/lean_runner.py) with the toolchain
and Mathlib installed by e18_setup.py: a proof is verified only by Lean, never by a peer or a vote.

The model is served by llama-server with the chat template of its GGUF (--jinja), as in E4 (run_solo.ChatServer),
with the sampling settings and prompt of the prover's own model card (e18_lean.PROVERS).

    uv run python run_e18.py --model Goedel-LM/Goedel-Prover-V2-8B --gguf ../models/X.gguf --suffix _colab
Writes results/e18_<model><suffix>_minif2f_<split>.jsonl (+ manifest), resumable, key (id, sample, round).
"""
from __future__ import annotations

import argparse
import hashlib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import e18_lean as E
from essaim import lean
from essaim.common import resolve_revision
from essaim.lean_runner import Checker, load_env
from essaim.results import ResultsFile, file_identity, read_rows
from run_solo import ChatServer


def seed(item_id: str, sample: int, rnd: int) -> int:
    return int.from_bytes(hashlib.sha256(f"e18|{item_id}|{sample}|{rnd}".encode()).digest()[:4], "big") % (2**31 - 1)


class PromptTooLong(Exception):
    pass


def chat(http, messages: list[dict], spec: E.Prover, max_tokens: int, s: int) -> dict:
    t0 = time.perf_counter()
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": spec.temperature, "top_p": spec.top_p,
            "top_k": spec.top_k, "seed": s}
    if spec.thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": spec.thinking}
    r = http.post("/v1/chat/completions", json=body)
    if r.status_code == 400 and "context" in r.text.lower():
        raise PromptTooLong(r.text[:300])
    if r.status_code != 200:
        raise RuntimeError(f"llama-server {r.status_code}: {r.text[:500]}")
    j = r.json()
    ch, usage = j["choices"][0], j.get("usage") or {}
    text = ch["message"].get("content") or ""
    reasoning = ch["message"].get("reasoning_content") or ""
    return {"text": text, "reasoning_chars": len(reasoning),
            # kept only when the answer has no Lean block (cut by the token limit): see e18_lean.candidate
            "reasoning_block": lean.last_code_block(reasoning) if lean.last_code_block(text) is None else None,
            "finish": ch.get("finish_reason"), "n_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"), "ms": round((time.perf_counter() - t0) * 1000)}


def attempt(srv, checker: Checker, spec: E.Prover, it: dict, sample: int, rnd: int, prev: dict | None,
            max_tokens: int) -> dict:
    """One generation and its Lean check. `prev`: the failed attempt this one revises (round >= 1)."""
    row = {"id": it["id"], "sample": sample, "round": rnd, "model": spec.model, "verified": False, "lean": None,
           "feedback": None}
    try:
        g = chat(srv.http, E.messages(spec, it, prev), spec, max_tokens, seed(it["id"], sample, rnd))
    except PromptTooLong as e:  # a revision whose prompt does not fit the slot: a failed attempt, no tokens
        row.update({"text": "", "reasoning_chars": 0, "reasoning_block": None, "finish": "prompt_too_long",
                    "n_tokens": 0, "prompt_tokens": None, "ms": 0, "build": "prompt_too_long", "reason": str(e),
                    "statement_rewritten": None, "code": None, "status": "prompt_too_long",
                    "feedback": "The previous answer was too long to be revised."})
        return row
    built = lean.build(it, E.candidate(g["text"], g["reasoning_block"]))
    row.update({**g, "build": built["status"], "reason": built.get("reason"),
                "statement_rewritten": built.get("statement_rewritten"), "code": built.get("code"),
                "status": built["status"]})
    if built["status"] != "ok":
        row["feedback"] = E.build_feedback(built)
        return row
    run = checker.check(lean.with_axioms(built["code"], built["name"]))
    rec = E.lean_record(run, built["axioms_line"])
    v = lean.verdict(rec, built["name"], built["axioms_line"])
    row.update({"status": v["status"], "verified": v["verified"], "axioms": v["axioms"], "lean": rec,
                "axioms_line": built["axioms_line"]})
    if not v["verified"]:
        row["feedback"] = (lean.feedback(rec, built["code"], E.FEEDBACK_MAX_ERRORS, E.FEEDBACK_MAX_CHARS)
                           if v["status"] == "lean_error" else E.status_feedback(v))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(E.PROVERS))
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--splits", nargs="+", default=list(E.SPLITS), choices=list(E.SPLITS))
    ap.add_argument("--n", type=int, default=None, help="first n problems of each split (smoke test)")
    ap.add_argument("--samples", type=int, default=E.SAMPLES)
    ap.add_argument("--chains", type=int, default=E.CHAINS)
    ap.add_argument("--revisions", type=int, default=E.REVISIONS)
    ap.add_argument("--max-tokens", type=int, default=E.MAX_TOKENS)
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--lean-workers", type=int, default=4)
    ap.add_argument("--oracle-suffix", default="_colab", help="suffix of the oracle files written by e18_setup.py")
    a = ap.parse_args()
    if not 0 <= a.chains <= a.samples:
        raise SystemExit("--chains doit être entre 0 et --samples")

    spec = E.PROVERS[a.model]
    env = load_env()
    E.check_lean_env(env)
    checker = Checker(env, timeout_s=E.CHECK_TIMEOUT_S, mem_mib=E.CHECK_MEM_MIB, workers=a.lean_workers)
    rev = resolve_revision(a.model)
    ctx = a.max_tokens + E.PROMPT_BUDGET[a.chains > 0 and a.revisions > 0]
    srv = ChatServer(a.gguf, a.parallel, ctx)
    try:
        weights = file_identity(a.gguf)
        if (weights["sha256"], weights["size"]) != (spec.sha256, spec.size):
            raise SystemExit(f"{a.gguf} ne correspond pas à l'empreinte épinglée dans e18_lean.PROVERS")
        for split in a.splits:
            items = E.usable(split, a.oracle_suffix, a.n)
            manifest = {"model": a.model, "revision": rev, "backend": "llama.cpp (--jinja, OpenAI chat)",
                        "gguf": Path(a.gguf).name, "weights": weights, "engine": srv.version,
                        "data": E.dataset_identity(), "bench": E.BENCH, "split": split, "n": len(items),
                        "problems": hashlib.sha256("\n".join(it["id"] for it in items).encode()).hexdigest(),
                        "prompt": E.PROMPT_VERSION, "style": spec.style, "thinking": spec.thinking,
                        "sampling": {"temperature": spec.temperature, "top_p": spec.top_p, "top_k": spec.top_k,
                                     "seed": "sha256(e18|id|sample|round)"},
                        "max_tokens": a.max_tokens, "ctx_per_slot": ctx, "samples": a.samples, "chains": a.chains,
                        "revisions": a.revisions, "lean": E.lean_identity(env), "checker": E.checker_identity(checker),
                        "screen": E.SCREEN_VERSION}
            out = ResultsFile(E.result_path(a.model, a.suffix, split), manifest, key=("id", "sample", "round"))
            old = {(r["id"], r["sample"], r["round"]): r for r in read_rows(out.path)}
            lock = threading.Lock()
            count = [0]

            def chain(it: dict, sample: int) -> None:
                last = 0 if sample >= a.chains else a.revisions
                prev = None
                for rnd in range(last + 1):
                    row = old.get((it["id"], sample, rnd))
                    if row is None:
                        row = attempt(srv, checker, spec, it, sample, rnd, prev, a.max_tokens)
                        with lock:
                            out.write(row)
                            count[0] += 1
                    if row["verified"]:
                        return
                    prev = row

            t0 = time.perf_counter()
            ex = ThreadPoolExecutor(max_workers=a.parallel + a.lean_workers)
            try:
                futs = [ex.submit(chain, it, s) for it in items for s in range(a.samples)]
                for fut in as_completed(futs):
                    fut.result()
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)
                srv.close()
                raise
            ex.shutdown()
            out.release()
            wall = time.perf_counter() - t0
            print(f"{split}: {count[0]} nouvelles tentatives en {wall:.0f} s", flush=True)
            # Wall time of this run, for the cost calibration (a resumed run only measures its own part).
            timing = ResultsFile(E.wall_path(a.model, a.suffix, split),
                                 {"kind": "e18 wall time", "model": a.model, "split": split}, key=("run",))
            timing.write({"run": uuid.uuid4().hex, "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                          "seconds": round(wall, 1), "new_attempts": count[0], "parallel": a.parallel,
                          "lean_workers": a.lean_workers})
            timing.release()
    finally:
        srv.close()
    print("done", a.model, flush=True)


if __name__ == "__main__":
    main()
