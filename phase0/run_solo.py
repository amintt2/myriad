"""E4: every model answers every question alone, once (greedy), in batches on one GPU.

All swarm decisions (votes over any subset of models, certificates, scaling curves) are then computed
offline from these answers (analyze_e4.py): one round trip per request is exactly what the app does,
so the peers' independent answers are all that a vote needs.

The model is served by llama-server with the chat template embedded in its GGUF (--jinja) through
its OpenAI-compatible endpoint: the same path as the app (app/myriad/engine.py). Thinking is disabled
(chat_template_kwargs enable_thinking=false) for the families that support it.

    uv run python run_solo.py --model Qwen/Qwen3-1.7B --gguf ../models/Qwen3-1.7B-Q8_0.gguf --suffix _colab \
        --benches gsm8k math500 arc mmlupro --splits dev test --parallel 16
Writes results/solo_<model><suffix>_<bench>_<split>.jsonl (+ manifest), resumable.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import secrets
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from essaim import answers, data
from essaim.llamacpp import SERVER, STARTUP_S, free_port
from essaim.common import resolve_revision
from essaim.results import ResultsFile, file_identity

RESULTS = Path(__file__).resolve().parent / "results"
PROMPT_VERSION = "solo-v1"
LOADERS = {"gsm8k": data.gsm8k, "math500": data.math500, "arc": data.arc, "mmlupro": data.mmlu_pro}
MAX_TOKENS = {"gsm8k": 1024, "math500": 2048, "arc": 1024, "mmlupro": 1024}
CTX_PER_SLOT = 3072


def prompt(bench: str, item: dict) -> str:
    if bench == "gsm8k":
        return (f"{item['question']}\n\nSolve it step by step, briefly. Finish with the sentence: "
                "\"The answer is N.\" where N is a number.")
    if bench == "math500":
        return f"{item['question']}\n\nSolve the problem step by step, briefly. Put the final answer in \\boxed{{}}."
    letters = [chr(65 + i) for i in range(len(item["options"]))]
    body = "\n".join(f"{L}. {o}" for L, o in zip(letters, item["options"]))
    return (f"{item['question']}\n\n{body}\n\nThink step by step, briefly, then finish with the sentence: "
            f"\"The answer is (X).\" where X is one of {letters[0]}-{letters[-1]}.")


class ChatServer:
    """Our own llama-server (--jinja, N slots with one KV cache each), checked to serve our GGUF."""

    def __init__(self, gguf: str, parallel: int, ctx_per_slot: int = CTX_PER_SLOT):
        self.gguf, self.parallel, self.ctx_per_slot = gguf, parallel, ctx_per_slot
        v = subprocess.run([str(SERVER), "--version"], capture_output=True, text=True, timeout=60)
        self.version = " | ".join(l.strip() for l in (v.stdout + v.stderr).splitlines()
                                  if "version" in l.lower() or "built with" in l.lower())
        for _ in range(5):
            try:
                if self._start():
                    return
            except BaseException:  # never leave a server holding GPU memory behind
                self.close()
                raise
            self.close()
        raise RuntimeError(f"llama-server could not start, see {self.log_path}")

    def _start(self) -> bool:
        self.port, key = free_port(), secrets.token_hex(16)
        logs = Path(self.gguf).parent / "server-logs"
        logs.mkdir(exist_ok=True)
        self.log_path = logs / f"{Path(self.gguf).stem}.solo.{os.getpid()}.{self.port}.log"
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [str(SERVER), "-m", self.gguf, "--port", str(self.port), "--host", "127.0.0.1", "-ngl", "999",
             "-c", str(self.ctx_per_slot * self.parallel), "-np", str(self.parallel), "--no-kv-unified", "--jinja",
             "--no-webui", "--api-key", key], stdout=self._log, stderr=subprocess.STDOUT)
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=900,
                                 headers={"Authorization": f"Bearer {key}"})
        deadline = time.monotonic() + STARTUP_S
        while True:
            if self.proc.poll() is not None:
                return False
            try:
                if self.http.get("/health", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"llama-server not ready, see {self.log_path}")
            time.sleep(0.5)
        r = self.http.get("/props", timeout=10)
        served = str(r.json().get("model_path", "")) if r.status_code == 200 else ""
        return os.path.normcase(os.path.abspath(served)) == os.path.normcase(os.path.abspath(self.gguf))

    def chat(self, user: str, max_tokens: int, seed: int) -> dict:
        t0 = time.perf_counter()
        r = self.http.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": 0.0,
            "seed": seed, "chat_template_kwargs": {"enable_thinking": False}})
        # No logprobs: llama-server computes them with a softmax over the whole vocabulary (~250k entries)
        # on the CPU for every token and slot, which made generation CPU-bound (11 tok/s per slot on an
        # A100). Ties in votes are broken by the peers' weights instead (analyze_e4.decide).
        r.raise_for_status()
        j = r.json()
        ch = j["choices"][0]
        lps = [c["logprob"] for c in ((ch.get("logprobs") or {}).get("content") or []) if c.get("logprob") is not None]
        text = ch["message"].get("content") or ""
        return {"text": text, "reasoning": ch["message"].get("reasoning_content"), "finish": ch.get("finish_reason"),
                "n_tokens": j.get("usage", {}).get("completion_tokens"),
                "mean_logp": (sum(lps) / len(lps)) if lps else None, "ms": round((time.perf_counter() - t0) * 1000)}

    def close(self):
        if getattr(self, "http", None):
            self.http.close()
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_log", None):
            self._log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--benches", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    ap.add_argument("--splits", nargs="+", default=list(data.SPLITS), choices=list(data.SPLITS))
    ap.add_argument("--n", type=int, default=None, help="first n questions of each split (default: all)")
    ap.add_argument("--parallel", type=int, default=16)
    a = ap.parse_args()

    rev = resolve_revision(a.model)
    srv = ChatServer(a.gguf, a.parallel)
    try:
        for bench in a.benches:
            for split in a.splits:
                items = LOADERS[bench](a.n or data.PART[bench], split=split)
                manifest = {"model": a.model, "revision": rev, "backend": "llama.cpp (--jinja, OpenAI chat)",
                            "gguf": Path(a.gguf).name, "weights": file_identity(a.gguf), "engine": srv.version,
                            "data": data.dataset_identity(bench), "bench": bench, "split": split, "n": len(items),
                            "prompt": PROMPT_VERSION, "max_tokens": MAX_TOKENS[bench], "temperature": 0.0,
                            "thinking": False, "ctx_per_slot": CTX_PER_SLOT, "logprobs": False}
                path = RESULTS / f"solo_{a.model.replace('/', '__')}{a.suffix}_{bench}_{split}.jsonl"
                out = ResultsFile(path, manifest, key=("id",))
                todo = [it for it in items if (it["id"],) not in out.done]
                t0, k = time.perf_counter(), 0
                ex = ThreadPoolExecutor(max_workers=a.parallel)
                try:
                    futs = {ex.submit(srv.chat, prompt(bench, it), MAX_TOKENS[bench], 7): it for it in todo}
                    for fut in as_completed(futs):
                        it, g = futs[fut], fut.result()
                        ended = g["finish"] == "stop"
                        out.write({"id": it["id"], "model": a.model, "bench": bench, "text": g["text"],
                                   "answer": answers.extract(bench, g["text"], it, ended), "gold": answers.gold(bench, it),
                                   "finish": g["finish"], "n_tokens": g["n_tokens"], "mean_logp": g["mean_logp"],
                                   "ms": g["ms"], "reasoning": bool(g["reasoning"])})
                        k += 1
                except BaseException:
                    ex.shutdown(wait=False, cancel_futures=True)
                    srv.close()
                    raise
                ex.shutdown()
                out.release()
                el = time.perf_counter() - t0
                print(f"{bench} {split}: {k} nouvelles réponses en {el:.0f} s", flush=True)
    finally:
        srv.close()
    print("done", a.model, flush=True)


if __name__ == "__main__":
    main()
