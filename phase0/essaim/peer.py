"""A mode-B peer: one frozen model behind a tiny HTTP API.

    uv run python -m essaim.peer --model Qwen/Qwen3-1.7B --gguf ../models/Qwen3-1.7B-Q8_0.gguf --port 8101
    ESSAIM_TOKEN=... uv run python -m essaim.peer --model google/gemma-4-E2B-it --port 8104 --host 0.0.0.0   # Mac

Calls are serialised (one model, one cache). Listening beyond 127.0.0.1 requires a shared
token (header X-Essaim-Token); request sizes are bounded so that one client cannot
monopolise the peer.
"""
from __future__ import annotations

import argparse
import hmac
import os
import threading
import time

from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .common import PROMPT_VERSION, resolve_revision
from .results import file_identity

app = FastAPI()
LOCK = threading.Lock()
lm = None
TOKEN: str | None = None
MAX_TEXT = 32_000


class Ctx(BaseModel):
    user: str = Field(max_length=MAX_TEXT)
    system: str | None = Field(default=None, max_length=MAX_TEXT)
    assistant_prefix: str = Field(default="", max_length=MAX_TEXT)


# Stop rules are chosen by name (common.STOPS): clients never send regular expressions.
MAX_PROMPT_TOKENS = 3072     # rendered prompt (template + user + system + assistant prefix)
MAX_CANDIDATE_TOKENS = 1024  # sum over the candidates of one /score call
DEADLINE_S = 120.0           # generation stops after this, whatever n is


def check_tokens(text: str, limit: int, what: str):
    n = len(lm.encode(text))
    if n > limit:
        raise HTTPException(status_code=413, detail=f"{what} : {n} jetons, maximum {limit}")


class ProposeReq(Ctx):
    n: int = Field(default=16, ge=1, le=512)
    stop: list[Literal["final_answer"]] | None = Field(default=None, max_length=1)


class ScoreReq(Ctx):
    candidates: list[Annotated[str, Field(max_length=4000)]] = Field(max_length=8)


class LettersReq(BaseModel):
    user: str = Field(max_length=MAX_TEXT)
    letters: list[Annotated[str, Field(min_length=1, max_length=1)]] = Field(max_length=26)
    system: str | None = Field(default=None, max_length=MAX_TEXT)
    answer_prefix: str = Field(default="Answer:", max_length=200)


def check(token: str | None):
    if TOKEN and not (token and hmac.compare_digest(token, TOKEN)):
        raise HTTPException(status_code=401, detail="jeton manquant ou invalide")


ADMIT = threading.BoundedSemaphore(2)  # one request computing + one waiting; more are refused


def timed(fn):
    if not ADMIT.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="pair occupé")
    try:
        q0 = time.perf_counter()
        if not LOCK.acquire(timeout=600):
            raise HTTPException(status_code=503, detail="pair occupé")
        try:
            t0 = time.perf_counter()
            res = fn()
            return {"result": res, "compute_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "queue_ms": round((t0 - q0) * 1000, 1)}
        finally:
            LOCK.release()
    finally:
        ADMIT.release()


IDENTITY: dict = {}


@app.get("/info")
def info(x_essaim_token: str | None = Header(default=None)):
    check(x_essaim_token)
    return {**lm.info(), **IDENTITY}


@app.post("/propose")
def propose(r: ProposeReq, x_essaim_token: str | None = Header(default=None)):
    check(x_essaim_token)
    check_tokens(lm.render(r.user, r.system, r.assistant_prefix), MAX_PROMPT_TOKENS, "prompt")
    return timed(lambda: lm.propose(r.user, r.assistant_prefix, r.n, r.system, list(r.stop or []), DEADLINE_S))


@app.post("/score")
def score(r: ScoreReq, x_essaim_token: str | None = Header(default=None)):
    check(x_essaim_token)
    base = lm.render(r.user, r.system, r.assistant_prefix)
    check_tokens(base, MAX_PROMPT_TOKENS, "prompt")
    total = sum(len(lm.encode(c)) for c in r.candidates)  # sum of the separate tokenisations
    if total > MAX_CANDIDATE_TOKENS:
        raise HTTPException(status_code=413, detail=f"candidats : {total} jetons, maximum {MAX_CANDIDATE_TOKENS}")
    for c in r.candidates:  # every context actually sent to the backend
        check_tokens(base + c, MAX_PROMPT_TOKENS + MAX_CANDIDATE_TOKENS, "prompt + candidat")
    return timed(lambda: lm.score(r.user, r.assistant_prefix, r.candidates, r.system))


@app.post("/letters")
def letters(r: LettersReq, x_essaim_token: str | None = Header(default=None)):
    check(x_essaim_token)
    return timed(lambda: lm.letter_probs(r.user, r.letters, r.system, r.answer_prefix))


def main():
    global lm, TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--gguf", default=None, help="serve through llama.cpp (all layers on the GPU)")
    a = ap.parse_args()
    TOKEN = os.environ.get("ESSAIM_TOKEN") or None
    if a.host not in ("127.0.0.1", "localhost", "::1") and not TOKEN:
        raise SystemExit("Écoute réseau refusée sans jeton : définis ESSAIM_TOKEN (même valeur chez le coordinateur).")
    rev = resolve_revision(a.model)
    if a.gguf:
        from .llamacpp import LlamaCppLM
        lm = LlamaCppLM(a.model, a.gguf, revision=rev)
    else:
        from .lm import LM
        lm = LM(a.model, a.device, a.dtype, a.threads, revision=rev)
    IDENTITY.update({**lm.identity(), "prompt": PROMPT_VERSION,
                     "weights": file_identity(a.gguf) if a.gguf else None})
    print("ready:", {**lm.info(), **IDENTITY}, flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
