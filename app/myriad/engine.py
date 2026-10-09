"""Inference engines: llama-server (llama.cpp) driven over its OpenAI-compatible API, and a fake
engine with scripted answers for tests."""
from __future__ import annotations

import asyncio
import atexit
import os
import secrets
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from .config import default_llama_server

STARTUP_S = 300


class EngineError(RuntimeError):
    pass


@dataclass
class Generation:
    text: str
    finish_reason: str | None
    completion_tokens: int
    mean_logprob: float | None
    compute_ms: float


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def resolve_binary(path: str | None) -> str:
    """Explicit path, else env LLAMA_SERVER, else 'llama-server' on the PATH."""
    cand = path or default_llama_server()
    if Path(cand).exists():
        return str(Path(cand))
    found = shutil.which(cand)
    if found:
        return found
    raise EngineError(f"llama-server introuvable ({cand}) : indiquez son chemin (LLAMA_SERVER ou config llama_server)")


class LlamaServerEngine:
    """Our own llama-server child process: 127.0.0.1 only, a fresh random API key, a free port,
    --jinja (the GGUF's chat template), every layer on the GPU when there is one (-ngl 999)."""

    def __init__(self, gguf: str, model_id: str, binary: str | None = None, ctx: int = 4096, parallel: int = 1,
                 n_gpu_layers: int = 999, log_dir: Path | None = None, startup_s: float = STARTUP_S):
        self.gguf, self.model = str(gguf), model_id
        self.binary = binary
        self.ctx, self.parallel, self.n_gpu_layers = ctx, max(1, parallel), n_gpu_layers
        self.log_dir = Path(log_dir) if log_dir else Path(self.gguf).parent
        self.startup_s = startup_s
        self.proc: subprocess.Popen | None = None
        self.http: httpx.AsyncClient | None = None
        self._log = None
        self.port: int | None = None
        self._state = "arrêté"
        atexit.register(self._kill)  # never leave a llama-server behind

    @property
    def state(self) -> str:
        """'prêt' only while our llama-server process is alive: a crash makes the node unavailable."""
        if self._state == "prêt" and (self.proc is None or self.proc.poll() is not None):
            self._state = "erreur"
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        self._state = value

    def _kill(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    async def start(self) -> None:
        if not Path(self.gguf).exists():
            raise EngineError(f"fichier GGUF introuvable : {self.gguf}")
        self.binary = resolve_binary(self.binary)
        self.state = "démarrage"
        for _ in range(5):  # a port picked as free can be taken before llama-server binds it
            try:
                if await self._start_once():
                    self.state = "prêt"
                    return
            except BaseException:
                await self.close()
                raise
            await self.close()
        self.state = "erreur"
        raise EngineError(f"llama-server n'a pas démarré, voir {self.log_path}")

    async def _start_once(self) -> bool:
        self.port = free_port()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{Path(self.gguf).stem}.{os.getpid()}.{self.port}.log"
        self._log = open(self.log_path, "w", encoding="utf-8")
        key = secrets.token_hex(16)  # only we can query this server
        cmd = [self.binary, "-m", self.gguf, "--host", "127.0.0.1", "--port", str(self.port), "--jinja",
               "-ngl", str(self.n_gpu_layers), "-c", str(self.ctx * self.parallel), "-np", str(self.parallel),
               "--no-webui", "--api-key", key]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     creationflags=flags)
        self.http = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}",
                                      headers={"Authorization": f"Bearer {key}"}, timeout=30)
        deadline = time.monotonic() + self.startup_s
        while True:
            if self.proc.poll() is not None:  # ours died (port in use...): never talk to another server
                return False
            try:
                if (await self.http.get("/health", timeout=5)).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise EngineError(f"llama-server pas prêt après {self.startup_s} s, voir {self.log_path}")
            await asyncio.sleep(0.5)
        r = await self.http.get("/props", timeout=10)  # needs our key: a foreign server answers 401
        if r.status_code != 200 or self.proc.poll() is not None:
            return False
        served = str(r.json().get("model_path", ""))
        return os.path.normcase(os.path.abspath(served)) == os.path.normcase(os.path.abspath(self.gguf))

    async def generate(self, messages: list[dict], max_tokens: int, temperature: float, seed: int,
                       timeout_s: float, thinking: bool = False) -> Generation:
        if self.http is None or self.state != "prêt":
            raise EngineError("llama-server n'est pas lancé")
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
                "logprobs": True, "top_logprobs": 1, "stream": False,
                "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
        t0 = time.perf_counter()
        try:
            r = await self.http.post("/v1/chat/completions", json=body, timeout=max(1.0, timeout_s))
        except httpx.TimeoutException as e:
            raise EngineError("délai dépassé") from e
        except httpx.TransportError as e:
            raise EngineError(f"llama-server injoignable : {e}") from e
        if r.status_code != 200:
            raise EngineError(f"llama-server HTTP {r.status_code}: {r.text[:200]}")
        return parse_chat_completion(r.json(), (time.perf_counter() - t0) * 1000)

    async def health(self) -> bool:
        """Liveness probe for the tracker's pings: our llama-server process answers /health."""
        if self.http is None or self.state != "prêt":
            return False
        try:
            return (await self.http.get("/health", timeout=2.0)).status_code == 200
        except httpx.HTTPError:
            return False

    def status(self) -> dict:
        return {"engine": "llama-server", "state": self.state, "model": self.model, "gguf": Path(self.gguf).name,
                "slots": self.parallel, "ctx": self.ctx}

    async def close(self) -> None:
        if self.http is not None:
            await self.http.aclose()
            self.http = None
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                await asyncio.to_thread(self.proc.wait, 20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if self._log is not None and not self._log.closed:
            self._log.close()
        if self._state != "erreur":
            self.state = "arrêté"


def parse_chat_completion(j: dict, compute_ms: float) -> Generation:
    try:
        ch = j["choices"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise EngineError("réponse de llama-server sans choix") from e
    text = (ch.get("message") or {}).get("content") or ""
    lps = [c.get("logprob") for c in ((ch.get("logprobs") or {}).get("content") or [])]
    lps = [float(x) for x in lps if isinstance(x, (int, float)) and x == x and x > -1e9]
    mean = min(0.0, sum(lps) / len(lps)) if lps else None
    usage = j.get("usage") or {}
    tokens = int(usage.get("completion_tokens") or len(lps))
    return Generation(text=text, finish_reason=ch.get("finish_reason"), completion_tokens=max(0, tokens),
                      mean_logprob=mean, compute_ms=compute_ms)


class FakeEngine:
    """Scripted engine for tests and simulations: fixed (or computed) reply, optional delay (fixed, or
    computed from the messages, e.g. sampled), optional failure. `hung`: the engine is stuck, its
    generations and its health probe never return (a frozen inference stack)."""

    def __init__(self, model: str = "fake/model", reply: str | Callable[[list[dict]], str] = "The answer is 42.",
                 delay_s: float | Callable[[list[dict]], float] = 0.0, fail: bool = False,
                 mean_logprob: float | None = -0.1,
                 tokens: int | None = None, hung: bool = False):
        self.model, self.reply, self.delay_s, self.fail = model, reply, delay_s, fail
        self.mean_logprob, self.tokens = mean_logprob, tokens
        self.hung = hung
        self.calls = 0
        self.cancelled = 0
        self.state = "prêt"

    async def health(self) -> bool:
        if self.hung:
            await asyncio.sleep(3600)
        return self.state == "prêt"

    async def start(self) -> None:
        self.state = "prêt"

    async def generate(self, messages: list[dict], max_tokens: int, temperature: float, seed: int,
                       timeout_s: float, thinking: bool = False) -> Generation:
        self.calls += 1
        t0 = time.perf_counter()
        try:
            await asyncio.sleep(3600 if self.hung else self.delay_s(messages) if callable(self.delay_s) else self.delay_s)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.fail:
            raise EngineError("panne simulée")
        text = self.reply(messages) if callable(self.reply) else self.reply
        tokens = min(max_tokens, self.tokens if self.tokens is not None else max(1, len(text.split())))
        return Generation(text=text, finish_reason="stop", completion_tokens=tokens, mean_logprob=self.mean_logprob,
                          compute_ms=(time.perf_counter() - t0) * 1000)

    def status(self) -> dict:
        return {"engine": "fake", "state": self.state, "model": self.model}

    async def close(self) -> None:
        self.state = "arrêté"
