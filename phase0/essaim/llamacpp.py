"""Same peer interface as essaim.lm.LM, backed by a llama.cpp server with every layer on the GPU.

The Hugging Face tokenizer renders the chat template and tokenises (identical prompts and
token ids to the PyTorch backend); llama-server runs the model. Exact log-probabilities of
chosen tokens are obtained by *forcing* them with a token-id grammar: with
post_sampling_probs=False, the reported logprob of a forced token is the raw (pre-grammar)
log-probability.

Requires the llama.cpp binaries (tools/llama.cpp-bin, or LLAMA_SERVER) and a GGUF file.
"""
from __future__ import annotations

import atexit
import math
import os
import re
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from .common import close_thought, letter_token_ids, render_chat, stop_point, think_spec, word_spans

ROOT = Path(__file__).resolve().parents[2]
SERVER = Path(os.environ.get("LLAMA_SERVER", ROOT / "tools" / "llama.cpp-bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")))
STARTUP_S = 300


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaCppLM:
    def __init__(self, model_id: str, gguf: str, ctx: int = 4096, revision: str | None = None, parallel: int = 1):
        """`ctx` is the context of ONE request; `parallel` > 1 gives the server that many slots
        (continuous batching), so that many requests can be sent at once from different threads."""
        self.model_id, self.gguf, self.revision, self.parallel = model_id, gguf, revision, parallel
        self._lock = threading.Lock()  # tokenizer and stats are shared by the request threads
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        try:
            v = subprocess.run([str(SERVER), "--version"], capture_output=True, text=True, timeout=30)
            lines = [l.strip() for l in (v.stdout + v.stderr).splitlines() if l.strip()]
            self.server_version = " | ".join(l for l in lines if "version" in l.lower() or "built with" in l.lower()) or None
        except (OSError, subprocess.TimeoutExpired):
            self.server_version = None
        self.think = think_spec(str(self.tok.chat_template or ""))
        self._letter_ids: dict[tuple, list[list[int]]] = {}
        self.stats = {"forward_s": 0.0, "calls": 0}  # forward_s sums request times (overlapping if parallel)
        self.proc, self.http, self._log, self.offload = None, None, None, None
        atexit.register(self.close)
        try:
            for attempt in range(5):  # a port picked as free can be taken by another job before we bind it
                if self._start(gguf, ctx):
                    break
                self.close()
            else:
                raise RuntimeError(f"llama-server could not start, see {self.log_path}")
        except BaseException:
            self.close()
            raise

    def _start(self, gguf: str, ctx: int) -> bool:
        """Start our own llama-server; True once it is healthy AND serves this GGUF from our process."""
        self.port = free_port()
        logs = Path(gguf).parent / "server-logs"
        logs.mkdir(exist_ok=True)
        self.log_path = logs / f"{Path(gguf).stem}.{os.getpid()}.{self.port}.log"  # one log per server
        self._log = open(self.log_path, "w", encoding="utf-8")
        key = secrets.token_hex(16)  # only our own server accepts requests signed with this key
        self.proc = subprocess.Popen(
            [str(SERVER), "-m", gguf, "--port", str(self.port), "--host", "127.0.0.1",
             "-ngl", "999", "-c", str(ctx * self.parallel), "-np", str(self.parallel), "--no-kv-unified",
             "--no-webui", "--api-key", key],
            stdout=self._log, stderr=subprocess.STDOUT)
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=600,
                                 headers={"Authorization": f"Bearer {key}"})
        deadline = time.monotonic() + STARTUP_S
        while True:
            if self.proc.poll() is not None:  # ours died (e.g. port already in use): never talk to another server
                return False
            try:
                if self.http.get("/health", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"llama-server not ready after {STARTUP_S}s, see {self.log_path}")
            time.sleep(0.5)
        r = self.http.get("/props", timeout=10)  # requires our API key: a foreign server answers 401
        if r.status_code != 200 or self.proc.poll() is not None:
            return False
        served = str(r.json().get("model_path", ""))
        if os.path.normcase(os.path.abspath(served)) != os.path.normcase(os.path.abspath(gguf)):
            return False  # not serving exactly our file
        self._log.flush()
        m = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", self.log_path.read_text(encoding="utf-8", errors="replace"))
        if m:
            self.offload = f"{m[-1][0]}/{m[-1][1]} layers on GPU"
            if m[-1][0] != m[-1][1]:
                raise RuntimeError(f"only {self.offload}: the model does not fit entirely on the GPU")
        return True

    def close(self):
        if self.http is not None:
            self.http.close()
            self.http = None
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if self._log is not None and not self._log.closed:
            self._log.close()

    # ---------- prompt ----------
    def render(self, user: str, system: str | None = None, assistant_prefix: str = "", think: bool = False) -> str:
        """Chat-template text, BOS included (prompts are sent as token ids, so no BOS is added twice)."""
        with self._lock:
            return render_chat(self.tok, user, system, think) + assistant_prefix

    def encode(self, text: str, offsets: bool = False):
        with self._lock:
            enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=offsets)
        return (enc["input_ids"], enc["offset_mapping"]) if offsets else enc["input_ids"]

    def _post(self, body: dict, timeout: float | None = None) -> dict:
        t0 = time.perf_counter()
        r = self.http.post("/completion", json={"cache_prompt": True, "post_sampling_probs": False, **body},
                           **({"timeout": timeout} if timeout is not None else {}))
        r.raise_for_status()
        with self._lock:
            self.stats["forward_s"] += time.perf_counter() - t0
            self.stats["calls"] += 1
        return r.json()

    def _forced_logprobs(self, prompt_ids: list[int], forced: list[int]) -> list[float]:
        """Teacher-forced log-probabilities of `forced` after `prompt_ids`."""
        j = self._post({"prompt": prompt_ids, "n_predict": len(forced), "n_probs": 1, "temperature": -1,
                        "grammar": "root ::= " + " ".join(f"<[{t}]>" for t in forced)})
        probs = j.get("completion_probabilities", [])
        if len(probs) != len(forced):
            raise RuntimeError(f"forced {len(forced)} tokens, got {len(probs)}")
        return [c["logprob"] for c in probs]

    # ---------- operations ----------
    def letter_probs(self, user: str, letters: list[str], system: str | None = None,
                     answer_prefix: str = "Answer:") -> list[float]:
        return self._letters_after(self.render(user, system, answer_prefix), letters)

    def _letters_after(self, prompt: str, letters: list[str]) -> list[float]:
        """Exact probability of each letter (all its single-token spellings), renormalised over the letters."""
        key = tuple(letters)
        with self._lock:
            if key not in self._letter_ids:
                self._letter_ids[key] = letter_token_ids(self.tok, letters)
        ids = self.encode(prompt)
        # One call gives the exact log-probs of the top 512 tokens; only letter tokens outside
        # that list are forced one by one (exact as well). No value is ever estimated.
        j = self._post({"prompt": ids, "n_predict": 1, "n_probs": 512, "temperature": -1})
        top = {t["id"]: t["logprob"] for t in j["completion_probabilities"][0]["top_logprobs"]}
        scores = []
        for cands in self._letter_ids[key]:
            lps = [top[t] if t in top else self._forced_logprobs(ids, [t])[0] for t in cands]
            m = max(lps)
            scores.append(m + math.log(sum(math.exp(x - m) for x in lps)))
        m = max(scores)
        ex = [math.exp(s - m) for s in scores]
        z = sum(ex)
        return [e / z for e in ex]

    def think_then_letters(self, user: str, letters: list[str], budget: int, seed: int,
                           temperature: float = 0.6) -> dict:
        """Thinking mode with a token budget (s1-style budget forcing): sample the reasoning,
        close it if the budget runs out, then read the exact answer-letter distribution."""
        open_tag, close_tag, _ = self.think
        prompt = self.render(user, think=True)
        j = self._post({"prompt": self.encode(prompt), "n_predict": budget, "temperature": temperature,
                        "top_p": 0.95, "top_k": 20, "seed": seed, "stop": [close_tag], "return_tokens": True})
        # Rebuild the reasoning from the generated token ids, special tokens included, so that
        # an opening tag emitted as a special token is not lost from the text; trailing special
        # tokens (end of turn / EOS, which llama.cpp keeps in `tokens`) are removed, as in lm.py.
        toks = list(j.get("tokens") or [])
        specials = set(self.tok.all_special_ids)
        while toks and toks[-1] in specials:
            toks.pop()
        with self._lock:
            thought = self.tok.decode(toks, skip_special_tokens=False) if j.get("tokens") else j["content"]
        thought = thought.split(close_tag)[0]
        stop = j.get("stop_type")  # "limit" (budget), "word" (closing tag) or "eos" (ended before closing)
        closing = close_thought(thought, self.think, prompt.rstrip().endswith(open_tag))
        probs = self._letters_after(prompt + thought + closing + "Answer:", letters)
        return {"probs": probs, "n_tokens": j.get("tokens_predicted", 0), "forced": stop == "limit", "stop": stop}

    def propose(self, user: str, assistant_prefix: str, n: int, system: str | None = None,
                stop: list[str] | None = None, deadline_s: float | None = None, chunk: int = 32) -> dict:
        """Greedy draft, generated in chunks of `chunk` tokens. After each chunk the named `stop`s
        (common.STOPS, judged with the assistant prefix so an open thinking block is respected) and the
        deadline are checked; text, token count and confidence are cut at the stop point. Each HTTP call
        is bounded by the remaining time, so the call returns at the deadline; the server may still
        finish the abandoned chunk (at most `chunk` tokens) before serving the next request."""
        ids = self.encode(self.render(user, system, assistant_prefix))
        names = list(stop or [])
        t_end = time.monotonic() + deadline_s if deadline_s else None
        toks, lps, why = [], [], "limit"
        specials = set(self.tok.all_special_ids)
        while len(toks) < n:
            step = min(n - len(toks), chunk)
            body = {"prompt": ids + toks, "n_predict": step, "n_probs": 1, "temperature": -1, "return_tokens": True}
            remaining = (t_end - time.monotonic()) if t_end else None
            if remaining is not None and remaining <= 0:
                why = "deadline"
                break
            try:  # the HTTP call itself is bounded by the remaining time: the deadline is enforced
                j = self._post(body, timeout=remaining)
            except httpx.TimeoutException:
                why = "deadline"
                break
            new = list(j.get("tokens") or [])
            probs = j.get("completion_probabilities", [])
            ended = j.get("stop_type") == "eos"
            while new and new[-1] in specials and ended:  # end of turn: not part of the draft
                new.pop()
            toks += new
            lps += [c["logprob"] for c in probs[: len(new)] if "logprob" in c]
            if names:
                text = self.tok.decode(toks, skip_special_tokens=False)
                cut = stop_point(assistant_prefix, text, names)
                if cut is not None:
                    keep = next(k for k in range(1, len(toks) + 1)
                                if len(self.tok.decode(toks[:k], skip_special_tokens=False)) >= cut)
                    toks, lps = toks[:keep], lps[:keep]
                    return {"text": text[:cut], "n_tokens": keep, "eos": False, "stop": "stop",
                            "mean_logp": sum(lps) / len(lps) if lps else 0.0}
            if ended:
                why = "eos"
                break
            if t_end and time.monotonic() > t_end:
                why = "deadline"
                break
            if not new:
                break
        return {"text": self.tok.decode(toks, skip_special_tokens=False), "n_tokens": len(toks), "eos": why == "eos",
                "stop": why, "mean_logp": sum(lps) / len(lps) if lps else 0.0}

    def score(self, user: str, assistant_prefix: str, candidates: list[str], system: str | None = None) -> list[dict]:
        """Per-word log-probabilities of each candidate (canonical tokenisation, teacher-forced)."""
        base = self.render(user, system, assistant_prefix)
        b = len(base)
        specials = set(self.tok.all_special_ids)
        out = []
        for cand in candidates:
            spans = word_spans(cand)
            words = [0.0] * len(spans)
            ids, offs = self.encode(base + cand, offsets=True)
            first = next((j for j, (s, e) in enumerate(offs) if e > b), len(ids))
            if any(t in specials for t in ids[first:]):
                # The candidate's text contains one of OUR control tokens (e.g. another family wrote
                # "<|im_end|>" as plain characters): llama-server refuses to force it (HTTP 500), and it
                # is not text this model would score as prose. Reported as not scoreable.
                out.append({"word_logp": words, "words": [cand[s:e] for s, e in spans],
                            "boundary_merged": first < len(ids) and offs[first][0] < b, "special_token": True})
                continue
            if ids[first:]:
                for j, lp in zip(range(first, len(ids)), self._forced_logprobs(ids[:first], ids[first:])):
                    pos = max(offs[j][1] - 1, b) - b
                    k = next((i for i, (ws, we) in enumerate(spans) if ws <= pos < we), len(spans) - 1)
                    if k >= 0:
                        words[k] += lp
            out.append({"word_logp": words, "words": [cand[s:e] for s, e in spans],
                        "boundary_merged": first < len(ids) and offs[first][0] < b})
        return out

    def identity(self) -> dict:
        """What was actually loaded: used in manifests and compared on resume."""
        import transformers
        return {"model": self.model_id, "revision": self.revision, "backend": "llama.cpp", "gguf": Path(self.gguf).name,
                "device": "gpu, all layers offloaded (-ngl 999)", "dtype": "per GGUF file (see weights)",
                "engine": {"llama-server": self.server_version, "transformers (tokenizer)": transformers.__version__}}

    def info(self) -> dict:
        return {**self.identity(), **self.stats, "parallel": self.parallel}
