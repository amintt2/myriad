"""Concrete llama.cpp and Linux E11 sandbox adapters; imports are lazy for offline preparation."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from verif1_protocol import file_sha, plain, sha, utc


class GGUFEngine:
    """Reuse E11's owned ChatServer, retaining the complete OpenAI usage counters without retries."""

    def __init__(self, catalogue: dict, model_dir: Path):
        self.catalogue, self.model_dir = catalogue, model_dir
        self.family, self.server = None, None

    def close(self):
        if self.server is not None:
            self.server.close()
        self.family, self.server = None, None

    def activate(self, family):
        if family == self.family:
            return
        self.close()
        from run_solo import ChatServer
        entry = self.catalogue[family]
        path = self.model_dir / entry["gguf"]
        if path.stat().st_size != entry["weights"]["size"] or file_sha(path) != entry["weights"]["sha256"]:
            raise ValueError("GGUF does not match frozen E11 weights")
        self.server = ChatServer(str(path), parallel=1, ctx_per_slot=8192)
        try:
            if "build 11505" not in self.server.version or "ff5888f99" not in self.server.version:
                raise ValueError("llama.cpp engine differs from pinned E11 build")
            self.server._log.flush()
            text = self.server.log_path.read_text(encoding="utf-8", errors="replace")
            offload = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
            if not offload or offload[-1][0] != offload[-1][1]:
                raise ValueError("full GPU offload was not attested")
        except BaseException:
            self.close()
            raise
        self.family = family

    def generate(self, family, prompt, max_tokens, seed, decoder=None):
        self.activate(family)
        started, t0 = utc(), time.monotonic()
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                **(decoder or {"temperature": 0.0, "top_p": 1.0}), "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False}}
        request_hash = sha(body)
        response = self.server.http.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        raw = response.json()
        body = raw if isinstance(raw, dict) else {}
        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        prompt_n, output_n = usage.get("prompt_tokens"), usage.get("completion_tokens")
        text, finish = message.get("content") or "", choice.get("finish_reason", "invalid_response")
        return {"text": text, "reasoning": message.get("reasoning_content"),
                "finish": finish, "prompt_tokens": prompt_n, "completion_tokens": output_n,
                "truncated": finish != "stop", "started_utc": started, "ended_utc": utc(),
                "seconds": time.monotonic() - t0, "request_sha256": request_hash,
                "response_sha256": sha(raw), "response": raw,
                "provenance": {**self.catalogue[family], "engine": self.server.version,
                               "offload": "all GPU layers", "context": 8192, "parallel": 1}}


class LinuxSandbox:
    """Only the existing untrusted CHILD executes candidate code; expected values never reach it."""

    def __init__(self):
        if not sys.platform.startswith("linux"):
            raise ValueError("candidate execution requires Linux E11 sandbox")
        from essaim import sandbox
        self.sandbox = sandbox
        self.identity = sandbox.isolation()

    @staticmethod
    def input_value(arguments):
        from essaim.sandbox import enc_text
        encoded = enc_text(tuple(arguments))
        if encoded is None:
            raise ValueError("input exceeds codec limits")
        return encoded

    def run(self, candidate, contract, encoded_inputs):
        started, t0 = utc(), time.monotonic()
        load, records, status, milliseconds = self.sandbox._candidate(
            candidate["code"], contract["entry"], contract["header"], "visible", encoded_inputs)
        cells = []
        for i in range(len(encoded_inputs)):
            record = records.get(i, {})
            cell = {"ok": False, "error": record.get("err") or load or "MissingOutput", "record": record}
            if "digest" in record or str(record.get("err", "")).startswith("Output"):
                cell.update(coverage_loss=True, error="output_serialization_coverage_loss")
            if load is None and "out" in record and "err" not in record:
                try:
                    value = self.sandbox.dec(record["out"])
                    plain(value)
                except (TypeError, ValueError):
                    cell.update(error="non_plain_JSON_output", coverage_loss=True)
                else:
                    cell.update(ok=True, value=value, error=None)
            cells.append(cell)
        return {"cells": cells, "load": load, "status": status, "ms": milliseconds,
                "seconds": time.monotonic() - t0, "started_utc": started, "ended_utc": utc(),
                "isolation": self.identity}

    def smoke(self):
        contract = {"entry": "f", "header": ""}
        inputs = [self.input_value([2]), self.input_value([3])]
        good = self.run({"code": "def f(x):\n    return x + 1"}, contract, inputs)
        bad = self.run({"code": "def f(x):\n    return 0"}, contract, inputs)
        forbidden = str(self.sandbox.PHASE0 / "verif1_protocol.py")
        denied = self.run({"code": f"def f(x):\n    return open({forbidden!r}).read()"}, contract, inputs)
        if ([c.get("value") for c in good["cells"]] != [3, 4]
                or [c.get("value") for c in bad["cells"]] != [0, 0] or any(c["ok"] for c in denied["cells"])):
            raise ValueError("Linux sandbox public smoke failed")
        return {"good": good, "bad": bad, "read_denied": denied, "isolation": self.identity}


class SimulatedEngine:
    """Explicit smoke-only engine. Never presented as an inference result."""

    def generate(self, family, prompt, max_tokens, seed, decoder=None):
        started = utc()
        text = (json.dumps({"score": 0.5, "reason": "simulated public judgment"}) if max_tokens == 256
                else json.dumps({"cases": [], "properties": []}))
        return {"text": text, "finish": "stop", "truncated": False, "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(text.split()), "started_utc": started, "ended_utc": utc(),
                "seconds": 0.0, "provenance": {"simulation": True, "family": family}}

    def close(self):
        pass


class SimulatedSandbox:
    identity = {"simulation": True, "platform": "no candidate execution"}
    input_value = staticmethod(LinuxSandbox.input_value)

    def run(self, candidate, contract, encoded_inputs):
        return {"cells": [{"ok": False, "error": "simulation_no_execution"} for _ in encoded_inputs],
                "status": "simulated", "ms": 0, "seconds": 0.0, "isolation": self.identity,
                "started_utc": utc(), "ended_utc": utc()}

    def smoke(self):
        return {"simulation": True, "isolation": self.identity}
