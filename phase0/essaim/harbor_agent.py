"""External Harbor agent. Import only in the dedicated, pinned Harbor environment."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict

import httpx
from harbor.agents.base import BaseAgent

from essaim.agent import GenerationLimit, HarborEnv, run_episode
from essaim.terminal_bench import validate_endpoint


class MyriadAgent(BaseAgent):
    def __init__(self, *args, families: list[str], strategy: str, max_steps: int, timeout_s: float,
                 cmd_timeout_s: float, max_tokens: int, total_tokens: int, seed: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.families, self.strategy = families, strategy
        self.max_steps, self.timeout_s, self.cmd_timeout_s = max_steps, timeout_s, cmd_timeout_s
        self.max_tokens, self.total_tokens, self.seed = max_tokens, total_tokens, seed

    @staticmethod
    def name() -> str:
        return "myriad-terminal"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        result = await environment.exec(command="command -v bash && command -v timeout", timeout_sec=10)
        if result.return_code:
            raise RuntimeError("bash and GNU timeout are required in the task environment")

    async def run(self, instruction, environment, context) -> None:
        base_url = os.environ.get("MYRIAD_BENCH_URL", "http://127.0.0.1:8400/v1")
        key = os.environ.get("MYRIAD_BENCH_TOKEN", "")
        validate_endpoint(base_url, key)
        env = HarborEnv(environment, asyncio.get_running_loop())
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        remaining, outcome = self.total_tokens, {"stopped": "error"}
        async with httpx.AsyncClient(base_url=base_url.rstrip("/") + "/", timeout=30,
                                     headers={"Authorization": f"Bearer {key}"}) as client:
            with (self.logs_dir / "episode.jsonl").open("w", encoding="utf-8") as log:
                def record(value):
                    log.write(json.dumps(value, ensure_ascii=False) + "\n")
                    log.flush()

                async def request(model, messages):
                    nonlocal remaining
                    if remaining <= 0:
                        raise GenerationLimit("episode completion allowance exhausted")
                    allowance = min(self.max_tokens, remaining)
                    remaining -= allowance
                    try:
                        response = await client.post("chat/completions", json={
                            "model": model, "messages": messages, "max_tokens": allowance,
                            "temperature": 0.0, "seed": self.seed})
                        response.raise_for_status()
                        data = response.json()
                    except (Exception, asyncio.CancelledError) as error:
                        record({"kind": "generation_error", "model": model, "allowance": allowance,
                                "exception_type": type(error).__name__})
                        raise
                    reply = data["choices"][0]["message"].get("content") or ""
                    record({"kind": "generation", "model": model, "allowance": allowance,
                            "served_model": data.get("model"), "usage": data.get("usage"), "reply": reply})
                    return reply

                worker = asyncio.create_task(asyncio.to_thread(
                    run_episode, lambda model, messages: env.call(request(model, messages)), env,
                    instruction, self.families, self.strategy, self.max_steps, self.cmd_timeout_s,
                    self.families, self.timeout_s, lambda step: record({"kind": "step", **asdict(step)})))
                try:
                    result = await asyncio.shield(worker)
                    outcome = {"stopped": result["stopped"], "steps": len(result["steps"])}
                except asyncio.CancelledError:
                    outcome = {"stopped": "timeout_or_cancelled"}
                    env.close()
                    try:
                        await worker
                    except (Exception, asyncio.CancelledError):
                        pass
                    raise
                finally:
                    env.close()
                    record({"kind": "end", **outcome, "remaining_allowance": remaining})
