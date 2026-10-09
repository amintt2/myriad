"""In-process swarm: a real tracker (uvicorn on a free port), nodes with scripted engines, a gateway."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx
import pytest
import uvicorn

from myriad.crypto import Identity
from myriad.engine import FakeEngine, free_port
from myriad.gateway import Gateway
from myriad.node import NodeClient
from myriad.priors import family_of, params_of
from myriad.protocol import MAX_FRAME_BYTES
from myriad.tracker import Tracker


@pytest.fixture(autouse=True)
def _no_update_checks(monkeypatch):
    """No periodic update check against the network during the tests (updater.py)."""
    monkeypatch.setenv("MYRIAD_UPDATE_CHECK", "0")


async def wait_until(pred, timeout: float = 5.0, step: float = 0.02):
    end = asyncio.get_running_loop().time() + timeout
    while True:
        v = pred()
        if v:
            return v
        if asyncio.get_running_loop().time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(step)


@dataclass
class Swarm:
    tracker: Tracker
    url: str
    server: uvicorn.Server
    server_task: asyncio.Task
    nodes: dict = field(default_factory=dict)
    tasks: list = field(default_factory=list)
    gateways: list = field(default_factory=list)

    async def add_node(self, name: str, engine=None, model: str | None = None, max_parallel: int = 2,
                       gguf: str = "model.gguf", identity: Identity | None = None, cls=NodeClient) -> NodeClient:
        n = cls(identity or Identity.generate(), self.url, engine=engine, model=model,
                family=family_of(model) if model else None, gguf=gguf if model else None,
                params_b=params_of(model) if model else None, ctx=4096, max_parallel=max_parallel, reconnect=False)
        self.tasks.append(asyncio.create_task(n.run()))
        await asyncio.wait_for(n.connected.wait(), 5)
        self.nodes[name] = n
        return n

    async def add_gateway(self, name: str = "client", **kw) -> tuple[Gateway, httpx.AsyncClient]:
        node = await self.add_node(name)
        gw = Gateway(node, peers_ttl_s=0.0, **kw)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://127.0.0.1:8400",
                                   timeout=30)
        self.gateways.append((gw, client))
        return gw, client

    async def close(self):
        for gw, client in self.gateways:
            await client.aclose()
            await gw.close()
        for n in self.nodes.values():
            await n.stop()
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.server.should_exit = True
        try:
            await asyncio.wait_for(self.server_task, 5)
        except asyncio.TimeoutError:
            self.server_task.cancel()
        self.tracker.ledger.close()


async def start_swarm(tmp_path, **tracker_kw) -> Swarm:
    kw = dict(starter_credit=1000.0, spot_rate=0.0, receipt_grace_s=60.0, sweep_s=0.05)
    kw.update(tracker_kw)
    tracker = Tracker(db_path=tmp_path / "tracker.sqlite", **kw)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(tracker.app, host="127.0.0.1", port=port, log_level="warning",
                                           ws_max_size=MAX_FRAME_BYTES, timeout_graceful_shutdown=1))
    task = asyncio.create_task(server.serve())
    await wait_until(lambda: server.started, 10)
    return Swarm(tracker=tracker, url=f"http://127.0.0.1:{port}", server=server, server_task=task)


MODELS = {
    "qwen": "Qwen/Qwen3-1.7B-GGUF",
    "smollm": "HuggingFaceTB/SmolLM3-3B-GGUF",
    "gemma": "ggml-org/gemma-4-E2B-it-GGUF",
    "granite": "ibm-granite/granite-3.3-2b-instruct-GGUF",
}


def scripted(math_answer: str, free_text: str):
    def reply(messages):
        q = messages[-1]["content"]
        if "capital" in q:
            return free_text
        return f"Let me compute step by step. The answer is {math_answer}."
    return reply


@pytest.fixture
async def swarm(tmp_path):
    """4 nodes of 4 families: qwen and smollm fast and right, gemma slow (and wrong), granite failing."""
    s = await start_swarm(tmp_path)
    engines = {
        "qwen": FakeEngine(MODELS["qwen"], scripted("42", "Paris is the capital of France."), tokens=10),
        "smollm": FakeEngine(MODELS["smollm"], scripted("42", "The capital of France is Paris."), tokens=10),
        "gemma": FakeEngine(MODELS["gemma"], scripted("7", "France's capital city is Paris."), delay_s=5.0, tokens=10),
        "granite": FakeEngine(MODELS["granite"], "unused", fail=True),
    }
    for name, eng in engines.items():
        await s.add_node(name, eng, MODELS[name])
    s.engines = engines
    try:
        yield s
    finally:
        await s.close()
