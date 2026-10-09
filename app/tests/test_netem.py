"""WAN emulation in the tracker relay (experiments only, off by default) and the early_stop switch."""
from __future__ import annotations

import asyncio
import math
import statistics
import time

import httpx
import pytest

from myriad.engine import FakeEngine
from myriad.gateway import chat_args, GatewayError
from myriad.netem import WanDelay, wan_from_rtt
from myriad.node import NodeClient
from myriad.tracker import Tracker

from .conftest import MODELS, start_swarm, wait_until

TICK = 0.016  # resolution of time.monotonic (the asyncio clock) on Windows
MATH = {"role": "user", "content": "Tom has 40 apples and buys 2 more. How many apples does he have?"}


def test_wan_delay_sampling():
    assert wan_from_rtt(0) is None and wan_from_rtt(-5) is None
    w = wan_from_rtt(100, sigma=0.0)
    assert w.median_ms == 50 and w.sample() == pytest.approx(0.05)
    assert WanDelay(0.0).sample() == 0.0 and not WanDelay(0.0).enabled
    w = WanDelay(median_ms=40, sigma=0.5, seed=1)
    xs = [w.sample() for _ in range(20000)]
    assert statistics.median(xs) == pytest.approx(0.040, rel=0.05)
    logs = [math.log(x) for x in xs]
    assert statistics.pstdev(logs) == pytest.approx(0.5, rel=0.05)
    capped = WanDelay(median_ms=1000, sigma=3.0, seed=2, max_ms=1500)
    assert max(capped.sample() for _ in range(2000)) <= 1.5
    for bad in (dict(median_ms=-1), dict(median_ms=10, sigma=-0.1), dict(median_ms=float("nan"))):
        with pytest.raises(ValueError):
            WanDelay(**bad)


def test_wan_off_by_default(tmp_path):
    t = Tracker(db_path=tmp_path / "t.sqlite")
    assert t.wan is None
    assert Tracker(db_path=tmp_path / "u.sqlite", wan=WanDelay(0.0)).wan is None  # median 0 = off
    t.ledger.close()


class OrderNode(NodeClient):
    """Records the order in which jobs reach the node."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.order: list[str] = []

    async def _on_job(self, f):
        self.order.append(f.job.messages[-1].content)
        await super()._on_job(f)


async def _ask_raw(gw, n: int, target: str) -> list[str]:
    """Send n signed jobs to one node in a burst, through the requester's connection."""
    from myriad.protocol import Job
    import uuid

    sent = []
    for i in range(n):
        content = f"job {i:03d}"
        job = Job(job_id=uuid.uuid4().hex, requester_id=gw.node.node_id, messages=[{"role": "user", "content": content}],
                  max_tokens=8, temperature=0.0, seed=1, deadline_ms=30_000, task_hint="free").signed_by(gw.node.identity)
        await gw.node.submit(target, job)
        sent.append(content)
    return sent


async def test_wan_delay_preserves_order_both_ways(tmp_path):
    """A large jitter (sigma 1.5) would reorder frames sampled independently: order must survive the
    requester -> tracker hop (inbound delay) and the tracker -> node hop (outbound delay)."""
    s = await start_swarm(tmp_path, wan=WanDelay(median_ms=5, sigma=1.5, seed=3))
    try:
        node = await s.add_node("qwen", FakeEngine(MODELS["qwen"], "ok"), MODELS["qwen"], max_parallel=64,
                                cls=OrderNode)
        gw, client = await s.add_gateway()
        sent = await _ask_raw(gw, 40, node.node_id)
        await wait_until(lambda: len(node.order) == 40, 10)
        assert node.order == sent
    finally:
        await s.close()


async def test_wan_delay_adds_four_hops_to_a_request(tmp_path):
    """requester -> tracker -> node -> tracker -> requester: at least 4 one-way delays."""
    s = await start_swarm(tmp_path, wan=WanDelay(median_ms=60, sigma=0.0))
    try:
        for name in ("qwen", "smollm"):
            await s.add_node(name, FakeEngine(MODELS[name], "The answer is 42."), MODELS[name])
        gw, client = await s.add_gateway()
        await gw.directory()  # the directory (HTTP, also delayed) is fetched outside the timed part
        t0 = time.perf_counter()
        ans = await gw.ask([MATH], k=2)
        elapsed = time.perf_counter() - t0
        assert ans.meta["answer"] == "42"
        # Lower bounds allow for the event loop's clock on Windows (15.6 ms ticks: a sleep can end early).
        assert 4 * 0.06 - 4 * TICK <= elapsed < 1.5, elapsed
        # HTTP to the tracker crosses two delayed hops too.
        async with httpx.AsyncClient(base_url=s.url) as http:
            t0 = time.perf_counter()
            assert (await http.get("/v1/health")).status_code == 200
            assert time.perf_counter() - t0 >= 2 * 0.06 - 2 * TICK
    finally:
        await s.close()


class CloseAfterResult(NodeClient):
    """Sends its result, then disconnects at once."""

    async def _execute(self, job):
        await super()._execute(job)
        await self._ws.close()


async def test_wan_disconnect_arrives_after_the_frames_sent_before_it(tmp_path):
    """The node's result is sent before its disconnection: with the delay, the tracker must still
    relay the result (not report the peer as disconnected)."""
    s = await start_swarm(tmp_path, wan=WanDelay(median_ms=80, sigma=0.0))
    try:
        await s.add_node("qwen", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"], cls=CloseAfterResult)
        gw, client = await s.add_gateway()
        ans = await gw.ask([MATH], k=1)
        assert ans.meta["peers_answered"] == 1 and ans.meta["answer"] == "42"
        await wait_until(lambda: not any(c.info.model for c in s.tracker.conns.values()), 5)
    finally:
        await s.close()


async def test_frame_counters(tmp_path):
    s = await start_swarm(tmp_path)
    try:
        await s.add_node("qwen", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"])
        gw, client = await s.add_gateway()
        await gw.ask([MATH], k=1)
        f = s.tracker.frames
        await wait_until(lambda: f["in:receipt"] == 1, 5)
        assert f["in:job"] == 1 and f["in:result"] == 1 and f["out"] >= 4  # 2 welcomes, job, result
    finally:
        await s.close()


async def test_early_stop_switch(tmp_path):
    """early_stop=False waits for every peer; the default stops as soon as the certificate holds."""
    s = await start_swarm(tmp_path)
    try:
        await s.add_node("qwen", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"])
        await s.add_node("smollm", FakeEngine(MODELS["smollm"], "The answer is 42."), MODELS["smollm"])
        await s.add_node("gemma", FakeEngine(MODELS["gemma"], "The answer is 7.", delay_s=lambda m: 1.0),
                         MODELS["gemma"])
        gw, client = await s.add_gateway()
        t0 = time.perf_counter()
        fast = await gw.ask([MATH], k=3)
        t_fast = time.perf_counter() - t0
        assert fast.meta["early_stop"] and fast.meta["peers_answered"] == 2 and t_fast < 0.9
        await asyncio.sleep(0.3)  # the cancelled straggler frees its slot
        t0 = time.perf_counter()
        full = await gw.ask([MATH], k=3, early_stop=False)
        t_full = time.perf_counter() - t0
        # asyncio may fire a timer one clock resolution early, on a clock that itself ticks by TICK: the
        # 1 s sleep can end up to 2 TICK early (no directory download hides it any more in essaim/1.1).
        assert not full.meta["early_stop"] and full.meta["peers_answered"] == 3 and t_full >= 1.0 - 2 * TICK
        assert full.meta["answer"] == fast.meta["answer"] == "42" and full.meta["certificate"]
    finally:
        await s.close()


def test_chat_args_early_stop():
    base = {"messages": [MATH]}
    assert chat_args(base)["early_stop"] is None
    assert chat_args({**base, "myriad": {"early_stop": False}})["early_stop"] is False
    with pytest.raises(GatewayError):
        chat_args({**base, "myriad": {"early_stop": "false"}})
    # The former name of the field and of the swarm model stay accepted (deprecated aliases).
    assert chat_args({**base, "essaim": {"early_stop": False}})["early_stop"] is False
    assert chat_args({**base, "myriad": {"k": 2}, "essaim": {"k": 3}})["k"] == 2
    for name in ("myriad", "essaim"):
        assert chat_args({**base, "model": name})["model"] is None
    assert chat_args({**base, "model": "Qwen/Qwen3-1.7B-GGUF"})["model"] == "Qwen/Qwen3-1.7B-GGUF"


class _WS:
    async def close(self):
        pass


async def test_wan_frames_of_a_replaced_session_are_ignored(tmp_path):
    """A node reconnects while frames of its old session are still delayed: like the direct path,
    which stops reading a replaced session, those frames must not be acted upon (audit Codex)."""
    from myriad.crypto import Identity
    from myriad.protocol import Job, JobFrame, NodeInfo
    from myriad.tracker import Conn

    t = Tracker(db_path=tmp_path / "t.sqlite", spot_rate=0, wan=WanDelay(20))
    a, b = Identity.generate(), Identity.generate()
    old = Conn(a.node_id, NodeInfo(node_id=a.node_id, pubkey=a.pubkey), _WS(), wan=t.wan, inbox=asyncio.Queue())
    target = Conn(b.node_id, NodeInfo(node_id=b.node_id, pubkey=b.pubkey, model=MODELS["qwen"], accepting=True), _WS())
    t.conns = {a.node_id: old, b.node_id: target}
    t.ledger.ensure_account(a.node_id, a.pubkey)
    job = Job(job_id="a" * 32, requester_id=a.node_id, messages=[{"role": "user", "content": "2+2?"}], max_tokens=8,
              temperature=0, seed=0, deadline_ms=1000).signed_by(a)
    old.inbox.put_nowait((time.monotonic() + 0.02, JobFrame(job=job, target=b.node_id)))
    old.inbox.put_nowait((time.monotonic() + 0.02, None))
    task = asyncio.create_task(t._delayed_dispatch(old))
    await asyncio.sleep(0)
    t._drop(old)
    t.conns[a.node_id] = Conn(a.node_id, old.info, _WS())
    await task
    assert job.job_id not in t.jobs and target.busy == 0 and old.submitted == 0
    t.ledger.close()
