"""Regression tests of the network-layer audit (audits/2026-10-09_audit_complet_net.md), one per
issue: each test reproduces the reported scenario and checks the fix."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from myriad.crypto import Identity
from myriad.engine import FakeEngine
from myriad.gateway import Gateway, GatewayError, chat_args, read_json
from myriad.netem import WanDelay
from myriad.protocol import JobFrame, JobResult, NodeInfo, ReceiptFrame, Receipt, ResultFrame, Route
from myriad.routing import live_routes, peer_family
from myriad.tracker import Conn, Tracker, spot_verdict

from .conftest import MODELS, start_swarm, wait_until
from .test_scale_v11 import MATH, fake_conn, sent_frames, signed_job


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=tmp_path / "t.sqlite", spot_rate=0.0, seed=3, receipt_grace_s=60.0)
    yield t
    t.ledger.close()


def requester(t: Tracker, credits: float) -> Conn:
    c = fake_conn(t, None)
    t.ledger.ensure_account(c.node_id, c.info.pubkey)
    t.ledger.db.execute("UPDATE accounts SET balance=? WHERE node_id=?", (int(credits * 1000), c.node_id))
    return c


# ---------------------------------------------------------------- 1. P1 credit overdraft
def test_1_unsettled_results_no_longer_recycle_the_quota(tracker):
    t = tracker
    node = fake_conn(t, MODELS["qwen"], "qwen", max_parallel=64)
    req = requester(t, 1.0)  # one credit
    accepted = 0
    for _ in range(200):
        job = signed_job(req.identity)
        t._submit(req, JobFrame(job=job, target=node.node_id))
        tj = t.jobs.get(job.job_id)
        if tj is None:
            continue
        accepted += 1
        res = JobResult(job_id=job.job_id, node_id=node.node_id, model=MODELS["qwen"], text="The answer is 42.",
                        completion_tokens=8).signed_by(node.identity)
        t._on_result(node, res)  # delivered at once, the receipt never comes (settled only after 60 s)
    assert accepted == 1  # was 200 before the fix: the in-flight quota was freed at each delivery
    errs = [f for f in sent_frames(req) if getattr(f, "error", None) == "insufficient_credits"]
    assert len(errs) == 199
    t.receipt_grace_s = 0.0
    t._delivered = type(t._delivered)((0.0, j) for _, j in t._delivered)
    t.sweep()  # automatic settlement releases the commitment
    assert t._committed.get(req.node_id, 0) == 0


def test_1b_a_fresh_account_has_a_small_in_flight_burst(tracker):
    """In-flight jobs commit no credits: a fresh account may only have new_account_inflight of them, the
    cap growing to max_inflight with the account's standing (age, credits spent on others' work)."""
    t = tracker
    node = fake_conn(t, MODELS["qwen"], "qwen", max_parallel=64)
    req = requester(t, 1000.0)
    for _ in range(20):  # no result comes back: every accepted job stays in flight
        t._submit(req, JobFrame(job=signed_job(req.identity), target=node.node_id))
    assert req.submitted == t.new_account_inflight == 8
    assert sum(getattr(f, "error", None) == "too_many_jobs" for f in sent_frames(req)) == 12
    old = requester(t, 1000.0)
    t.ledger.db.execute("UPDATE accounts SET created_at=? WHERE node_id=?", (time.time() - 30 * 86400, old.node_id))
    t.ledger.reporter_weight = lambda node_id, now: 1.0 if node_id == old.node_id else 0.0  # full standing
    assert t._inflight_cap(old) == t.max_inflight


# ---------------------------------------------------------------- 2. P1 unbounded state per account
def test_2_failed_selections_leave_no_group_and_state_is_capped(tracker):
    t = tracker
    req = requester(t, 1000.0)
    import myriad.tracker as tm
    for i in range(1500):  # no peer at all: nothing may be kept
        job = signed_job(req.identity, deadline_ms=600_000)
        t._submit(req, JobFrame(job=job, route=Route(group=f"{i:032x}")))
    assert len(t.groups) == 0 and not any(k == "group" for _, _, k, _ in t._timers)
    sent_frames(req)  # (drain the requester's outbox, which this burst of refusals filled)
    req.closed = False
    node = fake_conn(t, MODELS["qwen"], "qwen", max_parallel=64)
    for i in range(tm.MAX_GROUPS_PER_ACCOUNT + 5):
        job = signed_job(req.identity)
        t._submit(req, JobFrame(job=job, route=Route(group=f"{i:032x}")))
        if job.job_id in t.jobs:
            t._cancel(t.jobs[job.job_id])  # submit then cancel: the quota comes back at once...
    assert t._groups_of[req.node_id] == tm.MAX_GROUPS_PER_ACCOUNT  # ...but the groups are capped
    assert any(getattr(f, "error", None) == "too_many_groups" for f in sent_frames(req))
    cancelled = [tj for tj in t.jobs.values() if tj.cancelled]
    assert cancelled and all(not hasattr(tj.job, "messages") for tj in cancelled)  # prompts dropped at once


# ---------------------------------------------------------------- 3. P1 public evidence
async def test_3_evidence_never_returns_the_answer(tmp_path):
    from myriad.security import Security
    s = await start_swarm(tmp_path)
    try:
        await s.add_node("qwen", FakeEngine(MODELS["qwen"], "The secret answer is 42.", tokens=3), MODELS["qwen"],
                         e2e=False)
        gw, _ = await s.add_gateway(security=Security({"require_e2e": False}))  # even a plaintext job
        await gw.ask([MATH], k=1)
        await wait_until(lambda: any(e["kind"] == "receipt" for e in s.tracker.ledger.recent(10)), 5)
        jid = next(e["job_id"] for e in s.tracker.ledger.recent(10) if e["kind"] == "receipt")
        async with httpx.AsyncClient(base_url=s.url) as http:
            r = await http.get(f"/v1/evidence/{jid}")
        assert r.status_code == 200 and "secret answer" not in r.text and r.json()["result_sha256"]
        assert "requester_id" not in r.json() and "receipt" not in r.json()
    finally:
        await s.close()


# ---------------------------------------------------------------- 4. P2 node slot leak
async def test_4_a_job_cancelled_before_it_starts_frees_its_slot(tmp_path):
    s = await start_swarm(tmp_path)
    try:
        node = await s.add_node("qwen", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"], max_parallel=1)
        req = Identity.generate()
        job = signed_job(req)
        node._start(job)
        node._on_disconnect()  # cancelled before its first step: `_execute`'s finally never runs
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert node.running == {} and node.guard.running == {}
    finally:
        await s.close()


# ---------------------------------------------------------------- 5. P2 truncated answers in spot checks
def test_5_truncated_answers_are_inconclusive():
    m = [MATH]
    assert spot_verdict("Step 1: 2", "The answer is 4.", "math", m, ("length", "stop")) is None
    assert spot_verdict("The answer is 4.", "The answer is 4.", "math", m, ("stop", "stop")) is True
    assert spot_verdict("The answer is 5.", "The answer is 4.", "math", m, ("stop", "stop")) is False


# ---------------------------------------------------------------- 6. P2 family canonicalisation
def test_6_family_is_canonical_everywhere(tracker):
    t = tracker
    undeclared = fake_conn(t, MODELS["qwen"], None)  # a node without `family`
    liar = fake_conn(t, MODELS["qwen"], "gemma")  # a node declaring another family
    assert undeclared.family == liar.family == "qwen"
    chosen = t.select(2, only_family="qwen")
    assert chosen and chosen[0].family == "qwen"
    view = t._snapshot_view(liar, time.time())
    assert view["family"] == "qwen" and peer_family(view) == "qwen" and peer_family({"model": MODELS["qwen"]}) == "qwen"
    routes = live_routes([t._snapshot_view(c, time.time()) for c in (undeclared, liar)])
    assert routes["myriad:qwen"]["peers"] == 2 and "myriad:gemma" not in routes


# ---------------------------------------------------------------- 7. P2 client disconnect
async def test_7_client_disconnect_cancels_ask(swarm):
    gw, _ = await swarm.add_gateway()
    state = {"cancelled": False, "started": asyncio.Event()}

    async def slow_ask(**kw):
        state["started"].set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    gw.ask = slow_ask
    body = json.dumps({"messages": [MATH]}).encode()
    msgs = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if msgs:
            return msgs.pop(0)
        await state["started"].wait()  # the client leaves once the work has started
        return {"type": "http.disconnect"}

    sent = []

    async def send(m):
        sent.append(m)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions", "query_string": b"",
             "headers": [(b"host", b"127.0.0.1:8400"), (b"content-type", b"application/json"),
                         (b"content-length", str(len(body)).encode())],
             "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8400)}
    await asyncio.wait_for(gw.app(scope, receive, send), 5)
    assert state["cancelled"]


# ---------------------------------------------------------------- 8. P2 body size limit while reading
async def test_8_body_limit_applies_while_reading():
    from starlette.requests import Request

    def request(chunks, length=None):
        headers = [(b"content-type", b"application/json")]
        if length is not None:
            headers.append((b"content-length", str(length).encode()))
        it = iter(chunks)
        read = []

        async def receive():
            try:
                c = next(it)
            except StopIteration:
                return {"type": "http.request", "body": b"", "more_body": False}
            read.append(len(c))
            return {"type": "http.request", "body": c, "more_body": True}

        return Request({"type": "http", "method": "POST", "headers": headers, "path": "/"}, receive), read

    with pytest.raises(GatewayError) as e:  # announced too large: nothing is read
        r, read = request([b"x" * 1024] * 4096, length=4 << 20)
        await read_json(r)
    assert e.value.status == 413 and read == []
    r, read = request([b"x" * (64 << 10)] * 64)  # 4 MiB without a length: stops right after 1 MiB
    with pytest.raises(GatewayError) as e:
        await read_json(r)
    assert e.value.status == 413 and sum(read) <= (1 << 20) + (64 << 10)


# ---------------------------------------------------------------- 9. P2 invalid JSON types
async def test_9_invalid_types_are_400_not_500(swarm):
    gw, client = await swarm.add_gateway()
    m = json.dumps([MATH])
    bad = ['{"messages": [{"role": [], "content": "x"}]}', '{"model": {"x": 1}, "messages": %s}' % m,
           '{"messages": %s, "max_tokens": 1e309}' % m, '{"messages": %s, "temperature": "nan"}' % m,
           '{"messages": %s, "myriad": {"task_hint": ["math"]}}' % m, '{"messages": %s, "seed": -1e400}' % m]
    for body in bad:
        r = await client.post("/v1/chat/completions", content=body.encode(), headers={"Content-Type": "application/json"})
        assert r.status_code == 400, (body, r.status_code, r.text)
    r = await client.post("/v1/chat/completions", content=b'{"messages": [], "max_tokens": NaN}',
                          headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    with pytest.raises(GatewayError):
        chat_args({"messages": [MATH], "seed": float("inf")})


# ---------------------------------------------------------------- 10. P3 netem overflow
def test_10_wan_delay_never_overflows():
    d = WanDelay(1, sigma=1000, seed=1)
    for _ in range(100):
        assert 0 <= d.sample() <= d.max_ms / 1000
    assert WanDelay(5, sigma=1.0, max_ms=0.0).sample() == 0.0
