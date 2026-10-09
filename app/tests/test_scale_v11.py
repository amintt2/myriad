"""essaim/1.1: server-side peer selection (A), hung-node detection (B), replacement of refused or
failed peers with an exact stop certificate (C), and backward compatibility with essaim/1."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

import myriad.node as node_mod
from myriad.crypto import Identity
from myriad.engine import FakeEngine
from myriad.gateway import Gateway
from myriad.node import NodeClient
from myriad.protocol import (Assigned, Cancel, Job, JobError, JobFrame, NodeInfo, PeerCard, Ping, ResultFrame,
                             JobResult, Route, dump_frame, parse_frame)
from myriad.tracker import Conn, Tracker

from .conftest import MODELS, start_swarm, wait_until

MATH = {"role": "user", "content": "Tom has 40 apples and buys 2 more. How many apples does he have?"}


class _WS:
    async def close(self):
        pass


def fake_conn(t: Tracker, model: str | None, family: str | None = None, max_parallel: int = 1,
              accepting: bool = True) -> Conn:
    ident = Identity.generate()
    info = NodeInfo(node_id=ident.node_id, pubkey=ident.pubkey, model=model, family=family,
                    gguf=f"{family}.gguf" if family else None, max_parallel=max_parallel, accepting=accepting)
    c = Conn(ident.node_id, info, _WS())
    c.identity = ident
    t.conns[c.node_id] = c
    t._reindex(c)
    return c


def sent_frames(c: Conn) -> list:
    out = []
    while not c.outbox.empty():
        out.append(parse_frame(c.outbox.get_nowait()[1]))
    return out


def signed_job(ident: Identity, deadline_ms: int = 30_000) -> Job:
    import uuid
    return Job(job_id=uuid.uuid4().hex, requester_id=ident.node_id, messages=[MATH], max_tokens=8, temperature=0,
               seed=0, deadline_ms=deadline_ms, task_hint="math").signed_by(ident)


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=tmp_path / "t.sqlite", spot_rate=0.0, seed=3)
    yield t
    t.ledger.close()


# ============================================================ A. server-side selection
def test_select_distinct_families_most_reliable_first(tracker):
    t = tracker
    for fam in ("qwen", "smollm", "gemma", "granite"):
        for _ in range(5):
            fake_conn(t, MODELS[fam], fam)
    chosen = t.select(4)
    assert [c.family for c in chosen] == ["smollm", "gemma", "qwen", "granite"]  # priors: 0.865, 0.74, 0.535, 0.495
    assert [c.family for c in t.select(2)] == ["smollm", "gemma"]
    assert [c.family for c in t.select(4, exclude_families={"smollm"})] == ["gemma", "qwen", "granite"]
    only = t.select(3, only_family="qwen")
    assert len(only) == 1 and only[0].family == "qwen"
    m = t.select(1, model=MODELS["gemma"])
    assert len(m) == 1 and m[0].info.model == MODELS["gemma"]


def test_select_skips_busy_paused_suspended_excluded_and_low_reputation(tracker):
    t = tracker
    a = fake_conn(t, MODELS["smollm"], "smollm")
    b = fake_conn(t, MODELS["smollm"], "smollm")
    assert {t.select(1)[0].node_id for _ in range(40)} == {a.node_id, b.node_id}  # random spread
    a.busy = 1
    t._reindex(a)
    assert all(t.select(1)[0] is b for _ in range(20))  # a has no free slot
    assert t.select(1, exclude={b.node_id}) == []
    b.info = b.info.model_copy(update={"accepting": False})
    t._reindex(b)
    assert t.select(1) == []
    a.busy = 0
    t._reindex(a)
    t._suspend(a, "test")
    assert t.select(1) == [] and a.slot is None
    t._readmit(a)
    assert t.select(1) == [a]
    a.reputation = 0.1
    t._reindex(a)
    assert t.select(1) == []


def test_select_prefers_idle_then_least_loaded(tracker):
    t = tracker
    busy = fake_conn(t, MODELS["qwen"], "qwen", max_parallel=4)
    busy.busy = 3
    t._reindex(busy)
    light = fake_conn(t, MODELS["qwen"], "qwen", max_parallel=4)
    light.busy = 1
    t._reindex(light)
    assert busy.slot == (MODELS["qwen"], "partial") and light.slot == (MODELS["qwen"], "partial")
    picks = [t.select(1)[0] for _ in range(50)]
    assert picks.count(light) > picks.count(busy)  # best of two random draws
    idle = fake_conn(t, MODELS["qwen"], "qwen", max_parallel=4)
    assert all(t.select(1)[0] is idle for _ in range(20))


def test_select_cost_does_not_grow_with_n(tracker):
    """O(k) selection: 4096 nodes are not slower to select from than 64 (within a wide margin)."""
    t = tracker

    def timed(n_per_family: int) -> float:
        for fam in ("qwen", "smollm", "gemma", "granite"):
            for _ in range(n_per_family):
                fake_conn(t, MODELS[fam], fam)
        t0 = time.perf_counter()
        for _ in range(2000):
            t.select(4)
        return time.perf_counter() - t0

    small = timed(16)
    large = timed(1008)  # 4096 nodes in all
    assert large < 3 * small + 0.05, (small, large)


async def test_routed_job_gets_assigned_then_relayed(tracker):
    t = tracker
    req = fake_conn(t, None)
    t.ledger.ensure_account(req.node_id, req.info.pubkey)
    peers = [fake_conn(t, MODELS[f], f) for f in ("qwen", "smollm", "gemma")]
    group = "ab" * 16
    jobs = [signed_job(req.identity) for _ in range(4)]
    for j in jobs:
        t._dispatch(req, JobFrame(job=j, route=Route(group=group)))
    frames = sent_frames(req)
    assigned = [f for f in frames if isinstance(f, Assigned)]
    errors = [f for f in frames if isinstance(f, JobError)]
    assert len(assigned) == 3 and [e.error for e in errors] == ["no_peer"]  # only 3 families exist
    assert len({a.peer.family for a in assigned}) == 3
    assert assigned[0].peer.family == "smollm" and assigned[0].peer.reliability == pytest.approx(0.865, abs=0.01)
    for p in peers:  # each chosen node got its job, with the requester's key
        got = [f for f in sent_frames(p) if isinstance(f, JobFrame)]
        assert len(got) == 1 and got[0].requester_pubkey == req.info.pubkey and p.busy == 1
    # A replacement of the smollm job: no family left, so the same family, never the same node.
    extra = fake_conn(t, MODELS["smollm"], "smollm")
    rj = signed_job(req.identity)
    t._dispatch(req, JobFrame(job=rj, route=Route(group=group, replaces=assigned[0].job_id,
                                                  exclude=[assigned[0].peer.node_id])))
    (a2,) = [f for f in sent_frames(req) if isinstance(f, Assigned)]
    assert a2.peer.node_id == extra.node_id
    # A plain (non-replacement) job of the same group gets nothing: every family is used.
    t._dispatch(req, JobFrame(job=signed_job(req.identity), route=Route(group=group)))
    assert [f.error for f in sent_frames(req)] == ["no_peer"]


def test_targeted_job_to_a_suspended_node_is_refused(tracker):
    """An essaim/1 gateway may still name a suspended node from its directory (audit Codex)."""
    t = tracker
    req = fake_conn(t, None)
    t.ledger.ensure_account(req.node_id, req.info.pubkey)
    c = fake_conn(t, MODELS["qwen"], "qwen")
    t._suspend(c, "test")
    j = signed_job(req.identity)
    t._dispatch(req, JobFrame(job=j, target=c.node_id))
    assert [f.error for f in sent_frames(req)] == ["peer_suspended"] and c.busy == 0 and j.job_id not in t.jobs


def test_excluded_reputation_recovers_when_the_baseline_moves(tracker):
    """Reputations follow the model's baseline, even for a node no longer selected (audit Codex)."""
    t = tracker
    c = fake_conn(t, MODELS["qwen"], "qwen")
    c.spot = [4, 16]  # 80 % disagreement against a 5 % prior baseline
    t._spotted.add(c)
    t._refresh_reputation(c)
    assert c.reputation < 0.3 and c.slot is None
    t._mspot[MODELS["qwen"]] = [4 + 100, 16 + 400]  # the other nodes of the model disagree as much
    t._rep_at = -1e9
    t.sweep()
    assert c.reputation == 1.0 and c.slot is not None


def test_spot_checker_found_among_partly_busy_nodes(tracker):
    """Idle nodes serving another GGUF must not hide a compatible, partly busy checker (audit Codex)."""
    t = tracker
    other = [fake_conn(t, MODELS["qwen"], "qwen") for _ in range(5)]
    for o in other:
        o.info = o.info.model_copy(update={"gguf": "other.gguf"})
    same = fake_conn(t, MODELS["qwen"], "qwen", max_parallel=2)
    same.busy = 1
    t._reindex(same)
    pool = t._pools[MODELS["qwen"]]
    assert same in pool.partial and all(o in pool.idle for o in other)
    assert all(t._spot_checker(pool, "qwen.gguf", set()) is same for _ in range(20))
    assert t._spot_checker(pool, "qwen.gguf", {same.node_id}) is None


def test_jobframe_without_route_stays_essaim1(tracker):
    """A node of essaim/1.1 talking to an essaim/1 tracker: no unknown field in its frames."""
    ident = Identity.generate()
    raw = dump_frame(JobFrame(job=signed_job(ident), target="cd" * 16))
    assert '"route"' not in raw
    routed = dump_frame(JobFrame(job=signed_job(ident), route=Route(group="ab" * 16)))
    assert isinstance(parse_frame(routed), JobFrame) and '"route"' in routed


async def test_peers_snapshot_is_cached_and_chunked(tracker):
    t = tracker
    t.peers_snapshot_s = 0.2
    for i in range(600):  # more than two chunks
        fake_conn(t, MODELS["qwen"], "qwen")
    body = json.loads(await t.peers_snapshot())
    assert len(body["peers"]) == 600 and {"busy", "reputation", "suspended", "pubkey"} <= set(body["peers"][0])
    fake_conn(t, MODELS["gemma"], "gemma")
    again = json.loads(await t.peers_snapshot())
    assert len(again["peers"]) == 600  # cached: the new node is not there yet
    await asyncio.sleep(0.25)
    stale = json.loads(await t.peers_snapshot())  # stale one served while the rebuild starts
    assert len(stale["peers"]) == 600
    await t._snapshot_task
    assert len(json.loads(await t.peers_snapshot())["peers"]) == 601


async def test_gateway_uses_tracker_selection_not_directory(tmp_path):
    s = await start_swarm(tmp_path)
    try:
        for fam in ("qwen", "smollm", "gemma", "granite"):
            await s.add_node(fam, FakeEngine(MODELS[fam], "The answer is 42."), MODELS[fam])
        gw, client = await s.add_gateway()
        calls: list[str] = []

        async def spy(request):
            calls.append(request.url.path)

        gw.http.event_hooks["request"].append(spy)
        r = await client.post("/v1/chat/completions", json={"messages": [MATH], "myriad": {"k": 4}})
        meta = r.json()["myriad"]
        assert r.status_code == 200 and meta["routing"] == "tracker" and meta["answer"] == "42"
        assert len({p["family"] for p in meta["peers"]}) == meta["peers_asked"] >= 2
        assert "/v1/peers" not in calls and "/v1/reliability" not in calls and calls.count("/v1/health") == 1
        async with httpx.AsyncClient(base_url=s.url) as http:
            sel = (await http.get("/v1/select", params={"k": 4})).json()["peers"]
            assert [p["family"] for p in sel] == ["smollm", "gemma", "qwen", "granite"]
            ex = (await http.get("/v1/select", params={"k": 4, "exclude": sel[0]["node_id"]})).json()["peers"]
            assert sel[0]["node_id"] not in {p["node_id"] for p in ex} and len(ex) == 3
            assert (await http.get("/v1/select", params={"exclude": "zz"})).status_code == 422
            h = (await http.get("/v1/health")).json()
            assert h["protocol"] == "essaim/1" and h["protocol_version"] == "essaim/1.1" and "route" in h["features"]
        # An essaim/1-style gateway (directory) still works against the new tracker.
        legacy = Gateway(gw.node, routing="directory", peers_ttl_s=0.0)
        ans = await legacy.ask([MATH], k=4)
        assert ans.meta["routing"] == "directory" and ans.meta["answer"] == "42"
        await legacy.close()
    finally:
        await s.close()


# ============================================================ B. hung-node detection
def test_timeout_strikes_suspend_with_exponential_backoff(tracker):
    t = tracker
    t.suspend_base_s, t.timeout_strikes = 10.0, 2
    c = fake_conn(t, MODELS["qwen"], "qwen")
    c.pings = False  # an essaim/1 node: readmitted by timer
    t._strike(c, "timeout")
    assert c.suspended_until is None and c.slot is not None  # one timeout is not enough
    t._strike(c, "timeout")
    assert c.suspended_until is not None and c.slot is None
    assert c.suspended_until - time.monotonic() == pytest.approx(10.0, abs=0.5)
    t._readmit(c)
    assert c.slot is not None
    t._strike(c, "timeout")  # on probation: one more strike suspends it again, twice as long
    assert c.suspended_until - time.monotonic() == pytest.approx(20.0, abs=0.5)
    t._readmit(c)
    t._strike(c, "timeout")
    assert c.suspended_until - time.monotonic() == pytest.approx(40.0, abs=0.5)
    # Readmission of a node that does not answer pings happens on a timer.
    c.suspended_until = time.monotonic() - 0.01
    t._schedule(c.suspended_until, "readmit", c)
    t.sweep()
    assert c.suspended_until is None and c.slot is not None
    assert t.health_events["suspend_timeout"] == 3 and t.health_events["readmitted"] == 3


def test_short_deadlines_are_not_strikes(tracker):
    """A requester cannot get honest nodes suspended by giving them impossible deadlines."""
    t = tracker
    req = fake_conn(t, None)
    t.ledger.ensure_account(req.node_id, req.info.pubkey)
    c = fake_conn(t, MODELS["qwen"], "qwen")
    for _ in range(3):
        j = signed_job(req.identity, deadline_ms=500)
        t._dispatch(req, JobFrame(job=j, target=c.node_id))
        t._dispatch(c, JobError(job_id=j.job_id, node_id=c.node_id, error="deadline"))
    assert c.strikes == 0 and c.suspended_until is None


async def test_expired_jobs_are_strikes_and_swept_by_heap(tracker):
    t = tracker
    t.strike_min_deadline_s, t.job_ttl_s = 0.0, 0.0
    req = fake_conn(t, None)
    t.ledger.ensure_account(req.node_id, req.info.pubkey)
    c = fake_conn(t, MODELS["qwen"], "qwen")
    jobs = []
    for _ in range(2):
        j = signed_job(req.identity, deadline_ms=100)
        t._dispatch(req, JobFrame(job=j, target=c.node_id))
        jobs.append(j)
        tj = t.jobs[j.job_id]
        tj.deadline = time.monotonic() - 2  # its deadline (and the 1 s grace) are over
        t._deadlines = [(tj.deadline + 1.0, 0, j.job_id)] + [x for x in t._deadlines if x[2] != j.job_id]
        t.sweep()
        c.busy = 0
        t._reindex(c)
    errs = [f for f in sent_frames(req) if isinstance(f, JobError)]
    assert [e.error for e in errs] == ["deadline", "deadline"]
    assert c.suspended_until is not None and t.health_events["suspend_timeout"] == 1
    await asyncio.sleep(0.05)  # Windows' monotonic clock ticks by 15.6 ms
    t.sweep()  # job TTL 0: finished jobs are forgotten
    assert not any(j.job_id in t.jobs for j in jobs) and not req.jobs_out and not c.jobs_in


class SilentNode(NodeClient):
    """Connected, but its event loop no longer answers pings (an essaim/1.1 node that froze)."""
    frozen = False

    async def _pong(self, seq):
        if not self.frozen:
            await super()._pong(seq)


class LegacyNode(NodeClient):
    """An essaim/1 node: ignores the frames it does not know (pings)."""

    async def _dispatch(self, frame):
        if isinstance(frame, Ping):
            return
        await super()._dispatch(frame)


async def test_hung_nodes_are_suspended_and_readmitted(tmp_path, monkeypatch):
    monkeypatch.setattr(node_mod, "PROBE_TIMEOUT_S", 0.1)
    s = await start_swarm(tmp_path, ping_s=0.2, ping_timeout_s=0.3, suspend_base_s=0.6)
    try:
        frozen = await s.add_node("frozen", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"],
                                  cls=SilentNode)
        stuck_engine = FakeEngine(MODELS["gemma"], "The answer is 42.")
        stuck = await s.add_node("stuck", stuck_engine, MODELS["gemma"])
        legacy = await s.add_node("legacy", FakeEngine(MODELS["granite"], "The answer is 42."), MODELS["granite"],
                                  cls=LegacyNode)
        await s.add_node("smollm", FakeEngine(MODELS["smollm"], "The answer is 42."), MODELS["smollm"])
        conns = s.tracker.conns
        await wait_until(lambda: conns[frozen.node_id].pings and conns[stuck.node_id].pings, 5)
        await wait_until(lambda: conns[legacy.node_id].pings is False, 5)  # one unanswered ping: essaim/1
        frozen.frozen = True  # no Pong any more
        stuck_engine.hung = True  # the engine's health probe hangs: Pong(engine_ok=False)
        await wait_until(lambda: conns[frozen.node_id].suspended_until and conns[stuck.node_id].suspended_until, 5)
        assert conns[frozen.node_id].suspend_reason == "ping_missed"
        assert conns[stuck.node_id].suspend_reason == "engine_down"
        assert conns[legacy.node_id].suspended_until is None  # never judged on pings
        gw, client = await s.add_gateway()
        ans = await gw.ask([MATH], k=4)
        asked = {p["node_id"] for p in ans.meta["peers"]}
        assert asked == {legacy.node_id, s.nodes["smollm"].node_id}
        # Still down at the end of the suspension: suspended again, for longer.
        await wait_until(lambda: s.tracker.health_events["suspend_engine_down"] >= 2, 5)
        assert conns[stuck.node_id].level >= 2
        # Back to health: readmitted after a good Pong.
        frozen.frozen = False
        stuck_engine.hung = False
        await wait_until(lambda: conns[frozen.node_id].suspended_until is None
                         and conns[stuck.node_id].suspended_until is None, 10)
        ans = await gw.ask([MATH], k=4)
        assert ans.meta["peers_asked"] == 4
    finally:
        await s.close()


async def test_engine_hang_with_healthy_probe_is_caught_by_timeouts(tmp_path):
    """Generation stuck but health probe fine: the job timeouts suspend the node."""
    s = await start_swarm(tmp_path, strike_min_deadline_s=0.5, timeout_strikes=2, suspend_base_s=30)
    try:
        eng = FakeEngine(MODELS["qwen"], "The answer is 42.", delay_s=3600)
        q = await s.add_node("qwen", eng, MODELS["qwen"])
        gw, client = await s.add_gateway()
        for _ in range(2):
            r = await client.post("/v1/chat/completions", json={"model": MODELS["qwen"], "messages": [MATH],
                                                                "myriad": {"timeout_s": 1.0}})
            assert r.status_code == 502 and "deadline" in r.json()["error"]["message"]
            await asyncio.sleep(0.1)  # the node's slot is freed
        c = s.tracker.conns[q.node_id]
        await wait_until(lambda: c.suspended_until is not None, 3)
        assert c.suspend_reason == "timeout"
        r = await client.post("/v1/chat/completions", json={"model": MODELS["qwen"], "messages": [MATH]})
        assert r.status_code == 503 and r.json()["error"]["type"] == "no_peers"
    finally:
        await s.close()


async def test_cancel_frees_the_node_slot_at_once():
    """The tracker may route the next job right behind a Cancel: the node must not answer busy."""
    a, req = Identity.generate(), Identity.generate()
    n = NodeClient(a, "http://127.0.0.1:1", engine=FakeEngine("x/y", delay_s=3600), model="x/y", max_parallel=1)
    sent = []

    class WS:
        async def send(self, text):
            sent.append(parse_frame(text))

    n._ws = WS()
    j1, j2 = signed_job(req), signed_job(req)
    await n._dispatch(JobFrame(job=j1, requester_pubkey=req.pubkey))
    await n._dispatch(Cancel(job_id=j1.job_id))
    await n._dispatch(JobFrame(job=j2, requester_pubkey=req.pubkey))
    assert not [f for f in sent if isinstance(f, JobError)] and set(n.running) == {j2.job_id}
    for task in list(n.running.values()):
        task.cancel()
    await asyncio.sleep(0)


# ============================================================ C. replacement
class RefusingNode(NodeClient):
    """Accepted by the tracker, then refuses every job itself (a node-side 'busy')."""

    async def _on_job(self, f):
        await self._reject(f.job, "busy")


async def test_refused_peer_is_replaced_in_an_unused_family(tmp_path):
    s = await start_swarm(tmp_path)
    try:
        ref = await s.add_node("smollm", FakeEngine(MODELS["smollm"]), MODELS["smollm"], cls=RefusingNode)
        await s.add_node("gemma", FakeEngine(MODELS["gemma"], "The answer is 42."), MODELS["gemma"])
        await s.add_node("qwen", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"])
        gw, client = await s.add_gateway()
        ans = await gw.ask([MATH], k=2)
        peers = {p["family"]: p for p in ans.meta["peers"]}
        assert set(peers) == {"smollm", "gemma", "qwen"} and ans.meta["replacements"] == 1
        assert peers["smollm"]["status"] == "erreur" and peers["smollm"]["error"] == "busy"
        assert peers["qwen"]["replaces"] is not None and peers["qwen"]["status"] in ("ok", "annulé")
        assert ans.meta["answer"] == "42" and ans.meta["peers_answered"] >= 1
        assert ref.node_id not in {p["node_id"] for p in ans.meta["peers"] if p["status"] == "ok"}
        # Without replacement (ablation): the refused peer is simply lost.
        gw.replace = False
        ans = await gw.ask([MATH], k=2)
        assert ans.meta["replacements"] == 0 and ans.meta["peers_asked"] == 2
    finally:
        await s.close()


async def test_refused_peer_is_replaced_in_the_same_family_once(tmp_path):
    s = await start_swarm(tmp_path)
    try:
        await s.add_node("r1", FakeEngine(MODELS["qwen"]), MODELS["qwen"], cls=RefusingNode)
        await s.add_node("r2", FakeEngine(MODELS["qwen"]), MODELS["qwen"], cls=RefusingNode)
        await s.add_node("ok", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"])
        gw, client = await s.add_gateway()
        for routing in ("tracker", "directory"):
            gw.routing = routing
            got = []
            # The first peer is drawn at random: ask until a refuser was drawn first and replaced by the
            # node that answers (each ask has a probability of about 1/3 of doing so; 6 asks were not
            # enough to make the test deterministic: it failed about once in 8 runs).
            for _ in range(40):
                try:
                    ans = await gw.ask([MATH], model=MODELS["qwen"])
                    got.append(ans.meta)
                except Exception as e:  # both refusers drawn: one replacement only, then no answer
                    got.append({"error": str(e)})
                if got[-1].get("replacements") == 1 and got[-1].get("peers_answered") == 1:
                    break
            assert any(m.get("replacements") == 1 and m.get("peers_answered") == 1 for m in got), got
            for m in got:
                assert m.get("replacements", 0) <= 1
    finally:
        await s.close()


class FakeGatewayNode:
    """Just enough of a NodeClient for the gateway, with scripted tracker events."""

    def __init__(self):
        self.identity = Identity.generate()
        self.node_id = self.identity.node_id
        self.tracker_url = "http://127.0.0.1:1"
        self.connected = asyncio.Event()
        self.connected.set()
        self.session = 1
        self.waiters: dict = {}
        self.routed: list = []
        self.cancelled: list = []

    def register(self, ids, q):
        for j in ids:
            self.waiters[j] = q

    def unregister(self, ids):
        for j in ids:
            self.waiters.pop(j, None)

    async def route(self, job, route):
        self.routed.append((job, route))

    async def cancel(self, job_id):
        self.cancelled.append(job_id)

    async def send_receipt(self, rc):
        pass

    def push(self, frame):
        jid = frame.result.job_id if isinstance(frame, ResultFrame) else frame.job_id
        self.waiters[jid].put_nowait(frame)


def peer(fam: str, p: float):
    ident = Identity.generate()
    return ident, PeerCard(node_id=ident.node_id, pubkey=ident.pubkey, model=MODELS[fam], family=fam, reliability=p)


def result(ident: Identity, card: PeerCard, job_id: str, answer: str) -> ResultFrame:
    return ResultFrame(result=JobResult(job_id=job_id, node_id=card.node_id, model=card.model,
                                        text=f"The answer is {answer}.", completion_tokens=4).signed_by(ident))


@pytest.mark.parametrize("replacement_p,stops", [(0.5, True), (0.99, False)])
async def test_certificate_counts_pending_replacements(replacement_p, stops):
    """No early stop while a replacement's weight is unknown; once known, it counts as pending."""
    node = FakeGatewayNode()
    gw = Gateway(node, routing="tracker")
    try:
        task = asyncio.create_task(gw.ask([MATH], k=3, timeout_s=10))
        await wait_until(lambda: len(node.routed) == 3)
        # math (c = 0.05): w(0.6) = 3.41, w(0.5) = 3.00, w(0.99) = 7.59
        cards = [peer("smollm", 0.6), peer("gemma", 0.6), peer("qwen", 0.6)]
        jobs = [j for j, _ in node.routed]
        for (ident, card), job in zip(cards, jobs):
            node.push(Assigned(job_id=job.job_id, peer=card))
        node.push(JobError(job_id=jobs[2].job_id, node_id=cards[2][1].node_id, error="peer_busy"))
        await wait_until(lambda: len(node.routed) == 4)  # the replacement leaves
        rjob, rroute = node.routed[3]
        assert rroute.replaces == jobs[2].job_id and set(rroute.exclude) == {c.node_id for _, c in cards}
        assert rroute.group == node.routed[0][1].group
        for (ident, card), job in zip(cards[:2], jobs[:2]):
            node.push(result(ident, card, job.job_id, "42"))
        await asyncio.sleep(0.1)
        assert not task.done()  # the replacement's weight is unknown: no certificate yet
        rident, rcard = peer("granite", replacement_p)
        node.push(Assigned(job_id=rjob.job_id, peer=rcard))
        if stops:  # 2 x w(0.6) > w(0.5): certified with the replacement counted as pending
            ans = await asyncio.wait_for(task, 2)
            assert ans.meta["early_stop"] and ans.meta["certificate"] and ans.meta["replacements"] == 1
            st = {p["family"]: p["status"] for p in ans.meta["peers"]}
            assert st == {"smollm": "ok", "gemma": "ok", "qwen": "erreur", "granite": "annulé"}
            assert rjob.job_id in node.cancelled
        else:  # the replacement could still overturn the vote: wait for it
            await asyncio.sleep(0.1)
            assert not task.done()
            node.push(result(rident, rcard, rjob.job_id, "7"))
            ans = await asyncio.wait_for(task, 2)
            assert not ans.meta["early_stop"] and ans.meta["peers_answered"] == 3 and ans.meta["answer"] == "7"
    finally:
        await gw.close()


async def test_no_replacement_when_time_is_short_or_error_is_final():
    node = FakeGatewayNode()
    gw = Gateway(node, routing="tracker")
    try:
        task = asyncio.create_task(gw.ask([MATH], k=2, timeout_s=10))
        await wait_until(lambda: len(node.routed) == 2)
        (i1, c1), (i2, c2) = peer("smollm", 0.9), peer("gemma", 0.8)
        j1, j2 = (j for j, _ in node.routed)
        node.push(Assigned(job_id=j1.job_id, peer=c1))
        node.push(Assigned(job_id=j2.job_id, peer=c2))
        node.push(JobError(job_id=j2.job_id, node_id=c2.node_id, error="insufficient_credits"))
        node.push(result(i1, c1, j1.job_id, "42"))
        ans = await asyncio.wait_for(task, 2)
        assert ans.meta["replacements"] == 0 and len(node.routed) == 2
    finally:
        await gw.close()
