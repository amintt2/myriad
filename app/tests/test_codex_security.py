"""Regression tests of the Codex reviews of the security branch (audits/2026-10-09_audit_app_securite*.md)."""
from __future__ import annotations

import asyncio
import time

import pytest

import myriad.tracker as tracker_mod
from myriad.crypto import Identity
from myriad.e2e import Keyring, ReplayCache, open_job, seal_job, seal_result
from myriad.engine import FakeEngine
from myriad.node import NodeClient
from myriad.protocol import Cancel, Job, JobFrame, KxFrame, NodeInfo, Reserve, Route, SealedJobFrame
from myriad.security import Security, ServeGuard
from myriad.tracker import Tracker

from .conftest import MODELS, start_swarm, wait_until
from .test_scale_v11 import MATH, fake_conn, sent_frames


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=tmp_path / "t.sqlite", spot_rate=0.0, seed=3)
    yield t
    t.ledger.close()


def e2e_conn(t: Tracker, fam: str = "qwen"):
    c = fake_conn(t, MODELS[fam], fam, max_parallel=4)
    c.keyring = Keyring(c.identity)
    c.info = c.info.model_copy(update={"kx": c.keyring.cert()})
    t._reindex(c)
    return c


def requester(t: Tracker):
    r = fake_conn(t, None)
    t.ledger.ensure_account(r.node_id, r.info.pubkey)
    return r


def sealed_for(peer, req_ident) -> tuple:
    job = Job(job_id=Identity.generate().node_id, requester_id=req_ident.node_id, messages=[MATH], max_tokens=8,
              deadline_ms=30_000, task_hint="math")
    return seal_job(job, peer.node_id, peer.keyring.cert())


def test_a_cancelled_reservation_cannot_be_filled(tracker):
    t = tracker
    peer, req = e2e_conn(t), requester(t)
    jid = Identity.generate().node_id
    t._reserve(req, Reserve(job_id=jid, route=Route(group="1" * 32, e2e=True), max_tokens=8, deadline_ms=30_000))
    assert t.jobs[jid].reserved
    t._dispatch(req, Cancel(job_id=jid))
    sj, s = sealed_for(peer, req.identity)
    sj = sj.model_copy(update={"job_id": jid, "max_tokens": 8, "deadline_ms": 30_000}).signed_by(s.pseudonym)
    sent_frames(peer)
    t._submit_sealed(req, SealedJobFrame(job=sj, requester_pubkey=s.pseudonym.pubkey))
    assert not any(isinstance(f, SealedJobFrame) for f in sent_frames(peer))  # never relayed
    assert any(getattr(f, "error", None) == "duplicate_job" for f in sent_frames(req))
    assert peer.busy == 0


def test_a_result_racing_a_cancellation_is_still_accepted_and_paid(tracker):
    t = tracker
    t.receipt_grace_s = 0.0
    peer, req = e2e_conn(t), requester(t)
    sj, s = sealed_for(peer, req.identity)
    t._submit_sealed(req, SealedJobFrame(job=sj, requester_pubkey=s.pseudonym.pubkey))
    tj = t.jobs[sj.job_id]
    t._cancel(tj)  # the payload is dropped at once...
    assert tj.job.ct == "AAAA"  # its payload is dropped, its header kept (for a dispute)
    _, ps = open_job(sj, s.pseudonym.pubkey, peer.keyring, ReplayCache(), peer.node_id)
    sr = seal_result(peer.identity, ps, MODELS["qwen"], "The answer is 42.", "stop", 5, None, 1.0)
    t._on_result(peer, sr)  # ...but a late result still matches its kind
    assert tj.status == "delivered"
    t._delivered = type(t._delivered)((0.0, j) for _, j in t._delivered)
    t.sweep()
    assert tj.settled  # paid (it was relayed to the requester)


def test_plaintext_spot_check_survives_a_cancelled_original(tmp_path):
    t = Tracker(db_path=tmp_path / "t.sqlite", spot_rate=1.0, seed=1)
    try:
        a, b = fake_conn(t, MODELS["qwen"], "qwen"), fake_conn(t, MODELS["qwen"], "qwen")
        b.info = b.info.model_copy(update={"gguf": a.info.gguf})
        req = requester(t)
        job = Job(job_id=Identity.generate().node_id, requester_id=req.node_id, messages=[MATH], max_tokens=8,
                  temperature=0, deadline_ms=30_000, task_hint="math").signed_by(req.identity)
        t._submit(req, JobFrame(job=job, target=a.node_id))
        orig = t.jobs[job.job_id]
        assert orig.spot_job is not None
        t._cancel(orig)
        from myriad.protocol import JobResult
        for c, jid in ((a, job.job_id), (b, orig.spot_job)):
            res = JobResult(job_id=jid, node_id=c.node_id, model=MODELS["qwen"], text="The answer is 42.",
                            completion_tokens=4).signed_by(c.identity)
            t._on_result(c, res)  # no AttributeError on a stripped job
        assert orig.compared
    finally:
        t.ledger.close()


async def test_requester_quota_is_released_before_the_scrub():
    class SlowScrub(FakeEngine):
        async def scrub(self, busy=0):
            await asyncio.sleep(5)

    n = NodeClient(Identity.generate(), "http://127.0.0.1:1", engine=SlowScrub(MODELS["qwen"]), model=MODELS["qwen"],
                   security=Security({"max_concurrent": 1}))
    req = Identity.generate()
    job = Job(job_id="ab" * 16, requester_id=req.node_id, messages=[MATH])
    n._start(job)
    await wait_until(lambda: n.engine.calls == 1)
    await wait_until(lambda: n.guard.running == {}, 2)  # released while the scrub still runs
    assert n.guard.check(req.node_id, 10) is None
    for task in list(n.running.values()):
        task.cancel()


async def test_a_rotation_triggered_outside_the_heartbeat_is_announced():
    clock = [1_800_000_000.0]
    n = NodeClient(Identity.generate(), "http://127.0.0.1:1", engine=FakeEngine(MODELS["qwen"]), model=MODELS["qwen"])
    n.keyring = Keyring(n.identity, clock=lambda: clock[0])
    sent = []

    async def send(frame):
        sent.append(frame)
    n.send = send
    n._announced_kx = n.keyring.cert().kx
    clock[0] += 24 * 3600 + 1
    n.status()  # rotates the key through info()
    await n.announce_kx()
    await n.announce_kx()
    assert [type(f) for f in sent] == [KxFrame] and sent[0].kx.kx == n._announced_kx


def test_serve_guard_keeps_no_history_of_pseudonyms():
    g = ServeGuard(Security())
    for _ in range(10_000):
        assert g.check(Identity.generate().node_id, 10, pseudonymous=True) is None
    assert g.calls == {}
    clock = [0.0]
    g = ServeGuard(Security({"rate_per_min": 5}), clock=lambda: clock[0])
    for i in range(5000):
        clock[0] += 0.1
        g.check(f"{i:032x}", 10)
    assert len(g.calls) <= 4097  # idle requesters are dropped


async def test_an_older_tracker_with_a_policy_uses_local_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker_mod, "FEATURES", ("route", "ping", "select", "tags"))
    s = await start_swarm(tmp_path)
    try:
        q = await s.add_node("qwen", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"])
        await s.add_node("smollm", FakeEngine(MODELS["smollm"], "The answer is 42."), MODELS["smollm"])
        sec = Security({"require_e2e": False, "blocked": [{"node_id": q.node_id, "reason": "", "ts": 0}]})
        gw, _ = await s.add_gateway(security=sec)
        for _ in range(4):
            ans = await gw.ask([MATH], k=2)
            assert ans.meta["routing"] == "directory"
        assert q.engine.calls == 0  # the blocked node never received the question
    finally:
        await s.close()


# ================================================================ review of the protocol (audits/..._protocole.md)
class GarbageNode(NodeClient):
    """Answers encrypted jobs with a well-signed envelope of random bytes, claiming many tokens."""

    async def _execute(self, job, session=None):
        import base64
        import os
        from myriad.protocol import SealedResult, SealedResultFrame
        if session is None:
            return await super()._execute(job, session)
        sr = SealedResult(job_id=job.job_id, node_id=self.node_id, model=self.model, completion_tokens=500,
                          chunks=[base64.b64encode(os.urandom(4096)).decode()]).signed_by(self.identity)
        await self.send(SealedResultFrame(result=sr))
        self.running.pop(job.job_id, None)


async def test_an_undecryptable_answer_is_disputed_and_not_paid(tmp_path):
    s = await start_swarm(tmp_path, receipt_grace_s=0.3)
    try:
        bad = await s.add_node("bad", FakeEngine(MODELS["qwen"]), MODELS["qwen"], cls=GarbageNode)
        gw, _ = await s.add_gateway()
        with pytest.raises(Exception):
            await gw.ask([MATH], model=MODELS["qwen"])
        led = s.tracker.ledger
        await wait_until(lambda: any(e["kind"] == "refused" for e in led.recent(20)), 5)
        await asyncio.sleep(0.6)  # past the receipt grace: no automatic payment either
        assert led.balances()[bad.node_id] == pytest.approx(1000)
        assert led.balances()[gw.node.node_id] == pytest.approx(1000)
        assert s.tracker.health_events["refused_dispute"] == 1
    finally:
        await s.close()


def test_a_false_or_wrong_dispute_changes_nothing(tracker):
    from myriad.protocol import Dispute
    t = tracker
    peer, req = e2e_conn(t), requester(t)
    sj, s = sealed_for(peer, req.identity)
    t._submit_sealed(req, SealedJobFrame(job=sj, requester_pubkey=s.pseudonym.pubkey))
    _, ps = open_job(sj, s.pseudonym.pubkey, peer.keyring, ReplayCache(), peer.node_id)
    t._on_result(peer, seal_result(peer.identity, ps, MODELS["qwen"], "The answer is 42.", "stop", 5, None, 1.0))
    tj = t.jobs[sj.job_id]
    sent_frames(req)
    t._dispatch(req, Dispute(job_id=sj.job_id, eph_secret="11" * 32))  # not the job's secret
    assert not tj.settled and any("wrong_secret" in getattr(f, "error", "") for f in sent_frames(req))
    t._dispatch(req, Dispute(job_id=sj.job_id, eph_secret=s.eph_secret.hex()))  # the answer does open
    assert tj.settled and t.health_events["dispute_rejected"] == 1
    assert t.ledger.recent(5)[0]["kind"] == "auto" and t.ledger.recent(5)[0]["amount"] > 0


async def test_canaries_without_a_checker_use_the_known_answer_and_unopened_canaries_are_not_paid(tmp_path):
    s = await start_swarm(tmp_path, spot_rate=1.0, seed=4, canary_delay_s=(0.0, 0.05))
    try:
        await s.add_node("solo", FakeEngine(MODELS["qwen"], lambda m: "The answer is 42." if "Tom" in m[-1]["content"]
                                            else "The answer is 1.", tokens=4), MODELS["qwen"])
        gw, _ = await s.add_gateway()
        await gw.ask([MATH], model=MODELS["qwen"])
        await wait_until(lambda: s.tracker.canary_stats["known_wrong"] + s.tracker.canary_stats["known_ok"] == 1, 5)
        assert s.tracker.canary_stats["no_checker"] == 1 and s.tracker.canary_stats["known_wrong"] == 1
    finally:
        await s.close()
    s = await start_swarm(tmp_path / "b", spot_rate=1.0, seed=4, canary_delay_s=(0.0, 0.05))
    try:
        garbage = await s.add_node("g", FakeEngine(MODELS["qwen"]), MODELS["qwen"], cls=GarbageNode)
        s.tracker._run_canary(garbage.node_id, {"max_tokens": 64, "deadline_ms": 30_000, "bucket": 1024})
        await wait_until(lambda: s.tracker.canary_stats["undecryptable"] == 1, 5)
        assert s.tracker.ledger.balances()[garbage.node_id] == pytest.approx(1000)  # not paid
    finally:
        await s.close()


def test_a_late_garbage_answer_after_a_cancel_can_still_be_disputed(tracker):
    import base64
    import os
    from myriad.protocol import Dispute, SealedResult
    t = tracker
    peer, req = e2e_conn(t), requester(t)
    sj, s = sealed_for(peer, req.identity)
    t._submit_sealed(req, SealedJobFrame(job=sj, requester_pubkey=s.pseudonym.pubkey))
    tj = t.jobs[sj.job_id]
    t._cancel(tj)
    junk = SealedResult(job_id=sj.job_id, node_id=peer.node_id, model=MODELS["qwen"], completion_tokens=8,
                        chunks=[base64.b64encode(os.urandom(2048)).decode()]).signed_by(peer.identity)
    t._on_result(peer, junk)
    assert tj.status == "delivered" and len(tj.job.ct) == 4  # the payload is gone, the header kept
    t._dispatch(req, Dispute(job_id=sj.job_id, eph_secret=s.eph_secret.hex()))
    assert tj.settled and t.health_events["refused_dispute"] == 1
    assert t.ledger.recent(1)[0]["kind"] == "refused" and t.ledger.recent(1)[0]["amount"] == 0


def test_an_unban_reaches_the_running_tracker(tracker):
    t = tracker
    nid = Identity.generate().node_id
    t.ban(nid, "honeytoken (test)", "honeytoken")
    assert nid in t.banned
    t.ledger.remove_ban(nid)  # what `myriad tracker-bans --remove` does, from another process
    t._ban_checked = -1e9
    t.sweep()
    assert nid not in t.banned


def test_a_busy_peer_does_not_dodge_its_canary(tracker):
    t = tracker
    peer = e2e_conn(t)
    peer.busy = peer.info.max_parallel
    t._reindex(peer)
    t._run_canary(peer.node_id, {"max_tokens": 64, "deadline_ms": 30_000, "bucket": 1024})
    assert t.canary_stats["postponed"] == 1 and any(k == "canary" for _, _, k, _ in t._timers)


def test_a_claimed_truncation_must_be_credible():
    from myriad.tracker import spot_verdict, truncated_credibly
    assert not truncated_credibly("The answer is 999.", "length", 512)  # a short "cut-short" answer: a lie
    assert truncated_credibly("x " * 300, "length", 512)
    assert spot_verdict("The answer is 999.", "The answer is 42.", "math", [], ("stop", "stop")) is False


def test_tracker_validates_certificates_like_the_requesters(tracker):
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from myriad.e2e import make_cert
    from myriad.protocol import Hello
    ident = Identity.generate()
    long = make_cert(ident, X25519PrivateKey.generate(), int(time.time()), lifetime=365 * 86400)
    info = NodeInfo(node_id=ident.node_id, pubkey=ident.pubkey, model=MODELS["qwen"], kx=long)
    assert tracker._check_hello(Hello.make(ident, info, "ab" * 32), "ab" * 32) == "bad_kx"
    other = make_cert(ident, X25519PrivateKey.generate(), int(time.time()), model=MODELS["gemma"])
    info = info.model_copy(update={"kx": other})
    assert tracker._check_hello(Hello.make(ident, info, "ab" * 32), "ab" * 32) == "bad_kx"  # another model


async def test_the_tracker_cannot_relabel_a_peers_model_or_count_a_peer_twice(swarm, monkeypatch):
    t = swarm.tracker
    orig_card = t._card
    monkeypatch.setattr(t, "_card", lambda c, with_tags=False, full=False: orig_card(c, with_tags, full).model_copy(
        update={"model": MODELS["smollm"]}) if c.info.model == MODELS["gemma"] else orig_card(c, with_tags, full))
    gw, _ = await swarm.add_gateway(security=Security({"blocked_families": [{"family": "smollm", "reason": "", "ts": 0}]}))
    gw.security.settings.blocked_families = []  # the policy is not the point: the certificate names the model
    calls = swarm.engines["gemma"].calls
    ans = await gw.ask([MATH], k=4)
    gem = [p for p in ans.meta["peers"] if p["node_id"] == swarm.nodes["gemma"].node_id]
    assert gem and gem[0]["error"].startswith("policy:") and swarm.engines["gemma"].calls == calls
    monkeypatch.setattr(t, "_card", orig_card)
    qwen = t.conns[swarm.nodes["qwen"].node_id]
    monkeypatch.setattr(t, "select", lambda *a, **kw: [qwen])  # the same peer for every job of the request
    calls = swarm.engines["qwen"].calls
    ans = await gw.ask([MATH], k=3)
    assert swarm.engines["qwen"].calls == calls + 1 and ans.meta["peers_answered"] == 1


async def test_a_policy_aware_requesters_plaintext_job_is_never_duplicated(tmp_path):
    s = await start_swarm(tmp_path, spot_rate=1.0, seed=1)
    try:
        a = await s.add_node("a", FakeEngine(MODELS["qwen"], "The answer is 42.", tokens=4), MODELS["qwen"], e2e=False)
        b = await s.add_node("b", FakeEngine(MODELS["qwen"], "The answer is 42.", tokens=4), MODELS["qwen"], e2e=False)
        gw, _ = await s.add_gateway(security=Security({"require_e2e": False}))
        await gw.ask([MATH], model=MODELS["qwen"], temperature=0.0)
        await asyncio.sleep(0.3)
        assert a.engine.calls + b.engine.calls == 1  # no copy to a peer the requester did not check
    finally:
        await s.close()


async def test_tracker_bodies_are_limited_while_reading(swarm):
    import httpx
    async with httpx.AsyncClient(base_url=swarm.url) as http:
        r = await http.post("/v1/report", content=b"x" * 100_000, headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        r = await http.post("/v1/honeytoken", content=b"x" * 70_000)
        assert r.status_code == 413


async def test_a_plaintext_job_cannot_be_replayed_to_a_node(tmp_path):
    s = await start_swarm(tmp_path)
    try:
        node = await s.add_node("qwen", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"], e2e=False)
        req = Identity.generate()
        job = Job(job_id="ef" * 16, requester_id=req.node_id, messages=[MATH]).signed_by(req)
        frame = JobFrame(job=job, requester_pubkey=req.pubkey)
        rejected = []
        orig = node._reject

        async def spy(j, err):
            rejected.append(err)
        node._reject = spy
        await node._on_job(frame)
        await wait_until(lambda: node.engine.calls == 1)
        await node._on_job(frame)  # a malicious tracker sends it again
        assert rejected == ["replay"] and node.engine.calls == 1
        node._reject = orig
    finally:
        await s.close()
