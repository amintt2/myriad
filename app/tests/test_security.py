"""essaim/1.3 end to end, without GPU or network: what the tracker sees, canaries, policies, limits,
reports, honeytokens, the privacy guard and the local-only mode."""
from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
import pytest

import myriad.tracker as tracker_mod
from myriad import canary
from myriad.crypto import Identity
from myriad.engine import FakeEngine
from myriad.gateway import GatewayError
from myriad.protocol import NodeReport, SealedJobFrame, dump_frame
from myriad.security import Security, swarm_key

from .conftest import MODELS, start_swarm, wait_until

SECRET = "Zanzibar-7731"  # a marker that must never be readable by the tracker
Q = {"role": "user", "content": f"Code {SECRET}: Tom has 40 apples and buys 2 more. How many apples does he have?"}


def capture(tracker, monkeypatch) -> list[tuple[str, str, str]]:
    """Every frame the tracker receives or sends, as JSON: (direction, node id, text)."""
    seen: list[tuple[str, str, str]] = []
    orig_dispatch, orig_send = tracker._dispatch, tracker_mod.Conn.send

    def dispatch(conn, frame):
        seen.append(("in", conn.node_id, dump_frame(frame)))
        return orig_dispatch(conn, frame)

    def send(self, frame):
        seen.append(("out", self.node_id, dump_frame(frame)))
        return orig_send(self, frame)

    monkeypatch.setattr(tracker, "_dispatch", dispatch)
    monkeypatch.setattr(tracker_mod.Conn, "send", send)
    return seen


def keys(obj, prefix="") -> set:
    """The JSON structure (paths of keys) of a frame."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(prefix + k)
            out |= keys(v, prefix + k + ".")
    return out


async def test_tracker_never_sees_plaintext(swarm, monkeypatch):
    seen = capture(swarm.tracker, monkeypatch)
    gw, client = await swarm.add_gateway()
    r = await client.post("/v1/chat/completions", json={"model": "myriad", "messages": [Q]})
    assert r.status_code == 200 and r.json()["myriad"]["answer"] == "42"
    meta = r.json()["myriad"]
    assert meta["privacy"]["e2e"] and all(p["e2e"] for p in meta["peers"])
    # the peers ran the question in clear (they must), the tracker never saw it
    assert any(SECRET in e.last_messages[-1]["content"] for e in swarm.engines.values() if e.last_messages)
    await wait_until(lambda: any('"t":"receipt"' in t for _, _, t in seen), 5)
    texts = [t for _, _, t in seen]
    assert texts and not any(SECRET in t or "apples" in t or "The answer is" in t for t in texts)
    kinds = {json.loads(t)["t"] for t in texts}
    assert {"reserve", "sjob", "sresult", "assigned"} <= kinds and "job" not in kinds and "result" not in kinds
    # the peer only saw a pseudonym, never the requester's node id
    client_id = swarm.nodes["client"].node_id
    relayed = [json.loads(t) for d, n, t in seen if d == "out" and '"t":"sjob"' in t]
    assert relayed and all(f["job"]["requester_id"] != client_id for f in relayed)
    assert len({f["job"]["requester_id"] for f in relayed}) == len(relayed)  # one pseudonym per job
    # nor does the public ledger name the requester, nor does the evidence give the content
    async with httpx.AsyncClient(base_url=swarm.url) as http:
        assert all("requester_id" not in e for e in (await http.get("/v1/ledger")).json()["entries"])
        jid = next(e["job_id"] for e in swarm.tracker.ledger.recent(50) if e["kind"] == "receipt")
        ev = (await http.get(f"/v1/evidence/{jid}")).json()
        assert "result" not in ev and "receipt" not in ev and ev["result_sha256"] and ev["encrypted"]


async def test_canaries_are_indistinguishable_and_feed_reputation(tmp_path, monkeypatch):
    s = await start_swarm(tmp_path, spot_rate=1.0, seed=2, canary_delay_s=(0.0, 0.05))
    try:
        seen = capture(s.tracker, monkeypatch)
        a = await s.add_node("a", FakeEngine(MODELS["qwen"], "The answer is 42.", tokens=4), MODELS["qwen"])
        b = await s.add_node("b", FakeEngine(MODELS["qwen"], "The answer is 42.", tokens=4), MODELS["qwen"])
        gw, client = await s.add_gateway()
        r = await client.post("/v1/chat/completions", json={"model": MODELS["qwen"], "messages": [Q]})
        assert r.status_code == 200
        await wait_until(lambda: s.tracker.canary_stats["agree"] == 1, 5)
        st = s.tracker.ledger
        assert st.node_spot(a.node_id, MODELS["qwen"]) == (1, 0) and st.node_spot(b.node_id, MODELS["qwen"]) == (1, 0)
        assert st.model_spot_stats()[MODELS["qwen"]] == (1, 0)
        jobs = [json.loads(t) for d, n, t in seen if d == "out" and '"t":"sjob"' in t]
        assert len(jobs) == 3  # the real job, then the canary to its peer and to a checker
        real, canaries = jobs[0], jobs[1:]
        for c in canaries:
            assert keys(c) == keys(real)  # same frame shape
            assert len(c["job"]["ct"]) == len(real["job"]["ct"])  # same padded size
            assert (c["job"]["max_tokens"], c["job"]["deadline_ms"]) == (real["job"]["max_tokens"],
                                                                         real["job"]["deadline_ms"])
            assert c["job"]["requester_id"] != s.tracker.identity.node_id  # not the tracker's id: a pseudonym
        assert len({j["job"]["requester_id"] for j in jobs}) == 3
        # the peers' ledger view cannot tell the canary apart either: same kind, no requester shown
        async with httpx.AsyncClient(base_url=s.url) as http:
            await wait_until(lambda: len([e for e in st.recent(50) if e["kind"] == "receipt"]) == 3, 5)
            kinds = {e["kind"] for e in (await http.get("/v1/ledger")).json()["entries"] if e["kind"] != "starter"}
            assert kinds == {"receipt"}
    finally:
        await s.close()


async def test_honeytoken_sighting_bans_the_leaking_peer(tmp_path, monkeypatch):
    s = await start_swarm(tmp_path, spot_rate=1.0, seed=3, canary_delay_s=(0.0, 0.05),
                          public_url="https://tracker.example.org")
    orig = canary.draw
    monkeypatch.setattr(canary, "draw", lambda rng: {**orig(rng), "honey": True, "key": True})
    try:
        a = await s.add_node("a", FakeEngine(MODELS["qwen"], "The answer is 42.", tokens=4), MODELS["qwen"])
        await s.add_node("b", FakeEngine(MODELS["qwen"], "The answer is 42.", tokens=4), MODELS["qwen"])
        gw, client = await s.add_gateway()
        await client.post("/v1/chat/completions", json={"model": MODELS["qwen"], "messages": [Q]})
        await wait_until(lambda: s.tracker.canary_stats["sent"] == 1, 5)
        rows = s.tracker.ledger.db.execute("SELECT token, node_id FROM honeytokens WHERE node_id=?",
                                           (a.node_id,)).fetchall()
        assert {r["token"][:3] for r in rows} >= {"/h/", "myr"}  # url and key recorded for this peer
        await wait_until(lambda: s.tracker.canary_stats["agree"] == 1, 5)  # a got the canary last
        leaked = a.engine.last_messages[-1]["content"]  # what peer a saw (and leaked, in this story)
        assert "myr_live_" in leaked and "/h/" in leaked
        async with httpx.AsyncClient(base_url=s.url) as http:
            found = (await http.post("/v1/honeytoken", json={"text": "seen on a paste site: " + leaked})).json()
            assert found["found"] >= 2
        assert a.node_id in s.tracker.banned
        await wait_until(lambda: a.node_id not in s.tracker.conns, 5)
        async with httpx.AsyncClient(base_url=s.url) as http:  # a random token is not recognised
            assert (await http.post("/v1/honeytoken", json={"text": "x.0123456789@tracker.example.org"})).json() == {"found": 0}
    finally:
        await s.close()


@pytest.mark.parametrize("routing", ["tracker", "directory"])
async def test_require_e2e_excludes_plaintext_peers(tmp_path, routing):
    s = await start_swarm(tmp_path)
    try:
        old = await s.add_node("old", FakeEngine(MODELS["qwen"], "The answer is 42."), MODELS["qwen"], e2e=False)
        new = await s.add_node("new", FakeEngine(MODELS["smollm"], "The answer is 42."), MODELS["smollm"])
        gw, _ = await s.add_gateway(routing=routing)
        for _ in range(3):
            ans = await gw.ask([Q], k=2)
            assert {p["node_id"] for p in ans.meta["peers"]} == {new.node_id}
        assert old.engine.calls == 0
        # mixed network, plaintext allowed by the user: the old peer is used, in clear, the new one encrypted
        lax, _ = await s.add_gateway("lax", routing=routing, security=Security({"require_e2e": False}))
        ans = await lax.ask([Q], k=2)
        e2e = {p["node_id"]: p["e2e"] for p in ans.meta["peers"]}
        assert e2e == {old.node_id: False, new.node_id: True} and old.engine.calls == 1
    finally:
        await s.close()


@pytest.mark.parametrize("routing", ["tracker", "directory"])
async def test_blocklist_allowlist_and_private_swarm(tmp_path, routing):
    s = await start_swarm(tmp_path)
    try:
        key = swarm_key("our lab's long shared secret")
        nodes = {}
        for fam in ("qwen", "smollm", "gemma", "granite"):
            sec = Security({"swarm_key": key}) if fam in ("gemma", "granite") else None
            nodes[fam] = await s.add_node(fam, FakeEngine(MODELS[fam], "The answer is 42."), MODELS[fam], security=sec)
        sec = Security({"blocked": [{"node_id": nodes["qwen"].node_id, "reason": "test", "ts": 0}],
                        "blocked_families": [{"family": "smollm", "reason": "", "ts": 0}]})
        gw, _ = await s.add_gateway(routing=routing, security=sec)
        ans = await gw.ask([Q], k=4)
        asked = {p["node_id"] for p in ans.meta["peers"]}
        assert asked == {nodes["gemma"].node_id, nodes["granite"].node_id}
        assert nodes["qwen"].engine.calls == nodes["smollm"].engine.calls == 0
        sec.settings.blocked, sec.settings.blocked_families = [], []
        sec.settings.trusted_only = True
        sec.trust(nodes["smollm"].node_id)
        ans = await gw.ask([Q], k=4)
        assert {p["node_id"] for p in ans.meta["peers"]} == {nodes["smollm"].node_id}
        sec.settings.trusted_only = False
        sec.settings.swarm_key = key  # private swarm: only the members proving the key
        ans = await gw.ask([Q], k=4)
        assert {p["node_id"] for p in ans.meta["peers"]} == {nodes["gemma"].node_id, nodes["granite"].node_id}
    finally:
        await s.close()


async def test_a_malicious_tracker_cannot_substitute_keys_or_force_a_blocked_peer(swarm, monkeypatch):
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from myriad.e2e import make_cert
    t = swarm.tracker
    evil = Identity.generate()
    orig_card = t._card

    def card(c, with_tags=False, full=False):  # the tracker's own key, presented as the peer's
        pc = orig_card(c, with_tags, full)
        if full:
            fake = make_cert(evil, X25519PrivateKey.generate(), int(time.time())).model_copy(update={"node_id": c.node_id})
            pc = pc.model_copy(update={"kx": fake})
        return pc

    monkeypatch.setattr(t, "_card", card)
    gw, _ = await swarm.add_gateway()
    calls = {n: e.calls for n, e in swarm.engines.items()}
    with pytest.raises(GatewayError):
        await gw.ask([Q], k=2)
    assert {n: e.calls for n, e in swarm.engines.items()} == calls  # nobody received the job
    monkeypatch.setattr(t, "_card", orig_card)
    # the tracker ignores the blocklist and assigns the blocked peer anyway: the gateway sends nothing to it
    qwen = swarm.nodes["qwen"].node_id
    gw.security.block(qwen, "test")
    monkeypatch.setattr(t, "_eligible", lambda c, pol: True)
    monkeypatch.setattr(t, "select", lambda *a, **kw: [t.conns[qwen]] if t.conns[qwen].busy == 0 else [])
    with pytest.raises(GatewayError):
        await gw.ask([Q], k=1)
    assert swarm.engines["qwen"].calls == calls["qwen"]


async def test_serving_limits_are_enforced_per_paying_account(tmp_path):
    s = await start_swarm(tmp_path)
    try:
        sec = Security({"max_concurrent": 1, "rate_per_min": 2})
        node = await s.add_node("qwen", FakeEngine(MODELS["qwen"], "The answer is 42.", delay_s=0.3), MODELS["qwen"],
                                security=sec)
        await wait_until(lambda: s.tracker.conns[node.node_id].serve_rate == 2)
        gw, _ = await s.add_gateway()
        res = await asyncio.gather(*(gw.ask([Q], model=MODELS["qwen"]) for _ in range(2)), return_exceptions=True)
        assert sum(not isinstance(r, Exception) for r in res) == 1  # one at a time for this account
        await gw.ask([Q], model=MODELS["qwen"])  # the second call of the minute
        with pytest.raises(GatewayError):
            await gw.ask([Q], model=MODELS["qwen"])  # the third: rate limit (the pseudonyms do not help)
        assert node.engine.calls == 2
        other, _ = await s.add_gateway("other")
        assert (await other.ask([Q], model=MODELS["qwen"])).meta["peers_answered"] == 1  # another account
        sec.block(other.node.node_id, "abuse")
        await node.send_serve_policy()
        await wait_until(lambda: other.node.node_id in s.tracker.conns[node.node_id].serve_deny)
        with pytest.raises(GatewayError):
            await other.ask([Q], model=MODELS["qwen"])
    finally:
        await s.close()


def test_reports_are_weighted_and_never_ban(tmp_path):
    t = tracker_mod.Tracker(db_path=tmp_path / "t.sqlite", spot_rate=0)
    target = Identity.generate()
    t.ledger.ensure_account(target.node_id, target.pubkey)
    now = time.time()

    def reporter(age_days: float, spent: float) -> Identity:
        r = Identity.generate()
        t.ledger.ensure_account(r.node_id, r.pubkey)
        t.ledger.db.execute("UPDATE accounts SET created_at=? WHERE node_id=?", (now - age_days * 86400, r.node_id))
        if spent:
            t.ledger.settle(Identity.generate().node_id, r.node_id, target.node_id, "m", 1, int(spent * 1000), "receipt")
        return r

    def report(r, reason="abuse", job_id=None):
        rep = NodeReport(reporter_id=r.node_id, node_id=target.node_id, reason=reason, job_id=job_id,
                         ts=int(time.time())).signed_by(r)
        return t.accept_report(rep)

    sybils = [reporter(0, 0) for _ in range(20)]
    for r in sybils:
        ok, why, w = report(r)
        assert ok and w == 0.0  # fresh accounts weigh nothing
    assert t.report_scores.get(target.node_id, 0) == 0 and target.node_id not in t.banned
    assert report(sybils[0])[1] == "duplicate"
    forged = NodeReport(reporter_id=sybils[1].node_id, node_id=target.node_id, reason="spam",
                        ts=int(time.time())).signed_by(Identity.generate())
    assert t.accept_report(forged)[1] == "bad_signature"
    assert report(sybils[2], "disagreement", "ab" * 16)[1] == "job_not_paid"  # must be a job it paid for
    veterans = [reporter(30, 100) for _ in range(4)]
    for r in veterans[:2]:
        assert report(r)[2] == 1.0
    assert t.report_scores.get(target.node_id, 0) == 0  # fewer than 3 weighted reporters: nothing
    for r in veterans[2:]:
        report(r)
    assert t.report_scores[target.node_id] >= tracker_mod.REPORT_FLAG_SCORE
    assert target.node_id not in t.banned  # deprioritised, never banned by reports
    t.ledger.close()


async def test_privacy_guard_masks_and_restores(swarm, monkeypatch):
    seen = capture(swarm.tracker, monkeypatch)
    key = "sk-proj-abcdefghijklmnopqrstuvwx1234"
    for e in swarm.engines.values():
        e.reply = lambda m: f"Use CLE_1 carefully. The answer is 42. ({m[-1]['content'][:40]})"
    gw, client = await swarm.add_gateway()
    q = {"role": "user", "content": f"My key is {key}. Tom has 40 apples and buys 2 more. How many?"}
    r = await client.post("/v1/chat/completions", json={"messages": [q]})
    assert r.status_code == 200
    body = r.json()
    assert key in body["choices"][0]["message"]["content"]  # restored locally
    assert body["myriad"]["privacy"]["masked"] == {"api_key": 1}
    assert not any(key in e.last_messages[-1]["content"] for e in swarm.engines.values() if e.last_messages)
    assert not any(key in t for _, _, t in seen)
    # an e-mail (reported, not masked by default) needs a confirmation the first time
    q2 = {"role": "user", "content": "Write to bob@example.org: what is 2+2?"}
    r = await client.post("/v1/chat/completions", json={"messages": [q2]})
    assert r.status_code == 409 and r.json()["error"]["type"] == "confirmation_required"
    assert r.json()["error"]["privacy"]["confirm_types"] == ["email"]
    r = await client.post("/v1/chat/completions", json={"messages": [q2], "myriad": {"confirm_sensitive": True}})
    assert r.status_code == 200


async def test_local_only_mode_sends_nothing(swarm, monkeypatch):
    seen = capture(swarm.tracker, monkeypatch)
    gw, _ = await swarm.add_gateway(security=Security({"local_only": True}))
    before = len(seen)
    with pytest.raises(GatewayError) as e:  # a client-only node: no local model, nothing leaves
        await gw.ask([Q])
    assert e.value.code == "kept_local" and len(seen) == before
    qwen = swarm.nodes["qwen"]  # a serving node keeps the question for its own model
    gw2, _ = await swarm.add_gateway("local", security=Security({"privacy": {"modes": {"iban": "local"}}}))
    gw2.node.engine = qwen.engine
    calls = qwen.engine.calls
    before = len(seen)
    ans = await gw2.ask([{"role": "user", "content": "My IBAN is DE89370400440532013000, 40+2?"}])
    assert ans.meta["decision"] == "local" and qwen.engine.calls == calls + 1 and len(seen) == before


async def test_no_job_text_reaches_logs_or_disk(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    log_file = tmp_path / "all.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    s = await start_swarm(tmp_path)
    try:
        fail = FakeEngine(MODELS["granite"], "unused", fail=True)
        await s.add_node("granite", fail, MODELS["granite"])
        qwen = FakeEngine(MODELS["qwen"], f"The answer is 42 ({SECRET})", tokens=5)
        await s.add_node("qwen", qwen, MODELS["qwen"])
        gw, client = await s.add_gateway()
        r = await client.post("/v1/chat/completions", json={"messages": [Q], "myriad": {"k": 2}})
        assert r.status_code == 200 and SECRET in r.text
        await wait_until(lambda: any(e["kind"] == "receipt" for e in s.tracker.ledger.recent(20)), 5)
        assert qwen.scrubbed >= 1  # the engine's cache is erased after each job
    finally:
        await s.close()
        logging.getLogger().removeHandler(handler)
        handler.close()
    assert not any(SECRET in rec.getMessage() for rec in caplog.records)
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert SECRET.encode() not in f.read_bytes(), f


async def test_security_json_api(swarm, tmp_path):
    from myriad.config import Config
    from myriad.security import invite_code
    from myriad.ui import make_ui_app
    gw, _ = await swarm.add_gateway()
    node = swarm.nodes["client"]
    Config(tracker_url=swarm.url).save(tmp_path)
    app = make_ui_app(node, gw, Config(tracker_url=swarm.url), tmp_path, "tok")
    hdr = {"X-Myriad-Token": "tok"}
    qwen = swarm.nodes["qwen"].node_id
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401") as c:
        s = (await c.get("/api/security")).json()
        assert s["require_e2e"] and s["tracker_e2e"] and s["invite"].startswith("myr1-") and "swarm_key" not in s
        assert (await c.post("/api/security/block", json={"node_id": qwen})).status_code == 403  # token required
        r = await c.post("/api/security/block", json={"node_id": qwen, "reason": "test"}, headers=hdr)
        assert r.status_code == 200 and r.json()["security"]["blocked"][0]["node_id"] == qwen
        assert Config.load(tmp_path).security["blocked"][0]["reason"] == "test"  # persisted
        assert gw.security.is_blocked(qwen)
        ans = await gw.ask([Q], k=4)
        assert qwen not in {p["node_id"] for p in ans.meta["peers"]}
        await c.post("/api/security/unblock", json={"node_id": qwen}, headers=hdr)
        r = await c.post("/api/security/trust", json={"invite": invite_code(qwen), "label": "x"}, headers=hdr)
        assert r.json()["node_id"] == qwen
        r = await c.post("/api/security/swarm", json={"secret": "a long shared secret!"}, headers=hdr)
        assert r.json()["security"]["swarm"]["enabled"] and "a long" not in json.dumps(Config.load(tmp_path).security)
        assert (await c.post("/api/security/settings", json={"require_e2e": "yes"}, headers=hdr)).status_code == 400
        r = await c.post("/api/security/settings", json={"min_reliability": 0.5}, headers=hdr)
        assert r.json()["security"]["min_reliability"] == 0.5
        chk = (await c.post("/api/security/check", json={"text": "key sk-proj-abcdefghijklmnopqrstuvwx1234"},
                            headers=hdr)).json()
        assert chk["masked"] == {"api_key": 1} and "sk-proj" not in json.dumps(chk) and chk["readers"]["e2e_required"]
        r = await c.post("/api/security/confirm", json={"types": ["email"]}, headers=hdr)
        assert "email" in r.json()["security"]["privacy"]["confirmed"]
        assert (await c.get("/static/security.js")).status_code == 200
