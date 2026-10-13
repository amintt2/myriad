"""Free tier, second part: gateway queue, partial results, upfront refusal over HTTP, the free daily
allowance and the availability rewards (tracker side)."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

import myriad.canary as canary
from myriad.agents import AgentRun, parse_request
from myriad.crypto import account_params
from myriad.engine import FakeEngine
from myriad.gateway import GatewayError, _Slots
from myriad.ledger import Ledger, hour_of

from .conftest import MODELS, start_swarm, wait_until
from .test_agents import agent_reply, task
from .test_free_tier import swarm_of


# ================================================================ gateway queue
async def test_slots_queue_in_order_and_time_out():
    sl = _Slots()
    assert await sl.acquire(3, 1) == 0  # no cap known: no limit
    sl.set_cap(4)
    assert await sl.acquire(3, 1) == 3
    order = []

    async def want(name, n):
        got = await sl.acquire(n, 5)
        order.append(name)
        return got

    t1 = asyncio.create_task(want("a", 3))
    await asyncio.sleep(0.05)
    t2 = asyncio.create_task(want("b", 1))  # would fit (1 free) but a is first in line
    await asyncio.sleep(0.05)
    assert order == []
    sl.release(3)
    assert await t1 == 3 and await t2 == 1 and order == ["a", "b"]
    assert sl.used == 4
    with pytest.raises(GatewayError) as e:
        await sl.acquire(9, 0.2)  # clamped to the cap, still waits for room
    assert e.value.code == "too_many_jobs" and e.value.status == 429
    assert not sl._waiting


async def test_subtask_retries_when_the_tracker_still_counts_jobs_in_flight(tmp_path, monkeypatch):
    """The gateway queue and the tracker's count can disagree for a moment (a cancelled straggler not yet
    released): a too_many_jobs answer is waited out, not turned into a failed sub-task."""
    s = await swarm_of(tmp_path, 3)
    try:
        gw, _ = await s.add_gateway()
        real, calls = gw.ask, []

        async def flaky(*a, **kw):
            calls.append(1)
            if len(calls) <= 2:
                raise GatewayError(429, "trop de jobs", "too_many_jobs")
            return await real(*a, **kw)

        monkeypatch.setattr(gw, "ask", flaky)
        monkeypatch.setattr("myriad.agents.NO_PEER_PAUSE_S", 0.05)
        res = await AgentRun(gw, parse_request({"task": "x", "plan": [task(1, escalate=False)]})).execute()
        assert res["status"] == "ok" and len(calls) == 3
    finally:
        await s.close()


# ================================================================ credits running out mid-run, partial results
async def test_credits_exhausted_mid_run_keeps_the_partial_results(tmp_path):
    s = await start_swarm(tmp_path)  # starter credit 1000, no free allowance
    try:
        for i in range(3):
            await s.add_node(f"q{i}", FakeEngine(MODELS["qwen"], agent_reply(f"q{i}"), tokens=400), MODELS["qwen"])
        gw, _ = await s.add_gateway()
        chain = [task(0, max_tokens=400, escalate=False)] + [task(i, [f"t{i - 1}"], max_tokens=400, escalate=False)
                                                           for i in range(1, 4)]
        # 4 x 400 tokens x 0.5 typical fill x 0.85 = 680 credits: it fits (1000), but each answer really costs 340
        res = await AgentRun(gw, parse_request({"task": "x", "plan": chain, "combine": "merge",
                                                          "merge": {"max_tokens": 100}})).execute()
        st = {x["id"]: x for x in res["subtasks"]}
        assert res["estimate"]["credits"] < res["estimate"]["available"]
        assert [st[f"t{i}"]["status"] for i in range(4)] == ["ok", "ok", "ok", "failed"], st
        assert res["status"] == "partial" and "crédits épuisés" in res["error"]
        assert "## t0" in res["result"] and "## t2" in res["result"]  # what was obtained is kept
        assert not res["merged"]  # no merge call is started without credits
    finally:
        await s.close()


# ================================================================ HTTP: refusal upfront, options
async def test_http_refusal_before_anything_starts(tmp_path):
    s = await swarm_of(tmp_path, 4, tokens=300, max_parallel=4, starter_credit=1000.0)
    try:
        gw, client = await s.add_gateway()
        body = {"task": "x", "plan": [task(i, max_tokens=300, escalate=False) for i in range(10)],
                "budget": {"max_tokens": 4000}}
        r = await client.post("/v1/agents/run", json=body)
        assert r.status_code == 402
        err = r.json()["error"]
        assert err["type"] == "insufficient_credits" and "non lancée" in err["message"]
        assert {o["id"] for o in err["options"]} >= {"k", "max_tokens", "local"}
        assert err["estimate"]["credits"] > err["estimate"]["available"] and err["estimate"]["suggested_max_tokens"]
        r = await client.post("/v1/agents/run", json={**body, "stream": True})
        assert r.status_code == 402  # also for the event stream: no stream is opened
        assert not s.tracker.jobs  # nothing reached a peer
        small = {"task": "x", "plan": [task(i, max_tokens=100, escalate=False) for i in range(3)]}
        r = await client.post("/v1/agents/run", json=small)
        assert r.status_code == 200 and r.json()["status"] == "ok"
    finally:
        await s.close()


async def test_auto_plan_is_checked_once_written(tmp_path):
    plan = json.dumps({"subtasks": [{"id": f"s{i}", "prompt": f"Do part {i}. TOKEN=s{i}"} for i in range(8)]})
    s = await start_swarm(tmp_path)
    try:
        for i in range(2):
            await s.add_node(f"q{i}", FakeEngine(MODELS["qwen"], agent_reply(f"q{i}", plan=plan), tokens=20),
                             MODELS["qwen"], max_parallel=4)
        gw, _ = await s.add_gateway()
        req = parse_request({"task": "big", "plan": "auto", "budget": {"max_tokens": 200_000}, "auto_k": 4})
        events = []
        res = await AgentRun(gw, req, events.append).execute()
        assert res["status"] == "refused" and res["planner"]["status"] == "ok"
        assert [e for e in events if e["type"] == "job" and e["subtask"] != "plan"] == []
    finally:
        await s.close()


# ================================================================ free daily allowance
def old_account(t, node_id, days=10, passes=0):
    """Age an account and give it passed availability canaries (the tracker's own records)."""
    t.ledger.db.execute("UPDATE accounts SET created_at=? WHERE node_id=?", (time.time() - days * 86400, node_id))
    if passes:
        t.ledger.db.execute("INSERT OR REPLACE INTO avail(node_id, hour, paid, passed) VALUES (?,?,0,?)",
                            (node_id, hour_of(time.time()) - 2, passes))  # completed hours are what counts
    t._allow.clear()


async def test_allowance_scales_with_age_and_verified_availability(tmp_path):
    s = await start_swarm(tmp_path, free_daily_tokens=40_000.0)
    try:
        t = s.tracker
        a = await s.add_node("a", FakeEngine(MODELS["qwen"], "x"), MODELS["qwen"])
        assert t.free_allowance_milli(a.node_id) / 1000 == pytest.approx(2000.0)  # brand-new account: 5 %
        old_account(t, a.node_id, days=10)  # old, but nobody verified its availability
        assert t.free_allowance_milli(a.node_id) / 1000 == pytest.approx(2000.0)
        old_account(t, a.node_id, days=1.5, passes=24)  # verified but young: half way
        mid = t.free_allowance_milli(a.node_id) / 1000
        assert 2000 < mid < 40_000 * 0.6
        old_account(t, a.node_id, days=10, passes=24)
        assert t.free_allowance_milli(a.node_id) / 1000 == pytest.approx(40_000.0)
        t.ledger.avail_fail(a.node_id, time.time(), hard=True)  # failed a canary: back to the floor
        t._allow.clear()
        assert t.free_allowance_milli(a.node_id) / 1000 == pytest.approx(2000.0)
        t.banned[a.node_id] = "test"
        t._allow.clear()
        assert t.free_allowance_milli(a.node_id) == 0
        t.banned.clear()
        t.free_daily = 0  # the tier switched off
        t._allow.clear()
        assert t.free_allowance_milli(a.node_id) == 0
    finally:
        await s.close()


async def test_free_allowance_pays_instead_of_the_balance_and_is_bounded_per_day(tmp_path):
    s = await start_swarm(tmp_path, starter_credit=0.0, free_daily_tokens=40_000.0)
    try:
        await s.add_node("q", FakeEngine(MODELS["qwen"], "The answer is 42.", tokens=1000), MODELS["qwen"],
                         max_parallel=4)
        gw, client = await s.add_gateway()
        me = s.nodes["client"].node_id
        led = s.tracker.ledger
        q = {"model": MODELS["qwen"], "messages": [{"role": "user", "content": "Tom has 40 apples and buys 2 more."}]}
        ok = 0
        for _ in range(6):
            r = await client.post("/v1/chat/completions", json=q)
            if r.status_code != 200:
                assert r.status_code == 402 and r.json()["error"]["type"] == "insufficient_credits"
                break
            ok += 1
            await wait_until(lambda: len([e for e in led.recent(50) if e["kind"] == "receipt"]) >= ok, 5)
        per = led.db.execute("SELECT amount FROM ledger WHERE requester_id=?", (me,)).fetchone()["amount"] / 1000
        # answers are admitted while the allowance still covers what is owed: the last one may overdraw a little
        assert ok == int(2000 // per) + 1 and 1 < ok < 6, (ok, per)
        assert led.free_used(me) == 2000 * 1000  # the whole allowance, never more
        assert led.balance(me) == pytest.approx(2000 - ok * per)  # the rest of the last answer is a debt
        info = await gw.account(fresh=True)
        assert info["free"]["daily"] == 2000 and info["free"]["left_today"] == 0
        assert info["inflight_cap"] == 8 and info["balance"] == pytest.approx(2000 - ok * per)
        assert gw.slots.cap == 8
        rows = led.db.execute("SELECT amount, free FROM ledger WHERE requester_id=?", (me,)).fetchall()
        assert sum(r["free"] for r in rows) == 2000 * 1000 and all(r["free"] <= r["amount"] for r in rows)
    finally:
        await s.close()


async def test_preflight_counts_the_free_allowance(tmp_path):
    s = await swarm_of(tmp_path, 3, starter_credit=100.0, free_daily_tokens=40_000.0)
    try:
        gw, _ = await s.add_gateway()
        # 4 x 300 tokens x 0.5 x ~1 = ~600 credits: far above 100 of balance, within 100 + 2000 of allowance
        req = parse_request({"task": "x", "plan": [task(i, max_tokens=300, escalate=False) for i in range(4)]})
        res = await AgentRun(gw, req).execute()
        assert res["estimate"]["free_left_today"] == 2000 and res["estimate"]["available"] == 2100
        assert res["status"] == "ok"
    finally:
        await s.close()


# ================================================================ availability rewards
def fixed_canary(monkeypatch):
    monkeypatch.setattr(canary, "question", lambda p, tokens=None: ("What is 2+2?", "4"))


async def avail_swarm(tmp_path, engines: dict, **kw):
    kw = dict(avail_interval_s=0.2, canary_delay_s=(0.0, 0.01), **kw)
    s = await start_swarm(tmp_path, **kw)
    nodes = {}
    for name, eng in engines.items():
        nodes[name] = await s.add_node(name, eng, MODELS["qwen"])
    return s, nodes


async def test_available_node_is_paid_without_any_user_job(tmp_path, monkeypatch):
    fixed_canary(monkeypatch)
    s, n = await avail_swarm(tmp_path, {"a": FakeEngine(MODELS["qwen"], "The answer is 4.", tokens=4)})
    try:
        t = s.tracker
        me = n["a"].node_id
        await wait_until(lambda: t.canary_stats["avail_passed"] >= 4, 15)
        # each canary pays like a job: 4 tokens x 0.85 = 3.4 credits, on top of the starter credit
        await wait_until(lambda: t.ledger.balance(me) >= 1000 + 4 * 3.4 - 1e-6, 5)
        assert t.ledger.balance(me) == pytest.approx(1000 + t.canary_stats["avail_paid_milli"] / 1000)
        # the ledger row of a canary is an ordinary receipt: no label for the peer to learn from
        assert {e["kind"] for e in t.ledger.recent(50)} == {"starter", "receipt"}
        assert t.ledger.avail_paid_today(me, time.time()) == t.canary_stats["avail_paid_milli"]
        # what the owner is shown only counts completed hours: no counter moves right after a canary
        view = t.account_view(me)["availability"]
        assert view["serving"] and not view["blocked"] and view["earned_24h"] == 0 and view["passed_24h"] == 0
        assert view["estimated_daily"] > 0 and view["max_daily"] >= view["estimated_daily"]
        # a node that serves nothing (a pure client) gets no canary and no earnings
        c = await s.add_node("client", None)
        await asyncio.sleep(0.8)
        assert t.ledger.avail_paid_today(c.node_id, time.time()) == 0 and t.ledger.balance(c.node_id) == 1000
        assert t.account_view(c.node_id)["availability"]["estimated_daily"] == 0
    finally:
        await s.close()


async def test_daily_cap_bounds_what_canaries_pay(tmp_path, monkeypatch):
    fixed_canary(monkeypatch)
    s, n = await avail_swarm(tmp_path, {"a": FakeEngine(MODELS["qwen"], "The answer is 4.", tokens=4)},
                             avail_daily_cap=10.0)
    try:
        t = s.tracker
        old_account(t, n["a"].node_id, days=10)  # full cap: 10 x 0.85 = 8.5 credits; a canary pays 3.4
        await wait_until(lambda: t.canary_stats["avail_capped"] >= 2, 15)
        paid = t.ledger.avail_paid_today(n["a"].node_id, time.time()) / 1000
        assert paid == pytest.approx(3 * 3.4)  # stops after the canary that reached the cap (2 x 3.4 < 8.5)
        sent = t.canary_stats["sent"]
        await asyncio.sleep(0.8)
        assert t.canary_stats["sent"] == sent  # and nothing more is sent that day
    finally:
        await s.close()


async def test_new_account_cap_is_a_quarter(tmp_path):
    s = await start_swarm(tmp_path, avail_interval_s=900.0, avail_daily_cap=9_600.0)
    try:
        t = s.tracker
        a = await s.add_node("a", FakeEngine(MODELS["qwen"], "x"), MODELS["qwen"])
        assert t._avail_cap(a.node_id, MODELS["qwen"], None)[0] / 1000 == pytest.approx(9_600 * 0.85 * 0.25, rel=0.01)
        old_account(t, a.node_id, days=10)
        assert t._avail_cap(a.node_id, MODELS["qwen"], None)[0] / 1000 == pytest.approx(9_600 * 0.85)
        assert t._avail_cap(a.node_id, "org/Huge-70B-GGUF", 70.0)[1] == 4.0  # the model factor is capped
    finally:
        await s.close()


async def test_node_failing_canaries_is_not_paid(tmp_path, monkeypatch):
    fixed_canary(monkeypatch)
    s, n = await avail_swarm(tmp_path, {"liar": FakeEngine(MODELS["qwen"], "The answer is 5.", tokens=4)})
    try:
        t = s.tracker
        me = n["liar"].node_id
        old_account(t, me, days=10)
        await wait_until(lambda: t.canary_stats["avail_failed"] >= 1, 15)
        await asyncio.sleep(0.8)  # more canaries would have come by now
        assert t.canary_stats["avail_passed"] == 0 and t.ledger.balance(me) == 1000  # nothing earned
        assert t.ledger.avail_paid_today(me, time.time()) == 0
        assert any(e["kind"] == "refused" and e["node_id"] == me for e in t.ledger.recent(50))
        # wrong answers block the node once they are neither rare nor a minority
        led, now = t.ledger, time.time()
        led.avail_fail("x", now)
        led.avail_fail("x", now)
        assert not led.avail_blocked("x", now)  # two are not enough
        led.avail_fail("x", now)
        assert led.avail_blocked("x", now)
        led.db.execute("INSERT OR REPLACE INTO avail(node_id, hour, passed, failed) VALUES ('y', ?, 40, 3)",
                       (hour_of(now),))
        assert not led.avail_blocked("y", now)  # 3 failures among 43 canaries: a rare mistake
    finally:
        await s.close()


async def test_banned_and_blocked_nodes_get_no_canary(tmp_path, monkeypatch):
    fixed_canary(monkeypatch)
    s, n = await avail_swarm(tmp_path, {"a": FakeEngine(MODELS["qwen"], "The answer is 4.", tokens=4),
                                        "b": FakeEngine(MODELS["qwen"], "The answer is 4.", tokens=4)},
                             free_daily_tokens=40_000.0)
    try:
        t = s.tracker
        a, b = n["a"].node_id, n["b"].node_id
        t.ban(a, "honeytoken")  # a banned node is dropped: its chain of canaries ends with its connection
        t.ledger.avail_fail(b, time.time(), hard=True)  # b answered a canary with something that does not open
        t._allow.clear()
        await wait_until(lambda: t.canary_stats["avail_withheld"] >= 2, 10)
        await asyncio.sleep(0.5)
        assert t.canary_stats["sent"] == 0 and t.ledger.balance(b) == 1000
        assert a not in t.conns and t.free_allowance_milli(a) == 0
        assert t.free_allowance_milli(b) == int(t.free_daily * 0.05)
    finally:
        await s.close()


async def test_bad_signature_on_an_availability_canary_blocks_the_node(tmp_path):
    """A result rejected before it is decrypted (bad signature, kind or model) is a cheating-like failure."""
    from types import SimpleNamespace
    from myriad.tracker import CanaryPair
    s = await start_swarm(tmp_path, avail_interval_s=0.0)
    try:
        t = s.tracker
        a = await s.add_node("a", FakeEngine(MODELS["qwen"], "x"), MODELS["qwen"])
        t._avail_cheat(SimpleNamespace(canary=None, target=a.node_id))  # a real job: not an availability matter
        assert not t.ledger.avail_blocked(a.node_id, time.time())
        t._avail_cheat(SimpleNamespace(canary=(CanaryPair(params={}, avail=False), None), target=a.node_id))
        assert not t.ledger.avail_blocked(a.node_id, time.time())  # an ordinary canary: not either
        t._avail_cheat(SimpleNamespace(canary=(CanaryPair(params={}, avail=True), None), target=a.node_id))
        assert t.ledger.avail_blocked(a.node_id, time.time())
    finally:
        await s.close()


async def test_availability_chain_follows_the_connection(tmp_path, monkeypatch):
    """One chain of canaries per connection: a reconnection does not double the rate."""
    fixed_canary(monkeypatch)
    s, n = await avail_swarm(tmp_path, {"a": FakeEngine(MODELS["qwen"], "The answer is 4.", tokens=4)},
                             avail_daily_cap=1e9)
    try:
        t = s.tracker
        ident = n["a"].identity
        await wait_until(lambda: t.canary_stats["sent"] >= 1, 10)
        await n["a"].stop()
        again = await s.add_node("a2", FakeEngine(MODELS["qwen"], "The answer is 4.", tokens=4), MODELS["qwen"],
                                 identity=ident)
        sent0, t0 = t.canary_stats["sent"], time.monotonic()
        await asyncio.sleep(3.0)
        rate = (t.canary_stats["sent"] - sent0) / (time.monotonic() - t0)
        assert rate < 1.0 / 0.2 * 1.3  # about one per 0.2 s (jitter 0.75 - 1.25), not two chains
        assert again.node_id == ident.node_id
    finally:
        await s.close()


# ================================================================ endpoints, migration
async def test_account_endpoint_is_for_the_owner_only(tmp_path):
    s = await start_swarm(tmp_path, free_daily_tokens=40_000.0, avail_interval_s=900.0)
    try:
        a = await s.add_node("a", FakeEngine(MODELS["qwen"], "x"), MODELS["qwen"])
        b = await s.add_node("b", None)
        async with httpx.AsyncClient(base_url=s.url) as http:
            assert (await http.get(f"/v1/account/{a.node_id}")).status_code == 403
            assert (await http.get(f"/v1/account/{a.node_id}", params=account_params(b.identity))).status_code == 403
            r = await http.get(f"/v1/account/{a.node_id}", params=account_params(a.identity))
            assert r.status_code == 200
            d = r.json()
            assert d["balance"] == 1000 and d["free"]["daily"] == 2000 and d["free"]["left_today"] == 2000
            assert d["inflight_cap"] == 8 and d["availability"]["serving"]
            assert d["availability"]["max_daily"] > 0 and d["availability"]["estimated_daily"] > 0
            r = await http.get(f"/v1/account/{b.node_id}", params=account_params(b.identity))
            assert r.json()["availability"]["estimated_daily"] == 0 and not r.json()["availability"]["serving"]
    finally:
        await s.close()


def test_migrates_an_existing_ledger(tmp_path):
    """A ledger created before the free tier gets its `free` column."""
    import sqlite3
    db = sqlite3.connect(tmp_path / "old.sqlite")
    db.executescript("""CREATE TABLE ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, job_id TEXT UNIQUE,
      requester_id TEXT, node_id TEXT NOT NULL, model TEXT, tokens INTEGER NOT NULL DEFAULT 0,
      amount INTEGER NOT NULL, kind TEXT NOT NULL, receipt TEXT, result TEXT);""")
    db.close()
    led = Ledger(tmp_path / "old.sqlite")
    assert "free" in {r["name"] for r in led.db.execute("PRAGMA table_info(ledger)")}
    led.close()


# ================================================================ review points
async def test_preflight_uses_fresh_funds_not_the_dashboard_cache(tmp_path):
    s = await swarm_of(tmp_path, 3, starter_credit=50.0)
    try:
        gw, _ = await s.add_gateway()
        gw._acct = (time.monotonic(), {"balance": 1e9, "free": {"left_today": 0}, "inflight_cap": 8})  # stale, rich
        req = parse_request({"task": "x", "plan": [task(i, max_tokens=400, escalate=False) for i in range(4)]})
        res = await AgentRun(gw, req).execute()
        assert res["status"] == "refused" and res["estimate"]["balance"] == 50.0
    finally:
        await s.close()


async def test_estimate_counts_one_candidate_for_a_pinned_model(tmp_path):
    s = await swarm_of(tmp_path, 4, starter_credit=1000.0)
    try:
        gw, _ = await s.add_gateway()
        pinned = task(1, max_tokens=512, k=4, model=MODELS["qwen"], escalate=False)  # the gateway sends ONE job
        res = await AgentRun(gw, parse_request({"task": "x", "plan": [pinned]})).execute()
        assert res["estimate"]["max_credits"] == pytest.approx(512 * 0.85, abs=0.1) and res["status"] == "ok"
    finally:
        await s.close()


def test_unreleased_availability_table_is_replaced(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "dev.sqlite")
    db.executescript("CREATE TABLE avail (node_id TEXT, day INTEGER, paid INTEGER, passed INTEGER, pending INTEGER);")
    db.close()
    led = Ledger(tmp_path / "dev.sqlite")
    assert led.avail_passes("x", time.time()) == 0 and led.avail_paid_today("x", time.time()) == 0
    led.close()
