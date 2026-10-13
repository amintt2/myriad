"""Free tier: runs that start must finish. Reproduces why sub-agent runs of a fresh account started and
never finished (in-flight cap, starter credit, peers busy for longer than the retry window), then checks
the fixes: upfront refusal, gateway-side queue, free daily allowance, availability rewards."""
from __future__ import annotations

import pytest

from myriad.agents import AgentRun, parse_request
from myriad.engine import FakeEngine

from .conftest import MODELS, start_swarm
from .test_agents import agent_reply, task


async def swarm_of(tmp_path, n, delay=0.0, tokens=10, max_parallel=1, **tracker_kw):
    tracker_kw.setdefault("starter_credit", 1_000_000.0)  # the budget is not what these tests are about
    s = await start_swarm(tmp_path, **tracker_kw)
    fams = list(MODELS)
    for i in range(n):
        fam = fams[i % len(fams)]
        eng = FakeEngine(MODELS[fam], agent_reply(f"n{i}"), delay_s=delay, tokens=tokens)
        await s.add_node(f"n{i}", eng, MODELS[fam], max_parallel=max_parallel)
    return s


# ---------------------------------------------------------------- cause 1: the fresh-account in-flight cap
async def test_more_parallel_jobs_than_the_inflight_cap_queue_instead_of_failing(tmp_path):
    """6 sub-tasks x 2 candidates = 12 jobs in flight against the cap of 8 of a new account: the tracker
    answered too_many_jobs for the last ones and their sub-tasks failed."""
    s = await swarm_of(tmp_path, 12, delay=0.2)
    try:
        gw, _ = await s.add_gateway()
        req = parse_request({"task": "x", "plan": [task(i, k=2, escalate=False) for i in range(6)],
                             "budget": {"max_parallel": 8}})
        res = await AgentRun(gw, req).execute()
        assert res["status"] == "ok", [(x["id"], x["status"], x["error"]) for x in res["subtasks"]]
    finally:
        await s.close()


# ---------------------------------------------------------------- cause 2: the starter credit runs out mid-run
async def test_run_that_cannot_be_paid_is_refused_before_it_starts(tmp_path):
    """10 sub-tasks of 200 tokens on a 1.7B model cost ~1700 credits against a starter credit of 1000: the
    run used to start, serve ~6 sub-tasks, then fail the rest with insufficient_credits."""
    s = await swarm_of(tmp_path, 4, tokens=300, max_parallel=4, starter_credit=1000.0)
    try:
        gw, _ = await s.add_gateway()
        req = parse_request({"task": "x", "plan": [task(i, max_tokens=300, escalate=False) for i in range(10)],
                             "budget": {"max_tokens": 4000}})
        events = []
        res = await AgentRun(gw, req, events.append).execute()
        assert res["status"] == "refused" and "crédits" in res["error"], res["error"]
        assert "k" in res["error"] and "local" in res["error"]  # the options
        assert res["estimate"]["credits"] > res["estimate"]["available"]
        assert not [e for e in events if e["type"] == "job"]  # nothing was sent to a peer
    finally:
        await s.close()


# ---------------------------------------------------------------- cause 3: peers busy for longer than the retry window
async def test_subtasks_wait_for_a_busy_peer_as_long_as_the_run_allows(tmp_path):
    """6 sub-tasks on 2 peers that take 4 s each: the retry window used to be a fixed 7.5 s, so the third
    wave (which waits 8 s) failed with no_peers although the peers were only busy."""
    s = await swarm_of(tmp_path, 2, delay=4.0, max_parallel=1)
    try:
        gw, _ = await s.add_gateway()
        req = parse_request({"task": "x", "plan": [task(i, escalate=False) for i in range(6)],
                             "budget": {"deadline_s": 60}})
        res = await AgentRun(gw, req).execute()
        assert res["status"] == "ok", [(x["id"], x["status"], x["error"]) for x in res["subtasks"]]
    finally:
        await s.close()
