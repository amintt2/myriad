"""Regression tests of the sub-agents audit (2026-10-09): verification processes left behind,
preparation errors during concurrent verifications, the chat stream and forged results, context
budget vs dependencies, deadline during the plan or the merge, reserved ids in automatic plans."""
from __future__ import annotations

import asyncio
import json
import threading
import time
import types
import uuid

import httpx
import pytest

import myriad.agents as ag
from myriad.agents import AgentRun, AgentsError, SubState, VerifySpec, parse_request, run_agents, run_verification
from myriad.crypto import Identity
from myriad.engine import FakeEngine
from myriad.protocol import MAX_CONTENT_CHARS, Job, JobFrame, JobResult, ResultFrame
from myriad.ui import make_ui_app

from .conftest import MODELS, start_swarm
from .test_agents import GOOD, PLAN_JSON, PY, SYNTAX, agent_reply, make_swarm, tagged, task


# ------------------------------------------------------------------ 1. processes left behind
def test_verification_kills_children_left_behind(tmp_path):
    flag = tmp_path / "child.txt"
    child = f"import time, pathlib; time.sleep(1.5); pathlib.Path({str(flag)!r}).write_text('x')"
    parent = [PY, "-c", "import subprocess, sys; subprocess.Popen([sys.executable, '-c', " + repr(child) + "], "
                        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print('ok')"]
    v = run_verification(parent, GOOD, VerifySpec(run="spawn", timeout_s=30))
    assert v["passed"] and "ok" in v["output"], v
    time.sleep(2.5)
    assert not flag.exists()  # the child did not outlive the verification


# ------------------------------------------------------------------ 2. preparation errors
def test_preparation_error_is_a_verdict(tmp_path):
    work = tmp_path / "proj"
    (work / "candidate.py").mkdir(parents=True)  # the target is a directory: writing it fails
    v = run_verification(SYNTAX["syntax"], GOOD, VerifySpec(run="syntax", workdir=str(work), target="candidate.py"))
    assert v["passed"] is False and v["error"], v


async def test_concurrent_verifications_are_not_abandoned(tmp_path, monkeypatch):
    calls = {"n": 0, "finished": False}
    lock = threading.Lock()

    def fake(argv, text, spec, cancel=None):
        with lock:
            calls["n"] += 1
            first = calls["n"] == 1
        if first:
            raise OSError("préparation impossible")
        time.sleep(1.0)
        calls["finished"] = True
        return {"passed": True, "exit_code": 0, "error": None, "ms": 1.0, "output": ""}

    monkeypatch.setattr(ag, "run_verification", fake)
    s = await make_swarm(tmp_path, n=3)
    try:
        gw, _ = await s.add_gateway()
        gw.verify_commands = SYNTAX
        res = await run_agents(gw, {"plan": [task(1, k=2, escalate=False, verify={"run": "syntax"})]})
        assert calls["finished"], "the run ended while a verification was still running"
        st = res["subtasks"][0]
        assert st["status"] == "ok" and st["verify"]["passed"], st  # the other candidate passed
        assert ag._VERIFY_SEM._value == ag.VERIFY_CONCURRENCY
    finally:
        await s.close()


# ------------------------------------------------------------------ 3. forged results in the chat stream
class _FakeNode:
    def __init__(self):
        self.observers = []

    def add_observer(self, f):
        self.observers.append(f)

    def remove_observer(self, f):
        self.observers.remove(f)


async def test_chat_stream_shows_only_verified_answers():
    me, peer, evil = Identity.generate(), Identity.generate(), Identity.generate()
    node = _FakeNode()
    card = {"node_id": peer.node_id, "pubkey": peer.pubkey, "model": "org/M-GGUF", "family": "qwen"}

    class Gw:
        default_k = 2

        async def directory(self, fresh=False):
            return [card], {}

        async def ask(self, messages, **kw):
            from myriad.gateway import GatewayError
            obs = node.observers[0]
            jobs = []
            for _ in range(2):
                job = Job(job_id=uuid.uuid4().hex, requester_id=me.node_id, messages=messages).signed_by(me)
                obs("out", JobFrame(job=job, target=peer.node_id))
                jobs.append(job)
            forged = JobResult(job_id=jobs[0].job_id, node_id=peer.node_id, model="org/M-GGUF",
                               text="The answer is 666. FORGED").signed_by(evil)
            obs("in", ResultFrame(result=forged))
            good = JobResult(job_id=jobs[1].job_id, node_id=peer.node_id, model="org/M-GGUF",
                             text="The answer is 42.").signed_by(peer)
            obs("in", ResultFrame(result=good))
            raise GatewayError(502, "bad_signature")

    app = make_ui_app(node, Gw(), token="tok")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401") as c:
        r = await c.post("/api/chat/stream", json={"message": "What is 6 x 7?", "task_hint": "math"},
                         headers={"X-Myriad-Token": "tok"})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert "FORGED" not in r.text and "666" not in r.text, events
    answered = [e for e in events if e["type"] == "answered"]
    assert len(answered) == 1 and answered[0]["answer"] == "42"
    assert any(e["type"] == "failed" and "signature" in e["error"] for e in events)


# ------------------------------------------------------------------ 5. context budget vs dependencies
def test_global_contexts_count_in_the_budget():
    ctx = [{"text": "a" * 20_000}, {"text": "b" * 20_000}]
    with pytest.raises(AgentsError, match="trop longs"):
        parse_request({"context": ctx, "plan": [task(1, uses_context=[0, 1])]})
    parse_request({"context": ctx, "plan": [task(1, uses_context=[0]), task(2, uses_context=[1])]})  # each fits


def test_dependency_results_keep_their_room():
    req = parse_request({"task": "t", "context": [{"text": "c" * 25_000, "label": "big"}],
                         "plan": [task(0), task(1, ["t0"], uses_context=[0])]})
    run = AgentRun(types.SimpleNamespace(verify_commands={}), req)
    for s in req.plan:
        run.subs[s.id] = SubState(spec=s)
    run.subs["t0"].text = "Q" * 10_000
    user = run._messages(run.subs["t1"])[1]["content"]
    assert len(user) <= MAX_CONTENT_CHARS
    assert user.count("Q") == 10_000  # the dependency's result is whole
    assert user.count("c") >= 15_000 and "[tronqué]" in user  # the context gave way, visibly


# ------------------------------------------------------------------ 6. deadline during the plan or the merge
async def _slow_orchestrator_swarm(tmp_path, plan=PLAN_JSON):
    s = await start_swarm(tmp_path)
    await s.add_node("fast", FakeEngine(MODELS["qwen"], agent_reply("fast", plan=plan), tokens=10), MODELS["qwen"])
    await s.add_node("slow", FakeEngine(MODELS["smollm"], agent_reply("slow", plan=plan), delay_s=6.0, tokens=10),
                     MODELS["smollm"], cls=tagged(["orchestrator"]))
    return s


async def test_deadline_during_planning_ends_the_planner(tmp_path):
    s = await _slow_orchestrator_swarm(tmp_path)
    try:
        gw, _ = await s.add_gateway()
        events = []
        res = await run_agents(gw, {"task": "x", "plan": "auto", "budget": {"deadline_s": 1.5}}, events.append)
        assert res["status"] == "timeout"
        assert res["planner"]["status"] == "cancelled" and res["planner"]["ended_ms"] is not None, res["planner"]
    finally:
        await s.close()


async def test_deadline_during_merge_ends_the_merge(tmp_path):
    s = await _slow_orchestrator_swarm(tmp_path)
    try:
        gw, _ = await s.add_gateway()
        res = await run_agents(gw, {"plan": [task(1, family="qwen")], "combine": "merge",
                                    "budget": {"deadline_s": 2.0}})
        assert res["status"] == "timeout" and res["subtasks"][0]["status"] == "ok"
        assert res["merge"]["status"] == "cancelled" and res["merge"]["ended_ms"] is not None, res["merge"]
    finally:
        await s.close()


# ------------------------------------------------------------------ 8. reserved ids in an automatic plan
async def test_auto_plan_with_reserved_id_gets_its_second_chance(tmp_path):
    bad = json.dumps({"subtasks": [{"id": "merge", "prompt": "x TOKEN=merge"}], "combine": "concat"})
    calls = []

    def reply(messages):
        system = messages[0]["content"]
        if "orchestrator of Myriad" in system:
            calls.append(messages)
            return PLAN_JSON if "Invalid plan" in messages[-1]["content"] else bad
        return agent_reply("n", plan=PLAN_JSON)(messages)

    s = await start_swarm(tmp_path)
    try:
        for i, fam in enumerate(["qwen", "smollm"]):
            await s.add_node(f"n{i}", FakeEngine(MODELS[fam], reply, tokens=10), MODELS[fam],
                             cls=tagged(["orchestrator"] if i == 0 else None))
        gw, _ = await s.add_gateway()
        res = await run_agents(gw, {"task": "add, documented", "plan": "auto"})
        assert len(calls) == 2 and "réservé" in calls[1][-1]["content"]
        assert res["status"] == "ok" and res["planner"]["status"] == "ok", res
    finally:
        await s.close()


def test_planner_prompt_names_the_reserved_ids():
    assert "plan" in ag.PLANNER_SYSTEM and "merge" in ag.PLANNER_SYSTEM and "reserved" in ag.PLANNER_SYSTEM
