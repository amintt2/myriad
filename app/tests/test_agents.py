"""Sub-agents (POST /v1/agents/run): plan validation, parallel dispatch with dependencies (wall time =
longest branch), local verification with an allow-list, cascade, budgets, auto plans, events."""
from __future__ import annotations

import functools
import json
import re
import sys
import time

import httpx
import pytest

from myriad.agents import (AgentRun, AgentsError, allowed_commands, demo_plan, demo_verify_commands, extract_code,
                           parse_auto_plan, parse_request, pick_candidate, run_agents, run_verification, VerifySpec)
from myriad.engine import FakeEngine
from myriad.node import NodeClient

from .conftest import MODELS, start_swarm

PY = sys.executable
SYNTAX = {"syntax": [PY, "-I", "-m", "py_compile", "{file}"]}
GOOD = "Here it is:\n```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b:\n    return a +\n```"
PLAN_JSON = json.dumps({"subtasks": [
    {"id": "code", "role": "coder", "skill": "python", "prompt": "Write add(a, b). TOKEN=code", "kind": "code"},
    {"id": "doc", "role": "writer", "prompt": "Document add. TOKEN=doc", "depends_on": ["code"]}],
    "combine": "merge"})


def tagged(tags):
    return functools.partial(NodeClient, tags=tags)


def agent_reply(name: str, code: str | None = None, plan: str = PLAN_JSON, log: list | None = None):
    def reply(messages):
        if log is not None:
            log.append((name, messages))
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        if "orchestrator of Myriad" in system:
            return plan
        if "final agent of Myriad" in system:
            return "MERGED " + " + ".join(re.findall(r"## Result of (\w+)", user))
        if code is not None and "TOKEN=code" in user:
            return code
        m = re.search(r"TOKEN=(\w+)", user)
        return f"{name} did {m.group(1) if m else '?'}"
    return reply


async def make_swarm(tmp_path, n=6, delay=0.0, codes=None, tags=None, log=None, plan=PLAN_JSON, max_parallel=1,
                     tokens=10):
    """n nodes cycling over the 4 test families; codes[i] / tags[i] per node."""
    s = await start_swarm(tmp_path)
    fams = list(MODELS)
    for i in range(n):
        fam = fams[i % len(fams)]
        eng = FakeEngine(MODELS[fam], agent_reply(f"n{i}", (codes or {}).get(i), plan, log), delay_s=delay, tokens=tokens)
        await s.add_node(f"n{i}", eng, MODELS[fam], max_parallel=max_parallel, cls=tagged((tags or {}).get(i)))
    return s


def task(i, deps=(), **kw):
    return {"id": f"t{i}", "prompt": f"Do part {i}. TOKEN=t{i}", "depends_on": list(deps), **kw}


# ============================================================ validation
def test_plan_validation_rejects_bad_plans(tmp_path):
    ok = parse_request({"task": "x", "plan": [task(1), task(2, ["t1"])]})
    assert [s.id for s in ok.plan] == ["t1", "t2"]
    bad = [
        ({"plan": [task(1), task(1)]}, "double"),
        ({"plan": [task(1, ["t2"]), task(2, ["t1"])]}, "cycle"),
        ({"plan": [task(1, ["t9"])]}, "inconnue"),
        ({"plan": [task(1, ["t1"])]}, "elle-même"),
        ({"plan": [{"id": "merge", "prompt": "x"}]}, "réservé"),
        ({"plan": [task(i) for i in range(3)], "budget": {"max_subtasks": 2}}, "trop de sous-tâches"),
        ({"plan": [task(1, verify={"run": "rm"})]}, "non autorisée"),
        ({"plan": [task(1, verify={"run": "syntax", "target": "../x.py"})], "allow_commands": SYNTAX}, "relatif"),
        ({"plan": [task(1, skill="python", family="qwen")]}, "une seule route"),
        ({"plan": [task(1, context=[{"text": "y" * 29_990}])]}, "trop longs"),
        ({"plan": [task(1, context=[{"file": str(tmp_path / "nope.txt")}])]}, "introuvable"),
        ({"plan": [task(1, uses_context=[0])]}, "uses_context"),
        ({"plan": [task(1, k=9)]}, "plan"),
        ({"plan": [task(1, unknown=1)]}, "plan"),
        ({"plan": "auto"}, "task"),
        ({"task": "x", "plan": "auto", "verify": {"run": "pytest"}}, "non autorisée"),
    ]
    for body, why in bad:
        with pytest.raises(AgentsError) as e:
            parse_request(body)
        assert e.value.status == 400 and why in e.value.message, (body, e.value.message)
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (300 * 1024))
    with pytest.raises(AgentsError, match="trop gros"):
        parse_request({"plan": [task(1, context=[{"file": str(big)}])]})
    # the configuration's allow-list counts too, the request's adds to it
    assert parse_request({"plan": [task(1, verify={"run": "syntax"})]}, SYNTAX).plan[0].verify.run == "syntax"
    assert allowed_commands({"a": ["x"], "bad name": ["y"], "b": []}, {"c": ["z"]}) == {"a": ["x"], "c": ["z"]}


def test_auto_plan_schema_is_strict():
    p = parse_auto_plan(f"Sure!\n```json\n{PLAN_JSON}\n```", 8)
    assert [s.id for s in p.subtasks] == ["code", "doc"] and p.combine == "merge"
    for evil in ({"subtasks": [{"id": "a", "prompt": "x", "verify": {"run": "syntax"}}]},  # a peer names a command
                 {"subtasks": [{"id": "a", "prompt": "x", "context": [{"file": "/etc/passwd"}]}]},  # or a local file
                 {"subtasks": [{"id": "a", "prompt": "x", "model": "m"}]}, {"subtasks": []}):
        with pytest.raises(ValueError):
            parse_auto_plan(json.dumps(evil), 8)
    with pytest.raises(ValueError, match="trop de"):
        parse_auto_plan(json.dumps({"subtasks": [{"id": f"a{i}", "prompt": "x"} for i in range(5)]}), 4)
    with pytest.raises(ValueError):
        parse_auto_plan("no json here", 8)


def test_demo_plan_is_valid():
    body = demo_plan(set(demo_verify_commands()))
    req = parse_request(body, demo_verify_commands())
    assert len(req.plan) == 6 and sum(1 for s in req.plan if s.verify) == 3 and req.combine == "merge"
    assert all(s.verify is None for s in parse_request(demo_plan(set()), {}).plan)


# ============================================================ verification
def test_verification_runs_only_in_a_temporary_copy(tmp_path):
    work = tmp_path / "proj"
    work.mkdir()
    (work / "marker.txt").write_text("here")
    (work / ".git").mkdir()
    (work / ".git" / "big").write_text("skipped")
    check = [PY, "-I", "-c", "import pathlib, sys; sys.exit(0 if pathlib.Path('marker.txt').exists() and "
                             "pathlib.Path('src/add.py').exists() and not pathlib.Path('.git').exists() else 3)"]
    spec = VerifySpec(run="check", workdir=str(work), target="src/add.py")
    v = run_verification(check, GOOD, spec)
    assert v["passed"] and v["exit_code"] == 0
    assert sorted(p.name for p in work.iterdir()) == [".git", "marker.txt"]  # untouched
    v = run_verification(SYNTAX["syntax"], BAD, VerifySpec(run="syntax"))
    assert not v["passed"] and v["exit_code"] != 0 and "SyntaxError" in v["output"]
    assert run_verification(SYNTAX["syntax"], GOOD, VerifySpec(run="syntax"))["passed"]
    t0 = time.monotonic()
    v = run_verification([PY, "-c", "import time; time.sleep(20)"], GOOD, VerifySpec(run="slow", timeout_s=1))
    assert not v["passed"] and "délai" in v["error"] and time.monotonic() - t0 < 10
    v = run_verification(["definitely-not-a-command-xyz"], GOOD, VerifySpec(run="x"))
    assert not v["passed"] and v["error"]
    assert extract_code("a\n```py\nshort\n```\n```py\nlonger code\n```") == "longer code\n"


def test_pick_candidate_weighted_agreement():
    c = [{"text": "```\nx=1\n```", "weight": 0.4}, {"text": "Code:\n```py\n\nx=1\n\n```", "weight": 0.3},
         {"text": "```\ny=2\n```", "weight": 0.6}]
    assert pick_candidate(c) == 0  # x=1 pools 0.7 > 0.6, its heaviest peer is kept
    assert pick_candidate([c[2]]) == 0


# ============================================================ parallel dispatch
async def test_independent_subtasks_run_in_parallel_on_different_peers(tmp_path):
    D = 1.0
    s = await make_swarm(tmp_path, n=6, delay=D)
    try:
        gw, _ = await s.add_gateway()
        events = []
        req = parse_request({"task": "six parts", "plan": [task(i) for i in range(6)]})
        res = await AgentRun(gw, req, events.append).execute()
        assert res["status"] == "ok" and all(st["status"] == "ok" for st in res["subtasks"])
        wall, total = res["timing"]["wall_ms"] / 1000, res["timing"]["subtasks_sum_ms"] / 1000
        assert total > 5.5 * D and wall < 2.0 * D, res["timing"]  # ≈ one sub-task, not six
        assert len({st["peer"]["node_id"] for st in res["subtasks"]}) == 6
        for st in res["subtasks"]:
            assert st["text"].endswith(f"did {st['id']}")
        kinds = [e["type"] for e in events]
        assert kinds[0] == "plan" and kinds[-1] == "done" and kinds.count("job") == 6 and kinds.count("answered") == 6
        assert "## t0" in res["result"] and res["usage"]["completion_tokens"] == 60
    finally:
        await s.close()


async def test_dependencies_wait_and_receive_results(tmp_path):
    D = 0.6
    log: list = []
    s = await make_swarm(tmp_path, n=4, delay=D, log=log)
    try:
        gw, _ = await s.add_gateway()
        # a -> b -> c (chain) next to d; e depends on a and d (diamond)
        plan = [task("a"), task("b", ["ta"]), task("c", ["tb"]), task("d"), task("e", ["ta", "td"])]
        res = await AgentRun(gw, parse_request({"task": "chain", "plan": plan})).execute()
        assert res["status"] == "ok"
        by = {st["id"]: st for st in res["subtasks"]}
        assert by["tb"]["started_ms"] >= by["ta"]["ended_ms"] and by["tc"]["started_ms"] >= by["tb"]["ended_ms"]
        assert by["te"]["started_ms"] >= max(by["ta"]["ended_ms"], by["td"]["ended_ms"])
        assert by["td"]["started_ms"] < by["ta"]["ended_ms"]  # independent branches overlap
        wall = res["timing"]["wall_ms"] / 1000
        assert 3 * D <= wall < 3 * D + 1.2, res["timing"]  # the longest branch (a, b, c), not the sum (5 D)
        assert [st["level"] for st in res["subtasks"]] == [0, 1, 2, 0, 1]
        msgs = {m[-1]["content"].split("TOKEN=")[1].split()[0]: m for _, m in log}
        assert "## Result of sub-task ta" in msgs["tb"][-1]["content"] and "did ta" in msgs["tb"][-1]["content"]
        assert "did tb" in msgs["tc"][-1]["content"] and "did ta" not in msgs["tc"][-1]["content"]  # small context
        assert "did ta" in msgs["te"][-1]["content"] and "did td" in msgs["te"][-1]["content"]
    finally:
        await s.close()


async def test_failed_dependency_skips_dependents(tmp_path):
    s = await make_swarm(tmp_path, n=2)
    try:
        gw, _ = await s.add_gateway()
        plan = [task(1, model="nobody/none-GGUF", escalate=False), task(2, ["t1"]), task(3)]
        res = await AgentRun(gw, parse_request({"plan": plan})).execute()
        st = {x["id"]: x for x in res["subtasks"]}
        assert st["t1"]["status"] == "failed" and st["t2"]["status"] == "skipped" and st["t3"]["status"] == "ok"
        assert res["status"] == "partial"
    finally:
        await s.close()


# ============================================================ verification in a run, cascade
async def test_verification_failure_escalates_to_the_swarm(tmp_path):
    # node 0 (qwen) is the only "python" peer and writes broken code; the others write good code
    s = await make_swarm(tmp_path, n=4, codes={0: BAD, 1: GOOD, 2: GOOD, 3: GOOD}, tags={0: ["python"]})
    try:
        gw, _ = await s.add_gateway()
        gw.verify_commands = SYNTAX
        events = []
        body = {"plan": [{"id": "code", "skill": "python", "prompt": "Write add. TOKEN=code",
                          "verify": {"run": "syntax", "target": "add.py"}}]}
        res = await run_agents(gw, body, events.append)
        st = res["subtasks"][0]
        assert st["status"] == "ok" and st["escalated"] and len(st["attempts"]) == 2
        assert st["attempts"][0]["status"] == "failed" and "vérification" in st["attempts"][0]["error"]
        assert st["verify"]["passed"] and st["peer"]["family"] != "qwen" and "return a + b" in st["text"]
        assert any(c["chosen"] for c in st["candidates"]) and st["route"]["k"] == 3
        kinds = [e["type"] for e in events]
        assert "escalate" in kinds and kinds.count("verifying") == 2
        assert sum(1 for e in events if e["type"] == "verified" and not e["passed"]) >= 1
        # without the cascade the sub-task fails
        body["plan"][0]["escalate"] = False
        res = await run_agents(gw, body)
        assert res["subtasks"][0]["status"] == "failed" and res["status"] == "failed"
    finally:
        await s.close()


async def test_peers_cannot_choose_commands(tmp_path):
    evil = json.dumps({"subtasks": [{"id": "x", "prompt": "p", "verify": {"run": "syntax"}}]})
    s = await make_swarm(tmp_path, n=2, plan=evil, tags={0: ["orchestrator"]})
    try:
        gw, _ = await s.add_gateway()
        gw.verify_commands = SYNTAX
        res = await run_agents(gw, {"task": "do it", "plan": "auto"})
        assert res["status"] == "failed" and "plan automatique invalide" in res["error"]
        assert res["planner"]["status"] == "failed" and res["subtasks"] == []
    finally:
        await s.close()


async def test_auto_plan_runs_with_user_verification(tmp_path):
    log: list = []
    s = await make_swarm(tmp_path, n=4, codes={i: GOOD for i in range(4)}, tags={1: ["orchestrator"]}, log=log)
    try:
        gw, _ = await s.add_gateway()
        events = []
        res = await run_agents(gw, {"task": "add two numbers, documented", "plan": "auto",
                                    "verify": {"run": "syntax"}, "allow_commands": SYNTAX}, events.append)
        assert res["status"] == "ok" and res["plan_source"] == "auto" and res["merged"]
        assert res["planner"]["status"] == "ok" and res["planner"]["peer"]["node_id"] == s.nodes["n1"].node_id
        st = {x["id"]: x for x in res["subtasks"]}
        assert st["code"]["verify"]["passed"] and st["doc"]["verify"] is None  # only "code" sub-tasks
        assert res["result"] == "MERGED code + doc" and res["merge"]["status"] == "ok"
        kinds = [e["type"] for e in events]
        assert kinds[0] == "planning" and kinds.index("plan") > kinds.index("job")
        assert [e["planner"]["status"] for e in events if e["type"] == "planning"] == ["running", "ok"]
    finally:
        await s.close()


# ============================================================ budgets
async def test_token_budget_and_deadline(tmp_path):
    s = await make_swarm(tmp_path, n=2, delay=0.3, tokens=50)
    try:
        gw, _ = await s.add_gateway()
        res = await run_agents(gw, {"plan": [task(1), task(2, ["t1"])], "budget": {"max_tokens": 64}})
        st = {x["id"]: x for x in res["subtasks"]}
        assert st["t1"]["status"] == "ok" and st["t2"]["status"] == "failed" and "budget" in st["t2"]["error"]
        assert res["usage"]["completion_tokens"] <= 64
    finally:
        await s.close()
    s = await make_swarm(tmp_path / "b", n=2, delay=5.0)
    try:
        gw, _ = await s.add_gateway()
        t0 = time.monotonic()
        res = await run_agents(gw, {"plan": [task(1), task(2)], "budget": {"deadline_s": 1.5}})
        assert time.monotonic() - t0 < 3.5 and res["status"] == "timeout"
        assert {x["status"] for x in res["subtasks"]} == {"cancelled"}
    finally:
        await s.close()


async def test_max_parallel_limits_concurrency(tmp_path):
    s = await make_swarm(tmp_path, n=4, delay=0.5)
    try:
        gw, _ = await s.add_gateway()
        res = await run_agents(gw, {"plan": [task(i) for i in range(4)], "budget": {"max_parallel": 2}})
        assert res["status"] == "ok" and res["timing"]["wall_ms"] >= 950  # two waves
    finally:
        await s.close()


# ============================================================ HTTP and events
def sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


async def test_http_api_json_sse_and_errors(tmp_path):
    s = await make_swarm(tmp_path, n=3, delay=0.1)
    try:
        gw, client = await s.add_gateway()
        r = await client.post("/v1/agents/run", json={"task": "t", "plan": [task(1), task(2, ["t1"])]})
        assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["object"] == "myriad.agents.run"
        r = await client.post("/v1/agents/run", json={"plan": [task(1, verify={"run": "rm-rf"})]})
        assert r.status_code == 400 and "non autorisée" in r.json()["error"]["message"]
        r = await client.post("/v1/agents/run", json={"plan": [task(1)]}, headers={"Origin": "https://evil.example"})
        assert r.status_code == 403
        r = await client.post("/v1/agents/run", json={"plan": [task(1), task(2)], "stream": True})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        evs = sse(r.text)
        kinds = [e["type"] for e in evs]
        assert kinds[0] == "start" and kinds[1] == "plan" and kinds[-1] == "done" and r.text.rstrip().endswith("[DONE]")
        assert {"subtask", "job", "answered"} <= set(kinds)
        assert evs[-1]["result"]["status"] == "ok"
        sch = (await client.get("/v1/agents/schema")).json()
        assert "request" in sch and "auto_plan" in sch and sch["verify_commands"] == []
    finally:
        await s.close()


async def test_cli_agents_run(tmp_path, capsys):
    import asyncio

    import uvicorn

    from myriad import cli
    from myriad.engine import free_port

    s = await make_swarm(tmp_path, n=3, delay=0.05, codes={i: GOOD for i in range(3)})
    try:
        gw, _ = await s.add_gateway()
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(gw.app, host="127.0.0.1", port=port, log_level="warning"))
        srv = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.02)
        plan = tmp_path / "plan.json"
        plan.write_text(json.dumps([task(1), {"id": "code", "prompt": "TOKEN=code", "depends_on": ["t1"],
                                              "verify": {"run": "syntax"}}]), encoding="utf-8")
        argv = ["--home", str(tmp_path / "home"), "agents", "run", str(plan), "--gateway", f"http://127.0.0.1:{port}",
                "--allow", f"syntax=\"{PY}\" -I -m py_compile {{file}}"]
        rc = await asyncio.to_thread(cli.main, argv)
        out = capsys.readouterr()
        assert rc == 0, out.err
        assert "## t1" in out.out and "return a + b" in out.out
        assert "t1 ok" in out.err and "vérification OK" in out.err and "2/2 sous-tâches" in out.err
        rc = await asyncio.to_thread(cli.main, argv[:5] + ["--gateway", f"http://127.0.0.1:{port}"])
        assert rc == 1 and "non autorisée" in capsys.readouterr().err  # no --allow: the command is refused
        server.should_exit = True
        await srv
    finally:
        await s.close()



# ============================================================ audit fixes (verification, merge)
async def test_verification_is_bounded_and_cancellable(tmp_path):
    import asyncio

    import myriad.agents as ag

    noisy = [PY, "-c", "import sys\nfor i in range(200000): sys.stdout.write('x' * 50 + chr(10))"]
    v = run_verification(noisy, GOOD, VerifySpec(run="noisy", timeout_s=60))
    assert v["passed"] and len(v["output"]) <= ag.MAX_OUTPUT_CHARS
    flag = tmp_path / "survived.txt"
    slow = [PY, "-c", f"import time, pathlib; time.sleep(3); pathlib.Path({str(flag)!r}).write_text('x')"]
    task = asyncio.ensure_future(ag.verify_async(slow, GOOD, VerifySpec(run="slow", timeout_s=60)))
    await asyncio.sleep(0.8)
    t0 = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - t0 < 2.5 and ag._VERIFY_SEM._value == ag.VERIFY_CONCURRENCY
    await asyncio.sleep(3)
    assert not flag.exists()  # the command was killed, not left running
    # a command that exits at once but leaves a child holding its output: the deadline still applies
    spawn = [PY, "-c", "import subprocess, sys; subprocess.Popen([sys.executable, '-c', "
                       "'import time; time.sleep(6); print(1)']); print('parent done')"]
    t0 = time.monotonic()
    v = await asyncio.to_thread(run_verification, spawn, GOOD, VerifySpec(run="spawn", timeout_s=1))
    assert not v["passed"] and "délai" in v["error"] and time.monotonic() - t0 < 5


def test_pick_candidate_keeps_inner_whitespace():
    c = [{"text": "```\nreturn 'a b'\n```", "weight": 0.5}, {"text": "```\nreturn 'ab'\n```", "weight": 0.4},
         {"text": "```\nreturn 'ab'  \n```", "weight": 0.2}]
    assert pick_candidate(c) == 1  # 'ab' with trailing blanks pools with 'ab' (0.6) over 'a b' (0.5)
    assert pick_candidate(c[:2]) == 0  # different programs are not pooled


async def test_merge_of_many_subtasks_and_merge_failure(tmp_path):
    s = await make_swarm(tmp_path, n=4)
    try:
        gw, _ = await s.add_gateway()
        plan = [task(i) for i in range(18)]
        res = await run_agents(gw, {"plan": plan, "combine": "merge", "budget": {"max_subtasks": 18, "max_parallel": 16}})
        assert res["status"] == "ok" and res["merged"] and len(res["merge"]["depends_on"]) == 18
        res = await run_agents(gw, {"plan": plan[:2], "combine": "merge", "merge": {"model": "nobody/none-GGUF", "skill": None}})
        assert res["status"] == "partial" and "fusion finale" in res["error"] and not res["merged"]
        assert "## t0" in res["result"]  # the sub-task results are kept, concatenated
    finally:
        await s.close()
