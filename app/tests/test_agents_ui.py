"""The Agents view of the local interface: demo plan, token-checked run stream, assets and dictionary."""
from __future__ import annotations

import httpx

from myriad.agents import demo_verify_commands
from myriad.ui import make_ui_app

from .test_agents import make_swarm, sse, task


async def test_ui_agents_endpoints(tmp_path):
    s = await make_swarm(tmp_path, n=3, delay=0.05)
    try:
        gw, _ = await s.add_gateway()
        gw.verify_commands = demo_verify_commands()
        app = make_ui_app(gw.node, gw, token="tok")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8401") as c:
            demo = (await c.get("/api/agents/demo")).json()
            assert len(demo["plan"]) == 6 and demo["plan"][0]["verify"]["run"] == "python-syntax"
            assert "to-do" in (await c.get("/api/agents/demo", params={"lang": "en"})).json()["task"]
            r = await c.post("/api/agents/run", json={"plan": [task(1)]})
            assert r.status_code == 403  # no interface token
            r = await c.post("/api/agents/run", json={"plan": [task(1, verify={"run": "rm"})]},
                             headers={"X-Myriad-Token": "tok"})
            assert r.status_code == 400 and "non autorisée" in r.json()["error"]
            r = await c.post("/api/agents/run", json={"plan": [task(1), task(2, ["t1"])]},
                             headers={"X-Myriad-Token": "tok"})
            evs = sse(r.text)
            assert evs[-1]["type"] == "done" and evs[-1]["result"]["status"] == "ok"
            js = await c.get("/static/agents.js")
            assert js.status_code == 200 and "ag.title" in js.text
            page = (await c.get("/")).text
            assert 'id="view-agents"' in page and 'data-view="agents"' in page and "/static/agents.js" in page
            assert page.index("/static/agents.js") < page.index("/static/app.js")  # its keys before apply()
    finally:
        await s.close()
