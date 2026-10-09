"""Sub-agents on the in-process demo swarm (no GPU): parallel dispatch measured.

The 16 simulated peers of scripts/demo_swarm.py (six model families), each generation taking exactly
`--gen-s` seconds (3 s by default). Three plans:
- parallel: 6 independent sub-tasks -> wall time ≈ one sub-task, not six;
- chain: a -> b -> c -> wall time ≈ three sub-tasks, and each one receives the previous result;
- fan-in: 6 independent sub-tasks then one depending on all of them -> ≈ two sub-tasks.

    uv run python -m bench.bench_agents            # writes bench/results/agents_parallel.json
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import tempfile
import time
from pathlib import Path

import uvicorn

from myriad.agents import AgentRun, parse_request
from myriad.crypto import Identity
from myriad.engine import FakeEngine, free_port
from myriad.gateway import Gateway
from myriad.node import NodeClient
from myriad.priors import family_of, params_of
from myriad.protocol import MAX_FRAME_BYTES
from myriad.tracker import Tracker

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "results" / "agents_parallel.json"


def demo_peers() -> list:
    spec = importlib.util.spec_from_file_location("demo_swarm", ROOT / "scripts" / "demo_swarm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PEERS


def echo(log: list):
    def reply(messages):
        user = messages[-1]["content"]
        log.append(user)
        m = re.search(r"TOKEN=(\w+)", user)
        return f"result of {m.group(1) if m else '?'}"
    return reply


async def main(args) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="myriad-agents-bench-"))
    tracker = Tracker(db_path=tmp / "tracker.sqlite", starter_credit=10**9, spot_rate=0.0, sweep_s=0.1)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(tracker.app, host="127.0.0.1", port=port, log_level="warning",
                                           ws_max_size=MAX_FRAME_BYTES))
    tasks = [asyncio.create_task(server.serve())]
    while not server.started:
        await asyncio.sleep(0.05)
    url = f"http://127.0.0.1:{port}"
    log: list = []
    nodes = []
    for model, _, _ in demo_peers():
        n = NodeClient(Identity.generate(), url, engine=FakeEngine(model, echo(log), delay_s=args.gen_s, tokens=50),
                       model=model, family=family_of(model), gguf="model.gguf", params_b=params_of(model), ctx=4096,
                       max_parallel=1)
        nodes.append(n)
        tasks.append(asyncio.create_task(n.run()))
    client = NodeClient(Identity.generate(), url)
    tasks.append(asyncio.create_task(client.run()))
    await asyncio.wait_for(asyncio.gather(*(n.connected.wait() for n in nodes + [client])), 10)
    await asyncio.sleep(0.5)
    gw = Gateway(client, timeout_s=60)

    def t(i, deps=()):
        return {"id": f"s{i}", "prompt": f"Part {i}. TOKEN=s{i}", "depends_on": list(deps)}

    plans = {
        "parallel": [t(i) for i in range(6)],
        "chain": [t("a"), t("b", ["sa"]), t("c", ["sb"])],
        "fan-in": [t(i) for i in range(6)] + [t("z", [f"s{i}" for i in range(6)])],
    }
    out = {"gen_s": args.gen_s, "peers": len(nodes), "families": len({family_of(n.model) for n in nodes}), "runs": {}}
    try:
        for name, plan in plans.items():
            log.clear()
            t0 = time.perf_counter()
            res = await AgentRun(gw, parse_request({"task": f"bench {name}", "plan": plan})).execute()
            wall = time.perf_counter() - t0
            peers = [s["peer"]["node_id"] for s in res["subtasks"] if s.get("peer")]
            row = {"status": res["status"], "subtasks": len(plan), "wall_s": round(wall, 2),
                   "sum_s": round(res["timing"]["subtasks_sum_ms"] / 1000, 2),
                   "speedup": res["timing"]["parallel_speedup"], "distinct_peers": len(set(peers)),
                   "families": sorted({s["peer"]["family"] for s in res["subtasks"] if s.get("peer")})}
            if name == "chain":  # each step received the previous one's result
                row["context_passed"] = (any("result of sa" in u and "TOKEN=sb" in u for u in log)
                                         and any("result of sb" in u and "TOKEN=sc" in u for u in log)
                                         and not any("result of sa" in u and "TOKEN=sc" in u for u in log))
            out["runs"][name] = row
            print(f"{name:9s} {row}", flush=True)
    finally:
        await gw.close()
        for n in nodes + [client]:
            await n.stop()
        for x in tasks[1:]:
            x.cancel()
        await asyncio.gather(*tasks[1:], return_exceptions=True)
        server.should_exit = True  # a clean stop (cancelling uvicorn logs a traceback)
        await asyncio.wait({tasks[0]}, timeout=5)
        tracker.ledger.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-s", type=float, default=3.0, help="simulated generation time of every peer (s)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    result = asyncio.run(main(a))
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", a.out)
