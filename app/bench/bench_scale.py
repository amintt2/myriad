"""E7: scalability of the Myriad protocol (essaim/1) and of its single tracker, without GPU.

A real tracker (its own process, so its CPU time is measured alone), N simulated nodes (FakeEngine with
the answer and compute-time models of bench/sim.py, spread over a few processes) and requesters
(real Gateway objects, in their own processes) talking through the real relay, optionally with an
emulated WAN delay injected in the tracker (myriad/netem.py).

    uv run python -m bench.bench_scale run --plan smoke          # 1 minute, checks the setup
    uv run python -m bench.bench_scale run --plan all            # latency, scale, throughput, failure
    uv run python -m bench.bench_scale report                    # results/e7_report.md from the JSON files

Roles of the same script (started by `run`): `tracker`, `nodes`, `requesters`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # allow `python bench/bench_scale.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "bench"

from bench import sim
from bench.common import RESULTS, Worker, emit, pct, r, stdin_commands

TIMEOUT_S = 60.0  # gateway timeout per request in the benchmark (production default: 120 s)
STARTER_CREDIT = 1e9  # requesters must not run out of credits during a run (production: 1000)


# ============================================================ tracker process
async def tracker_main(a) -> None:
    import uvicorn
    from myriad import priors
    from myriad.netem import wan_from_rtt
    from myriad.protocol import MAX_FRAME_BYTES
    from myriad.tracker import Tracker

    fams = sim.families(a.families)
    priors.load()["models"].update(sim.prior_entries(fams))  # in memory only
    wan = wan_from_rtt(a.rtt_ms, a.wan_sigma, seed=a.seed)
    tracker = Tracker(db_path=a.db, starter_credit=STARTER_CREDIT, seed=a.seed, wan=wan)
    http_calls: Counter = Counter()
    lag: list[float] = []

    @tracker.app.middleware("http")
    async def count_http(request, call_next):
        path = request.url.path
        http_calls["/v1/balance" if path.startswith("/v1/balance/") else path] += 1
        return await call_next(request)

    async def stats(reset: bool = False):
        lags = sorted(lag)
        out = {"cpu_s": time.process_time(), "wall": time.time(), "frames": dict(tracker.frames),
               "http": dict(http_calls), "conns": len(tracker.conns),
               "health": dict(getattr(tracker, "health_events", {})),
               "suspended": sum(1 for c in tracker.conns.values() if getattr(c, "suspended_until", None) is not None),
               "serving": sum(1 for c in tracker.conns.values() if c.info.model), "jobs_retained": len(tracker.jobs),
               "lag_ms": {"n": len(lags), "p50": r(pct(lags, 50) and pct(lags, 50) * 1000),
                          "p99": r(pct(lags, 99) and pct(lags, 99) * 1000), "max": r(lags[-1] * 1000 if lags else None)}}
        if reset:
            lag.clear()
        return out

    async def peers_cost(n: int = 20):
        """Time of answering /v1/peers (all serving peers) as FastAPI does it (view, then
        jsonable_encoder, then json), without any network delay; and of /v1/reliability. Wall time
        per step (perf_counter; the event loop is blocked meanwhile), plus the process CPU time of
        the whole /v1/peers batch (process_time ticks by 15.6 ms on Windows: only meaningful when
        the batch is long, as at large N)."""
        from fastapi.encoders import jsonable_encoder

        serving = [c for c in tracker.conns.values() if c.info.model]
        c0 = time.process_time()
        t0 = time.perf_counter()
        for _ in range(n):
            views = [tracker._peer_view(c) for c in serving]
        t1 = time.perf_counter()
        for _ in range(n):
            json.dumps(jsonable_encoder({"peers": views}))
        t2 = time.perf_counter()
        c2 = time.process_time()
        for _ in range(n):
            json.dumps(jsonable_encoder({"models": tracker.reliability()}))
        t3 = time.perf_counter()
        out = {"peers": len(serving), "peers_view_ms": (t1 - t0) / n * 1000, "peers_encode_ms": (t2 - t1) / n * 1000,
               "peers_ms": (t2 - t0) / n * 1000, "reliability_ms": (t3 - t2) / n * 1000,
               "peers_cpu_ms": (c2 - c0) / n * 1000, "batch_cpu_s": c2 - c0}
        if hasattr(tracker, "_rebuild_snapshot"):  # essaim/1.1: chunked snapshot and O(k) selection
            work, block = [], []
            for _ in range(n):
                await tracker._rebuild_snapshot()
                work.append(tracker.snapshot_stats["work_ms"])
                block.append(tracker.snapshot_stats["max_chunk_ms"])
            t5 = time.perf_counter()
            reps = 2000
            for _ in range(reps):
                tracker.select(4)
            t6 = time.perf_counter()
            # work: CPU-bound time of one rebuild; block: longest stretch without yielding to the loop
            out.update(snapshot_work_ms=statistics.mean(work), snapshot_block_ms=max(block),
                       snapshot_bytes=len(tracker._snapshot or b""), select4_us=(t6 - t5) / reps * 1e6)
        return out

    tracker.app.add_api_route("/bench/stats", stats, methods=["GET"])
    tracker.app.add_api_route("/bench/peers_cost", peers_cost, methods=["GET"])

    async def probe():
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.02)
            lag.append(max(0.0, time.perf_counter() - t - 0.02))
            if len(lag) > 200_000:
                del lag[:100_000]

    prof = None
    if a.profile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    server = uvicorn.Server(uvicorn.Config(tracker.app, host="127.0.0.1", port=a.port, log_level="warning",
                                           ws_max_size=MAX_FRAME_BYTES, timeout_graceful_shutdown=2, backlog=4096))
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            emit({"error": "tracker did not start"})
            return
        await asyncio.sleep(0.05)
    asyncio.create_task(probe())
    emit({"ready": True, "port": a.port, "wan": wan.describe() if wan else None})
    q: asyncio.Queue = asyncio.Queue()
    stdin_commands(asyncio.get_running_loop(), q)
    while (await q.get()).get("cmd") != "quit":
        pass
    if prof is not None:
        import pstats
        prof.disable()
        with open(a.profile, "w", encoding="utf-8") as f:
            pstats.Stats(prof, stream=f).sort_stats("cumulative").print_stats(60)
            pstats.Stats(prof, stream=f).sort_stats("tottime").print_stats(40)
    server.should_exit = True
    await task
    tracker.ledger.close()


# ============================================================ node worker process
async def nodes_main(a) -> None:
    from myriad.crypto import Identity
    from myriad.engine import FakeEngine
    from myriad.node import NodeClient

    fams = sim.families(a.families)
    am = sim.AnswerModel(a.collision, a.seed)
    cm = sim.ComputeModel(a.median_s, a.sigma, a.scale, a.node_sigma, per_family=not a.global_median, seed=a.seed)
    nodes, peers, tasks = [], [], []
    for i in range(a.offset, a.offset + a.count):
        fam = fams[i % len(fams)]
        peer = sim.SimPeer(f"n{i}", fam, am, cm)
        eng = FakeEngine(fam.model, reply=peer.reply, delay_s=peer.delay)
        # reconnect=True only while connecting: with thousands of nodes, the tracker's listen backlog
        # (capped by Windows) can refuse a connection, which is then retried after 1 s.
        nodes.append(NodeClient(Identity.generate(), a.tracker, engine=eng, model=fam.model, family=fam.family,
                                gguf=f"{fam.family}.gguf", params_b=fam.params_b, ctx=4096,
                                max_parallel=a.max_parallel, reconnect=True))
        peers.append(peer)
    for lo in range(0, len(nodes), 64):  # connect in batches
        batch = nodes[lo:lo + 64]
        tasks += [asyncio.create_task(n.run()) for n in batch]
        await asyncio.wait_for(asyncio.gather(*(n.connected.wait() for n in batch)), 120)
    for n in nodes:  # from now on a lost connection stays lost (the crash scenario relies on it)
        n.reconnect = False
    emit({"ready": True, "nodes": len(nodes),
          "connect_retries": sum(1 for n in nodes if n.session > 1 or n.last_error)})
    killed: set[int] = set()
    q: asyncio.Queue = asyncio.Queue()
    stdin_commands(asyncio.get_running_loop(), q)
    while True:
        cmd = await q.get()
        c = cmd.get("cmd")
        if c == "quit":
            break
        if c == "kill":
            alive = [i for i in range(len(nodes)) if i not in killed]
            chosen = random.Random(cmd.get("seed", 0)).sample(alive, round(cmd["frac"] * len(alive)))
            for i in chosen:
                killed.add(i)
                if cmd["mode"] == "crash":  # abrupt: the TCP connection is reset, no close handshake
                    ws = nodes[i]._ws
                    if ws is not None and ws.transport is not None:
                        ws.transport.abort()
                elif cmd["mode"] == "hang":  # connected, but its engine never answers again (nor its health probe)
                    peers[i].hung = True
                    nodes[i].engine.hung = True
                else:  # "hang_engine": generations never end, the engine's health probe still says OK
                    peers[i].hung = True
            emit({"killed": len(chosen), "mode": cmd["mode"]})
        elif c == "stats":
            emit({"cpu_s": time.process_time(), "wall": time.time(),
                  "calls": sum(n.engine.calls for n in nodes), "cancelled": sum(n.engine.cancelled for n in nodes),
                  "connected": sum(1 for n in nodes if n.connected.is_set())})
    for n in nodes:
        n.reconnect = False
    await asyncio.gather(*(n.stop() for n in nodes), return_exceptions=True)
    for t in tasks:
        t.cancel()


# ============================================================ requester worker process
async def requesters_main(a) -> None:
    from myriad.crypto import Identity
    from myriad.gateway import Gateway, GatewayError
    from myriad.node import NodeClient

    am = sim.AnswerModel(a.collision, a.seed)
    gws, tasks = [], []
    for _ in range(a.count):
        node = NodeClient(Identity.generate(), a.tracker, reconnect=False)
        tasks.append(asyncio.create_task(node.run()))
        await asyncio.wait_for(node.connected.wait(), 60)
        kw = {} if a.routing == "auto" else {"routing": a.routing}
        if a.no_replace:
            kw["replace"] = False
        gws.append(Gateway(node, default_k=a.k, timeout_s=TIMEOUT_S, peers_ttl_s=a.peers_ttl, **kw))
    emit({"ready": True, "gateways": len(gws)})

    async def one(g: int, qid: int, p: dict, inflight: list[int], out: list[dict]) -> None:
        inflight[g] += 1
        t_sub = time.time()
        t0 = time.perf_counter()
        rec = {"qid": qid, "gw": g, "t": t_sub, "gold": am.gold(qid)}
        try:
            ans = await gws[g].ask([{"role": "user", "content": sim.question(qid)}], max_tokens=64, temperature=0.0,
                                   seed=0, k=p["k"], task_hint="math", timeout_s=p["timeout_s"],
                                   early_stop=p["early_stop"])
            m = ans.meta
            rec.update(ok=True, answer=m["answer"], correct=m["answer"] == rec["gold"], asked=m["peers_asked"],
                       answered=m["peers_answered"], early=m["early_stop"], cert=m["certificate"],
                       decision=m["decision"], repl=m.get("replacements", 0), routing=m.get("routing", "directory"),
                       peers=[[x["family"], x["status"], x["error"], x["latency_ms"]] for x in m["peers"]])
        except GatewayError as e:
            rec.update(ok=False, error=e.code, msg=e.message[:300])
        except Exception as e:  # noqa: BLE001 - recorded, never fatal for the run
            rec.update(ok=False, error=type(e).__name__, msg=str(e)[:300])
        finally:
            inflight[g] -= 1
        rec["lat"] = time.perf_counter() - t0
        rec["done"] = time.time()
        out.append(rec)

    q: asyncio.Queue = asyncio.Queue()
    stdin_commands(asyncio.get_running_loop(), q)
    while True:
        cmd = await q.get()
        if cmd.get("cmd") == "quit":
            break
        if cmd.get("cmd") != "run":
            continue
        p = cmd
        await asyncio.sleep(max(0.0, p["start_at"] - time.time()))
        cpu0, wall0 = time.process_time(), time.time()
        rng = random.Random(p["seed"])
        inflight = [0] * len(gws)
        out: list[dict] = []
        running = []
        t_start = time.perf_counter()
        t_next, i = 0.0, 0
        while True:
            t_next += rng.expovariate(p["rate"])
            if t_next > p["duration"]:
                break
            await asyncio.sleep(max(0.0, t_start + t_next - time.perf_counter()))
            g = min(range(len(gws)), key=lambda j: (inflight[j], j))  # least loaded gateway
            running.append(asyncio.create_task(one(g, p["qid_base"] + i, p, inflight, out)))
            i += 1
        await asyncio.gather(*running)
        Path(p["out"]).write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")
        emit({"done": len(out), "out": p["out"], "cpu_s": time.process_time() - cpu0, "wall_s": time.time() - wall0})
    for gw in gws:
        await gw.close()
    for gw in gws:
        await gw.node.stop()
    for t in tasks:
        t.cancel()


# ============================================================ orchestrator
def http_get(url: str, timeout: float = 30.0) -> dict:
    import httpx
    return httpx.get(url, timeout=timeout).raise_for_status().json()


def expected_vote_accuracy(fams: list[sim.Family], k: int, collision: float, n: int = 20000, seed: int = 0) -> float:
    """Monte-Carlo accuracy of the gateway's full weighted vote under the answer model, with the prior
    weights (what the measured accuracy should approach when every asked peer answers)."""
    from myriad.fusion import reliability_weight, weighted_vote
    from myriad.priors import collision_prob

    am = sim.AnswerModel(collision, seed + 99)
    chosen = sorted(fams, key=lambda f: f.accuracy, reverse=True)[:k]
    w = [reliability_weight(f.accuracy, collision_prob("math")) for f in chosen]
    good = 0
    for q in range(n):
        answers = [am.answer(f"x{j}", f.accuracy, q) for j, f in enumerate(chosen)]
        good += weighted_vote(answers, w).answer == am.gold(q)
    return good / n


def summarize(recs: list[dict], duration: float, t0: float, s0: dict, s1: dict, n_node_workers_cpu: float | None,
              req_cpu: float | None) -> dict:
    ok = [x for x in recs if x.get("ok")]
    lat = [x["lat"] for x in ok]
    peer_err = Counter(p[2] for x in ok for p in x["peers"] if p[1] == "erreur")
    peer_status = Counter(p[1] for x in ok for p in x["peers"])
    last_done = max((x["done"] for x in recs), default=t0 + duration)
    dcpu, dwall = s1["cpu_s"] - s0["cpu_s"], s1["wall"] - s0["wall"]
    fin = {k: s1["frames"].get(k, 0) - s0["frames"].get(k, 0) for k in set(s1["frames"]) | set(s0["frames"])}
    frames_in = sum(v for k, v in fin.items() if k.startswith("in:"))
    frames_out = fin.get("out", 0)
    http = {k: s1["http"].get(k, 0) - s0["http"].get(k, 0) for k in s1["http"]}
    n = len(recs)
    return {
        "requests": n, "ok": len(ok), "failed": n - len(ok),
        "fail_codes": dict(Counter(x.get("error") for x in recs if not x.get("ok"))),
        "accuracy_ok": r(statistics.mean(x["correct"] for x in ok) if ok else None, 4),
        "accuracy_all": r(sum(x.get("correct", False) for x in recs) / n if n else None, 4),
        "latency_s": {"p50": r(pct(lat, 50)), "p95": r(pct(lat, 95)), "p99": r(pct(lat, 99)),
                      "mean": r(statistics.mean(lat) if lat else None), "max": r(max(lat) if lat else None)},
        "peers_asked_mean": r(statistics.mean(x["asked"] for x in ok) if ok else None),
        "peers_answered_mean": r(statistics.mean(x["answered"] for x in ok) if ok else None),
        "peers_answered_hist": dict(sorted(Counter(x["answered"] for x in ok).items())),
        "early_stop_frac": r(statistics.mean(bool(x["early"]) for x in ok) if ok else None),
        "certified_frac": r(statistics.mean(bool(x["cert"]) for x in ok) if ok else None),
        "peer_status": dict(peer_status), "peer_errors": dict(peer_err),
        "replacements_per_request": r(statistics.mean(x.get("repl", 0) for x in ok) if ok else None),
        "requests_with_replacement": r(statistics.mean(bool(x.get("repl", 0)) for x in ok) if ok else None),
        "routing": dict(Counter(x.get("routing", "directory") for x in ok)),
        "health": {k: s1.get("health", {}).get(k, 0) - s0.get("health", {}).get(k, 0)
                   for k in s1.get("health", {}) if s1["health"][k] - s0.get("health", {}).get(k, 0)},
        "suspended_end": s1.get("suspended"),
        "offered_rps": r(n / duration), "goodput_rps": r(len(ok) / max(duration, last_done - t0)),
        "drain_s": r(max(0.0, last_done - t0 - duration)),
        "tracker": {"cpu_s": r(dcpu), "wall_s": r(dwall), "cpu_util": r(dcpu / dwall if dwall else None),
                    "frames_in": frames_in, "frames_out": frames_out,
                    "frames_in_by_type": {k[3:]: v for k, v in sorted(fin.items()) if k.startswith("in:") and v},
                    "frames_per_request": r((frames_in + frames_out) / n if n else None, 2),
                    "cpu_us_per_frame": r(dcpu / (frames_in + frames_out) * 1e6 if frames_in + frames_out else None, 1),
                    "cpu_ms_per_request": r(dcpu / n * 1000 if n else None, 2),
                    "http_calls": http, "http_peers_per_s": r(http.get("/v1/peers", 0) / dwall if dwall else None, 2),
                    "loop_lag_ms": s1["lag_ms"], "jobs_retained": s1["jobs_retained"], "conns": s1["conns"]},
        "node_workers_cpu_util": r(n_node_workers_cpu), "requester_workers_cpu_util": r(req_cpu),
    }


def run_scenario(sc: dict, logdir: Path) -> dict:
    """Start a tracker, node workers and requester workers; run the phases; stop everything."""
    from myriad.engine import free_port

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    tag = sc["name"]
    tmp = Path(tempfile.mkdtemp(prefix="e7_"))
    common = ["--families", str(sc["families"]), "--seed", str(sc.get("seed", 0)),
              "--collision", str(sc.get("collision", sim.MEASURED_COLLISION))]
    workers: list[Worker] = []
    t_setup = time.time()
    try:
        tr = Worker("tracker", ["-m", "bench.bench_scale", "tracker", "--port", str(port), "--db",
                                str(tmp / "tracker.sqlite"), "--rtt-ms", str(sc["rtt_ms"]),
                                "--wan-sigma", str(sc.get("wan_sigma", 0.25)), *common,
                                *(["--profile", str(logdir / f"{tag}.profile.txt")] if sc.get("profile") else [])],
                    logdir / f"{tag}.tracker.log")
        workers.append(tr)
        tr.expect(60)
        N = sc["nodes"]
        n_w = max(1, min(8, math.ceil(N / 256)))  # 1 to 4 processes up to 1024 nodes (as in E7), 8 for 4096
        node_ws = []
        for w in range(n_w):
            lo, hi = N * w // n_w, N * (w + 1) // n_w
            nw = Worker(f"nodes{w}", ["-m", "bench.bench_scale", "nodes", "--tracker", url, "--offset", str(lo),
                                      "--count", str(hi - lo), "--scale", str(sc["scale"]),
                                      "--node-sigma", str(sc.get("node_sigma", 0.0)), *common],
                        logdir / f"{tag}.nodes{w}.log")
            workers.append(nw)
            node_ws.append(nw)
        for nw in node_ws:
            nw.expect(300)
        req_ws = []
        for g in range(sc["req_workers"]):
            rw = Worker(f"req{g}", ["-m", "bench.bench_scale", "requesters", "--tracker", url, "--count",
                                    str(sc["gateways_per_worker"]), "--k", str(sc["k"]),
                                    "--peers-ttl", str(sc.get("peers_ttl_s", 2.0)),
                                    "--routing", sc.get("routing", "auto"),
                                    *(["--no-replace"] if sc.get("no_replace") else []), *common],
                        logdir / f"{tag}.req{g}.log")
            workers.append(rw)
            req_ws.append(rw)
        for rw in req_ws:
            rw.expect(120)
        health = http_get(f"{url}/v1/health")
        assert health["serving"] == N, health
        setup_s = time.time() - t_setup
        cost = http_get(f"{url}/bench/peers_cost?n={sc.get('cost_reps', 5)}", 300)
        if sc.get("http_probe"):  # the same answer through HTTP, as a gateway sees it (no WAN delay at rtt 0)
            import httpx
            times = []
            with httpx.Client(timeout=60) as hc:
                for _ in range(20):
                    t0 = time.perf_counter()
                    body = hc.get(f"{url}/v1/peers").raise_for_status().content
                    times.append(time.perf_counter() - t0)
            cost["http_peers_ms_p50"] = r(pct(times, 50) * 1000, 2)
            cost["http_peers_bytes"] = len(body)
        phases = []
        for pi, ph in enumerate(sc["phases"]):
            s0 = http_get(f"{url}/bench/stats?reset=true")
            n0 = [nw.call({"cmd": "stats"}, 30) for nw in node_ws]
            start_at = time.time() + 1.0
            G = len(req_ws)
            outs = []
            for g, rw in enumerate(req_ws):
                out = tmp / f"p{pi}_r{g}.jsonl"
                outs.append(out)
                rw.send({"cmd": "run", "rate": ph["rate"] / G, "duration": ph["duration"], "start_at": start_at,
                         "qid_base": ph.get("qid_base", pi * 1_000_000) + g * 100_000, "seed": 1000 * pi + g,
                         "k": sc["k"], "timeout_s": ph.get("timeout_s", TIMEOUT_S),
                         "early_stop": ph.get("early_stop", True), "out": str(out)})
            timeline, kills, stop_poll = [], [], threading.Event()

            def poll():
                while not stop_poll.wait(2.0):
                    try:
                        s = http_get(f"{url}/bench/stats", 10)
                        timeline.append({"t": r(s["wall"] - start_at, 1), "cpu_s": s["cpu_s"], "wall": s["wall"],
                                         "conns": s["conns"], "serving": s["serving"],
                                         "suspended": s.get("suspended")})
                    except Exception:  # noqa: BLE001
                        pass
            poller = threading.Thread(target=poll, daemon=True)
            poller.start()
            if ph.get("kill"):
                kd = ph["kill"]
                time.sleep(max(0.0, start_at + kd["at_s"] - time.time()))
                for w, nw in enumerate(node_ws):
                    kills.append(nw.call({"cmd": "kill", "mode": kd["mode"], "frac": kd["frac"], "seed": w}, 30))
            dones = [rw.expect(ph["duration"] + ph.get("timeout_s", TIMEOUT_S) + 120) for rw in req_ws]
            stop_poll.set()
            s1 = http_get(f"{url}/bench/stats")
            n1 = [nw.call({"cmd": "stats"}, 30) for nw in node_ws]
            recs = []
            for out in outs:
                recs += [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
            node_cpu = max((b["cpu_s"] - a_["cpu_s"]) / max(1e-9, b["wall"] - a_["wall"]) for a_, b in zip(n0, n1))
            req_cpu = max(d["cpu_s"] / max(1e-9, d["wall_s"]) for d in dones)
            summ = summarize(recs, ph["duration"], start_at, s0, s1, node_cpu, req_cpu)
            summ["phase"] = {k: v for k, v in ph.items() if k != "qid_base"}
            summ["timeline_cpu_util"] = [
                {"t": b["t"], "util": r((b["cpu_s"] - a_["cpu_s"]) / max(1e-9, b["wall"] - a_["wall"]), 2),
                 "serving": b["serving"], "suspended": b.get("suspended")}
                for a_, b in zip(timeline, timeline[1:])]
            if kills:
                summ["killed"] = kills
                summ["windows"] = windows(recs, start_at, ph["kill"]["at_s"], ph["duration"])
            phases.append(summ)
            print(f"[{tag}] phase {pi}: {json.dumps(brief(summ))}", flush=True)
            if ph.get("stop_if_saturated") and saturated(summ, phases[0]):
                print(f"[{tag}] saturated, ladder stopped", flush=True)
                break
            time.sleep(sc.get("pause_s", 5.0))
        return {"scenario": {k: v for k, v in sc.items() if k != "phases"}, "setup_s": r(setup_s, 1),
                "directory_cost": cost, "phases": phases}
    finally:
        for w in reversed(workers):
            w.close()


def windows(recs: list[dict], t0: float, at_s: float, duration: float) -> dict:
    """Before / during (10 s after the failure) / after, by submission time."""
    out = {}
    for name, lo, hi in (("before", 0, at_s), ("during_10s", at_s, at_s + 10), ("after", at_s + 10, duration)):
        sel = [x for x in recs if lo <= x["t"] - t0 < hi]
        ok = [x for x in sel if x.get("ok")]
        lat = [x["lat"] for x in ok]
        out[name] = {"requests": len(sel), "ok": len(ok),
                     "accuracy_all": r(sum(x.get("correct", False) for x in sel) / len(sel) if sel else None, 4),
                     "p50": r(pct(lat, 50)), "p95": r(pct(lat, 95)), "max": r(max(lat) if lat else None),
                     "peers_answered_mean": r(statistics.mean(x["answered"] for x in ok) if ok else None),
                     "peer_errors": dict(Counter(p[2] for x in ok for p in x["peers"] if p[1] == "erreur")),
                     "fail_codes": dict(Counter(x.get("error") for x in sel if not x.get("ok")))}
    return out


def saturated(s: dict, first: dict) -> bool:
    base = first["latency_s"]["p95"] or 0.1
    return (s["goodput_rps"] < 0.9 * s["offered_rps"] or (s["latency_s"]["p95"] or 1e9) > 3 * base + 1.0
            or s["failed"] > 0.05 * max(1, s["requests"]))


def brief(s: dict) -> dict:
    t = s["tracker"]
    return {"offered": s["offered_rps"], "goodput": s["goodput_rps"], "ok": s["ok"], "fail": s["failed"],
            "acc": s["accuracy_all"], "p50": s["latency_s"]["p50"], "p95": s["latency_s"]["p95"],
            "answered": s["peers_answered_mean"], "cpu": t["cpu_util"], "us/frame": t["cpu_us_per_frame"],
            "lag99": t["loop_lag_ms"]["p99"], "busy": s["peer_errors"].get("peer_busy", 0)}


# ------------------------------------------------------------ plans
def plan_smoke() -> list[dict]:
    return [dict(name="smoke", nodes=16, families=4, k=4, rtt_ms=100, scale=0.05, req_workers=1,
                 gateways_per_worker=2, phases=[dict(rate=4, duration=8, early_stop=True),
                                                dict(rate=4, duration=8, early_stop=False)])]


def plan_latency() -> list[dict]:
    """A: measured compute times (scale 1), light load, RTT x {certificate, wait for all}; then k = 5..7
    and heterogeneous node speeds at RTT 100 ms."""
    out = []
    base = dict(nodes=256, scale=1.0, req_workers=1, gateways_per_worker=6, pause_s=8.0)
    both = [dict(rate=2.0, duration=75, early_stop=True, qid_base=0),
            dict(rate=2.0, duration=75, early_stop=False, qid_base=0)]
    for rtt in (0, 50, 100, 150):
        out.append(dict(base, name=f"lat_rtt{rtt}_k4", families=4, k=4, rtt_ms=rtt, phases=both))
    for k in (5, 6, 7):
        out.append(dict(base, name=f"lat_rtt100_k{k}", families=k, k=k, rtt_ms=100, phases=both))
    out.append(dict(base, name="lat_rtt100_k4_hetero", families=4, k=4, rtt_ms=100, node_sigma=0.5, phases=both))
    return out


def plan_scale() -> list[dict]:
    """B: N x RTT at a fixed per-node load (~30 % of node capacity, capped), compute times / 10."""
    out = []
    for n in (4, 16, 64, 256, 1024):
        rate = min(0.15 * n, 12.0)
        dur = max(20.0, 60.0 / rate)
        for rtt in (0, 50, 100, 150):
            out.append(dict(name=f"scale_n{n}_rtt{rtt}", nodes=n, families=4, k=4, rtt_ms=rtt, scale=0.1,
                            req_workers=1 if rate < 20 else 2, gateways_per_worker=8,
                            phases=[dict(rate=rate, duration=dur, early_stop=True)]))
    return out


LADDER = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300, 400)


def plan_throughput() -> list[dict]:
    """C: offered load ladder until the system saturates, compute times / 40 so that nodes are not
    the bottleneck."""
    out = []
    for n, rtt, ttl in ((64, 0, 2.0), (256, 0, 2.0), (1024, 0, 2.0), (1024, 100, 2.0), (1024, 0, 30.0)):
        name = f"tput_n{n}_rtt{rtt}" + ("" if ttl == 2.0 else f"_ttl{ttl:g}")
        out.append(dict(name=name, nodes=n, families=4, k=4, rtt_ms=rtt, scale=0.025, peers_ttl_s=ttl,
                        req_workers=3, gateways_per_worker=8, pause_s=5.0,
                        phases=[dict(rate=x, duration=15, early_stop=True, stop_if_saturated=True,
                                     qid_base=i * 1_000_000) for i, x in enumerate(LADDER)]))
    return out


def plan_failure() -> list[dict]:
    """D: 25 % of the nodes fail 40 s into a 120 s run (measured compute times, RTT 50 ms)."""
    out = []
    for mode in ("crash", "hang"):
        out.append(dict(name=f"fail_{mode}", nodes=256, families=4, k=4, rtt_ms=50, scale=1.0, req_workers=1,
                        gateways_per_worker=6,
                        phases=[dict(rate=2.5, duration=120, early_stop=True, timeout_s=30.0,
                                     kill=dict(at_s=40.0, mode=mode, frac=0.25))]))
    return out


def plan_profile() -> list[dict]:
    """Diagnostic (not reported as a result): where the tracker spends its CPU at N = 1024."""
    return [dict(name="profile_n1024", nodes=1024, families=4, k=4, rtt_ms=0, scale=0.025, req_workers=3,
                 gateways_per_worker=8, profile=True, phases=[dict(rate=15, duration=30, early_stop=True)]),
            dict(name="profile_n64", nodes=64, families=4, k=4, rtt_ms=0, scale=0.025, req_workers=3,
                 gateways_per_worker=8, profile=True, phases=[dict(rate=30, duration=30, early_stop=True)])]


def plan_dircost() -> list[dict]:
    """E: cost of one /v1/peers answer against N (no load), the term that grows with the network."""
    return [dict(name=f"dircost_n{n}", nodes=n, families=4, k=4, rtt_ms=0, scale=1.0, req_workers=0,
                 gateways_per_worker=1, cost_reps=20, http_probe=True, phases=[]) for n in (4, 16, 64, 256, 1024)]


# ------------------------------------------------------------ essaim/1.1 (results in e7_v11_*.json)
V11_N = (64, 256, 1024, 4096)


def plan_v11_scale() -> list[dict]:
    """B again with essaim/1.1: fixed load, RTT 100 ms, N up to 4096."""
    out = []
    for n in V11_N:
        rate = min(0.15 * n, 12.0)
        out.append(dict(name=f"scale_n{n}_rtt100", nodes=n, families=4, k=4, rtt_ms=100, scale=0.1, req_workers=1,
                        gateways_per_worker=8, phases=[dict(rate=rate, duration=max(20.0, 60.0 / rate),
                                                            early_stop=True)]))
    return out


def plan_v11_throughput() -> list[dict]:
    """C again with essaim/1.1: RTT 100 ms for every N, and RTT 0 where E7 measured it (comparison)."""
    out = []
    for n, rtt in [(n, 100) for n in V11_N] + [(64, 0), (256, 0), (1024, 0)]:
        out.append(dict(name=f"tput_n{n}_rtt{rtt}", nodes=n, families=4, k=4, rtt_ms=rtt, scale=0.025,
                        peers_ttl_s=2.0, req_workers=3, gateways_per_worker=8, pause_s=5.0,
                        phases=[dict(rate=x, duration=15, early_stop=True, stop_if_saturated=True,
                                     qid_base=i * 1_000_000) for i, x in enumerate(LADDER)]))
    return out


def plan_v11_failure() -> list[dict]:
    """D again with essaim/1.1, plus a hang that the engine's health probe does not see."""
    out = []
    for mode in ("crash", "hang", "hang_engine"):
        out.append(dict(name=f"fail_{mode}", nodes=256, families=4, k=4, rtt_ms=50, scale=1.0, req_workers=1,
                        gateways_per_worker=6,
                        phases=[dict(rate=2.5, duration=120, early_stop=True, timeout_s=30.0,
                                     kill=dict(at_s=40.0, mode=mode, frac=0.25))]))
    return out


def plan_v11_busy() -> list[dict]:
    """Busy peers: N = 64 nodes at node saturation (compute / 10), with and without replacement."""
    out = []
    for suffix, routing, repl in (("", "auto", True), ("_noreplace", "auto", False),
                                  ("_directory", "directory", False)):  # the last one behaves as essaim/1
        out.append(dict(name=f"busy_n64{suffix}", nodes=64, families=4, k=4, rtt_ms=0, scale=0.1, req_workers=3,
                        gateways_per_worker=8, routing=routing, no_replace=not repl, pause_s=5.0,
                        phases=[dict(rate=x, duration=20, early_stop=True, qid_base=i * 1_000_000)
                                for i, x in enumerate((10, 20, 30, 40))]))
    return out


def plan_v11_dircost() -> list[dict]:
    return [dict(name=f"dircost_n{n}", nodes=n, families=4, k=4, rtt_ms=0, scale=1.0, req_workers=0,
                 gateways_per_worker=1, cost_reps=20, http_probe=True, phases=[]) for n in (64, 256, 1024, 4096)]


PLANS = {"smoke": plan_smoke, "profile": plan_profile, "dircost": plan_dircost, "latency": plan_latency, "scale": plan_scale, "throughput": plan_throughput,
         "failure": plan_failure, "v11_dircost": plan_v11_dircost, "v11_scale": plan_v11_scale,
         "v11_throughput": plan_v11_throughput, "v11_failure": plan_v11_failure, "v11_busy": plan_v11_busy}


def orchestrate(a) -> None:
    import platform

    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.plan == "all":  # E7 as published (essaim/1 plans)
        names = [n for n in PLANS if n not in ("smoke", "profile") and not n.startswith("v11_")]
    elif a.plan == "v11":  # the essaim/1.1 re-run
        names = [n for n in PLANS if n.startswith("v11_")]
    else:
        names = [a.plan]
    logdir = Path(tempfile.mkdtemp(prefix="e7_logs_"))
    print(f"logs: {logdir}", flush=True)
    for name in names:
        scenarios = PLANS[name]()
        if a.only:
            scenarios = [s for s in scenarios if a.only in s["name"]]
        out_path = RESULTS / f"e7_{name}.json"
        # Scenarios already in the file are kept (with --only or --resume); --resume also skips them.
        kept: dict[str, dict] = {}
        if out_path.exists() and (a.resume or a.only):
            kept = {x["scenario"]["name"]: x for x in json.loads(out_path.read_text(encoding="utf-8"))["scenarios"]}
        by_name = dict(kept)
        for sc in scenarios:
            if a.resume and sc["name"] in kept:
                continue
            print(f"== {sc['name']}", flush=True)
            t0 = time.time()
            res = run_scenario(sc, logdir)
            fams = sim.families(sc["families"])
            res["expected_full_vote_accuracy"] = r(expected_vote_accuracy(fams, sc["k"], sc.get("collision",
                                                   sim.MEASURED_COLLISION)), 4)
            res["best_single_accuracy"] = max(f.accuracy for f in fams)
            res["elapsed_s"] = r(time.time() - t0, 1)
            by_name[sc["name"]] = res
            write_json(out_path, name, list(by_name.values()), platform)
        write_json(out_path, name, list(by_name.values()), platform)
    report()


def write_json(path: Path, plan: str, results: list[dict], platform) -> None:
    import inspect
    import os

    import myriad
    from myriad.tracker import Tracker

    tracker_defaults = {k: p.default for k, p in inspect.signature(Tracker.__init__).parameters.items()
                        if p.default is not inspect.Parameter.empty and k not in ("db_path", "seed", "wan")}
    meta = {"experiment": "E7", "plan": plan, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol_version": getattr(myriad, "PROTOCOL_VERSION", myriad.PROTOCOL),
            "tracker_defaults": tracker_defaults,
            "machine": {"platform": platform.platform(), "python": platform.python_version(),
                        "cpu": platform.processor(), "cpus": os.cpu_count()},
            "settings": {"gateway_timeout_s": TIMEOUT_S, "starter_credit": STARTER_CREDIT,
                         "answer_model": {"collision": sim.MEASURED_COLLISION,
                                          "popular_share": math.sqrt(sim.MEASURED_COLLISION)},
                         "compute_model": {"median_s": sim.MEASURED_MEDIAN_S, "sigma": sim.MEASURED_SIGMA,
                                           "per_family_medians": {f.family: f.median_s for f in sim.MEASURED_FAMILIES}},
                         "families": [f.__dict__ for f in sim.families(7)],
                         "node_max_parallel": 1, "wan_sigma": 0.25, "gateway_peers_ttl_s": 2.0,
                         "tracker": "production defaults (spot_rate 0.05, sweep 0.25 s, job TTL 600 s), SQLite on disk"},
            "scenarios": results}
    path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ------------------------------------------------------------ report
def report() -> None:
    from bench.report_e7 import write_report

    write_report(RESULTS)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="role", required=True)

    def common(p):
        p.add_argument("--families", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--collision", type=float, default=sim.MEASURED_COLLISION)

    p = sub.add_parser("run", help="run a plan (orchestrator)")
    p.add_argument("--plan", choices=["all", "v11", *PLANS], default="smoke")
    p.add_argument("--only", help="only the scenarios whose name contains this text")
    p.add_argument("--resume", action="store_true", help="keep the scenarios already in the JSON file")
    p = sub.add_parser("report", help="write results/e7_report.md from the JSON files")
    p = sub.add_parser("tracker")
    common(p)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--rtt-ms", type=float, default=0.0)
    p.add_argument("--wan-sigma", type=float, default=0.25)
    p.add_argument("--profile", help="write a cProfile report of the tracker to this file (slows it down)")
    p = sub.add_parser("nodes")
    common(p)
    p.add_argument("--tracker", required=True)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--median-s", type=float, default=sim.MEASURED_MEDIAN_S)
    p.add_argument("--sigma", type=float, default=sim.MEASURED_SIGMA)
    p.add_argument("--node-sigma", type=float, default=0.0)
    p.add_argument("--global-median", action="store_true", help="same median for every family")
    p.add_argument("--max-parallel", type=int, default=1)
    p = sub.add_parser("requesters")
    common(p)
    p.add_argument("--tracker", required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--peers-ttl", type=float, default=2.0, help="gateway directory cache (production: 2 s)")
    p.add_argument("--routing", choices=["auto", "tracker", "directory"], default="auto")
    p.add_argument("--no-replace", action="store_true", help="do not replace refused peers (ablation)")
    a = ap.parse_args(argv)
    if a.role == "run":
        orchestrate(a)
    elif a.role == "report":
        report()
    else:
        import logging
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
        fn = {"tracker": tracker_main, "nodes": nodes_main, "requesters": requesters_main}[a.role]
        try:
            asyncio.run(fn(a))
        except Exception as e:  # noqa: BLE001 - reported to the orchestrator
            emit({"error": f"{type(e).__name__}: {e}"[:500]})
            raise


if __name__ == "__main__":
    main()
