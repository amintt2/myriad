"""Cost of the end-to-end encryption (essaim/1.3): CPU and latency per job.

1. Primitives (no network): sealing and opening a job (X25519 + HKDF + ChaCha20-Poly1305 + the
   pseudonym's ed25519 signature and its check) and an answer, for several sizes. CPU time
   (process_time) and wall time, median over N runs.
2. In process, a real tracker and nodes with instant fake engines: the latency of a request with
   encryption (Reserve -> Assigned -> SealedJob) and without (plaintext, routed in one frame), k = 1 and
   k = 4, without and with an emulated WAN (one-way delay): the reservation costs one more
   requester <-> tracker round trip.

    uv run python -m bench.bench_crypto [--n 200] [--asks 40]
Writes bench/results/crypto.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "bench"

from bench.common import RESULTS  # noqa: E402
from myriad.crypto import Identity  # noqa: E402
from myriad.e2e import Keyring, ReplayCache, open_job, open_result, seal_job, seal_result  # noqa: E402
from myriad.protocol import Job  # noqa: E402


def timed(fn, n: int) -> dict:
    """Median and p90 wall time per call; CPU time per call averaged over the n calls (the process
    clock ticks every 15.6 ms on Windows, too coarse for one call)."""
    wall = []
    c0 = time.process_time()
    for _ in range(n):
        w0 = time.perf_counter()
        fn()
        wall.append((time.perf_counter() - w0) * 1000)
    cpu = (time.process_time() - c0) * 1000 / n
    return {"cpu_ms": round(cpu, 3), "wall_ms": round(statistics.median(wall), 3),
            "wall_p90_ms": round(sorted(wall)[int(0.9 * (n - 1))], 3)}


def primitives(n: int) -> list[dict]:
    req, peer = Identity.generate(), Identity.generate()
    ring = Keyring(peer)
    out = []
    for prompt_chars, answer_chars in ((300, 600), (4_000, 2_000), (30_000, 8_000), (120_000, 60_000)):
        content = ("x" * 99 + " ") * (prompt_chars // 100)
        msgs = [{"role": "user", "content": content[i:i + 30_000]} for i in range(0, len(content), 30_000)]
        jobs = [Job(job_id=f"{i:032x}", requester_id=req.node_id, messages=msgs, max_tokens=512) for i in range(n)]
        it = iter(jobs)
        sealed = []
        seal = timed(lambda: sealed.append(seal_job(next(it), peer.node_id, ring.cert())), n)
        it2 = iter(sealed)
        cache = ReplayCache()
        opened = []
        open_t = timed(lambda: opened.append(open_job(*(lambda sj_s: (sj_s[0], sj_s[1].pseudonym.pubkey))(next(it2)),
                                                       ring, cache, peer.node_id)), n)
        text = "y" * answer_chars
        it3 = iter(opened)
        results = []
        seal_r = timed(lambda: results.append(seal_result(peer, next(it3)[1], "m", text, "stop", 100, -0.1, 1.0)), n)
        it4 = iter(zip(results, sealed))
        open_r = timed(lambda: (lambda sr, s: open_result(sr, s[1], peer.pubkey))(*next(it4)), n)
        size = len(sealed[0][0].ct)
        out.append({"prompt_chars": prompt_chars, "answer_chars": answer_chars, "ciphertext_b64_chars": size,
                    "seal_job": seal, "open_job": open_t, "seal_result": seal_r, "open_result": open_r,
                    "total_cpu_ms": round(sum(x["cpu_ms"] for x in (seal, open_t, seal_r, open_r)), 3)})
    return out


async def swarm_latency(asks: int, wan_ms: float) -> list[dict]:
    from myriad.engine import FakeEngine
    from myriad.netem import WanDelay
    from myriad.security import Security
    from tests.conftest import MODELS, start_swarm

    out = []
    for e2e in (True, False):
        tmp = Path(tempfile.mkdtemp())
        kw = {"wan": WanDelay(wan_ms)} if wan_ms else {}
        kw["starter_credit"] = 1e7  # the asks really spend credits
        s = await start_swarm(tmp, **kw)
        try:
            unlimited = {"rate_per_min": 0, "max_concurrent": 0}
            for fam in ("qwen", "smollm", "gemma", "granite"):
                await s.add_node(fam, FakeEngine(MODELS[fam], "The answer is 42.", tokens=8), MODELS[fam],
                                 max_parallel=8, e2e=e2e, security=Security(unlimited))
            gw, _ = await s.add_gateway(security=Security({"require_e2e": e2e, "rotate_peers": False, **unlimited}))
            q = [{"role": "user", "content": "Tom has 40 apples and buys 2 more. How many apples does he have?"}]
            if not e2e:  # the reference: the essaim/1.2 protocol (routed job in one frame, no reservation)
                feats = await gw.features()
                gw._feat = (gw.node.session, feats - {"e2e", "policy", "report"})
            for k in (1, 4):
                await gw.ask(q, k=k, early_stop=False)  # warm-up
                lat = []
                for _ in range(asks):
                    t0 = time.perf_counter()
                    ans = await gw.ask(q, k=k, early_stop=False)
                    lat.append((time.perf_counter() - t0) * 1000)
                    assert ans.meta["privacy"]["e2e"] == e2e
                out.append({"e2e": e2e, "k": k, "wan_one_way_ms": wan_ms, "asks": asks,
                            "median_ms": round(statistics.median(lat), 2),
                            "p90_ms": round(sorted(lat)[int(0.9 * (asks - 1))], 2)})
        finally:
            await s.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--asks", type=int, default=40)
    args = ap.parse_args()
    prim = primitives(args.n)
    lat = []
    for wan in (0.0, 25.0):
        lat += asyncio.run(swarm_latency(args.asks if not wan else max(10, args.asks // 4), wan))
    out = {"machine": {"platform": platform.platform(), "processor": platform.processor(),
                       "python": platform.python_version()},
           "primitives": prim, "swarm_latency": lat}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "crypto.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
