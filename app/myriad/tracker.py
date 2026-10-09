"""Tracker: rendezvous, WebSocket relay, credits ledger and reliability statistics.

Nodes connect OUT to the tracker (so a node behind a NAT box needs no open port) and prove their
identity by signing a challenge. A requester sends signed jobs naming a target node; the tracker
checks them and forwards them; the target sends back a signed result, which is relayed to the
requester; the requester then sends a signed receipt and the tracker settles the credits.
The tracker never runs a model.

essaim/1.1 (backward compatible, see protocol.py):
- Server-side peer selection. Serving nodes that can take a job now (connected, accepting, not
  suspended, reputation >= 0.3, a free slot) sit in per-model indexes (idle / partly busy), kept up to
  date at every change in O(1). A job sent without a target but with a `route` is given a peer of a
  family not yet used by its request group: models are walked by decreasing reliability, and the peer
  is the better of two random draws in the model's idle set (fewer recent strikes), else in its
  partly busy set (lower load). Cost O(k) per request, independent of the number of nodes N; the
  requester learns the chosen peer from an Assigned frame. GET /v1/select gives the same choice.
- Health. Each node answering pings is pinged every `ping_s`; a missing Pong after `ping_timeout_s`,
  or a Pong saying the engine is down, suspends the node. Jobs that time out (requester deadline of at
  least `strike_min_deadline_s`) are strikes; `timeout_strikes` consecutive strikes suspend the node.
  A suspension lasts suspend_base_s * 2^level (capped), the level growing at each new suspension and
  falling back to 0 at the next delivered result; a pinging node is readmitted only after a good Pong.
  Suspended nodes are never selected.
- GET /v1/peers (the full directory, for the UI and essaim/1 gateways) is a cached snapshot, rebuilt
  in the background in chunks at most every `peers_snapshot_s`: answering it never costs O(N).
- Housekeeping uses deadline heaps and FIFO queues: no periodic scan of all jobs.

essaim/1.2 (feature "tags"): nodes advertise skill tags (NodeInfo.tags). Besides its model pool, a
selectable node sits in one pool per (tag, model), kept up to date by the same O(1) reindexing, so a
job routed by tag (Route.tag) is given a peer in O(k) too: the tag's models are walked by decreasing
reliability. When no selectable node has the tag, any node is chosen and Assigned.tag_match says so.
Route.family restricts the choice to one model family (no fallback).
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
import math
import random
import secrets
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import ValidationError

from . import FEATURES, PROTOCOL, PROTOCOL_VERSION, __version__
from .crypto import Identity, pubkey_matches
from .fusion import detect_task_hint, extract_answer
from .ledger import Ledger
from .netem import WanDelay
from .priors import MILLI, beta_mean, credit_factor, family_of, prior_accuracy, trusted_params
from .protocol import (MAX_FRAME_BYTES, MAX_PROMPT_CHARS, Assigned, Cancel, Challenge, ErrorFrame, Hello, Job,
                       JobError, JobFrame, JobResult, NodeInfo, PeerCard, Ping, Pong, ReceiptFrame, ResultFrame,
                       Route, Status, Welcome, dump_frame, parse_frame)

log = logging.getLogger("myriad.tracker")
HELLO_TIMEOUT_S = 10
OUTBOX = 1000
MIN_REPUTATION = 0.3  # below this a node is never selected (the essaim/1 gateways apply the same rule)
SNAPSHOT_CHUNK = 256  # peers encoded between two yields to the event loop
GROUP_GRACE_S = 5.0
MAX_GROUP_JOBS = 32  # jobs (first picks and replacements) of one request group
MAX_SELECT_K = 16
TIMEOUT_ERRORS = ("deadline",)  # node-reported job errors that count as timeouts
REPUTATION_REFRESH_S = 5.0


class IndexedSet:
    """A set with O(1) add, discard and uniform random draw."""
    __slots__ = ("items", "pos")

    def __init__(self):
        self.items: list = []
        self.pos: dict = {}

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, x) -> bool:
        return x in self.pos

    def add(self, x) -> None:
        if x not in self.pos:
            self.pos[x] = len(self.items)
            self.items.append(x)

    def discard(self, x) -> None:
        i = self.pos.pop(x, None)
        if i is None:
            return
        last = self.items.pop()
        if i < len(self.items):
            self.items[i] = last
            self.pos[last] = i


@dataclass(eq=False)
class Pool:
    """Selectable nodes serving one model: idle (no job) and partly busy (a free slot left)."""
    model: str
    family: str
    idle: IndexedSet = field(default_factory=IndexedSet)
    partial: IndexedSet = field(default_factory=IndexedSet)

    def __len__(self) -> int:
        return len(self.idle) + len(self.partial)


@dataclass(eq=False)
class Conn:
    node_id: str
    info: NodeInfo
    ws: WebSocket
    outbox: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(OUTBOX))
    busy: int = 0  # jobs routed to this node and not finished
    submitted: int = 0  # jobs this node requested and still in flight
    connected_at: float = field(default_factory=time.time)
    closed: bool = False
    wan: WanDelay | None = None  # WAN emulation (experiments only): delay of each frame sent
    inbox: asyncio.Queue | None = None  # WAN emulation: received frames waiting for their delivery time
    _out_due: float = 0.0
    _in_due: float = 0.0
    # essaim/1.1 bookkeeping
    jobs_in: set = field(default_factory=set)  # pending jobs routed to this node
    jobs_out: set = field(default_factory=set)  # pending jobs this node requested
    slot: tuple | None = None  # (model, "idle" | "partial") while selectable
    reputation: float = 1.0
    spot: list = field(default_factory=lambda: [0, 0])  # spot-check agreements, disagreements on its model
    pings: bool | None = None  # None: unknown yet; True: answers pings (essaim/1.1); False: does not
    ping_seq: int = 0
    ping_sent: float | None = None  # monotonic time of the ping waiting for its Pong
    ping_due: float | None = None  # due time of the one scheduled ping
    rtt_ms: float | None = None
    engine_ok: bool = True
    strikes: int = 0  # consecutive bad outcomes (timeouts)
    level: int = 0  # suspension backoff exponent; back to 0 after a delivered result
    suspended_until: float | None = None
    suspend_reason: str | None = None

    def send(self, frame) -> None:
        if self.closed:
            return
        due = 0.0
        if self.wan is not None:  # never before the frame sent ahead of it: order is preserved
            due = self._out_due = max(time.monotonic() + self.wan.sample(), self._out_due)
        try:
            self.outbox.put_nowait((due, dump_frame(frame)))
        except asyncio.QueueFull:  # a peer that does not read its messages is dropped
            self.closed = True

    def in_due(self) -> float:
        """Delivery time of a frame received now (WAN emulation), in arrival order."""
        self._in_due = max(time.monotonic() + self.wan.sample(), self._in_due)
        return self._in_due

    @property
    def family(self) -> str:
        return self.info.family or self.info.model or ""

    @property
    def tags(self) -> tuple:
        return tuple(self.info.tags or ())


@dataclass(eq=False)
class TrackedJob:
    job: Job
    requester: str
    target: str
    model: str
    factor: float
    deadline: float
    status: str = "pending"  # pending, delivered, failed, expired, cancelled
    result: JobResult | None = None
    tokens: int = 0
    delivered_at: float | None = None
    cancelled: bool = False
    settled: bool = False
    released: bool = False
    finished_at: float | None = None
    spot_job: str | None = None  # id of the spot-check duplicate of this job
    spot_of: str | None = None  # this job is the duplicate of that job
    compared: bool = False
    target_conn: Conn | None = None
    requester_conn: Conn | None = None


@dataclass(eq=False)
class Group:
    """The jobs of one routed request: their peers are of distinct families."""
    families: set = field(default_factory=set)
    nodes: set = field(default_factory=set)
    job_family: dict = field(default_factory=dict)  # job id -> family of its peer
    expires: float = 0.0


class Tracker:
    def __init__(self, db_path: str | Path = ":memory:", starter_credit: float = 1000.0, spot_rate: float = 0.05,
                 receipt_grace_s: float = 60.0, max_inflight: int = 64, sweep_s: float = 0.25,
                 job_ttl_s: float = 600.0, seed: int | None = None, wan: WanDelay | None = None,
                 ping_s: float = 10.0, ping_timeout_s: float = 5.0, suspend_base_s: float = 10.0,
                 suspend_max_s: float = 300.0, timeout_strikes: int = 2, strike_min_deadline_s: float = 10.0,
                 peers_snapshot_s: float = 3.0, landing: bool = True):
        self.landing = landing  # serve the public landing page at / (myriad/landing.py)
        self.ledger = Ledger(db_path, starter_credit)
        pem = self.ledger.get_meta("tracker_key")
        if pem:
            self.identity = Identity.from_pem(pem.encode())
        else:
            self.identity = Identity.generate()
            self.ledger.set_meta("tracker_key", self.identity.to_pem().decode())
        self.spot_rate, self.receipt_grace_s = spot_rate, receipt_grace_s
        self.max_inflight, self.sweep_s, self.job_ttl_s = max_inflight, sweep_s, job_ttl_s
        self.ping_s, self.ping_timeout_s = ping_s, ping_timeout_s
        self.suspend_base_s, self.suspend_max_s = suspend_base_s, suspend_max_s
        self.timeout_strikes, self.strike_min_deadline_s = max(1, int(timeout_strikes)), strike_min_deadline_s
        self.peers_snapshot_s = peers_snapshot_s
        self.rng = random.Random(seed)
        self.conns: dict[str, Conn] = {}
        self.jobs: dict[str, TrackedJob] = {}
        self._sweeper: asyncio.Task | None = None
        # WAN emulation for experiments (off by default): see myriad/netem.py.
        self.wan = wan if wan is not None and wan.enabled else None
        self.frames: Counter = Counter()  # frames received per type ("in:<t>") and sent ("out")
        self.health_events: Counter = Counter()  # suspensions per reason, readmissions, missed pings...
        # selection indexes
        self._pools: dict[str, Pool] = {}
        self._tag_pools: dict[str, dict[str, Pool]] = {}  # tag -> model -> selectable nodes with that tag
        self._tag_order: dict[str, tuple[float, list[str]]] = {}  # tag -> (sorted at, models by reliability)
        self._tag_match: bool | None = None  # set by _route_target for a job routed by tag
        self._order: list[str] = []
        self._order_at = -1.0
        self._order_dirty = True
        self._mstats: dict[str, list[int]] = {m: [a, d] for m, (a, d) in self.ledger.model_stats().items()}
        # Spot-check counts in memory (mirrors of the ledger), for reputations computed in O(1).
        self._mspot: dict[str, list[int]] = {m: [a, d] for m, (a, d) in self.ledger.model_spot_stats().items()}
        self._spotted: set[Conn] = set()  # connected nodes with spot checks: their reputation can move
        self._rep_at = 0.0
        self.groups: dict[tuple[str, str], Group] = {}
        # housekeeping queues
        self._timers: list = []  # heap of (due, seq, kind, obj)
        self._tseq = itertools.count()
        self._deadlines: list = []  # heap of (expiry time, seq, job id)
        self._delivered: deque = deque()  # (delivered_at, job id), in time order
        self._finished: deque = deque()  # (finished_at, job id), in time order
        # /v1/peers snapshot
        self._snapshot: bytes | None = None
        self._snapshot_at = -1.0
        self._snapshot_task: asyncio.Task | None = None
        self.snapshot_stats: dict = {}
        self.app = self._make_app()

    # ---------- app ----------
    def _make_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app):
            self._sweeper = asyncio.create_task(self._sweep_loop())
            try:
                yield
            finally:
                self._sweeper.cancel()
                if self._snapshot_task is not None:
                    self._snapshot_task.cancel()
                for c in list(self.conns.values()):
                    c.closed = True

        app = FastAPI(title="myriad tracker", version=__version__, lifespan=lifespan)

        if self.wan is not None:
            @app.middleware("http")
            async def wan_delay(request, call_next):  # one delayed hop each way
                await asyncio.sleep(self.wan.sample())
                response = await call_next(request)
                await asyncio.sleep(self.wan.sample())
                return response

        @app.get("/v1/health")
        async def health():
            return {"ok": True, "protocol": PROTOCOL, "protocol_version": PROTOCOL_VERSION, "features": list(FEATURES),
                    "version": __version__, "tracker_id": self.identity.node_id, "nodes": len(self.conns),
                    "serving": sum(1 for c in self.conns.values() if c.info.model),
                    "selectable": sum(len(p) for p in self._pools.values())}

        @app.get("/v1/peers")
        async def peers():
            return Response(await self.peers_snapshot(), media_type="application/json")

        @app.get("/v1/select")
        async def select(k: int = Query(4, ge=1, le=MAX_SELECT_K), model: str | None = Query(None, max_length=200),
                         exclude: str = Query("", max_length=64 * 33), tag: str | None = Query(None, max_length=32)):
            ids = {x for x in exclude.split(",") if x}
            if len(ids) > 64 or any(len(x) != 32 or any(ch not in "0123456789abcdef" for ch in x) for x in ids):
                raise HTTPException(422, "exclude : jusqu'à 64 identifiants de nœud séparés par des virgules")
            chosen = self.select(k, model=model or None, exclude=ids, tag=tag or None)
            return {"peers": [self._card(c, with_tags=True).model_dump(mode="json") for c in chosen]}

        @app.get("/v1/reliability")
        async def reliability():
            return {"models": self.reliability()}

        @app.get("/v1/balances")
        async def balances():
            return {"balances": self.ledger.balances()}

        @app.get("/v1/balance/{node_id}")
        async def balance(node_id: str):
            b = self.ledger.balance(node_id)
            if b is None:
                raise HTTPException(404, "compte inconnu")
            return {"node_id": node_id, "balance": b}

        @app.get("/v1/ledger")
        async def ledger(limit: int = 50):
            return {"entries": self.ledger.recent(max(1, min(limit, 500)))}

        @app.get("/v1/evidence/{job_id}")
        async def evidence(job_id: str):
            e = self.ledger.evidence(job_id)
            if e is None:
                raise HTTPException(404, "aucun règlement pour ce job")
            return e

        @app.websocket("/v1/ws")
        async def ws_endpoint(ws: WebSocket):
            await self._serve_ws(ws)

        from .stats import install as install_stats
        install_stats(app, self)  # GET /v1/stats (read-only dashboard figures)
        if self.landing:
            from .landing import install as install_landing
            install_landing(app)  # GET / and /static/landing/* (static page, never under /v1/)
        return app

    def _peer_view(self, c: Conn) -> dict:
        """One peer as essaim/1 published it, reputation recomputed from the ledger (two queries).
        Kept for diagnostics; the directory itself is served from the snapshot (`peers_snapshot`)."""
        n_agree, n_dis = self.ledger.node_spot(c.node_id, c.info.model)
        m_agree, m_dis = self.ledger.model_spot_stats().get(c.info.model, (0, 0))
        # Baseline from the OTHER comparisons on this model (a cheater does not raise its own bar);
        # both counts are for the model this node serves now.
        alpha = spot_baseline(max(0, m_agree - n_agree), max(0, m_dis - n_dis))
        rep = reputation(n_agree, n_dis, alpha)
        return {**c.info.model_dump(mode="json"), "busy": c.busy, "reputation": round(rep, 4),
                "connected_s": round(time.time() - c.connected_at, 1)}

    @staticmethod
    def _snapshot_view(c: Conn, wall: float) -> dict:
        return {**c.info.model_dump(mode="json"), "busy": c.busy, "reputation": round(c.reputation, 4),
                "connected_s": round(wall - c.connected_at, 1), "suspended": c.suspended_until is not None}

    async def peers_snapshot(self) -> bytes:
        """The /v1/peers body. Served from a snapshot at most `peers_snapshot_s` old (plus one rebuild):
        a stale snapshot is returned at once while a new one is built in the background, in chunks,
        so the event loop is never blocked for O(N)."""
        now = time.monotonic()
        if self._snapshot is None:
            await self._rebuild_snapshot()
        elif now - self._snapshot_at > self.peers_snapshot_s and (self._snapshot_task is None or self._snapshot_task.done()):
            self._snapshot_task = asyncio.create_task(self._rebuild_snapshot())
        return self._snapshot

    async def _rebuild_snapshot(self) -> None:
        started = time.monotonic()
        t = time.perf_counter()
        conns = [c for c in self.conns.values() if c.info.model]
        parts: list[str] = []
        work = longest = 0.0
        for lo in range(0, len(conns), SNAPSHOT_CHUNK):
            if lo:
                dt = time.perf_counter() - t
                work, longest = work + dt, max(longest, dt)
                await asyncio.sleep(0)  # let the relay run between chunks
                t = time.perf_counter()
            wall = time.time()
            views = [self._snapshot_view(c, wall) for c in conns[lo:lo + SNAPSHOT_CHUNK] if not c.closed]
            if views:
                parts.append(json.dumps(views, ensure_ascii=False, allow_nan=False)[1:-1])
        self._snapshot = ('{"peers":[' + ",".join(parts) + "]}").encode("utf-8")
        self._snapshot_at = started
        dt = time.perf_counter() - t
        # work: time spent building; max_chunk: longest stretch the event loop was not given back
        self.snapshot_stats = {"peers": len(conns), "work_ms": (work + dt) * 1000,
                               "max_chunk_ms": max(longest, dt) * 1000}

    def reliability(self) -> dict[str, dict]:
        stats = self.ledger.model_stats()
        spots = self.ledger.model_spot_stats()
        models = set(stats) | {c.info.model for c in self.conns.values() if c.info.model}
        out = {}
        for m in sorted(models):
            agree, disagree = stats.get(m, (0, 0))
            prior = prior_accuracy(m)
            out[m] = {"family": family_of(m), "prior": prior, "agree": agree, "disagree": disagree,
                      "p": round(beta_mean(prior, agree, disagree), 6),
                      "spot_checks": sum(spots.get(m, (0, 0))),
                      "spot_alpha": round(spot_baseline(*spots.get(m, (0, 0))), 6)}
        return out

    def model_p(self, model: str) -> float:
        """Published reliability of a model (same value as /v1/reliability), from in-memory counts."""
        a, d = self._mstats.get(model, (0, 0))
        return beta_mean(prior_accuracy(model), a, d)

    # ---------- selection (essaim/1.1) ----------
    def _tier(self, c: Conn) -> str | None:
        if (c.closed or not c.info.model or not c.info.accepting or c.suspended_until is not None
                or c.reputation < MIN_REPUTATION or self.conns.get(c.node_id) is not c):
            return None
        cap = max(1, c.info.max_parallel)
        if c.busy >= cap:
            return None
        return "idle" if c.busy == 0 else "partial"

    def _reindex(self, c: Conn) -> None:
        """Move a connection to the index set matching its state, in O(1)."""
        tier = self._tier(c)
        new = (c.info.model, tier) if tier else None
        if new == c.slot:
            return
        if c.slot is not None:
            pool = self._pools.get(c.slot[0])
            if pool is not None:
                getattr(pool, c.slot[1]).discard(c)
                if not len(pool):
                    del self._pools[pool.model]
                    self._order_dirty = True
        if new is not None:
            pool = self._pools.get(new[0])
            if pool is None:
                pool = self._pools[new[0]] = Pool(new[0], c.family)
                self._order_dirty = True
            getattr(pool, new[1]).add(c)
        for tag in c.tags:  # the same move in the node's tag pools: O(number of tags)
            pools = self._tag_pools.get(tag)
            if c.slot is not None and pools is not None:
                tp = pools.get(c.slot[0])
                if tp is not None:
                    getattr(tp, c.slot[1]).discard(c)
                    if not len(tp):
                        del pools[tp.model]
                        self._tag_order.pop(tag, None)
                if not pools:
                    del self._tag_pools[tag]
            if new is not None:
                pools = self._tag_pools.setdefault(tag, {})
                tp = pools.get(new[0])
                if tp is None:
                    tp = pools[new[0]] = Pool(new[0], c.family)
                    self._tag_order.pop(tag, None)
                getattr(tp, new[1]).add(c)
        c.slot = new

    def _ordered_models(self) -> list[str]:
        """Models with selectable nodes, most reliable first (re-sorted when the set of models changes,
        and at least every second as reliabilities move)."""
        now = time.monotonic()
        if self._order_dirty or now - self._order_at > 1.0:
            self._order = sorted(self._pools, key=lambda m: (-self.model_p(m), m))
            self._order_dirty, self._order_at = False, now
        return self._order

    def _ordered_tag_models(self, tag: str) -> list[str]:
        """Models with selectable nodes advertising `tag`, most reliable first (re-sorted when that set
        changes, and at least every second, like `_ordered_models`)."""
        pools = self._tag_pools.get(tag)
        if not pools:  # nothing cached for a tag no selectable node has (any string can be asked for)
            self._tag_order.pop(tag, None)
            return []
        now = time.monotonic()
        hit = self._tag_order.get(tag)
        if hit is None or now - hit[0] > 1.0:
            hit = self._tag_order[tag] = (now, sorted(pools, key=lambda m: (-self.model_p(m), m)))
        return hit[1]

    @staticmethod
    def _load_key(c: Conn) -> tuple:
        return (c.busy / max(1, c.info.max_parallel), c.strikes)

    def _pick(self, pool: Pool, exclude) -> Conn | None:
        """Best of two random draws (lower load, then fewer strikes), idle nodes first; O(|exclude|)."""
        for tier in (pool.idle, pool.partial):
            n = len(tier)
            if not n:
                continue
            best, seen = None, 0
            for _ in range(2 * (len(exclude) + 2)):
                c = tier.items[self.rng.randrange(n)]
                if c.node_id in exclude or c.closed:
                    continue
                seen += 1
                if best is None or self._load_key(c) < self._load_key(best):
                    best = c
                if seen >= 2:
                    break
            if best is None:  # unlucky draws: scan |exclude| + 1 consecutive items from a random start
                start = self.rng.randrange(n)
                for i in range(min(n, len(exclude) + 2)):
                    c = tier.items[(start + i) % n]
                    if c.node_id not in exclude and not c.closed:
                        best = c
                        break
            if best is not None:
                return best
        return None

    def select(self, k: int, model: str | None = None, exclude=(), exclude_families=(),
               only_family: str | None = None, tag: str | None = None) -> list[Conn]:
        """Up to k selectable nodes of distinct families, most reliable model of each family first;
        with `tag`, only nodes advertising it. Cost: O(models walked + k * |exclude|), independent of
        the number of nodes."""
        exclude = set(exclude)
        pools = self._pools if tag is None else self._tag_pools.get(tag, {})
        if model is not None:
            pool = pools.get(model)
            if pool is not None and only_family is not None and pool.family != only_family:
                pool = None
            c = self._pick(pool, exclude) if pool is not None else None
            return [c] if c is not None and k >= 1 else []
        out: list[Conn] = []
        fams = set(exclude_families)
        for m in (self._ordered_models() if tag is None else self._ordered_tag_models(tag)):
            pool = pools.get(m)
            if pool is None or pool.family in fams or (only_family is not None and pool.family != only_family):
                continue
            c = self._pick(pool, exclude)
            if c is None:
                continue
            out.append(c)
            fams.add(pool.family)
            if len(out) >= k:
                break
        return out

    def _card(self, c: Conn, with_tags: bool = False) -> PeerCard:
        return PeerCard(node_id=c.node_id, pubkey=c.info.pubkey, model=c.info.model, family=c.info.family,
                        gguf=c.info.gguf, params_b=c.info.params_b, reliability=round(self.model_p(c.info.model), 6),
                        tags=list(c.tags) if with_tags else None)

    def _route_target(self, req: Conn, job: Job, route: Route) -> Conn | None:
        """Pick the peer of a routed job: a family not yet used by its group (for a replacement, else
        the replaced job's family), never a node already used by the group or excluded.
        route.family: only that family (a replacement stays in it). route.tag: a node advertising the
        tag first, else any node; `self._tag_match` tells which."""
        key = (req.node_id, route.group)
        g = self.groups.get(key)
        if g is None:
            g = self.groups[key] = Group()
            self._schedule(time.monotonic() + job.deadline_ms / 1000 + GROUP_GRACE_S, "group", key)
        if len(g.job_family) >= MAX_GROUP_JOBS:
            return None
        excl = g.nodes | set(route.exclude)
        tag = route.tag
        self._tag_match = None
        if route.family:  # strict family; within it, the tag is a preference
            chosen = self.select(1, model=route.model, exclude=excl, only_family=route.family, tag=tag) if tag else []
            if not chosen:
                chosen = self.select(1, model=route.model, exclude=excl, only_family=route.family)
        elif route.model:
            chosen = self.select(1, model=route.model, exclude=excl, tag=tag) if tag else []
            if not chosen:
                chosen = self.select(1, model=route.model, exclude=excl)
        else:
            chosen = self.select(1, exclude=excl, exclude_families=g.families, tag=tag) if tag else []
            if not chosen:
                chosen = self.select(1, exclude=excl, exclude_families=g.families)
            fam = g.job_family.get(route.replaces) if route.replaces else None
            if not chosen and fam is not None:
                chosen = self.select(1, exclude=excl, only_family=fam, tag=tag) if tag else []
                if not chosen:
                    chosen = self.select(1, exclude=excl, only_family=fam)
        if not chosen:
            return None
        if tag:
            self._tag_match = tag in chosen[0].tags
        c = chosen[0]
        g.nodes.add(c.node_id)
        g.families.add(c.family)
        g.job_family[job.job_id] = c.family
        g.expires = max(g.expires, time.monotonic() + job.deadline_ms / 1000 + GROUP_GRACE_S)
        return c

    # ---------- health (essaim/1.1) ----------
    def _schedule(self, due: float, kind: str, obj) -> None:
        heapq.heappush(self._timers, (due, next(self._tseq), kind, obj))

    def _schedule_ping(self, c: Conn, delay: float | None = None) -> None:
        now = time.monotonic()
        if delay is None:
            delay = self.ping_s * self.rng.uniform(0.9, 1.1)
        due = now + delay
        if c.suspended_until is not None:  # a suspended node is probed when its suspension ends
            due = max(c.suspended_until, now)
        c.ping_due = due
        self._schedule(due, "ping", c)

    def _current(self, c: Conn) -> bool:
        return not c.closed and self.conns.get(c.node_id) is c

    def _on_timer(self, kind: str, obj, now: float) -> None:
        if kind == "ping":
            c = obj
            if not self._current(c) or c.ping_due is None or now < c.ping_due or c.ping_sent is not None:
                return  # gone, or a superseded schedule
            c.ping_due = None
            c.ping_seq += 1
            c.ping_sent = now
            c.send(Ping(seq=c.ping_seq))
            self._schedule(now + self.ping_timeout_s, "ping_check", (c, c.ping_seq))
        elif kind == "ping_check":
            c, seq = obj
            if not self._current(c) or c.ping_sent is None or c.ping_seq != seq:
                return
            c.ping_sent = None
            if c.pings:
                self.health_events["ping_missed"] += 1
                self._suspend(c, "ping_missed")
                self._schedule_ping(c)
            else:  # never answered a ping: an essaim/1 node, judged on its jobs only
                c.pings = False
                if c.suspended_until is not None:
                    self._schedule(c.suspended_until, "readmit", c)
        elif kind == "readmit":
            c = obj
            if self._current(c) and not c.pings and c.suspended_until is not None and now >= c.suspended_until:
                self._readmit(c)
        elif kind == "group":
            g = self.groups.get(obj)
            if g is None:
                return
            if g.expires > now:
                self._schedule(g.expires, "group", obj)
            else:
                del self.groups[obj]

    def _strike(self, c: Conn, reason: str) -> None:
        if not self._current(c):
            return
        c.strikes += 1
        self.health_events[f"strike_{reason}"] += 1
        if c.strikes >= self.timeout_strikes and c.suspended_until is None:
            self._suspend(c, reason)

    def _suspend(self, c: Conn, reason: str) -> None:
        now = time.monotonic()
        dur = min(self.suspend_max_s, self.suspend_base_s * (2 ** c.level))
        c.level = min(c.level + 1, 30)
        c.suspended_until = max(c.suspended_until or 0.0, now + dur)
        c.suspend_reason = reason
        c.strikes = self.timeout_strikes - 1  # on probation once readmitted: one more strike suspends it again
        self.health_events[f"suspend_{reason}"] += 1
        self._reindex(c)
        if c.pings:
            if c.ping_sent is None:
                self._schedule_ping(c)
        else:
            self._schedule(c.suspended_until, "readmit", c)
        log.info("node %s suspended %.0f s (%s)", c.node_id[:8], dur, reason)

    def _readmit(self, c: Conn) -> None:
        c.suspended_until, c.suspend_reason = None, None
        self.health_events["readmitted"] += 1
        self._reindex(c)

    def _on_pong(self, c: Conn, f: Pong) -> None:
        if c.ping_sent is None or f.seq != c.ping_seq:
            return  # late (already counted as missed) or unsolicited
        now = time.monotonic()
        c.rtt_ms = (now - c.ping_sent) * 1000
        c.ping_sent = None
        c.pings = True
        c.engine_ok = f.engine_ok
        if c.info.accepting != f.accepting or c.info.max_parallel != f.max_parallel:
            c.info = c.info.model_copy(update={"accepting": f.accepting, "max_parallel": f.max_parallel})
        if not f.engine_ok:
            self.health_events["engine_down"] += 1
            if c.suspended_until is None or now >= c.suspended_until:  # (still) down: suspended, longer each time
                c.suspended_until = None
                self._suspend(c, "engine_down")
        elif c.suspended_until is not None and now >= c.suspended_until:
            self._readmit(c)
        self._reindex(c)
        self._schedule_ping(c)

    def _timeout_strike(self, tj: TrackedJob) -> None:
        if tj.job.deadline_ms / 1000 >= self.strike_min_deadline_s and tj.target_conn is not None:
            self._strike(tj.target_conn, "timeout")

    # ---------- websocket ----------
    async def _serve_ws(self, ws: WebSocket) -> None:
        await ws.accept()
        nonce = secrets.token_hex(32)
        await ws.send_text(dump_frame(Challenge(nonce=nonce, tracker_id=self.identity.node_id,
                                                tracker_pubkey=self.identity.pubkey)))
        try:
            raw = await asyncio.wait_for(self._receive(ws), HELLO_TIMEOUT_S)
            hello = parse_frame(raw) if raw is not None else None
        except (asyncio.TimeoutError, ValueError, ValidationError, WebSocketDisconnect):
            hello = None
        reason = self._check_hello(hello, nonce)
        if reason:
            try:
                await ws.send_text(dump_frame(ErrorFrame(error=reason)))
                await ws.close(code=1008)
            except Exception:
                pass
            return
        info = hello.info
        old = self.conns.get(info.node_id)
        if old is not None:  # the same node reconnected: the new session wins
            self._drop(old)
        self.ledger.ensure_account(info.node_id, info.pubkey)
        conn = Conn(node_id=info.node_id, info=info, ws=ws, wan=self.wan)
        if info.model:
            conn.spot = list(self.ledger.node_spot(info.node_id, info.model))
            self._refresh_reputation(conn)
            if sum(conn.spot):
                self._spotted.add(conn)
        self.conns[info.node_id] = conn
        writer = asyncio.create_task(self._writer(conn))
        delayed = None
        if self.wan is not None:
            conn.inbox = asyncio.Queue(OUTBOX)
            delayed = asyncio.create_task(self._delayed_dispatch(conn))
        conn.send(Welcome(node_id=info.node_id, balance=self.ledger.balance(info.node_id) or 0.0))
        if info.model:
            self._reindex(conn)
            self._schedule_ping(conn, self.rng.uniform(0.2, 1.0))  # the first ping tells an essaim/1.1 node
        log.info("node %s connected (%s)", info.node_id[:8], info.model or "client")
        try:
            while not conn.closed:
                raw = await self._receive(ws)
                if raw is None:
                    break
                try:
                    frame = parse_frame(raw)
                except (ValueError, ValidationError) as e:
                    conn.send(ErrorFrame(error=f"invalid_frame: {str(e)[:200]}"))
                    continue
                if delayed is None:
                    self._dispatch(conn, frame)
                else:
                    try:
                        conn.inbox.put_nowait((conn.in_due(), frame))
                    except asyncio.QueueFull:
                        conn.closed = True
        except WebSocketDisconnect:
            pass
        finally:
            if delayed is not None:
                # The disconnection reaches the tracker after the frames sent before it.
                try:
                    conn.inbox.put_nowait((conn.in_due(), None))
                    await delayed
                except asyncio.QueueFull:
                    delayed.cancel()
                except asyncio.CancelledError:
                    delayed.cancel()
                    writer.cancel()
                    self._drop(conn)
                    raise
            writer.cancel()
            self._drop(conn)
            log.info("node %s disconnected", info.node_id[:8])

    async def _delayed_dispatch(self, conn: Conn) -> None:
        """WAN emulation: hand each received frame to the tracker at its delivery time, in order."""
        while True:
            due, frame = await conn.inbox.get()
            wait = due - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            if frame is None:
                return
            if self.conns.get(conn.node_id) is not conn:
                # Replaced by a newer session of the same node: as on the direct path, which stops
                # reading an old session, its frames are not acted upon any more.
                continue
            try:
                self._dispatch(conn, frame)
            except Exception:  # keep the connection's queue moving whatever one frame does
                log.exception("dispatch failed")

    @staticmethod
    async def _receive(ws: WebSocket) -> str | None:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            return None
        data = msg.get("text")
        if data is None and msg.get("bytes") is not None:
            data = msg["bytes"].decode("utf-8", errors="replace")
        if data is None:
            return None
        if len(data) > MAX_FRAME_BYTES:
            raise WebSocketDisconnect(code=1009)
        return data

    def _check_hello(self, hello, nonce: str) -> str | None:
        if not isinstance(hello, Hello):
            return "hello_expected"
        if hello.nonce != nonce:
            return "bad_nonce"
        if not pubkey_matches(hello.info.node_id, hello.info.pubkey):
            return "node_id_mismatch"
        if not hello.valid():
            return "bad_signature"
        if hello.info.node_id == self.identity.node_id:
            return "reserved_id"
        return None

    async def _writer(self, conn: Conn) -> None:
        try:
            while True:
                due, text = await conn.outbox.get()
                if due:
                    wait = due - time.monotonic()
                    if wait > 0:
                        await asyncio.sleep(wait)
                await conn.ws.send_text(text)
                self.frames["out"] += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            conn.closed = True
            try:
                await conn.ws.close()
            except Exception:
                pass

    def _drop(self, conn: Conn) -> None:
        """Forget a connection: jobs it was serving fail, jobs it requested are cancelled."""
        conn.closed = True
        self._reindex(conn)  # out of the selection indexes, whichever session it is
        self._spotted.discard(conn)
        if self.conns.get(conn.node_id) is conn:
            del self.conns[conn.node_id]
        else:
            return  # an older session already replaced: its jobs belong to the node, kept
        for jid in list(conn.jobs_in):
            tj = self.jobs.get(jid)
            if tj is not None and tj.status == "pending":
                self._fail(tj, "peer_disconnected")
        for jid in list(conn.jobs_out):
            tj = self.jobs.get(jid)
            if tj is not None and tj.status == "pending":
                self._cancel(tj)
        try:
            asyncio.get_running_loop().create_task(_close_quietly(conn.ws))
        except RuntimeError:  # no running loop (tests driving the tracker directly)
            pass

    def _dispatch(self, conn: Conn, frame) -> None:
        self.frames[f"in:{frame.t}"] += 1
        if isinstance(frame, Status):
            conn.info = conn.info.model_copy(update={"accepting": frame.accepting, "max_parallel": frame.max_parallel})
            self._reindex(conn)
        elif isinstance(frame, JobFrame):
            self._submit(conn, frame)
        elif isinstance(frame, ResultFrame):
            self._on_result(conn, frame.result)
        elif isinstance(frame, JobError):
            self._on_job_error(conn, frame)
        elif isinstance(frame, Cancel):
            tj = self.jobs.get(frame.job_id)
            if tj is not None and tj.requester == conn.node_id and tj.status == "pending":
                self._cancel(tj)
        elif isinstance(frame, ReceiptFrame):
            self._on_receipt(conn, frame)
        elif isinstance(frame, Pong):
            self._on_pong(conn, frame)
        else:
            conn.send(ErrorFrame(error=f"unexpected_frame: {frame.t}"))

    # ---------- jobs ----------
    def _release(self, tj: TrackedJob) -> None:
        if tj.released:
            return
        tj.released = True
        tj.finished_at = time.monotonic()
        self._finished.append((tj.finished_at, tj.job.job_id))
        t = tj.target_conn
        if t is not None:
            t.jobs_in.discard(tj.job.job_id)
            if t.busy > 0:
                t.busy -= 1
            self._reindex(t)
        r = tj.requester_conn
        if r is not None:
            r.jobs_out.discard(tj.job.job_id)
            if r.submitted > 0:
                r.submitted -= 1

    def _submit(self, req: Conn, f: JobFrame) -> None:
        job = f.job

        def reject(err: str) -> None:
            req.send(JobError(job_id=job.job_id, node_id=f.target, error=err))

        if f.target is None and f.route is None:
            return reject("no_target")
        if job.requester_id != req.node_id:
            return reject("requester_mismatch")
        if not job.verify(req.info.pubkey):
            return reject("bad_signature")
        if job.job_id in self.jobs or self.ledger.has_job(job.job_id):
            return reject("duplicate_job")
        if job.prompt_chars() > MAX_PROMPT_CHARS:
            return reject("prompt_too_large")
        bal = self.ledger.balance_milli(req.node_id)
        if bal is None or bal < 0:
            return reject("insufficient_credits")
        if req.submitted >= self.max_inflight:
            return reject("too_many_jobs")
        if f.target is None:  # essaim/1.1: the tracker picks the peer and tells the requester first
            tgt = self._route_target(req, job, f.route)
            if tgt is None:
                return reject("no_peer")
            tagged = f.route.tag is not None  # only a tags-aware requester sets it: it parses the new fields
            req.send(Assigned(job_id=job.job_id, peer=self._card(tgt, with_tags=tagged),
                              tag_match=self._tag_match if tagged else None))
        else:
            tgt = self.conns.get(f.target)
            if tgt is None or not tgt.info.model:
                return reject("peer_offline")
            if not tgt.info.accepting:
                return reject("peer_paused")
            if tgt.suspended_until is not None:  # e.g. an essaim/1 gateway with an older directory
                return reject("peer_suspended")
            if tgt.busy >= max(1, tgt.info.max_parallel):
                return reject("peer_busy")
        tj = self._route(job, req.node_id, tgt, req.info.pubkey, req)
        req.submitted += 1
        self._maybe_spot_check(tj, req.node_id)

    def _route(self, job: Job, requester: str, tgt: Conn, requester_pubkey: str,
               req_conn: Conn | None = None) -> TrackedJob:
        params = trusted_params(tgt.info.model, tgt.info.params_b)
        tj = TrackedJob(job=job, requester=requester, target=tgt.node_id, model=tgt.info.model,
                        factor=credit_factor(params), deadline=time.monotonic() + job.deadline_ms / 1000,
                        target_conn=tgt, requester_conn=req_conn)
        self.jobs[job.job_id] = tj
        heapq.heappush(self._deadlines, (tj.deadline + 1.0, next(self._tseq), job.job_id))
        tgt.busy += 1
        tgt.jobs_in.add(job.job_id)
        if req_conn is not None:
            req_conn.jobs_out.add(job.job_id)
        self._reindex(tgt)
        tgt.send(JobFrame(job=job, requester_pubkey=requester_pubkey))
        return tj

    def _maybe_spot_check(self, tj: TrackedJob, requester: str) -> None:
        """Duplicate a small fraction of jobs on another node running the same model file. Only greedy
        jobs (temperature 0, same seed) with an extractable answer (math, multiple choice) are checked:
        comparing two SAMPLED texts would punish honest nodes more than a cheater returning the modal
        answer, since the false-positive rate is then at least the detection rate."""
        if self.spot_rate <= 0 or tj.job.temperature != 0 or self.rng.random() >= self.spot_rate:
            return
        if (tj.job.task_hint or detect_task_hint([m.model_dump() for m in tj.job.messages])) not in ("math", "mc"):
            return
        tgt = tj.target_conn
        pool = self._pools.get(tgt.info.model) if tgt is not None else None
        if pool is None:
            return
        checker = self._spot_checker(pool, tgt.info.gguf, {tj.target, requester})
        if checker is None:
            return
        dup = Job(job_id=uuid.uuid4().hex, requester_id=self.identity.node_id, messages=tj.job.messages,
                  max_tokens=tj.job.max_tokens, temperature=0.0, seed=tj.job.seed, deadline_ms=tj.job.deadline_ms,
                  task_hint=tj.job.task_hint, thinking=tj.job.thinking).signed_by(self.identity)
        spot = self._route(dup, self.identity.node_id, checker, self.identity.pubkey)
        spot.spot_of, tj.spot_job = tj.job.job_id, dup.job_id

    def _spot_checker(self, pool: Pool, gguf: str | None, exclude: set) -> Conn | None:
        """A selectable node of the pool serving the same GGUF file, idle or partly busy: random draws in
        both sets, then a bounded scan (best effort, O(1): with very few compatible nodes among many,
        a spot check may be skipped)."""
        def ok(c: Conn) -> bool:
            return c.node_id not in exclude and not c.closed and c.info.gguf == gguf

        for tier in (pool.idle, pool.partial):
            n = len(tier)
            for _ in range(min(n, 8)):
                c = tier.items[self.rng.randrange(n)]
                if ok(c):
                    return c
        for tier in (pool.idle, pool.partial):
            n = len(tier)
            start = self.rng.randrange(n) if n else 0
            for i in range(min(n, 64)):
                c = tier.items[(start + i) % n]
                if ok(c):
                    return c
        return None

    def _on_result(self, node: Conn, res: JobResult) -> None:
        tj = self.jobs.get(res.job_id)
        if tj is None or tj.target != node.node_id or res.node_id != node.node_id:
            return node.send(ErrorFrame(error=f"unknown_job: {res.job_id}"))
        if tj.status not in ("pending", "cancelled"):
            return  # late duplicate, deadline passed or already failed
        if not res.verify(node.info.pubkey):
            node.send(ErrorFrame(error=f"bad_signature: {res.job_id}"))
            return self._fail(tj, "bad_result_signature")
        if res.model != tj.model:
            return self._fail(tj, "model_mismatch")
        # Self-reported token counts are bounded by the request and by the text actually returned.
        tj.tokens = min(res.completion_tokens, tj.job.max_tokens, len(res.text.encode("utf-8")) + 16)
        tj.status, tj.result, tj.delivered_at = "delivered", res, time.monotonic()
        self._delivered.append((tj.delivered_at, res.job_id))
        node.strikes, node.level = 0, 0  # a delivered result clears the node's health record
        self._release(tj)
        if tj.spot_of is not None:  # our own spot check: paid by the network
            amount = int(round(tj.tokens * tj.factor * MILLI))
            self.ledger.settle(res.job_id, None, node.node_id, res.model, tj.tokens, amount, "spot",
                               result=res.model_dump(mode="json"))
            tj.settled = True
        else:
            req = self.conns.get(tj.requester)
            if req is not None:
                req.send(ResultFrame(result=res))
        self._compare_spot(tj)

    def _on_job_error(self, node: Conn, f: JobError) -> None:
        tj = self.jobs.get(f.job_id)
        if tj is None or tj.target != node.node_id or tj.status != "pending":
            return
        self.ledger.record_failure(node.node_id)
        if f.error in TIMEOUT_ERRORS:
            self._timeout_strike(tj)
        self._fail(tj, f.error[:200])

    def _fail(self, tj: TrackedJob, error: str) -> None:
        tj.status = "failed"
        self._release(tj)
        req = self.conns.get(tj.requester)
        if req is not None and tj.spot_of is None:
            req.send(JobError(job_id=tj.job.job_id, node_id=tj.target, error=error))

    def _cancel(self, tj: TrackedJob) -> None:
        tj.status, tj.cancelled = "cancelled", True
        self._release(tj)
        tgt = self.conns.get(tj.target)
        if tgt is not None:
            tgt.send(Cancel(job_id=tj.job.job_id))

    def _on_receipt(self, req: Conn, f: ReceiptFrame) -> None:
        rc = f.receipt
        tj = self.jobs.get(rc.job_id)

        def bad(why: str) -> None:
            req.send(ErrorFrame(error=f"bad_receipt: {why}: {rc.job_id}"))

        if tj is None:
            return bad("unknown_job")
        if rc.requester_id != req.node_id or tj.requester != req.node_id:
            return bad("not_requester")
        if not rc.verify(req.info.pubkey):
            return bad("bad_signature")
        if tj.status != "delivered" or tj.result is None:
            return bad("no_result")
        if rc.node_id != tj.target or rc.model != tj.result.model or rc.completion_tokens != tj.result.completion_tokens:
            return bad("mismatch")
        if tj.settled:
            return bad("already_settled")
        self._settle(tj, "receipt", rc.model_dump(mode="json"))
        if rc.agreed is not None and tj.requester != tj.target:
            self.ledger.record_agreement(tj.model, rc.agreed)
            self._mstats.setdefault(tj.model, [0, 0])[0 if rc.agreed else 1] += 1

    def _settle(self, tj: TrackedJob, kind: str, receipt: dict | None) -> None:
        amount = int(round(tj.tokens * tj.factor * MILLI))
        self.ledger.settle(tj.job.job_id, tj.requester, tj.target, tj.model, tj.tokens, amount, kind,
                           receipt=receipt, result=tj.result.model_dump(mode="json"))
        tj.settled = True

    def _refresh_reputation(self, c: Conn) -> None:
        """Same formula as `_peer_view` (baseline from the OTHER comparisons on the model), from the
        in-memory counts: O(1)."""
        n_agree, n_dis = c.spot
        m_agree, m_dis = self._mspot.get(c.info.model, (0, 0))
        c.reputation = reputation(n_agree, n_dis, spot_baseline(max(0, m_agree - n_agree), max(0, m_dis - n_dis)))
        self._reindex(c)

    def _compare_spot(self, tj: TrackedJob) -> None:
        orig = self.jobs.get(tj.spot_of) if tj.spot_of else tj
        spot = self.jobs.get(orig.spot_job) if orig is not None and orig.spot_job else None
        if orig is None or spot is None or orig.compared or orig.result is None or spot.result is None:
            return
        orig.compared = True
        agree = spot_verdict(orig.result.text, spot.result.text, orig.job.task_hint,
                             [m.model_dump() for m in orig.job.messages])
        if agree is None:  # no answer to compare (cut short...): inconclusive, nobody is blamed
            return
        self.ledger.record_model_spot(orig.model, agree)
        self.ledger.record_spot(orig.target, orig.model, agree)
        self.ledger.record_spot(spot.target, spot.model, agree)
        self._mspot.setdefault(orig.model, [0, 0])[0 if agree else 1] += 1
        for x in (orig, spot):  # the reputation used by the selection follows the new counts
            c = self.conns.get(x.target)
            if c is not None and c.info.model == x.model:
                c.spot[0 if agree else 1] += 1
                self._spotted.add(c)
                self._refresh_reputation(c)
        # The model's baseline moved for its other nodes too: they are refreshed by the sweep.

    # ---------- housekeeping ----------
    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.sweep_s)
            try:
                self.sweep()
            except Exception:  # never let housekeeping die
                log.exception("sweep failed")

    def sweep(self) -> None:
        """Due work only (heaps and FIFO queues), plus a scan of the connections for closed ones."""
        now = time.monotonic()
        for c in list(self.conns.values()):
            if c.closed:
                self._drop(c)
        while self._timers and self._timers[0][0] <= now:
            _, _, kind, obj = heapq.heappop(self._timers)
            self._on_timer(kind, obj, now)
        if now - self._rep_at >= REPUTATION_REFRESH_S:
            # Spot checks move each model's baseline: the reputations of every node with spot checks
            # (selectable or excluded) follow it, in O(1) each, a node excluded below 0.3 included.
            self._rep_at = now
            for c in list(self._spotted):
                if self._current(c):
                    self._refresh_reputation(c)
                else:
                    self._spotted.discard(c)
        while self._deadlines and self._deadlines[0][0] < now:
            _, _, jid = heapq.heappop(self._deadlines)
            tj = self.jobs.get(jid)
            if tj is None or tj.status != "pending":
                continue
            tj.status = "expired"
            self._release(tj)
            self._timeout_strike(tj)
            req = self.conns.get(tj.requester)
            if req is not None and tj.spot_of is None:
                req.send(JobError(job_id=jid, node_id=tj.target, error="deadline"))
            tgt = self.conns.get(tj.target)
            if tgt is not None:
                tgt.send(Cancel(job_id=jid))
        while self._delivered and now - self._delivered[0][0] > self.receipt_grace_s:
            _, jid = self._delivered.popleft()
            tj = self.jobs.get(jid)
            if tj is not None and tj.status == "delivered" and not tj.settled and not tj.cancelled:
                # The requester got the signed result but sent no receipt: the node is paid anyway.
                self._settle(tj, "auto", None)
        while self._finished and now - self._finished[0][0] > self.job_ttl_s:
            _, jid = self._finished.popleft()
            tj = self.jobs.get(jid)
            if tj is None:
                continue
            if tj.settled or tj.cancelled or tj.status != "delivered":
                del self.jobs[jid]
            else:  # delivered, receipt still awaited: look again later
                self._finished.append((now, jid))


async def _close_quietly(ws) -> None:
    """Close a WebSocket that may already be closed (the peer left first): nothing to report then."""
    try:
        await ws.close()
    except Exception:
        pass


def spot_verdict(a: str, b: str, hint: str | None, messages: list[dict]) -> bool | None:
    """Two greedy answers to the same job: do their EXTRACTED answers agree? None if either has none."""
    hint = hint or detect_task_hint(messages)
    ea, eb = extract_answer(a, hint), extract_answer(b, hint)
    if ea is None or eb is None:
        return None
    return ea == eb


# Beta(1, 19) prior on the honest disagreement rate of two greedy runs of the same model file (mean 5 %):
# llama.cpp is not bit-exact across hardware and batch sizes, so honest nodes do disagree sometimes.
SPOT_PRIOR = (1.0, 19.0)


def spot_baseline(agree: int, disagree: int) -> float:
    """Measured disagreement rate alpha between honest runs of a model (posterior mean)."""
    a, b = SPOT_PRIOR
    return (disagree + a) / (agree + disagree + a + b)


def reputation(agree: int, disagree: int, alpha: float) -> float:
    """1.0 unless the node disagrees well above the baseline alpha (more than 3 standard deviations
    plus one); then it falls with the excess rate, down to 0 for a node that always disagrees."""
    n = agree + disagree
    if n == 0:
        return 1.0
    alpha = min(max(alpha, 1e-6), 1 - 1e-6)
    if disagree <= n * alpha + 3 * math.sqrt(n * alpha * (1 - alpha)) + 1:
        return 1.0
    return max(0.0, 1.0 - (disagree / n - alpha) / (1 - alpha))


def run(host: str = "127.0.0.1", port: int = 8500, db: str | Path = "tracker.sqlite", **kw) -> None:
    import uvicorn

    tracker = Tracker(db_path=db, **kw)
    if tracker.wan is not None:
        log.warning("WAN emulation ON: %s (experiments only)", tracker.wan.describe())
    uvicorn.run(tracker.app, host=host, port=port, ws_max_size=MAX_FRAME_BYTES, log_level="info")
