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

essaim/1.3 (features "e2e", "policy", "report"; see protocol.py, e2e.py and docs/08_securite.md):
- End-to-end encryption. The tracker relays SealedJob / SealedResult envelopes it cannot read: it checks
  the signatures (a one-job pseudonym's for the job, the peer's for the result), the sizes and the
  credits, and settles on the cleartext header (token limit and count). It charges the authenticated
  connection that sent the job; the peer only sees the pseudonym. A routed encrypted job starts with a
  Reserve frame: the tracker picks and holds the peer, sends the Assigned frame (with the peer's key
  certificate), and relays the SealedJob that follows (RESERVE_TTL_S at most, no strike if it never comes).
- Canary jobs replace the spot checks of encrypted jobs (canary.py): after a real encrypted job, with
  probability `spot_rate`, the tracker sends its own encrypted question (header and padded size copied
  from that job, a fresh pseudonym, a random delay) to the same peer and to a peer serving the same
  model file, decrypts both answers and feeds the same reputation logic. Plaintext jobs of older
  gateways keep the former spot checks. Canaries may carry honeytokens: a sighting (POST /v1/honeytoken
  or GET /h/<token>) bans the peer that received it.
- Requester policy in the selection: Route.deny / deny_families / only / swarm / e2e / min_rel /
  soft_avoid, also accepted by POST /v1/select. Peers' serving policies (ServePolicy: refused requesters,
  per-requester rate and concurrency) are enforced here, per paying account.
- Signed reports (POST /v1/report), weighted by the reporter's account age and spending (sybil
  resistance): they only lower a node's priority, never ban it. Bans come from the admin's ban file
  and from honeytoken sightings.
- The public ledger views (/v1/ledger, /v1/evidence) no longer name the requester of a job.

App updates (feature "update", see release.py): with a release repository configured (`run()`: env
MYRIAD_RELEASE_REPO, default amintt2/myriad; empty or "off" disables it), a background task polls GitHub
for the latest release every MYRIAD_RELEASE_POLL_S seconds (600 by default) and serves it on
GET /v1/version. Nodes that connect with `?features=update` get its version in their Welcome frame and an
UpdateAvailable frame when it changes. A Tracker built directly (tests) watches nothing.
"""
from __future__ import annotations

import asyncio
import hashlib
import heapq
import itertools
import json
import logging
import math
import os
import random
import re
import secrets
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response
from pydantic import ValidationError

from . import FEATURES, PROTOCOL, PROTOCOL_VERSION, __version__
from . import canary as canaries
from .crypto import Identity, account_authorized, pubkey_matches
from .e2e import E2EError, dispute_session, kx_problem, open_result, seal_job, sealed_size
from .fusion import FORMAT_INSTRUCTIONS, detect_task_hint, extract_answer
from .ledger import Ledger
from .netem import WanDelay
from .priors import MILLI, beta_mean, credit_factor, family_of, prior_accuracy, trusted_params
from .protocol import (MAX_FRAME_BYTES, MAX_POLICY_IDS, MAX_PROMPT_CHARS, Assigned, Cancel, Challenge, Dispute, ErrorFrame,
                       Hello, Job, JobError, JobFrame, JobResult, KxFrame, NodeInfo, NodeReport, PeerCard, Ping, Pong,
                       ReceiptFrame, Reserve, ResultFrame, Route, SealedJob, SealedJobFrame, SealedResult,
                       SealedResultFrame, ServePolicy, Status, UpdateAvailable, Welcome, dump_frame, parse_frame)
from .release import DEFAULT_RELEASE_REPO, ReleaseWatcher

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
VERSION_CACHE_S = 300  # Cache-Control max-age of GET /v1/version
# optional features a node may ask for (WebSocket query `features`); "e2e": a requester that applies a peer
# policy (essaim/1.3): its plaintext jobs are never duplicated to a peer it did not choose
NODE_FEATURES = frozenset({"update", "e2e"})
RESERVE_TTL_S = 15.0  # a reserved peer waits this long for the job's content
# Bounded state (audit net #2): request groups, and jobs kept after they end (for late receipts)
MAX_GROUPS_PER_ACCOUNT = 256
MAX_GROUPS = 50_000
NEW_ACCOUNT_INFLIGHT = 8  # in-flight jobs of a fresh account (see Tracker._inflight_cap)
MAX_RETAINED_PER_ACCOUNT = 2048
MAX_JOBS = 200_000
KX_MARGIN_S = 60.0  # a key certificate expiring sooner does not count as end-to-end capable
CANARY_DELAY_S = (0.5, 8.0)  # a canary leaves this long after the real job it imitates
CANARY_RETRIES = 4  # a canary for a busy peer is postponed this many times before it is dropped
TRUNCATION_CHARS_PER_TOKEN = 1.0  # a "length" answer shorter than this per allowed token was not cut short
INCONCLUSIVE_LIMIT = 3  # cut-short audits of one peer before one counts as a disagreement
REPORT_WINDOW_S = 30 * 86400
REPORT_FLAG_SCORE = 3.0  # weighted score from which a node is deprioritised (never banned)
REPORT_MIN_REPORTERS = 3
REPORTS_PER_HOUR = 20
HONEY_RE = re.compile(r"myr_live_[0-9a-f]{32}|/h/[0-9a-f]{24}|[a-z]+\.[0-9a-f]{10}@[A-Za-z0-9.\-]+")


@dataclass
class Header:
    """A reserved job (essaim/1.3) before its content arrives: what the tracker knows of it."""
    job_id: str
    requester_id: str
    max_tokens: int
    deadline_ms: int


@dataclass(frozen=True)
class _Settled:
    """What is kept of a settled job's result: enough to answer a late duplicate receipt."""
    model: str
    completion_tokens: int


@dataclass(frozen=True)
class Policy:
    """The requester's peer policy (Route fields of essaim/1.3) plus the requester's account, for the
    peers' serving policies."""
    requester: str | None = None
    deny: frozenset = frozenset()
    deny_fams: frozenset = frozenset()
    only: frozenset | None = None
    swarm: str | None = None
    e2e: bool = False
    min_rel: float | None = None
    soft: frozenset = frozenset()

    @classmethod
    def of(cls, route: Route | None, requester: str | None) -> "Policy":
        if route is None:
            return cls(requester=requester)
        return cls(requester=requester, deny=frozenset(route.deny), deny_fams=frozenset(f.lower() for f in route.deny_families),
                   only=frozenset(route.only) if route.only is not None else None, swarm=route.swarm,
                   e2e=bool(route.e2e), min_rel=route.min_rel, soft=frozenset(route.soft_avoid))


@dataclass(eq=False)
class CanaryPair:
    """One canary question sent to two peers of the same model file."""
    params: dict
    texts: dict = field(default_factory=dict)  # job id -> decrypted answer (None: undecryptable)
    reasons: dict = field(default_factory=dict)  # job id -> finish_reason
    expected: str | None = None  # the known answer (used when there is no checker)
    jobs: list = field(default_factory=list)  # TrackedJob of each copy


async def read_limited(request: Request, limit: int) -> bytes:
    """The request body, refused (413) as soon as it exceeds `limit`: never loaded whole first."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, "requête trop grande")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, "requête trop grande")
        chunks.append(chunk)
    return b"".join(chunks)


def requested_features(raw: str | None) -> frozenset:
    """`?features=update,...` -> the known features asked for (unknown names and oversize input ignored)."""
    if not raw or len(raw) > 200:
        return frozenset()
    return frozenset(x.strip() for x in raw.split(",")) & NODE_FEATURES


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
    features: frozenset = frozenset()  # optional features this node asked for when it connected
    # essaim/1.3: the node's serving policy (ServePolicy), enforced per paying account
    serve_deny: frozenset = frozenset()
    serve_rate: int = 0
    serve_conc: int = 0
    req_calls: dict = field(default_factory=dict)  # requester -> deque of job start times (last minute)
    req_running: Counter = field(default_factory=Counter)
    reported: bool = False  # weighted reports above the flag score: drawn last

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
        """The canonical family of the model this node serves (priors.family_of), whatever the node
        declares: the same reading in the directory, the selection and the gateways (audit net #6)."""
        return family_of(self.info.model) if self.info.model else ""

    @property
    def tags(self) -> tuple:
        return tuple(self.info.tags or ())

    @property
    def swarm_groups(self) -> tuple:
        return tuple(s.g for s in (self.info.swarms or ()))

    def kx_ok(self, margin: float = KX_MARGIN_S) -> bool:
        kx = self.info.kx
        return kx is not None and kx.expires > time.time() + margin


@dataclass(eq=False)
class TrackedJob:
    job: object  # Job (plaintext), SealedJob (encrypted) or Header (reserved, content not arrived yet)
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
    reserved: bool = False  # essaim/1.3: peer held, content not arrived yet
    canary: tuple | None = None  # (CanaryPair, e2e.Session): a canary job of the tracker
    counted: bool = False  # counted in the target's per-requester running jobs
    commits: bool = False  # a requester's job: it holds credits (`committed`) until settled
    committed: int = 0
    sealed: bool = False  # an encrypted job (kept when its payload is dropped: a late result must match)


@dataclass(eq=False)
class Group:
    """The jobs of one routed request: their peers are of distinct families."""
    families: set = field(default_factory=set)
    nodes: set = field(default_factory=set)
    job_family: dict = field(default_factory=dict)  # job id -> family of its peer
    expires: float = 0.0


class Tracker:
    def __init__(self, db_path: str | Path = ":memory:", starter_credit: float = 1000.0, spot_rate: float = 0.05,
                 receipt_grace_s: float = 60.0, max_inflight: int = 64, new_account_inflight: int = NEW_ACCOUNT_INFLIGHT,
                 sweep_s: float = 0.25,
                 job_ttl_s: float = 600.0, seed: int | None = None, wan: WanDelay | None = None,
                 ping_s: float = 10.0, ping_timeout_s: float = 5.0, suspend_base_s: float = 10.0,
                 suspend_max_s: float = 300.0, timeout_strikes: int = 2, strike_min_deadline_s: float = 10.0,
                 peers_snapshot_s: float = 3.0, release_repo: str | None = None, release_poll_s: float = 600.0,
                 release_http=None, landing: bool = True, public_url: str | None = None,
                 ban_file: str | Path | None = None, canary_delay_s: tuple = CANARY_DELAY_S,
                 max_overdraft: float | None = None):
        self.landing = landing  # serve the public landing page at / (myriad/landing.py)
        self.public_url = public_url  # honeytoken URLs and e-mail domain (none without it)
        self.ban_file = Path(ban_file) if ban_file else None
        self._ban_mtime: float | None = None
        self._ban_checked = -1e9
        self.canary_delay_s = canary_delay_s
        self.canary_stats: Counter = Counter()
        self.ledger = Ledger(db_path, starter_credit)
        pem = self.ledger.get_meta("tracker_key")
        if pem:
            self.identity = Identity.from_pem(pem.encode())
        else:
            self.identity = Identity.generate()
            self.ledger.set_meta("tracker_key", self.identity.to_pem().decode())
        self.spot_rate, self.receipt_grace_s = spot_rate, receipt_grace_s
        self.max_inflight, self.sweep_s, self.job_ttl_s = max_inflight, sweep_s, job_ttl_s
        self.new_account_inflight = new_account_inflight
        # Credits committed by delivered but not yet settled results, per account, in thousandths (audit
        # net #1): a result's exact cost is committed when it is delivered and released when it is settled
        # (receipt, or automatically after receipt_grace_s). A requester is admitted only while its balance
        # is >= 0 and balance - committed >= -max_overdraft, so it can no longer recycle its in-flight quota
        # through unsettled results: what it can owe is bounded by max_overdraft plus one burst of at most
        # max_inflight jobs. (Committing each job's MAXIMUM cost at admission would refuse ordinary parallel
        # requests: k jobs of 512 tokens already exceed the starter credit.)
        self.max_overdraft = int(round((max_overdraft or 0.0) * MILLI))
        self._committed: Counter = Counter()
        self._retained: Counter = Counter()  # jobs kept in self.jobs, per requesting account
        self._groups_of: Counter = Counter()  # request groups alive, per account
        self._inconclusive: Counter = Counter()  # cut-short audits per peer (see _inconclusive_by)
        self._route_error: str | None = None
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
        # essaim/1.3
        self._swarm_members: dict[str, set] = {}  # swarm group tag -> connections claiming it
        self.banned: dict[str, str] = {}  # node id -> reason (ban file and honeytoken sightings)
        self._file_bans: dict[str, str] = {}
        self._reload_bans(force=True)
        self.report_scores: dict[str, float] = self.ledger.report_scores(time.time() - REPORT_WINDOW_S,
                                                                         REPORT_MIN_REPORTERS)
        # Latest app release (feature "update"): None when no repository is watched.
        self.releases = (ReleaseWatcher(release_repo, poll_s=release_poll_s, http=release_http,
                                        on_change=self._on_release) if release_repo else None)
        self.app = self._make_app()

    # ---------- app ----------
    def _make_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app):
            self._sweeper = asyncio.create_task(self._sweep_loop())
            if self.releases is not None:
                self.releases.start()
            try:
                yield
            finally:
                self._sweeper.cancel()
                if self.releases is not None:
                    await self.releases.stop()
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

        @app.get("/v1/version")
        async def version():
            """Latest app release known to the tracker (a hint: clients verify everything themselves)."""
            body = self.releases.public() if self.releases is not None else {"repo": None, "latest": None,
                                                                              "checked_at": None}
            return Response(json.dumps(body, ensure_ascii=False, allow_nan=False), media_type="application/json",
                            headers={"Cache-Control": f"public, max-age={VERSION_CACHE_S}"})

        @app.get("/v1/peers")
        async def peers():
            return Response(await self.peers_snapshot(), media_type="application/json")

        @app.get("/v1/select")
        async def select(k: int = Query(4, ge=1, le=MAX_SELECT_K), model: str | None = Query(None, max_length=200),
                         exclude: str = Query("", max_length=64 * 33), tag: str | None = Query(None, max_length=32),
                         swarm: str | None = Query(None, pattern=r"^[0-9a-f]{32}$"), e2e: bool = False,
                         deny_families: str = Query("", max_length=32 * 65)):
            ids = {x for x in exclude.split(",") if x}
            if len(ids) > 64 or any(len(x) != 32 or any(ch not in "0123456789abcdef" for ch in x) for x in ids):
                raise HTTPException(422, "exclude : jusqu'à 64 identifiants de nœud séparés par des virgules")
            pol = Policy(swarm=swarm, e2e=e2e, deny_fams=frozenset(x.lower() for x in deny_families.split(",") if x))
            chosen = self.select(k, model=model or None, exclude=ids, tag=tag or None, policy=pol)
            return {"peers": [self._card(c, with_tags=True, full=True).model_dump(mode="json", exclude_none=True)
                              for c in chosen]}

        @app.post("/v1/select")
        async def select_post(request: Request):
            """The same choice with the full policy of essaim/1.3 in a JSON body: {k, model, tag, exclude,
            deny, deny_families, only, swarm, e2e, min_rel} (long lists do not fit in a URL)."""
            raw = await read_limited(request, 128 * 1024)
            try:
                body = json.loads(raw)
                k = int(body.get("k", 4))
                route = Route(group="0" * 32, **{f: body[f] for f in ("model", "tag", "deny", "deny_families", "only",
                                                                     "swarm", "e2e", "min_rel") if f in body})
                exclude = Route(group="0" * 32, exclude=body.get("exclude") or []).exclude
            except (ValueError, TypeError, AttributeError, ValidationError) as e:
                raise HTTPException(422, f"requête invalide : {str(e)[:200]}") from None
            if not 1 <= k <= MAX_SELECT_K:
                raise HTTPException(422, f"k entre 1 et {MAX_SELECT_K}")
            chosen = self.select(k, model=route.model, exclude=set(exclude), tag=route.tag, policy=Policy.of(route, None))
            return {"peers": [self._card(c, with_tags=True, full=True).model_dump(mode="json", exclude_none=True)
                              for c in chosen]}

        @app.get("/v1/reliability")
        async def reliability():
            return {"models": self.reliability()}

        @app.get("/v1/balances")
        async def balances():
            # essaim/1.3: no per-account listing any more (a peer could match a debit to the job it
            # served and so learn who asked: Codex review); only totals.
            b = self.ledger.balances()
            return {"accounts": len(b), "total": round(sum(b.values()), 3)}

        @app.get("/v1/balance/{node_id}")
        async def balance(node_id: str, ts: int | None = None, sig: str | None = None):
            """An account's balance, for its owner only (query signed with the account's key)."""
            if not account_authorized(node_id, self.ledger.pubkey(node_id), ts, sig):
                raise HTTPException(403, "requête signée par le titulaire du compte attendue")
            b = self.ledger.balance(node_id)
            if b is None:
                raise HTTPException(404, "compte inconnu")
            return {"node_id": node_id, "balance": b}

        @app.get("/v1/ledger")
        async def ledger(limit: int = 50):
            # essaim/1.3: without the requester, which the peers must not learn (unlinkability)
            return {"entries": [{k: v for k, v in e.items() if k != "requester_id"}
                                for e in self.ledger.recent(max(1, min(limit, 500)))]}

        @app.get("/v1/evidence/{job_id}")
        async def evidence(job_id: str):
            """The settlement of a job and a DIGEST of the peer's signed result: never its content (audit
            net #3), nor the requester and its receipt (unlinkability). The full signed evidence stays
            in the tracker's ledger for disputes."""
            e = self.ledger.evidence(job_id)
            if e is None:
                raise HTTPException(404, "aucun règlement pour ce job")
            res = e.get("result") or {}
            out = {k: v for k, v in e.items() if k not in ("requester_id", "receipt", "result")}
            out["result_sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True, ensure_ascii=False)
                                                  .encode("utf-8")).hexdigest() if res else None
            out["result_signature"] = res.get("signature")
            out["encrypted"] = "chunks" in res
            return out

        @app.post("/v1/report")
        async def report(request: Request):
            raw = await read_limited(request, 8192)
            try:
                rep = NodeReport.model_validate_json(raw)
            except ValidationError as e:
                raise HTTPException(422, f"signalement invalide : {str(e)[:200]}") from None
            ok, why, score = self.accept_report(rep)
            if not ok:
                raise HTTPException(400 if why != "rate_limited" else 429, why)
            return {"ok": True, "weight": round(score, 3)}

        @app.post("/v1/honeytoken")
        async def honeytoken(request: Request):
            """Report text where a honeytoken might appear (a leak seen elsewhere): the peer that received
            it is identified and banned. Answers only how many tokens were recognised."""
            raw = await read_limited(request, 65536)
            try:
                text = str(json.loads(raw).get("text", ""))
            except (ValueError, AttributeError):
                raise HTTPException(422, "JSON {\"text\": ...} attendu") from None
            return {"found": self.honeytoken_sighting(text, "report")}

        @app.get("/h/{token}")
        async def honey_url(token: str):
            if re.fullmatch(r"[0-9a-f]{24}", token):
                self.honeytoken_sighting(f"/h/{token}", "url")
            return PlainTextResponse("Not Found", status_code=404)

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
        info = c.info.model_dump(mode="json")
        for f in c.info.unset():  # as an older tracker published it when the node has no such field
            info.pop(f, None)
        if c.family:
            info["family"] = c.family  # canonical, as the selection reads it (audit net #6)
        return {**info, "busy": c.busy, "reputation": round(c.reputation, 4),
                "connected_s": round(wall - c.connected_at, 1), "suspended": c.suspended_until is not None,
                **({"reported": True} if c.reported else {})}

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
                or c.reputation < MIN_REPUTATION or self.conns.get(c.node_id) is not c or c.node_id in self.banned):
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
        return (c.reported, c.busy / max(1, c.info.max_parallel), c.strikes)

    def _serves(self, c: Conn, requester: str | None) -> bool:
        """The node's serving policy (ServePolicy) allows one more job from this paying account now."""
        if requester is None:
            return True
        if requester in c.serve_deny:
            return False
        if c.serve_conc and c.req_running.get(requester, 0) >= c.serve_conc:
            return False
        if c.serve_rate:
            q = c.req_calls.get(requester)
            if q:
                now = time.monotonic()
                while q and now - q[0] > 60.0:
                    q.popleft()
                if len(q) >= c.serve_rate:
                    return False
        return True

    def _eligible(self, c: Conn, pol: Policy) -> bool:
        if c.node_id in pol.deny or c.node_id in self.banned:
            return False
        if pol.e2e and not c.kx_ok():
            return False
        if pol.swarm is not None and pol.swarm not in c.swarm_groups:
            return False
        if pol.only is not None and c.node_id not in pol.only:
            return False
        if pol.deny_fams and (c.family or "").lower() in pol.deny_fams:
            return False
        return self._serves(c, pol.requester)

    def _pick(self, pool: Pool, exclude, pred=None) -> Conn | None:
        """Best of two random draws (lower load, then fewer strikes), idle nodes first; O(|exclude|).
        `pred`: an extra condition (requester policy); with it, the fallback scan looks further."""
        extra = 64 if pred is not None else 0
        for tier in (pool.idle, pool.partial):
            n = len(tier)
            if not n:
                continue
            best, seen = None, 0
            for _ in range(2 * (len(exclude) + 2) + extra):
                c = tier.items[self.rng.randrange(n)]
                if c.node_id in exclude or c.closed or (pred is not None and not pred(c)):
                    continue
                seen += 1
                if best is None or self._load_key(c) < self._load_key(best):
                    best = c
                if seen >= 2:
                    break
            if best is None:  # unlucky draws: scan |exclude| + 1 consecutive items from a random start
                start = self.rng.randrange(n)
                for i in range(min(n, len(exclude) + 2 + extra)):
                    c = tier.items[(start + i) % n]
                    if c.node_id not in exclude and not c.closed and (pred is None or pred(c)):
                        best = c
                        break
            if best is not None:
                return best
        return None

    def select(self, k: int, model: str | None = None, exclude=(), exclude_families=(),
               only_family: str | None = None, tag: str | None = None, policy: Policy | None = None) -> list[Conn]:
        """Up to k selectable nodes of distinct families, most reliable model of each family first;
        with `tag`, only nodes advertising it; with `policy` (essaim/1.3), only nodes the requester's
        policy and the nodes' serving policies allow, the soft-avoided ones last. Cost: O(models walked
        + k * |exclude|), independent of the number of nodes (an allowlist or a private swarm: O(its size))."""
        if policy is None:
            return self._select(k, model, set(exclude), set(exclude_families), only_family, tag, None)
        exclude = set(exclude) | policy.deny
        fams = set(exclude_families) | policy.deny_fams
        if policy.soft:
            out = self._select(k, model, exclude | policy.soft, fams, only_family, tag, policy)
            if len(out) < k:  # not enough peers besides the recent ones: complete with them
                more = self._select(k - len(out), model, exclude | {c.node_id for c in out},
                                    fams | {c.family.lower() for c in out} if model is None else fams,
                                    only_family, tag, policy)
                out += more
            return out
        return self._select(k, model, exclude, fams, only_family, tag, policy)

    def _select(self, k, model, exclude: set, fams: set, only_family, tag, policy: Policy | None) -> list[Conn]:
        pred = None if policy is None else (lambda c: self._eligible(c, policy))
        lfams = {f.lower() for f in fams}
        if policy is not None and (policy.only is not None or policy.swarm is not None):
            return self._select_small(k, model, exclude, lfams, only_family, tag, policy)
        pools = self._pools if tag is None else self._tag_pools.get(tag, {})
        min_rel = policy.min_rel if policy is not None else None
        if model is not None:
            pool = pools.get(model)
            if pool is not None and only_family is not None and pool.family != only_family:
                pool = None
            if pool is not None and policy is not None and pool.family.lower() in policy.deny_fams:
                pool = None
            if pool is not None and min_rel is not None and self.model_p(model) < min_rel:
                pool = None
            c = self._pick(pool, exclude, pred) if pool is not None else None
            return [c] if c is not None and k >= 1 else []
        out: list[Conn] = []
        for m in (self._ordered_models() if tag is None else self._ordered_tag_models(tag)):
            pool = pools.get(m)
            if (pool is None or pool.family in fams or pool.family.lower() in lfams
                    or (only_family is not None and pool.family != only_family)):
                continue
            if min_rel is not None and self.model_p(m) < min_rel:
                continue
            c = self._pick(pool, exclude, pred)
            if c is None:
                continue
            out.append(c)
            fams.add(pool.family)
            lfams.add(pool.family.lower())
            if len(out) >= k:
                break
        return out

    def _select_small(self, k, model, exclude: set, lfams: set, only_family, tag, policy: Policy) -> list[Conn]:
        """Selection among an allowlist or the members of a private swarm: O(size of that set)."""
        if policy.only is not None:
            cands = [self.conns[n] for n in policy.only if n in self.conns]
        else:
            cands = list(self._swarm_members.get(policy.swarm, ()))
        best: dict[str, tuple] = {}
        for c in cands:
            if (self._tier(c) is None or c.node_id in exclude or (c.family or "").lower() in lfams
                    or (only_family is not None and c.family != only_family) or (model is not None and c.info.model != model)
                    or (tag is not None and tag not in c.tags) or not self._eligible(c, policy)):
                continue
            p = self.model_p(c.info.model)
            if policy.min_rel is not None and p < policy.min_rel:
                continue
            key = (-p, self._load_key(c), self.rng.random())
            fam = c.family.lower()
            if fam not in best or key < best[fam][0]:
                best[fam] = (key, c)
        return [c for _, c in sorted(best.values(), key=lambda kc: kc[0])[:k]]

    def _card(self, c: Conn, with_tags: bool = False, full: bool = False) -> PeerCard:
        """`full` (essaim/1.3 requesters only): with the key certificate, swarm proofs and reputation."""
        return PeerCard(node_id=c.node_id, pubkey=c.info.pubkey, model=c.info.model, family=c.family or None,
                        gguf=c.info.gguf, params_b=c.info.params_b, reliability=round(self.model_p(c.info.model), 6),
                        tags=list(c.tags) if with_tags else None,
                        kx=c.info.kx if full else None, swarms=c.info.swarms if full else None,
                        reputation=round(c.reputation, 4) if full else None)

    def _route_target(self, req: Conn, job, route: Route) -> Conn | None:
        """Pick the peer of a routed job: a family not yet used by its group (for a replacement, else
        the replaced job's family), never a node already used by the group or excluded.
        route.family: only that family (a replacement stays in it). route.tag: a node advertising the
        tag first, else any node; `self._tag_match` tells which. The requester's policy (essaim/1.3) and
        the peers' serving policies always apply."""
        self._route_error = None
        key = (req.node_id, route.group)
        g = self.groups.get(key)
        new_group = g is None
        if new_group:  # created only once a peer is found (audit net #2), within per-account caps
            if self._groups_of[req.node_id] >= MAX_GROUPS_PER_ACCOUNT or len(self.groups) >= MAX_GROUPS:
                self._route_error = "too_many_groups"
                return None
            g = Group()
        if len(g.job_family) >= MAX_GROUP_JOBS:
            return None
        excl = g.nodes | set(route.exclude)
        tag = route.tag
        pol = Policy.of(route, req.node_id)
        self._tag_match = None
        if route.family:  # strict family; within it, the tag is a preference
            chosen = (self.select(1, model=route.model, exclude=excl, only_family=route.family, tag=tag, policy=pol)
                      if tag else [])
            if not chosen:
                chosen = self.select(1, model=route.model, exclude=excl, only_family=route.family, policy=pol)
        elif route.model:
            chosen = self.select(1, model=route.model, exclude=excl, tag=tag, policy=pol) if tag else []
            if not chosen:
                chosen = self.select(1, model=route.model, exclude=excl, policy=pol)
        else:
            chosen = self.select(1, exclude=excl, exclude_families=g.families, tag=tag, policy=pol) if tag else []
            if not chosen:
                chosen = self.select(1, exclude=excl, exclude_families=g.families, policy=pol)
            fam = g.job_family.get(route.replaces) if route.replaces else None
            if not chosen and fam is not None:
                chosen = self.select(1, exclude=excl, only_family=fam, tag=tag, policy=pol) if tag else []
                if not chosen:
                    chosen = self.select(1, exclude=excl, only_family=fam, policy=pol)
        if not chosen:
            return None
        if tag:
            self._tag_match = tag in chosen[0].tags
        c = chosen[0]
        if new_group:
            self.groups[key] = g
            self._groups_of[req.node_id] += 1
            self._schedule(time.monotonic() + job.deadline_ms / 1000 + GROUP_GRACE_S, "group", key)
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
        elif kind == "canary":
            self._run_canary(*obj)
        elif kind == "group":
            g = self.groups.get(obj)
            if g is None:
                return
            if g.expires > now:
                self._schedule(g.expires, "group", obj)
            else:
                del self.groups[obj]
                n = self._groups_of[obj[0]] - 1
                if n > 0:
                    self._groups_of[obj[0]] = n
                else:
                    self._groups_of.pop(obj[0], None)

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
    def latest_version(self) -> str | None:
        latest = self.releases.latest if self.releases is not None else None
        return latest.get("version") if latest else None

    def _on_release(self, info: dict) -> None:
        """A new latest release: tell the connected nodes that asked for it (O(N), once per release)."""
        frame = UpdateAvailable(version=info["version"])
        n = 0
        for c in list(self.conns.values()):
            if "update" in c.features and not c.closed:
                c.send(frame)
                n += 1
        log.info("release %s announced to %d node(s)", info["version"], n)

    async def _serve_ws(self, ws: WebSocket) -> None:
        features = requested_features(ws.query_params.get("features"))
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
        conn = Conn(node_id=info.node_id, info=info, ws=ws, wan=self.wan, features=features)
        if info.model:
            conn.spot = list(self.ledger.node_spot(info.node_id, info.model))
            self._refresh_reputation(conn)
            if sum(conn.spot):
                self._spotted.add(conn)
        self.conns[info.node_id] = conn
        conn.reported = self.report_scores.get(info.node_id, 0.0) >= REPORT_FLAG_SCORE
        for g in conn.swarm_groups:
            self._swarm_members.setdefault(g, set()).add(conn)
        writer = asyncio.create_task(self._writer(conn))
        delayed = None
        if self.wan is not None:
            conn.inbox = asyncio.Queue(OUTBOX)
            delayed = asyncio.create_task(self._delayed_dispatch(conn))
        conn.send(Welcome(node_id=info.node_id, balance=self.ledger.balance(info.node_id) or 0.0,
                          latest_version=self.latest_version() if "update" in features else None))
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
        if hello.info.node_id in self.banned:
            return "banned"
        kx = hello.info.kx
        if kx is not None and kx_problem(hello.info.node_id, hello.info.pubkey, kx, model=hello.info.model):
            return "bad_kx"  # the same validation as the requesters (Codex review)
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
        for g in conn.swarm_groups:
            members = self._swarm_members.get(g)
            if members is not None:
                members.discard(conn)
                if not members:
                    del self._swarm_members[g]
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
        elif isinstance(frame, Reserve):
            self._reserve(conn, frame)
        elif isinstance(frame, SealedJobFrame):
            self._submit_sealed(conn, frame)
        elif isinstance(frame, (ResultFrame, SealedResultFrame)):
            self._on_result(conn, frame.result)
        elif isinstance(frame, KxFrame):
            kx = frame.kx
            if kx_problem(conn.node_id, conn.info.pubkey, kx, model=conn.info.model):
                conn.send(ErrorFrame(error="bad_kx"))
            elif conn.info.model:
                conn.info = conn.info.model_copy(update={"kx": kx})
        elif isinstance(frame, Dispute):
            self._on_dispute(conn, frame)
        elif isinstance(frame, ServePolicy):
            conn.serve_deny = frozenset(frame.deny)
            conn.serve_rate, conn.serve_conc = frame.rate_per_min, frame.max_concurrent
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
        if tj.counted and t is not None:
            n = t.req_running.get(tj.requester, 0) - 1
            if n > 0:
                t.req_running[tj.requester] = n
            else:
                t.req_running.pop(tj.requester, None)

    def _can_request(self, req: Conn, job_id: str) -> str | None:
        """Checks common to every job a requester sends (None: accepted)."""
        if req.node_id in self.banned:
            return "banned"
        if job_id in self.jobs or self.ledger.has_job(job_id):
            return "duplicate_job"
        bal = self.ledger.balance_milli(req.node_id)
        if bal is None or bal < 0 or bal - self._committed.get(req.node_id, 0) < -self.max_overdraft:
            return "insufficient_credits"
        if req.submitted >= self._inflight_cap(req) or self._retained.get(req.node_id, 0) >= MAX_RETAINED_PER_ACCOUNT:
            return "too_many_jobs"
        if len(self.jobs) >= MAX_JOBS:
            return "tracker_busy"
        return None

    def _inflight_cap(self, req: Conn) -> int:
        """Jobs an account may have in flight. A new account gets new_account_inflight; the cap grows to
        max_inflight with the account's standing (ledger.reporter_weight: age and credits spent on others'
        work). In-flight jobs commit no credits (see max_overdraft), so this bounds what a fresh, possibly
        throwaway account can owe in one burst (audit net #1). Cached on the connection for a minute."""
        now = time.monotonic()
        cached = getattr(req, "_cap", None)
        if cached is not None and now - cached[1] < 60.0:
            return cached[0]
        base = min(self.new_account_inflight, self.max_inflight)
        w = self.ledger.reporter_weight(req.node_id, time.time())
        cap = base + int((self.max_inflight - base) * w)
        req._cap = (cap, now)
        return cap

    def _commit(self, tj: TrackedJob, amount: int) -> None:
        """Set the credits this job holds on its requester's account (see max_overdraft)."""
        if not tj.commits:
            return  # tracker jobs (canaries, spot checks): nobody's credits
        self._committed[tj.requester] += amount - tj.committed
        tj.committed = amount
        if self._committed[tj.requester] <= 0:
            self._committed.pop(tj.requester, None)

    def _strip(self, tj: TrackedJob) -> None:
        """Drop what is no longer needed of a finished job (its prompt; its answer once settled and
        compared), so that jobs kept for late receipts hold little memory (audit net #2)."""
        if (tj.spot_job is not None or tj.spot_of is not None) and not tj.compared:
            return  # a plaintext spot comparison may still need the prompt and the answer (a late result)
        if isinstance(tj.job, SealedJob):
            if not tj.settled:  # a late answer may still be disputed: keep the header (and its keys), not ct
                if tj.job.ct != "AAAA":
                    tj.job = tj.job.model_copy(update={"ct": "AAAA"})
            else:
                tj.job = Header(job_id=tj.job.job_id, requester_id=tj.requester, max_tokens=tj.job.max_tokens,
                                deadline_ms=tj.job.deadline_ms)
        elif not isinstance(tj.job, Header):
            tj.job = Header(job_id=tj.job.job_id, requester_id=tj.requester, max_tokens=tj.job.max_tokens,
                            deadline_ms=tj.job.deadline_ms)
        if tj.settled and tj.result is not None and not isinstance(tj.result, _Settled):
            tj.result = _Settled(model=tj.result.model, completion_tokens=tj.result.completion_tokens)

    def _target_problem(self, req: Conn, target: str, sealed: bool) -> tuple[Conn | None, str | None]:
        """A job sent to a named peer: the peer, or why it cannot take the job."""
        tgt = self.conns.get(target)
        if tgt is None or not tgt.info.model:
            return None, "peer_offline"
        if not tgt.info.accepting:
            return None, "peer_paused"
        if tgt.suspended_until is not None:  # e.g. an essaim/1 gateway with an older directory
            return None, "peer_suspended"
        if tgt.busy >= max(1, tgt.info.max_parallel):
            return None, "peer_busy"
        if tgt.node_id in self.banned:
            return None, "peer_banned"
        if not self._serves(tgt, req.node_id):
            return None, "peer_refuses"
        if sealed and not tgt.kx_ok(margin=0):
            return None, "peer_no_e2e"
        return tgt, None

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
        if job.prompt_chars() > MAX_PROMPT_CHARS:
            return reject("prompt_too_large")
        held = self.jobs.get(job.job_id)
        if held is not None and self._open_reservation(held, req) and f.target is not None:
            # essaim/1.3: the content of a reserved job, in clear (the peer has no key and the
            # requester's settings allow plaintext)
            if f.target != held.target or job.max_tokens != held.job.max_tokens or job.deadline_ms != held.job.deadline_ms:
                self._cancel(held)
                return reject("reservation_mismatch")
            self._fill(held, job, JobFrame(job=job, requester_pubkey=req.info.pubkey))
            self._maybe_spot_check(held, req.node_id)
            return
        why = self._can_request(req, job.job_id)
        if why:
            return reject(why)
        if f.target is None:  # essaim/1.1: the tracker picks the peer and tells the requester first
            tgt = self._route_target(req, job, f.route)
            if tgt is None:
                return reject(self._route_error or "no_peer")
            tagged = f.route.tag is not None  # only a tags-aware requester sets it: it parses the new fields
            req.send(Assigned(job_id=job.job_id, peer=self._card(tgt, with_tags=tagged),
                              tag_match=self._tag_match if tagged else None))
        else:
            tgt, why = self._target_problem(req, f.target, sealed=False)
            if why:
                return reject(why)
        tj = self._route(job, req.node_id, tgt, req.info.pubkey, req)
        req.submitted += 1
        self._maybe_spot_check(tj, req.node_id)

    def _reserve(self, req: Conn, f: Reserve) -> None:
        """essaim/1.3: pick and hold a peer for a routed job whose content will follow."""
        def reject(err: str) -> None:
            req.send(JobError(job_id=f.job_id, error=err))

        why = self._can_request(req, f.job_id)
        if why:
            return reject(why)
        hdr = Header(job_id=f.job_id, requester_id=req.node_id, max_tokens=f.max_tokens, deadline_ms=f.deadline_ms)
        tgt = self._route_target(req, hdr, f.route)
        if tgt is None:
            return reject(self._route_error or "no_peer")
        tagged = f.route.tag is not None
        req.send(Assigned(job_id=f.job_id, peer=self._card(tgt, with_tags=tagged, full=True),
                          tag_match=self._tag_match if tagged else None))
        self._track(hdr, req.node_id, tgt, req, time.monotonic() + RESERVE_TTL_S, reserved=True)
        req.submitted += 1

    def _submit_sealed(self, req: Conn, f: SealedJobFrame) -> None:
        """essaim/1.3: an encrypted job, for a reserved peer or a named one (directory mode)."""
        sj = f.job

        def reject(err: str) -> None:
            req.send(JobError(job_id=sj.job_id, node_id=sj.target, error=err))

        if not pubkey_matches(sj.requester_id, f.requester_pubkey) or not sj.verify(f.requester_pubkey):
            return reject("bad_signature")
        held = self.jobs.get(sj.job_id)
        if held is not None and self._open_reservation(held, req):
            if (sj.target != held.target or sj.max_tokens != held.job.max_tokens
                    or sj.deadline_ms != held.job.deadline_ms):
                self._cancel(held)
                return reject("reservation_mismatch")
            self._fill(held, sj, SealedJobFrame(job=sj, requester_pubkey=f.requester_pubkey))
            self._maybe_canary(held)
            return
        why = self._can_request(req, sj.job_id)
        if why:
            return reject(why)
        tgt, why = self._target_problem(req, sj.target, sealed=True)
        if why:
            return reject(why)
        tj = self._track(sj, req.node_id, tgt, req, time.monotonic() + sj.deadline_ms / 1000)
        req.submitted += 1
        tgt.send(SealedJobFrame(job=sj, requester_pubkey=f.requester_pubkey))
        self._maybe_canary(tj)

    def _track(self, job, requester: str, tgt: Conn, req_conn: Conn | None, deadline: float,
               reserved: bool = False, count: bool = True) -> TrackedJob:
        """Register a job on its peer (busy slot, deadline) without sending anything."""
        params = trusted_params(tgt.info.model, tgt.info.params_b)
        tj = TrackedJob(job=job, requester=requester, target=tgt.node_id, model=tgt.info.model,
                        factor=credit_factor(params), deadline=deadline, target_conn=tgt, requester_conn=req_conn,
                        reserved=reserved, sealed=isinstance(job, SealedJob))
        self.jobs[job.job_id] = tj
        heapq.heappush(self._deadlines, (tj.deadline + 1.0, next(self._tseq), job.job_id))
        tgt.busy += 1
        tgt.jobs_in.add(job.job_id)
        if req_conn is not None:
            req_conn.jobs_out.add(job.job_id)
            tj.commits = True  # its exact cost is committed once delivered, until settled
            self._retained[requester] += 1
        if count and req_conn is not None:  # the peer's per-requester limits (ServePolicy)
            tgt.req_calls.setdefault(requester, deque()).append(time.monotonic())
            tgt.req_running[requester] += 1
            tj.counted = True
            if len(tgt.req_calls) > 4096:  # bounded memory: forget idle requesters
                for k in [k for k, q in tgt.req_calls.items() if not q or time.monotonic() - q[-1] > 60][:1024]:
                    tgt.req_calls.pop(k, None)
        self._reindex(tgt)
        return tj

    @staticmethod
    def _open_reservation(tj: TrackedJob, req: Conn) -> bool:
        """A reservation of this requester still waiting for its content: never one already cancelled,
        expired or released (Codex: Reserve, Cancel, then content would be relayed unaccounted)."""
        return tj.reserved and tj.requester == req.node_id and tj.status == "pending" and not tj.released

    def _fill(self, tj: TrackedJob, job, frame) -> None:
        """The content of a reserved job arrived: its own deadline starts now, the peer gets it."""
        tj.job, tj.reserved = job, False
        tj.sealed = isinstance(job, SealedJob)
        tj.deadline = time.monotonic() + job.deadline_ms / 1000
        heapq.heappush(self._deadlines, (tj.deadline + 1.0, next(self._tseq), job.job_id))
        if tj.target_conn is not None:
            tj.target_conn.send(frame)

    def _route(self, job: Job, requester: str, tgt: Conn, requester_pubkey: str,
               req_conn: Conn | None = None) -> TrackedJob:
        tj = self._track(job, requester, tgt, req_conn, time.monotonic() + job.deadline_ms / 1000)
        tgt.send(JobFrame(job=job, requester_pubkey=requester_pubkey))
        return tj

    def _maybe_spot_check(self, tj: TrackedJob, requester: str) -> None:
        """Duplicate a small fraction of PLAINTEXT jobs (older gateways, or peers without encryption) on
        another node running the same model file. Only greedy jobs (temperature 0, same seed) with an
        extractable answer (math, multiple choice) are checked: comparing two SAMPLED texts would punish
        honest nodes more than a cheater returning the modal answer, since the false-positive rate is
        then at least the detection rate. Encrypted jobs are audited by canaries (`_maybe_canary`)."""
        if not isinstance(tj.job, Job):
            return
        rc = self.conns.get(requester)
        if rc is not None and "e2e" in rc.features:
            return  # a policy-aware requester: never send its question to a peer it did not check (Codex review)
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

    def _spot_checker(self, pool: Pool, gguf: str | None, exclude: set, e2e: bool = False) -> Conn | None:
        """A selectable node of the pool serving the same GGUF file, idle or partly busy: random draws in
        both sets, then a bounded scan (best effort, O(1): with very few compatible nodes among many,
        a spot check may be skipped). `e2e`: only a node with a valid key (a canary)."""
        def ok(c: Conn) -> bool:
            return (c.node_id not in exclude and not c.closed and c.info.gguf == gguf
                    and (not e2e or c.kx_ok()) and c.node_id not in self.banned)

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

    def _on_result(self, node: Conn, res: JobResult | SealedResult) -> None:
        tj = self.jobs.get(res.job_id)
        if tj is None or tj.target != node.node_id or res.node_id != node.node_id or tj.reserved:
            return node.send(ErrorFrame(error=f"unknown_job: {res.job_id}"))
        if tj.status not in ("pending", "cancelled"):
            return  # late duplicate, deadline passed or already failed
        sealed = isinstance(res, SealedResult)
        if sealed != tj.sealed:
            return self._fail(tj, "bad_result_kind")
        if not res.verify(node.info.pubkey):
            node.send(ErrorFrame(error=f"bad_signature: {res.job_id}"))
            return self._fail(tj, "bad_result_signature")
        if res.model != tj.model:
            return self._fail(tj, "model_mismatch")
        # Self-reported token counts are bounded by the request and by the text actually returned (for
        # an encrypted result: by its padded size, all the tracker can see).
        size = sum(len(c) for c in res.chunks) * 3 // 4 if sealed else len(res.text.encode("utf-8"))
        tj.tokens = min(res.completion_tokens, tj.job.max_tokens, size + 16)
        tj.status, tj.result, tj.delivered_at = "delivered", res, time.monotonic()
        self._commit(tj, int(round(tj.tokens * tj.factor * MILLI)))  # now the exact cost, until settled
        self._delivered.append((tj.delivered_at, res.job_id))
        node.strikes, node.level = 0, 0  # a delivered result clears the node's health record
        self._release(tj)
        if tj.canary is not None:  # our own canary: decrypted and compared here, paid by the network if it opens
            if self._canary_result(tj, node):
                self._settle(tj, "receipt", None, minted=True)
            else:
                self._refuse_payment(tj, "canary")
            return
        if tj.spot_of is not None:  # our own spot check: paid by the network
            amount = int(round(tj.tokens * tj.factor * MILLI))
            self.ledger.settle(res.job_id, None, node.node_id, res.model, tj.tokens, amount, "spot",
                               result=res.model_dump(mode="json"))
            tj.settled = True
        else:
            req = self.conns.get(tj.requester)
            if req is not None:
                req.send(SealedResultFrame(result=res) if sealed else ResultFrame(result=res))
        self._compare_spot(tj)

    # ---------- canaries (essaim/1.3) ----------
    def _maybe_canary(self, tj: TrackedJob) -> None:
        """After a real encrypted job, with probability spot_rate, schedule a canary for the same peer,
        imitating that job's header and padded size, after a random delay."""
        if self.spot_rate <= 0 or tj.canary is not None or not isinstance(tj.job, SealedJob):
            return
        if self.rng.random() >= self.spot_rate:
            return
        sj = tj.job
        tmpl = {"max_tokens": sj.max_tokens, "deadline_ms": sj.deadline_ms, "bucket": sealed_size(sj)}
        lo, hi = self.canary_delay_s
        self._schedule(time.monotonic() + self.rng.uniform(lo, hi), "canary", (tj.target, tmpl))

    def _run_canary(self, target_id: str, tmpl: dict) -> None:
        t = self.conns.get(target_id)
        if t is None or not self._current(t) or not t.kx_ok():
            self.canary_stats["skipped"] += 1
            return
        if self._tier(t) is None:
            # Busy (or paused): the audit is postponed, not dropped, so that a peer cannot dodge it by
            # holding its slots during the canary window (Codex review).
            if tmpl.get("attempt", 0) < CANARY_RETRIES and t.suspended_until is None:
                lo, hi = self.canary_delay_s
                self._schedule(time.monotonic() + max(1.0, self.rng.uniform(lo, hi)), "canary",
                               (target_id, {**tmpl, "attempt": tmpl.get("attempt", 0) + 1}))
                self.canary_stats["postponed"] += 1
            else:
                self.canary_stats["skipped"] += 1
            return
        pool = self._pools.get(t.info.model)
        checker = self._spot_checker(pool, t.info.gguf, {target_id}, e2e=True) if pool is not None else None
        if checker is None:  # no other peer with this model file: the known answer is the reference
            self.canary_stats["no_checker"] += 1
        pair = CanaryPair(params=canaries.draw(self.rng))
        seed = self.rng.randrange(2**31 - 1)
        for node in ((t, checker) if checker is not None else (t,)):
            tokens = canaries.honeytokens(self.public_url, pair.params["name"])
            text, pair.expected = canaries.question(pair.params, tokens)
            msg = text + "\n\n" + FORMAT_INSTRUCTIONS["math"]  # as a gateway words a math question
            pseudo = Identity.generate()
            job = Job(job_id=uuid.uuid4().hex, requester_id=pseudo.node_id, messages=[{"role": "user", "content": msg}],
                      max_tokens=tmpl["max_tokens"], temperature=0.0, seed=seed, deadline_ms=tmpl["deadline_ms"],
                      task_hint="math")
            try:
                sj, session = seal_job(job, node.node_id, node.info.kx, pseudo, min_bucket=tmpl["bucket"])
            except E2EError:
                self.canary_stats["seal_failed"] += 1
                return
            tj = self._track(sj, pseudo.node_id, node, None, time.monotonic() + sj.deadline_ms / 1000, count=False)
            tj.canary = (pair, session)
            pair.jobs.append(tj)
            used = canaries.used_tokens(pair.params, tokens)
            if used:
                self.ledger.record_honeytokens(node.node_id, sj.job_id, used)
            node.send(SealedJobFrame(job=sj, requester_pubkey=pseudo.pubkey))
        self.canary_stats["sent"] += 1

    def _canary_result(self, tj: TrackedJob, node: Conn) -> bool:
        """Decrypt one canary answer (before it is paid: Codex review); compare when the pair is complete.
        Returns whether the answer opened. Without a checker, the known answer is the reference."""
        pair, session = tj.canary
        jid = tj.job.job_id
        try:
            plain, _ = open_result(tj.result, session, node.info.pubkey)
            pair.texts[jid] = plain.text
            pair.reasons[jid] = plain.finish_reason if truncated_credibly(plain.text, plain.finish_reason,
                                                                          tj.job.max_tokens) else "stop"
        except E2EError:
            pair.texts[jid] = None
            self.canary_stats["undecryptable"] += 1
            self._record_spot(tj.model, [(tj.target, tj.model)], False, baseline=False)  # this peer only
            return False
        if len(pair.jobs) == 1:  # no checker for this model file: the known answer decides, for this peer only
            if pair.reasons[jid] == "length":
                self.canary_stats["inconclusive"] += 1
                self._inconclusive_by(tj.target, tj.model)
                return True
            got = extract_answer(pair.texts[jid], "math")
            ok = got is not None and got == pair.expected
            self.canary_stats["known_ok" if ok else "known_wrong"] += 1
            self._record_spot(tj.model, [(tj.target, tj.model)], ok, baseline=False)
            return True
        if len(pair.texts) < 2 or any(v is None for v in pair.texts.values()):
            return True
        a, b = pair.jobs
        agree = spot_verdict(pair.texts[a.job.job_id], pair.texts[b.job.job_id], "math", [],
                             (pair.reasons.get(a.job.job_id), pair.reasons.get(b.job.job_id)))
        if agree is None:
            self.canary_stats["inconclusive"] += 1
            for x in (a, b):
                if pair.reasons.get(x.job.job_id) == "length":
                    self._inconclusive_by(x.target, x.model)
            return True
        self.canary_stats["agree" if agree else "disagree"] += 1
        self._record_spot(a.model, [(a.target, a.model), (b.target, b.model)], agree)
        return True

    def _inconclusive_by(self, node_id: str, model: str) -> None:
        """Audits a peer made inconclusive by a cut-short answer: every INCONCLUSIVE_LIMIT-th one counts
        as a disagreement for that peer (inconclusive audits are watched, not free)."""
        self._inconclusive[node_id] += 1
        if self._inconclusive[node_id] >= INCONCLUSIVE_LIMIT:
            self._inconclusive.pop(node_id, None)
            self._record_spot(model, [(node_id, model)], False, baseline=False)

    def _refuse_payment(self, tj: TrackedJob, why: str) -> None:
        """An answer that does not open: settled at zero (recorded), counted against the peer."""
        self.ledger.settle(tj.job.job_id, None if tj.canary is not None else tj.requester, tj.target, tj.model,
                           0, 0, "refused", result=tj.result.model_dump(mode="json"))
        tj.settled = True
        self._commit(tj, 0)
        self.ledger.record_failure(tj.target)
        self.health_events[f"refused_{why}"] += 1

    def _on_dispute(self, req: Conn, f: Dispute) -> None:
        """The requester says an encrypted answer does not open, and reveals the job's ephemeral secret
        so that the tracker can check it itself (verifiable: a wrong secret is refused, an answer that
        opens is paid as usual)."""
        tj = self.jobs.get(f.job_id)

        def bad(why: str) -> None:
            req.send(ErrorFrame(error=f"bad_dispute: {why}: {f.job_id}"))

        if (tj is None or tj.requester != req.node_id or not tj.sealed or tj.status != "delivered" or tj.settled
                or not isinstance(tj.job, SealedJob) or not isinstance(tj.result, SealedResult)):
            return bad("not_disputable")
        try:
            session = dispute_session(tj.job, f.eph_secret)
        except E2EError:
            return bad("wrong_secret")
        pub = tj.target_conn.info.pubkey if tj.target_conn is not None else self.ledger.pubkey(tj.target)
        try:
            open_result(tj.result, session, pub or "")
        except E2EError:
            self._refuse_payment(tj, "dispute")
            self._record_spot(tj.model, [(tj.target, tj.model)], False, baseline=False)
            self._strip(tj)
            return
        self.health_events["dispute_rejected"] += 1
        self._settle(tj, "auto", None)  # it opens: the requester pays

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
        self._commit(tj, 0)
        req = self.conns.get(tj.requester)
        if req is not None and tj.spot_of is None:
            req.send(JobError(job_id=tj.job.job_id, node_id=tj.target, error=error))
        self._strip(tj)

    def _cancel(self, tj: TrackedJob) -> None:
        tj.status, tj.cancelled = "cancelled", True
        self._release(tj)
        self._commit(tj, 0)  # a result racing the cancel commits its exact cost again (_on_result)
        tgt = self.conns.get(tj.target)
        if tgt is not None and not tj.reserved:  # a reserved peer never heard of the job
            tgt.send(Cancel(job_id=tj.job.job_id))
        self._strip(tj)

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

    def _settle(self, tj: TrackedJob, kind: str, receipt: dict | None, minted: bool = False) -> None:
        """`minted`: paid by the network (a canary), no requester account is debited; it is recorded
        with the same kind as a real job so that the public ledger does not tell canaries apart."""
        amount = int(round(tj.tokens * tj.factor * MILLI))
        self.ledger.settle(tj.job.job_id, None if minted else tj.requester, tj.target, tj.model, tj.tokens, amount,
                           kind, receipt=receipt, result=tj.result.model_dump(mode="json"))
        tj.settled = True
        self._commit(tj, 0)
        if tj.canary is None:
            self._strip(tj)

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
        if not isinstance(orig.job, Job) or isinstance(orig.result, _Settled) or isinstance(spot.result, _Settled):
            return  # (defensive) a stripped job has nothing left to compare
        orig.compared = True
        reasons = tuple("length" if truncated_credibly(x.result.text, x.result.finish_reason, orig.job.max_tokens)
                        else "stop" for x in (orig, spot))
        agree = spot_verdict(orig.result.text, spot.result.text, orig.job.task_hint,
                             [m.model_dump() for m in orig.job.messages], reasons)
        if agree is None:
            for x, r in zip((orig, spot), reasons):
                if r == "length":
                    self._inconclusive_by(x.target, x.model)
        if agree is None:  # no answer to compare (cut short...): inconclusive, nobody is blamed
            return
        self._record_spot(orig.model, [(orig.target, orig.model), (spot.target, spot.model)], agree)

    def _record_spot(self, model: str, nodes: list, agree: bool, baseline: bool = True) -> None:
        """One comparison (spot check or canary pair). `baseline`: it also counts in the model's honest
        disagreement rate (False for a single peer's undecryptable answer)."""
        if baseline:
            self.ledger.record_model_spot(model, agree)
            self._mspot.setdefault(model, [0, 0])[0 if agree else 1] += 1
        for node_id, m in nodes:
            self.ledger.record_spot(node_id, m, agree)
            c = self.conns.get(node_id)  # the reputation used by the selection follows the new counts
            if c is not None and c.info.model == m:
                c.spot[0 if agree else 1] += 1
                self._spotted.add(c)
                self._refresh_reputation(c)
        # The model's baseline moved for its other nodes too: they are refreshed by the sweep.

    # ---------- bans, reports, honeytokens (essaim/1.3) ----------
    def _reload_bans(self, force: bool = False) -> None:
        """Admin ban file (JSON: a list of node ids or of {node_id, reason}), re-read when it changes,
        plus the bans recorded in the ledger (honeytoken sightings)."""
        if self.ban_file is not None:
            try:
                mtime = self.ban_file.stat().st_mtime
            except OSError:
                mtime = None
            if force or mtime != self._ban_mtime:
                self._ban_mtime = mtime
                bans: dict[str, str] = {}
                if mtime is not None:
                    try:
                        for x in json.loads(self.ban_file.read_text(encoding="utf-8-sig")):
                            nid = x.get("node_id") if isinstance(x, dict) else x
                            if isinstance(nid, str) and re.fullmatch(r"[0-9a-f]{32}", nid):
                                bans[nid] = str(x.get("reason", "admin") if isinstance(x, dict) else "admin")[:200]
                    except (OSError, ValueError, AttributeError):
                        log.warning("ban file unreadable: %s", self.ban_file)
                        bans = dict(self._file_bans)
                self._file_bans = bans
        self.banned = {**self.ledger.bans(), **self._file_bans}
        for nid in list(self.banned):
            c = self.conns.get(nid)
            if c is not None:
                c.send(ErrorFrame(error="banned"))
                self._drop(c)

    def ban(self, node_id: str, reason: str, source: str = "admin") -> None:
        self.ledger.add_ban(node_id, reason, source)
        self._reload_bans()

    def honeytoken_sighting(self, text: str, source: str) -> int:
        """Look for recorded honeytokens in `text`; ban the peers that received the ones found."""
        found = 0
        for m in {x.rstrip(".") for x in HONEY_RE.findall(text or "")[:64]}:
            hit = self.ledger.honeytoken(m)
            if hit is None:
                continue
            found += 1
            node_id, job_id = hit
            self.ledger.record_sighting(m, node_id, source)
            log.warning("honeytoken of job %s seen (%s): node %s banned", job_id[:8], source, node_id[:8])
            self.ban(node_id, f"honeytoken ({source})", "honeytoken")
        return found

    def accept_report(self, rep: NodeReport) -> tuple[bool, str, float]:
        """Check a signed report and store it with the reporter's weight (sybil resistance: a new or
        idle account weighs nothing). Returns (accepted, reason, weight)."""
        now = time.time()
        pub = self.ledger.pubkey(rep.reporter_id)
        if pub is None:
            return False, "unknown_reporter", 0.0
        if not rep.verify(pub):
            return False, "bad_signature", 0.0
        if abs(rep.ts - now) > 600:
            return False, "stale", 0.0
        if rep.reporter_id == rep.node_id:
            return False, "self_report", 0.0
        if rep.reason == "disagreement":
            if rep.job_id is None or not self.ledger.job_between(rep.job_id, rep.reporter_id, rep.node_id):
                return False, "job_not_paid", 0.0
        if self.ledger.reports_since(rep.reporter_id, now - 3600) >= REPORTS_PER_HOUR:
            return False, "rate_limited", 0.0
        weight = self.ledger.reporter_weight(rep.reporter_id, now)
        if not self.ledger.add_report(rep.reporter_id, rep.node_id, rep.reason, rep.job_id, rep.ts, weight):
            return False, "duplicate", 0.0
        self.report_scores = self.ledger.report_scores(now - REPORT_WINDOW_S, REPORT_MIN_REPORTERS)
        c = self.conns.get(rep.node_id)
        if c is not None:
            c.reported = self.report_scores.get(c.node_id, 0.0) >= REPORT_FLAG_SCORE
        return True, "ok", weight

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
        if now - self._ban_checked >= 2.0:  # the ban file and the ledger's bans (`myriad tracker-bans`)
            self._ban_checked = now
            self._reload_bans()
        while self._deadlines and self._deadlines[0][0] < now:
            _, _, jid = heapq.heappop(self._deadlines)
            tj = self.jobs.get(jid)
            if tj is None or tj.status != "pending" or tj.deadline + 1.0 > now:
                continue  # done, or a newer deadline was set (the content of a reserved job arrived)
            tj.status = "expired"
            self._release(tj)
            self._commit(tj, 0)
            req = self.conns.get(tj.requester)
            if tj.reserved:  # the content never came: the requester's doing, not the peer's
                if req is not None:
                    req.send(JobError(job_id=jid, node_id=tj.target, error="reservation_expired"))
                continue
            self._timeout_strike(tj)
            if req is not None and tj.spot_of is None:
                req.send(JobError(job_id=jid, node_id=tj.target, error="deadline"))
            tgt = self.conns.get(tj.target)
            if tgt is not None:
                tgt.send(Cancel(job_id=jid))
        while self._delivered and now - self._delivered[0][0] > self.receipt_grace_s:
            _, jid = self._delivered.popleft()
            tj = self.jobs.get(jid)
            if tj is not None and tj.status == "delivered" and not tj.settled:
                # The requester got the signed result but sent no receipt: the node is paid anyway (also
                # for a result that raced a cancellation: it was relayed, so it was received).
                self._settle(tj, "auto", None)
        while self._finished and now - self._finished[0][0] > self.job_ttl_s:
            _, jid = self._finished.popleft()
            tj = self.jobs.get(jid)
            if tj is None:
                continue
            if tj.settled or tj.cancelled or tj.status != "delivered":
                del self.jobs[jid]
                self._commit(tj, 0)
                if tj.commits:
                    n = self._retained[tj.requester] - 1
                    if n > 0:
                        self._retained[tj.requester] = n
                    else:
                        self._retained.pop(tj.requester, None)
            else:  # delivered, receipt still awaited: look again later
                self._finished.append((now, jid))


async def _close_quietly(ws) -> None:
    """Close a WebSocket that may already be closed (the peer left first): nothing to report then."""
    try:
        await ws.close()
    except Exception:
        pass


def truncated_credibly(text: str, reason: str | None, limit: int) -> bool:
    """A self-declared "length" finish is believed only for an answer about as long as its token limit
    allows (a peer could otherwise declare "length" to make any audit inconclusive: Codex review)."""
    return reason == "length" and len(text or "") >= TRUNCATION_CHARS_PER_TOKEN * min(limit, 256)


def spot_verdict(a: str, b: str, hint: str | None, messages: list[dict], reasons=(None, None)) -> bool | None:
    """Two greedy answers to the same job: do their EXTRACTED answers agree? None (inconclusive) if
    either has none, or if either was cut short (finish_reason "length": nodes may cap the output
    differently, and a truncated answer's last number is not its answer; audit net #5)."""
    if any(r == "length" for r in reasons):
        return None
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

    if "release_repo" not in kw:  # the public tracker watches the app's releases unless told not to
        repo = os.environ.get("MYRIAD_RELEASE_REPO", DEFAULT_RELEASE_REPO).strip()
        kw["release_repo"] = None if repo.lower() in ("", "off", "none", "0") else repo
    if "release_poll_s" not in kw and os.environ.get("MYRIAD_RELEASE_POLL_S", "").strip():
        kw["release_poll_s"] = float(os.environ["MYRIAD_RELEASE_POLL_S"])
    tracker = Tracker(db_path=db, **kw)
    if tracker.wan is not None:
        log.warning("WAN emulation ON: %s (experiments only)", tracker.wan.describe())
    uvicorn.run(tracker.app, host=host, port=port, ws_max_size=MAX_FRAME_BYTES, log_level="info")
