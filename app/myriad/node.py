"""Node: one outbound WebSocket to the tracker (works behind a NAT box), challenge-response login,
execution of the jobs it receives within the user's limits, signed results. The same connection
carries the jobs this node's gateway sends to other peers.

essaim/1.3: a serving node advertises an X25519 key certified by its identity (e2e.Keyring, rotated
daily, a KxFrame announcing each new key) and serves encrypted jobs (SealedJobFrame): it checks the
pseudonym's signature, the target, the expiry and the replay cache, decrypts, applies its limits
(security.ServeGuard), runs the job and seals the answer for the requester. The job's content only
lives in memory; after each job the engine's cache is scrubbed (engine.scrub) and nothing about the
content is logged. Errors sent back for an encrypted job are coarse codes, never engine messages
(they could quote the prompt). The node's blocklist and per-requester limits are also sent to the
tracker (ServePolicy), which enforces them per paying account (the node only sees pseudonyms).

Compatibility: a tracker before essaim/1.3 refuses a Hello carrying `kx`/`swarms` ("hello_expected"),
one before essaim/1.2 also refuses `tags`: the node retries at once without them (`_compat`)."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import time
from collections import deque
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from . import PROTOCOL
from .config import within_hours
from .crypto import Identity, pubkey_matches
from .e2e import E2EError, Keyring, ReplayCache, open_job, seal_result
from .protocol import (MAX_FRAME_BYTES, MAX_TEXT_CHARS, MAX_TOKENS, Assigned, Cancel, Challenge, ErrorFrame, Hello,
                       Job, JobError, JobFrame, JobResult, KxFrame, NodeInfo, Ping, Pong, ReceiptFrame, Reserve,
                       ResultFrame, Route, SealedJob, SealedJobFrame, SealedResultFrame, ServePolicy, Status,
                       SwarmProof, UpdateAvailable, Welcome, dump_frame, parse_frame)
from .security import Security, ServeGuard

log = logging.getLogger("myriad.node")
HEARTBEAT_S = 15.0
PROBE_TIMEOUT_S = 2.0  # engine health probe before answering a ping (the tracker waits 5 s by default)
PLAIN_REPLAY_S = 3600.0
# Content of jobs never reaches the logs. MYRIAD_LOG_CONTENT=1 (debugging only, off by default) lets
# the parse errors of invalid frames be logged in full (they can quote a frame's content).
LOG_CONTENT = os.environ.get("MYRIAD_LOG_CONTENT") == "1"


class _Refused(ValueError):
    """An older tracker refused a Hello carrying fields it does not know."""


def ws_url(tracker_url: str) -> str:
    """http://host:port -> ws://host:port/v1/ws (https -> wss)."""
    u = urlsplit(tracker_url.rstrip("/"))
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(u.scheme)
    if scheme is None:
        raise ValueError(f"URL de traqueur invalide : {tracker_url}")
    path = u.path if u.path.endswith("/v1/ws") else (u.path + "/v1/ws")
    return urlunsplit((scheme, u.netloc, path, "", ""))


def transport_protected(tracker_url: str) -> bool:
    """TLS (https/wss), or a tracker on this machine or a private network."""
    import ipaddress
    u = urlsplit(tracker_url)
    if u.scheme in ("https", "wss"):
        return True
    host = (u.hostname or "").lower()
    if host in ("localhost",) or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def http_url(tracker_url: str) -> str:
    u = urlsplit(tracker_url.rstrip("/"))
    scheme = {"ws": "http", "wss": "https"}.get(u.scheme, u.scheme)
    path = u.path[: -len("/v1/ws")] if u.path.endswith("/v1/ws") else u.path
    return urlunsplit((scheme, u.netloc, path, "", ""))


class NodeClient:
    # Optional features asked for in the WebSocket URL (`?features=...`): an older tracker ignores the
    # query string, a newer one only sends the matching optional frames to nodes that asked.
    FEATURES: tuple = ("update", "e2e")

    def __init__(self, identity: Identity, tracker_url: str, engine=None, model: str | None = None,
                 family: str | None = None, gguf: str | None = None, params_b: float | None = None, ctx: int = 0,
                 max_parallel: int = 1, accepting: bool = True, active_hours: str | None = None,
                 max_job_tokens: int = 1024, reconnect: bool = True, tags: list[str] | None = None,
                 security: Security | None = None, e2e: bool = True):
        self.identity = identity
        # Skill tags (essaim/1.2), advertised in the Hello. An older tracker refuses a Hello carrying
        # them: the node then reconnects without tags to that tracker (`_compat`).
        self.tags = list(tags) if tags else None
        # 0: every field; 1: without the essaim/1.3 fields (kx, swarms); 2: without tags either
        self._compat = 0
        self.e2e = e2e  # False: never advertise a key (tests of mixed networks)
        self.security = security or Security()
        self.guard = ServeGuard(self.security)
        self.keyring = Keyring(identity, model=model)
        self.replay = ReplayCache()
        self.plain_replay = ReplayCache(50_000)
        self.security.listeners.append(self._security_changed)
        self.node_id = identity.node_id
        self.tracker_url = tracker_url
        self.engine = engine
        self.model, self.family, self.gguf, self.params_b, self.ctx = model, family, gguf, params_b, ctx
        self.max_parallel, self.accepting, self.active_hours = max_parallel, accepting, active_hours
        self.max_job_tokens = max_job_tokens
        self.reconnect = reconnect
        self.state = "déconnecté"
        self.connected = asyncio.Event()
        self.tracker_id: str | None = None
        self.balance: float | None = None
        self.last_error: str | None = None
        self.running: dict[str, asyncio.Task] = {}
        self.waiters: dict[str, asyncio.Queue] = {}
        self.recent: deque = deque(maxlen=50)
        self.stats = {"served": 0, "failed": 0, "cancelled": 0, "rejected": 0, "tokens": 0, "sealed": 0}
        self._ws = None
        self._send_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self.session = 0  # number of tracker sessions opened so far (a new one may be another tracker version)
        self._last_ping = -1e9  # monotonic time of the last essaim/1.1 ping received
        self._pongs: set[asyncio.Task] = set()
        self._observers: set = set()  # callables (direction, frame), e.g. the UI's live chat view
        self._hello_swarms = None  # swarm proofs announced in the current session
        self._announced_kx: str | None = None  # key-agreement key the tracker knows (Hello or KxFrame)
        self._quota: dict[str, str] = {}  # job id -> requester whose concurrency slot it holds
        # Feature "update": latest app version announced by the tracker (a hint, validated by updater.py),
        # and the callable (version, source) told about it.
        self.latest_version: str | None = None
        self.on_update = None

    @property
    def _tags_refused(self) -> bool:
        return self._compat >= 2

    # ---------- limits ----------
    def serving(self) -> bool:
        return bool(self.model) and self.engine is not None

    def accepting_now(self) -> bool:
        engine_ready = self.engine is not None and getattr(self.engine, "state", "prêt") == "prêt"
        return (self.serving() and engine_ready and self.accepting and self.max_parallel > 0
                and within_hours(self.active_hours, dt.datetime.now().hour))

    def e2e_fields(self) -> bool:
        """Whether the Hello carries the essaim/1.3 fields (the tracker accepts them)."""
        return self._compat == 0 and self.e2e and bool(self.model)

    def info(self) -> NodeInfo:
        e2e = self.e2e_fields()
        swarms = self.security.swarm_proofs(self.node_id) if e2e else None
        return NodeInfo(node_id=self.node_id, pubkey=self.identity.pubkey, model=self.model, family=self.family,
                        gguf=self.gguf, params_b=self.params_b, ctx=self.ctx, max_parallel=self.max_parallel,
                        version=PROTOCOL, accepting=self.accepting_now(),
                        tags=self.tags if self.tags and self.model and self._compat < 2 else None,
                        kx=self.keyring.cert() if e2e else None,
                        swarms=[SwarmProof(**p) for p in swarms] if swarms else None)

    async def set_limits(self, max_parallel: int | None = None, accepting: bool | None = None,
                         active_hours: str | None | type(...) = ...) -> None:
        if max_parallel is not None:
            self.max_parallel = max(0, min(int(max_parallel), 64))
        if accepting is not None:
            self.accepting = bool(accepting)
        if active_hours is not ...:
            self.active_hours = active_hours or None
        await self.send_status()

    async def send_status(self) -> None:
        if self.connected.is_set():
            await self.send(Status(accepting=self.accepting_now(), max_parallel=self.max_parallel))

    async def send_serve_policy(self) -> None:
        if self.connected.is_set() and self.e2e_fields():
            await self.send(ServePolicy(**self.guard.serve_policy()))

    def _security_changed(self) -> None:
        """Settings changed (UI): re-send the serving policy; a new private swarm needs a new Hello."""
        if not self.connected.is_set():
            return
        loop = asyncio.get_event_loop()
        loop.create_task(self._apply_security())

    async def _apply_security(self) -> None:
        try:
            await self.send_serve_policy()
            now = self.security.swarm_proofs(self.node_id) if self.e2e_fields() else None
            if now != self._hello_swarms and self._ws is not None and self.reconnect:
                await self._ws.close()  # the run loop reconnects with the new proofs
        except Exception as e:  # noqa: BLE001 - best effort, applied at the next connection anyway
            log.debug("security settings not applied now: %s", type(e).__name__)

    # ---------- connection ----------
    async def send(self, frame) -> None:
        ws = self._ws
        if ws is None:
            raise ConnectionError("non connecté au traqueur")
        self._observe("out", frame)
        async with self._send_lock:
            await ws.send(dump_frame(frame))

    async def run(self) -> None:
        """Connect, serve, reconnect with backoff until stop()."""
        if not transport_protected(self.tracker_url):
            # End-to-end encryption protects the content, not the session: on a plain ws:// link, an active
            # attacker could take over the authenticated connection and spend this account (Codex review).
            log.warning("tracker reached without TLS (%s): use https:// for a tracker outside this machine "
                        "or local network", urlsplit(self.tracker_url).hostname)
        backoff = 1.0
        while not self._stop.is_set():
            self.state = "connexion"
            started = time.monotonic()
            try:
                while True:
                    try:
                        await self._session()
                        break
                    except _Refused:  # an older tracker: at once again, without the fields it does not know
                        level = self._compat
                        self._compat = 1 if level == 0 else 2
                        log.info("tracker refuses newer hello fields: retrying at compatibility level %d", self._compat)
                        self._on_disconnect()
                        self.state = "connexion"
            except (OSError, WebSocketException, asyncio.TimeoutError, ValueError, ValidationError) as e:
                self.last_error = f"{type(e).__name__}: {e}"[:300]
                log.warning("tracker connection: %s", self.last_error)
            finally:
                self._on_disconnect()
            if not self.reconnect or self._stop.is_set():
                break
            if time.monotonic() - started > 30:
                backoff = 1.0
            try:
                await asyncio.wait_for(self._stop.wait(), backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, 30.0)
        self.state = "arrêté"

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            await self._ws.close()

    async def _session(self) -> None:
        url = ws_url(self.tracker_url)
        if self.FEATURES:
            url += "?features=" + ",".join(self.FEATURES)
        async with connect(url, max_size=MAX_FRAME_BYTES, open_timeout=10,
                           ping_interval=20, ping_timeout=20) as ws:
            ch = parse_frame(await asyncio.wait_for(ws.recv(), 10))
            if not isinstance(ch, Challenge) or not pubkey_matches(ch.tracker_id, ch.tracker_pubkey):
                raise ValueError("défi du traqueur invalide")
            info = self.info()
            await ws.send(dump_frame(Hello.make(self.identity, info, ch.nonce)))
            first = parse_frame(await asyncio.wait_for(ws.recv(), 10))
            if isinstance(first, ErrorFrame):
                newer = info.kx is not None or info.swarms is not None or info.tags is not None
                if newer and first.error == "hello_expected" and self._compat < 2:  # unknown fields
                    raise _Refused(first.error)
                raise ValueError(f"refusé par le traqueur : {first.error}")
            if not isinstance(first, Welcome) or first.node_id != self.node_id:
                raise ValueError("réponse inattendue du traqueur")
            self.tracker_id, self.balance = ch.tracker_id, first.balance
            self._ws = ws
            self._hello_swarms = [p.model_dump() for p in info.swarms] if info.swarms else None
            self._announced_kx = info.kx.kx if info.kx is not None else None
            self.session += 1
            self._last_ping = -1e9
            self.state, self.last_error = "connecté", None
            self.connected.set()
            log.info("connected to tracker as %s", self.node_id[:8])
            if first.latest_version:
                self._update_hint(first.latest_version, "welcome")
            await self.send_serve_policy()
            beat = asyncio.create_task(self._heartbeat())
            try:
                async for raw in ws:
                    try:
                        frame = parse_frame(raw)
                    except (ValueError, ValidationError) as e:
                        # a parse error can quote the frame's content: only its type is logged
                        log.warning("invalid frame from tracker: %s", e if LOG_CONTENT else type(e).__name__)
                        continue
                    await self._dispatch(frame)
            finally:
                beat.cancel()

    def _on_disconnect(self) -> None:
        self._ws = None
        self.connected.clear()
        if self.state != "arrêté":
            self.state = "déconnecté"
        for task in list(self.running.values()):  # the tracker fails them anyway
            task.cancel()
        for task in list(self._pongs):
            task.cancel()
        for job_id, q in list(self.waiters.items()):
            q.put_nowait(JobError(job_id=job_id, error="tracker_disconnected"))

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            try:
                await self.announce_kx()
            except Exception:
                return
            if time.monotonic() - self._last_ping < 2 * HEARTBEAT_S:
                continue  # an essaim/1.1 tracker pings us: our Pongs carry the same information
            try:
                await self.send_status()
            except Exception:
                return

    async def announce_kx(self) -> None:
        """Daily key: tell the tracker about the current key whenever it differs from the announced one,
        whatever triggered the rotation (a status() call may rotate before the heartbeat does)."""
        if not self.e2e_fields() or self._announced_kx is None:
            return
        cert = self.keyring.cert()
        if cert.kx != self._announced_kx:
            await self.send(KxFrame(kx=cert))
            self._announced_kx = cert.kx

    async def engine_ok(self) -> bool:
        """Short health probe of the engine (True for a client-only node)."""
        if not self.serving():
            return True
        probe = getattr(self.engine, "health", None)
        if probe is None:
            return getattr(self.engine, "state", "prêt") == "prêt"
        try:
            return bool(await asyncio.wait_for(probe(), PROBE_TIMEOUT_S))
        except Exception:  # timeout or failure: the engine does not answer
            return False

    async def _pong(self, seq: int) -> None:
        ok = await self.engine_ok()
        try:
            await self.send(Pong(seq=seq, accepting=self.accepting_now(), max_parallel=self.max_parallel,
                                 engine_ok=ok, running=min(len(self.running), 1024)))
        except Exception:
            pass

    # ---------- observers (read-only taps on the frames, for the local UI) ----------
    def add_observer(self, cb) -> None:
        self._observers.add(cb)

    def remove_observer(self, cb) -> None:
        self._observers.discard(cb)

    def _observe(self, direction: str, frame) -> None:
        for cb in list(self._observers):
            try:
                cb(direction, frame)
            except Exception as e:  # an observer must never break the protocol
                log.debug("observer failed: %s", type(e).__name__)

    def observe_local(self, direction: str, frame) -> None:
        """Show the local observers the PLAINTEXT view of an encrypted exchange (what this machine sent
        or decrypted): the UI's live chat and the agents view keep working with end-to-end encryption.
        Nothing is sent. While it runs, `local_view` is True: the frame was decrypted and verified by
        this machine's gateway (observers need not, and cannot, re-check its plaintext signature)."""
        self._local_view = True
        try:
            self._observe(direction, frame)
        finally:
            self._local_view = False

    @property
    def local_view(self) -> bool:
        return getattr(self, "_local_view", False)

    async def _dispatch(self, frame) -> None:
        self._observe("in", frame)
        if isinstance(frame, JobFrame):
            await self._on_job(frame)
        elif isinstance(frame, SealedJobFrame):
            await self._on_sealed_job(frame)
        elif isinstance(frame, Cancel):
            # The slot is free at once: the tracker may route the next job right behind this Cancel.
            task = self.running.pop(frame.job_id, None)
            if task is not None:
                task.cancel()
        elif isinstance(frame, Ping):
            self._last_ping = time.monotonic()
            task = asyncio.create_task(self._pong(frame.seq))  # the probe must not stall the receive loop
            self._pongs.add(task)
            task.add_done_callback(self._pongs.discard)
        elif isinstance(frame, (ResultFrame, SealedResultFrame, JobError, Assigned)):
            job_id = frame.result.job_id if isinstance(frame, (ResultFrame, SealedResultFrame)) else frame.job_id
            q = self.waiters.get(job_id)
            if q is not None:
                q.put_nowait(frame)
        elif isinstance(frame, UpdateAvailable):
            self._update_hint(frame.version, "push")
        elif isinstance(frame, ErrorFrame):
            self.last_error = frame.error
            log.warning("tracker error: %s", frame.error)

    def _update_hint(self, version: str, source: str) -> None:
        self.latest_version = version
        cb = self.on_update
        if cb is not None:
            try:
                cb(version, source)
            except Exception as e:  # the updater must never break the protocol
                log.warning("update listener failed: %s", e)

    # ---------- serving ----------
    async def _reject(self, job, error: str) -> None:
        self.stats["rejected"] += 1
        await self.send(JobError(job_id=job.job_id, node_id=self.node_id, error=error))

    async def _on_job(self, f: JobFrame) -> None:
        job = f.job
        pk = f.requester_pubkey
        if not pk or not pubkey_matches(job.requester_id, pk) or not job.verify(pk):
            return await self._reject(job, "bad_job_signature")
        if not self.accepting_now():
            return await self._reject(job, "paused")
        if len(self.running) >= self.max_parallel:
            return await self._reject(job, "busy")
        if job.job_id in self.running:
            return
        why = self.guard.check(job.requester_id, job.prompt_chars())
        if why:
            return await self._reject(job, why)
        # a plaintext job has no expiry of its own: its id is remembered for an hour (a malicious tracker
        # cannot make the node run it twice; Codex review)
        now = time.time()
        if not self.plain_replay.add(job.job_id, now + PLAIN_REPLAY_S, now):
            return await self._reject(job, "replay")
        self._start(job)

    def _start(self, job: Job, session=None) -> None:
        """Run a job; its slot is freed when its task ends, even if it is cancelled before its first
        step (then `_execute`'s finally never runs: audit net #4)."""
        task = asyncio.create_task(self._execute(job, session))
        self.running[job.job_id] = task
        self.guard.started(job.requester_id)
        self._quota[job.job_id] = job.requester_id

        def done(t: asyncio.Task, jid: str = job.job_id) -> None:
            if self.running.get(jid) is t:
                self.running.pop(jid, None)
            self._release_quota(jid)

        task.add_done_callback(done)

    def _release_quota(self, job_id: str) -> None:
        """The requester's concurrency slot, released once (when the answer is sent, or by the task's
        done callback if it never ran)."""
        rid = self._quota.pop(job_id, None)
        if rid is not None:
            self.guard.finished(rid)

    async def _on_sealed_job(self, f: SealedJobFrame) -> None:
        sj: SealedJob = f.job
        if not self.accepting_now():
            return await self._reject(sj, "paused")
        if len(self.running) >= self.max_parallel:
            return await self._reject(sj, "busy")
        if sj.job_id in self.running:
            return
        try:
            job, session = open_job(sj, f.requester_pubkey, self.keyring, self.replay, self.node_id)
        except E2EError as e:
            return await self._reject(sj, e.code)
        why = self.guard.check(job.requester_id, job.prompt_chars(), pseudonymous=True)
        if why:
            return await self._reject(sj, why)
        self.stats["sealed"] += 1
        self._start(job, session)

    async def _execute(self, job: Job, session=None) -> None:
        t0 = time.monotonic()
        entry = {"job_id": job.job_id, "requester": job.requester_id[:8], "ts": time.time(), "status": "en cours",
                 "e2e": session is not None}
        self.recent.appendleft(entry)
        timeout = job.deadline_ms / 1000
        try:
            gen = await asyncio.wait_for(
                self.engine.generate([m.model_dump() for m in job.messages], min(job.max_tokens, self.max_job_tokens),
                                     job.temperature, job.seed, timeout, thinking=job.thinking), timeout)
            tokens = max(0, min(gen.completion_tokens, MAX_TOKENS))
            reason = (gen.finish_reason or None) and str(gen.finish_reason)[:200]
            lp = gen.mean_logprob if gen.mean_logprob is None else min(0.0, gen.mean_logprob)
            if session is not None:
                sr = seal_result(self.identity, session, self.model, gen.text[:MAX_TEXT_CHARS], reason, tokens, lp,
                                 gen.compute_ms)
                await self.send(SealedResultFrame(result=sr))
            else:
                res = JobResult(job_id=job.job_id, node_id=self.node_id, model=self.model,
                                text=gen.text[:MAX_TEXT_CHARS], finish_reason=reason, completion_tokens=tokens,
                                mean_logprob=lp, compute_ms=round(gen.compute_ms, 1)).signed_by(self.identity)
                await self.send(ResultFrame(result=res))
            self.stats["served"] += 1
            self.stats["tokens"] += tokens
            entry.update(status="servi", tokens=tokens)
        except asyncio.CancelledError:
            self.stats["cancelled"] += 1
            entry["status"] = "annulé"
        except Exception as e:
            self.stats["failed"] += 1
            if isinstance(e, asyncio.TimeoutError):
                err = "deadline"
            elif session is not None:  # an engine message could quote the encrypted prompt: a code only
                err = "engine_error"
            else:
                err = f"{type(e).__name__}: {e}"[:200]
            entry.update(status="échec", error=err)
            try:
                await self.send(JobError(job_id=job.job_id, node_id=self.node_id, error=err))
                if not self.accepting_now():  # e.g. llama-server died: stop being routed jobs now
                    await self.send_status()
            except Exception:
                pass
        finally:
            entry["ms"] = round((time.monotonic() - t0) * 1000)
            entry["done_ts"] = time.time()  # completion time (the throughput window uses it)
            if self.running.get(job.job_id) is asyncio.current_task():
                self.running.pop(job.job_id, None)
            self._release_quota(job.job_id)  # before the scrub, which may take a while
            await self._scrub()

    async def _scrub(self) -> None:
        """Erase what the engine keeps of past jobs (its KV cache slots), when it can."""
        scrub = getattr(self.engine, "scrub", None)
        if scrub is None:
            return
        try:
            await asyncio.wait_for(scrub(busy=len(self.running)), 5.0)
        except Exception as e:  # noqa: BLE001 - best effort, never fails a job
            log.debug("engine scrub failed: %s", type(e).__name__)

    # ---------- requesting (used by the gateway) ----------
    def register(self, job_ids, queue: asyncio.Queue) -> None:
        for j in job_ids:
            self.waiters[j] = queue

    def unregister(self, job_ids) -> None:
        for j in job_ids:
            self.waiters.pop(j, None)

    async def submit(self, target: str, job: Job) -> None:
        await self.send(JobFrame(job=job, target=target))

    async def route(self, job: Job, route: Route) -> None:
        """essaim/1.1: let the tracker choose the peer (an Assigned frame names it)."""
        await self.send(JobFrame(job=job, route=route))

    async def reserve(self, job: Job, route: Route) -> None:
        """essaim/1.3: let the tracker choose and hold a peer; nothing about the content is sent."""
        await self.send(Reserve(job_id=job.job_id, route=route, max_tokens=job.max_tokens, deadline_ms=job.deadline_ms))

    async def submit_sealed(self, sj: SealedJob, pseudonym_pubkey: str) -> None:
        await self.send(SealedJobFrame(job=sj, requester_pubkey=pseudonym_pubkey))

    async def cancel(self, job_id: str) -> None:
        await self.send(Cancel(job_id=job_id))

    async def send_receipt(self, receipt) -> None:
        await self.send(ReceiptFrame(receipt=receipt.signed_by(self.identity)))

    def status(self) -> dict:
        return {"node_id": self.node_id, "state": self.state, "tracker": self.tracker_url, "model": self.model,
                "family": self.family, "tags": self.info().tags, "gguf": self.gguf, "params_b": self.params_b, "serving": self.serving(),
                "accepting": self.accepting, "accepting_now": self.accepting_now(), "max_parallel": self.max_parallel,
                "active_hours": self.active_hours, "running": len(self.running), "stats": dict(self.stats),
                "recent": list(self.recent)[:20], "last_error": self.last_error, "e2e": self.e2e_fields(),
                "engine": self.engine.status() if self.engine is not None else None}
