"""Node: one outbound WebSocket to the tracker (works behind a NAT box), challenge-response login,
execution of the jobs it receives within the user's limits, signed results. The same connection
carries the jobs this node's gateway sends to other peers."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from collections import deque
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from . import PROTOCOL
from .config import within_hours
from .crypto import Identity, pubkey_matches
from .protocol import (MAX_FRAME_BYTES, MAX_TEXT_CHARS, MAX_TOKENS, Assigned, Cancel, Challenge, ErrorFrame, Hello,
                       Job, JobError, JobFrame, JobResult, NodeInfo, Ping, Pong, ReceiptFrame, ResultFrame, Route,
                       Status, Welcome, dump_frame, parse_frame)

log = logging.getLogger("myriad.node")
HEARTBEAT_S = 15.0
PROBE_TIMEOUT_S = 2.0  # engine health probe before answering a ping (the tracker waits 5 s by default)


class _TagsRefused(ValueError):
    """An older tracker (before essaim/1.2) refused a Hello carrying skill tags."""


def ws_url(tracker_url: str) -> str:
    """http://host:port -> ws://host:port/v1/ws (https -> wss)."""
    u = urlsplit(tracker_url.rstrip("/"))
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(u.scheme)
    if scheme is None:
        raise ValueError(f"URL de traqueur invalide : {tracker_url}")
    path = u.path if u.path.endswith("/v1/ws") else (u.path + "/v1/ws")
    return urlunsplit((scheme, u.netloc, path, "", ""))


def http_url(tracker_url: str) -> str:
    u = urlsplit(tracker_url.rstrip("/"))
    scheme = {"ws": "http", "wss": "https"}.get(u.scheme, u.scheme)
    path = u.path[: -len("/v1/ws")] if u.path.endswith("/v1/ws") else u.path
    return urlunsplit((scheme, u.netloc, path, "", ""))


class NodeClient:
    def __init__(self, identity: Identity, tracker_url: str, engine=None, model: str | None = None,
                 family: str | None = None, gguf: str | None = None, params_b: float | None = None, ctx: int = 0,
                 max_parallel: int = 1, accepting: bool = True, active_hours: str | None = None,
                 max_job_tokens: int = 1024, reconnect: bool = True, tags: list[str] | None = None):
        self.identity = identity
        # Skill tags (essaim/1.2), advertised in the Hello. An older tracker refuses a Hello carrying
        # them: the node then reconnects without tags to that tracker (`_tags_refused`).
        self.tags = list(tags) if tags else None
        self._tags_refused = False
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
        self.stats = {"served": 0, "failed": 0, "cancelled": 0, "rejected": 0, "tokens": 0}
        self._ws = None
        self._send_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self.session = 0  # number of tracker sessions opened so far (a new one may be another tracker version)
        self._last_ping = -1e9  # monotonic time of the last essaim/1.1 ping received
        self._pongs: set[asyncio.Task] = set()
        self._observers: set = set()  # callables (direction, frame), e.g. the UI's live chat view

    # ---------- limits ----------
    def serving(self) -> bool:
        return bool(self.model) and self.engine is not None

    def accepting_now(self) -> bool:
        engine_ready = self.engine is not None and getattr(self.engine, "state", "prêt") == "prêt"
        return (self.serving() and engine_ready and self.accepting and self.max_parallel > 0
                and within_hours(self.active_hours, dt.datetime.now().hour))

    def info(self) -> NodeInfo:
        return NodeInfo(node_id=self.node_id, pubkey=self.identity.pubkey, model=self.model, family=self.family,
                        gguf=self.gguf, params_b=self.params_b, ctx=self.ctx, max_parallel=self.max_parallel,
                        version=PROTOCOL, accepting=self.accepting_now(),
                        tags=self.tags if self.tags and self.model and not self._tags_refused else None)

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
        backoff = 1.0
        while not self._stop.is_set():
            self.state = "connexion"
            started = time.monotonic()
            try:
                try:
                    await self._session()
                except _TagsRefused:  # an older tracker: at once again, without the tags
                    self._tags_refused = True
                    log.info("tracker does not accept skill tags: connecting without them")
                    self._on_disconnect()
                    self.state = "connexion"
                    await self._session()
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
        async with connect(ws_url(self.tracker_url), max_size=MAX_FRAME_BYTES, open_timeout=10,
                           ping_interval=20, ping_timeout=20) as ws:
            ch = parse_frame(await asyncio.wait_for(ws.recv(), 10))
            if not isinstance(ch, Challenge) or not pubkey_matches(ch.tracker_id, ch.tracker_pubkey):
                raise ValueError("défi du traqueur invalide")
            info = self.info()
            await ws.send(dump_frame(Hello.make(self.identity, info, ch.nonce)))
            first = parse_frame(await asyncio.wait_for(ws.recv(), 10))
            if isinstance(first, ErrorFrame):
                if info.tags is not None and first.error == "hello_expected":  # unknown field `tags`
                    raise _TagsRefused(first.error)
                raise ValueError(f"refusé par le traqueur : {first.error}")
            if not isinstance(first, Welcome) or first.node_id != self.node_id:
                raise ValueError("réponse inattendue du traqueur")
            self.tracker_id, self.balance = ch.tracker_id, first.balance
            self._ws = ws
            self.session += 1
            self._last_ping = -1e9
            self.state, self.last_error = "connecté", None
            self.connected.set()
            log.info("connected to tracker as %s", self.node_id[:8])
            beat = asyncio.create_task(self._heartbeat())
            try:
                async for raw in ws:
                    try:
                        frame = parse_frame(raw)
                    except (ValueError, ValidationError) as e:
                        log.warning("invalid frame from tracker: %s", e)
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
            if time.monotonic() - self._last_ping < 2 * HEARTBEAT_S:
                continue  # an essaim/1.1 tracker pings us: our Pongs carry the same information
            try:
                await self.send_status()
            except Exception:
                return

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
                log.debug("observer failed: %s", e)

    async def _dispatch(self, frame) -> None:
        self._observe("in", frame)
        if isinstance(frame, JobFrame):
            await self._on_job(frame)
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
        elif isinstance(frame, (ResultFrame, JobError, Assigned)):
            job_id = frame.result.job_id if isinstance(frame, ResultFrame) else frame.job_id
            q = self.waiters.get(job_id)
            if q is not None:
                q.put_nowait(frame)
        elif isinstance(frame, ErrorFrame):
            self.last_error = frame.error
            log.warning("tracker error: %s", frame.error)

    # ---------- serving ----------
    async def _reject(self, job: Job, error: str) -> None:
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
        self.running[job.job_id] = asyncio.create_task(self._execute(job))

    async def _execute(self, job: Job) -> None:
        t0 = time.monotonic()
        entry = {"job_id": job.job_id, "requester": job.requester_id[:8], "ts": time.time(), "status": "en cours"}
        self.recent.appendleft(entry)
        timeout = job.deadline_ms / 1000
        try:
            gen = await asyncio.wait_for(
                self.engine.generate([m.model_dump() for m in job.messages], min(job.max_tokens, self.max_job_tokens),
                                     job.temperature, job.seed, timeout, thinking=job.thinking), timeout)
            res = JobResult(job_id=job.job_id, node_id=self.node_id, model=self.model, text=gen.text[:MAX_TEXT_CHARS],
                            finish_reason=(gen.finish_reason or None) and str(gen.finish_reason)[:200],
                            completion_tokens=max(0, min(gen.completion_tokens, MAX_TOKENS)),
                            mean_logprob=gen.mean_logprob if gen.mean_logprob is None else min(0.0, gen.mean_logprob),
                            compute_ms=round(gen.compute_ms, 1)).signed_by(self.identity)
            await self.send(ResultFrame(result=res))
            self.stats["served"] += 1
            self.stats["tokens"] += res.completion_tokens
            entry.update(status="servi", tokens=res.completion_tokens)
        except asyncio.CancelledError:
            self.stats["cancelled"] += 1
            entry["status"] = "annulé"
        except Exception as e:
            self.stats["failed"] += 1
            err = "deadline" if isinstance(e, asyncio.TimeoutError) else f"{type(e).__name__}: {e}"[:200]
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
            self.running.pop(job.job_id, None)

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

    async def cancel(self, job_id: str) -> None:
        await self.send(Cancel(job_id=job_id))

    async def send_receipt(self, receipt) -> None:
        await self.send(ReceiptFrame(receipt=receipt.signed_by(self.identity)))

    def status(self) -> dict:
        return {"node_id": self.node_id, "state": self.state, "tracker": self.tracker_url, "model": self.model,
                "family": self.family, "tags": self.info().tags, "gguf": self.gguf, "params_b": self.params_b, "serving": self.serving(),
                "accepting": self.accepting, "accepting_now": self.accepting_now(), "max_parallel": self.max_parallel,
                "active_hours": self.active_hours, "running": len(self.running), "stats": dict(self.stats),
                "recent": list(self.recent)[:20], "last_error": self.last_error,
                "engine": self.engine.status() if self.engine is not None else None}
