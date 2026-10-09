"""Local OpenAI-compatible gateway (127.0.0.1:8400).

One request = ONE round trip: k peers of different families (most reliable first) get the same job
at once through the tracker relay, then the gateway fuses what comes back: weighted vote when a final
answer can be extracted (math, multiple choice), medoid otherwise. It answers as soon as the stop
certificate holds, cancels the stragglers, and pays with signed receipts.

Peer selection. With an essaim/1.1 tracker (feature "route"), the gateway sends its k jobs without a
target and the tracker picks the peers (distinct families, idle and reliable nodes first) from its
indexes; an Assigned frame names each peer before anything else about its job arrives. With an
essaim/1 tracker, the gateway downloads the directory (cached `peers_ttl_s`) and picks the peers itself.

Replacement. A peer that refuses the job (busy, paused, gone) or fails fast is replaced once, if at
least a quarter of the request's time is left: the replacement job goes to a family not used yet by
the request, else to the failed peer's family, never to a node already asked.

Stop certificate with replacements. The certificate is computed over every job still pending,
replacements included: a pending job counts with its peer's weight (whether or not it will answer).
A routed job's weight is only known once its Assigned frame has arrived, so the certificate is not
evaluated while any pending job is still unassigned. The gateway sends no new job after it stops, so
the decision is exactly the weighted vote that all the peers asked (the refused or failed ones
abstaining, the replacements included) would have produced.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from . import PROTOCOL, __version__
from .crypto import pubkey_matches
from .fusion import (FORMAT_INSTRUCTIONS, count_options, detect_task_hint, extract_answer, medoid,
                     reliability_weight, stop_certificate, weighted_vote)
from .node import NodeClient, http_url
from .priors import collision_prob, prior_accuracy
from .protocol import (MAX_DEADLINE_MS, MAX_PROMPT_CHARS, MAX_TOKENS, Assigned, ChatMessage, Job, JobError, Receipt,
                       ResultFrame, Route)

log = logging.getLogger("myriad.gateway")
MAX_K = 8
LATE_GRACE_S = 5.0
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
# Errors after which a peer is NOT replaced: the request itself is at fault, there is no peer to be
# had, or the time is up.
NO_REPLACEMENT = frozenset({"insufficient_credits", "too_many_jobs", "bad_signature", "requester_mismatch",
                            "duplicate_job", "prompt_too_large", "no_target", "no_peer", "deadline",
                            "tracker_disconnected", "bad_job_signature", "unassigned"})
REPLACE_MIN_LEFT = 0.25  # share of the request's time that must be left to ask for a replacement
# The swarm's model id in the OpenAI-compatible API, and the name of the extension field (request:
# options; response: peers, answers, weights, decision). "essaim" (the package's former name) is kept
# as a deprecated alias of both, so that existing clients keep working.
SWARM_MODEL = "myriad"
LEGACY_SWARM_MODEL = "essaim"
SWARM_MODELS = frozenset({SWARM_MODEL, LEGACY_SWARM_MODEL})


class GatewayError(Exception):
    def __init__(self, status: int, message: str, code: str = "essaim_error"):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


@dataclass
class PeerOutcome:
    peer: dict | None  # None while a routed job waits for its Assigned frame (or got no peer)
    job_id: str
    weight: float | None  # None while the peer is unknown
    status: str = "en attente"  # en attente, ok, erreur, annulé
    text: str | None = None
    answer: str | None = None
    mean_logprob: float | None = None
    completion_tokens: int = 0
    compute_ms: float | None = None
    latency_ms: float | None = None
    finish_reason: str | None = None
    error: str | None = None
    result: object = None
    job: object = None
    replaces: str | None = None  # job id of the refused or failed job this one replaces
    replaced_by: str | None = None
    exclude: list = field(default_factory=list)  # routed replacement: nodes the tracker must not choose

    def public(self, chosen: bool) -> dict:
        p = self.peer
        return {"node_id": p["node_id"], "model": p["model"], "family": p.get("family"),
                "weight": round(self.weight or 0.0, 4), "status": self.status, "answer": self.answer,
                "mean_logprob": self.mean_logprob, "completion_tokens": self.completion_tokens,
                "compute_ms": self.compute_ms, "latency_ms": self.latency_ms, "error": self.error, "chosen": chosen,
                "replaces": self.replaces}


@dataclass
class _Request:
    """What a replacement needs to know about the request it belongs to."""
    routed: bool
    group: str
    model: str | None
    collision: float
    deadline_at: float  # monotonic time at which the request's own deadline ends
    timeout: float
    make_job: object
    peers: list | None = None  # directory (essaim/1 tracker only)
    rel: dict | None = None


@dataclass
class SwarmAnswer:
    text: str
    finish_reason: str | None
    completion_tokens: int
    meta: dict = field(default_factory=dict)


def normalize_messages(raw) -> list[dict]:
    """OpenAI messages -> [{'role', 'content': str}]; text parts are joined, other parts refused."""
    if not isinstance(raw, list) or not raw:
        raise GatewayError(400, "messages doit être une liste non vide", "invalid_request_error")
    out = []
    for m in raw:
        if not isinstance(m, dict):
            raise GatewayError(400, "message invalide", "invalid_request_error")
        role = {"developer": "system"}.get(m.get("role"), m.get("role"))
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if not isinstance(p, dict) or p.get("type") != "text":
                    raise GatewayError(400, "seul le texte est accepté", "invalid_request_error")
                parts.append(str(p.get("text", "")))
            content = "\n".join(parts)
        if content is None:
            content = ""
        try:
            out.append(ChatMessage(role=role, content=content).model_dump())
        except ValidationError as e:
            raise GatewayError(400, f"message invalide : {e.errors()[0]['msg']}", "invalid_request_error") from e
    return out


def with_instruction(messages: list[dict], hint: str) -> list[dict]:
    """Append the answer-format sentence to the last user message (as in the phase-0 measurements)."""
    instr = FORMAT_INSTRUCTIONS.get(hint)
    if not instr:
        return messages
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m["role"] == "user":
            if instr not in m["content"]:
                m["content"] = f"{m['content']}\n\n{instr}"
            break
    return out


def peer_reliability(p: dict, reliability: dict[str, dict]) -> float:
    r = reliability.get(p["model"])
    return float(r["p"]) if r and r.get("p") is not None else prior_accuracy(p["model"], p.get("family"))


def select_peers(peers: list[dict], reliability: dict[str, dict], k: int, model: str | None = None,
                 rng: random.Random | None = None, exclude=(), exclude_families=(),
                 only_family: str | None = None) -> list[dict]:
    """Best available node of each family, families ranked by model reliability; at most k (directory
    mode, for an essaim/1 tracker). `exclude`: node ids never chosen; `exclude_families`: families
    already used; `only_family`: restrict to that family."""
    rng = rng or random.Random()
    exclude, exclude_families = set(exclude), set(exclude_families)

    def fam_of(p: dict) -> str:
        return p.get("family") or p["model"]

    cands = [p for p in peers if p.get("model") and p.get("accepting") and p.get("reputation", 1.0) >= 0.3
             and not p.get("suspended") and p.get("busy", 0) < max(1, p.get("max_parallel", 1))
             and p.get("node_id") not in exclude and fam_of(p) not in exclude_families
             and (only_family is None or fam_of(p) == only_family)]
    if model:
        cands = [p for p in cands if p["model"] == model]
    keyed = sorted(((peer_reliability(p, reliability), p.get("reputation", 1.0), -p.get("busy", 0), rng.random()), p)
                   for p in cands)
    best: dict[str, tuple] = {}
    for key, p in reversed(keyed):
        fam = fam_of(p)
        if fam not in best:
            best[fam] = (key, p)
    ranked = sorted(best.values(), key=lambda kp: kp[0], reverse=True)
    return [p for _, p in ranked[:k]]


class Gateway:
    def __init__(self, node: NodeClient, default_k: int = 4, timeout_s: float = 120.0, peers_ttl_s: float = 2.0,
                 http: httpx.AsyncClient | None = None, early_stop: bool = True, routing: str = "auto",
                 replace: bool = True):
        if routing not in ("auto", "tracker", "directory"):
            raise ValueError("routing must be auto, tracker or directory")
        self.node = node
        self.default_k, self.timeout_s, self.peers_ttl_s = default_k, timeout_s, peers_ttl_s
        self.early_stop = early_stop  # False: always wait for every peer (ablation of the stop certificate)
        # auto: the tracker picks the peers when it supports it (essaim/1.1), else the directory is used
        self.routing = routing
        self.replace = replace  # False: refused or failed peers are not replaced (ablation)
        self.http = http or httpx.AsyncClient(base_url=http_url(node.tracker_url), timeout=10)
        self._dir: tuple[float, list, dict] | None = None
        self._feat: tuple[int, frozenset] | None = None
        self._bg: set[asyncio.Task] = set()
        self.rng = random.Random()
        self.history: list[dict] = []
        self.app = self._make_app()

    async def close(self) -> None:
        for t in list(self._bg):
            t.cancel()
        await self.http.aclose()

    async def directory(self, fresh: bool = False) -> tuple[list[dict], dict]:
        now = time.monotonic()
        if not fresh and self._dir and now - self._dir[0] < self.peers_ttl_s:
            return self._dir[1], self._dir[2]
        try:
            peers = (await self.http.get("/v1/peers")).raise_for_status().json()["peers"]
            rel = (await self.http.get("/v1/reliability")).raise_for_status().json()["models"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise GatewayError(503, f"traqueur injoignable : {e}", "tracker_unavailable") from e
        self._dir = (now, peers, rel)
        return peers, rel

    async def features(self) -> frozenset:
        """Features announced by the tracker (GET /v1/health), checked again after each reconnection."""
        session = getattr(self.node, "session", 0)
        if self._feat is not None and self._feat[0] == session:
            return self._feat[1]
        try:
            h = (await self.http.get("/v1/health")).raise_for_status().json()
        except (httpx.HTTPError, ValueError) as e:
            raise GatewayError(503, f"traqueur injoignable : {e}", "tracker_unavailable") from e
        feats = h.get("features") if isinstance(h, dict) else None
        feats = frozenset(x for x in feats if isinstance(x, str)) if isinstance(feats, list) else frozenset()
        self._feat = (session, feats)
        return feats

    async def routed(self) -> bool:
        if self.routing == "directory":
            return False
        return self.routing == "tracker" or "route" in await self.features()

    async def balance(self) -> float | None:
        try:
            r = await self.http.get(f"/v1/balance/{self.node.node_id}")
            return r.json()["balance"] if r.status_code == 200 else None
        except (httpx.HTTPError, KeyError, ValueError):
            return None

    async def models(self) -> list[str]:
        peers, _ = await self.directory()
        return sorted({p["model"] for p in peers if p.get("model")})

    # ---------- core ----------
    async def ask(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0,
                  seed: int | None = None, k: int | None = None, task_hint: str | None = None,
                  model: str | None = None, format_instruction: bool = True,
                  timeout_s: float | None = None, early_stop: bool | None = None) -> SwarmAnswer:
        if not self.node.connected.is_set():
            raise GatewayError(503, "le nœud n'est pas connecté au traqueur", "not_connected")
        t0 = time.perf_counter()
        hint = task_hint or detect_task_hint(messages)
        if hint not in ("math", "mc", "free"):
            raise GatewayError(400, "task_hint doit valoir math, mc ou free", "invalid_request_error")
        sent = with_instruction(messages, hint) if (format_instruction and hint in ("math", "mc")) else messages
        k = max(1, min(int(k or self.default_k), MAX_K))
        if model:
            k = 1
        timeout = max(1.0, min(float(timeout_s or self.timeout_s), MAX_DEADLINE_MS / 1000))
        if sum(len(m["content"]) for m in sent) > MAX_PROMPT_CHARS:
            raise GatewayError(400, "requête trop longue", "invalid_request_error")
        collision = collision_prob(hint, count_options(messages) if hint == "mc" else None)
        routed = await self.routed()

        def make_job(deadline_s: float) -> Job:
            try:
                return Job(job_id=uuid.uuid4().hex, requester_id=self.node.node_id, messages=sent,
                           max_tokens=max(1, min(int(max_tokens), MAX_TOKENS)), temperature=temperature,
                           seed=seed if seed is not None else self.rng.randrange(2**31 - 1),
                           deadline_ms=max(100, int(deadline_s * 1000)), task_hint=hint).signed_by(self.node.identity)
            except ValidationError as e:
                raise GatewayError(400, f"requête invalide : {e.errors()[0]['msg']}", "invalid_request_error") from e

        req = _Request(routed=routed, group=uuid.uuid4().hex, model=model, collision=collision,
                       deadline_at=time.monotonic() + timeout, timeout=timeout, make_job=make_job)
        outcomes: dict[str, PeerOutcome] = {}
        if routed:  # the tracker picks the k peers (no directory download)
            for _ in range(k):
                job = make_job(timeout)
                outcomes[job.job_id] = PeerOutcome(peer=None, job_id=job.job_id, weight=None, job=job)
        else:  # essaim/1 tracker: pick from the directory
            req.peers, req.rel = await self.directory()
            chosen = select_peers(req.peers, req.rel, k, model, self.rng)
            if not chosen:
                raise GatewayError(503, "aucun pair disponible pour cette requête", "no_peers")
            for p in chosen:
                job = make_job(timeout)
                outcomes[job.job_id] = PeerOutcome(peer=p, job_id=job.job_id, job=job,
                                                   weight=reliability_weight(peer_reliability(p, req.rel), collision))
        queue: asyncio.Queue = asyncio.Queue()
        self.node.register(outcomes, queue)
        pending = set(outcomes)
        vote_mode = hint in ("math", "mc") and not model
        stop_early = self.early_stop if early_stop is None else bool(early_stop)
        early = False
        end = time.monotonic() + timeout + 2.0
        answer: SwarmAnswer | None = None
        try:
            try:
                for o in list(outcomes.values()):  # every job leaves at once: one round trip
                    await self._send(o, req)
            except Exception as e:
                raise GatewayError(503, f"envoi impossible : {e}", "not_connected") from e
            while pending:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    ev = await asyncio.wait_for(queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if isinstance(ev, Assigned):
                    o = outcomes.get(ev.job_id)
                    if o is None or o.peer is not None or o.status != "en attente":
                        continue
                    o.peer = ev.peer.model_dump(mode="json")
                    o.weight = reliability_weight(ev.peer.reliability, collision)
                else:
                    jid = self._absorb(ev, outcomes, hint, t0)
                    if jid is None or jid not in pending:
                        continue
                    pending.discard(jid)
                    if outcomes[jid].status == "erreur":
                        new = await self._replacement(outcomes[jid], outcomes, req, queue)
                        if new is not None:
                            pending.add(new.job_id)
                # Certificate over every pending job, replacements included, once all their weights are known.
                if (vote_mode and pending and stop_early
                        and all(outcomes[j].weight is not None for j in pending)):
                    partial: dict[str, float] = {}
                    for o in outcomes.values():
                        if o.status == "ok" and o.answer is not None:
                            partial[o.answer] = partial.get(o.answer, 0.0) + o.weight
                    if stop_certificate(partial, sum(outcomes[j].weight for j in pending)) is not None:
                        early = True
                        break
            for jid in pending:
                outcomes[jid].status = "annulé" if early else "sans réponse"
            answer = self._decide(outcomes, hint, vote_mode, bool(model), pending, early, t0, routed)
        finally:
            # Always, even on error or cancellation: cancel what is still running, pay what was
            # received, and drop the job registrations.
            for jid in pending:  # error or cancellation path: a result racing the cancel is still paid
                if outcomes[jid].status == "en attente":
                    outcomes[jid].status = "annulé"
            fused = answer.meta.get("answer") if answer is not None else None
            judged = bool(answer.meta.pop("judged", False)) if answer is not None else False
            task = asyncio.create_task(self._finish(outcomes, pending, queue, fused, judged, hint, t0, end))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)
        self.history.insert(0, {"ts": time.time(), "decision": answer.meta["decision"], "answer": answer.meta["answer"],
                                "latency_ms": answer.meta["latency_ms"], "peers": answer.meta["peers_answered"]})
        del self.history[50:]
        return answer

    async def _send(self, o: PeerOutcome, req: _Request) -> None:
        if req.routed:
            await self.node.route(o.job, Route(group=req.group, model=req.model, exclude=o.exclude,
                                               replaces=o.replaces))
        else:
            await self.node.submit(o.peer["node_id"], o.job)

    async def _replacement(self, failed: PeerOutcome, outcomes: dict[str, PeerOutcome], req: _Request,
                           queue: asyncio.Queue) -> PeerOutcome | None:
        """Replace, once, a peer that refused the job or failed fast, while enough time is left: a
        family not used yet by the request first, else the failed peer's family; never a node already
        asked. The new job is registered before it is sent (its Assigned frame must not be missed)."""
        if (not self.replace or failed.peer is None or failed.replaces is not None or failed.replaced_by is not None
                or (failed.error or "") in NO_REPLACEMENT):
            return None
        left = req.deadline_at - time.monotonic()
        if left < max(1.0, REPLACE_MIN_LEFT * req.timeout):
            return None
        used = [o.peer["node_id"] for o in outcomes.values() if o.peer is not None]
        if req.routed:
            new = PeerOutcome(peer=None, job_id="", weight=None, replaces=failed.job_id, exclude=used[-64:])
        else:
            fams = set() if req.model else {o.peer.get("family") or o.peer["model"]
                                            for o in outcomes.values() if o.peer is not None}
            cand = select_peers(req.peers, req.rel, 1, req.model, self.rng, exclude=used, exclude_families=fams)
            if not cand and not req.model:
                cand = select_peers(req.peers, req.rel, 1, None, self.rng, exclude=used,
                                    only_family=failed.peer.get("family") or failed.peer["model"])
            if not cand:
                return None
            new = PeerOutcome(peer=cand[0], job_id="", replaces=failed.job_id,
                              weight=reliability_weight(peer_reliability(cand[0], req.rel), req.collision))
        new.job = req.make_job(left)
        new.job_id = new.job.job_id
        failed.replaced_by = new.job_id
        outcomes[new.job_id] = new
        self.node.register([new.job_id], queue)
        try:
            await self._send(new, req)
        except Exception as e:  # noqa: BLE001 - the connection dropped: the replacement simply fails
            new.status, new.error = "erreur", f"envoi impossible : {e}"[:200]
            return None
        return new

    def _absorb(self, ev, outcomes: dict[str, PeerOutcome], hint: str, t0: float,
                accept: tuple[str, ...] = ("en attente",)) -> str | None:
        """Record one event; a result whose signature does not check out counts as an error."""
        if isinstance(ev, ResultFrame):
            res = ev.result
            o = outcomes.get(res.job_id)
            if o is None or o.status not in accept:
                return None
            p = o.peer
            if p is None:  # cannot happen with an honest tracker (Assigned comes first): unverifiable
                o.status, o.error = "erreur", "unassigned"
                return res.job_id
            if (res.node_id != p["node_id"] or res.model != p["model"] or not pubkey_matches(p["node_id"], p["pubkey"])
                    or not res.verify(p["pubkey"])):
                o.status, o.error = "erreur", "bad_signature"
                return res.job_id
            o.status, o.text, o.result = "ok", res.text, res
            o.answer = extract_answer(res.text, hint)
            o.mean_logprob, o.completion_tokens = res.mean_logprob, res.completion_tokens
            o.compute_ms, o.finish_reason = res.compute_ms, res.finish_reason
            o.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return res.job_id
        if isinstance(ev, JobError):
            o = outcomes.get(ev.job_id)
            if o is None or o.status != "en attente":
                return None
            o.status, o.error = "erreur", ev.error
            o.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return ev.job_id
        return None

    def _decide(self, outcomes: dict[str, PeerOutcome], hint: str, vote_mode: bool, single: bool,
                pending: set[str], early: bool, t0: float, routed: bool = False) -> SwarmAnswer:
        order = list(outcomes.values())
        asked = [o for o in order if o.peer is not None]  # jobs that never got a peer are not listed
        ok = [o for o in asked if o.status == "ok"]
        if not ok:
            if not asked and order and all(o.error == "no_peer" for o in order):
                raise GatewayError(503, "aucun pair disponible pour cette requête", "no_peers")
            errs = "; ".join(f"{o.peer['model'] if o.peer else 'aucun pair'}: {o.error or o.status}" for o in order)
            code = "insufficient_credits" if any(o.error == "insufficient_credits" for o in order) else "no_answer"
            raise GatewayError(402 if code == "insufficient_credits" else 502,
                               f"aucun pair n'a répondu ({errs})", code)
        decision, fused, rep, scores, certified = "medoid", None, None, {}, False
        if single:
            decision, rep = "single", ok[0]
        elif vote_mode:
            vote = weighted_vote([o.answer for o in ok], [o.weight for o in ok], [o.mean_logprob for o in ok])
            if vote.answer is not None:
                decision, fused, rep, scores = "vote", vote.answer, ok[vote.representative], vote.scores
                if all(outcomes[j].weight is not None for j in pending):  # an unknown weight certifies nothing
                    certified = stop_certificate(scores, sum(outcomes[j].weight for j in pending)) is not None
        if rep is None:
            idx = medoid([o.text for o in ok], [o.weight for o in ok])
            rep = ok[idx if idx is not None else 0]
        judged = decision == "vote" and certified and sum(1 for o in ok if o.answer is not None) >= 2
        meta = {"protocol": PROTOCOL, "task_hint": hint, "decision": decision, "answer": fused,
                "scores": {a: round(s, 4) for a, s in scores.items()}, "certificate": certified, "early_stop": early,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "peers_asked": len(asked),
                "peers_answered": len(ok), "peers": [o.public(o is rep) for o in asked],
                "replacements": sum(1 for o in asked if o.replaces is not None),
                "routing": "tracker" if routed else "directory",
                "total_completion_tokens": sum(o.completion_tokens for o in ok), "judged": judged}
        return SwarmAnswer(text=rep.text or "", finish_reason=rep.finish_reason or "stop",
                           completion_tokens=rep.completion_tokens, meta=meta)

    async def _finish(self, outcomes: dict[str, PeerOutcome], pending: set[str], queue: asyncio.Queue,
                      fused: str | None, judged: bool, hint: str, t0: float, end: float) -> None:
        """After the answer: cancel the stragglers, pay every result received (late ones too) with a
        signed receipt, and report whether each answer agreed with the decision."""
        try:
            for jid in pending:
                try:
                    await self.node.cancel(jid)
                except Exception:
                    pass

            async def pay(o: PeerOutcome) -> None:
                agreed = (o.answer == fused) if judged else None
                rc = Receipt(job_id=o.job_id, requester_id=self.node.node_id, node_id=o.peer["node_id"],
                             completion_tokens=o.result.completion_tokens, model=o.result.model, agreed=agreed)
                try:
                    await self.node.send_receipt(rc)
                except Exception as e:
                    log.warning("receipt not sent: %s", e)

            for o in outcomes.values():
                if o.status == "ok":
                    await pay(o)
            late_end = min(end, time.monotonic() + LATE_GRACE_S)
            waiting = set(pending)
            while waiting:  # a result racing the cancellation is still paid for
                remaining = late_end - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    ev = await asyncio.wait_for(queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if isinstance(ev, Assigned):  # a late assignment: needed to verify a late result
                    o = outcomes.get(ev.job_id)
                    if o is not None and o.peer is None:
                        o.peer = ev.peer.model_dump(mode="json")
                    continue
                if not isinstance(ev, ResultFrame) or ev.result.job_id not in waiting:
                    continue
                jid = self._absorb(ev, outcomes, hint, t0, accept=("annulé", "sans réponse"))
                if jid is not None:
                    waiting.discard(jid)
                    if outcomes[jid].status == "ok":
                        await pay(outcomes[jid])
        finally:
            self.node.unregister(outcomes)

    # ---------- HTTP ----------
    def _make_app(self) -> FastAPI:
        app = FastAPI(title="myriad gateway", version=__version__)

        @app.middleware("http")
        async def local_only(request: Request, call_next):
            bad = check_local(request)
            if bad:
                return JSONResponse({"error": {"message": bad, "type": "forbidden"}}, status_code=403)
            return await call_next(request)

        @app.get("/v1/models")
        async def list_models():
            try:
                names = await self.models()
            except GatewayError:
                names = []
            data = [{"id": SWARM_MODEL, "object": "model", "created": 0, "owned_by": "myriad"},
                    {"id": LEGACY_SWARM_MODEL, "object": "model", "created": 0, "owned_by": "myriad"}]  # deprecated
            data += [{"id": m, "object": "model", "created": 0, "owned_by": "myriad-peer"} for m in names]
            return {"object": "list", "data": data}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            try:
                body = await read_json(request)
                stream = bool(body.get("stream", False))
                result = await self.ask(**chat_args(body))
            except GatewayError as e:
                return JSONResponse({"error": {"message": e.message, "type": e.code}}, status_code=e.status)
            model = body.get("model") or SWARM_MODEL
            if stream:
                return StreamingResponse(sse_chunks(result, model), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache"})
            return completion_body(result, model)

        @app.get("/v1/myriad/status")
        @app.get("/v1/essaim/status")  # deprecated alias
        async def status():
            return {"node": self.node.status(), "balance": await self.balance(), "history": self.history[:20]}

        return app


def check_local(request: Request) -> str | None:
    """Refuse DNS-rebinding (Host) and cross-site browser requests (Origin) on the local servers."""
    host = request.headers.get("host", "")
    hostname = host[1:].split("]")[0] if host.startswith("[") else host.rsplit(":", 1)[0]
    if hostname not in LOCAL_HOSTS:
        return "hôte non autorisé"
    origin = request.headers.get("origin")
    if origin and origin != "null":
        from urllib.parse import urlsplit
        if (urlsplit(origin).hostname or "") not in LOCAL_HOSTS:
            return "origine non autorisée"
    elif origin == "null":
        return "origine non autorisée"
    return None


async def read_json(request: Request) -> dict:
    if not request.headers.get("content-type", "").lower().startswith("application/json"):
        raise GatewayError(415, "Content-Type application/json attendu", "invalid_request_error")
    raw = await request.body()
    if len(raw) > 1 << 20:
        raise GatewayError(413, "requête trop grande", "invalid_request_error")
    try:
        body = json.loads(raw)
    except ValueError as e:
        raise GatewayError(400, "JSON invalide", "invalid_request_error") from e
    if not isinstance(body, dict):
        raise GatewayError(400, "objet JSON attendu", "invalid_request_error")
    return body


def chat_args(body: dict) -> dict:
    """OpenAI request body -> Gateway.ask arguments. `model`: 'myriad' (the swarm; 'essaim' is a deprecated
    alias) or one network model. Options in the `myriad` field (or the deprecated `essaim` field)."""
    field_name = SWARM_MODEL if body.get(SWARM_MODEL) is not None else LEGACY_SWARM_MODEL
    ext = body.get(field_name) or {}
    if not isinstance(ext, dict):
        raise GatewayError(400, f"le champ {field_name} doit être un objet", "invalid_request_error")
    model = body.get("model")
    try:
        temperature = float(body.get("temperature") if body.get("temperature") is not None else 0.0)
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 512)
        seed = body.get("seed")
        seed = None if seed is None else int(seed) % (2**31 - 1)
        k = ext.get("k")
        k = None if k is None else int(k)
        timeout_s = ext.get("timeout_s")
        timeout_s = None if timeout_s is None else max(1.0, min(float(timeout_s), 600.0))
    except (TypeError, ValueError) as e:
        raise GatewayError(400, f"paramètre invalide : {e}", "invalid_request_error") from e
    if not (0.0 <= temperature <= 2.0):
        raise GatewayError(400, "temperature hors de [0, 2]", "invalid_request_error")
    early_stop = ext.get("early_stop")
    if early_stop is not None and not isinstance(early_stop, bool):
        raise GatewayError(400, f"{field_name}.early_stop doit être un booléen", "invalid_request_error")
    return {"messages": normalize_messages(body.get("messages")), "max_tokens": max_tokens, "temperature": temperature,
            "seed": seed, "k": k, "task_hint": ext.get("task_hint"),
            "model": model if model and model not in SWARM_MODELS else None,
            "format_instruction": bool(ext.get("format_instruction", True)), "timeout_s": timeout_s,
            "early_stop": early_stop}


def completion_body(ans: SwarmAnswer, model: str) -> dict:
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": ans.text},
                                         "finish_reason": ans.finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": ans.completion_tokens,
                      "total_tokens": ans.completion_tokens},
            **swarm_meta(ans)}


def swarm_meta(ans: SwarmAnswer) -> dict:
    """The swarm's metadata under `myriad`, and under `essaim` (deprecated alias, same content)."""
    return {SWARM_MODEL: ans.meta, LEGACY_SWARM_MODEL: ans.meta}


async def sse_chunks(ans: SwarmAnswer, model: str, size: int = 200):
    """Emulated streaming: the fused answer is complete, it is sent in a few chunks."""
    cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

    def chunk(delta: dict, finish=None, extra: dict | None = None) -> str:
        body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **(extra or {})}
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant"})
    for i in range(0, len(ans.text), size):
        yield chunk({"content": ans.text[i:i + size]})
    yield chunk({}, ans.finish_reason, swarm_meta(ans))
    yield "data: [DONE]\n\n"
