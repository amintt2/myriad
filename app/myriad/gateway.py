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

Privacy and protection (essaim/1.3, security.py, privacy.py, e2e.py):
- Before anything leaves the machine: the history is capped (`max_history_turns`), the privacy guard
  scans the messages, masks secrets with placeholders (restored in the answer, locally), asks for a
  first-time confirmation before sensitive content leaves unmasked, and keeps the question local when a
  rule says so (or in local-only mode): it is then answered by this machine's own model, or refused.
- End-to-end encryption (required by default): with a tracker announcing "e2e", a routed job starts
  with a Reserve frame; when the Assigned frame names the peer, the gateway checks the peer against the
  user's policy (blocklist, trusted nodes, private swarm, minimum reliability, quarantine) and its key
  certificate, and only then seals the job for that peer with a one-job pseudonym. In directory mode,
  the directory is filtered by the same policy before choosing. A peer that breaks the policy gets
  nothing (its reservation is cancelled and it is replaced). Without "e2e" on the tracker, the gateway
  refuses to send unless the user allowed plaintext.
- Local quarantine: peers that disagree with certified majorities again and again, or send answers that
  do not decrypt or verify, are excluded for a while; disagreements are also reported to the tracker
  (signed NodeReport), and peers are rotated (recently used ones avoided when possible).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from . import PROTOCOL, __version__
from .crypto import account_params, pubkey_matches
from .e2e import E2EError, kx_problem, open_result, seal_job
from .fusion import (FORMAT_INSTRUCTIONS, count_options, detect_task_hint, extract_answer, medoid,
                     reliability_weight, stop_certificate, weighted_vote)
from .node import NodeClient, http_url
from .priors import collision_prob, prior_accuracy
from .privacy import analyze, restore
from .protocol import (MAX_DEADLINE_MS, MAX_PROMPT_CHARS, MAX_TOKENS, Assigned, ChatMessage, Dispute, Job, JobError, JobFrame,
                       KxCert, NodeReport, Receipt, ResultFrame, Route, SealedResultFrame)
from .routing import PREFIXES, RouteError, RouteSpec, known_families, live_routes, parse_model, peer_family
from .security import Security

log = logging.getLogger("myriad.gateway")
MAX_K = 8
LATE_GRACE_S = 5.0
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
# Errors after which a peer is NOT replaced: the request itself is at fault, there is no peer to be
# had, or the time is up.
NO_REPLACEMENT = frozenset({"insufficient_credits", "too_many_jobs", "bad_signature", "requester_mismatch",
                            "duplicate_job", "prompt_too_large", "no_target", "no_peer", "deadline",
                            "tracker_disconnected", "bad_job_signature", "unassigned", "banned",
                            "reservation_expired", "reservation_mismatch"})
REPLACE_MIN_LEFT = 0.25  # share of the request's time that must be left to ask for a replacement
# The swarm's model id in the OpenAI-compatible API, and the name of the extension field (request:
# options; response: peers, answers, weights, decision). "essaim" (the package's former name) is kept
# as a deprecated alias of both, so that existing clients keep working.
SWARM_MODEL = "myriad"
LEGACY_SWARM_MODEL = "essaim"
SWARM_MODELS = frozenset({SWARM_MODEL, LEGACY_SWARM_MODEL})


class _PolicyError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class GatewayError(Exception):
    def __init__(self, status: int, message: str, code: str = "essaim_error", extra: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code
        self.extra = extra or {}  # e.g. the privacy summary of a question that needs a confirmation

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.code, **self.extra}}


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
    tag_match: bool | None = None  # routed by tag: does the peer advertise it (False: fallback peer)
    reserved: bool = False  # essaim/1.3: the content goes once the Assigned peer has been checked
    session: object = None  # e2e.Session of an encrypted job (None: plaintext)
    attest: str | None = None  # the peer's signature over the plaintext answer's digest

    def public(self, chosen: bool) -> dict:
        p = self.peer
        return {"node_id": p["node_id"], "model": p["model"], "family": p.get("family"),
                "weight": round(self.weight or 0.0, 4), "status": self.status, "answer": self.answer,
                "mean_logprob": self.mean_logprob, "completion_tokens": self.completion_tokens,
                "compute_ms": self.compute_ms, "latency_ms": self.latency_ms, "error": self.error, "chosen": chosen,
                "replaces": self.replaces, "tag_match": self.tag_match, "e2e": self.session is not None}


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
    tag: str | None = None  # essaim/1.2: a peer advertising this skill tag (any peer if none)
    family: str | None = None  # essaim/1.2: only this model family
    avoid: list = field(default_factory=list)  # node ids the caller asked not to use (replacements too)
    reserve: bool = False  # essaim/1.3 tracker: Reserve first, content after the peer check
    e2e: bool = False  # the tracker relays encrypted jobs
    require_e2e: bool = True
    policy: dict = field(default_factory=dict)  # Route policy fields (tracker with "policy")
    mapping: dict = field(default_factory=dict)  # privacy placeholders -> original values (local only)


@dataclass
class SwarmAnswer:
    text: str
    finish_reason: str | None
    completion_tokens: int
    meta: dict = field(default_factory=dict)
    # Every answer received ({text, node_id, model, family, weight, completion_tokens, tag_match...}):
    # the sub-agent runner verifies candidates one by one.
    candidates: list = field(default_factory=list)


def normalize_messages(raw) -> list[dict]:
    """OpenAI messages -> [{'role', 'content': str}]; text parts are joined, other parts refused."""
    if not isinstance(raw, list) or not raw:
        raise GatewayError(400, "messages doit être une liste non vide", "invalid_request_error")
    out = []
    for m in raw:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise GatewayError(400, "message invalide", "invalid_request_error")
        role = {"developer": "system"}.get(m["role"], m["role"])
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


def trim_history(messages: list[dict], max_turns: int) -> tuple[list[dict], int]:
    """Context minimisation: keep the system messages and the last `max_turns` user turns (with the
    answers between them); returns the messages and how many were dropped. 0: no cap."""
    users = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if not max_turns or len(users) <= max_turns:
        return messages, 0
    cut = users[-max_turns]
    kept = [m for i, m in enumerate(messages) if i >= cut or m["role"] == "system"]
    return kept, len(messages) - len(kept)


def peer_reliability(p: dict, reliability: dict[str, dict]) -> float:
    r = reliability.get(p["model"])
    return float(r["p"]) if r and r.get("p") is not None else prior_accuracy(p["model"], p.get("family"))


def select_peers(peers: list[dict], reliability: dict[str, dict], k: int, model: str | None = None,
                 rng: random.Random | None = None, exclude=(), exclude_families=(),
                 only_family: str | None = None, tag: str | None = None) -> list[dict]:
    """Best available node of each family, families ranked by model reliability; at most k (directory
    mode, for an essaim/1 tracker). `exclude`: node ids never chosen; `exclude_families`: families
    already used; `only_family`: restrict to that family; `tag`: only nodes advertising that tag."""
    if tag is not None:
        peers = [p for p in peers if tag in (p.get("tags") or ())]
    rng = rng or random.Random()
    exclude, exclude_families = set(exclude), set(exclude_families)

    def fam_of(p: dict) -> str:
        return peer_family(p)

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
                 replace: bool = True, security: Security | None = None):
        if routing not in ("auto", "tracker", "directory"):
            raise ValueError("routing must be auto, tracker or directory")
        self.node = node
        # The node's protection settings (shared with its serving side), defaults when none.
        self.security = security or getattr(node, "security", None) or Security()
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
        # Sub-agents (agents.py): local verification commands allowed by the user, by name.
        self.verify_commands: dict[str, list[str]] = {}
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
            r = await self.http.get(f"/v1/balance/{self.node.node_id}", params=account_params(self.node.identity))
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
                  timeout_s: float | None = None, early_stop: bool | None = None, tag: str | None = None,
                  family: str | None = None, avoid=(), confirm: bool = False, local: bool = False) -> SwarmAnswer:
        """`tag`: peers advertising this skill tag (any peer for the jobs no such peer can take: flagged
        in meta["route"]); `family`: one peer of this model family; `avoid`: node ids not to choose
        (at most 64, e.g. peers busy with the other sub-tasks of an agent run). `confirm`: the user
        agreed to send the sensitive content found this time; `local`: answer on this machine only."""
        avoid = [a for a in avoid if isinstance(a, str)][:64]
        st = self.security.settings
        messages, trimmed = trim_history(messages, st.max_history_turns)
        hint = task_hint or detect_task_hint(messages)
        if hint not in ("math", "mc", "free"):
            raise GatewayError(400, "task_hint doit valoir math, mc ou free", "invalid_request_error")
        guard = analyze(messages, self.security.privacy, confirm=confirm)
        privacy = guard.summary()
        if local or st.local_only or guard.decision == "local":
            return await self._ask_local(messages, max_tokens, temperature, seed, timeout_s, privacy,
                                         "réglage" if (local or st.local_only) else "règle de confidentialité")
        if guard.decision == "ask":
            raise GatewayError(409, "contenu sensible : confirmation requise avant de l'envoyer au réseau "
                                    f"({', '.join(guard.confirm_types)})", "confirmation_required",
                               {"privacy": privacy})
        messages = guard.messages  # secrets replaced by placeholders, restored in the answer
        if not self.node.connected.is_set():
            raise GatewayError(503, "le nœud n'est pas connecté au traqueur", "not_connected")
        t0 = time.perf_counter()
        sent = with_instruction(messages, hint) if (format_instruction and hint in ("math", "mc")) else messages
        k = max(1, min(int(k or self.default_k), MAX_K))
        if model or family:
            k = 1
        timeout = max(1.0, min(float(timeout_s or self.timeout_s), MAX_DEADLINE_MS / 1000))
        if sum(len(m["content"]) for m in sent) > MAX_PROMPT_CHARS:
            raise GatewayError(400, "requête trop longue", "invalid_request_error")
        collision = collision_prob(hint, count_options(messages) if hint == "mc" else None)
        routed = await self.routed()
        feats = await self.features()
        if routed and (tag or family) and "tags" not in feats:
            routed = False  # a tracker before essaim/1.2 cannot route by tag or family: the directory can
        e2e = "e2e" in feats
        if routed and not e2e and self.security.policy_active():
            # An older tracker relays a routed job before naming its peer: the user's policy could only
            # be checked after the question had left. Choose the peer here instead (Codex review).
            routed = False
        if st.require_e2e and not e2e:
            raise GatewayError(503, "le traqueur ne prend pas en charge le chiffrement de bout en bout ; la "
                                    "question n'est pas envoyée (réglage « Exiger le chiffrement de bout en bout »)",
                               "e2e_unavailable")

        def make_job(deadline_s: float) -> Job:
            try:
                return Job(job_id=uuid.uuid4().hex, requester_id=self.node.node_id, messages=sent,
                           max_tokens=max(1, min(int(max_tokens), MAX_TOKENS)), temperature=temperature,
                           seed=seed if seed is not None else self.rng.randrange(2**31 - 1),
                           deadline_ms=max(100, int(deadline_s * 1000)), task_hint=hint).signed_by(self.node.identity)
            except ValidationError as e:
                raise GatewayError(400, f"requête invalide : {e.errors()[0]['msg']}", "invalid_request_error") from e

        req = _Request(routed=routed, group=uuid.uuid4().hex, model=model, collision=collision,
                       deadline_at=time.monotonic() + timeout, timeout=timeout, make_job=make_job, tag=tag or None,
                       family=family or None, avoid=avoid, reserve=routed and e2e, e2e=e2e,
                       require_e2e=st.require_e2e, mapping=guard.mapping,
                       policy=self.security.route_fields() if "policy" in feats else {})
        outcomes: dict[str, PeerOutcome] = {}
        if routed:  # the tracker picks the k peers (no directory download)
            for _ in range(k):
                job = make_job(timeout)
                outcomes[job.job_id] = PeerOutcome(peer=None, job_id=job.job_id, weight=None, job=job,
                                                   exclude=list(avoid))
        else:  # essaim/1 tracker: pick from the directory
            req.peers, req.rel = await self.directory()
            req.peers = self._allowed_peers(req.peers, req.rel, req)
            recent = set(self.security.soft_avoid())
            fresh = [p for p in req.peers if p["node_id"] not in recent]
            chosen = select_peers(fresh, req.rel, k, model, self.rng, only_family=req.family, tag=req.tag,
                                  exclude=avoid)
            if len(chosen) < k:  # peer rotation is a preference: complete with the recent peers
                chosen += select_peers(req.peers, req.rel, k - len(chosen), model, self.rng, only_family=req.family,
                                       tag=req.tag, exclude=[p["node_id"] for p in chosen] + avoid,
                                       exclude_families=[peer_family(p) for p in chosen]
                                       if not (model or req.family) else ())
            if req.tag and len(chosen) < k:  # not enough peers with the tag: any peer for the rest
                chosen += select_peers(req.peers, req.rel, k - len(chosen), model, self.rng, only_family=req.family,
                                       exclude=[p["node_id"] for p in chosen] + avoid,
                                       exclude_families=[peer_family(p) for p in chosen])
            if not chosen:
                raise GatewayError(503, "aucun pair disponible pour cette requête", "no_peers")
            for p in chosen:
                job = make_job(timeout)
                outcomes[job.job_id] = PeerOutcome(peer=p, job_id=job.job_id, job=job,
                                                   weight=reliability_weight(peer_reliability(p, req.rel), collision),
                                                   tag_match=(req.tag in (p.get("tags") or ())) if req.tag else None)
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
                    o.peer = ev.peer.model_dump(mode="json", exclude_none=True)
                    o.weight = reliability_weight(ev.peer.reliability, collision)
                    o.tag_match = ev.tag_match if req.tag else None
                    why = self._duplicate(o, outcomes, req) or self._peer_problem(o.peer, ev.peer.reliability, req)
                    if why is None and o.reserved:
                        try:
                            await self._send_content(o, req)
                        except _PolicyError as e:
                            why = e.reason
                        except Exception as e:  # noqa: BLE001 - the connection dropped
                            o.status, o.error = "erreur", f"envoi impossible : {type(e).__name__}"
                            pending.discard(o.job_id)
                            continue
                    if why is not None:  # the tracker assigned a peer the user's policy refuses
                        o.status, o.error = "erreur", f"policy:{why}"
                        pending.discard(o.job_id)
                        try:
                            await self.node.cancel(o.job_id)  # frees the reservation; nothing was sent
                        except Exception:  # noqa: BLE001
                            pass
                        new = await self._replacement(o, outcomes, req, queue)
                        if new is not None:
                            pending.add(new.job_id)
                else:
                    jid = self._absorb(ev, outcomes, hint, t0, mapping=req.mapping)
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
            answer = self._decide(outcomes, hint, vote_mode, bool(model or family), pending, early, t0, routed)
            asked = [o for o in outcomes.values() if o.peer is not None]
            answer.meta["route"] = {"tag": req.tag, "family": req.family,
                                    "fallback": bool(req.tag) and any(o.tag_match is False for o in asked)}
            answer.candidates = [
                {"text": o.text or "", "job_id": o.job_id, "node_id": o.peer["node_id"], "model": o.peer["model"],
                 "family": o.peer.get("family"), "weight": o.weight or 0.0, "completion_tokens": o.completion_tokens,
                 "compute_ms": o.compute_ms, "latency_ms": o.latency_ms, "tag_match": o.tag_match}
                for o in asked if o.status == "ok"]
        finally:
            # Always, even on error or cancellation: cancel what is still running, pay what was
            # received, and drop the job registrations.
            for jid in pending:  # error or cancellation path: a result racing the cancel is still paid
                if outcomes[jid].status == "en attente":
                    outcomes[jid].status = "annulé"
            fused = answer.meta.get("answer") if answer is not None else None
            judged = bool(answer.meta.pop("judged", False)) if answer is not None else False
            task = asyncio.create_task(self._finish(outcomes, pending, queue, fused, judged, hint, t0, end,
                                                    req.mapping))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)
        for o in outcomes.values():
            if o.peer is not None:
                self.security.used_peer(o.peer["node_id"])  # peer rotation
        asked = [o for o in outcomes.values() if o.peer is not None]
        answer.meta["privacy"] = {**privacy, "history_trimmed": trimmed,
                                  "e2e": bool(asked) and all(o.session is not None for o in asked),
                                  "readers": [{"node_id": o.peer["node_id"], "model": o.peer["model"],
                                               "e2e": o.session is not None,
                                               "trusted": o.peer["node_id"] in self.security.trusted_ids()}
                                              for o in asked]}
        if req.mapping:  # put the original values back, on this machine only
            m = req.mapping
            answer.text = restore(answer.text, m)
            answer.meta["answer"] = restore(answer.meta["answer"], m) if answer.meta.get("answer") else answer.meta.get("answer")
            answer.meta["scores"] = {restore(a, m): s for a, s in answer.meta.get("scores", {}).items()}
            for p in answer.meta["peers"]:
                if p.get("answer"):
                    p["answer"] = restore(p["answer"], m)
            for c in answer.candidates:
                c["text"] = restore(c["text"], m)
        self.history.insert(0, {"ts": time.time(), "decision": answer.meta["decision"], "answer": answer.meta["answer"],
                                "latency_ms": answer.meta["latency_ms"], "peers": answer.meta["peers_answered"]})
        del self.history[50:]
        return answer

    async def _send(self, o: PeerOutcome, req: _Request) -> None:
        if req.routed:
            policy = req.policy
            if o.replaces is not None and policy.get("soft_avoid"):
                # rotation is a preference for first picks: a replacement may need the recent peers
                policy = {**policy, "soft_avoid": []}
            route = Route(group=req.group, model=req.model, exclude=o.exclude, replaces=o.replaces, tag=req.tag,
                          family=req.family, **policy)
            if req.reserve:  # essaim/1.3: nothing about the content leaves before the peer is checked
                o.reserved = True
                self.node.observe_local("out", JobFrame(job=o.job, route=route))  # the local UI's view
                await self.node.reserve(o.job, route)
            else:
                await self.node.route(o.job, route)
        else:
            await self._send_content(o, req)

    async def _send_content(self, o: PeerOutcome, req: _Request) -> None:
        """The job itself, to the peer already checked: sealed for it when it has a valid key, in clear
        only if the user allows it. Raises _PolicyError when neither is possible."""
        p = o.peer
        e2e_ok = req.e2e and kx_problem(p["node_id"], p["pubkey"], p.get("kx"), model=p.get("model")) is None
        if e2e_ok:
            try:
                sj, session = seal_job(o.job, p["node_id"], KxCert.model_validate(p["kx"]))
            except E2EError as e:
                raise _PolicyError("e2e_" + e.code) from e
            o.session = session
            if not o.reserved:
                self.node.observe_local("out", JobFrame(job=o.job, target=p["node_id"]))  # the local UI's view
            await self.node.submit_sealed(sj, session.pseudonym.pubkey)
        elif not req.require_e2e:
            await self.node.submit(p["node_id"], o.job)
        else:
            raise _PolicyError("no_e2e")

    def _peer_problem(self, p: dict, reliability: float | None, req: _Request) -> str | None:
        """Why the user's policy refuses this peer (None: it may receive the job)."""
        e2e_ok = None
        if req.require_e2e:
            e2e_ok = req.e2e and kx_problem(p["node_id"], p.get("pubkey", ""), p.get("kx"), model=p.get("model")) is None
        if not pubkey_matches(p["node_id"], p.get("pubkey", "")):
            return "pubkey_mismatch"
        return self.security.check_peer(p, reliability, e2e_ok)

    @staticmethod
    def _duplicate(o: PeerOutcome, outcomes: dict, req: _Request) -> str | None:
        """One vote per peer, and distinct families for the swarm: a tracker assigning the same peer (or
        family) twice in one request would count one answer several times (Codex review)."""
        others = [x for x in outcomes.values() if x is not o and x.peer is not None]
        if any(x.peer["node_id"] == o.peer["node_id"] for x in others):
            return "duplicate_peer"
        if not (req.model or req.family):
            live = {peer_family(x.peer) for x in others if x.status != "erreur"}
            if peer_family(o.peer) in live:
                return "duplicate_family"
        return None

    def _allowed_peers(self, peers: list[dict], rel: dict, req: _Request) -> list[dict]:
        return [p for p in peers if p.get("model")
                and self._peer_problem(p, peer_reliability(p, rel), req) is None]

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
        # never a node already asked, nor one the caller asked to avoid (the tracker also excludes the
        # group's nodes itself, so the caller's list comes first within the 64 ids a Route carries)
        used = list(dict.fromkeys(req.avoid + [o.peer["node_id"] for o in outcomes.values() if o.peer is not None]))
        if req.routed:
            new = PeerOutcome(peer=None, job_id="", weight=None, replaces=failed.job_id, exclude=used[:64])
        else:
            fams = set() if (req.model or req.family) else {peer_family(o.peer)
                                                            for o in outcomes.values() if o.peer is not None}
            cand = []
            if req.tag:
                cand = select_peers(req.peers, req.rel, 1, req.model, self.rng, exclude=used, exclude_families=fams,
                                    only_family=req.family, tag=req.tag)
            if not cand:
                cand = select_peers(req.peers, req.rel, 1, req.model, self.rng, exclude=used, exclude_families=fams,
                                    only_family=req.family)
            if not cand and not req.model and not req.family:
                cand = select_peers(req.peers, req.rel, 1, None, self.rng, exclude=used,
                                    only_family=peer_family(failed.peer))
            if not cand:
                return None
            new = PeerOutcome(peer=cand[0], job_id="", replaces=failed.job_id,
                              weight=reliability_weight(peer_reliability(cand[0], req.rel), req.collision),
                              tag_match=(req.tag in (cand[0].get("tags") or ())) if req.tag else None)
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
                accept: tuple[str, ...] = ("en attente",), mapping: dict | None = None) -> str | None:
        """Record one event; a result whose signature does not check out (or that does not decrypt)
        counts as an error."""
        if isinstance(ev, (ResultFrame, SealedResultFrame)):
            res = ev.result
            o = outcomes.get(res.job_id)
            if o is None or o.status not in accept:
                return None
            p = o.peer
            if p is None:  # cannot happen with an honest tracker (Assigned comes first): unverifiable
                o.status, o.error = "erreur", "unassigned"
                return res.job_id
            if isinstance(ev, SealedResultFrame) or o.session is not None:
                if not isinstance(ev, SealedResultFrame) or o.session is None:
                    o.status, o.error = "erreur", "e2e_mismatch"  # a plaintext answer to an encrypted job
                    return res.job_id
                try:
                    if res.model != p["model"]:
                        raise E2EError("bad_signature")
                    res, o.attest = open_result(res, o.session, p["pubkey"])
                except E2EError as e:
                    o.status, o.error = "erreur", e.code
                    self.security.note(p["node_id"], "severe", "réponse indéchiffrable ou mal signée")
                    if e.code != "bad_signature" and o.session.eph_secret:
                        # contest the payment: reveals THIS job's ephemeral secret to the tracker, which
                        # checks for itself that the answer does not open (Codex review)
                        self._spawn(self.node.send(Dispute(job_id=o.job_id, eph_secret=o.session.eph_secret.hex())))
                    return ev.result.job_id
                # the local UI's view of the answer (placeholders put back): nothing is sent
                self.node.observe_local("in", ResultFrame(result=res.model_copy(
                    update={"text": restore(res.text, mapping or {})})))
            elif (res.node_id != p["node_id"] or res.model != p["model"] or not pubkey_matches(p["node_id"], p["pubkey"])
                    or not res.verify(p["pubkey"])):
                o.status, o.error = "erreur", "bad_signature"
                self.security.note(p["node_id"], "severe", "réponse mal signée")
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
                      fused: str | None, judged: bool, hint: str, t0: float, end: float,
                      mapping: dict | None = None) -> None:
        """After the answer: cancel the stragglers, pay every result received (late ones too) with a
        signed receipt, and report whether each answer agreed with the decision (to the tracker, and to
        the local quarantine; a disagreement with a certified majority is also a signed report)."""
        try:
            for jid in pending:
                try:
                    await self.node.cancel(jid)
                except Exception:
                    pass
            disagreed: list[PeerOutcome] = []

            async def pay(o: PeerOutcome) -> None:
                agreed = (o.answer == fused) if judged else None
                rc = Receipt(job_id=o.job_id, requester_id=self.node.node_id, node_id=o.peer["node_id"],
                             completion_tokens=o.result.completion_tokens, model=o.result.model, agreed=agreed)
                if agreed is not None and o.peer["node_id"] != self.node.node_id:
                    self.security.note(o.peer["node_id"], "agree" if agreed else "disagree")
                    if not agreed:
                        disagreed.append(o)
                try:
                    await self.node.send_receipt(rc)
                except Exception as e:
                    log.warning("receipt not sent: %s", type(e).__name__)

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
                        o.peer = ev.peer.model_dump(mode="json", exclude_none=True)
                    continue
                if not isinstance(ev, (ResultFrame, SealedResultFrame)) or ev.result.job_id not in waiting:
                    continue
                jid = self._absorb(ev, outcomes, hint, t0, accept=("annulé", "sans réponse"), mapping=mapping)
                if jid is not None:
                    waiting.discard(jid)
                    if outcomes[jid].status == "ok":
                        await pay(outcomes[jid])
            if disagreed and "report" in (self._feat[1] if self._feat else ()):
                for o in disagreed:  # signed, tied to a job this node paid: the tracker can weigh them
                    try:
                        await self.report(o.peer["node_id"], "disagreement", o.job_id)
                    except Exception:  # noqa: BLE001 - best effort
                        pass
        finally:
            self.node.unregister(outcomes)

    def _spawn(self, coro) -> None:
        async def quiet():
            try:
                await coro
            except Exception as e:  # noqa: BLE001 - best effort (connection gone...)
                log.debug("background send failed: %s", type(e).__name__)
        task = asyncio.create_task(quiet())
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def report(self, node_id: str, reason: str, job_id: str | None = None, note: str = "") -> dict:
        """Send a signed report about a node to the tracker (POST /v1/report)."""
        try:
            rep = NodeReport(reporter_id=self.node.node_id, node_id=node_id, reason=reason, job_id=job_id or None,
                             ts=int(time.time()), note=note[:200]).signed_by(self.node.identity)
        except ValidationError as e:
            raise GatewayError(400, f"signalement invalide : {e.errors()[0]['msg']}", "invalid_request_error") from e
        try:
            r = await self.http.post("/v1/report", content=rep.model_dump_json(),
                                     headers={"Content-Type": "application/json"})
        except httpx.HTTPError as e:
            raise GatewayError(503, f"traqueur injoignable : {type(e).__name__}", "tracker_unavailable") from e
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code != 200:
            raise GatewayError(r.status_code if 400 <= r.status_code < 500 else 502,
                               f"signalement refusé : {body.get('detail', r.status_code)}", "report_refused")
        return body

    async def _ask_local(self, messages: list[dict], max_tokens: int, temperature: float, seed: int | None,
                         timeout_s: float | None, privacy: dict, why: str) -> SwarmAnswer:
        """Answer with this machine's own model only (local-only mode, or a « toujours local » rule)."""
        eng = self.node.engine
        if eng is None or getattr(eng, "state", "prêt") != "prêt":
            raise GatewayError(409, f"question gardée sur cette machine ({why}) et aucun modèle local n'est prêt : "
                                    "rien n'a été envoyé au réseau", "kept_local", {"privacy": privacy})
        t0 = time.perf_counter()
        timeout = max(1.0, min(float(timeout_s or self.timeout_s), MAX_DEADLINE_MS / 1000))
        try:
            gen = await asyncio.wait_for(eng.generate(messages, max(1, min(int(max_tokens), MAX_TOKENS)), temperature,
                                                      seed if seed is not None else self.rng.randrange(2**31 - 1),
                                                      timeout), timeout)
        except Exception as e:  # noqa: BLE001
            raise GatewayError(502, f"modèle local : {type(e).__name__}", "local_error") from e
        finally:
            scrub = getattr(eng, "scrub", None)  # the local path keeps nothing either (Codex review)
            if scrub is not None:
                try:
                    await asyncio.wait_for(scrub(busy=len(getattr(self.node, "running", ()))), 5.0)
                except Exception:  # noqa: BLE001 - best effort
                    pass
        meta = {"protocol": PROTOCOL, "decision": "local", "answer": None, "scores": {}, "certificate": False,
                "early_stop": False, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "peers_asked": 0,
                "peers_answered": 0, "peers": [], "replacements": 0, "routing": "local",
                "total_completion_tokens": gen.completion_tokens, "task_hint": None,
                "privacy": {**privacy, "kept_local": why, "readers": []}}
        return SwarmAnswer(text=gen.text, finish_reason=gen.finish_reason or "stop",
                           completion_tokens=gen.completion_tokens, meta=meta)

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
            """The swarm, then every routing name with live counts (`myriad`: peers, available), then
            the network's models. Extra fields are ignored by OpenAI clients."""
            try:
                peers, _ = await self.directory()
            except GatewayError:
                peers = []
            routes = live_routes(peers)
            sw = routes.pop("myriad")
            data = [{"id": SWARM_MODEL, "object": "model", "created": 0, "owned_by": "myriad", "myriad": sw},
                    {"id": LEGACY_SWARM_MODEL, "object": "model", "created": 0, "owned_by": "myriad",  # deprecated
                     "myriad": sw}]
            order = {"family": 0, "tag": 1, "model": 2}
            for name, r in sorted(routes.items(), key=lambda kv: (order[kv[1]["kind"]], kv[0])):
                data.append({"id": name, "object": "model", "created": 0,
                             "owned_by": "myriad-peer" if r["kind"] == "model" else "myriad", "myriad": r})
            return {"object": "list", "data": data}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            try:
                body = await read_json(request)
                stream = bool(body.get("stream", False))
                args = chat_args(body)
                spec = await self.resolve_route(body.get("model"))
                if spec.kind != "model":
                    args["model"] = None
                if spec.kind == "family":
                    args["family"] = spec.value
                elif spec.kind == "tag":
                    args["tag"] = spec.value
                    opts = body.get(SWARM_MODEL) if body.get(SWARM_MODEL) is not None else body.get(LEGACY_SWARM_MODEL)
                    if (opts or {}).get("k") is None:
                        args["k"] = 1  # one specialist by default; myriad.k asks for more (fused)
                result = await until_disconnect(request, self.ask(**args))
                if result is None:  # the client left: ask() was cancelled (its cleanup cancels the peers' jobs)
                    return Response(status_code=499)
                result.meta["route"] = {**spec.public(), **result.meta.get("route", {}),
                                        "fallback": bool(result.meta.get("route", {}).get("fallback"))}
            except GatewayError as e:
                return JSONResponse(e.body(), status_code=e.status)
            model = body.get("model") or SWARM_MODEL
            if stream:
                return StreamingResponse(sse_chunks(result, model), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache"})
            return completion_body(result, model)

        @app.get("/v1/myriad/status")
        @app.get("/v1/essaim/status")  # deprecated alias
        async def status():
            return {"node": self.node.status(), "balance": await self.balance(), "history": self.history[:20]}

        from .agents import install as install_agents
        install_agents(app, self)  # POST /v1/agents/run (sub-agents), GET /v1/agents/schema
        return app

    async def resolve_route(self, name) -> RouteSpec:
        """`model` field -> RouteSpec: `myriad:<x>` is a family when x is a known family (priors or the
        live directory), else a skill tag."""
        if isinstance(name, str) and name.strip().lower().startswith(PREFIXES):
            try:
                peers = (await self.directory())[0]
            except GatewayError:
                peers = []
            fams = known_families(peers)
        else:
            fams = None
        try:
            return parse_model(name, fams)
        except RouteError as e:
            raise GatewayError(400, str(e), "invalid_request_error") from e


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


MAX_BODY_BYTES = 1 << 20


async def until_disconnect(request: Request, coro):
    """Run `coro` while the HTTP client stays connected; cancel it (and wait for its cleanup) as soon as
    the client disconnects, then return None (audit net #7)."""
    task = asyncio.ensure_future(coro)

    async def watch() -> None:  # the body has been read: the next message can only be a disconnect
        while True:
            msg = await request.receive()
            if msg.get("type") == "http.disconnect":
                return

    watcher = asyncio.ensure_future(watch())
    try:
        done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return None
    except asyncio.CancelledError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    finally:
        watcher.cancel()


def _no_constant(name: str):
    raise ValueError(f"constante JSON non admise : {name}")  # NaN, Infinity


def _finite(x, what: str) -> float:
    """A finite number from JSON (bool and non-finite values refused)."""
    if isinstance(x, bool) or not isinstance(x, (int, float, str)):
        raise ValueError(f"{what} : nombre attendu")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{what} : nombre fini attendu")
    return v


async def read_json(request: Request) -> dict:
    if not request.headers.get("content-type", "").lower().startswith("application/json"):
        raise GatewayError(415, "Content-Type application/json attendu", "invalid_request_error")
    # The size limit is applied while reading (Content-Length first, then the bytes actually received),
    # never after loading a large body into memory (audit net #8).
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_BODY_BYTES:
                raise GatewayError(413, "requête trop grande", "invalid_request_error")
        except ValueError:
            raise GatewayError(400, "Content-Length invalide", "invalid_request_error") from None
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise GatewayError(413, "requête trop grande", "invalid_request_error")
        chunks.append(chunk)
    raw = b"".join(chunks)
    try:
        body = json.loads(raw, parse_constant=_no_constant)
    except (ValueError, RecursionError) as e:
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
    if model is not None and not isinstance(model, str):  # an unhashable value must not reach a set (audit net #9)
        raise GatewayError(400, "model doit être une chaîne", "invalid_request_error")
    hint = ext.get("task_hint")
    if hint is not None and not isinstance(hint, str):
        raise GatewayError(400, "task_hint doit valoir math, mc ou free", "invalid_request_error")
    try:
        temperature = _finite(body.get("temperature") if body.get("temperature") is not None else 0.0, "temperature")
        max_tokens = int(_finite(body.get("max_completion_tokens") or body.get("max_tokens") or 512, "max_tokens"))
        seed = body.get("seed")
        seed = None if seed is None else int(_finite(seed, "seed")) % (2**31 - 1)
        k = ext.get("k")
        k = None if k is None else int(_finite(k, "k"))
        timeout_s = ext.get("timeout_s")
        timeout_s = None if timeout_s is None else max(1.0, min(_finite(timeout_s, "timeout_s"), 600.0))
    except (TypeError, ValueError, OverflowError) as e:
        raise GatewayError(400, f"paramètre invalide : {e}", "invalid_request_error") from e
    if not (0.0 <= temperature <= 2.0):
        raise GatewayError(400, "temperature hors de [0, 2]", "invalid_request_error")
    early_stop = ext.get("early_stop")
    if early_stop is not None and not isinstance(early_stop, bool):
        raise GatewayError(400, f"{field_name}.early_stop doit être un booléen", "invalid_request_error")
    for flag in ("confirm_sensitive", "local"):
        if ext.get(flag) is not None and not isinstance(ext.get(flag), bool):
            raise GatewayError(400, f"{field_name}.{flag} doit être un booléen", "invalid_request_error")
    return {"messages": normalize_messages(body.get("messages")), "max_tokens": max_tokens, "temperature": temperature,
            "seed": seed, "k": k, "task_hint": hint,
            "model": model if model and model not in SWARM_MODELS else None,
            "format_instruction": bool(ext.get("format_instruction", True)), "timeout_s": timeout_s,
            "early_stop": early_stop, "confirm": bool(ext.get("confirm_sensitive")), "local": bool(ext.get("local"))}


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
