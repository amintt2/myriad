"""Protection settings of a node and the logic behind them (all local; persisted in config.json under
`security`). The interface only calls the JSON API installed by `install_ui`, so it can be restyled.

As a requester (Security.check_peer, Security.route_fields, used by the gateway on BOTH routing paths:
the tracker's selection receives the policy, and every assigned peer is checked again locally before
anything is sent to it):
- require_e2e (default on): only peers with a valid end-to-end key; plaintext peers are never chosen.
- blocklist of node ids (with a reason and a date) and of model families;
- trusted mode: only the node ids of the allowlist (added by id or by invite code);
- private swarm: only nodes proving knowledge of the swarm key (protocol.SwarmProof). The shared secret
  itself is never stored nor sent: K = scrypt(secret), the group tag is HMAC(K, "group");
- minimum model reliability;
- local quarantine of peers that repeatedly disagree with certified majorities, send undecryptable or
  badly signed answers, or whose tracker reputation (fed by canary jobs) is low. Counters decay
  (half-life QUARANTINE_HALF_LIFE_S); a quarantine always expires (30 min, doubling, at most 24 h) and
  becomes permanent only if the user blocks the node;
- local-only mode (never send questions to the network), the privacy guard (privacy.py), the history
  cap (only the last `max_history_turns` user turns are sent) and peer rotation (soft avoidance of the
  peers used recently, so that no single peer accumulates a user's history).
As a server (ServeGuard): jobs from blocked nodes are refused; per-requester rate and concurrency
limits; maximum prompt and output tokens. Requesters of encrypted jobs are pseudonymous, so the
per-requester limits and the blocklist are also sent to the tracker (protocol.ServePolicy), which knows
the paying account and enforces them.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from fastapi import Request  # module level: the endpoints' annotations are resolved here

from .priors import family_of
from .privacy import DEFAULT_MODES, MODES, TYPES, PrivacySettings, analyze

QUARANTINE_HALF_LIFE_S = 6 * 3600
QUARANTINE_BASE_S = 30 * 60
QUARANTINE_MAX_S = 24 * 3600
LOW_REPUTATION = 0.5
MAX_LIST = 1024
INVITE_PREFIX = "myr1-"
SWARM_SALT = b"myriad-swarm-v1"


def _hex32(x) -> str | None:
    if isinstance(x, str) and len(x) == 32 and all(c in "0123456789abcdef" for c in x):
        return x
    return None


# ---------- invite codes, private swarms ----------
def invite_code(node_id: str) -> str:
    """A short code that encodes a node id (with a checksum): myr1-xxxxxxxx..."""
    raw = bytes.fromhex(node_id)
    raw += hashlib.sha256(b"myriad-invite" + raw).digest()[:2]
    return INVITE_PREFIX + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def parse_invite(code: str) -> str | None:
    """Node id of an invite code (or of a bare node id); None if invalid."""
    code = (code or "").strip().lower()
    if _hex32(code):
        return code
    if not code.startswith(INVITE_PREFIX):
        return None
    body = code[len(INVITE_PREFIX):].upper()
    try:
        raw = base64.b32decode(body + "=" * (-len(body) % 8))
    except (ValueError, TypeError):
        return None
    if len(raw) != 18 or hashlib.sha256(b"myriad-invite" + raw[:16]).digest()[:2] != raw[16:]:
        return None
    return raw[:16].hex()


def swarm_key(secret: str) -> str:
    """K = scrypt(secret): slow on purpose, the group tag being public (a short secret could otherwise be
    guessed offline from it). Use a long random secret (the interface can generate one)."""
    return hashlib.scrypt(secret.encode("utf-8"), salt=SWARM_SALT, n=2**14, r=8, p=1, dklen=32).hex()


def swarm_group(key_hex: str) -> str:
    return hmac.new(bytes.fromhex(key_hex), b"group", hashlib.sha256).hexdigest()[:32]


def swarm_proof(key_hex: str, node_id: str) -> str:
    return hmac.new(bytes.fromhex(key_hex), b"member\x00" + node_id.encode("ascii"), hashlib.sha256).hexdigest()[:32]


# ---------- settings ----------
@dataclass
class SecuritySettings:
    require_e2e: bool = True  # « Exiger le chiffrement de bout en bout »
    local_only: bool = False  # never send questions to the network
    trusted_only: bool = False  # « Seulement mes nœuds de confiance »
    trusted: list = field(default_factory=list)  # [{node_id, label, ts}]
    blocked: list = field(default_factory=list)  # [{node_id, reason, ts}]
    blocked_families: list = field(default_factory=list)  # [{family, reason, ts}]
    swarm_key: str | None = None  # scrypt(secret) of the private swarm (the secret itself is never stored)
    min_reliability: float = 0.0
    quarantine: bool = True
    rotate_peers: bool = True
    max_history_turns: int = 8  # user turns sent to peers (0: no cap)
    planner_context_chars: int = 600  # sub-agents: excerpt of each context item shown to the planner peer
    privacy: dict = field(default_factory=lambda: PrivacySettings().to_dict())
    # serving side
    rate_per_min: int = 120  # jobs per requester and minute (0: no limit)
    max_concurrent: int = 4  # jobs of one requester at a time (0: no limit)
    max_prompt_tokens: int = 8000  # estimated (characters / 4)
    mlock: str = "auto"  # keep the model and its cache in RAM (no swap): auto, on, off

    @classmethod
    def from_dict(cls, d: dict | None) -> "SecuritySettings":
        s = cls()
        d = d if isinstance(d, dict) else {}
        for k in ("require_e2e", "local_only", "trusted_only", "quarantine", "rotate_peers"):
            if isinstance(d.get(k), bool):
                setattr(s, k, d[k])
        for k, lo, hi in (("max_history_turns", 0, 1000), ("rate_per_min", 0, 10_000), ("max_concurrent", 0, 64),
                          ("max_prompt_tokens", 1, 1_000_000), ("planner_context_chars", 0, 3000)):
            v = d.get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                setattr(s, k, max(lo, min(hi, v)))
        v = d.get("min_reliability")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            s.min_reliability = max(0.0, min(1.0, float(v)))
        s.trusted = [x for x in (d.get("trusted") or []) if isinstance(x, dict) and _hex32(x.get("node_id"))][:MAX_LIST]
        s.blocked = [x for x in (d.get("blocked") or []) if isinstance(x, dict) and _hex32(x.get("node_id"))][:MAX_LIST]
        s.blocked_families = [x for x in (d.get("blocked_families") or [])
                              if isinstance(x, dict) and isinstance(x.get("family"), str) and x["family"]][:32]
        k = d.get("swarm_key")
        s.swarm_key = k if isinstance(k, str) and len(k) == 64 and all(c in "0123456789abcdef" for c in k) else None
        s.privacy = PrivacySettings.from_dict(d.get("privacy")).to_dict()
        if d.get("mlock") in ("auto", "on", "off"):
            s.mlock = d["mlock"]
        return s

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- quarantine ----------
@dataclass
class _Rec:
    agree: float = 0.0
    disagree: float = 0.0
    severe: float = 0.0
    t: float = 0.0
    until: float | None = None
    level: int = 0
    reason: str | None = None
    last_end: float = 0.0


class Quarantine:
    """Local, automatic, temporary exclusion of misbehaving peers (never permanent: see module doc)."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self.rec: dict[str, _Rec] = {}

    def _get(self, node_id: str, now: float) -> _Rec:
        r = self.rec.get(node_id)
        if r is None:
            r = self.rec[node_id] = _Rec(t=now)
            if len(self.rec) > 10_000:  # bounded memory: forget the oldest records
                for k in list(self.rec)[:1000]:
                    self.rec.pop(k, None)
        f = 0.5 ** (max(0.0, now - r.t) / QUARANTINE_HALF_LIFE_S)
        r.agree, r.disagree, r.severe, r.t = r.agree * f, r.disagree * f, r.severe * f, now
        if r.until is not None and now >= r.until:
            r.until, r.last_end = None, now
        if r.until is None and r.level and now - r.last_end > QUARANTINE_MAX_S:
            r.level, r.last_end = r.level - 1, now  # the backoff also decays
        return r

    def note(self, node_id: str, kind: str, reason: str | None = None) -> bool:
        """Record one observation (agree, disagree, severe). True if the peer is now quarantined."""
        now = self.clock()
        r = self._get(node_id, now)
        if kind == "agree":
            r.agree += 1
        elif kind == "disagree":
            r.disagree += 1
        else:
            r.severe += 1
        if r.until is None and (r.severe >= 2 or (r.disagree >= 4 and r.disagree / (r.agree + r.disagree) >= 0.6)):
            dur = min(QUARANTINE_MAX_S, QUARANTINE_BASE_S * 2 ** r.level)
            r.until, r.level = now + dur, min(r.level + 1, 10)
            r.reason = reason or ("désaccords répétés" if r.severe < 2 else "réponses invalides")
            r.agree = r.disagree = r.severe = 0.0
            return True
        return False

    def active(self, node_id: str) -> bool:
        r = self.rec.get(node_id)
        if r is None or r.until is None:
            return False
        return self._get(node_id, self.clock()).until is not None

    def lift(self, node_id: str) -> bool:
        r = self.rec.pop(node_id, None)
        return r is not None

    def ids(self) -> list[str]:
        return [n for n in list(self.rec) if self.active(n)]

    def public(self) -> list[dict]:
        now = self.clock()
        return [{"node_id": n, "until": self.rec[n].until, "remaining_s": round(self.rec[n].until - now),
                 "reason": self.rec[n].reason, "level": self.rec[n].level} for n in self.ids()]


# ---------- the node's security state ----------
class Security:
    def __init__(self, settings: SecuritySettings | dict | None = None, save=None, clock=time.time):
        self.settings = settings if isinstance(settings, SecuritySettings) else SecuritySettings.from_dict(settings)
        self._save = save
        self.clock = clock
        self.quarantine = Quarantine(clock)
        self.recent_peers: deque = deque(maxlen=64)  # (time, node_id) of the peers asked lately (rotation)
        self.listeners: list = []  # callables run after a change (the node re-sends its ServePolicy...)

    # ----- persistence -----
    def save(self) -> None:
        if self._save is not None:
            self._save(self.settings.to_dict())
        for cb in list(self.listeners):
            try:
                cb()
            except Exception:  # noqa: BLE001 - a listener must not break the change
                pass

    @property
    def privacy(self) -> PrivacySettings:
        return PrivacySettings.from_dict(self.settings.privacy)

    # ----- lists -----
    def blocked_ids(self) -> set[str]:
        return {x["node_id"] for x in self.settings.blocked}

    def blocked_families(self) -> set[str]:
        return {x["family"].lower() for x in self.settings.blocked_families}

    def trusted_ids(self) -> set[str]:
        return {x["node_id"] for x in self.settings.trusted}

    def is_blocked(self, node_id: str) -> bool:
        return node_id in self.blocked_ids()

    # ----- private swarm -----
    def swarm_group(self) -> str | None:
        return swarm_group(self.settings.swarm_key) if self.settings.swarm_key else None

    def swarm_proofs(self, node_id: str) -> list[dict] | None:
        k = self.settings.swarm_key
        return [{"g": swarm_group(k), "p": swarm_proof(k, node_id)}] if k else None

    def swarm_member(self, node_id: str, proofs) -> bool:
        k = self.settings.swarm_key
        if not k:
            return True
        g, want = swarm_group(k), swarm_proof(k, node_id)
        for pr in proofs or []:
            pr = pr if isinstance(pr, dict) else getattr(pr, "model_dump", lambda: {})()
            if pr.get("g") == g and isinstance(pr.get("p"), str) and hmac.compare_digest(pr["p"], want):
                return True
        return False

    # ----- peer policy -----
    def deny_ids(self) -> list[str]:
        ids = list(dict.fromkeys([x["node_id"] for x in self.settings.blocked] + self.quarantine.ids()))
        return ids[:MAX_LIST]

    def policy_active(self) -> bool:
        """Whether some peers must be refused (then a peer is checked BEFORE anything is sent to it)."""
        s = self.settings
        return bool(s.blocked or s.blocked_families or s.trusted_only or s.swarm_key or s.min_reliability
                    or (s.quarantine and self.quarantine.ids()))

    def soft_avoid(self, window_s: float = 600.0) -> list[str]:
        if not self.settings.rotate_peers:
            return []
        now = self.clock()
        return list(dict.fromkeys(n for t, n in reversed(self.recent_peers) if now - t <= window_s))[:64]

    def used_peer(self, node_id: str) -> None:
        self.recent_peers.append((self.clock(), node_id))

    def route_fields(self) -> dict:
        s = self.settings
        out = {"deny": self.deny_ids(), "deny_families": sorted(self.blocked_families())[:32],
               "e2e": True if s.require_e2e else None, "swarm": self.swarm_group(),
               "min_rel": s.min_reliability or None, "soft_avoid": self.soft_avoid()}
        if s.trusted_only:
            out["only"] = sorted(self.trusted_ids())[:MAX_LIST]
        return out

    def check_peer(self, p: dict, reliability: float | None = None, e2e_ok: bool | None = None) -> str | None:
        """Why this peer must not receive a job (None: it may). `p`: a directory entry or a PeerCard dump;
        `e2e_ok`: whether its key certificate checked out (None: not evaluated)."""
        nid = p.get("node_id")
        s = self.settings
        if nid in self.blocked_ids():
            return "blocked"
        fam = (family_of(p["model"]) if p.get("model") else str(p.get("family") or "")).lower()
        if fam in self.blocked_families():
            return "family_blocked"
        if s.quarantine and self.quarantine.active(nid):
            return "quarantined"
        if s.trusted_only and nid not in self.trusted_ids():
            return "not_trusted"
        if s.swarm_key and not self.swarm_member(nid, p.get("swarms")):
            return "not_in_swarm"
        if s.min_reliability and reliability is not None and reliability < s.min_reliability:
            return "low_reliability"
        rep = p.get("reputation")
        if s.quarantine and isinstance(rep, (int, float)) and rep < LOW_REPUTATION:
            self.quarantine.note(nid, "severe", "réputation basse (canaris)")
            self.quarantine.note(nid, "severe", "réputation basse (canaris)")
            return "quarantined"
        if s.require_e2e and e2e_ok is False:
            return "no_e2e"
        return None

    def note(self, node_id: str, kind: str, reason: str | None = None) -> None:
        if self.settings.quarantine:
            self.quarantine.note(node_id, kind, reason)

    # ----- mutations (the caller saves) -----
    def block(self, node_id: str, reason: str = "") -> None:
        if not _hex32(node_id):
            raise ValueError("identifiant de nœud invalide")
        self.settings.blocked = [x for x in self.settings.blocked if x["node_id"] != node_id]
        self.settings.blocked.append({"node_id": node_id, "reason": str(reason)[:200], "ts": int(self.clock())})
        del self.settings.blocked[:-MAX_LIST]
        self.settings.trusted = [x for x in self.settings.trusted if x["node_id"] != node_id]

    def unblock(self, node_id: str) -> bool:
        n = len(self.settings.blocked)
        self.settings.blocked = [x for x in self.settings.blocked if x["node_id"] != node_id]
        return len(self.settings.blocked) != n

    def block_family(self, family: str, reason: str = "") -> None:
        fam = str(family or "").strip().lower()[:64]
        if not fam:
            raise ValueError("famille invalide")
        self.settings.blocked_families = [x for x in self.settings.blocked_families if x["family"] != fam]
        self.settings.blocked_families.append({"family": fam, "reason": str(reason)[:200], "ts": int(self.clock())})
        del self.settings.blocked_families[:-32]

    def unblock_family(self, family: str) -> bool:
        fam = str(family or "").strip().lower()
        n = len(self.settings.blocked_families)
        self.settings.blocked_families = [x for x in self.settings.blocked_families if x["family"] != fam]
        return len(self.settings.blocked_families) != n

    def trust(self, node_or_invite: str, label: str = "") -> str:
        nid = parse_invite(node_or_invite)
        if nid is None:
            raise ValueError("identifiant ou code d'invitation invalide")
        self.settings.trusted = [x for x in self.settings.trusted if x["node_id"] != nid]
        self.settings.trusted.append({"node_id": nid, "label": str(label)[:80], "ts": int(self.clock())})
        del self.settings.trusted[:-MAX_LIST]
        self.unblock(nid)
        return nid

    def untrust(self, node_id: str) -> bool:
        n = len(self.settings.trusted)
        self.settings.trusted = [x for x in self.settings.trusted if x["node_id"] != node_id]
        return len(self.settings.trusted) != n

    def set_swarm_secret(self, secret: str | None) -> None:
        if secret is None or secret == "":
            self.settings.swarm_key = None
            return
        if len(secret) < 12:
            raise ValueError("secret trop court (12 caractères au moins ; un secret aléatoire long est préférable)")
        self.settings.swarm_key = swarm_key(secret)

    def update(self, body: dict) -> None:
        """Apply the simple settings of a JSON body (validated like the configuration file)."""
        for k in ("require_e2e", "local_only", "trusted_only", "quarantine", "rotate_peers"):
            if k in body and not isinstance(body[k], bool):
                raise ValueError(f"{k} doit être un booléen")
        if "mlock" in body and body["mlock"] not in ("auto", "on", "off"):
            raise ValueError("mlock : auto, on ou off")
        cur = self.settings.to_dict()
        for k in ("require_e2e", "local_only", "trusted_only", "quarantine", "rotate_peers", "min_reliability",
                  "max_history_turns", "rate_per_min", "max_concurrent", "max_prompt_tokens", "mlock",
                  "planner_context_chars"):
            if k in body:
                cur[k] = body[k]
        if isinstance(body.get("privacy"), dict):
            p = dict(cur["privacy"])
            pb = body["privacy"]
            if isinstance(pb.get("modes"), dict):
                p["modes"] = {**p["modes"], **{k: v for k, v in pb["modes"].items() if k in DEFAULT_MODES and v in MODES}}
            if isinstance(pb.get("terms"), list):
                p["terms"] = pb["terms"]
            if isinstance(pb.get("confirmed"), list):
                p["confirmed"] = [t for t in pb["confirmed"] if t in TYPES]
            cur["privacy"] = p
        self.settings = SecuritySettings.from_dict(cur)

    def confirm_types(self, types) -> None:
        p = self.privacy
        p.confirmed = sorted(set(p.confirmed) | {t for t in types if t in TYPES})
        self.settings.privacy = p.to_dict()

    def public(self, node_id: str | None = None) -> dict:
        d = self.settings.to_dict()
        d.pop("swarm_key", None)
        d["swarm"] = {"enabled": bool(self.settings.swarm_key),
                      "group": (self.swarm_group() or "")[:8] or None}
        d["quarantined"] = self.quarantine.public()
        if node_id:
            d["invite"] = invite_code(node_id)
            d["node_id"] = node_id
        return d


# ---------- serving side ----------
class ServeGuard:
    """Limits a serving node applies to the jobs it receives (requesters of encrypted jobs are
    pseudonyms: the tracker enforces the per-account part, see protocol.ServePolicy)."""

    def __init__(self, security: Security | None = None, clock=time.monotonic):
        self.security = security or Security()
        self.clock = clock
        self.calls: dict[str, deque] = {}
        self.running: dict[str, int] = {}

    def check(self, requester_id: str, prompt_chars: int, pseudonymous: bool = False) -> str | None:
        """`pseudonymous`: an encrypted job, whose requester id is a one-job pseudonym: per-requester
        limits are meaningless here (the tracker enforces them per paying account, ServePolicy), only
        the size limit applies."""
        s = self.security.settings
        if math.ceil(prompt_chars / 4) > s.max_prompt_tokens:
            return "prompt_too_long"
        if pseudonymous:
            return None
        if self.security.is_blocked(requester_id):
            return "blocked"
        if s.max_concurrent and self.running.get(requester_id, 0) >= s.max_concurrent:
            return "busy_requester"
        if s.rate_per_min:
            now = self.clock()
            if len(self.calls) > 4096:  # bounded memory: drop the requesters idle for a minute
                for k in [k for k, v in self.calls.items() if not v or now - v[-1] > 60.0]:
                    del self.calls[k]
                if len(self.calls) > 4096:
                    return "rate_limited"  # fail closed rather than grow without bound
            q = self.calls.setdefault(requester_id, deque())
            while q and now - q[0] > 60.0:
                q.popleft()
            if len(q) >= s.rate_per_min:
                return "rate_limited"
            q.append(now)
        return None

    def started(self, requester_id: str) -> None:
        self.running[requester_id] = self.running.get(requester_id, 0) + 1

    def finished(self, requester_id: str) -> None:
        n = self.running.get(requester_id, 0) - 1
        if n > 0:
            self.running[requester_id] = n
        else:
            self.running.pop(requester_id, None)

    def serve_policy(self) -> dict:
        s = self.security.settings
        return {"deny": [x["node_id"] for x in s.blocked][:MAX_LIST], "rate_per_min": s.rate_per_min,
                "max_concurrent": s.max_concurrent}


# ---------- local JSON API (the interface) ----------
def install_ui(app, rt) -> None:
    """GET /api/security and the POST /api/security/* actions (token-checked by the UI middleware)."""
    from fastapi.responses import JSONResponse

    from .gateway import GatewayError, read_json

    def sec() -> Security:
        gw = rt.gateway
        if gw is not None:
            return gw.security
        s = getattr(rt, "_security", None)
        if s is None:
            s = rt._security = Security(getattr(rt.config, "security", None))
        return s

    def persist(s: Security) -> None:
        from .config import Config
        rt.config.security = s.settings.to_dict()
        if rt.home is not None and (rt.home / "config.json").exists():
            saved = Config.load(rt.home)
            saved.security = s.settings.to_dict()
            saved.save(rt.home)
        for cb in list(s.listeners):
            try:
                cb()
            except Exception:  # noqa: BLE001
                pass

    def node_id() -> str | None:
        return rt.node.node_id if rt.node is not None else None

    async def features() -> list:
        gw = rt.gateway
        if gw is None:
            return []
        try:
            return sorted(await gw.features())
        except Exception:  # noqa: BLE001 - tracker unreachable
            return []

    @app.get("/api/security")
    async def get_security():
        s = sec()
        feats = await features()
        return {**s.public(node_id()), "tracker_features": feats, "tracker_e2e": "e2e" in feats}

    async def mutate(request: Request, fn):
        try:
            body = await read_json(request)
            s = sec()
            out = fn(s, body)
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        except (TypeError, ValueError, KeyError) as e:
            return JSONResponse({"error": f"valeur invalide : {e}"}, status_code=400)
        persist(s)
        return {"ok": True, **(out or {}), "security": s.public(node_id())}

    @app.post("/api/security/settings")
    async def set_settings(request: Request):
        return await mutate(request, lambda s, b: s.update(b))

    @app.post("/api/security/block")
    async def block(request: Request):
        def f(s, b):
            nid = parse_invite(str(b.get("node_id", "")))
            if nid is None:
                raise ValueError("identifiant de nœud invalide")
            s.block(nid, str(b.get("reason") or ""))
        return await mutate(request, f)

    @app.post("/api/security/unblock")
    async def unblock(request: Request):
        return await mutate(request, lambda s, b: {"removed": s.unblock(str(b.get("node_id", "")))})

    @app.post("/api/security/block_family")
    async def block_family(request: Request):
        return await mutate(request, lambda s, b: s.block_family(str(b.get("family", "")), str(b.get("reason") or "")))

    @app.post("/api/security/unblock_family")
    async def unblock_family(request: Request):
        return await mutate(request, lambda s, b: {"removed": s.unblock_family(str(b.get("family", "")))})

    @app.post("/api/security/trust")
    async def trust(request: Request):
        return await mutate(request, lambda s, b: {"node_id": s.trust(str(b.get("node_id") or b.get("invite") or ""),
                                                                      str(b.get("label") or ""))})

    @app.post("/api/security/untrust")
    async def untrust(request: Request):
        return await mutate(request, lambda s, b: {"removed": s.untrust(str(b.get("node_id", "")))})

    @app.post("/api/security/swarm")
    async def swarm(request: Request):
        def f(s, b):
            secret = b.get("secret")
            if secret is not None and not isinstance(secret, str):
                raise ValueError("secret invalide")
            s.set_swarm_secret(secret)
        return await mutate(request, f)

    @app.post("/api/security/quarantine/lift")
    async def lift(request: Request):
        return await mutate(request, lambda s, b: {"removed": s.quarantine.lift(str(b.get("node_id", "")))})

    @app.post("/api/security/confirm")
    async def confirm(request: Request):
        """The user agrees, once, that these types of content may leave the machine unmasked."""
        return await mutate(request, lambda s, b: s.confirm_types(b.get("types") or []))

    @app.post("/api/security/check")
    async def check(request: Request):
        """Sensitivity indicator of a question before it is sent: what was detected, what will be
        masked, the decision, and who will read it (no original value is echoed)."""
        try:
            body = await read_json(request)
            text = str(body.get("text") or "")
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        s = sec()
        a = analyze([{"role": "user", "content": text}], s.privacy)
        st = s.settings
        readers = {"local_only": st.local_only or a.decision == "local",
                   "e2e_required": st.require_e2e,
                   "trust": ("trusted" if st.trusted_only else "swarm" if st.swarm_key else "network"),
                   "k": getattr(rt.gateway, "default_k", None),
                   "note": "le pair qui calcule lit la question (après masquage) ; le traqueur ne voit que des métadonnées"
                   if st.require_e2e else "sans chiffrement exigé, le traqueur peut aussi lire la question"}
        return {**a.summary(), "masked_text": a.messages[0]["content"], "readers": readers}

    @app.post("/api/security/report")
    async def report(request: Request):
        try:
            body = await read_json(request)
            nid = parse_invite(str(body.get("node_id", "")))
            if nid is None:
                raise ValueError("identifiant de nœud invalide")
            gw = rt.gateway
            if gw is None:
                return JSONResponse({"error": "le nœud démarre"}, status_code=503)
            r = await gw.report(nid, str(body.get("reason") or "other"), body.get("job_id"), str(body.get("note") or ""))
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        except (TypeError, ValueError) as e:
            return JSONResponse({"error": f"valeur invalide : {e}"}, status_code=400)
        return {"ok": True, "tracker": r}
