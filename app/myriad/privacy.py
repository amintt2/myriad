"""Local privacy guard: detect secrets and personal data in a question BEFORE it leaves the machine,
replace them by stable placeholders (reversible pseudonymisation), and put the original values back
in the answer. The mapping placeholder -> value lives in memory for one request and never leaves the
machine.

Detectors (regular expressions with validity checks, no network):
- private_key: PEM private key blocks;            -> CLE_PRIVEE_n
- api_key: well-known token formats (OpenAI, Anthropic, GitHub, GitLab, Slack, AWS, Google, Hugging
  Face, JWT, Bearer) and `password=...`-style assignments (only the value is replaced);   -> CLE_n
- card: 13-19 digits with a card prefix and a valid Luhn checksum;   -> CARTE_n
- iban: country code, check digits and a valid ISO 13616 mod-97 checksum;   -> IBAN_n
- email;   -> EMAIL_n
- phone: French numbers (0X XX XX XX XX, +33) and international +CC numbers;   -> TEL_n
- path: the user name inside home-directory paths (C:\\Users\\name, /home/name, /Users/name);   -> UTILISATEUR_n
- name: a capitalised name after a cue ("je m'appelle", "M.", "Mme", "Dr", "my name is", "Dear"...); a
  heuristic, it misses names without a cue;   -> PERSONNE_n
- custom: terms the user listed (project names, clients...);   -> TERME_n
A second detector can be plugged in: if the module `myriad.privacy_model` exists, its
`scan(text) -> {"p_sensitive", "decision", "spans": [{start, end, type, score}]}` is called too; spans are
united with conservative overlap coverage, the most conservative model-only decision wins,
and an error of that detector counts as sensitive content (fail-closed: confirmation required).

Each type has a mode: "mask" (replaced before sending), "warn" (sent as is, reported; the first time a
type would leave the machine unmasked, a confirmation is required), "local" (the question is never sent
to the network: answered by this machine's own model or refused) or "off". Defaults: secrets (keys,
cards, IBAN, custom terms) and user names in paths masked, emails and phones reported, names off.

Limits (stated in docs/08_securite.md): detection is heuristic, a placeholder can be lost or altered
by the model, and pseudonymisation does not hide what the rest of the text says.
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field

from . import privacy_rules
from .privacy_rules import iban_ok, luhn_ok

TYPES = ("private_key", "api_key", "card", "iban", "email", "phone", "path", "name", "custom", "other")
PREFIX = {"private_key": "CLE_PRIVEE", "api_key": "CLE", "card": "CARTE", "iban": "IBAN", "email": "EMAIL",
          "phone": "TEL", "path": "UTILISATEUR", "name": "PERSONNE", "custom": "TERME", "other": "DONNEE"}
SECRET_TYPES = frozenset({"private_key", "api_key", "card", "iban"})
MODES = ("mask", "warn", "local", "off")
DEFAULT_MODES = {"private_key": "mask", "api_key": "mask", "card": "mask", "iban": "mask", "email": "warn",
                 "phone": "warn", "path": "mask", "name": "off", "custom": "mask", "other": "mask"}
DECISIONS = ("send", "mask", "ask", "local")  # increasingly conservative
MAX_TERMS = 200
_PLACEHOLDER = re.compile(r"\b(" + "|".join(sorted(PREFIX.values(), key=len, reverse=True)) + r")_(\d{1,4})\b")

@dataclass
class Span:
    start: int
    end: int
    type: str
    source: str = "regex"
    types: tuple = ()  # after merging overlaps: the types of every detection it covers


def detect(text: str, terms=()) -> list[Span]:
    """Every sensitive span found by the regular-expression detectors (may overlap)."""
    out = [Span(s["start"], s["end"], s.get("guard_type", privacy_rules.GUARD_TYPES[s["type"]]))
           for s in privacy_rules.matches(text)]
    for term in terms:
        if not term:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, flags=re.IGNORECASE):
            out.append(Span(m.start(), m.end(), "custom"))
    return out


def _plugin():
    """The optional model-based detector (another branch): None when the module is absent."""
    try:
        return importlib.import_module("myriad.privacy_model")
    except ModuleNotFoundError as e:
        if e.name in ("myriad.privacy_model",):
            return None
        raise


_PLUGIN_TYPES = {"person": "name", "per": "name", "name": "name", "email": "email", "phone": "phone",
                 "iban": "iban", "card": "card", "credit_card": "card", "api_key": "api_key", "secret": "api_key",
                 "token": "api_key", "private_key": "private_key", "path": "path", "username": "path"}


def plugin_scan(text: str) -> tuple[list[Span], str]:
    """Spans and decision of the plugged-in detector; ("ask" + no span) when it fails (fail-closed)."""
    try:
        mod = _plugin()
    except Exception:  # noqa: BLE001 - a broken plugin: treated as sensitive
        return [], "ask"
    if mod is None:
        return [], "send"
    try:
        r = mod.scan(text, include_rules=False) if getattr(mod, "SHARED_RULES", False) else mod.scan(text)
        # Canonical rule hits already follow the guard's settings; only the model adds a decision.
        decision = r.get("model_decision", r.get("decision", "ask"))
        if decision not in DECISIONS:
            decision = "ask"
        spans = []
        for s in r.get("spans") or []:
            if s.get("source") == "rules":
                continue
            a, b = int(s["start"]), int(s["end"])
            if 0 <= a < b <= len(text):
                spans.append(Span(a, b, _PLUGIN_TYPES.get(str(s.get("type", "")).lower(), "other"), "model"))
        return spans, decision
    except Exception:  # noqa: BLE001 - fail-closed
        return [], "ask"


_RANK = {"local": 3, "mask": 2, "warn": 1}


def _resolve(spans: list[tuple[Span, str]]) -> list[tuple[Span, str]]:
    """Non-overlapping (span, mode) pairs: overlapping detections are merged into one span covering them
    all, with the strictest mode (and the type of the strictest detection). A lenient detection never
    hides a stricter one, and a partial mask never leaves the rest of a detection exposed (Codex
    reviews)."""
    out: list[tuple[Span, str]] = []
    for s, mode in sorted(spans, key=lambda sm: (sm[0].start, -(sm[0].end - sm[0].start))):
        if out and s.start < out[-1][0].end:
            prev, pmode = out[-1]
            best = (s, mode) if _RANK[mode] > _RANK[pmode] else (prev, pmode)
            merged = Span(prev.start, max(prev.end, s.end), best[0].type, best[0].source)
            # every constituent type is kept: an unmasked merged span needs each one's confirmation
            merged.types = tuple(dict.fromkeys(prev.types + (s.type,)))
            out[-1] = (merged, best[1])
        else:
            span = Span(s.start, s.end, s.type, s.source)
            span.types = (s.type,)
            out.append((span, mode))
    return out


@dataclass
class PrivacySettings:
    modes: dict = field(default_factory=lambda: dict(DEFAULT_MODES))
    terms: list = field(default_factory=list)  # user-defined sensitive terms (custom)
    confirmed: list = field(default_factory=list)  # types the user agreed to send unmasked

    @classmethod
    def from_dict(cls, d: dict | None) -> "PrivacySettings":
        d = d or {}
        modes = dict(DEFAULT_MODES)
        for k, v in (d.get("modes") or {}).items():
            if k in DEFAULT_MODES and v in MODES:
                modes[k] = v
        terms = [str(t).strip()[:100] for t in (d.get("terms") or []) if str(t).strip()][:MAX_TERMS]
        confirmed = [t for t in (d.get("confirmed") or []) if t in TYPES]
        return cls(modes, terms, confirmed)

    def to_dict(self) -> dict:
        return {"modes": dict(self.modes), "terms": list(self.terms), "confirmed": list(self.confirmed)}


@dataclass
class Finding:
    type: str
    action: str  # mask, warn, local
    source: str
    message: int  # index of the message
    start: int
    end: int


@dataclass
class Analysis:
    findings: list
    decision: str  # send, mask, ask, local
    messages: list  # what would be sent (placeholders in place of the masked values)
    mapping: dict  # placeholder -> original value (never leaves the machine)
    confirm_types: list  # types that need a first-time confirmation before leaving unmasked
    plugin_decision: str = "send"

    def summary(self) -> dict:
        """What was found and what will happen, without any original value."""
        detected: dict[str, int] = {}
        masked: dict[str, int] = {}
        unmasked: dict[str, int] = {}
        for f in self.findings:
            detected[f.type] = detected.get(f.type, 0) + 1
            bucket = masked if f.action == "mask" else unmasked
            bucket[f.type] = bucket.get(f.type, 0) + 1
        level = ("high" if any(t in SECRET_TYPES for t in detected) or self.decision == "local"
                 else "medium" if detected else "none")
        return {"decision": self.decision, "level": level, "detected": detected, "masked": masked,
                "unmasked": unmasked, "confirm_types": list(self.confirm_types),
                "placeholders": sorted(self.mapping), "plugin_decision": self.plugin_decision}


def _more(a: str, b: str) -> str:
    return a if DECISIONS.index(a) >= DECISIONS.index(b) else b


def analyze(messages: list[dict], settings: PrivacySettings | None = None, confirm: bool = False) -> Analysis:
    """Scan every message, decide, and build the pseudonymised messages. `confirm`: the caller agreed to
    send unmasked findings this time (no first-time confirmation needed)."""
    st = settings or PrivacySettings()
    taken = set()
    for m in messages:  # never reuse a placeholder that the text already contains
        taken.update(f"{a}_{b}" for a, b in _PLACEHOLDER.findall(m.get("content") or ""))
    counters: dict[str, int] = {}
    by_value: dict[tuple, str] = {}
    mapping: dict[str, str] = {}
    findings: list[Finding] = []
    out = []
    decision, plugin_decision = "send", "send"
    plugin_ask = False
    for idx, m in enumerate(messages):
        text = m.get("content") or ""
        spans = detect(text, st.terms)
        p_spans, p_dec = plugin_scan(text)
        plugin_decision = _more(plugin_decision, p_dec)
        plugin_ask = plugin_ask or p_dec == "ask"  # per message: one failing message is enough
        moded = [(s, st.modes.get(s.type, "mask")) for s in spans + p_spans]
        moded = [(s, mode) for s, mode in moded if mode != "off"]
        if any(mode == "local" for _, mode in moded):  # decided on every detection, before overlaps
            decision = _more(decision, "local")
        parts, pos = [], 0
        for s, mode in _resolve(moded):
            for ty in s.types or (s.type,):  # one finding per detected type (confirmations per type)
                findings.append(Finding(ty, mode, s.source, idx, s.start, s.end))
            if mode != "mask":
                continue
            value = text[s.start:s.end]
            key = (s.type, value)
            ph = by_value.get(key)
            if ph is None:
                n = counters.get(s.type, 0)
                while True:
                    n += 1
                    ph = f"{PREFIX[s.type]}_{n}"
                    if ph not in taken:
                        break
                counters[s.type] = n
                by_value[key] = ph
                mapping[ph] = value
            parts.append(text[pos:s.start])
            parts.append(ph)
            pos = s.end
            decision = _more(decision, "mask")
        parts.append(text[pos:])
        out.append({**m, "content": "".join(parts)})
    confirm_types = sorted({f.type for f in findings if f.action == "warn" and f.type not in st.confirmed})
    if plugin_ask:  # the detector failed or asks on some message: always a confirmation (fail-closed)
        confirm_types = sorted(set(confirm_types) | {"other"})
    if confirm_types and not confirm:
        decision = _more(decision, "ask")
    if plugin_decision != "ask":
        decision = _more(decision, plugin_decision)
    return Analysis(findings, decision, out, mapping, [] if confirm else confirm_types, plugin_decision)


def restore(text: str, mapping: dict) -> str:
    """Put the original values back in place of the placeholders (local only)."""
    if not mapping or not text:
        return text
    return _PLACEHOLDER.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
