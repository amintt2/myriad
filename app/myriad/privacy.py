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
united (overlaps resolved in favour of the earliest, longest span), the most conservative decision wins,
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

_PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[\s\S]+?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----")
_API_KEYS = [re.compile(p) for p in (
    r"\bsk-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_\-]{20,}",
    r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", r"\bgithub_pat_[A-Za-z0-9_]{40,}\b", r"\bglpat-[A-Za-z0-9_\-]{20,}\b",
    r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", r"\bAIza[0-9A-Za-z_\-]{35}\b",
    r"\bhf_[A-Za-z0-9]{30,}\b", r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",
    r"(?<=Bearer )[A-Za-z0-9._~+/\-]{20,}=*")]
_ASSIGN = re.compile(r"(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|token|password|passwd|"
                     r"pwd|mot[_ ]de[_ ]passe)\b\s*[:=]\s*[\"']?([^\s\"',;]{6,})")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}\b")
_PHONE = re.compile(r"(?<![\w+])(?:(?:\+33\s?|0033\s?|0)[1-9](?:[\s.\-]?\d{2}){4}|\+(?:[1-9]\d{0,2})[\s.\-]?"
                    r"(?:\(?\d{1,4}\)?[\s.\-]?){2,5}\d{2,4})(?![\w])")
_CARD = re.compile(r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
_PATH = re.compile(r"(?i)(?:[A-Z]:\\(?:Users|Documents and Settings)\\|/home/|/Users/)([^\\/\s\"'<>|:*?]{2,64})")
_NAME = re.compile(r"(?:\b(?:[Jj]e m'appelle|[Jj]e suis|[Mm]y name is|I am|I'm|[Mm]onsieur|[Mm]adame|"
                   r"[Mm]ademoiselle|Mme|Mlle|Mr|Mrs|Ms|Dr|Pr|Dear|Cher|Chère|Bonjour|Hello|Hi)\.?|\bM\.)[ \t]+"
                   r"([A-ZÀ-Ý][a-zà-ÿ'\-]+(?:[ \t]+[A-ZÀ-Ý][a-zà-ÿ'\-]+){0,2})")
_NOT_NAMES = frozenset({"Je", "I", "Le", "La", "Les", "The", "A", "Un", "Une", "Tout", "All", "Not", "Here",
                        "Ici", "Ok", "Merci", "Thanks", "Monsieur", "Madame"})


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def iban_ok(raw: str) -> bool:
    s = raw.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    s = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in s)
    return int(num) % 97 == 1


@dataclass
class Span:
    start: int
    end: int
    type: str
    source: str = "regex"
    types: tuple = ()  # after merging overlaps: the types of every detection it covers


def detect(text: str, terms=()) -> list[Span]:
    """Every sensitive span found by the regular-expression detectors (may overlap)."""
    out: list[Span] = []
    for m in _PRIVATE_KEY.finditer(text):
        out.append(Span(m.start(), m.end(), "private_key"))
    for rx in _API_KEYS:
        for m in rx.finditer(text):
            out.append(Span(m.start(), m.end(), "api_key"))
    for m in _ASSIGN.finditer(text):
        out.append(Span(m.start(1), m.end(1), "api_key"))
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and digits[0] in "23456" and luhn_ok(digits):
            out.append(Span(m.start(), m.end(), "card"))
    for m in _IBAN.finditer(text):
        if iban_ok(m.group()):
            out.append(Span(m.start(), m.end(), "iban"))
    for m in _EMAIL.finditer(text):
        out.append(Span(m.start(), m.end(), "email"))
    for m in _PHONE.finditer(text):
        if 9 <= len(re.sub(r"\D", "", m.group())) <= 15:
            out.append(Span(m.start(), m.end(), "phone"))
    for m in _PATH.finditer(text):
        out.append(Span(m.start(1), m.end(1), "path"))
    for m in _NAME.finditer(text):
        name = m.group(1)
        if name.split()[0] not in _NOT_NAMES:
            out.append(Span(m.start(1), m.end(1), "name"))
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
        r = mod.scan(text)
        decision = r.get("decision", "ask")
        if decision not in DECISIONS:
            decision = "ask"
        spans = []
        for s in r.get("spans") or []:
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
