"""Deterministic PII and secret rules (regex + checksums), the always-available layer of the privacy scan.

Each rule yields spans `{"start", "end", "type", "score"}` on the original text. The score says how sure
the rule is: 1.0 when a checksum validates the value (Luhn, IBAN mod 97, NIR key) or the format is
unambiguous (private key block, provider-prefixed API key), lower for format-only matches (phone
numbers, IPv4) and cue-based matches (a name after « je m'appelle »).

Types (shared with the model layer, see privacy_model.TYPES): PERSON, EMAIL, PHONE, ADDRESS, ID,
FINANCIAL, SECRET, IP, USERNAME, DATE_OF_BIRTH.

The guard, scanner and benchmark all use this canonical rule set."""
from __future__ import annotations

import re
from typing import Callable, Iterator

Span = dict


def luhn_ok(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        s += d
        alt = not alt
    return s % 10 == 0


def iban_ok(raw: str) -> bool:
    s = re.sub(r"[\s-]", "", raw).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s):
        return False
    r = s[4:] + s[:4]
    n = "".join(str(int(c, 36)) for c in r)
    return int(n) % 97 == 1


def nir_ok(raw: str) -> bool:
    """French social security number (NIR): 13 characters + 2-digit key, key = 97 - (n mod 97).
    Corsica departments 2A / 2B are replaced by 19 / 18 for the computation."""
    s = re.sub(r"[\s.-]", "", raw).upper()
    if len(s) != 15:
        return False
    body, key = s[:13], s[13:]
    body = body.replace("2A", "19", 1) if body[5:7] == "2A" else body.replace("2B", "18", 1) if body[5:7] == "2B" else body
    if not body.isdigit() or not key.isdigit():
        return False
    return 97 - int(body) % 97 == int(key)


def _rx(p: str, flags: int = 0) -> re.Pattern:
    return re.compile(p, flags)


EMAIL = _rx(r"(?<![\w.%+-])[\w.%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}(?![\w-])")
# French numbers (0X XX XX XX XX, +33 X XX XX XX XX) and international E.164-like numbers
PHONE_FR = _rx(r"(?<![\w+])(?:(?:\+|00)33[\s.-]?(?:\(0\)[\s.-]?)?|0)[1-9](?:[\s.-]?\d{2}){4}(?!\d)")
PHONE_INTL = _rx(r"(?<![\w+])(?:\+|00)[1-9]\d{0,2}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?\d{1,4}(?:[\s.-]?\d{2,4}){2,4}(?!\d)")
PHONE_US = _rx(r"(?<![\w-])\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
CARD = _rx(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
IBAN_CANDIDATE = _rx(r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{2}(?:[ -]?[A-Za-z0-9]{2,4}){3,8}(?![A-Za-z0-9])")
NIR = _rx(r"(?<!\d)[12][\s.]?\d{2}[\s.]?(?:0[1-9]|1[0-2]|[2-9]\d)[\s.]?(?:\d{2}|2[AaBb])[\s.]?\d{3}[\s.]?\d{3}[\s.]?\d{2}(?!\d)")
SSN_US = _rx(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
IPV4 = _rx(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
IPV6 = _rx(r"(?<![\w:])(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}(?![\w:])")

SECRET_PATTERNS: list[tuple[re.Pattern, float]] = [
    (_rx(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|$)"), 1.0),
    (_rx(r"(?<![A-Z0-9])(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[A-Z0-9]{16}(?![A-Z0-9])"), 1.0),  # AWS access key id
    (_rx(r"(?<![\w])(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}(?![\w])"), 1.0),  # GitHub tokens
    (_rx(r"(?<![\w])github_pat_[A-Za-z0-9_]{22,255}(?![\w])"), 1.0),
    (_rx(r"(?<![\w])glpat-[A-Za-z0-9_-]{20,}(?![\w-])"), 1.0),  # GitLab
    (_rx(r"(?<![\w-])sk-(?:proj-|ant-(?:(?:api|admin)\d\d-)?|svcacct-|admin-|live-|test-)?"
         r"[A-Za-z0-9_-]{20,}(?![\w-])"), 1.0),  # OpenAI / Anthropic
    (_rx(r"(?<![\w])(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}(?![\w])"), 1.0),  # Stripe
    (_rx(r"(?<![\w-])xox[abposr]-[A-Za-z0-9-]{10,}(?![\w-])"), 1.0),  # Slack
    (_rx(r"(?<![\w-])AIza[0-9A-Za-z_-]{35}(?![\w-])"), 1.0),  # Google API key
    (_rx(r"(?<![\w-])hf_[A-Za-z0-9]{30,}(?![\w-])"), 1.0),  # Hugging Face
    (_rx(r"(?<![\w-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![\w-])"), 1.0),  # JWT
    (_rx(r"(?<![\w])(?:SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}|npm_[A-Za-z0-9]{36})(?![\w-])"), 1.0),
    # credentials inside a URL: scheme://user:password@host
    (_rx(r"(?<=://)[^\s/:@]{1,64}:[^\s/@]{1,128}(?=@[\w.-]+)"), 1.0),
]
# key = value assignments whose key names a secret; the value is the span
ASSIGN = _rx(
    r"""(?ix)(?<![\w.-])["']?(?P<key>[\w.-]+(?:[ ]de[ ]passe)?)
    ["']?\s*(?:=>|:=|=|:)""")
ASSIGN_VALUE = _rx(r"\s*[\"']?(?P<val>[^\s\"',;]{6,})")
SECRET_KEY = _rx(r"(?i)pass(?:word|wd|phrase)?|pwd|mot[_ ]?de[_ ]?passe|mdp|secret|token|api[_-]?key|apikey|"
                 r"access[_-]?key|private[_-]?key|client[_-]?secret|auth|credential|bearer")
AUTH_HEADER = _rx(r"(?i)\b(?:authorization\s*:\s*)?(?:bearer|basic)\s+(?P<val>[A-Za-z0-9._~+/=-]{16,})")
# home directories reveal the account name
PATH_USER = _rx(r"(?i)(?:[A-Z]:\\(?:Users|Documents and Settings)\\|/(?:home|Users)/)(?P<val>[^\\/\s\"'<>|:*?]{1,64})")
# street addresses (French and English street types), number first
ADDRESS = _rx(
    r"(?<![\w])\d{1,4}(?:[ \t]?(?:bis|ter|[a-c]))?,?[ \t]+"
    r"(?i:rue|avenue|av\.|boulevard|bd|bld|place|pl\.|chemin|impasse|all[ée]e|quai|route|cours|square|"
    r"passage|street|st\.|road|rd\.|ave\.|lane|ln\.|drive|dr\.|court|ct\.|way|terrace|close|"
    r"stra[ßs]e|calle|via|viale|piazza|straat|laan)[ \t]+"
    r"(?:(?i:de|du|des|la|le|les|l['’]|d['’])[ \t]*)*"
    r"[A-ZÀ-Ý][\w'’-]*(?:[ \t]+(?:(?i:de|du|des|la|le|les)[ \t]+)*[A-ZÀ-Ý][\w'’-]*){0,5}"
    r"(?:,?[ \t]+\d{5}[ \t]+[A-ZÀ-Ý][\w'’-]*(?:[ \t]+[A-ZÀ-Ý][\w'’-]*){0,2})?")
ADDRESS_EN = _rx(
    r"""(?x)(?<![\w])\d{1,5}\s+(?:[A-Z][\w'’.-]+\s+){1,3}
    (?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Way|Boulevard|Blvd\.?|
       Place|Pl\.?|Terrace|Close|Crescent|Square|Parkway|Highway)\b
    (?:,?\s+(?:Apt\.?|Suite|Unit|Flat)\s*\w+)?
    (?:,\s*[A-Z][\w'’ -]{1,30})?(?:,?\s+(?:[A-Z]{2}\s+)?(?:\d{5}(?:-\d{4})?|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}))?""")
# a password given in a sentence: « mon mot de passe (Wi-Fi) est X », "password is X", "password X"
PASSWORD_CUE = _rx(r"(?i)\b(?:mot\s+de\s+passe|mdp|password|passwd|passcode|code\s+pin|pin)(?:\s+[\w-]+){0,2}?"
                   r"\s*(?:est|is|:|=|\s)\s*[«\"'“]?\s*(?P<val>(?=[^\s]*[\d!?#$%&*@])[^\s«»\"'“”,;]{6,64})")
# user@host of ssh/scp/sftp commands (not caught by the e-mail rule when the host is an IP)
SSH_USER = _rx(r"\b(?:ssh|scp|sftp|rsync|mosh)\s+(?:-\S+(?:\s+\S+)?\s+)*?(?P<val>[A-Za-z_][\w.-]{0,31})@[\w.-]+")
_NAME = re.compile(r"(?:\b(?:[Jj]e m'appelle|[Jj]e suis|[Mm]on nom est|[Nn]om\s*:|[Nn]ame\s*:|[Mm]y name is|"
                   r"I am|I'm|[Mm]onsieur|[Mm]adame|"
                   r"[Mm]ademoiselle|Mme|Mlle|Mr|Mrs|Ms|Dr|Pr|Dear|Cher|Chère|Bonjour|Hello|Hi)\.?|\bM\.)[ \t]+"
                   r"([A-ZÀ-Ý][a-zà-ÿ'\-]+(?:[ \t]+[A-ZÀ-Ý][a-zà-ÿ'\-]+){0,2})")
_NOT_NAMES = frozenset({"Je", "I", "Le", "La", "Les", "The", "A", "Un", "Une", "Tout", "All", "Not", "Here",
                        "Ici", "Ok", "Merci", "Thanks", "Monsieur", "Madame"})
DOB = _rx(r"(?i)(?:n[ée]e?\s+le|date\s+de\s+naissance\s*:?|born\s+on|date\s+of\s+birth\s*:?|dob\s*:?)\s*"
          r"(?P<val>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+\w+\s+\d{4})")


def _iter(pattern: re.Pattern, text: str, typ: str, score: float, group: str | int = 0,
          check: Callable[[str], bool] | None = None) -> Iterator[Span]:
    for m in pattern.finditer(text):
        v = m.group(group)
        if v is None or (check and not check(v)):
            continue
        yield {"start": m.start(group), "end": m.end(group), "type": typ, "score": score}


def _card_ok(v: str) -> bool:
    d = re.sub(r"\D", "", v)
    return 13 <= len(d) <= 19 and luhn_ok(d) and len(set(d)) >= 2


def _ipv4_ok(v: str) -> bool:
    return v not in {"0.0.0.0", "127.0.0.1", "255.255.255.255"} and not v.startswith("127.")


VERSION_BEFORE = _rx(r"(?i)(?:version|release|v|pin|==|>=|<=|~=)\s*$")


def _not_version(text: str, spans: list[Span]) -> list[Span]:
    return [s for s in spans if not VERSION_BEFORE.search(text[max(0, s["start"] - 12):s["start"]])]


GUARD_TYPES = {"PERSON": "name", "EMAIL": "email", "PHONE": "phone", "ADDRESS": "other", "ID": "other",
               "FINANCIAL": "other", "SECRET": "api_key", "IP": "other", "USERNAME": "path", "DATE_OF_BIRTH": "other"}


def _ibans(text: str) -> Iterator[Span]:
    # Try checksum-valid endpoints from longest to shortest; trailing prose must not hide a valid IBAN.
    pos = 0
    while m := IBAN_CANDIDATE.search(text, pos):
        raw = m.group()
        pos = m.start() + 1  # invalid candidate: keep searching inside it, with guaranteed progress
        for n in range(len(raw), 3, -1):
            if not raw[n - 1].isalnum() or (m.start() + n < len(text) and text[m.start() + n].isalnum()):
                continue
            if iban_ok(raw[:n]):
                yield {"start": m.start(), "end": m.start() + n, "type": "FINANCIAL", "score": 1.0,
                       "guard_type": "iban"}
                pos = m.start() + n  # the overlong candidate may contain the next IBAN's start
                break


def _assignments(text: str) -> Iterator[Span]:
    # Extract a whole key once, then inspect its name. Never restart within a long dotted/hyphenated key.
    pos = 0
    while m := ASSIGN.search(text, pos):
        pos = m.end()
        if SECRET_KEY.search(m.group("key")) and (v := ASSIGN_VALUE.match(text, pos)):
            yield {"start": v.start("val"), "end": v.end("val"), "type": "SECRET", "score": 0.9}
            pos = v.end()


def matches(text: str) -> list[Span]:
    """Every canonical detection, unmerged: preferences must see all constituent fine types."""
    out: list[Span] = []
    out += _iter(EMAIL, text, "EMAIL", 1.0)
    out += _iter(NIR, text, "ID", 1.0, check=nir_ok)
    out += _iter(SSN_US, text, "ID", 0.9)
    out += _ibans(text)
    out += [dict(s, guard_type="card") for s in _iter(CARD, text, "FINANCIAL", 1.0, check=_card_ok)]
    out += _iter(PHONE_FR, text, "PHONE", 0.95)
    out += _iter(PHONE_INTL, text, "PHONE", 0.85)
    out += _iter(PHONE_US, text, "PHONE", 0.7)
    out += _not_version(text, list(_iter(IPV4, text, "IP", 0.8, check=_ipv4_ok)))
    out += _iter(IPV6, text, "IP", 0.9)
    for i, (p, score) in enumerate(SECRET_PATTERNS):
        out += [dict(s, guard_type="private_key" if i == 0 else "api_key")
                for s in _iter(p, text, "SECRET", score)]
    out += _iter(PASSWORD_CUE, text, "SECRET", 0.8, group="val")
    out += _iter(SSH_USER, text, "USERNAME", 0.8, group="val")
    out += _assignments(text)
    out += _iter(AUTH_HEADER, text, "SECRET", 0.9, group="val")
    out += _iter(PATH_USER, text, "USERNAME", 0.9, group="val")
    out += _iter(ADDRESS, text, "ADDRESS", 0.8)
    out += _iter(ADDRESS_EN, text, "ADDRESS", 0.8)
    out += _iter(_NAME, text, "PERSON", 0.7, group=1, check=lambda v: v.split()[0] not in _NOT_NAMES)
    out += _iter(DOB, text, "DATE_OF_BIRTH", 0.9, group="val")
    return out


def find(text: str) -> list[Span]:
    """Benchmark/model adapter: broad types and merged coverage from the same canonical rules."""
    return merge([{k: v for k, v in s.items() if k != "guard_type"} for s in matches(text)])


def merge(spans: list[Span]) -> list[Span]:
    """Merge overlapping spans; the merged span keeps the type of its highest-scoring part."""
    spans = sorted(spans, key=lambda s: (s["start"], -s["end"]))
    out: list[Span] = []
    for s in spans:
        if out and s["start"] < out[-1]["end"]:
            last = out[-1]
            if s["score"] > last["score"]:
                last["type"], last["score"] = s["type"], s["score"]
            last["end"] = max(last["end"], s["end"])
        else:
            out.append(dict(s))
    return out
