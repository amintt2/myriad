"""Fusion of the answers of several peers. Pure functions, no I/O.

- extract_answer: the final number (GSM8K style) or the multiple-choice letter of a text.
- reliability_weight: K-class Nitzan-Paroush weight w = logit(p) - ln(c), clipped at >= 0, where p is the
  model's reliability and c the probability that two wrong peers give the SAME wrong answer (c = 1 gives
  the binary log-odds ln(p / (1 - p)); c = 1 / (K - 1) for K options; c ~ 0.05 for open numeric answers).
  Under independent errors this is the optimal weighted majority rule (Nitzan & Paroush, 1982). With
  binary weights on open answers, a single strong peer outweighs all the others (phase-0 replay, GSM8K
  dev: 82 % with c = 1 against 87 % with c <= 0.1).
- weighted_vote: sum of the weights per answer, ties broken by the supporters' mean log-probability.
- stop_certificate: the answer that no set of missing answers can overturn any more
  (docs/03_idees_codex.md, idea 2): answer a is final once L_a > max_{b != a} L_b + W, where L are the
  partial vote sums and W the total weight of the peers that have not answered yet.
- medoid: for free text, the answer closest to the others (weighted word-multiset Jaccard).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

W_MIN, W_MAX = 0.0, 10.0
_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_OPEN_TAGS, _CLOSE_TAGS = ("<think>", "<|channel>"), ("</think>", "<channel|>")
# "The answer is 42", "Final answer: $42", "la réponse est 42", "**The answer is 16.**"
_ANSWER_NUM = re.compile(r"(?i:answer|r[ée]ponse)\s*(?i:is|est|:|=)\s*[:\s]*\**\s*\$?\s*\\?\(?\s*(" + _NUM + r")")
_BOXED_NUM = re.compile(r"\\boxed\{\s*\$?\s*(" + _NUM + r")")
_ANY_NUM = re.compile(_NUM)
_ANSWER_LETTER = re.compile(r"(?i:answer|r[ée]ponse)\s*(?i:is|est|:|=)\s*[:\s]*\**\s*\(?([A-J])\)?(?![A-Za-z0-9])")
_BOXED_LETTER = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-J])\)?\s*\}?\s*\}")
_LONE_LETTER = re.compile(r"^\s*\**\s*\(?([A-J])\)?\s*\**\s*[\.\):]?\s*\**\s*$")
_MC_OPTION = re.compile(r"^\s*\(?([A-J])[\).:]\s+\S", re.M)
_MATH_CUES = re.compile(
    r"how (?:many|much|long|far|old)|calculate|compute|solve|what is the (?:total|sum|value|result|product|cost)"
    r"|combien|calcul|r[ée]sou|quel est le (?:total|r[ée]sultat|prix)|\d\s*[-+*/×x÷^]\s*\d", re.I)
MAX_NUMBER_CHARS = 40


def visible_text(text: str) -> str:
    """The text after the last closed thinking block, minus any thinking block still open."""
    for tag in _CLOSE_TAGS:
        if tag in text:
            text = text.split(tag)[-1]
    for tag in _OPEN_TAGS:
        if tag in text:
            text = text.split(tag)[0]
    return text


def normalize_number(s: str) -> str | None:
    s = s.replace(",", "").strip()
    if not s or len(s) > MAX_NUMBER_CHARS:
        return None
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    if not d.is_finite():
        return None
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def extract_answer(text: str | None, task_hint: str | None) -> str | None:
    """Final answer of `text` for a 'math' or 'mc' task; None for free text or when absent."""
    if not text or task_hint not in ("math", "mc"):
        return None
    vis = visible_text(text)
    if task_hint == "math":
        for rx in (_ANSWER_NUM, _BOXED_NUM, _ANY_NUM):
            found = rx.findall(vis)
            for cand in reversed(found):
                n = normalize_number(cand)
                if n is not None:
                    return n
        return None
    for rx in (_ANSWER_LETTER, _BOXED_LETTER):
        found = rx.findall(vis)
        if found:
            return found[-1].upper()
    lines = [ln for ln in vis.strip().splitlines() if ln.strip()]
    for ln in (lines[-1:] + lines[:1]) if lines else []:  # a lone letter as the last (or only) line
        m = _LONE_LETTER.match(ln)
        if m:
            return m.group(1)
    return None


def detect_task_hint(messages: list[dict]) -> str:
    """'mc' when the last user message lists options A), B)...; 'math' for a numeric question; else 'free'."""
    user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    letters = {m.group(1) for m in _MC_OPTION.finditer(user)}
    if {"A", "B"} <= letters:
        return "mc"
    if re.search(r"\d", user) and _MATH_CUES.search(user):
        return "math"
    return "free"


FORMAT_INSTRUCTIONS = {
    "math": 'Solve it step by step, briefly. Finish with the sentence: "The answer is N." where N is a number.',
    "mc": 'Think briefly, then finish with the sentence: "The answer is X." where X is the letter of the correct option.',
}


def reliability_weight(p: float, collision: float = 1.0, w_min: float = W_MIN, w_max: float = W_MAX) -> float:
    """w = logit(p) - ln(collision), clipped to [w_min, w_max]: never negative, a weak model never votes against."""
    if p is None or not math.isfinite(p):
        p = 0.5
    p = min(max(p, 1e-6), 1 - 1e-6)
    c = collision if collision is not None and math.isfinite(collision) else 1.0
    c = min(max(c, 1e-6), 1.0)
    return min(max(math.log(p / (1 - p)) - math.log(c), w_min), w_max)


def count_options(messages: list[dict]) -> int:
    """Number of options A), B), ... listed in the last user message (at least 2)."""
    user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    return max(2, len({m.group(1) for m in _MC_OPTION.finditer(user)}))


def _tol(*xs: float) -> float:
    return 1e-12 * (1.0 + sum(abs(x) for x in xs))


@dataclass
class VoteResult:
    answer: str | None
    scores: dict[str, float] = field(default_factory=dict)
    representative: int | None = None  # index of the answer whose text is returned
    tie_broken: bool = False


def weighted_vote(answers: list[str | None], weights: list[float],
                  confidences: list[float | None] | None = None) -> VoteResult:
    """Weighted plurality. None answers abstain. Exact ties (up to float noise) are broken by the
    highest mean log-probability among the supporters, then by the answer string (deterministic)."""
    if len(answers) != len(weights):
        raise ValueError("answers and weights differ in length")
    conf = list(confidences) if confidences is not None else [None] * len(answers)
    scores: dict[str, float] = {}
    for a, w in zip(answers, weights):
        if w < 0 or not math.isfinite(w):
            raise ValueError("weights must be finite and >= 0")
        if a is not None:
            scores[a] = scores.get(a, 0.0) + w
    if not scores:
        return VoteResult(None)
    top = max(scores.values())
    tol = _tol(*scores.values())
    tied = sorted(a for a, s in scores.items() if top - s <= tol)

    def best_conf(a: str) -> float:
        cs = [c for x, c in zip(answers, conf) if x == a and c is not None and math.isfinite(c)]
        return max(cs) if cs else -math.inf

    winner = min(tied, key=lambda a: (-best_conf(a), a))
    rep = max((i for i, a in enumerate(answers) if a == winner),
              key=lambda i: (weights[i], conf[i] if conf[i] is not None else -math.inf, -i))
    return VoteResult(winner, scores, rep, len(tied) > 1)


def certificate_margin(*xs: float) -> float:
    """Safety margin of the certificate; much larger than the tie tolerance of weighted_vote, so float
    rounding in sums taken in different orders can never turn a certified answer into a tie."""
    return 1e-9 * (1.0 + sum(abs(x) for x in xs))


def stop_certificate(partial_votes: dict[str, float], remaining_weight: float) -> str | None:
    """Answer a such that L_a > max(max_{b != a} L_b, 0) + W: whatever the missing peers answer
    (or if they abstain), a stays the strict winner of the full weighted vote. None otherwise."""
    if not partial_votes:
        return None
    if remaining_weight < 0 or not math.isfinite(remaining_weight):
        raise ValueError("remaining_weight must be finite and >= 0")
    ranked = sorted(partial_votes.items(), key=lambda kv: (-kv[1], kv[0]))
    a, la = ranked[0]
    runner = max(ranked[1][1], 0.0) if len(ranked) > 1 else 0.0  # an unseen answer starts at 0
    if la > runner + remaining_weight + certificate_margin(*partial_votes.values(), remaining_weight):
        return a
    return None


_WORD = re.compile(r"\w+", re.UNICODE)


def _bag(text: str) -> Counter:
    return Counter(_WORD.findall(text.lower()))


def similarity(a: Counter, b: Counter) -> float:
    """Multiset Jaccard: sum of minimum counts over sum of maximum counts (1.0 for two empty texts)."""
    keys = a.keys() | b.keys()
    if not keys:
        return 1.0
    return sum(min(a[k], b[k]) for k in keys) / sum(max(a[k], b[k]) for k in keys)


def medoid(texts: list[str | None], weights: list[float]) -> int | None:
    """Index of the weighted medoid of the non-empty texts; None if all are empty."""
    idx = [i for i, t in enumerate(texts) if t and t.strip()]
    if not idx:
        return None
    if len(idx) == 1:
        return idx[0]
    bags = {i: _bag(visible_text(texts[i])) for i in idx}

    def centrality(i: int) -> float:
        # Weighted medoid: minimises sum_j w_j (1 - sim(i, j)) over ALL answers, its own included
        # (sim = 1): without the self term, the least reliable of two similar answers would win.
        return sum(weights[j] * (1.0 if j == i else similarity(bags[i], bags[j])) for j in idx)

    return max(idx, key=lambda i: (centrality(i), weights[i], -i))
