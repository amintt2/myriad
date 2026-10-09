"""E10, parallel sections (Skeleton-of-Thought across peers of different families).

One peer writes a numbered skeleton of the answer; point i is then expanded by peer i mod P, all
points at once; the answer is the points and their expansions, in order. Shared by run_sot.py
(generation), judge_sot.py (pairwise judge) and analyze_sot.py (quality and speed model).

Prompts follow the two-stage scheme of Ning et al. (2023), "Skeleton-of-Thought", with our own wording,
3 to 8 points of at most 12 words, and expansions of 1 to 3 short paragraphs.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

from .common import visible_answer_text
from .results import read_manifest, read_rows

RESULTS = Path(__file__).resolve().parent.parent / "results"
PROMPT_VERSION = "sot-v1"  # bump whenever a generation prompt, the parsing or the assembly changes
JUDGE_VERSION = "sot-judge-v1"
MIN_POINTS, MAX_POINTS, MAX_POINT_WORDS = 3, 8, 12
BASE_MAX_TOKENS, OUTLINE_MAX_TOKENS, EXPAND_MAX_TOKENS = 1024, 256, 256
SEED = 7


def slug(model: str) -> str:
    return model.replace("/", "__")


def base_path(model: str, suffix: str, split: str) -> Path:
    return RESULTS / f"sot_base_{slug(model)}{suffix}_{split}.jsonl"


def outline_path(model: str, suffix: str, split: str) -> Path:
    return RESULTS / f"sot_outline_{slug(model)}{suffix}_{split}.jsonl"


def expand_path(tag: str, model: str, suffix: str, split: str) -> Path:
    return RESULTS / f"sot_expand_{tag}_{slug(model)}{suffix}_{split}.jsonl"


def judge_path(tag: str, judge: str, suffix: str, split: str) -> Path:
    return RESULTS / f"sot_judge_{tag}_{slug(judge)}{suffix}_{split}.jsonl"


# ---------- prompts ----------

def baseline_prompt(question: str) -> str:
    return question


def outline_prompt(question: str) -> str:
    return ("You are planning the answer to the question below, not writing it. Give only the skeleton of a "
            f"good answer: a numbered list of {MIN_POINTS} to {MAX_POINTS} points (1., 2., 3., ...). Each point is a "
            f"short phrase of at most {MAX_POINT_WORDS} words, not a full sentence. Write nothing else: no title, "
            "no introduction, no explanation.\n\n"
            f"Question:\n{question}")


def skeleton_text(points: list[str]) -> str:
    return "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))


def expand_prompt(question: str, points: list[str], i: int) -> str:
    """Point i (0-based) of the skeleton."""
    return ("You are writing one part of a longer answer to the question below. Other writers handle the other "
            "points of the skeleton, at the same time.\n\n"
            f"Question:\n{question}\n\n"
            f"Skeleton of the full answer:\n{skeleton_text(points)}\n\n"
            f"Write ONLY point {i + 1} (\"{points[i]}\"), in 1 to 3 short paragraphs. Do not repeat the point's "
            "title or number, do not write an introduction or a conclusion for the whole answer, and do not "
            "cover the other points.")


# ---------- skeleton parsing ----------

_NUMBERED = re.compile(r"^\s*(?:[#>*_]+\s*)*(?:point|step)?\s*(\d{1,2})\s*(?:[.):]|\s-)\s*(.*)$", re.I)
_BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
_MARKUP = re.compile(r"[*_`#]+")


def _clean(point: str) -> str:
    p = _MARKUP.sub("", point).strip()
    p = re.sub(r"\s+", " ", p).strip(" :;-–—")
    return p


def parse_skeleton(text: str) -> dict:
    """Points of a skeleton answer. Reads the first numbered list 1, 2, 3, ... (markdown tolerated,
    other lines ignored, a second list starting again at 1 ends the first); without one, a bullet
    list at column 0. status: "ok", "truncated" (more than MAX_POINTS: the first MAX_POINTS are kept),
    "bullets" (read from bullets), or "fallback" (fewer than MIN_POINTS points: no skeleton)."""
    text = visible_answer_text(text)
    points, expect = [], 1
    for line in text.splitlines():
        m = _NUMBERED.match(line)
        if not m:
            continue
        n, p = int(m.group(1)), _clean(m.group(2))
        if n == expect and p:
            points.append(p)
            expect += 1
        elif n == 1 and points:
            break  # a second list (e.g. sub-points numbered again): the skeleton is the first one
    status = "ok"
    if len(points) < MIN_POINTS:
        bullets = [_clean(m.group(1)) for line in text.splitlines()
                   if not line.startswith((" ", "\t")) and (m := _BULLET.match(line))]
        bullets = [b for b in bullets if b]
        if len(bullets) >= MIN_POINTS:
            points, status = bullets, "bullets"
    if len(points) > MAX_POINTS:
        points, status = points[:MAX_POINTS], "truncated"
    if len(points) < MIN_POINTS:
        return {"points": [], "status": "fallback", "n_long": 0}
    return {"points": points, "status": status,
            "n_long": sum(len(p.split()) > MAX_POINT_WORDS for p in points)}


# ---------- assignment and assembly ----------

def assign(n_points: int, peers: list[str]) -> list[str]:
    """Peer of each point: round-robin, point i to peer i mod P."""
    return [peers[i % len(peers)] for i in range(n_points)]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def clean_expansion(text: str, i: int, point: str) -> str:
    """The expansion body: thinking removed, and a first line that only repeats the point's number
    and/or title dropped (the assembled answer already shows them)."""
    body = visible_answer_text(text).strip()
    lines = body.split("\n", 1)
    first = _clean(re.sub(rf"^\s*(?:[#>*_]+\s*)*(?:point\s*)?{i + 1}\s*[.):]?\s*", "", lines[0], flags=re.I))
    if _norm(first) in ("", _norm(point)) and len(lines) > 1:
        body = lines[1].strip()
    return body


def assemble(points: list[str], expansions: list[str]) -> str:
    return "\n\n".join(f"**{i + 1}. {p}**\n\n{clean_expansion(e, i, p)}"
                       for i, (p, e) in enumerate(zip(points, expansions)))


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def complete(path: Path, ids: list[str], key: tuple[str, ...] = ("id",), what: str = "") -> list[dict]:
    """Rows of a finished stage file; refuses a missing file, a file without manifest, or missing ids."""
    if not path.exists() or read_manifest(path) is None:
        raise SystemExit(f"{path.name} absent (ou sans manifeste) : lancer d'abord l'étape {what}")
    rows = read_rows(path, key)
    missing = sorted(set(ids) - {r["id"] for r in rows})
    if missing:
        raise SystemExit(f"{path.name} incomplet : {len(missing)} prompts manquent (ex. {missing[0]})")
    return [r for r in rows if r["id"] in set(ids)]


def outline_inputs(outline_model: str, suffix: str, split: str, ids: list[str]) -> tuple[dict, str]:
    """Parsed skeletons by id, and their fingerprint (what the expansions were asked to write)."""
    rows = {r["id"]: r for r in complete(outline_path(outline_model, suffix, split), ids, what="outline")}
    sk = {i: {"points": rows[i]["points"], "status": rows[i]["status"]} for i in ids}
    return rows, canonical_sha(sk)


def load_sot(tag: str, outline_model: str, peers: list[str], suffix: str, split: str, ids: list[str]) -> dict:
    """The parallel answer of every prompt, with what the speed model needs.

    A prompt whose skeleton could not be read falls back to the outline model answering alone
    (its baseline answer), as a deployed system would; it is flagged `fallback`."""
    outlines, sha = outline_inputs(outline_model, suffix, split, ids)
    exp = {}
    for m in dict.fromkeys(peers):
        p = expand_path(tag, m, suffix, split)
        man = read_manifest(p)
        if man is None:
            raise SystemExit(f"{p.name} absent (ou sans manifeste) : lancer d'abord l'étape expand de {m}")
        if man.get("peers") != peers or man.get("outline_model") != outline_model or man.get("outline_sha256") != sha:
            raise SystemExit(f"{p.name} : expansions faites avec d'autres pairs, un autre squelette ou un autre --n")
        for r in read_rows(p, ("id", "point")):
            exp[(r["id"], r["point"])] = r
    base_o = None
    out = {}
    for i in ids:
        o = outlines[i]
        if o["status"] == "fallback":
            if base_o is None:
                base_o = {r["id"]: r for r in complete(base_path(outline_model, suffix, split), ids, what="baseline")}
            b = base_o[i]
            out[i] = {"text": b["text"], "fallback": True, "points": [], "outline_tokens": o["n_tokens"],
                      "outline_prompt_tokens": o.get("prompt_tokens"), "fallback_tokens": b["n_tokens"],
                      "fallback_prompt_tokens": b.get("prompt_tokens"), "expansions": []}
            continue
        rows = []
        for k, peer in enumerate(assign(len(o["points"]), peers)):
            r = exp.get((i, k))
            if r is None or r["model"] != peer:
                raise SystemExit(f"expansion manquante : {i} point {k + 1} ({peer})")
            rows.append(r)
        out[i] = {"text": assemble(o["points"], [r["text"] for r in rows]), "fallback": False,
                  "points": o["points"], "outline_tokens": o["n_tokens"],
                  "outline_prompt_tokens": o.get("prompt_tokens"),
                  "expansions": [{"model": r["model"], "n_tokens": r["n_tokens"], "finish": r["finish"],
                                  "prompt_tokens": r.get("prompt_tokens")} for r in rows]}
    return out


# ---------- one chat call ----------

def chat(http, messages: list[dict], max_tokens: int, **extra) -> dict:
    """One greedy chat completion (thinking off) on our llama-server; wall time is measured while
    the server batches other requests, so it is not a single-stream latency."""
    t0 = time.perf_counter()
    r = http.post("/v1/chat/completions", json={
        "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": False}, **extra})
    if r.status_code != 200:
        raise RuntimeError(f"llama-server {r.status_code}: {r.text[:500]}")
    j = r.json()
    ch, usage, t = j["choices"][0], j.get("usage") or {}, j.get("timings") or {}
    return {"text": ch["message"].get("content") or "", "reasoning": bool(ch["message"].get("reasoning_content")),
            "finish": ch.get("finish_reason"), "n_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"), "ms": round((time.perf_counter() - t0) * 1000),
            "decode_ms": t.get("predicted_ms"), "prompt_ms": t.get("prompt_ms"),
            "logprobs": (ch.get("logprobs") or {}).get("content")}


# ---------- pairwise judge ----------

JUDGE_GRAMMAR = 'root ::= "A" | "B" | "C"'
JUDGE_TOP = 20


def judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    """Pairwise comparison in the style of MT-Bench (Zheng et al., 2023), with a one-letter verdict."""
    return ("Please act as an impartial judge and evaluate the quality of the responses provided by two AI "
            "assistants to the user question displayed below. Choose the assistant that follows the user's "
            "instructions and answers the user's question better. Consider helpfulness, relevance, accuracy, "
            "depth, creativity, coherence and level of detail. Avoid any position bias: the order in which the "
            "responses are presented must not influence your decision. Do not let the length of the responses "
            "influence your evaluation, and do not favour any particular formatting. Be as objective as "
            "possible.\n\n"
            f"[User Question]\n{question}\n\n"
            f"[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
            f"[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]\n\n"
            "Output your final verdict as a single letter, and nothing else: A if assistant A is better, "
            "B if assistant B is better, C for a tie.")


def letter_probs(top: list[dict] | None) -> dict | None:
    """Probabilities of A, B and C (renormalised over the three) from the raw (pre-grammar) top
    log-probabilities of the verdict token; a letter absent from the top list counts as 0.
    Returns {"A", "B", "C", "mass"} where mass is the raw probability of the three letters."""
    if not top:
        return None
    p = {"A": 0.0, "B": 0.0, "C": 0.0}
    for t in top:
        s = (t.get("token") or "").strip()
        if s in p and t.get("logprob") is not None:
            p[s] += math.exp(t["logprob"])
    mass = sum(p.values())
    if mass <= 0:
        return None
    return {**{k: v / mass for k, v in p.items()}, "mass": mass}


def sot_outcome(verdict: str | None, sot_is: str) -> float | None:
    """Score of the parallel answer for one display order: 1 win, 0.5 tie, 0 loss."""
    if verdict == "C":
        return 0.5
    if verdict in ("A", "B"):
        return 1.0 if verdict == sot_is else 0.0
    return None
