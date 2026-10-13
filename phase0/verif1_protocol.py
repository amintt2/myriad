"""Frozen, label-free VERIF-1 data, scoring and exclusive append-only accounting."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

VERSION = "VERIF-1-20261010-v3-A5"
AUTHORS = ("Qwen", "Gemma", "Granite", "SmolLM", "Mistral", "Phi", "OLMo")
MODELS = (
    "Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it", "ibm-granite/granite-4.2-3b",
    "HuggingFaceTB/SmolLM3-3B", "mistralai/Ministral-3-3B-Instruct-2512",
    "microsoft/Phi-4-mini-instruct", "allenai/OLMo-2-0425-1B-Instruct",
)
FAMILIES = dict(zip(MODELS, AUTHORS))
AUTHOR_DECODER = {"temperature": 0.7, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0}
JUDGE_DECODER = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0}
WEIGHTS = ((1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
AUTHOR_INSTRUCTION = (
    'Write fallible public tests from this contract only. Return ONLY JSON with exactly cases and properties. '
    'At most 4 each. Case: {"args":[JSON arguments],"expected":JSON value}. '
    'Property: {"left":[JSON arguments],"right":[JSON arguments],'
    '"relation":"equal|length_equal|permutation","domain":"why inputs satisfy preconditions",'
    '"uncertain":true|false}. Use only finite plain JSON values. No Python, expressions or extra keys. '
    'Set uncertain=true if the domain is not explicit. Do not infer universal properties. '
    'Keep the whole response within 512 tokens.\nCONTRACT:\n'
)
JUDGE_INSTRUCTION = (
    'Judge this anonymous candidate using only the contract, code and public execution observations below. '
    'Agreement is not correctness; tests may be wrong and domains uncertain. '
    'Return ONLY JSON with exactly {"score":number from 0 to 1,"reason":"brief justification"}. '
    'No additional keys. Keep within 256 tokens.\n'
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive(path: Path):
    """An orphan lock is deliberately not stolen, even when its recorded PID is absent."""
    write_new(path, {"pid": os.getpid(), "utc": utc()})
    try:
        yield
    finally:
        path.unlink()


def plain(value, depth=0, budget=None) -> None:
    budget = [1024] if budget is None else budget
    budget[0] -= 1
    if depth > 12 or budget[0] < 0:
        raise ValueError("JSON value exceeds structural limits")
    if value is None or type(value) in (bool, int):
        if type(value) is int and value.bit_length() > 256:
            raise ValueError("integer too large")
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is str and len(value) <= 4096:
        return
    if type(value) is list:
        for item in value:
            plain(item, depth + 1, budget)
        return
    if type(value) is dict and all(type(k) is str and len(k) <= 4096 for k in value):
        for item in value.values():
            plain(item, depth + 1, budget)
        return
    raise ValueError("only bounded finite JSON values allowed")


def args(value) -> None:
    if type(value) is not list or len(value) > 32:
        raise ValueError("arguments must be a bounded JSON list")
    plain(value)


def exact_keys(value, keys) -> None:
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("unexpected schema keys")


def strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    return json.loads(text, object_pairs_hook=pairs)


def parse_proposal(text: str) -> dict:
    if len(text) > 32768:
        raise ValueError("oversized proposal")
    proposal = strict_json(text)
    exact_keys(proposal, ("cases", "properties"))
    for kind in ("cases", "properties"):
        if type(proposal[kind]) is not list or len(proposal[kind]) > 4:
            raise ValueError("at most four cases/properties")
    for case in proposal["cases"]:
        exact_keys(case, ("args", "expected"))
        args(case["args"])
        plain(case["expected"])
    for prop in proposal["properties"]:
        exact_keys(prop, ("left", "right", "relation", "domain", "uncertain"))
        args(prop["left"])
        args(prop["right"])
        if (prop["relation"] not in ("equal", "length_equal", "permutation")
                or type(prop["uncertain"]) is not bool or type(prop["domain"]) is not str
                or not 0 < len(prop["domain"]) <= 1000):
            raise ValueError("invalid finite relation/domain")
    return proposal


def parse_judgment(text: str) -> dict:
    value = strict_json(text)
    exact_keys(value, ("score", "reason"))
    if (type(value["score"]) not in (int, float) or not 0 <= value["score"] <= 1
            or not math.isfinite(value["score"]) or type(value["reason"]) is not str
            or not 0 < len(value["reason"]) <= 4096):
        raise ValueError("invalid judgment")
    return value


def relation(left, right, name) -> bool:
    if name == "equal":
        return canonical(left) == canonical(right)
    if name == "length_equal":
        return (type(left) in (list, str, dict) and type(right) is type(left) and len(left) == len(right))
    if name == "permutation":
        return (type(left) is list and type(right) is list
                and Counter(map(canonical, left)) == Counter(map(canonical, right)))
    raise ValueError("unknown relation")


def shortlist(candidates: list[dict]) -> list[int]:
    """A2 public judge shortlist: family count, fixed catalogue/sample order, no learned weights."""
    groups = defaultdict(list)
    for i, cand in enumerate(candidates):
        sig = cand["public"]["extra"]
        valid = bool(sig) and all(isinstance(s, str) and re.fullmatch(r"(?:[0-9a-f]{16}|D[0-9a-f]{15})", s)
                                  for s in sig)
        groups[("signature", tuple(sig)) if valid else ("fallback", i)].append(i)

    def order(i):
        c = candidates[i]
        return MODELS.index(c["model"]), c["sample"], i

    def rank(group):
        return -len({candidates[i]["family"] for i in group}), min(order(i) for i in group)

    return [min(g, key=order) for g in sorted(groups.values(), key=rank)[:3]]


def supported_scores(candidates, proposals, matrix, threshold, kind):
    """Support never includes author OR evaluated candidate family. Matrix cells contain plain values."""
    if threshold not in (1, 2) or kind not in ("cases", "properties"):
        raise ValueError("undeclared variant")
    scores, coverage = [], []
    for i, target in enumerate(candidates):
        by_author = defaultdict(list)
        for p in proposals:
            author, item = p["author"], p["item"]
            if p["kind"] != kind or author == target["family"] or item.get("uncertain", False):
                continue
            passed = []
            for cells in matrix:
                left = cells[p["indices"][0]]
                if kind == "cases":
                    ok = left["ok"] and canonical(left["value"]) == canonical(item["expected"])
                else:
                    right = cells[p["indices"][1]]
                    ok = left["ok"] and right["ok"] and relation(left["value"], right["value"], item["relation"])
                passed.append(bool(ok))
            voters = {c["family"] for j, c in enumerate(candidates)
                      if c["family"] not in (author, target["family"])
                      and c["public"]["visible"] and passed[j]}
            missing = any(matrix[i][index].get("coverage_loss", False) for index in p["indices"])
            if len(voters) >= threshold and not missing:
                by_author[author].append(int(passed[i]))
        rates = [sum(v) / len(v) for v in by_author.values()]
        scores.append(sum(rates) / len(rates) if rates else None)
        coverage.append({a: len(v) for a, v in by_author.items()})
    return scores, coverage


def differential_scores(candidates, matrix, indices):
    representatives = {}
    for i, cand in enumerate(candidates):
        if cand["public"]["visible"]:
            representatives.setdefault(cand["family"], i)
    scores = []
    for i, cand in enumerate(candidates):
        peers = [j for family, j in representatives.items() if family != cand["family"]]
        rates = []
        for index in indices:
            eligible_peers = [j for j in peers if not matrix[j][index].get("coverage_loss", False)]
            if not eligible_peers or matrix[i][index].get("coverage_loss", False):
                continue
            cell = matrix[i][index]
            matches = sum(cell["ok"] and matrix[j][index]["ok"]
                          and canonical(cell["value"]) == canonical(matrix[j][index]["value"])
                          for j in eligible_peers)
            rates.append(matches / len(eligible_peers))
        scores.append(sum(rates) / len(rates) if rates else None)
    return scores


def author_acceptances(candidates, proposals, matrix, threshold):
    """Diagnostic only: all supported cases pass; missing evidence is absent, never a new selection rule."""
    result = []
    for i, target in enumerate(candidates):
        outcomes = defaultdict(list)
        for p in proposals:
            author = p["author"]
            if p["kind"] != "cases" or author == target["family"]:
                continue
            index = p["indices"][0]
            passed = [row[index]["ok"] and canonical(row[index]["value"]) == canonical(p["item"]["expected"])
                      for row in matrix]
            voters = {c["family"] for j, c in enumerate(candidates)
                      if c["family"] not in (author, target["family"]) and c["public"]["visible"] and passed[j]}
            if len(voters) >= threshold:
                outcomes[author].append(None if matrix[i][index].get("coverage_loss") else bool(passed[i]))
        result.append({a: (all(outcomes[a]) if outcomes[a] and None not in outcomes[a] else None)
                       for a in AUTHORS})
    return result


def combine(signals, weights):
    if tuple(weights) not in WEIGHTS:
        raise ValueError("unregistered combination")
    values = [(s, w) for s, w in zip(signals, weights) if s is not None and w]
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s, _ in values):
        raise ValueError("signal outside normalized range")
    return sum(s * w for s, w in values) / sum(w for _, w in values) if values else None


def mean_present(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def select(scores, baseline_order):
    present = [i for i in baseline_order if scores[i] is not None]
    return max(present, key=lambda i: scores[i]) if present else baseline_order[0]


def cost(n):
    if type(n) is not int or n <= 0:
        raise ValueError("positive question count required")
    authors, judges = n * len(AUTHORS) * 2 * 512, n * 6 * 256
    return {"questions": n, "author_calls": n * len(AUTHORS) * 2, "judge_calls_max": n * 6,
            "author_decode_cap": authors, "judge_decode_cap": judges, "decode_cap": authors + judges,
            "decode_gpu_hours_100_50_tps": [(authors + judges) / 360000, (authors + judges) / 180000],
            "prefill_cpu_load_bootstrap_extra": True, "monetary": "measured_hours * declared_rate"}


class Ledger:
    """Hash chained terminal results. Callers hold the campaign lock for the entire lifetime."""

    def __init__(self, path: Path, binding: str):
        self.path, self.binding = path, binding
        self.head, self.events, self.done, self.pending = "0" * 64, [], {}, {}
        if path.exists():
            raw = path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise ValueError("partial ledger line")
            for line in raw.splitlines():
                event = json.loads(line)
                digest = event.pop("sha256")
                if digest != sha(event) or event["previous"] != self.head or event["binding"] != binding:
                    raise ValueError("ledger chain/binding mismatch")
                self._accept(event)
                self.head = digest
        if self.pending:
            raise ValueError("unfinished reservation: fail-closed, no retry")

    def _accept(self, event):
        key = event["key"]
        if event["kind"] == "reserve":
            if key in self.pending or key in self.done:
                raise ValueError("duplicate ledger operation")
            self.pending[key] = event
        elif event["kind"] == "result":
            reserve = self.pending.pop(key)
            if event["input_sha256"] != reserve["input_sha256"]:
                raise ValueError("ledger input mismatch")
            if event["status"] == "ok" and event.get("output_sha256") != sha(event["output"]):
                raise ValueError("ledger output hash mismatch")
            self.done[key] = event
        else:
            raise ValueError("unknown ledger event")
        self.events.append(event)

    def append(self, kind, key, input_hash, **fields):
        event = {"kind": kind, "key": key, "input_sha256": input_hash, "binding": self.binding,
                 "previous": self.head, "utc": utc(), **fields}
        digest = sha(event)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical({**event, "sha256": digest}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._accept(event)
        self.head = digest

    def operation(self, key, inputs, action, ceiling=0):
        digest = sha(inputs)
        if key in self.done:
            event = self.done[key]
            if event["input_sha256"] != digest:
                raise ValueError("resumed input changed")
            if event["status"] != "ok":
                raise ValueError("infrastructure failure is terminal")
            return event["output"]
        self.append("reserve", key, digest, inputs=inputs, token_ceiling=ceiling)
        started = time.monotonic()
        try:
            output = action()
        except Exception as error:
            self.append("result", key, digest, status="error", error=type(error).__name__,
                        seconds=time.monotonic() - started, prompt_tokens=None, completion_tokens=None,
                        token_ceiling=ceiling)
            raise
        self.append("result", key, digest, status="ok", output=output, output_sha256=sha(output),
                    seconds=time.monotonic() - started)
        return output


def confirmation_open(registry: Path, variant: str, frozen_dev: Path, expected_sha: str,
                      bindings: dict, test_path: Path):
    """Reserve permanently BEFORE opening. Parent must use a single shared registry for this campaign."""
    if variant not in {f"V{i:02d}" for i in range(1, 8)}:
        raise ValueError("unknown variant")
    if len(expected_sha) != 64 or file_sha(frozen_dev) != expected_sha:
        raise ValueError("frozen dev configuration SHA required")
    frozen = json.loads(frozen_dev.read_text(encoding="utf-8"))
    exact_keys(bindings, ("code_sha256", "plan_sha256", "data_sha256"))
    if (frozen.get("split") != "dev" or frozen.get("variant") != variant
            or frozen.get("bindings") != bindings or not frozen.get("configuration")):
        raise ValueError("dev freeze and confirmation bindings differ")
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
           for v in bindings.values()):
        raise ValueError("invalid confirmation hashes")
    write_new(registry / f"VERIF-1-{variant}.opened.json",
              {"variant": variant, "dev_sha256": expected_sha, "bindings": bindings, "utc": utc(),
               "test_path": str(test_path.resolve()), "consumed": True})
    return test_path.open("rb")
