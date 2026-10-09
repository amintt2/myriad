"""Benchmark samples, fixed seed, cached under phase0/data/.

Two fixed, disjoint partitions per benchmark, from the same seeded shuffle, whatever the
number of questions requested (n only takes the first n of a partition):
  dev  = shuffled items [0, PART)        used to choose the fusion method
  test = shuffled items [PART, 2*PART)   touched only once the method is frozen
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from .common import final_answer

DATA = Path(__file__).resolve().parent.parent / "data"
LETTERS = "ABCDEFGHIJ"
SPLITS = ("dev", "test")


REPOS = {"arc": "allenai/ai2_arc", "mmlupro": "TIGER-Lab/MMLU-Pro", "gsm8k": "openai/gsm8k",
         "math500": "HuggingFaceH4/MATH-500", "mtbench": "HuggingFaceH4/mt_bench_prompts"}
# Dataset commits pinned for the whole phase 0, on every machine.
REVISIONS = {"arc": "210d026faf9955653af8916fad021475a3f00453",
             "mmlupro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
             "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
             "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
             "mtbench": "e3a795c5e9a82ee40611c416b8a7786c73198991"}
_PREFIX = {"arc": "ARC-Challenge/test", "mmlupro": "data/test", "gsm8k": "main/test"}


def _parquets(bench: str) -> list[str]:
    repo, rev = REPOS[bench], REVISIONS[bench]
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
    return [hf_hub_download(repo, f, repo_type="dataset", revision=rev)
            for f in sorted(files) if f.startswith(_PREFIX[bench]) and f.endswith(".parquet")]


def _canonical(items: list[dict]) -> bytes:
    """Platform-independent bytes of the cached questions (LF, sorted keys, UTF-8)."""
    return "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in items).encode("utf-8")


def _rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        rows += pq.read_table(p).to_pylist()
    return rows


def _cached(name: str, build, bench: str) -> list[dict]:
    """Build once from the pinned dataset revision; the cache records that revision and the SHA-256
    of its canonical bytes (identical on Windows and macOS)."""
    DATA.mkdir(exist_ok=True)
    f = DATA / f"{name}_v2.jsonl"
    meta = DATA / f"{name}_v2.meta.json"
    if not f.exists() or not meta.exists():
        raw = _canonical(build())
        tmp = f.with_name(f.name + f".tmp{os.getpid()}")
        tmp.write_bytes(raw)
        os.replace(tmp, f)  # atomic: a reader never sees a half-written file
        m = {"dataset": REPOS[bench], "revision": REVISIONS[bench], "sha256": hashlib.sha256(raw).hexdigest()}
        meta.write_bytes((json.dumps(m, indent=2) + "\n").encode("utf-8"))
    raw = f.read_bytes()
    m = json.loads(meta.read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != m["sha256"] or m.get("revision") != REVISIONS[bench]:
        raise SystemExit(f"{f.name} ne correspond plus à son empreinte ou à la révision fixée : supprime-le pour le reconstruire")
    _IDENTITY[bench] = m
    return [json.loads(l) for l in raw.decode("utf-8").splitlines()]


_IDENTITY: dict[str, dict] = {}


def dataset_identity(bench: str) -> dict | None:
    """Dataset revision and content hash of the questions actually loaded (call after loading)."""
    return _IDENTITY.get(bench)


PART = {"arc": 300, "mmlupro": 300, "gsm8k": 300, "math500": 250}


def _split(items: list[dict], n: int, split: str, bench: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    part = PART[bench]
    if not 0 < n <= part:
        raise ValueError(f"n={n} : la partition {split} de {bench} a {part} questions")
    lo = 0 if split == "dev" else part
    out = items[lo: lo + part][:n]
    if len(out) < n:
        raise ValueError(f"not enough items for {split} ({len(out)} < {n})")
    return out


def arc(n: int = 300, seed: int = 0, split: str = "dev") -> list[dict]:
    def build():
        rows = _rows(_parquets("arc"))
        random.Random(seed).shuffle(rows)
        return [{"id": r["id"], "question": r["question"], "options": r["choices"]["text"],
                 "answer": r["choices"]["label"].index(r["answerKey"])}
                for r in rows if len(r["choices"]["label"]) == 4 and r["answerKey"] in r["choices"]["label"]]
    return _split(_cached(f"arc_all_s{seed}", build, "arc"), n, split, "arc")


def mmlu_pro(n: int = 300, seed: int = 0, split: str = "dev") -> list[dict]:
    def build():
        rows = [r for r in _rows(_parquets("mmlupro")) if len(r["options"]) == 10]
        random.Random(seed).shuffle(rows)
        return [{"id": str(r["question_id"]), "question": r["question"], "options": r["options"],
                 "answer": int(r["answer_index"]), "category": r["category"]} for r in rows]
    return _split(_cached(f"mmlupro_all_s{seed}", build, "mmlupro"), n, split, "mmlupro")


def gsm8k(n: int = 60, seed: int = 0, split: str = "dev") -> list[dict]:
    def build():
        rows = [{"id": f"gsm8k-test-{i}", **r} for i, r in enumerate(_rows(_parquets("gsm8k")))]
        random.Random(seed).shuffle(rows)  # ids are stable: the row index before shuffling
        return [{"id": r["id"], "question": r["question"],
                 "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for r in rows]
    return _split(_cached(f"gsm8k_all_s{seed}", build, "gsm8k"), n, split, "gsm8k")


def math500(n: int = 250, seed: int = 0, split: str = "dev") -> list[dict]:
    """MATH-500 (the 500-problem subset of MATH used by most reports): 250 dev, 250 test."""
    def build():
        path = hf_hub_download(REPOS["math500"], "test.jsonl", repo_type="dataset", revision=REVISIONS["math500"])
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        random.Random(seed).shuffle(rows)
        return [{"id": r["unique_id"], "question": r["problem"], "answer": r["answer"],
                 "subject": r["subject"], "level": r["level"]} for r in rows]
    return _split(_cached(f"math500_all_s{seed}", build, "math500"), n, split, "math500")


# E10 (Skeleton-of-Thought): MT-Bench first turns (80 prompts, 10 per category, Apache-2.0).
# Kept: the categories whose good answer is a long free-form text made of several separable parts.
# Set apart ("other", reported separately): math, coding, extraction (one exact result or format,
# a skeleton would split a single chain of computation) and reasoning (short logic puzzles whose
# answer is one sentence). Ning et al. (2023) report that SoT suits answers made of independent
# points and hurts questions that need step-by-step reasoning (math, coding, Fermi estimates).
MT_KEPT = ("writing", "roleplay", "stem", "humanities")
MT_SPLITS = ("dev", "test", "other")
MT_PART = {"dev": 16, "test": 24}  # 40 kept prompts; dev only picks the "best single model"


def mt_bench(n: int | None = None, seed: int = 0, split: str = "dev") -> list[dict]:
    """MT-Bench first-turn prompts. dev/test: a fixed seeded shuffle of the kept categories
    (16 + 24); other: the excluded categories in question order (40)."""
    if split not in MT_SPLITS:
        raise ValueError(f"split must be one of {MT_SPLITS}")

    def build():
        path = hf_hub_download(REPOS["mtbench"], "raw/question.jsonl", repo_type="dataset",
                               revision=REVISIONS["mtbench"])
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows.sort(key=lambda r: r["question_id"])
        kept = [r for r in rows if r["category"] in MT_KEPT]
        random.Random(seed).shuffle(kept)
        other = [r for r in rows if r["category"] not in MT_KEPT]
        cut = MT_PART["dev"]
        return [{"id": f"mtbench-{r['question_id']}", "question": r["prompt"][0], "category": r["category"],
                 "split": "dev" if i < cut else "test" if r["category"] in MT_KEPT else "other"}
                for i, r in enumerate(kept + other)]
    items = [x for x in _cached(f"mtbench_all_s{seed}", build, "mtbench") if x["split"] == split]
    if split in MT_PART and len(items) != MT_PART[split]:
        raise ValueError(f"MT-Bench {split}: {len(items)} prompts, {MT_PART[split]} attendus")
    if n is not None and not 0 < n <= len(items):
        raise ValueError(f"n={n} : la partition {split} de MT-Bench a {len(items)} prompts")
    return items[:n] if n else items


def mc_prompt(item: dict) -> tuple[str, list[str]]:
    letters = list(LETTERS[: len(item["options"])])
    body = "\n".join(f"{L}. {o}" for L, o in zip(letters, item["options"]))
    user = (f"{item['question']}\n\n{body}\n\n"
            f"Reply with the letter of the correct answer only ({letters[0]}-{letters[-1]}).")
    return user, letters


def balanced_perm(m: int, run: int, runs: int) -> list[int]:
    """Option order shown in pass `run`: perm[pos] = original index at displayed position pos.

    Cyclic rotations spread each option over distinct positions (a Latin-square design);
    once all m rotations are used, the reversed order is rotated. At most 2m distinct orders.
    The rotation step is coprime with m (v2): with step 2 and 10 options, an option only ever
    visited positions of its own parity, so a position bias could not average out and the
    option-position graph was disconnected (docs/03_idees_codex.md)."""
    limit = 2 * m if m > 2 else math.factorial(m)  # with 2 options, the reversed order is a rotation
    if runs > limit:
        raise ValueError(f"{runs} passages pour {m} options : au plus {limit} ordres distincts")
    step = max(1, m // runs) if runs <= m else 1
    while math.gcd(step, m) != 1:
        step += 1
    if run < m:  # rotations of the original order: distinct, since the step is coprime with m
        base, shift = list(range(m)), (run * step) % m
    else:
        base, shift = list(reversed(range(m))), ((run - m) * step) % m
    return base[shift:] + base[:shift]


def extract_number(text: str, ended: bool = False) -> str | None:
    """GSM8K: last explicitly finalised answer (see common.final_answer); None if absent."""
    return final_answer(text, ended)
