"""E18: Lean-checked proofs by a swarm of small open provers (docs/14_e18_lean.md). Tables, data, prompts and plan.

Pure data and helpers, shared by e18_setup.py (Lean toolchain, controls, oracle), run_e18.py (generation and
checking), colab_jobs.py (plans e18-1 and e18-pilot) and analyze_e18.py. Nothing here touches the E4/E11/E16 tables.

Result files (the Colab pull picks up `e18_*_colab_*.jsonl`):
    e18_<model><suffix>_minif2f_<split>.jsonl        run_e18.py, key (id, sample, round)
    e18_oracle<suffix>_minif2f_<split>.jsonl         e18_setup.py: every statement elaborates with `sorry`
    e18_setup<suffix>_runs.jsonl                     e18_setup.py: measured setup times, versions, controls
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from essaim import lean

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = HERE / "data"
BENCH = "minif2f"
SPLITS = ("dev", "test")

# ---------- data: miniF2F in Lean 4, as released with DeepSeek-Prover-V1.5 (Lean 4.9 era, both splits) ----------
DATA_URL = ("https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/"
            "2c4ba9119eef74d0d611f494261b2c5bae98c69a/datasets/minif2f.jsonl")
DATA_SHA256 = "d6654c88476ff14db19d57dea25bd955a240525fb52171e460f89d98f83d7dd5"
DATA_LICENSE = "MIT (LICENSE-CODE of deepseek-ai/DeepSeek-Prover-V1.5; miniF2F itself: MIT, openai/miniF2F)"
SOURCE_SPLIT = {"dev": "valid", "test": "test"}
DEV_N, DEV_SEED = 122, 0  # dev: half of miniF2F-valid, a fixed seeded draw (cost); test: all 244 of miniF2F-test

# ---------- Lean: the toolchain and Mathlib of the provers' own evaluations (Lean 4.9, DeepSeek-Prover-V1.5) ----------
TOOLCHAIN = "leanprover/lean4:v4.9.0"
MATHLIB_REPO = "https://github.com/leanprover-community/mathlib4"
MATHLIB_COMMIT = "f0957a7575317490107578ebaee9efaf8e62a4ab"  # tag v4.9.0, "chore: bump toolchain to v4.9.0"
ELAN_VERSION = "v4.2.3"
ELAN_URL = f"https://github.com/leanprover/elan/releases/download/{ELAN_VERSION}/elan-x86_64-unknown-linux-gnu.tar.gz"
ELAN_SHA256 = "df0b2b3a439961ffcbb3985214365ffe40f49bc871df04dff268c7d8e21ca8b2"
CHECK_TIMEOUT_S = 150.0  # wall clock per Lean file (the dataset header sets maxHeartbeats 0: no other bound)
CHECK_MEM_MIB = 6144     # anonymous resident memory per Lean process
SCREEN_VERSION = "screen-v1"

# ---------- protocol (pre-registered, docs/14_e18_lean.md) ----------
PROMPT_VERSION = "e18-v1"
MAX_TOKENS = 8192        # the quick-start limit of the provers' cards (their papers use up to 30k-40k)
PROMPT_BUDGET = {False: 1536, True: 6144}  # context per slot beyond MAX_TOKENS, without / with revision rounds
SAMPLES, CHAINS, REVISIONS = 3, 1, 1       # test, small provers (defaults of run_e18.py)
FEEDBACK_MAX_ERRORS, FEEDBACK_MAX_CHARS = 8, 6000
GIB = 2**30


@dataclass(frozen=True)
class Prover:
    model: str          # Hugging Face id (tokenizer/config revision pinned by essaim.common.resolve_revision)
    style: str          # prompt style: goedel (Goedel-Prover-V2 / Pythagoras) or kimina
    repo: str           # GGUF repository (third party: the authors publish no GGUF)
    file: str
    size: int           # bytes (Hugging Face API, 2026-10-13)
    sha256: str         # LFS hash: prepare_e18 refuses any other file
    revision: str       # commit of the GGUF repository when the table was written
    kv_bytes: int       # KV cache bytes per token in f16: layers x KV heads x head dim x 2 (K, V) x 2 bytes
    slots: int          # parallel requests of its llama-server
    params_b: float
    license: str
    temperature: float = 0.6  # generation_config.json of each model (Qwen3 defaults)
    top_p: float = 0.95
    top_k: int = 20
    thinking: bool | None = None  # None: the chat template's default (thinking on for these models)

    def local(self, models: Path) -> Path:
        """Where the GGUF file is kept: one directory per GGUF repository (some files have generic names)."""
        return models / "e18" / self.repo.replace("/", "__") / self.file

    def reserve_gib(self, revisions: bool) -> int:
        """GPU memory of its job: weights + f16 KV cache of every slot + 2 GiB (compute buffers, CUDA context)."""
        ctx = MAX_TOKENS + PROMPT_BUDGET[revisions]
        return math.ceil(self.size / GIB + self.slots * ctx * self.kv_bytes / GIB + 2)


_KV_QWEN3_8B = 36 * 8 * 128 * 2 * 2   # Qwen3-8B and Qwen3-4B: 36 layers, 8 KV heads of 128
_KV_QWEN3_32B = 64 * 8 * 128 * 2 * 2  # Qwen3-32B: 64 layers, 8 KV heads of 128
PROVERS = {p.model: p for p in (
    Prover("Goedel-LM/Goedel-Prover-V2-8B", "goedel", "mradermacher/Goedel-Prover-V2-8B-GGUF",
           "Goedel-Prover-V2-8B.Q8_0.gguf", 8709518592,
           "753fcff9da9865081f218d41a880f2e7181f0139bfbb0d0005daadcaf9598e0c",
           "43fab68ec71738a75b0a84738e18b222d2d4e249", _KV_QWEN3_8B, 12, 8.2, "apache-2.0"),
    Prover("AI-MO/Kimina-Prover-Distill-8B", "kimina", "mradermacher/Kimina-Prover-Distill-8B-GGUF",
           "Kimina-Prover-Distill-8B.Q8_0.gguf", 8709519008,
           "9c7a4346ae36152d1f7096b1158ba12e8d085e5176c97d5189d930ac6e17e9d8",
           "863f4acfbfd900b03addd4d052297c0ff0549675", _KV_QWEN3_8B, 12, 8.2, "apache-2.0"),
    Prover("Pythagoras-LM/Pythagoras-Prover-4B", "goedel", "tinyopsec/Pythagoras-Prover-4B-GGUF",
           "model_q8_0.gguf", 4693671552, "ed128a9adae81f32238832184158bffe562f8ebb6a643129e13ffa5447bc04df",
           "bd79dc7180b9387b1dc833ac4529f279b994fdbf", _KV_QWEN3_8B, 12, 4.4, "apache-2.0"),
    # reference: the strongest open prover that fits one A100 40GB (Q4_K_M: Q8_0 is 34.8 GB before any KV cache)
    Prover("Pythagoras-LM/Pythagoras-Prover-32B", "goedel", "mradermacher/Pythagoras-Prover-32B-GGUF",
           "Pythagoras-Prover-32B.Q4_K_M.gguf", 19762150400,
           "c713e51a2ce6609f3cb9b826a4fafade9f58c9ab181c2f0c5df3a69d8e1016d5",
           "1246c9cfa03750d44ada23ad2ca50a92230dd026", _KV_QWEN3_32B, 6, 32.8, "apache-2.0"),
)}
SWARM = ["Goedel-LM/Goedel-Prover-V2-8B", "AI-MO/Kimina-Prover-Distill-8B", "Pythagoras-LM/Pythagoras-Prover-4B"]
REFERENCE = "Pythagoras-LM/Pythagoras-Prover-32B"


# ---------- data ----------

def _download() -> bytes:
    DATA.mkdir(exist_ok=True)
    cache = DATA / "e18_minif2f_deepseek.jsonl"
    if cache.exists():
        raw = cache.read_bytes()
    else:
        with urllib.request.urlopen(DATA_URL, timeout=120) as r:
            raw = r.read()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise SystemExit(f"miniF2F : empreinte inattendue ({DATA_URL})")
    if not cache.exists():
        tmp = cache.with_name(cache.name + f".tmp{os.getpid()}")
        tmp.write_bytes(raw)
        os.replace(tmp, cache)
    return raw


def all_problems() -> list[dict]:
    rows = [json.loads(l) for l in _download().decode("utf-8").splitlines() if l.strip()]
    return [{"id": r["name"], "source_split": r["split"], "header": r["header"],
             "formal_statement": r["formal_statement"], "informal": r.get("informal_prefix", "")} for r in rows]


def problems(split: str) -> list[dict]:
    """dev: DEV_N problems of miniF2F-valid (seeded draw from the name order); test: all of miniF2F-test, by name."""
    if split not in SPLITS:
        raise ValueError(split)
    rows = sorted((p for p in all_problems() if p["source_split"] == SOURCE_SPLIT[split]), key=lambda p: p["id"])
    if len(rows) != 244 or len({p["id"] for p in rows}) != 244:
        raise SystemExit(f"miniF2F {SOURCE_SPLIT[split]} : {len(rows)} problèmes, 244 attendus")
    if split == "dev":
        random.Random(DEV_SEED).shuffle(rows)
        rows = rows[:DEV_N]
    return rows


def dataset_identity() -> dict:
    return {"dataset": DATA_URL, "sha256": DATA_SHA256, "license": DATA_LICENSE, "dev": f"valid, {DEV_N} (seed {DEV_SEED})",
            "test": "test, 244"}


def result_path(model: str, suffix: str, split: str, results: Path = RESULTS) -> Path:
    return results / f"e18_{model.replace('/', '__')}{suffix}_{BENCH}_{split}.jsonl"


def wall_path(model: str, suffix: str, split: str, results: Path = RESULTS) -> Path:
    return results / f"e18_{model.replace('/', '__')}{suffix}_wall_{split}.jsonl"


def oracle_path(suffix: str, split: str, results: Path = RESULTS) -> Path:
    return result_path("oracle", suffix, split, results)


def usable(split: str, suffix: str = "_colab", n: int | None = None, results: Path = RESULTS) -> list[dict]:
    """The split's problems whose statement elaborated with `sorry` in the oracle run (the others are excluded for
    every system); the first n of them for a smoke test."""
    from essaim.results import read_manifest, read_rows
    path = oracle_path(suffix, split, results)
    man = read_manifest(path)
    items = problems(split)
    if man is None:
        raise SystemExit(f"{path.name} absent : lancer e18_setup.py d'abord")
    ok = {r["id"]: r["ok"] for r in read_rows(path, key=("id",))}
    if set(ok) != {it["id"] for it in items}:
        raise SystemExit(f"{path.name} : oracle incomplet ({len(ok)} sur {len(items)})")
    items = [it for it in items if ok[it["id"]]]
    return items[:n] if n else items


# ---------- prompts ----------

def statement_with_sorry(it: dict) -> str:
    return it["header"] + lean.statement_prefix(it["formal_statement"]) + " by\n  sorry"


def informal(it: dict) -> str:
    t = it["informal"].strip()
    if t.startswith("/--"):
        t = t[3:]
    if t.endswith("-/"):
        t = t[:-2]
    return t.strip()


def first_prompt(spec: Prover, it: dict) -> list[dict]:
    if spec.style == "goedel":  # Goedel-Prover-V2 src/utils.py (DeepSeekCoTHandler), also Pythagoras' card
        return [{"role": "user", "content": (
            f"Complete the following Lean 4 code:\n\n```lean4\n{statement_with_sorry(it)}```\n\n"
            "Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan "
            "outlining the main proof steps and strategies.\nThe plan should highlight key ideas, intermediate lemmas, "
            "and proof structures that will guide the construction of the final formal proof.")}]
    if spec.style == "kimina":  # AI-MO/Kimina-Prover-Distill-8B model card
        return [{"role": "system", "content": "You are an expert in mathematics and Lean 4."},
                {"role": "user", "content": (
                    "Think about and solve the following problem step by step in Lean 4."
                    f"\n# Problem:{informal(it)}"
                    f"\n# Formal statement:\n```lean4\n{statement_with_sorry(it)}\n```\n")}]
    raise ValueError(spec.style)


def messages(spec: Prover, it: dict, prev: dict | None) -> list[dict]:
    """The first prompt, or (revision) the first prompt, the code Lean checked as the assistant's answer, and Lean's
    feedback in the format of Goedel-Prover-V2's self-correction (src/inference.py, generate_correction_prompt)."""
    msgs = first_prompt(spec, it)
    if prev is None:
        return msgs
    shown = prev.get("code") or lean.last_code_block(prev.get("text") or "") or ""
    return msgs + [{"role": "assistant", "content": f"```lean4\n{shown}\n```"},
                   {"role": "user", "content": (
                       f"The proof (Round {prev['round']}) is not correct. Following is the compilation error message, "
                       "where we use <error></error> to signal the position of the error.\n\n"
                       f"{prev['feedback']}\n\nBefore producing the Lean 4 code to formally prove the given theorem, "
                       "provide a detailed analysis of the error message.")}]


def candidate(text: str, reasoning_block: str | None) -> str:
    """What the checked file is built from: the final answer when it has a Lean block, else the last Lean block of
    the reasoning (an answer cut by the token limit), else the answer (no code)."""
    if lean.last_code_block(text) is None and reasoning_block:
        return f"```lean4\n{reasoning_block}\n```"
    return text


def build_feedback(built: dict) -> str:
    return {"no_code": "No Lean 4 code block was found in the answer.",
            "no_theorem": "The Lean 4 code does not contain the theorem with its original name and statement.",
            "forbidden": f"The code was refused before compilation ({built.get('reason')}): only theorem and lemma "
                         "declarations and tactic proofs are allowed."}.get(built["status"], built["status"])


def status_feedback(v: dict) -> str:
    return {"timeout": "Lean did not finish within the time limit (the proof is too slow or loops).",
            "memory": "Lean exceeded the memory limit.",
            "sorry": "The proof uses sorry.",
            "bad_axioms": f"The proof depends on non-standard axioms: {v.get('axioms')}."}.get(
        v["status"], f"The proof was not accepted ({v['status']}).")


def lean_record(run: dict, axioms_line: int, max_messages: int = 60, max_chars: int = 2000) -> dict:
    """What is stored of a Lean run: enough to recompute the verdict offline (lean.verdict accepts `messages`).
    Every error and the message on the axioms line are kept; other messages up to the bound."""
    msgs = lean.parse_messages(run.get("stdout", ""))
    keep = [m for m in msgs if m["severity"] == "error" or m["line"] == axioms_line
            or lean.SORRY_WARNING in m["data"]]
    rest = [m for m in msgs if m not in keep][:max(0, max_messages - len(keep))]
    kept = [{**m, "data": m["data"][:max_chars]} for m in msgs if m in keep or m in rest]
    return {"returncode": run.get("returncode"), "timeout": run.get("timeout"), "memory": run.get("memory"),
            "overflow": run.get("overflow"), "ms": run.get("ms"), "peak_mib": run.get("peak_mib"),
            "messages": kept, "n_messages": len(msgs)}


# ---------- environment ----------

def check_lean_env(env: dict):
    if env.get("toolchain") != TOOLCHAIN or env.get("mathlib_commit") != MATHLIB_COMMIT:
        raise SystemExit(f"environnement Lean inattendu : {env.get('toolchain')} {env.get('mathlib_commit')}")
    if not env.get("controls_ok"):
        raise SystemExit("les contrôles du vérificateur n'ont pas réussi : relancer e18_setup.py")


def lean_identity(env: dict) -> dict:
    return {k: env.get(k) for k in ("toolchain", "lean_version", "mathlib_commit", "elan_version", "mathlib_source")}


def checker_identity(checker) -> dict:
    return {"timeout_s": checker.timeout_s, "mem_mib": checker.mem_mib, "isolation": checker.isolation,
            "allowed_axioms": sorted(lean.ALLOWED_AXIOMS), "screen": SCREEN_VERSION}


# ---------- Colab plans (colab_jobs.py) ----------
# A job is (kind, model, split). e18-lean: Lean, controls and oracle on the CPU (stage 0, stops the plan if the
# checker fails its controls); e18-prove: run_e18.py for one model and split (stage 1).
KINDS = ("e18-lean", "e18-prove", "e18-pilot")
STAGES = {"e18-lean": 0, "e18-prove": 1, "e18-pilot": 1}
PILOT_N = 20


def protocol(kind: str, model: str, split: str) -> dict:
    """run_e18.py arguments of a job. dev: one attempt per problem (choice of the best small prover, no revision);
    test: SAMPLES attempts, chain 0 revised REVISIONS times; reference: SAMPLES attempts, no revision. The pilot (dev,
    PILOT_N problems) measures lengths and speeds with one attempt and the revisions."""
    ref = model == REFERENCE
    if kind == "e18-pilot":
        return {"samples": 1, "chains": 0 if ref else 1, "revisions": 0 if ref else REVISIONS, "n": PILOT_N,
                "suffix": "_pilot_colab"}
    if split == "dev":
        return {"samples": 1, "chains": 0, "revisions": 0, "n": None, "suffix": "_colab"}
    return {"samples": SAMPLES, "chains": 0 if ref else CHAINS, "revisions": 0 if ref else REVISIONS, "n": None,
            "suffix": "_colab"}


def reserve_gib(kind: str, model: str, split: str) -> int:
    if kind == "e18-lean":
        return 0
    p = protocol(kind, model, split)
    return PROVERS[model].reserve_gib(p["chains"] > 0 and p["revisions"] > 0)


def plan(pilot: bool = False) -> list[tuple[str, str, str]]:
    jobs = [("e18-lean", BENCH, "all")]
    if pilot:
        return jobs + [("e18-pilot", m, "dev") for m in SWARM + [REFERENCE]]
    return jobs + [("e18-prove", m, s) for s in SPLITS for m in SWARM] + [("e18-prove", REFERENCE, "test")]
