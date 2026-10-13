"""E16: a bigger model quantized to the same memory, instead of a small model in Q8 (docs/13_e16_quantification.md).

Pure data and helpers, shared by colab_jobs.py (plan e16-1), speed_quant.py and analyze_e16.py. Nothing here
touches the E4/E11 tables (colab_jobs.SOLO): the quantized files live in QUANT, and each (model, quantization) pair
writes its own result files, named with a suffix that still contains "_colab_" so that the Colab pull picks them up:

    solo_Qwen__Qwen3.5-9B_iq3xxs_colab_gsm8k_dev.jsonl      (run_solo.py --suffix _iq3xxs_colab)
    code_Qwen__Qwen3.5-9B_iq3xxs_colab_mbppplus_test.jsonl  (run_code.py)
    codeexec_e16_iq3xxs_colab_mbppplus_test.jsonl           (exec_code.py --tag e16, one file per quantization)

The protocol is the one of E4 and E11 (same prompts, 16 slots of 3072 tokens, greedy), so a quantized run and the
Q8_0 run of the same model differ only by the weights file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

GIB = 2**30
BUDGET_GIB = 6.5  # weights file of one peer on an 8 GB consumer GPU (KV cache of 4-8k tokens and buffers on top)
TIERS_GIB = (5.5, BUDGET_GIB)
EXEC_TAG = "e16"
SPEED_STEM = "e16speed"  # results/solo_e16speed_colab_bench_all.jsonl, written by speed_quant.py


@dataclass(frozen=True)
class Quant:
    model: str     # Hugging Face id of the original model (tokenizer, revision pinned by essaim.common)
    tag: str       # llama.cpp quantization type, e.g. IQ3_XXS
    repo: str      # GGUF repository
    file: str
    size: int      # bytes (Hugging Face metadata)
    sha256: str    # published LFS hash: fastdl.py refuses a file that does not match, the manifests record it
    revision: str  # commit of the GGUF repository when the table was written (documentation; the hash decides)

    @property
    def gib(self) -> float:
        return self.size / GIB

    @property
    def suffix(self) -> str:
        return f"_{slug(self.tag)}_colab"

    @property
    def exec_tag(self) -> str:
        """Tag of this file's sandbox results (exec_code.py --tag; no underscore). One execution file set per
        (model, quantization): exec_code.py records the list of models in its manifest and refuses to resume when it
        changes, so a later plan (e16-2) must never add a model to files an earlier plan wrote."""
        return EXEC_TAG + "".join(c for c in self.model.split("/")[-1].lower() if c.isalnum())

    def fits(self, budget_gib: float | None = None) -> bool:
        return self.gib <= (BUDGET_GIB if budget_gib is None else budget_gib)


def slug(tag: str) -> str:
    """Quantization tag as used in file names: IQ3_XXS -> iq3xxs (no underscore: it separates name parts)."""
    return tag.replace("_", "").lower()


_QWEN, _GEMMA, _MINI = "Qwen/Qwen3.5-9B", "google/gemma-4-12B-it", "mistralai/Ministral-3-14B-Instruct-2512"
_BQ, _BG, _BM = ("bartowski/Qwen_Qwen3.5-9B-GGUF", "bartowski/gemma-4-12B-it-GGUF",
                 "bartowski/mistralai_Ministral-3-14B-Instruct-2512-GGUF")
_RQ, _RG, _RM = "182be2fd6c7b", "2ae7d41be21c", "9289ba93995b"
_Q = lambda model, tag, repo, rev, name, size, sha: Quant(model, tag, repo, name, size, sha, rev)

# Files sized and hashed with the Hugging Face API on 2026-10-13. imatrix quantizations by bartowski (same recipe for
# the three families); the QAT file is Google's own (quantization-aware training, Q4_0).
QUANT: list[Quant] = [
    _Q(_QWEN, "IQ3_XXS", _BQ, _RQ, "Qwen_Qwen3.5-9B-IQ3_XXS.gguf", 4534579232, "478d619ab6f27ca84c8e8650667329ba54bb5cecda5d732a19dc91c6fd6b6948"),
    _Q(_QWEN, "IQ3_M", _BQ, _RQ, "Qwen_Qwen3.5-9B-IQ3_M.gguf", 4981092384, "cbeadcf49deaf24c756516c4b98e88b88ec8719108553038623d52907f84522d"),
    _Q(_QWEN, "Q3_K_M", _BQ, _RQ, "Qwen_Qwen3.5-9B-Q3_K_M.gguf", 5178748960, "24fccd8416fcdca792f939de4a2240eb57795b0f7a9ba92e62398eab3ec5c11e"),
    _Q(_QWEN, "IQ4_XS", _BQ, _RQ, "Qwen_Qwen3.5-9B-IQ4_XS.gguf", 5501202464, "7d977cc96c2e08616016d967f232083e354691a8a16f345b26f7d782ee5c9601"),
    _Q(_GEMMA, "IQ3_XXS", _BG, _RG, "gemma-4-12B-it-IQ3_XXS.gguf", 5145090528, "3a85fa01d72d3f5c30c4ae723f6a42e8ad0f6d4df366776b9628e1135e06f4c3"),
    _Q(_GEMMA, "IQ3_M", _BG, _RG, "gemma-4-12B-it-IQ3_M.gguf", 5969922528, "def80efba2f976bb0c9709f06845b237ba3092bf4d68d5b47fd84b4ef23f4439"),
    _Q(_GEMMA, "Q3_K_M", _BG, _RG, "gemma-4-12B-it-Q3_K_M.gguf", 6301391328, "b3a5c0834941b18251317f7dd1096e5d6e01b12e1b06b49670fd26467de8f6b4"),
    _Q(_GEMMA, "IQ4_XS", _BG, _RG, "gemma-4-12B-it-IQ4_XS.gguf", 6780746208, "60e817ec84aabab9d20ac1eeec5d0e5fc5859993eb2f234e0f0d95fef2aa87f7"),
    _Q(_GEMMA, "QAT_Q4_0", "google/gemma-4-12B-it-qat-q4_0-gguf", "29d097773436", "gemma-4-12b-it-qat-q4_0.gguf", 6975879296,
       "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"),
    _Q(_MINI, "IQ3_XXS", _BM, _RM, "mistralai_Ministral-3-14B-Instruct-2512-IQ3_XXS.gguf", 5427082656, "c048183d069e475819612e8ae24d159aaf07e2ac620fd31ec325e39bc410a4c1"),
    _Q(_MINI, "IQ3_M", _BM, _RM, "mistralai_Ministral-3-14B-Instruct-2512-IQ3_M.gguf", 6269875616, "0784b672b6a633208df04af863e857b421b6de3209bf614a6159ac994f5d79d6"),
    _Q(_MINI, "Q3_K_M", _BM, _RM, "mistralai_Ministral-3-14B-Instruct-2512-Q3_K_M.gguf", 6682097056, "b440f656c6a308f4c2cde69481efbebc91e00af083c1fb8e4714920f59990d25"),
]
# Left out because over the budget (7.4 GB = 6.9 GiB): Ministral-3-14B IQ4_XS (bartowski, sha256 2895179103887c0a...).
# Listed in the doc, never run.

# GPU memory to reserve besides the weights (KV cache of 16 slots x 3072 tokens, compute buffers, margin), in GiB:
# the reservation of the model's Q8_0 entry in colab_jobs.SOLO minus its Q8_0 file (tests/test_e16.py checks it).
OVERHEAD_GIB = {_QWEN: 9.2, _GEMMA: 11.2, _MINI: 11.7}
# Code generation (E11) is run for the lowest and the highest quantization of each model within the budget: the bounds,
# chosen before any result. e16-2 adds the intermediate ones.
CODE_BOUNDS = {(_QWEN, "IQ3_XXS"), (_QWEN, "IQ4_XS"), (_GEMMA, "IQ3_XXS"), (_GEMMA, "QAT_Q4_0"),
               (_MINI, "IQ3_XXS"), (_MINI, "Q3_K_M")}
DEFAULT_GPU_GIB = 40  # A100 40 GB; larger GPUs only run more jobs at once
# Q8_0 files timed next to the quantized ones: the E4 swarm (7 families), its two smaller siblings and the three
# references of the same size as the quantized models (not the 27B: a 28 GB download for a model out of the question).
SPEED_Q8 = ["Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it", "ibm-granite/granite-4.2-3b", "HuggingFaceTB/SmolLM3-3B",
            "mistralai/Ministral-3-3B-Instruct-2512", "microsoft/Phi-4-mini-instruct", "allenai/OLMo-2-0425-1B-Instruct",
            "Qwen/Qwen3.5-2B", "google/gemma-4-E2B-it", _QWEN, _GEMMA, _MINI]


# Calibration of the consumer-speed estimate: speeds_consumer.json (llama-bench tg128, Q8_0, RX 6650 XT 8 GB, 280 GB/s
# nominal) with the byte size of the Q8_0 file (Hugging Face API, 2026-10-13). Dense models only: Gemma E2B reads far
# fewer bytes per token than its file holds (per-layer embeddings) and Qwen3.5 is a hybrid linear-attention
# architecture whose speed is not set by the weights alone (Qwen3.5-2B reaches 72 GB/s against 120-220 for the others).
CONSUMER_NOMINAL_GBS = 280.0
CALIBRATION_DENSE: list[tuple[str, int]] = [
    ("Qwen/Qwen3-1.7B", 1834426016), ("Qwen/Qwen3-4B", 4280404704),
    ("ibm-granite/granite-4.2-3b", 3892651552), ("HuggingFaceTB/SmolLM3-3B", 3275574624)]


def get(model: str, tag: str) -> Quant:
    for q in QUANT:
        if q.model == model and q.tag == tag:
            return q
    raise KeyError(f"{model}@{tag}: pas de fichier quantifié dans QUANT")


def label(model: str, tag: str) -> str:
    return f"{model}@{tag}"


def split_label(lab: str) -> tuple[str, str]:
    model, _, tag = lab.rpartition("@")
    return model, tag


def reserve_gib(q: Quant) -> int:
    """GPU memory reserved for one job (MemoryBudget): the file plus the model's fixed overhead, rounded up."""
    return math.ceil(q.gib + OVERHEAD_GIB[q.model])


def models() -> list[str]:
    return list(dict.fromkeys(q.model for q in QUANT))


def groups() -> dict[str, list[Quant]]:
    """Quantizations by tag, in table order: the models quantized the same way form one 'big swarm'."""
    out: dict[str, list[Quant]] = {}
    for q in QUANT:
        out.setdefault(q.tag, []).append(q)
    return out


def plan(solo: bool = True, code: str = "bounds", speed: bool = True) -> list[tuple[str, str, str]]:
    """Jobs of a plan: (kind, model, quantization tag). solo: E4 benchmarks on the 12 files; code: 'bounds' (the lowest
    and highest quantization of each model, CODE_BOUNDS), 'rest' (the other ones) or 'none'; speed: llama-bench of every
    file, alone on the GPU. Stages (colab_jobs.STAGE): generation, then sandbox execution, then the speed measurement."""
    jobs = [("solo-q", q.model, q.tag) for q in QUANT] if solo else []
    if code != "none":
        rest = code == "rest"
        jobs += [("code-q", q.model, q.tag) for q in QUANT if ((q.model, q.tag) in CODE_BOUNDS) != rest]
        jobs += [("code-exec", EXEC_TAG, "all")]
    if speed:
        jobs += [("solo-speed", EXEC_TAG, "all")]
    return jobs


def speed_need_gib(largest_q8_gib: float = 13.4) -> int:
    """The speed job loads one file at a time, alone on the GPU (last stage)."""
    return math.ceil(largest_q8_gib + 3)


def a100_hours(measured_hours: float, n_parallel: int = 2, speedup: float = 1.0) -> float:
    """Wall-clock hours of a set of jobs whose individual durations (measured under contention) sum to
    `measured_hours`, run `n_parallel` at a time."""
    return measured_hours / max(n_parallel, 1) / speedup


# ---- consumer speed estimate (no GPU involved) ----

def implied_bandwidth_gbs(tokens_per_s: float, size_bytes: int) -> float:
    """Decoding reads every weight once per token: bytes read per second, in GB/s."""
    return tokens_per_s * size_bytes / 1e9


def estimate_consumer_tps(size_bytes: int, calibration: list[tuple[float, int]]) -> tuple[float, float, float]:
    """(low, mid, high) tokens/s on the consumer GPU of `calibration` = [(measured tok/s, file bytes)], by the
    memory-bandwidth model tps = B / bytes with B the implied bandwidth of the calibration files (min, median, max).
    Valid for batch-1 decoding with fully offloaded weights; it ignores the extra cost of dequantizing i-quants."""
    bws = sorted(implied_bandwidth_gbs(t, s) for t, s in calibration)
    if not bws:
        raise ValueError("calibration vide")
    mid = bws[len(bws) // 2] if len(bws) % 2 else (bws[len(bws) // 2 - 1] + bws[len(bws) // 2]) / 2
    return tuple(b * 1e9 / size_bytes for b in (bws[0], mid, bws[-1]))  # type: ignore[return-value]
