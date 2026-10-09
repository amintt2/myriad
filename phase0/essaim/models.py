"""Model sizes, one convention for the whole research code, the paper and its figures.

TOTAL parameters (billions), embeddings included: what a consumer device must hold. Gemma 4 "E2B" and
"E4B" are named after their EFFECTIVE size (the per-layer embeddings are excluded from the name: about 2B
and 4B effective, but 5B and 8B in total); they are recorded with both, and every size shown in the paper
or used to rank references (analyze_e4.py, analyze_code.py, paper/make_numbers.py, paper/make_figures.py)
is the total. Values are those of the model cards.
"""
from __future__ import annotations

# model id -> (total parameters, effective parameters if the name refers to them), in billions
SIZES: dict[str, tuple[float, float | None]] = {
    # E1-E3 (phase 0)
    "Qwen/Qwen3-1.7B": (1.7, None),
    "HuggingFaceTB/SmolLM3-3B": (3.1, None),
    "ibm-granite/granite-3.3-2b-instruct": (2.5, None),
    "google/gemma-4-E2B-it": (5.0, 2.0),
    "Qwen/Qwen3-4B": (4.0, None),
    # E4, E10, E11
    "Qwen/Qwen3.5-2B": (2.0, None),
    "Qwen/Qwen3.5-4B": (4.0, None),
    "google/gemma-4-E4B-it": (8.0, 4.0),
    "ibm-granite/granite-4.2-3b": (3.0, None),
    "mistralai/Ministral-3-3B-Instruct-2512": (3.4, None),
    "microsoft/Phi-4-mini-instruct": (3.8, None),
    "allenai/OLMo-2-0425-1B-Instruct": (1.5, None),
    "Qwen/Qwen3.5-9B": (9.0, None),
    "google/gemma-4-12B-it": (12.0, None),
    "mistralai/Ministral-3-14B-Instruct-2512": (14.0, None),
    "Qwen/Qwen3.8-27B": (27.0, None),
}


def total_b(model: str) -> float:
    """Total parameters in billions (KeyError for an unknown model: sizes are never guessed)."""
    return SIZES[model][0]


def effective_b(model: str) -> float | None:
    return SIZES[model][1]


def fmt_b(x: float) -> str:
    return f"{x:g}"
