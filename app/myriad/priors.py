"""Reliability priors per model family (phase-0 measurements), family/size detection, credit factor."""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from importlib import resources

# Family keywords, checked in order against the lower-cased model name.
_FAMILY_KEYS = (
    ("smollm", "smollm"), ("qwen", "qwen"), ("granite", "granite"), ("gemma", "gemma"),
    ("llama", "llama"), ("mistral", "mistral"), ("ministral", "mistral"), ("phi", "phi"),
    ("olmo", "olmo"), ("deepseek", "deepseek"), ("falcon", "falcon"), ("exaone", "exaone"),
)
_SIZE_RE = re.compile(r"(?:^|[-_.])e?(\d+(?:\.\d+)?)b(?:$|[-_.])", re.I)
# Credits are counted in thousandths, so the ledger only holds integers.
MILLI = 1000


@lru_cache(maxsize=1)
def load() -> dict:
    with resources.files("myriad").joinpath("priors.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def model_key(model_id: str | None) -> str:
    """'unsloth/Qwen3-1.7B-GGUF' -> 'qwen3-1.7b' (owner, GGUF suffix and quantisation removed)."""
    name = (model_id or "").split(":")[0].rsplit("/", 1)[-1].lower()
    name = re.sub(r"[-_.]gguf$", "", name)
    name = re.sub(r"[-_.](q\d.*|iq\d.*|f16|bf16|f32)$", "", name)
    return name


def family_of(model_id: str | None) -> str:
    key = model_key(model_id)
    known = load()["models"].get(key)
    if known:
        return known["family"]
    for needle, fam in _FAMILY_KEYS:
        if needle in key:
            return fam
    owner = (model_id or "").split("/", 1)[0].lower()
    return owner or "inconnu"


def params_of(model_id: str | None) -> float | None:
    """Size in billions of parameters: from the priors file, else parsed from the name ('1.7B', 'E2B')."""
    key = model_key(model_id)
    known = load()["models"].get(key)
    if known and "params_b" in known:
        return float(known["params_b"])
    m = _SIZE_RE.search(key)
    return float(m.group(1)) if m else None


def prior_accuracy(model_id: str | None, family: str | None = None) -> float:
    pri = load()
    known = pri["models"].get(model_key(model_id))
    if known and "accuracy" in known:
        return float(known["accuracy"])
    fam = family or family_of(model_id)
    return float(pri["families"].get(fam, pri["default"]))


def prior_strength() -> float:
    return float(load().get("prior_strength", 50))


def beta_mean(prior: float, agree: int, disagree: int, strength: float | None = None) -> float:
    """Posterior mean of a Beta(prior*n0, (1-prior)*n0) prior after agree/disagree observations."""
    n0 = prior_strength() if strength is None else strength
    return (prior * n0 + agree) / (n0 + agree + disagree)


def collision_prob(task_hint: str, n_options: int | None = None) -> float:
    """Probability c that two wrong peers give the same wrong answer, per task type (priors.json);
    1 / (K - 1) for a multiple-choice question with K options."""
    if task_hint == "mc":
        return 1.0 / max(1, (n_options or 4) - 1)
    table = load().get("same_wrong_answer", {})
    return float(table.get(task_hint, 1.0))


def credit_factor(params_b: float | None) -> float:
    """Credits earned per generated token: proportional to the model size, 1.0 for a 2B model."""
    p = 1.0 if params_b is None or not math.isfinite(params_b) else params_b
    return max(0.25, min(p, 70.0) / 2.0)


def trusted_params(model_id: str | None, declared: float | None, cap_unknown: float = 8.0) -> float:
    """Size used for credits: the known size when the model is in the priors file, else the declared
    size capped (a node cannot earn more by claiming to run a huge model)."""
    known = params_of(model_id) if model_key(model_id) in load()["models"] else None
    if known is not None:
        return known
    if declared is None or not math.isfinite(declared) or declared <= 0:
        return 1.0
    return min(declared, cap_unknown)
