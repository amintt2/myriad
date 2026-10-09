"""Simulated peers for the scalability benchmark (E7): an answer model and a compute-time model.

Answer model. Question q has a gold answer. Node i answers it correctly with probability p_i (its
model's accuracy), independently of the other nodes. When wrong, it gives the question's "popular"
wrong answer with probability a, otherwise a wrong answer of its own (drawn from a space so large that
two of them never coincide in practice). Two wrong peers therefore give the same wrong answer with
probability c = a^2: a = sqrt(c). Our measured GSM8K value is c ~ 0.09 (a = 0.3).

Compute-time model. The time a node takes for a question is lognormal: median m and log-scale sigma,
multiplied by a time scale (1.0 = real time; smaller values speed the run up and are reported).
Phase-0 measurements (phase0/results/gen_essaim4_colab_test.jsonl, solo mode, 200 GSM8K test
questions per model, one GPU): median 4.06 s and sigma 0.36 over the 4 models; per model medians
SmolLM3-3B 3.54 s, Qwen3-1.7B 3.91 s, granite-3.3-2b 3.91 s, gemma-4-E2B 5.02 s.

Every draw is a pure function of (seed, node, question): a node asked the same question twice gives
the same answer after the same time (greedy decoding), and runs are reproducible.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field

MEASURED_COLLISION = 0.09  # GSM8K: probability that two wrong peers give the same wrong answer
MEASURED_MEDIAN_S = 4.06
MEASURED_SIGMA = 0.36


@dataclass(frozen=True)
class Family:
    family: str
    model: str  # model id as the nodes announce it (known ids get their priors from myriad/priors.json)
    accuracy: float  # true probability of a correct answer (the oracle)
    median_s: float = MEASURED_MEDIAN_S
    params_b: float = 2.0
    measured: bool = True  # False: a synthetic family added to reach k > 4


# The 4 families measured in phase 0 (GSM8K test accuracy and per-model median time), then synthetic
# families, deliberately WEAKER than all measured ones (accuracy 0.45), used only when k > 4.
MEASURED_FAMILIES = (
    Family("smollm", "HuggingFaceTB/SmolLM3-3B-GGUF", 0.865, 3.54, 3.1),
    Family("gemma", "ggml-org/gemma-4-E2B-it-GGUF", 0.74, 5.02, 2.0),
    Family("qwen", "Qwen/Qwen3-1.7B-GGUF", 0.535, 3.91, 1.7),
    Family("granite", "ibm-granite/granite-3.3-2b-instruct-GGUF", 0.495, 3.91, 2.5),
)
SYNTHETIC_ACCURACY = 0.45


def families(n: int) -> list[Family]:
    """The first n families: the 4 measured ones, then synthetic ones (synth5, synth6, ...)."""
    out = list(MEASURED_FAMILIES[:n])
    for j in range(len(out), n):
        name = f"synth{j + 1}"
        out.append(Family(name, f"bench/{name}-2b", SYNTHETIC_ACCURACY, MEASURED_MEDIAN_S, 2.0, measured=False))
    return out


def prior_entries(fams: list[Family]) -> dict[str, dict]:
    """Entries for myriad/priors.json 'models' so that synthetic families get their true accuracy as
    prior, like the measured ones (patched in memory by the benchmark, never written to disk)."""
    from myriad.priors import model_key

    return {model_key(f.model): {"family": f.family, "params_b": f.params_b, "accuracy": f.accuracy}
            for f in fams if not f.measured}


def _u(*parts) -> random.Random:
    """A generator seeded by a stable hash of its key (independent of PYTHONHASHSEED)."""
    h = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=16).digest()
    return random.Random(int.from_bytes(h, "big"))


@dataclass(frozen=True)
class AnswerModel:
    collision: float = MEASURED_COLLISION
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.collision <= 1.0:
            raise ValueError("collision must be in [0, 1]")

    @property
    def popular_share(self) -> float:
        return math.sqrt(self.collision)

    def gold(self, qid: int) -> str:
        return str(10 + _u("gold", self.seed, qid).randrange(10_000))

    def popular_wrong(self, qid: int) -> str:
        g = int(self.gold(qid))
        return str(g + 1 + _u("popular", self.seed, qid).randrange(50))

    def answer(self, node: str, accuracy: float, qid: int) -> str:
        r = _u("answer", self.seed, node, qid)
        if r.random() < accuracy:
            return self.gold(qid)
        if r.random() < self.popular_share:
            return self.popular_wrong(qid)
        return str(100_000 + r.randrange(10**12))  # never the gold (< 10 010) nor the popular wrong answer


@dataclass(frozen=True)
class ComputeModel:
    median_s: float = MEASURED_MEDIAN_S
    sigma: float = MEASURED_SIGMA
    scale: float = 1.0  # time scale: 1.0 = measured times; 0.05 = 20 times faster
    node_sigma: float = 0.0  # spread of node speeds (hardware heterogeneity); 0 = identical nodes
    per_family: bool = True  # use each family's measured median instead of median_s
    seed: int = 0

    def node_speed(self, node: str) -> float:
        if self.node_sigma <= 0:
            return 1.0
        return math.exp(self.node_sigma * _u("speed", self.seed, node).gauss(0.0, 1.0))

    def seconds(self, node: str, qid: int, median_s: float | None = None) -> float:
        m = self.median_s if median_s is None else median_s
        z = _u("time", self.seed, node, qid).gauss(0.0, 1.0)
        return self.scale * m * self.node_speed(node) * math.exp(self.sigma * z)


QID_RE = re.compile(r"\[q(\d+)\]")


def question(qid: int) -> str:
    """A math prompt carrying its question id; detected as 'math' by the gateway."""
    return f"[q{qid}] A farmer has {qid % 97 + 3} crates of 12 apples. How many apples does he have in total?"


def qid_of(messages: list[dict]) -> int:
    for m in reversed(messages):
        found = QID_RE.search(m.get("content", ""))
        if found:
            return int(found.group(1))
    raise ValueError("no question id in the prompt")


@dataclass
class SimPeer:
    """Reply and delay functions for one simulated node (plugged into myriad.engine.FakeEngine)."""
    name: str
    family: Family
    answers: AnswerModel
    compute: ComputeModel
    hung: bool = False  # failure injection: the engine stops answering
    seen: list = field(default_factory=list)

    def reply(self, messages: list[dict]) -> str:
        q = qid_of(messages)
        return f"Each crate holds 12 apples, so we multiply. The answer is {self.answers.answer(self.name, self.family.accuracy, q)}."

    def delay(self, messages: list[dict]) -> float:
        if self.hung:
            return 3600.0
        median = self.family.median_s if self.compute.per_family else None
        return self.compute.seconds(self.name, qid_of(messages), median)
