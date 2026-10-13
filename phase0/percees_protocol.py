"""Offline PERC job planning and pure CPU operators; no model or candidate execution.

Only the explicitly enabled injectable inference seam calls external code. The sandbox seam is a
future adapter contract, not an alternative executor. This module uses the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence


VERSION = "PERC-20261010-v1-gpu-protocol-1"
IDEAS = tuple(f"P{i:02}" for i in range(1, 13))
PERMISSIVE_LICENSES = {"Apache-2.0", "MIT", "BSD-3-Clause"}
PROMPTS = {
    "standard": "Solve the following task. Return the final answer or the requested function only.\n\nTask:\n",
    "contract_edges": (
        "Solve the following task. Check its contract, types, and edge cases privately. "
        "Return the final answer or the requested function only.\n\nTask:\n"
    ),
    "concise_invariants": (
        "Solve the following task using a concise solution. Check the essential invariants privately. "
        "Return the final answer or the requested function only.\n\nTask:\n"
    ),
}
TEST_PROMPT = (
    "Generate fallible test proposals from the task statement alone. You have no candidate solution, "
    "reference solution, or hidden tests. Return JSON only with cases containing JSON inputs, proposed "
    "expected outputs, and a rationale, plus optional declarative metamorphic relations. Do not return "
    "executable assertions or code. State uncertainties; do not invent unspecified behavior.\n\nTask:\n"
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def finite(value: float, *, positive: bool = False) -> None:
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError("Expected a finite positive value" if positive else "Expected a finite nonnegative value")


@dataclass(frozen=True)
class Problem:
    benchmark: str
    id: str
    split: str
    domain: str
    prompt: str
    source: str
    synthetic_external: bool = False

    @property
    def key(self) -> str:
        return f"{self.benchmark}/{self.id}"

    @property
    def fingerprint(self) -> str:
        # Exact normalized duplicate guard, not a semantic contamination detector.
        return digest(" ".join(self.prompt.split()).casefold())

    def validate(self) -> None:
        if type(self.synthetic_external) is not bool:
            raise ValueError("Synthetic provenance flag must be a boolean")
        if self.split not in {"dev", "test", "train"}:
            raise ValueError("Unknown split")
        if not all((self.benchmark, self.id, self.domain, self.prompt.strip(), self.source)):
            raise ValueError("Missing problem identity, provenance, domain, or prompt")
        if "/" in self.benchmark or "/" in self.id:
            raise ValueError("Ambiguous problem key")
        if self.split == "train" and not self.synthetic_external:
            raise ValueError("P11 training requires synthetic TRAIN external to every benchmark")
        if self.split != "train" and self.synthetic_external:
            raise ValueError("Synthetic training records cannot be evaluation records")


@dataclass(frozen=True)
class Model:
    id: str
    family: str
    revision: str
    license: str
    license_evidence: str
    tokenizer: str
    native_chat_template: str
    parameter_billions: float
    prefill_tokens_s: float
    decode_tokens_s: float
    hourly_cost: float
    license_verified: bool = False
    weights_format: str = "safetensors"
    trust_remote_code: bool = False

    def validate(self) -> None:
        if type(self.license_verified) is not bool or type(self.trust_remote_code) is not bool:
            raise ValueError("License review and remote code flags must be booleans")
        if not all((self.id, self.family, self.revision, self.license_evidence, self.tokenizer,
                    self.native_chat_template)):
            raise ValueError("Catalogue requires pinned assets, native template, and license provenance")
        if self.license not in PERMISSIVE_LICENSES or self.weights_format != "safetensors" or self.trust_remote_code:
            raise ValueError("Only declared permissive safetensors models without remote code")
        for value in (self.parameter_billions, self.prefill_tokens_s, self.decode_tokens_s):
            finite(value, positive=True)
        finite(self.hourly_cost)


@dataclass(frozen=True)
class Config:
    id: str
    idea: str
    candidate_count: int = 1
    reflection_tokens: int = 0
    width: int = 1
    share_tokens: int = 0
    prompt_variant: str = "standard"
    draft_tokens: int = 4
    pool: str = "linear"
    temperature: float = 1.0

    def validate(self) -> None:
        if not self.id or self.idea not in IDEAS or self.prompt_variant not in PROMPTS:
            raise ValueError("Invalid configuration")
        if any(type(n) is not int for n in (self.candidate_count, self.reflection_tokens, self.width,
                                           self.share_tokens, self.draft_tokens)):
            raise ValueError("Integral variant budgets required")
        if self.candidate_count not in {1, 3, 5} or self.reflection_tokens not in {0, 1000, 4000, 16000}:
            raise ValueError("Variant outside preregistration")
        if self.width not in {1, 3, 5} or self.share_tokens not in {0, 128} or self.draft_tokens not in {4, 8, 16}:
            raise ValueError("Variant outside preregistration")
        if self.pool not in {"linear", "product"}:
            raise ValueError("Unknown probability pool")
        finite(self.temperature, positive=True)


def default_configs() -> tuple[Config, ...]:
    configs = [Config(f"P06-n{n}", "P06", candidate_count=n) for n in (1, 3, 5)]
    configs += [Config(f"P07-r{r}-w{w}-s{s}", "P07", reflection_tokens=r, width=w, share_tokens=s)
                for r in (0, 1000, 4000, 16000) for w in (1, 3, 5) for s in (0, 128)]
    configs += [Config(f"P08-{p}", "P08", prompt_variant=p) for p in PROMPTS]
    configs += [Config(f"P09-k{k}", "P09", draft_tokens=k) for k in (4, 8, 16)]
    configs += [Config(f"P10-{op}-t{t}", "P10", pool=op, temperature=t)
                for op in ("linear", "product") for t in (0.5, 1.0, 2.0)]
    configs += [Config(f"{idea}-plan", idea) for idea in (*IDEAS[:5], "P11", "P12")]
    return tuple(configs)


@dataclass(frozen=True)
class Campaign:
    problems: tuple[Problem, ...]
    models: tuple[Model, ...]
    configs: tuple[Config, ...]
    seeds: tuple[int, ...] = (20261010,)
    input_tokens_estimate: int = 512
    answer_tokens: int = 512
    test_generation_tokens: int = 512
    p07_proxy_budget: int = 100000
    candidate_cpu_seconds: int = 30
    calibration_minimum_trials: int = 30
    calibration_minimum_lower_bound: float = 0.8
    version: str = VERSION

    def validate(self) -> None:
        if self.version != VERSION or not self.problems or not self.models or not self.configs or not self.seeds:
            raise ValueError("Empty or incompatible campaign")
        if any(type(n) is not int or n <= 0 for n in (self.input_tokens_estimate, self.answer_tokens,
                                                    self.test_generation_tokens, self.p07_proxy_budget,
                                                    self.candidate_cpu_seconds)):
            raise ValueError("Positive integral budgets required")
        if self.candidate_cpu_seconds > 30:
            raise ValueError("Future adapter CPU ceiling exceeds existing extra-phase envelope")
        if type(self.calibration_minimum_trials) is not int or self.calibration_minimum_trials <= 0:
            raise ValueError("Positive calibration trial floor required")
        if not 0 < self.calibration_minimum_lower_bound <= 1:
            raise ValueError("Invalid frozen calibration threshold")
        if len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("Unique nonnegative integral seeds required")
        for objects in (self.problems, self.models, self.configs):
            for obj in objects:
                obj.validate()
        for config in self.configs:
            if config.idea == "P07":
                p07_budget(config, self.input_tokens_estimate, self.answer_tokens, self.p07_proxy_budget)
        for keys in ([p.key for p in self.problems], [m.id for m in self.models], [c.id for c in self.configs]):
            if len(keys) != len(set(keys)):
                raise ValueError("Duplicate identity across campaign/splits")
        fingerprints = [p.fingerprint for p in self.problems]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("Duplicate normalized problem; train/dev/test must be disjoint")

    @property
    def hash(self) -> str:
        self.validate()
        return digest({"campaign": asdict(self), "prompts": PROMPTS, "test_prompt": TEST_PROMPT,
                       "protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


@dataclass(frozen=True)
class Freeze:
    campaign_hash: str
    preregistration: str
    registered_at: str
    dev_ids: tuple[str, ...]
    chosen_configs: tuple[tuple[str, str], ...]
    family_prompts: tuple[tuple[str, str], ...]
    calibration_hash: str

    def validate(self, campaign: Campaign) -> None:
        if (self.campaign_hash != campaign.hash or not self.preregistration or not self.registered_at
                or not self.calibration_hash):
            raise ValueError("Test requires matching frozen preregistration")
        dev = {p.key for p in campaign.problems if p.split == "dev"}
        if not self.dev_ids or not set(self.dev_ids) <= dev or len(set(self.dev_ids)) != len(self.dev_ids):
            raise ValueError("Selection evidence must identify dev problems only")
        chosen = dict(self.chosen_configs)
        ideas = {c.idea for c in campaign.configs}
        if len(chosen) != len(self.chosen_configs) or set(chosen) != ideas:
            raise ValueError("Freeze exactly one configuration per registered idea, including inactive plans")
        catalogue = {c.id: c for c in campaign.configs}
        if any(cid not in catalogue or catalogue[cid].idea != idea for idea, cid in chosen.items()):
            raise ValueError("Unknown frozen configuration")
        prompts = dict(self.family_prompts)
        if len(prompts) != len(self.family_prompts) or set(prompts) != {m.family for m in campaign.models}:
            raise ValueError("Freeze prompt selection by family only")
        if any(p not in PROMPTS for p in prompts.values()):
            raise ValueError("Unknown frozen prompt")


def choose_family_prompts(scores: Sequence[tuple[str, str, str, float]], campaign: Campaign) -> dict[str, str]:
    """Choose by family on the same dev problem set per variant, never by test/model/question."""
    campaign.validate()
    dev = {p.key for p in campaign.problems if p.split == "dev"}
    families = {m.family for m in campaign.models}
    groups: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for family, variant, key, score in scores:
        if key not in dev or family not in families or variant not in PROMPTS:
            raise ValueError("Prompt selection is dev-only within registered families")
        finite(score)
        if key in groups[family, variant]:
            raise ValueError("One score per family/variant/problem; aggregate models before selection")
        groups[family, variant][key] = score
    selected = {}
    for family in sorted(families):
        sets = [set(groups[family, variant]) for variant in PROMPTS]
        if not sets[0] or any(keys != sets[0] for keys in sets):
            raise ValueError("All three prompts require identical dev coverage per family")
        selected[family] = min(PROMPTS, key=lambda v: (-sum(groups[family, v].values()) / len(sets[0]), v))
    return selected


def p07_budget(config: Config, input_tokens: int, answer_tokens: int, proxy_cap: int) -> dict:
    """Equal reservation under a token proxy; not FLOPs equality or measured runtime equality.

    Sharing is one broadcast: a producer generates 128 tokens, every other member prefills them.
    Unused allowance remains reserved, not silently spent on extra answers.
    """
    config.validate()
    if config.idea != "P07" or any(type(x) is not int or x <= 0 for x in (input_tokens, answer_tokens, proxy_cap)):
        raise ValueError("Invalid depth/width budget")
    prefill = config.width * input_tokens + (config.width - 1) * config.share_tokens
    decode = config.width * (config.reflection_tokens + answer_tokens) + config.share_tokens
    total = prefill + decode
    if total > proxy_cap:
        raise ValueError("Depth/width/share exceeds common proxy budget")
    return {"prefill_tokens": prefill, "decode_tokens": decode, "used_proxy": total,
            "reserved_proxy": proxy_cap, "unused_proxy": proxy_cap - total, "approximate": True,
            "sharing_rounds": int(config.share_tokens > 0)}


def estimate_cost(model: Model, prefill: int, decode: int) -> dict:
    model.validate()
    if any(type(n) is not int or n < 0 for n in (prefill, decode)):
        raise ValueError("Nonnegative integral token counts required")
    seconds = prefill / model.prefill_tokens_s + decode / model.decode_tokens_s
    return {"estimated_seconds_serial": seconds, "estimated_cost": seconds * model.hourly_cost / 3600,
            "estimated_parameter_token_proxy": model.parameter_billions * (prefill + decode),
            "measured": False, "excludes": ["queue", "network", "sandbox", "loading", "training", "verification"]}


def plan_jobs(campaign: Campaign, split: str = "dev", freeze: Freeze | None = None, *,
              only: tuple[str, str, str, int] | None = None) -> list[dict]:
    campaign.validate()
    if split not in {"dev", "test", "train"}:
        raise ValueError("Unknown split")
    if split == "test":
        if freeze is None:
            raise ValueError("Test jobs blocked until preregistration is frozen")
        freeze.validate(campaign)
    campaign_hash = campaign.hash
    jobs = []
    configs = campaign.configs
    if split == "test":
        configs = tuple(c for c in configs if c.id in dict(freeze.chosen_configs).values())
    for problem in campaign.problems:
        if problem.split != split:
            continue
        for config in configs:
            if split == "train" and config.idea != "P11":
                continue
            for model in campaign.models:
                for seed in campaign.seeds:
                    if only is not None and (problem.key, config.id, model.id, seed) != only:
                        continue
                    base = {"campaign_hash": campaign_hash, "split": split, "problem_id": problem.key,
                            "problem_fingerprint": problem.fingerprint, "domain": problem.domain,
                            "model": asdict(model), "seed": seed, "config": asdict(config),
                            "execution": "dry-run", "native_chat_template_required": True}
                    if config.idea == "P06":
                        # Separate payload construction: neither candidate nor gold is an argument here.
                        test_job = {**base, "kind": "test_generation", "messages": [
                            {"role": "user", "content": TEST_PROMPT + problem.prompt}],
                            "max_new_tokens": campaign.test_generation_tokens,
                            "candidate_independent": True, "generated_tests_are_fallible": True}
                        test_job["estimate"] = estimate_cost(model, campaign.input_tokens_estimate,
                                                             campaign.test_generation_tokens)
                        test_job["job_id"] = digest(test_job)
                        jobs.append(test_job)
                    variant = config.prompt_variant
                    if config.idea == "P08" and split == "test":
                        variant = dict(freeze.family_prompts)[model.family]
                    counts = {"prefill_tokens": campaign.input_tokens_estimate,
                              "decode_tokens": campaign.answer_tokens + config.reflection_tokens}
                    if config.idea == "P07":
                        counts = p07_budget(config, campaign.input_tokens_estimate, campaign.answer_tokens,
                                            campaign.p07_proxy_budget)
                    if config.idea == "P06":
                        counts = {k: v * config.candidate_count for k, v in counts.items()}
                    implemented = config.idea in {"P06", "P07", "P08"}
                    job = {**base, "kind": "candidate_plan" if implemented else "future_specification",
                           "messages": [{"role": "user", "content": PROMPTS[variant] + problem.prompt}],
                           "prompt_variant": variant, "token_budget": counts,
                           "max_new_tokens": campaign.answer_tokens + config.reflection_tokens,
                           "sandbox_contract": "essaim.sandbox:_candidate/extra; adapter required; never evaluate",
                           "candidate_cpu_seconds": campaign.candidate_cpu_seconds,
                           "estimate": estimate_cost(model, counts["prefill_tokens"], counts["decode_tokens"])}
                    job["job_id"] = digest(job)
                    jobs.append(job)
    return jobs


@dataclass(frozen=True)
class Calibration:
    domain: str
    dev_ids: tuple[str, ...]
    valid_proposals: int
    total_proposals: int
    minimum_trials: int = 30
    minimum_lower_bound: float = 0.8

    def validated(self, campaign: Campaign, domain: str) -> bool:
        campaign.validate()
        eligible = {p.key for p in campaign.problems if p.split == "dev" and p.domain == self.domain}
        if (not self.dev_ids or len(set(self.dev_ids)) != len(self.dev_ids)
                or not set(self.dev_ids) <= eligible):
            raise ValueError("Calibration evidence must be dev-only in the same domain")
        if (any(type(n) is not int for n in (self.valid_proposals, self.total_proposals, self.minimum_trials))
                or not 0 <= self.valid_proposals <= self.total_proposals or self.minimum_trials <= 0):
            raise ValueError("Invalid calibration counts")
        if not 0 < self.minimum_lower_bound <= 1:
            raise ValueError("Invalid calibration threshold")
        if (self.minimum_trials, self.minimum_lower_bound) != (campaign.calibration_minimum_trials,
                                                              campaign.calibration_minimum_lower_bound):
            raise ValueError("Calibration thresholds must match frozen campaign policy")
        if self.domain != domain or self.total_proposals < self.minimum_trials:
            return False
        n, z = self.total_proposals, 1.959963984540054
        p = self.valid_proposals / n
        lower = (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
        return lower >= self.minimum_lower_bound


@dataclass(frozen=True)
class CandidateScore:
    id: str
    family: str
    answer: str
    passed: int
    total: int
    test_set_hash: str


def select_functional(candidates: Sequence[CandidateScore], calibration: Calibration | None,
                      campaign: Campaign, problem_key: str, freeze: Freeze | None = None) -> str | None:
    """One representative and one vote per family on a common independently generated suite."""
    campaign.validate()
    problem = next((p for p in campaign.problems if p.key == problem_key), None)
    if problem is None or problem.split == "train":
        raise ValueError("Registered evaluation problem required")
    if problem.split == "test":
        if freeze is None:
            raise ValueError("Test selection requires frozen preregistration")
        freeze.validate(campaign)
        if digest(asdict(calibration) if calibration else None) != freeze.calibration_hash:
            raise ValueError("Test calibration must match frozen dev evidence")
    if calibration is None or not calibration.validated(campaign, problem.domain):
        return None
    if not candidates:
        return None
    if len({c.id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    if len({(c.test_set_hash, c.total) for c in candidates}) != 1:
        raise ValueError("All candidates require the same test suite and denominator")
    families = {m.family for m in campaign.models}
    representatives = {}
    for candidate in sorted(candidates, key=lambda c: (-c.passed, c.id)):
        if (candidate.family not in families or not candidate.test_set_hash or candidate.total <= 0
                or not 0 <= candidate.passed <= candidate.total or not candidate.answer
                or type(candidate.passed) is not int or type(candidate.total) is not int):
            raise ValueError("Invalid score")
        representatives.setdefault(candidate.family, candidate)
    # A partial/failed candidate is not a valid functional solution under this frozen policy.
    votes = Counter(c.answer for c in representatives.values() if c.passed == c.total)
    return min(votes, key=lambda answer: (-votes[answer], answer)) if votes else None


class CpuSandbox(Protocol):
    """Future adapter to the existing extra-phase sandbox, not implemented or called here.

    Each call must isolate one candidate, enforce CPU limits, and receive inputs only. Generated
    assertions must never reach the unsandboxed trusted grader: interpret a bounded declarative schema.
    """

    def run_candidate(self, *, code: str, entry: str, encoded_inputs: tuple[str, ...],
                      cpu_seconds: int) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class CandidateInputs:
    encoded_inputs: tuple[str, ...]


@dataclass(frozen=True)
class GraderGold:
    expected_outputs: tuple[str, ...]
    declarative_assertions: tuple[str, ...]


def candidate_payload(inputs: CandidateInputs) -> dict:
    if type(inputs) is not CandidateInputs or any(type(x) is not str for x in inputs.encoded_inputs):
        raise ValueError("Candidate receives encoded inputs only, never a grader bundle")
    return {"inputs": list(inputs.encoded_inputs)}


class Inference(Protocol):
    def generate(self, *, model: Model, messages: tuple[dict, ...], seed: int,
                 max_new_tokens: int, native_chat_template: str) -> str: ...


def infer(job: Mapping, backend: Inference, campaign: Campaign, *, freeze: Freeze | None = None,
          enabled: bool = False) -> str:
    """Explicit Python injection only. No HTTP client, endpoint discovery, retries, or downloads."""
    if enabled is not True:
        raise ValueError("Inference disabled; explicit opt-in required")
    identity = (job["problem_id"], job["config"]["id"], job["model"]["id"], job["seed"])
    if job not in plan_jobs(campaign, job["split"], freeze, only=identity):
        raise ValueError("Only unmodified jobs from the registered campaign are accepted")
    if job["kind"] not in {"test_generation", "candidate_plan"} or job["config"]["idea"] not in {"P06", "P08"}:
        raise ValueError("This seam supports individual P06/P08 generations only, not a full pipeline")
    model = Model(**job["model"])
    model.validate()
    if not model.license_verified:
        raise ValueError("Local license review required before inference")
    return backend.generate(model=model, messages=tuple(job["messages"]), seed=job["seed"],
                            max_new_tokens=job["max_new_tokens"], native_chat_template=model.native_chat_template)


@dataclass(frozen=True)
class SpeculationPlan:
    draft_tokenizers: tuple[str, ...]
    target_tokenizer: str
    draft_tokens: int
    alignment: str
    exact_verifier: str
    transport: str = "local"

    def validate(self) -> None:
        if not self.draft_tokenizers or not self.target_tokenizer or any(not t for t in self.draft_tokenizers):
            raise ValueError("Pinned tokenizer fingerprints required")
        if self.draft_tokens not in {4, 8, 16} or self.transport != "local":
            raise ValueError("Plan requires local block experiments, not WAN per token")
        if self.alignment == "identical-tokenizers":
            if any(t != self.target_tokenizer for t in self.draft_tokenizers):
                raise ValueError("Every draft tokenizer must match target")
            if self.exact_verifier != "target-ratio-residual":
                raise ValueError("Exact target ratio/residual verifier required")
        elif self.alignment == "retokenize-text":
            if self.exact_verifier != "target-string-law-residual":
                raise ValueError("Retokenization alone cannot establish exact target distribution")
        else:
            raise ValueError("Unsupported tokenizer alignment")


def speculation_metrics(proposed: int, accepted: int, rtt_seconds: Sequence[float]) -> dict:
    if any(type(n) is not int for n in (proposed, accepted)) or not 0 <= accepted <= proposed:
        raise ValueError("Invalid acceptance counts")
    for value in rtt_seconds:
        finite(value)
    return {"proposed": proposed, "accepted": accepted,
            "acceptance_rate": accepted / proposed if proposed else None,
            "rtt_mean_seconds": sum(rtt_seconds) / len(rtt_seconds) if rtt_seconds else None,
            "rtt_total_seconds": sum(rtt_seconds), "rtt_samples": len(rtt_seconds)}


@dataclass(frozen=True)
class Distribution:
    space: str
    labels: tuple[str, ...]
    probabilities: tuple[float, ...]

    def validate(self) -> None:
        if (not self.space or not self.labels or len(self.labels) != len(set(self.labels))
                or len(self.labels) != len(self.probabilities)):
            raise ValueError("Invalid response space")
        for p in self.probabilities:
            finite(p)
        if not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-9, rel_tol=0):
            raise ValueError("Normalized response probabilities required; raw logits are not accepted")


def align_distribution(source: Distribution, target_space: str, target_labels: tuple[str, ...],
                       mapping: Mapping[str, str]) -> Distribution:
    source.validate()
    if (set(mapping) != set(source.labels) or len(set(mapping.values())) != len(mapping)
            or set(mapping.values()) != set(target_labels) or len(set(target_labels)) != len(target_labels)):
        raise ValueError("Explicit complete bijection required; incompatible mappings are refused")
    values = {mapping[label]: p for label, p in zip(source.labels, source.probabilities)}
    aligned = Distribution(target_space, target_labels, tuple(values[label] for label in target_labels))
    aligned.validate()
    return aligned


def normalize_logs(log_values: Sequence[float]) -> tuple[float, ...]:
    maximum = max(log_values)
    if maximum == -math.inf:
        raise ValueError("No common probability support; abstain")
    values = [math.exp(x - maximum) for x in log_values]
    return tuple(x / sum(values) for x in values)


def temperature_logs(log_values: Sequence[float], temperature: float) -> tuple[float, ...]:
    maximum = max(log_values)
    if maximum == -math.inf:
        raise ValueError("No common probability support; abstain")
    return tuple((x - maximum) / temperature for x in log_values)


def pool_probabilities(distributions: Sequence[Distribution], *, operator: str = "linear",
                       temperature: float = 1.0, weights: Sequence[float] | None = None) -> Distribution:
    finite(temperature, positive=True)
    if not distributions or operator not in {"linear", "product"}:
        raise ValueError("Nonempty registered pool required")
    first = distributions[0]
    for distribution in distributions:
        distribution.validate()
        if (distribution.space, distribution.labels) != (first.space, first.labels):
            raise ValueError("Align common response spaces first; never mix tokenizer logits")
    weights = tuple(weights) if weights is not None else (1.0,) * len(distributions)
    if len(weights) != len(distributions):
        raise ValueError("One weight per distribution required")
    for weight in weights:
        finite(weight)
    finite(sum(weights), positive=True)
    normalized_weights = tuple(w / sum(weights) for w in weights)
    logs = [tuple(math.log(p) if p else -math.inf for p in d.probabilities) for d in distributions]
    if operator == "linear":
        softened = [normalize_logs(temperature_logs(values, temperature)) for values in logs]
        output = tuple(sum(w * p[i] for w, p in zip(normalized_weights, softened))
                       for i in range(len(first.labels)))
    else:
        # Per-model normalization constants cancel. Combine raw log support before any exponentiation
        # or division by temperature; otherwise small positive probabilities can turn into false zeros.
        active = [(weight, values) for original, weight, values in zip(weights, normalized_weights, logs)
                  if original > 0]
        combined = tuple(-math.inf if any(values[i] == -math.inf for _, values in active)
                         else sum(weight * values[i] for weight, values in active)
                         for i in range(len(first.labels)))
        output = normalize_logs(temperature_logs(combined, temperature))
    result = Distribution(first.space, first.labels, output)
    result.validate()
    return result


def p11_training_plan(campaign: Campaign, *, e13_ready: bool, e14_ready: bool,
                      p06_p07_gain_validated: bool) -> dict:
    campaign.validate()
    if not all(type(flag) is bool and flag for flag in (e13_ready, e14_ready, p06_p07_gain_validated)):
        raise ValueError("P11 blocked pending E13/E14 prerequisites and validated P06/P07 gain")
    train = [p for p in campaign.problems if p.split == "train"]
    if not train:
        raise ValueError("External synthetic TRAIN required")
    return {"execution": "future_specification", "train_ids": [p.key for p in train],
            "train_fingerprints": [p.fingerprint for p in train], "benchmark_training_allowed": False,
            "stages": ["sandbox-validated synthetic dialogues", "complementary family LoRA ensemble",
                       "teacher ensemble soft targets", "small student distillation"],
            "training_launched": False}


def load_campaign(path: Path) -> Campaign:
    """Strict prompt-only schema: solution/hidden/gold fields are refused rather than stripped."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data["problems"] = tuple(Problem(**p) for p in data["problems"])
    data["models"] = tuple(Model(**m) for m in data["models"])
    data["configs"] = tuple(Config(**c) for c in data["configs"])
    if "seeds" in data:
        data["seeds"] = tuple(data["seeds"])
    campaign = Campaign(**data)
    campaign.validate()
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Strict prompt-only JSON manifest")
    parser.add_argument("--output", type=Path, required=True, help="New dry-run JSONL file; never overwritten")
    parser.add_argument("--split", choices=("dev", "test", "train"), default="dev")
    parser.add_argument("--freeze", type=Path, help="Frozen preregistration JSON; mandatory for test")
    args = parser.parse_args()
    campaign = load_campaign(args.manifest)
    freeze = None
    if args.freeze:
        data = json.loads(args.freeze.read_text(encoding="utf-8"))
        data["dev_ids"] = tuple(data["dev_ids"])
        for field in ("chosen_configs", "family_prompts"):
            data[field] = tuple(tuple(pair) for pair in data[field])
        freeze = Freeze(**data)
    jobs = plan_jobs(campaign, args.split, freeze)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for job in jobs:
            stream.write(json.dumps(job, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    print(json.dumps({"campaign_hash": campaign.hash, "jobs": len(jobs), "execution": "dry-run"}))


if __name__ == "__main__":
    main()
