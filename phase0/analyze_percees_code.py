"""P06 exploratory DEV replay; parse stored logs only, never load datasets or execute candidates.

Run with Python from the repository root. Inputs are limited to a DEV-only snapshot outside git.
All choices, including ties and folds, are declared in CONFIG before opening any input.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from essaim import code
from essaim.models import total_b


SWARM = (
    "Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it", "ibm-granite/granite-4.2-3b",
    "HuggingFaceTB/SmolLM3-3B", "mistralai/Ministral-3-3B-Instruct-2512",
    "microsoft/Phi-4-mini-instruct", "allenai/OLMo-2-0425-1B-Instruct",
)
EXTRA = ("Qwen/Qwen3.5-2B", "google/gemma-4-E2B-it")
REFERENCES = (
    "Qwen/Qwen3.5-9B", "google/gemma-4-12B-it",
    "mistralai/Ministral-3-14B-Instruct-2512", "Qwen/Qwen3.8-27B",
)
MODELS = SWARM + EXTRA + REFERENCES
BENCHES = ("humanevalplus", "mbppplus")
EXPECTED_N = {"humanevalplus": 82, "mbppplus": 189}
RULES = ("text", "visible", "count", "families", "wfamilies")
VARIANTS = tuple((k, rule) for k in (1, 3, 5) for rule in RULES)
CONFIG = {
    "registration": "PERC-20261010-v1-A1", "outer_folds": 5, "inner_folds": 3,
    "fold_formula": "int(SHA256('percees-v1:' + qid), 16) % n_folds",
    "variant_order": list(VARIANTS), "tie_order": "higher internal macro score; lower k; RULES order",
    "model_order": list(MODELS), "primary_baseline": [5, "visible"],
    "functional_baseline": [5, "wfamilies"], "seed": 20261010, "bootstrap_draws": 2000,
    "weights": "train greedy logit clipped [0.02,0.98] minus log(train wrong visible collision)",
    "collision_clip": [0.001, 0.999], "collision_empty_default": 0.5,
    "count": "distinct program hashes within a functional group",
    "scope": "DEV only; reference failure exclusions before folds; no confirmatory p values",
}
PROTOCOL = ("data", "bench", "split", "n", "prompt", "max_tokens", "thinking", "ctx_per_slot",
            "samples", "greedy", "sampling")


@dataclass(frozen=True)
class Candidate:
    qid: str
    model: str
    sample: int
    prog: str
    text: str
    tokens: int
    prompt_tokens: int


@dataclass(frozen=True)
class PublicFeatures:
    load: bool
    visible: bool
    visible_count: int
    signature: tuple[str, ...]


@dataclass
class BenchData:
    bench: str
    candidates: dict[str, list[Candidate]]
    features: dict[Candidate, PublicFeatures]
    labels: dict[Candidate, bool]
    validation: dict


class IncompleteData(ValueError):
    """Any incomplete, inconsistent or unbound input blocks all scoring."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold(qid: str, n: int = 5) -> int:
    return int(hashlib.sha256(("percees-v1:" + qid).encode()).hexdigest(), 16) % n


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IncompleteData(message)


def read_jsonl(path: Path, keys: tuple[str, ...]) -> list[dict]:
    require(path.exists(), f"missing {path.name}")
    raw = path.read_bytes()
    require(bool(raw) and raw.endswith(b"\n"), f"empty or unterminated input {path.name}")
    rows, seen = [], set()
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
            key = tuple(row[k] for k in keys)
        except (ValueError, TypeError, KeyError) as exc:
            raise IncompleteData(f"partial/invalid row {path.name}:{number}: {exc}") from exc
        require(key not in seen, f"duplicate {key} in {path.name}")
        seen.add(key)
        rows.append(row)
    return rows


def manifest(path: Path) -> dict:
    meta = path.with_name(path.name + ".meta.json")
    require(meta.exists(), f"missing manifest {meta.name}")
    try:
        result = json.loads(meta.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise IncompleteData(f"invalid manifest {meta.name}") from exc
    require(isinstance(result, dict), f"invalid manifest object {meta.name}")
    return result


def check_generation(row: dict, man: dict, bench: str, model: str) -> None:
    fields = ("id", "sample", "model", "bench", "seed", "temperature", "text", "code", "prog",
              "extract", "finish", "n_tokens", "prompt_tokens", "ms", "reasoning")
    require(all(k in row for k in fields), f"partial generation {model}")
    sample = row["sample"]
    require(type(sample) is int and 0 <= sample <= 4, f"invalid sample {model}")
    require(row["model"] == model and row["bench"] == bench, f"wrong generation identity {model}")
    require(isinstance(row["id"], str) and bool(row["id"]), f"invalid ID {model}")
    require(isinstance(row["text"], str) and isinstance(row["code"], str), f"invalid source {model}")
    require(row["seed"] == (man["greedy"]["seed"] if sample == 0 else man["sampling"]["seeds"][sample - 1]),
            f"wrong sampling seed {model}")
    require(row["temperature"] == (0.0 if sample == 0 else man["sampling"]["temperature"]),
            f"wrong sampling temperature {model}")
    require(row["finish"] in ("stop", "length"), f"unfinished generation {model}")
    require(all(type(row[k]) is int and row[k] >= 0 for k in ("n_tokens", "prompt_tokens", "ms")),
            f"invalid cost counters {model}")
    require(row["prog"] == code.prog_id(row["code"]), f"stored program hash mismatch {model}/{row['id']}")
    # The logs do not retain entry points. Try names in PUBLIC generated text, and require exact re-extraction.
    # No dataset (and therefore no hidden test source) is opened to recover an entry point.
    names = set(re.findall(r"\bdef\s+([A-Za-z_]\w*)\s*\(", row["text"])) | {"__unknown_entry__"}
    require(any(code.extract_code(row["text"], name) == (row["code"], row["extract"]) for name in sorted(names)),
            f"raw text/extraction mismatch {model}/{row['id']}/{sample}")


def check_execution(row: dict, bench: str) -> None:
    require(all(k in row for k in ("id", "prog", "bench", "load", "visible", "visible_err", "extra",
                                  "hidden", "status", "ms")), f"partial execution {bench}")
    require(row["bench"] == bench and isinstance(row["id"], str), f"wrong execution identity {bench}")
    require(row["load"] is None or isinstance(row["load"], str), f"invalid load status {bench}")
    require(isinstance(row["visible"], list) and all(type(x) is bool for x in row["visible"]),
            f"invalid visible results {bench}")
    require(isinstance(row["visible_err"], list) and len(row["visible_err"]) == len(row["visible"]),
            f"partial visible results {bench}")
    require(all(x is None or isinstance(x, str) for x in row["visible_err"]), f"invalid visible errors {bench}")
    require(isinstance(row["extra"], list) and all(isinstance(x, str) for x in row["extra"]),
            f"invalid signatures {bench}")
    h = row["hidden"]
    require(isinstance(h, dict) and all(k in h for k in ("pass", "n_pass", "n", "err")),
            f"partial hidden result {bench}")
    require(type(h["pass"]) is bool and type(h["n"]) is int and type(h["n_pass"]) is int
            and 0 <= h["n_pass"] <= h["n"] and h["n"] > 0, f"invalid hidden counters {bench}")
    require(not h["pass"] or (h["n_pass"] == h["n"] and h["err"] is None),
            f"inconsistent successful hidden result {bench}")
    require(isinstance(row["status"], dict) and set(row["status"]) == {"visible", "extra", "hidden"},
            f"partial execution status {bench}")
    require(all(x in ("ok", "timeout", "overflow", "crash") for x in row["status"].values()),
            f"unknown execution status {bench}")
    require(not h["pass"] or row["status"]["hidden"] == "ok", f"successful unfinished execution {bench}")
    require(type(row["ms"]) is int and row["ms"] >= 0, f"invalid execution counter {bench}")


def load_bench(snapshot: Path, bench: str) -> BenchData:
    mans, generations, ids = {}, {}, None
    for model in MODELS:
        path = snapshot / f"code_{model.replace('/', '__')}_colab_{bench}_dev.jsonl"
        man = manifest(path)
        require(all(k in man for k in PROTOCOL + ("model", "revision", "weights", "engine", "backend", "gguf")),
                f"partial manifest {path.name}")
        require(man["model"] == model and man["bench"] == bench and man["split"] == "dev"
                and man["n"] == EXPECTED_N[bench] and man["samples"] == 4, f"wrong manifest {path.name}")
        require(man["prompt"] == code.PROMPT_VERSION and len(man["sampling"]["seeds"]) == 4,
                f"unexpected protocol {path.name}")
        require(re.fullmatch(r"[0-9a-f]{64}", man["weights"]["sha256"]) is not None
                and re.fullmatch(r"[0-9a-f]{64}", man["data"]["sha256"]) is not None,
                f"invalid provenance hashes {path.name}")
        rows = read_jsonl(path, ("id", "sample"))
        for row in rows:
            check_generation(row, man, bench, model)
        model_ids = {r["id"] for r in rows}
        require(len(model_ids) == man["n"] and len(rows) == man["n"] * 5, f"incomplete generation {path.name}")
        require({(r["id"], r["sample"]) for r in rows} == set(itertools.product(model_ids, range(5))),
                f"missing sample/ID {path.name}")
        require(ids is None or ids == model_ids, f"unaligned generation IDs {path.name}")
        ids = model_ids
        mans[model], generations[model] = man, rows
    require(len({json.dumps({k: man[k] for k in PROTOCOL}, sort_keys=True) for man in mans.values()}) == 1,
            f"generation protocols differ {bench}")
    ep = snapshot / f"codeexec_e11_colab_{bench}_dev.jsonl"
    em = manifest(ep)
    require(all(k in em for k in ("models", "data", "n", "bench", "split", "tests", "extract",
                                 "extra_n", "reference_only", "isolation", "python", "numpy")),
            f"partial execution manifest {bench}")
    require(em["bench"] == bench and em["split"] == "dev" and em["n"] == EXPECTED_N[bench]
            and em["data"] == next(iter(mans.values()))["data"] and em["tests"] == code.TESTS_VERSION
            and em["extract"] == code.EXTRACT_VERSION and em["reference_only"] is False
            and sorted(em["models"]) == sorted(MODELS) and em["extra_n"] == 16,
            f"execution/generation manifest mismatch {bench}")
    rows = read_jsonl(ep, ("id", "prog"))
    for row in rows:
        check_execution(row, bench)
    executions = {(r["id"], r["prog"]): r for r in rows}
    expected = {(q, "reference") for q in ids}
    expected.update((g["id"], g["prog"]) for gs in generations.values() for g in gs)
    require(set(executions) == expected,
            f"execution alignment {bench}: {len(expected - set(executions))} missing, "
            f"{len(set(executions) - expected)} unexpected")
    candidates, features, labels, excluded = {}, {}, {}, []
    for q in sorted(ids):
        ref = executions[q, "reference"]
        if not ref["hidden"]["pass"]:
            excluded.append({"id": q, "reason": ref["hidden"], "status": ref["status"]})
        valid = [i for i, ok in enumerate(ref["visible"]) if ok]
        candidates[q] = []
        for model in MODELS:
            for g in sorted((g for g in generations[model] if g["id"] == q), key=lambda g: g["sample"]):
                r = executions[q, g["prog"]]
                require(len(r["visible"]) == len(ref["visible"]) and len(r["extra"]) == len(ref["extra"])
                        and r["hidden"]["n"] == ref["hidden"]["n"], f"partial execution arrays {bench}/{q}")
                c = Candidate(q, model, g["sample"], g["prog"], code.normalized_text(g["code"]),
                              g["n_tokens"], g["prompt_tokens"])
                n_visible = sum(r["visible"][i] for i in valid)
                features[c] = PublicFeatures(r["load"] is None, r["load"] is None and n_visible == len(valid),
                                             n_visible, tuple(r["extra"]))
                labels[c] = r["hidden"]["pass"]
                candidates[q].append(c)
        if not ref["hidden"]["pass"]:
            del candidates[q]
    info = {
        "complete": True, "n_manifest": len(ids), "n_scored": len(candidates), "excluded": excluded,
        "generation_rows": {m: len(gs) for m, gs in generations.items()}, "execution_rows": len(rows),
        "linked_distinct_programs": len(expected) - len(ids), "generation_manifests": mans,
        "execution_manifest": em, "ids": sorted(ids),
        "rejected_visible_tests": sum(sum(not ok for ok in executions[q, "reference"]["visible"]) for q in ids),
        "status_counts": {phase: dict_counts(r["status"][phase] for r in rows)
                          for phase in ("visible", "extra", "hidden")},
        "linkage_limit": "manifests lack generation-file digests and official split-ID digests; "
                         "linkage is exact row-program hashes plus aligned ID/sample/protocol coverage; "
                         "entry points are inferred from public generated text, not independently attested",
    }
    return BenchData(bench, candidates, features, labels, info)


def dict_counts(values) -> dict:
    result = defaultdict(int)
    for value in values:
        result[value] += 1
    return dict(result)


def pool(data: BenchData, qid: str, models: tuple[str, ...], k: int) -> list[Candidate]:
    return [c for c in data.candidates[qid] if c.model in models and c.sample < k]


def rank(candidate: Candidate, weights: dict[str, float]):
    return (weights.get(candidate.model, 0), -candidate.sample, -MODELS.index(candidate.model))


def select(candidates: list[Candidate], features: dict[Candidate, PublicFeatures],
           weights: dict[str, float], rule: str) -> Candidate:
    """No labels, hidden fields, training data or evaluation object may cross this interface."""
    require(bool(candidates), "empty selection pool")
    pick = lambda group: max(group, key=lambda c: rank(c, weights))
    if rule == "text":
        groups = defaultdict(list)
        for c in candidates:
            if c.text:
                groups[c.text].append(c)
        if not groups:
            return pick(candidates)
        return pick(max(groups.values(), key=lambda g: (len(g), rank(pick(g), weights))))
    passing = [c for c in candidates if features[c].visible]
    if rule == "visible":
        if passing:
            return pick(passing)
        return max(candidates, key=lambda c: (features[c].visible_count, rank(c, weights)))
    require(rule in ("count", "families", "wfamilies"), f"unknown rule {rule}")
    eligible = passing or [c for c in candidates if features[c].load]
    groups = defaultdict(list)
    for c in eligible:
        if any(s != "!" for s in features[c].signature):
            groups[features[c].signature].append(c)
    if not groups:
        return pick(eligible or candidates)

    def group_rank(group):
        families = {c.model for c in group}
        distinct = len({c.prog for c in group})
        score = distinct if rule == "count" else len(families) if rule == "families" else sum(
            weights.get(m, 0) for m in sorted(families))
        return score, len(families), distinct, rank(pick(group), weights)

    return pick(max(groups.values(), key=group_rank))


def fit(data: BenchData, ids: list[str], models: tuple[str, ...]) -> dict:
    """Fit K weights ONLY on the supplied training IDs, using greedy candidates."""
    require(bool(ids), "empty training fold")
    greedy = {q: pool(data, q, models, 1) for q in ids}
    probabilities = {m: statistics.mean(data.labels[next(c for c in greedy[q] if c.model == m)] for q in ids)
                     for m in models}
    denominator = numerator = 0
    for q in ids:
        wrong = [c for c in greedy[q] if not data.labels[c] and data.features[c].visible
                 and any(s != "!" for s in data.features[c].signature)]
        for a, b in itertools.combinations(wrong, 2):
            denominator += 1
            numerator += data.features[a].signature == data.features[b].signature
    collision = numerator / denominator if denominator else 0.5
    collision = min(max(collision, 0.001), 0.999)
    weights = {}
    for m, p in probabilities.items():
        p = min(max(p, 0.02), 0.98)
        weights[m] = max(0.0, math.log(p / (1 - p)) - math.log(collision))
    return {"weights": weights, "greedy_accuracy": probabilities, "collision": collision,
            "collision_numerator": numerator, "collision_denominator": denominator, "train_ids": ids}


def predict(data: BenchData, ids: list[str], models: tuple[str, ...], variant: tuple[int, str],
            weights: dict) -> dict[str, Candidate]:
    k, rule = variant
    return {q: select(pool(data, q, models, k), data.features, weights, rule) for q in ids}


def score(data: BenchData, predictions: dict[str, Candidate]) -> float:
    require(bool(predictions), "empty scoring fold")
    return statistics.mean(data.labels[c] for c in predictions.values())


def choose_variant(datasets: dict[str, BenchData], train_ids: dict[str, list[str]],
                   models: tuple[str, ...], outer: int, name: str, journal: list[dict]) -> tuple[int, str]:
    all_predictions = {v: {b: {} for b in BENCHES} for v in VARIANTS}
    for inner in range(3):
        fitted, validation_ids = {}, {}
        for b, data in datasets.items():
            inner_train = [q for q in train_ids[b] if fold(q, 3) != inner]
            validation_ids[b] = [q for q in train_ids[b] if fold(q, 3) == inner]
            require(bool(validation_ids[b]), "empty inner validation fold")
            fitted[b] = fit(data, inner_train, models)
        for variant in VARIANTS:
            accuracies = {}
            for b, data in datasets.items():
                preds = predict(data, validation_ids[b], models, variant, fitted[b]["weights"])
                all_predictions[variant][b].update(preds)
                accuracies[b] = score(data, preds)
            journal.append({"type": "inner_trial", "experiment": name, "outer": outer, "inner": inner,
                            "variant": variant, "scores": accuracies, "macro": statistics.mean(accuracies.values()),
                            "fits": fitted, "validation_ids": validation_ids})
    internal = {v: {b: score(data, all_predictions[v][b]) for b, data in datasets.items()} for v in VARIANTS}
    best = max(VARIANTS, key=lambda v: (statistics.mean(internal[v].values()), -VARIANTS.index(v)))
    journal.append({"type": "decision", "experiment": name, "outer": outer, "selected": best,
                    "internal_scores": [{"variant": v, "scores": internal[v],
                                         "macro": statistics.mean(internal[v].values())} for v in VARIANTS]})
    return best


def costs(candidates: list[Candidate], rule: str) -> dict:
    output = sum(c.tokens for c in candidates)
    prompt = sum(c.prompt_tokens for c in candidates)
    return {"candidates": len(candidates), "output_tokens": output, "prompt_tokens": prompt,
            "generation_tokens": output + prompt,
            "params_b_times_tokens": sum(total_b(c.model) * (c.tokens + c.prompt_tokens) for c in candidates),
            "distinct_programs": len({c.prog for c in candidates}),
            "selector_distinct_executions": 0 if rule == "text" else len({c.prog for c in candidates})}


def bootstrap_difference(a: dict[str, list[bool]], b: dict[str, list[bool]]) -> dict:
    """Paired resampling of questions within each fixed benchmark; macro average of two benches."""
    rng = random.Random(CONFIG["seed"])
    differences = {bench: [int(x) - int(y) for x, y in zip(a[bench], b[bench], strict=True)] for bench in BENCHES}
    observed = 100 * statistics.mean(statistics.mean(d) for d in differences.values())
    draws = []
    for _ in range(CONFIG["bootstrap_draws"]):
        draws.append(100 * statistics.mean(sum(rng.choice(d) for _ in d) / len(d) for d in differences.values()))
    draws.sort()
    return {"difference_pp": observed, "ci95_descriptive_pp": [draws[49], draws[1949]],
            "draws": CONFIG["bootstrap_draws"], "seed": CONFIG["seed"]}


def analyze(datasets: dict[str, BenchData]) -> tuple[dict, list[dict], list[dict]]:
    journal, records, choices = [], [], []
    for outer in range(5):
        training = {b: [q for q in d.candidates if fold(q) != outer] for b, d in datasets.items()}
        evaluation = {b: [q for q in d.candidates if fold(q) == outer] for b, d in datasets.items()}
        swarm_fits = {b: fit(d, training[b], SWARM) for b, d in datasets.items()}
        best_solo = max(SWARM, key=lambda m: (statistics.mean(swarm_fits[b]["greedy_accuracy"][m] for b in BENCHES),
                                             -SWARM.index(m)))
        variant = choose_variant(datasets, training, SWARM, outer, "P06", journal)
        solo_variants = {m: choose_variant(datasets, training, (m,), outer, f"multi:{m}", journal) for m in MODELS}
        choices.append({"outer": outer, "P06": variant, "best_small_solo": best_solo,
                        "single_model_multi": solo_variants, "swarm_fits": swarm_fits})
        for b, data in datasets.items():
            weights = swarm_fits[b]["weights"]
            methods = {
                "P06": (SWARM, variant, weights), "visible_k5": (SWARM, (5, "visible"), weights),
                "wfamilies_k5": (SWARM, (5, "wfamilies"), weights),
                "best_small_greedy": ((best_solo,), (1, "visible"), {best_solo: 1.0}),
            }
            for m in MODELS:
                solo_fit = fit(data, training[b], (m,))
                methods[f"greedy:{m}"] = ((m,), (1, "visible"), solo_fit["weights"])
                methods[f"multi:{m}"] = ((m,), solo_variants[m], solo_fit["weights"])
            for method, (models, v, w) in methods.items():
                predictions = predict(data, evaluation[b], models, v, w)
                for q, c in predictions.items():
                    records.append({"bench": b, "id": q, "outer": outer, "method": method, "variant": v,
                                    "selected": {"model": c.model, "sample": c.sample, "prog": c.prog},
                                    "correct": data.labels[c], "cost": costs(pool(data, q, models, v[0]), v[1])})
    methods = sorted({r["method"] for r in records})
    results, paired = {}, {}
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        scores = {b: statistics.mean(r["correct"] for r in rows if r["bench"] == b) for b in BENCHES}
        paired[method] = {b: [r["correct"] for r in sorted(rows, key=lambda r: r["id"]) if r["bench"] == b]
                          for b in BENCHES}
        cost_keys = tuple(rows[0]["cost"])
        by_bench_cost = {b: {k: statistics.mean(r["cost"][k] for r in rows if r["bench"] == b) for k in cost_keys}
                         for b in BENCHES}
        results[method] = {"scores": scores, "macro": statistics.mean(scores.values()),
                           "cost_macro_per_question": {k: statistics.mean(by_bench_cost[b][k] for b in BENCHES)
                                                       for k in cost_keys}, "cost_by_bench": by_bench_cost}
    comparisons = {m: bootstrap_difference(paired["P06"], paired[m]) for m in methods if m != "P06"}
    diagnostics = {}
    for b, data in datasets.items():
        qids = list(data.candidates)
        oracle = {k: statistics.mean(any(data.labels[c] for c in pool(data, q, SWARM, k)) for q in qids)
                  for k in (1, 3, 5)}
        marginal = {}
        for m in SWARM:
            others = tuple(x for x in SWARM if x != m)
            marginal[m] = statistics.mean(any(data.labels[c] for c in pool(data, q, SWARM, 5))
                                           and not any(data.labels[c] for c in pool(data, q, others, 5)) for q in qids)
        eligible = {c for q in qids for c in pool(data, q, SWARM, 5)}
        visible = [c for c in eligible if data.features[c].visible]
        wrong = [c for c in eligible if not data.labels[c]]
        diagnostics[b] = {"oracle": oracle, "oracle_marginal_family": marginal,
                          "visible_false_discovery": (statistics.mean(not data.labels[c] for c in visible)
                                                       if visible else None),
                          "visible_false_acceptance": (statistics.mean(data.features[c].visible for c in wrong)
                                                       if wrong else None),
                          "visible_candidates_n": len(visible), "wrong_candidates_n": len(wrong),
                          "diagnostic_only": True}
    summary = {"status": "complete_dev_exploratory", "config": CONFIG, "results": results,
               "comparisons_P06_minus": comparisons, "choices": choices, "diagnostics": diagnostics,
               "validation": {b: d.validation for b, d in datasets.items()},
               "registry_counts": {"inner_trials": sum(r["type"] == "inner_trial" for r in journal),
                                   "decisions": sum(r["type"] == "decision" for r in journal),
                                   "variants_per_search": len(VARIANTS), "outer_folds": 5,
                                   "searches_per_outer": 1 + len(MODELS)}}
    return summary, journal, records


def report(summary: dict) -> str:
    if summary["status"] == "complete_validated_no_scores":
        return "# P06 E11/dev : validation seule\n\nCouverture DEV validée; aucun score calculé.\n"
    if summary["status"] == "blocked_incomplete":
        return "# P06 E11/dev : analyse bloquée\n\nAucun score calculé.\n\n" + "\n".join(
            f"- {error}" for error in summary["errors"]) + "\n"
    lines = ["# P06 E11/dev : rejeu exploratoire", "",
             "Les deux bancs DEV sont complets selon les manifestes et la liaison exacte des programmes. "
             "Aucun contenu test, dataset complet, GPU, nouvelle inférence ni programme généré n'a été exécuté.", "",
             "Les 13 modèles couvrent chaque ID et les indices 0 à 4, sans doublon ni ligne partielle. "
             "Chaque code stocké correspond à son empreinte et à une ré-extraction du texte public; "
             "chaque programme distinct et chaque référence possède une exécution complète dans le journal. "
             "Les SHA-256 des fichiers et du code sont conservés dans le manifeste du rejeu.", "",
             "Les manifestes historiques ne contiennent ni empreinte des fichiers de génération dans celui "
             "des exécutions, ni manifeste officiel des IDs DEV : la couverture et la liaison des programmes "
             "sont vérifiées, mais cette provenance n'est pas une attestation indépendante du split. "
             "Les points d'entrée sont inférés à partir du texte public généré, sans lecture du dataset.", "",
             "Cinq plis externes déterministes par ID, trois internes limités au train. Quinze variantes "
             "1/3/5 candidats × texte, visible+poids, count, families, wfamilies. Poids K réappris uniquement "
             "sur les questions train de chaque pli et banc. Les décisions utilisent les scores internes macro, "
             "puis moins de candidats, puis l'ordre de règles déclaré. Les fonctions de sélection ne reçoivent "
             "aucune étiquette cachée. Le comparateur primaire est visible+poids k=5; "
             "wfamilies k=5 est aussi affiché.", "",
             "Le groupe count compte les programmes distincts, families et wfamilies donnent une contribution "
             "par famille. Les références et tous les autres modèles disposent de la même recherche interne "
             "multi-échantillons; les solos gloutons sont affichés séparément. Le meilleur solo est choisi "
             "sur train parmi les sept modèles de l'essaim; les deux frères plus petits sont affichés "
             "séparément et ne participent pas à ce choix.", ""]
    for b, v in summary["validation"].items():
        lines.append(f"- {b} : {v['n_manifest']} IDs, {v['n_scored']} notés, "
                     f"{v['linked_distinct_programs']} programmes distincts liés, "
                     f"{v['execution_rows']} exécutions (références incluses). "
                     f"Exclusions de référence : {', '.join(x['id'] for x in v['excluded']) or 'aucune'}.")
    lines += ["", "| Méthode | HumanEval+ (%) | MBPP+ (%) | Macro (%) | Candidats | Programmes distincts | "
              "Tokens génération | Paramètres B × tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    order = ["P06", "visible_k5", "wfamilies_k5", "best_small_greedy"]
    order += [name for m in MODELS for name in (f"greedy:{m}", f"multi:{m}")]
    for name in order:
        r = summary["results"][name]
        c = r["cost_macro_per_question"]
        lines.append(f"| {name} | {100*r['scores'][BENCHES[0]]:.2f} | {100*r['scores'][BENCHES[1]]:.2f} | "
                     f"{100*r['macro']:.2f} | {c['candidates']:.2f} | {c['distinct_programs']:.2f} | "
                     f"{c['generation_tokens']:.0f} | {c['params_b_times_tokens']:.0f} |")
    lines += ["", "Différences appariées P06 − comparateur, points de macro; IC descriptif bootstrap "
              "par question au sein de chaque banc, 2 000 tirages, graine 20261010.", "",
              "| Comparateur | Différence (pp) | IC 95 % descriptif (pp) |", "|---|---:|---:|"]
    for name in ["visible_k5", "best_small_greedy", "wfamilies_k5"] + [f"multi:{m}" for m in REFERENCES]:
        r = summary["comparisons_P06_minus"][name]
        lo, hi = r["ci95_descriptive_pp"]
        lines.append(f"| {name} | {r['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    primary = summary["comparisons_P06_minus"]["visible_k5"]
    lines += ["", f"Le gain primaire observé vaut {primary['difference_pp']:+.2f} points. "
              "Ce résultat DEV reste exploratoire, y compris si son intervalle descriptif exclut zéro. "
              "Il ne démontre pas une percée confirmée. Le bootstrap ne capture pas toute l'incertitude "
              "des apprentissages sur des plis recouvrants. Aucun p confirmatoire ni verdict d'équivalence.", "",
              "Les coûts sont des budgets de génération réellement journalisés (sortie + prompt par appel), "
              "des proxies paramètres totaux × tokens, et des nombres de programmes distincts par question. "
              "La règle texte n'exige pas d'exécution pour sélectionner; les autres règles utilisent les "
              "journaux d'exécution de tous leurs programmes distincts. Ces proxies ne mesurent ni durée "
              "de réseau, ni énergie, ni accélération réelle. Le budget de l'essaim peut atteindre 35 "
              "candidats contre cinq pour une référence; il n'est pas égalisé.", "",
              "## Choix internes et diagnostics", "",
              "| Pli | Variante P06 | Solo train |", "|---|---|---|"]
    for choice in summary["choices"]:
        lines.append(f"| {choice['outer']} | {choice['P06']} | {choice['best_small_solo']} |")
    for b, d in summary["diagnostics"].items():
        false_discovery = d["visible_false_discovery"]
        false_discovery_text = "non applicable" if false_discovery is None else f"{100*false_discovery:.2f} %"
        lines += ["", f"{b} : plafond oracle k=1/3/5 : "
                  + "/".join(f"{100*v:.2f} %" for v in d["oracle"].values()) + ". "
                  + "Faux positifs parmi les candidats acceptés visiblement : "
                  + f"{false_discovery_text}."]
        lines.append("Gains oracle marginaux par famille (pp) : " + "; ".join(
            f"{m} {100*v:.2f}" for m, v in d["oracle_marginal_family"].items()) + ".")
    counts = summary["registry_counts"]
    lines += ["", f"Registre : {counts['inner_trials']} essais internes et {counts['decisions']} décisions; "
              "toutes les variantes, scores internes, poids, IDs train/validation et prédictions externes "
              "sont conservés, y compris les résultats négatifs. Les passes de production, vérifications "
              "indépendantes et corrections techniques sont consignées dans le manifeste et le journal "
              "des tentatives.", "",
              "Les plafonds oracle et contributions marginales utilisent les étiquettes DEV comme diagnostics. "
              "Ils ne sont pas des sélecteurs déployables. Les tests indépendants générés par LLM "
              "ne figurent pas dans ces données; leur gain et celui d'une vraie nouvelle campagne "
              "métamorphique restent non identifiables. Confiance limitée par la taille DEV, les "
              "bancs fixes et les limites du harnais : confirmation future sur de nouvelles tâches avec "
              "configuration et provenance figées avant ouverture.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=Path("C:/tmp/percees/e11-dev"))
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--technical-correction", action="append", default=[])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected_names = {f"code_{m.replace('/', '__')}_colab_{b}_dev.jsonl{suffix}"
                      for m in MODELS for b in BENCHES for suffix in ("", ".meta.json")}
    expected_names.update(f"codeexec_e11_colab_{b}_dev.jsonl{suffix}"
                          for b in BENCHES for suffix in ("", ".meta.json"))
    inventory = {p.name for p in args.snapshot.iterdir() if p.is_file()}
    errors, datasets = [], {}
    if inventory != expected_names:
        errors.append(f"snapshot inventory mismatch: missing={sorted(expected_names - inventory)}, "
                      f"unexpected={sorted(inventory - expected_names)}")
    history_path = args.output / "percees_code_attempts.jsonl"
    history = read_jsonl(history_path, ("attempt", "event")) if history_path.exists() else []
    attempt = 1 + max((r["attempt"] for r in history), default=0)
    # Preserve all earlier artifacts before updating the convenient latest-result paths.
    if attempt > 1:
        for suffix in ("summary.json", "manifest.json", "registry.jsonl", "predictions.jsonl", "report.md"):
            previous = args.output / f"percees_code_{suffix}"
            if previous.exists():
                archive = args.output / f"percees_code_attempt{attempt - 1:04d}_{suffix}"
                require(not archive.exists(), f"attempt archive exists: {archive.name}")
                archive.write_bytes(previous.read_bytes())
    provenance = {"attempt": attempt, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "config": CONFIG, "snapshot": str(args.snapshot.resolve()),
                  "input_sha256": {name: digest(args.snapshot / name) for name in sorted(inventory & expected_names)},
                  "source_sha256": {str(p.relative_to(Path(__file__).parent.parent)): digest(p) for p in (
                      Path(__file__), Path(__file__).parent / "essaim/code.py",
                      Path(__file__).parent / "essaim/models.py",
                      Path(__file__).parent.parent / "tests/test_percees_code.py")},
                  "attempts": [{"type": "validation", "config": "PERC-20261010-v1-A1"}],
                  "technical_corrections": args.technical_correction, "test_contents_opened": False}
    with history_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"attempt": attempt, "event": "started", "provenance": provenance,
                                 "validate_only": args.validate_only}, ensure_ascii=False) + "\n")
    if not errors:
        for b in BENCHES:
            try:
                datasets[b] = load_bench(args.snapshot, b)
            except (IncompleteData, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{b}: {exc}")
    if not errors:
        # Identity of a model must be stable across the two DEV benchmark runs.
        for m in MODELS:
            a, b = (datasets[bench].validation["generation_manifests"][m] for bench in BENCHES)
            if any(a[k] != b[k] for k in ("model", "revision", "gguf", "weights", "engine", "backend")):
                errors.append(f"model identity differs across benches: {m}")
        a, b = (datasets[bench].validation["execution_manifest"] for bench in BENCHES)
        if any(a[k] != b[k] for k in ("tests", "extract", "extra_n", "isolation", "python", "numpy")):
            errors.append("execution settings differ across DEV benches")
    if errors:
        summary, journal, predictions = {"status": "blocked_incomplete", "errors": errors}, [], []
    elif args.validate_only:
        summary = {"status": "complete_validated_no_scores",
                   "validation": {b: d.validation for b, d in datasets.items()}}
        journal, predictions = [], []
    else:
        provenance["attempts"].append({"type": "numerical_replay", "config": "PERC-20261010-v1-A1"})
        summary, journal, predictions = analyze(datasets)
    for suffix, obj in (("summary.json", summary), ("manifest.json", provenance)):
        (args.output / f"percees_code_{suffix}").write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                                                           encoding="utf-8", newline="\n")
    for suffix, rows in (("registry.jsonl", journal), ("predictions.jsonl", predictions)):
        (args.output / f"percees_code_{suffix}").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    (args.output / "percees_code_report.md").write_text(report(summary), encoding="utf-8", newline="\n")
    outputs = {suffix: digest(args.output / f"percees_code_{suffix}") for suffix in (
        "summary.json", "manifest.json", "registry.jsonl", "predictions.jsonl", "report.md")}
    with history_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"attempt": attempt, "event": "finished", "status": summary["status"],
                                 "errors": errors, "output_sha256": outputs,
                                 "finished_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
    print(json.dumps({"status": summary["status"], "errors": errors, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
