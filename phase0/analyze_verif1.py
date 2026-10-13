"""VERIF-1 CPU replay of the pinned DEV snapshot; no candidate execution or inference."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import analyze_percees_code as historical


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path("C:/tmp/percees/e11-dev")
FAMILIES = {
    **{m: "Qwen" for m in historical.MODELS if m.startswith("Qwen/")},
    **{m: "Gemma" for m in historical.MODELS if m.startswith("google/")},
    **{m: "Mistral" for m in historical.MODELS if m.startswith("mistralai/")},
    "ibm-granite/granite-4.2-3b": "Granite",
    "HuggingFaceTB/SmolLM3-3B": "SmolLM",
    "microsoft/Phi-4-mini-instruct": "Phi",
    "allenai/OLMo-2-0425-1B-Instruct": "OLMo",
}
GRID = tuple(itertools.product((0, 0.5, 1, 2), (10, 30)))
CONTROLS = ("visible_k5", "qwen27_multi", "qwen4_multi", "best_small_greedy")
REGISTRATION_HISTORY = (
    {"id": "initial", "parent_commit": "2d396eb", "campaign_variants": 6,
     "confirmation_alpha_per_bench_comparator": 0.05 / 24, "test_openings": 0},
    {"id": "A3", "parent_commit": "cb07e83cc122e9a9e766e06fe1aa84a91cd93456",
     "campaign_variants": 7, "confirmation_alpha_per_bench_comparator": 0.05 / 28,
     "test_openings": 0, "change": "V07 added in another worktree; V01/V02 rules unchanged"},
)
CONFIG = {
    "registration": "VERIF-1-20261010", "parent_commit": "2d396eb",
    "families": FAMILIES, "absent_catalogue_family": "Llama",
    "pool": list(historical.SWARM), "samples": 5, "candidates": 35,
    "V01": "family count, present-member family max-weight sum, representative K rank",
    "V02_grid": GRID, "V02_tie": "score, family count, representative K rank",
    "collision": "question-level wrong eligible signature-set intersection given both nonempty",
    "collision_prior": "pooled training pair counts; 0.5 if no observations",
    "signature": "nonempty, all entries nonempty and not !, extra status ok",
    "outer_folds": 5, "inner_folds": 3, "fold": historical.CONFIG["fold_formula"],
    "seed": 20261010, "bootstrap_draws": 2000, "test_openings": 0,
    "campaign_variants": 7, "confirmation_alpha_per_bench_comparator": 0.05 / 28,
    "registration_history": REGISTRATION_HISTORY,
    "active_amendment": REGISTRATION_HISTORY[-1],
    "confirmations_by_variant": {f"V{i:02d}": 0 for i in range(1, 8)},
    "scope": "exploratory DEV only; no p-values; V03-V06 inactive without alpha redistribution",
}


@dataclass(frozen=True)
class PublicData:
    """Selection can only access stored public candidate features, never labels."""
    candidates: dict[str, list[historical.Candidate]]
    features: dict[historical.Candidate, historical.PublicFeatures]


def usable(feature: historical.PublicFeatures) -> bool:
    return feature.load and bool(feature.signature) and all(s and s != "!" for s in feature.signature)


def eligible(candidates, features):
    passing = [c for c in candidates if features[c].visible]
    return passing or [c for c in candidates if features[c].load]


def group_families(group, weights):
    result = {}
    for c in group:
        family = FAMILIES[c.model]
        result[family] = max(result.get(family, 0.0), weights.get(c.model, 0.0))
    return result


def pair_key(a, b):
    return "|".join(sorted((a, b)))


def select(candidates, features, weights, collisions, variant="V01"):
    """Pure public interface; collisions and K weights are already fitted on training IDs."""
    historical.require(bool(candidates), "empty selection pool")
    groups = defaultdict(list)
    for c in eligible(candidates, features):
        if usable(features[c]):
            groups[features[c].signature].append(c)
    if not groups:
        return historical.select(candidates, features, weights, "visible"), {
            "fallback": True, "groups": 0, "positive_agreement": False, "family_count": 0}
    pick = lambda g: max(g, key=lambda c: historical.rank(c, weights))

    def ranking(group):
        families = group_families(group, weights)
        weight = sum(families.values())
        if variant == "V01":
            return len(families), weight, historical.rank(pick(group), weights)
        lam, _ = variant
        penalty = sum(collisions[pair_key(a, b)] for a, b in itertools.combinations(sorted(families), 2))
        return weight - lam * penalty, len(families), historical.rank(pick(group), weights)

    winner = max(groups.values(), key=ranking)
    n = len(group_families(winner, weights))
    return pick(winner), {"fallback": False, "groups": len(groups), "family_count": n,
                          "positive_agreement": n >= 2, "selected_group_score": ranking(winner)[0]}


def fit_collisions(public: PublicData, labels, ids, tau):
    """One bounded Bernoulli observation per training question and unordered lineage pair."""
    historical.require(bool(ids), "empty collision training fold")
    families = sorted({FAMILIES[m] for m in historical.SWARM})
    counts = {pair_key(a, b): [0, 0] for a, b in itertools.combinations(families, 2)}
    for q in ids:
        signatures = defaultdict(set)
        candidates = [c for c in public.candidates[q] if c.model in historical.SWARM and c.sample < 5]
        for c in eligible(candidates, public.features):
            if not labels[c] and usable(public.features[c]):
                signatures[FAMILIES[c.model]].add(public.features[c].signature)
        for a, b in itertools.combinations(families, 2):
            if signatures[a] and signatures[b]:
                counts[pair_key(a, b)][1] += 1
                counts[pair_key(a, b)][0] += bool(signatures[a] & signatures[b])
    numerator = sum(n for n, _ in counts.values())
    denominator = sum(d for _, d in counts.values())
    mean = numerator / denominator if denominator else 0.5
    values = {key: (n + tau * mean) / (d + tau) for key, (n, d) in counts.items()}
    return {"values": values, "counts": counts, "pooled_mean": mean, "tau": tau, "train_ids": list(ids)}


def predict(public, ids, weights, collision_fit, variant):
    predictions, signals = {}, {}
    for q in ids:
        cs = [c for c in public.candidates[q] if c.model in historical.SWARM and c.sample < 5]
        predictions[q], signals[q] = select(cs, public.features, weights, collision_fit["values"], variant)
    return predictions, signals


def choose_v02(datasets, public, train, outer, journal):
    combined = {v: {b: {} for b in historical.BENCHES} for v in GRID}
    for inner in range(3):
        fits, validation = {}, {}
        for b, d in datasets.items():
            ids = [q for q in train[b] if historical.fold(q, 3) != inner]
            validation[b] = [q for q in train[b] if historical.fold(q, 3) == inner]
            historical.require(bool(validation[b]), "empty inner validation fold")
            fits[b] = {"K": historical.fit(d, ids, historical.SWARM),
                       "collisions": {tau: fit_collisions(public[b], d.labels, ids, tau) for tau in (10, 30)}}
        for v in GRID:
            scores = {}
            for b, d in datasets.items():
                preds, _ = predict(public[b], validation[b], fits[b]["K"]["weights"],
                                   fits[b]["collisions"][v[1]], v)
                combined[v][b].update(preds)
                scores[b] = historical.score(d, preds)
            journal.append({"type": "inner_trial", "experiment": "V02", "outer": outer,
                            "inner": inner, "variant": v, "scores": scores, "fits": fits,
                            "validation_ids": validation})
    internal = [{"variant": v, "scores": {b: historical.score(d, combined[v][b])
                                          for b, d in datasets.items()}} for v in GRID]
    best = max(internal, key=lambda row: (statistics.mean(row["scores"].values()), -GRID.index(row["variant"])))
    journal.append({"type": "decision", "experiment": "V02", "outer": outer,
                    "selected": best["variant"], "internal_scores": internal})
    return best["variant"]


def bootstrap(a, b):
    rng = random.Random(CONFIG["seed"])
    delta = {bench: [int(x) - int(y) for x, y in zip(a[bench], b[bench], strict=True)]
             for bench in historical.BENCHES}
    draws = {bench: [] for bench in (*historical.BENCHES, "macro")}
    for _ in range(CONFIG["bootstrap_draws"]):
        sampled = {bench: 100 * sum(rng.choice(ds) for _ in ds) / len(ds) for bench, ds in delta.items()}
        for bench, value in sampled.items():
            draws[bench].append(value)
        draws["macro"].append(statistics.mean(sampled.values()))
    observed = {b: 100 * statistics.mean(ds) for b, ds in delta.items()}
    observed["macro"] = statistics.mean(observed.values())
    return {b: {"difference_pp": observed[b], "ci95_descriptive_pp": [sorted(ds)[49], sorted(ds)[1949]],
                "draws": 2000, "seed": CONFIG["seed"]} for b, ds in draws.items()}


def error_pair(a, b):
    pa, pb = statistics.mean(a), statistics.mean(b)
    joint = statistics.mean(x and y for x, y in zip(a, b, strict=True))
    variance = pa * (1 - pa) * pb * (1 - pb)
    return {"n": len(a), "error_a": pa, "error_b": pb, "joint_error": joint,
            "error_correlation": (joint - pa * pb) / math.sqrt(variance) if variance else None}


def diagnostics(data, models):
    features = getattr(data, "public_extra_features", data.features)
    ids = sorted(data.candidates)
    errors = {k: {m: [not any(data.labels[c] for c in historical.pool(data, q, (m,), k))
                     for q in ids] for m in models} for k in (1, 5)}
    pairs, aggregate = [], {scope: [0, 0] for scope in ("intra", "inter")}
    for a, b in itertools.combinations_with_replacement(models, 2):
        scope = "intra" if FAMILIES[a] == FAMILIES[b] else "inter"
        numerator = denominator = 0
        for q in ids:
            wrong = {m: [c for c in historical.pool(data, q, (m,), 5) if not data.labels[c]
                         and features[c].visible and usable(features[c])] for m in (a, b)}
            candidates = itertools.combinations(wrong[a], 2) if a == b else itertools.product(wrong[a], wrong[b])
            seen = set()
            for x, y in candidates:
                key = tuple(sorted((x.prog, y.prog)))
                if x.prog == y.prog or key in seen:
                    continue
                seen.add(key)
                denominator += 1
                numerator += features[x].signature == features[y].signature
        pairs.append({"models": [a, b], "families": [FAMILIES[a], FAMILIES[b]], "scope": scope,
                      "greedy_errors": error_pair(errors[1][a], errors[1][b]),
                      "five_draw_oracle_errors": error_pair(errors[5][a], errors[5][b]),
                      "wrong_same_signature": {"numerator": numerator, "denominator": denominator,
                                               "conditional_rate": numerator / denominator if denominator else None}})
    # Aggregate counts deduplicate again across model aliases sharing the same lineage scope.
    for q in ids:
        wrong = [c for c in historical.pool(data, q, models, 5) if not data.labels[c]
                 and features[c].visible and usable(features[c])]
        seen = set()
        for x, y in itertools.combinations(wrong, 2):
            scope = "intra" if FAMILIES[x.model] == FAMILIES[y.model] else "inter"
            key = (scope, *sorted((x.prog, y.prog)))
            if x.prog == y.prog or key in seen:
                continue
            seen.add(key)
            aggregate[scope][1] += 1
            aggregate[scope][0] += features[x].signature == features[y].signature
    return {"pairs": pairs, "wrong_same_signature": {
        scope: {"numerator": n, "denominator": d, "conditional_rate": n / d if d else None}
        for scope, (n, d) in aggregate.items()}, "diagnostic_only": True}


def analyze(datasets, execution_ms, journal=None, records=None):
    public = {b: PublicData(d.candidates, getattr(d, "public_extra_features", d.features)) for b, d in datasets.items()}
    journal = [] if journal is None else journal
    records = [] if records is None else records
    choices = []
    for outer in range(5):
        train = {b: [q for q in d.candidates if historical.fold(q) != outer] for b, d in datasets.items()}
        evaluate = {b: [q for q in d.candidates if historical.fold(q) == outer] for b, d in datasets.items()}
        ks = {b: historical.fit(d, train[b], historical.SWARM) for b, d in datasets.items()}
        collisions = {b: {tau: fit_collisions(public[b], d.labels, train[b], tau) for tau in (10, 30)}
                      for b, d in datasets.items()}
        best_solo = max(historical.SWARM, key=lambda m: (
            statistics.mean(ks[b]["greedy_accuracy"][m] for b in historical.BENCHES), -historical.SWARM.index(m)))
        v02 = choose_v02(datasets, public, train, outer, journal)
        solo_models = {"qwen27_multi": "Qwen/Qwen3.8-27B", "qwen4_multi": "Qwen/Qwen3.5-4B"}
        solo_variants = {name: historical.choose_variant(datasets, train, (m,), outer, name, journal)
                         for name, m in solo_models.items()}
        choices.append({"outer": outer, "V02": v02, "best_small_solo": best_solo,
                        "solo_variants": solo_variants, "K_fits": ks, "collision_fits": collisions,
                        "train_ids": train, "evaluation_ids": evaluate})
        for b, d in datasets.items():
            methods = {"V01": "V01", "V02": v02, **{f"V02_l{lam}_t{tau}": (lam, tau) for lam, tau in GRID}}
            for name, variant in methods.items():
                tau = 10 if variant == "V01" else variant[1]
                preds, signals = predict(public[b], evaluate[b], ks[b]["weights"], collisions[b][tau], variant)
                for q, c in preds.items():
                    cs = historical.pool(d, q, historical.SWARM, 5)
                    records.append(record(d, q, c, outer, name, variant, cs, "visible", signals[q], execution_ms[b]))
            controls = {"visible_k5": (historical.SWARM, (5, "visible"), ks[b]["weights"]),
                        "best_small_greedy": ((best_solo,), (1, "visible"), {best_solo: 1.0})}
            for name, m in solo_models.items():
                controls[name] = ((m,), solo_variants[name], historical.fit(d, train[b], (m,))["weights"])
            for name, (models, variant, weights) in controls.items():
                for q, c in historical.predict(d, evaluate[b], models, variant, weights).items():
                    records.append(record(d, q, c, outer, name, variant, historical.pool(d, q, models, variant[0]),
                                          variant[1], {}, execution_ms[b]))
    methods = sorted({r["method"] for r in records})
    results, paired = {}, {}
    oracle = {b: [any(d.labels[c] for c in historical.pool(d, q, historical.SWARM, 5))
                  for q in sorted(d.candidates)] for b, d in datasets.items()}
    oracle_scores = {b: statistics.mean(values) for b, values in oracle.items()}
    oracle_scores["macro"] = statistics.mean(oracle_scores.values())
    for name in methods:
        rows = [r for r in records if r["method"] == name]
        paired[name] = {b: [r["correct"] for r in sorted(rows, key=lambda row: row["id"]) if r["bench"] == b]
                        for b in historical.BENCHES}
        scores = {b: statistics.mean(values) for b, values in paired[name].items()}
        scores["macro"] = statistics.mean(scores.values())
        costs = {b: {k: statistics.mean(r["cost"][k] for r in rows if r["bench"] == b)
                     for k in rows[0]["cost"]} for b in historical.BENCHES}
        costs["macro"] = {k: statistics.mean(costs[b][k] for b in historical.BENCHES) for k in rows[0]["cost"]}
        coverage = {b: {key: sum(r["signal"].get(key, False) for r in rows if r["bench"] == b)
                        for key in ("fallback", "positive_agreement")} for b in historical.BENCHES}
        results[name] = {"scores": scores, "cost_per_question": costs, "signal_counts": coverage}
    comparisons = {}
    for name in methods:
        if not name.startswith("V0"):
            continue
        comparisons[name] = {}
        for control in CONTROLS:
            comparison = bootstrap(paired[name], paired[control])
            for b in historical.BENCHES:
                aa, bb = paired[name][b], paired[control][b]
                comparison[b].update(repairs=sum(a and not c for a, c in zip(aa, bb, strict=True)),
                                     breaks=sum(not a and c for a, c in zip(aa, bb, strict=True)))
            comparisons[name][control] = comparison
        results[name]["gap_fraction_recovered"] = {
            b: ((results[name]["scores"][b] - results["visible_k5"]["scores"][b]) /
                (oracle_scores[b] - results["visible_k5"]["scores"][b]))
            if oracle_scores[b] != results["visible_k5"]["scores"][b] else None
            for b in (*historical.BENCHES, "macro")}
    summary = {"status": "complete_dev_exploratory", "config": CONFIG, "results": results,
               "comparisons": comparisons, "oracle35": oracle_scores, "choices": choices,
               "validation": {b: d.validation for b, d in datasets.items()},
               "diagnostics": {b: {"fixed35": diagnostics(d, historical.SWARM),
                                    "catalogue13": diagnostics(d, historical.MODELS)} for b, d in datasets.items()},
               "multiplicity": {"campaign_variants": CONFIG["campaign_variants"],
                                "confirmation_alpha_per_bench_comparator": CONFIG["confirmation_alpha_per_bench_comparator"],
                                "confirmations_by_variant": CONFIG["confirmations_by_variant"],
                                "active_amendment": CONFIG["active_amendment"],
                                "V01_configurations": 1, "V02_configurations": 8,
                                "control_configurations_per_search": 15, "outer_folds": 5, "inner_folds": 3,
                                "inner_trials": sum(r["type"] == "inner_trial" for r in journal),
                                "decisions": sum(r["type"] == "decision" for r in journal),
                                "external_methods": len(methods), "test_openings": 0}}
    return summary, journal, records


def record(data, q, candidate, outer, method, variant, cs, rule, signal, execution_ms):
    cost = historical.costs(cs, rule)
    cost.update(authors=0, judges=0, new_inference_calls=0,
                historical_generation_ms=sum(getattr(data, "generation_ms", {}).get((q, c.model, c.sample), 0) for c in cs),
                historical_execution_ms=sum(execution_ms.get((q, prog), 0) for prog in {c.prog for c in cs})
                if rule != "text" else 0)
    return {"bench": data.bench, "id": q, "outer": outer, "method": method, "variant": variant,
            "selected": {"model": candidate.model, "sample": candidate.sample, "prog": candidate.prog},
            "correct": data.labels[candidate], "cost": cost, "signal": signal}


def report(summary):
    lines = ["# VERIF-1 : rejeu CPU exclusivement dev", "", f"Statut : {summary['status']}.", "",
             "Test ouvert : **0** ; nouvelle inférence, auteurs et juges : **0**. "
             "Signatures E11 existantes issues de mutations aléatoires publiques, aucun nouveau test généré.", ""]
    lines += ["Métadonnées amendées par A3 (`cb07e83`) : sept variantes V01–V07, "
              "alpha confirmatoire futur 0,05/28, confirmations zéro pour chacune. "
              "L'inscription initiale à six variantes est conservée dans l'historique et les anciennes tentatives. "
              "V07 relève d'un autre worktree ; règles, grilles et comparateurs V01/V02 inchangés.", ""]
    if summary["status"] != "complete_dev_exploratory":
        return "\n".join(lines + [json.dumps(summary.get("errors", []), ensure_ascii=False), ""])
    lines += ["| Méthode | HumanEval+ % | MBPP+ % | Macro % | Candidats | Préfill | Sortie | B×tokens |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("V01", "V02", *CONTROLS, *(f"V02_l{lam}_t{tau}" for lam, tau in GRID)):
        r = summary["results"][name]
        s, c = r["scores"], r["cost_per_question"]["macro"]
        lines.append(f"| {name} | {100*s['humanevalplus']:.2f} | {100*s['mbppplus']:.2f} | {100*s['macro']:.2f} | "
                     f"{c['candidates']:.2f} | {c['prompt_tokens']:.0f} | {c['output_tokens']:.0f} | {c['params_b_times_tokens']:.0f} |")
    lines += ["", "Les huit configurations externes V02 sont exhaustives et descriptives ; "
              "seul le choix interne définit V02. Aucun choix sur leurs scores externes.", "",
              "| Méthode − contrôle | Banc | Différence pp | IC95 descriptif pp | Réparations | Cassures |",
              "|---|---|---:|---|---:|---:|"]
    for name in ("V01", "V02"):
        for control, rows in summary["comparisons"][name].items():
            for b, r in rows.items():
                lo, hi = r["ci95_descriptive_pp"]
                lines.append(f"| {name} − {control} | {b} | {r['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                             f"{r.get('repairs', '—')} | {r.get('breaks', '—')} |")
    lines += ["", "| Banc | Oracle fixe35 % | Fraction du gap V01 | Fraction du gap V02 |",
              "|---|---:|---:|---:|"]
    for b, score in summary["oracle35"].items():
        values = [summary["results"][v]["gap_fraction_recovered"][b] for v in ("V01", "V02")]
        formatted = ["indéfinie" if x is None else f"{100*x:.2f} %" for x in values]
        lines.append(f"| {b} | {100*score:.2f} | {formatted[0]} | {formatted[1]} |")
    lines += ["", "| Pli | V02 interne | Qwen27 interne | Qwen4 interne | Meilleur solo train |",
              "|---|---|---|---|---|"]
    for c in summary["choices"]:
        lines.append(f"| {c['outer']} | {c['V02']} | {c['solo_variants']['qwen27_multi']} | "
                     f"{c['solo_variants']['qwen4_multi']} | {c['best_small_solo']} |")
    lines += ["", "| Banc/pool | Faux distincts intra : accords/paires | Inter : accords/paires |",
              "|---|---:|---:|"]
    for b, scopes in summary["diagnostics"].items():
        for scope, d in scopes.items():
            counts = d["wrong_same_signature"]
            lines.append(f"| {b}/{scope} | {counts['intra']['numerator']}/{counts['intra']['denominator']} | "
                         f"{counts['inter']['numerator']}/{counts['inter']['denominator']} |")
    lines += ["", "Les erreurs marginales/jointes et corrélations de correction par paire sont "
              "dans le résumé JSON ; une variance nulle donne une corrélation indéfinie. "
              "La collision conditionnelle est mesurée parmi les programmes faux, visibles, distincts "
              "et signatures utilisables ; elle ne mesure pas la corrélation de correction. "
              "Les paires de hashes sont dédupliquées par question, les coûts comptent tous les appels.", "",
              "Couverture des signaux (accord positif = au moins deux vraies familles ; "
              "absence d'accord inclut les groupes à une seule famille) :"]
    for name in ("V01", "V02"):
        for b, counts in summary["results"][name]["signal_counts"].items():
            n = summary["validation"][b]["n_scored"]
            lines.append(f"- {name}/{b} : {counts['positive_agreement']}/{n} accords positifs, "
                         f"{n-counts['positive_agreement']} sans accord, {counts['fallback']} replis sans groupe utilisable.")
    lines += ["", "Multiplicité exhaustive : " + json.dumps(summary["multiplicity"], ensure_ascii=False) + ".", "",
              "IC bootstrap appariés descriptifs, 2 000 tirages, graine 20261010 ; "
              "apprentissages recouvrants non entièrement capturés. Aucun p-value dev, aucune percée confirmée. "
              "Oracle et diagnostics lisent les labels uniquement après sélection, sans les fournir aux sélecteurs.", "",
              "Coûts CPU/temps mur du rejeu dans le manifeste ; compteurs historiques d'exécution "
              "incluent le harnais entier et les tests cachés, ils ne mesurent pas le temps propre du vérificateur. "
              "Tokens et B×tokens sont des proxies sans prix, énergie ou accélération inférée. "
              "Limite de provenance historique : pas d'empreinte indépendante du split officiel ni "
              "des fichiers génération dans le manifeste d'exécution. Confirmation future sur un lot disjoint.", ""]
    return "\n".join(lines)


def load_snapshot(snapshot):
    historical.require(snapshot.resolve() == SNAPSHOT.resolve(), "only pinned DEV snapshot is authorized")
    pinned = json.loads((ROOT / "phase0/results/percees_code_snapshot_copy.json").read_text(encoding="utf-8"))
    hashes = pinned["source_and_snapshot_sha256"]
    inventory = {p.name for p in snapshot.iterdir() if p.is_file()}
    historical.require(inventory == set(hashes), "pinned snapshot inventory mismatch")
    actual = {name: historical.digest(snapshot / name) for name in sorted(hashes)}
    historical.require(actual == hashes, "pinned snapshot SHA mismatch")
    datasets = {b: historical.load_bench(snapshot, b) for b in historical.BENCHES}
    for m in historical.MODELS:
        a, b = [datasets[bench].validation["generation_manifests"][m] for bench in historical.BENCHES]
        historical.require(all(a[k] == b[k] for k in ("model", "revision", "gguf", "weights", "engine", "backend")),
                           f"cross-benchmark model identity mismatch: {m}")
    a, b = [datasets[bench].validation["execution_manifest"] for bench in historical.BENCHES]
    historical.require(all(a[k] == b[k] for k in ("tests", "extract", "extra_n", "isolation", "python", "numpy")),
                       "cross-benchmark execution settings mismatch")
    execution_ms, generation_ms = {}, {}
    for bench, d in datasets.items():
        rows = historical.read_jsonl(snapshot / f"codeexec_e11_colab_{bench}_dev.jsonl", ("id", "prog"))
        statuses = {(r["id"], r["prog"]): r["status"]["extra"] for r in rows}
        execution_ms[bench] = {(r["id"], r["prog"]): r["ms"] for r in rows}
        # BenchData keeps labels separate. Only PUBLIC extra status sanitizes the feature interface.
        d.public_extra_features = dict(d.features)
        for c, f in d.features.items():
            if statuses[c.qid, c.prog] != "ok" or not usable(f):
                d.public_extra_features[c] = replace(f, signature=())
        generation_ms[bench], d.generation_ms = {}, {}
        for m in historical.MODELS:
            rows = historical.read_jsonl(snapshot / f"code_{m.replace('/', '__')}_colab_{bench}_dev.jsonl", ("id", "sample"))
            generation_ms[bench][m] = sum(r["ms"] for r in rows)
            d.generation_ms.update({(r["id"], m, r["sample"]): r["ms"] for r in rows})
    return datasets, execution_ms, actual, generation_ms


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=ROOT / "phase0/results")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--technical-correction", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    history_path = args.output / "verif1_attempts_dev.jsonl"
    history = historical.read_jsonl(history_path, ("attempt", "event")) if history_path.exists() else []
    attempt = max((r["attempt"] for r in history), default=0) + 1
    suffixes = ("summary_dev.json", "manifest_dev.json", "registry_dev.jsonl", "predictions_dev.jsonl", "report_dev.md")
    for suffix in suffixes:
        previous = args.output / f"verif1_{suffix}"
        if previous.exists():
            archived = args.output / f"verif1_attempt{attempt-1:04d}_{suffix}"
            historical.require(not archived.exists(), "attempt archive already exists")
            archived.write_bytes(previous.read_bytes())
    def event(name, **kwargs):
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"attempt": attempt, "event": name, "utc": datetime.now(timezone.utc).isoformat(),
                                     **kwargs}, ensure_ascii=False, allow_nan=False) + "\n")
    started, cpu = time.perf_counter(), time.process_time()
    provenance = {"attempt": attempt, "started_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
                  "config_sha256": hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest(),
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "snapshot": str(args.snapshot.resolve()), "technical_corrections": args.technical_correction,
                  "source_sha256": {str(p.relative_to(ROOT)): historical.digest(p) for p in (
                      Path(__file__), ROOT / "phase0/analyze_percees_code.py", ROOT / "phase0/essaim/code.py",
                      ROOT / "phase0/essaim/models.py", ROOT / "tests/test_verif1_replay.py",
                      ROOT / "docs/12_percees_v2_replay.md", ROOT / "docs/12_percees.md")},
                  "test_openings": 0, "new_inference_calls": 0}
    event("started", provenance=provenance, validate_only=args.validate_only)
    journal, records = [], []
    try:
        datasets, ms, hashes, generation_ms = load_snapshot(args.snapshot)
        provenance.update(input_sha256=hashes, historical_generation_ms=generation_ms)
        if args.validate_only:
            summary = {"status": "complete_validated_no_scores", "validation": {b: d.validation for b, d in datasets.items()}}
        else:
            summary, journal, records = analyze(datasets, ms, journal, records)
    except Exception as exc:
        summary = {"status": "failed_preserved", "errors": [f"{type(exc).__name__}: {exc}"]}
        event("failure", errors=summary["errors"])
    provenance.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "verif1_summary_dev.json", summary)
    write_json(args.output / "verif1_manifest_dev.json", provenance)
    for suffix, rows in (("registry_dev.jsonl", journal), ("predictions_dev.jsonl", records)):
        if suffix == "registry_dev.jsonl":
            rows = [{**row, "registration_metadata": CONFIG["active_amendment"]} for row in rows]
        (args.output / f"verif1_{suffix}").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                                                            for r in rows), encoding="utf-8", newline="\n")
    (args.output / "verif1_report_dev.md").write_text(report(summary), encoding="utf-8", newline="\n")
    event("finished", status=summary["status"], output_sha256={s: historical.digest(args.output / f"verif1_{s}") for s in suffixes})
    print(json.dumps({"status": summary["status"], "attempt": attempt, "wall_seconds": provenance["wall_seconds"],
                      "errors": summary.get("errors", [])}))
    if summary["status"] == "failed_preserved":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
