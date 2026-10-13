"""DEV-only nested verification analysis and automatic freeze; the sole VERIF label reader."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import analyze_percees_code as historical
from verif1_protocol import (AUTHORS, VERSION, WEIGHTS, Ledger, exclusive, file_sha, sha, utc, write_new)
from verif1_run import validate_bundle

GRIDS = {
    "V03": [f"V03:{t}" for t in (1, 2)],
    "V04": [f"V04:{t}:{n}" for t in (1, 2) for n in (16, 64)],
    "V05": ["V05"],
    "V06": [f"V06:{t}:{n}:" + "".join(map(str, w))
            for t in (1, 2) for n in (16, 64) for w in WEIGHTS],
}
CONFIG = {"registration": VERSION, "grids": GRIDS, "outer": 5, "inner": 3,
          "fold_formula": historical.CONFIG["fold_formula"], "seed": 20261010, "bootstrap": 2000,
          "selection": "visible_then_loaded; public_score; train_K_tie; absent_historical_visible_K",
          "tie": "first registered configuration", "alpha_future": .05 / 28,
          "freeze": "full_DEV_three_inner_folds; no_outer_score_choice"}


def scientific_select(candidates, features, scores, weights):
    """Public vectors only. Learned weights are supplied by the training fold, never by this question."""
    if len(candidates) != len(scores):
        raise ValueError("score/candidate alignment differs")
    eligible = eligible_indices(candidates, features)
    present = [i for i in eligible if scores[i] is not None]
    if not present:
        return historical.select(candidates, features, weights, "visible")
    if any(type(scores[i]) not in (int, float) or not math.isfinite(scores[i])
           or not 0 <= scores[i] <= 1 for i in present):
        raise ValueError("invalid normalized public score")
    return candidates[max(present, key=lambda i: (scores[i], historical.rank(candidates[i], weights)))]


def eligible_indices(candidates, features):
    visible = [i for i, c in enumerate(candidates) if features[c].visible]
    loaded = [i for i, c in enumerate(candidates) if features[c].load]
    return visible or loaded or list(range(len(candidates)))


def ordered(data, qid, question):
    lookup = {(c.model, c.sample, c.prog): c for c in historical.pool(data, qid, historical.SWARM, 5)}
    candidates = [lookup[(c["model"], c["sample"], c["prog"])] for c in question["candidates"]]
    if len(set(candidates)) != 35 or len(lookup) != 35:
        raise ValueError("public candidates do not match labeled DEV snapshot")
    return candidates


def predictions(data, ids, questions, vectors, config, weights):
    return {q: scientific_select(ordered(data, q, questions[q]), data.features,
                                 vectors[q]["scores"][config], weights) for q in ids}


def choose(datasets, ids, questions, vectors, experiment, outer, journal):
    trials = {config: {b: {} for b in historical.BENCHES} for config in GRIDS[experiment]}
    for inner in range(3):
        fits, valid = {}, {}
        for bench, data in datasets.items():
            train = [q for q in ids[bench] if historical.fold(q, 3) != inner]
            valid[bench] = [q for q in ids[bench] if historical.fold(q, 3) == inner]
            historical.require(bool(valid[bench]), "empty inner validation fold")
            fits[bench] = historical.fit(data, train, historical.SWARM)
        for config in GRIDS[experiment]:
            scores = {}
            for bench, data in datasets.items():
                pred = predictions(data, valid[bench], questions[bench], vectors[bench], config,
                                   fits[bench]["weights"])
                trials[config][bench].update(pred)
                scores[bench] = historical.score(data, pred)
            journal.append({"type": "inner_trial", "experiment": experiment, "outer": outer,
                            "inner": inner, "configuration": config, "scores": scores,
                            "macro": statistics.mean(scores.values()), "fits": fits,
                            "validation_ids": valid})
    results = {c: {b: historical.score(d, trials[c][b]) for b, d in datasets.items()} for c in trials}
    winner = max(GRIDS[experiment], key=lambda c: (statistics.mean(results[c].values()),
                                                -GRIDS[experiment].index(c)))
    journal.append({"type": "decision", "experiment": experiment, "outer": outer,
                    "configuration": winner, "all_configurations": results})
    return winner


def diagnostic_statistics(rows):
    correct = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    accepted = lambda r: all(r["accept"])
    out = {"candidates": len(rows), "questions": len({r["id"] for r in rows}),
           "correct_candidates": len(correct), "wrong_candidates": len(wrong),
           "correct_questions": len({r["id"] for r in correct}),
           "wrong_questions": len({r["id"] for r in wrong}),
           "false_reject": statistics.mean(not accepted(r) for r in correct) if correct else None,
           "false_agreement": statistics.mean(accepted(r) for r in wrong) if wrong else None}
    if wrong and len(wrong[0]["accept"]) == 2:
        x = statistics.mean(r["accept"][0] for r in wrong)
        y = statistics.mean(r["accept"][1] for r in wrong)
        denominator = math.sqrt(x * (1 - x) * y * (1 - y))
        out.update(false_margins=[x, y], rho_FA=(out["false_agreement"] - x * y) / denominator
                   if denominator else None, correlation_population="same_pair_present_EvalPlus_wrong")
    return out


def diagnostic_bootstrap(rows):
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["id"]].append(row)
    groups = list(by_question.values())
    if not groups:
        return {"draws": 2000, "seed": 20261010, "false_reject_ci": None, "false_agreement_ci": None}
    # Resample sufficient statistics per question; never treat 35 candidates as independent questions.
    counts = []
    for group in groups:
        counts.append((sum(r["correct"] for r in group),
                       sum(r["correct"] and not all(r["accept"]) for r in group),
                       sum(not r["correct"] for r in group),
                       sum(not r["correct"] and all(r["accept"]) for r in group)))
    rng, draws = random.Random(20261010), [[], []]
    for _ in range(2000):
        total = [0, 0, 0, 0]
        for _ in counts:
            for i, n in enumerate(rng.choice(counts)):
                total[i] += n
        for j, (den, num) in enumerate(((total[0], total[1]), (total[2], total[3]))):
            if den:
                draws[j].append(num / den)
    def interval(values):
        values.sort()
        return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]] if values else None
    return {"draws": 2000, "seed": 20261010, "false_reject_ci": interval(draws[0]),
            "false_agreement_ci": interval(draws[1]), "unit": "question"}


def diagnostics(datasets, questions, vectors):
    output = {}
    groups = [(a,) for a in AUTHORS] + list(itertools.combinations(AUTHORS, 2))
    for bench, data in datasets.items():
        output[bench] = {}
        for threshold in ("1", "2"):
            records = {group: [] for group in groups}
            absent = {group: 0 for group in groups}
            absent_questions = {group: set() for group in groups}
            for qid in questions[bench]:
                candidates = ordered(data, qid, questions[bench][qid])
                outcomes = vectors[bench][qid]["diagnostic_acceptances"][threshold]
                for i, c in enumerate(candidates):
                    for group in groups:
                        accept = [outcomes[i][a] for a in group]
                        if None in accept:
                            absent[group] += 1
                            absent_questions[group].add(qid)
                            continue
                        records[group].append({"id": qid, "correct": data.labels[c], "accept": accept})
            output[bench][threshold] = {"+".join(g): {
                **diagnostic_statistics(rows), "absent_candidates": absent[g],
                "absent_questions": len(absent_questions[g]),
                "bootstrap": diagnostic_bootstrap(rows)} for g, rows in records.items()}
    return {"by_bench": output, "diagnostic_only": True,
            "limitations": "EvalPlus proxy; inputs may violate contract; not an additional selector"}


def selection_diagnostics(datasets, paired, records):
    output = {}
    references = ("visible_k5", "multi:Qwen/Qwen3.8-27B")
    for experiment in GRIDS:
        banks = {}
        for bench, data in datasets.items():
            oracle = statistics.mean(any(data.labels[c] for c in historical.pool(data, q, historical.SWARM, 5))
                                     for q in data.candidates)
            new = paired[experiment][bench]
            rows = [r for r in records if r["method"] == experiment and r["bench"] == bench]
            banks[bench] = {"questions": len(new), "oracle_fixed35": oracle,
                            "signal_absent_questions": sum(r["public_signal_absent"] for r in rows),
                            "versus": {}}
            for reference in references:
                base = paired[reference][bench]
                gain = statistics.mean(new) - statistics.mean(base)
                gap = oracle - statistics.mean(base)
                banks[bench]["versus"][reference] = {
                    "repairs": sum(x and not y for x, y in zip(new, base, strict=True)),
                    "breaks": sum(y and not x for x, y in zip(new, base, strict=True)),
                    "gain": gain, "oracle_gap_recovered": gain / gap if gap > 0 else None}
        macro = {}
        for reference in references:
            gain = statistics.mean(b["versus"][reference]["gain"] for b in banks.values())
            gap = statistics.mean(b["oracle_fixed35"] - statistics.mean(paired[reference][name])
                                  for name, b in banks.items())
            macro[reference] = {"gain": gain, "oracle_gap_recovered": gain / gap if gap > 0 else None}
        output[experiment] = {"by_bench": banks, "macro_versus": macro, "diagnostic_only": True}
    return output


def analyze(datasets, questions, vectors):
    hist_summary, journal, records = historical.analyze(datasets)
    choices = []
    for outer in range(5):
        train = {b: [q for q in d.candidates if historical.fold(q) != outer] for b, d in datasets.items()}
        test = {b: [q for q in d.candidates if historical.fold(q) == outer] for b, d in datasets.items()}
        fits = {b: historical.fit(d, train[b], historical.SWARM) for b, d in datasets.items()}
        for experiment in GRIDS:
            config = choose(datasets, train, questions, vectors, experiment, outer, journal)
            choices.append({"outer": outer, "experiment": experiment, "configuration": config, "fits": fits})
            for bench, data in datasets.items():
                pred = predictions(data, test[bench], questions[bench], vectors[bench], config, fits[bench]["weights"])
                for qid, c in pred.items():
                    candidates = ordered(data, qid, questions[bench][qid])
                    present = any(vectors[bench][qid]["scores"][config][i] is not None
                                  for i in eligible_indices(candidates, data.features))
                    records.append({"bench": bench, "id": qid, "outer": outer, "method": experiment,
                                    "configuration": config, "correct": data.labels[c],
                                    "public_signal_absent": not present,
                                    "selected": {"model": c.model, "sample": c.sample, "prog": c.prog}})
    results, paired = {}, {}
    for method in sorted({r["method"] for r in records}):
        rows = sorted((r for r in records if r["method"] == method), key=lambda r: (r["bench"], r["id"]))
        paired[method] = {b: [r["correct"] for r in rows if r["bench"] == b] for b in historical.BENCHES}
        scores = {b: statistics.mean(paired[method][b]) for b in historical.BENCHES}
        results[method] = {"by_bench": scores, "macro": statistics.mean(scores.values())}
    comparators = list(hist_summary["results"])
    comparisons = {e: {m: historical.bootstrap_difference(paired[e], paired[m]) for m in comparators}
                   for e in GRIDS}
    ids = {b: list(d.candidates) for b, d in datasets.items()}
    frozen = {}
    final_fits = {b: historical.fit(d, ids[b], historical.SWARM) for b, d in datasets.items()}
    for experiment in GRIDS:
        frozen[experiment] = {"configuration": choose(datasets, ids, questions, vectors, experiment,
                                                      "freeze_full_dev", journal), "fits": final_fits}
    summary = {"status": "complete_real_dev_exploratory", "config": CONFIG, "results": results,
               "comparisons": comparisons, "outer_choices": choices, "freeze_choices": frozen,
               "historical": hist_summary, "verifier_diagnostics": diagnostics(datasets, questions, vectors),
               "selection_diagnostics": selection_diagnostics(datasets, paired, records),
               "registry_records": len(journal), "scientific_scores_synthetic": False}
    return {"summary": summary, "registry": journal, "predictions": records}


def technical_check(prepared, run, scope):
    """Read-only public artifact review. Never load labels or rank scientific configurations."""
    bundle = json.loads((prepared / "verif1_dev_bundle.json").read_text())
    plan = json.loads((prepared / "verif1_dev_plan.json").read_text())
    validate_bundle(bundle, plan)
    manifest = json.loads((run / "verif1_manifest.json").read_text())
    if manifest["simulation"] or manifest["sandbox"].get("simulation") or manifest["plan"] != plan:
        raise ValueError("technical check requires matching real DEV artifacts")
    binding = sha({"plan": plan, "simulation": False, "sandbox": manifest["sandbox"]})
    if manifest["binding"] != binding:
        raise ValueError("manifest binding differs")
    ledger = Ledger(run / "verif1_ledger.jsonl", binding)
    event = ledger.done.get("final:" + scope)
    if not event or event["status"] != "ok":
        raise ValueError("scope has no completed final record")
    final = event["output"]
    for name, value in final.items():
        if json.loads((run / f"verif1_{scope}_{name}.json").read_text()) != value:
            raise ValueError("public final artifact differs from ledger")
    expected = bundle["questions"]
    if scope == "pilot":
        expected = [q for b in historical.BENCHES for q in sorted(
            (q for q in expected if q["contract"]["bench"] == b), key=lambda q: q["contract"]["id"])[:32]]
    ids = [q["contract"]["bench"] + ":" + q["contract"]["id"] for q in expected]
    coverage = final["coverage"]
    if (coverage["simulation"] or not coverage["complete_structural_coverage"]
            or coverage["scope"] != scope or coverage["questions"] != (64 if scope == "pilot" else 270)
            or coverage["question_ids"] != ids or set(final["scores"]) != set(ids)):
        raise ValueError("scope structural coverage differs")
    grid = {c for values in GRIDS.values() for c in values}
    if any(set(v["scores"]) != grid or any(len(s) != 35 for s in v["scores"].values())
           for v in final["scores"].values()):
        raise ValueError("public score-vector structural coverage differs")
    judge_statuses, judges, truncated = {}, 0, 0
    for key, entry in ledger.done.items():
        if ":judge:" not in key or key.rsplit(":judge:", 1)[0] not in final["scores"]:
            continue
        if entry["status"] != "ok":
            raise ValueError("judge operation did not complete")
        result = entry["output"]
        status = result["status"]
        judge_statuses[status] = judge_statuses.get(status, 0) + 1
        judges += 1
        truncated += bool(result["raw"].get("truncated"))
    if judges != coverage["judges_expected"]:
        raise ValueError("judge request structural coverage differs")
    return {"status": "technical_review_only", "scope": scope, "ledger_sha256": file_sha(ledger.path),
            "coverage": coverage, "costs": final["costs"], "labels_loaded": False,
            "judge_diagnostics": {"statuses": judge_statuses, "requests": judges,
                                  "expected": coverage["judges_expected"], "raw_truncated": truncated},
            "limit": "inspect format/truncation/coverage and infrastructure errors; no scientific selection"}


def public_inputs(prepared, run, snapshot):
    bundle_path, plan_path = prepared / "verif1_dev_bundle.json", prepared / "verif1_dev_plan.json"
    bundle, plan = json.loads(bundle_path.read_text()), json.loads(plan_path.read_text())
    validate_bundle(bundle, plan)
    for name, digest in bundle["snapshot"].items():
        if not name.endswith(("_dev.jsonl", "_dev.jsonl.meta.json")) or file_sha(snapshot / name) != digest:
            raise ValueError("snapshot DEV binding differs")
    manifest = json.loads((run / "verif1_manifest.json").read_text())
    if manifest["simulation"] or manifest["sandbox"].get("simulation") or manifest["plan"] != plan:
        raise ValueError("scientific analysis requires matching real DEV run, never simulation")
    binding = sha({"plan": plan, "simulation": False, "sandbox": manifest["sandbox"]})
    if manifest["binding"] != binding:
        raise ValueError("manifest binding differs")
    ledger = Ledger(run / "verif1_ledger.jsonl", binding)
    final = ledger.done["final:full"]["output"]
    for name, value in final.items():
        if json.loads((run / f"verif1_full_{name}.json").read_text()) != value:
            raise ValueError("public final artifact differs from ledger")
    questions = {b: {} for b in historical.BENCHES}
    vectors = {b: {} for b in historical.BENCHES}
    for q in bundle["questions"]:
        b, qid = q["contract"]["bench"], q["contract"]["id"]
        questions[b][qid] = q
        vectors[b][qid] = final["scores"][b + ":" + qid]
        if set(vectors[b][qid]["scores"]) != {c for grid in GRIDS.values() for c in grid}:
            raise ValueError("configuration grid coverage differs")
    coverage = final["coverage"]
    if coverage["questions"] != 270 or coverage["simulation"] or not coverage["complete_structural_coverage"]:
        raise ValueError("full real coverage required")
    bindings = {"code_sha256": sha(plan["sources"]), "plan_sha256": file_sha(plan_path),
                "data_sha256": file_sha(bundle_path)}
    return questions, vectors, bindings, file_sha(ledger.path), final["costs"]


def run_analysis(prepared, run, snapshot, output):
    questions, vectors, bindings, ledger_hash, costs = public_inputs(prepared, run, snapshot)
    input_binding = {"bindings": bindings, "run_ledger_sha256": ledger_hash, "config": CONFIG}
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output / ".analysis.lock"):
        ledger = Ledger(output / "verif1_analysis_ledger.jsonl", sha(input_binding))
        def action():
            started = time.monotonic()
            # Labels enter only here, after provenance validation and append-only reservation.
            datasets = {b: historical.load_bench(snapshot, b) for b in historical.BENCHES}
            if any(set(d.candidates) != set(questions[b]) for b, d in datasets.items()):
                raise ValueError("question coverage differs from labeled DEV snapshot")
            result = analyze(datasets, questions, vectors)
            result["summary"].update(bindings=bindings, run_ledger_sha256=ledger_hash,
                                     measured_costs=costs, created_utc=utc(),
                                     analysis_cpu_wall_seconds=time.monotonic() - started)
            return result
        result = ledger.operation("analyze_dev", input_binding, action)
        for name, value in result.items():
            path = output / f"verif1_{name}_dev.json"
            if path.exists():
                if json.loads(path.read_text()) != value:
                    raise ValueError("analysis artifact changed")
            else:
                write_new(path, value)
    return result["summary"]


def freeze(analysis, output):
    from verif1_prepare import source_hashes
    summary_path = analysis / "verif1_summary_dev.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "complete_real_dev_exploratory" or summary["config"] != CONFIG:
        raise ValueError("real registered DEV analysis required")
    if sha(source_hashes()) != summary["bindings"]["code_sha256"]:
        raise ValueError("current frozen sources differ from DEV analysis")
    binding = {"bindings": summary["bindings"], "run_ledger_sha256": summary["run_ledger_sha256"], "config": CONFIG}
    ledger = Ledger(analysis / "verif1_analysis_ledger.jsonl", sha(binding))
    if ledger.done["analyze_dev"]["output"]["summary"] != summary:
        raise ValueError("freeze summary differs from append-only analysis")
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output / ".freeze.lock"):
        journal = Ledger(output / "verif1_freeze_ledger.jsonl", file_sha(summary_path))
        for variant, chosen in summary["freeze_choices"].items():
            frozen = journal.operation(variant, chosen, lambda: {
                "split": "dev", "variant": variant, "configuration": chosen["configuration"],
                "weights_by_bench": {b: f["weights"] for b, f in chosen["fits"].items()},
                "bindings": summary["bindings"], "analysis_sha256": file_sha(summary_path),
                "selection": CONFIG["selection"], "utc": utc()})
            path = output / f"verif1_{variant}_frozen_dev.json"
            if path.exists():
                if json.loads(path.read_text()) != frozen:
                    raise ValueError("frozen artifact changed")
            else:
                write_new(path, frozen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check_parser = commands.add_parser("check")
    for name in ("prepared", "run"):
        check_parser.add_argument("--" + name, type=Path, required=True)
    check_parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    analyze_parser = commands.add_parser("analyze")
    for name in ("prepared", "run", "snapshot", "output"):
        analyze_parser.add_argument("--" + name, type=Path, required=True)
    freeze_parser = commands.add_parser("freeze")
    for name in ("analysis", "output"):
        freeze_parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.action == "check":
        print(json.dumps(technical_check(args.prepared, args.run, args.scope), ensure_ascii=False))
    elif args.action == "freeze":
        freeze(args.analysis, args.output)
    else:
        run_analysis(args.prepared, args.run, args.snapshot, args.output)


if __name__ == "__main__":
    main()
