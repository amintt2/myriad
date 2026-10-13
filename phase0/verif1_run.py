"""Runnable VERIF-1 DEV stages: authors, Linux matrix, finite properties, blinded judges and combinations."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path

from verif1_protocol import (AUTHOR_DECODER, AUTHOR_INSTRUCTION, AUTHORS, JUDGE_DECODER,
                             JUDGE_INSTRUCTION, VERSION, WEIGHTS, Ledger,
                             author_acceptances, canonical, combine, cost, differential_scores,
                             exclusive, file_sha, mean_present,
                             parse_judgment, parse_proposal, select, sha, shortlist, supported_scores, utc, write_new)


def validate_bundle(bundle, plan):
    from verif1_prepare import source_hashes
    if (bundle["split"] != "dev" or plan["split"] != "dev" or bundle["version"] != VERSION
            or plan["version"] != VERSION or sha(bundle) != plan["bundle_sha256"]
            or sha(bundle["catalogue"]) != plan["catalogue_sha256"] or source_hashes() != plan["sources"]):
        raise ValueError("DEV bundle/plan/source provenance changed")
    if plan["prompts"] != {"author": AUTHOR_INSTRUCTION, "judge": JUDGE_INSTRUCTION}:
        raise ValueError("prompts differ from frozen protocol")
    fixed = {"seed": 20261010, "author_calls": 2, "author_tokens": 512, "judge_tokens": 256,
             "support_thresholds": [1, 2], "mutations": [16, 64], "shortlist": 3,
             "context": 8192, "parallel": 1, "authors": list(AUTHORS),
             "author_decoder": AUTHOR_DECODER, "judge_decoder": JUDGE_DECODER,
             "seeds": [20261010, 20261011], "shortlist_order": "family_count_then_catalogue_sample_no_K"}
    if any(plan["config"].get(k) != v for k, v in fixed.items()):
        raise ValueError("plan disagrees with implemented frozen limits")
    from verif1_protocol import MODELS, FAMILIES
    ids = set()
    for q in bundle["questions"]:
        contract = q["contract"]
        digest = contract["sha256"]
        if sha({k: v for k, v in contract.items() if k != "sha256"}) != digest:
            raise ValueError("contract hash differs")
        if set(contract) != {"id", "bench", "text", "entry", "header", "public_inputs", "public_only", "sha256"}:
            raise ValueError("contract is not public projection")
        identity = (contract["bench"], contract["id"])
        if identity in ids or not contract["public_only"]:
            raise ValueError("duplicate or private contract")
        ids.add(identity)
        if (len(q["candidates"]) != 35 or {(c["model"], c["sample"]) for c in q["candidates"]}
                != {(m, s) for m in MODELS for s in range(5)}):
            raise ValueError("candidate coverage differs")
        for c in q["candidates"]:
            if (set(c) != {"model", "family", "sample", "code", "code_sha256", "prog", "cost", "public"}
                    or c["family"] != FAMILIES[c["model"]]
                    or hashlib.sha256(c["code"].encode()).hexdigest() != c["code_sha256"]
                    or set(c["public"]) != {"load", "visible", "visible_cases", "visible_errors", "extra", "status"}):
                raise ValueError("candidate public projection/provenance differs")


def proposed(ledger, key, family, prompt, ceiling, seed, engine, parser):
    decoder = AUTHOR_DECODER if parser is parse_proposal else JUDGE_DECODER
    def action():
        raw = engine.generate(family, prompt, ceiling, seed, decoder=decoder)
        if (type(raw.get("prompt_tokens")) is not int or raw["prompt_tokens"] < 0
                or type(raw.get("completion_tokens")) is not int or not 0 <= raw["completion_tokens"] <= ceiling):
            return {"raw": raw, "status": "usage_error", "parsed": None, "prompt_sha256": sha(prompt)}
        result = {"raw": raw, "status": "valid", "parsed": None, "prompt_sha256": sha(prompt)}
        if raw["prompt_tokens"] + ceiling > 8192:
            result["status"] = "context_overflow"
        elif raw["truncated"] or raw["finish"] != "stop":
            result["status"] = "truncated"
        else:
            try:
                result["parsed"] = parser(raw["text"])
            except (ValueError, TypeError, KeyError) as error:
                result.update(status="format_error", error=type(error).__name__, error_message=str(error))
        return result
    return ledger.operation(key, {"family": family, "prompt": prompt, "max_tokens": ceiling,
                                 "seed": seed, "decoder": decoder},
                            action, ceiling)


def mutation_inputs(contract, count=64):
    """Only public inputs feed trusted deterministic E11 mutations; no reference/annotation fallback."""
    from essaim import code, sandbox
    base = [sandbox.dec(s) for s in contract["public_inputs"]]
    seed = int(hashlib.sha256((contract["id"] + ":verif1:20261010").encode()).hexdigest(), 16)
    rng = random.Random(seed)
    seen, outputs = set(contract["public_inputs"]), []
    for _ in range(40 * count):
        if len(outputs) >= count or not base:
            break
        original = rng.choice(base)
        if not original:
            continue
        values = list(original)
        index = rng.randrange(len(values))
        values[index] = code._mutate(values[index], rng)
        encoded = sandbox.enc_text(tuple(values))
        if encoded is not None and encoded not in seen and len(encoded) <= 20000:
            seen.add(encoded)
            outputs.append(encoded)
    return outputs


def inputs_for(contract, responses, sandbox):
    encoded, proposals = [], []

    def add(value):
        if value not in encoded:
            encoded.append(value)
        return encoded.index(value)

    for family in AUTHORS:
        seen = set()
        for response in responses[family]:
            if response["status"] != "valid":
                continue
            for kind in ("cases", "properties"):
                for item in response["parsed"][kind]:
                    identity = canonical([kind, item])
                    if identity in seen:
                        continue
                    seen.add(identity)
                    arguments = [item["args"]] if kind == "cases" else [item["left"], item["right"]]
                    indices = [add(sandbox.input_value(a)) for a in arguments]
                    proposals.append({"author": family, "kind": kind, "item": item, "indices": indices})
    public_indices = [add(value) for value in contract["public_inputs"]]
    mutations = mutation_inputs(contract)
    mutation_indices = [add(value) for value in mutations]
    return {"encoded": encoded, "proposals": proposals, "public_indices": public_indices,
            "mutation_indices": mutation_indices, "mutation_domain": "uncertain_public_type_preserving",
            "mutation_coverage": {str(n): min(n, len(mutations)) for n in (16, 64)}}


def judge_prompt(contract, candidate, cells, prepared):
    proposal_indices = list(dict.fromkeys(i for p in prepared["proposals"] for i in p["indices"]))
    indices = list(dict.fromkeys(prepared["public_indices"][:2] + proposal_indices[:2]
                                + prepared["mutation_indices"][:2]))
    observations = [{"input": prepared["encoded"][i][:2000], "observation": {
        "ok": cells[i]["ok"], "value_preview": canonical(cells[i].get("value"))[:1000],
        "error": cells[i].get("error"), "output_sha256": sha(cells[i])}} for i in indices]
    # No metadata, model, family, rank or score enters this projection. Raw record digests are public outputs.
    return JUDGE_INSTRUCTION + canonical({"contract": contract["text"], "candidate": candidate["code"],
                                          "original_public_results": candidate["public"],
                                          "executions": observations,
                                          "fallible_proposals_preview": canonical([
                                              {k: p[k] for k in ("kind", "item", "indices")}
                                              for p in prepared["proposals"][:2]])[:2000],
                                          "all_public_execution_sha256": sha(cells),
                                          "all_public_execution_counts": {
                                              "total": len(cells), "successful": sum(c["ok"] for c in cells)},
                                          "evidence_sampling": "first two per input source; bounded previews",
                                          "mutations_domain": prepared["mutation_domain"]})


def question_scores(q, prepared, matrix, judges):
    candidates = q["candidates"]
    baseline = shortlist(candidates)
    order = baseline + [i for i in range(len(candidates)) if i not in baseline]
    v03, v04, cov03, cov04 = {}, {}, {}, {}
    for threshold in (1, 2):
        v03[str(threshold)], cov03[str(threshold)] = supported_scores(
            candidates, prepared["proposals"], matrix, threshold, "cases")
        props, coverage = supported_scores(candidates, prepared["proposals"], matrix, threshold, "properties")
        for n in (16, 64):
            key = f"{threshold}:{n}"
            diff = differential_scores(candidates, matrix, prepared["mutation_indices"][:n])
            v04[key] = [mean_present((p, d)) for p, d in zip(props, diff)]
            cov04[key] = {"properties": coverage, "property_scores": props, "differential_scores": diff,
                          "mutations": min(n, len(prepared["mutation_indices"])), "domain": "uncertain"}
    v05 = [None] * len(candidates)
    for index, results in judges.items():
        v05[index] = mean_present([r["parsed"]["score"] for r in results if r["status"] == "valid"])
    v06 = {}
    for threshold in (1, 2):
        for n in (16, 64):
            for weights in WEIGHTS:
                key = f"{threshold}:{n}:" + "".join(map(str, weights))
                v06[key] = [combine((v03[str(threshold)][i], v04[f"{threshold}:{n}"][i], v05[i]), weights)
                            for i in range(len(candidates))]
    all_variants = {**{f"V03:{k}": v for k, v in v03.items()}, **{f"V04:{k}": v for k, v in v04.items()},
                    "V05": v05, **{f"V06:{k}": v for k, v in v06.items()}}
    return {"baseline": baseline[0], "shortlist": baseline, "scores": all_variants,
            "selected": {key: select(values, order) for key, values in all_variants.items()},
            "coverage": {"V03": cov03, "V04": cov04, "V05_valid_judges": {
                str(i): sum(r["status"] == "valid" for r in rs) for i, rs in judges.items()}},
            "diagnostic_acceptances": {str(t): author_acceptances(candidates, prepared["proposals"], matrix, t)
                                       for t in (1, 2)},
            "agreement_is_not_correctness": True}


def run_pipeline(bundle, plan, output, engine, sandbox, scope="pilot", simulation=False):
    """Stages are batched by family, so each model loads once per author/judge stage."""
    if scope not in ("pilot", "full"):
        raise ValueError("unknown scope")
    questions = bundle["questions"]
    if scope == "pilot":
        questions = [q for bench in ("humanevalplus", "mbppplus")
                     for q in sorted((x for x in questions if x["contract"]["bench"] == bench),
                                     key=lambda q: q["contract"]["id"])[:32]]
    output.mkdir(parents=True, exist_ok=True)
    binding = sha({"plan": plan, "simulation": simulation, "sandbox": sandbox.identity})
    with exclusive(output / ".verif1.lock"):
        manifest = {"binding": binding, "plan": plan, "simulation": simulation, "sandbox": sandbox.identity,
                    "python": platform.python_version(), "platform": platform.platform()}
        manifest_path = output / "verif1_manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("runtime provenance changed")
        else:
            if any(p.name != ".verif1.lock" for p in output.iterdir()):
                raise ValueError("new campaign directory must be empty")
            write_new(manifest_path, manifest)
        ledger = Ledger(output / "verif1_ledger.jsonl", binding)
        # Every invocation probes the actual current sandbox before any model request, including resumes.
        fresh_smoke = sandbox.smoke()
        if scope == "full" and not simulation:
            pilot = ledger.done.get("final:pilot", {}).get("output", {}).get("coverage", {})
            if pilot.get("questions") != 64 or not pilot.get("complete_structural_coverage"):
                raise ValueError("full extension requires the completed 64-question pilot in this same ledger")
        ledger.operation("smoke", sandbox.identity, lambda: fresh_smoke)
        ids = [q["contract"]["bench"] + ":" + q["contract"]["id"] for q in questions]
        author_results = {qid: {a: [] for a in AUTHORS} for qid in ids}
        try:
            for family in AUTHORS:
                for qid, q in zip(ids, questions):
                    prompt = AUTHOR_INSTRUCTION + q["contract"]["text"]
                    for attempt in range(2):
                        result = proposed(ledger, f"{qid}:author:{family}:{attempt}", family, prompt,
                                          512, 20261010 + attempt, engine, parse_proposal)
                        author_results[qid][family].append(result)
            engine.close()
            prepared_all, matrices = {}, {}
            for qid, q in zip(ids, questions):
                prepared = ledger.operation(f"{qid}:inputs", {"contract": q["contract"],
                                            "responses": author_results[qid]},
                                            lambda: inputs_for(q["contract"], author_results[qid], sandbox))
                prepared_all[qid] = prepared
                cells = []
                for i, candidate in enumerate(q["candidates"]):
                    execution = ledger.operation(f"{qid}:matrix:{i}", {"candidate": candidate,
                                                 "contract": q["contract"], "inputs": prepared["encoded"]},
                                                 lambda: sandbox.run(candidate, q["contract"], prepared["encoded"]))
                    if len(execution["cells"]) != len(prepared["encoded"]):
                        raise ValueError("incomplete program x input matrix")
                    cells.append(execution["cells"])
                matrices[qid] = cells
            judge_results = {qid: defaultdict(list) for qid in ids}
            shortlists = {qid: shortlist(q["candidates"])
                          for qid, q in zip(ids, questions)}
            for family in AUTHORS:
                for qid, q in zip(ids, questions):
                    for i in shortlists[qid]:
                        candidate = q["candidates"][i]
                        judges = [a for a in AUTHORS if a != candidate["family"]][:2]
                        if family not in judges:
                            continue
                        prompt = judge_prompt(q["contract"], candidate, matrices[qid][i], prepared_all[qid])
                        result = proposed(ledger, f"{qid}:judge:{i}:{family}", family, prompt,
                                          256, 20261010, engine, parse_judgment)
                        judge_results[qid][i].append(result)
            scores = {}
            for qid, q in zip(ids, questions):
                json_judges = {str(i): results for i, results in judge_results[qid].items()}
                scores[qid] = ledger.operation(f"{qid}:scores", {"prepared": prepared_all[qid],
                                              "matrix": matrices[qid], "judges": json_judges,
                                              "candidates": q["candidates"]},
                                              lambda: question_scores(q, prepared_all[qid], matrices[qid],
                                                                      judge_results[qid]))
        finally:
            engine.close()
        requests = [event["output"] for event in ledger.done.values()
                    if isinstance(event.get("output"), dict) and "raw" in event["output"]]
        coverage = {"scope": scope, "simulation": simulation, "questions": len(questions),
                    "question_ids": ids, "authors_expected": len(questions) * len(AUTHORS) * 2,
                    "judges_expected": sum(2 * len(v) for v in shortlists.values()),
                    "matrix_programs_expected": len(questions) * 35,
                    "author_statuses": dict(counter_status(author_results)),
                    "mutation_coverage": {qid: p["mutation_coverage"] for qid, p in prepared_all.items()},
                    "json_output_coverage_losses": sum(c.get("coverage_loss", False)
                                                       for rows in matrices.values() for row in rows for c in row),
                    "complete_structural_coverage": True, "extension_requires_technical_review": True}
        accounting = {"ceiling": cost(len(questions)), "known_prompt_tokens": sum(
            r["raw"]["prompt_tokens"] for r in requests if type(r["raw"].get("prompt_tokens")) is int),
                      "known_completion_tokens": sum(r["raw"]["completion_tokens"] for r in requests
                                                     if type(r["raw"].get("completion_tokens")) is int),
                      "unknown_usage_requests": sum(r["status"] == "usage_error" for r in requests),
                      "inference_request_seconds": sum(r["raw"]["seconds"] for r in requests),
                      "candidate_original_cost": {k: sum(c["cost"][k] for q in questions for c in q["candidates"])
                                                  for k in ("n_tokens", "prompt_tokens", "ms")},
                      "sandbox_seconds": sum(event["output"].get("seconds", 0) for event in ledger.done.values()
                                             if ":matrix:" in event["key"]),
                      "utc": utc(), "ledger_head": ledger.head,
                      "ledger_sha256": file_sha(ledger.path), "includes_original_35_candidates": True}
        accounting["operation_seconds_including_loads"] = sum(e.get("seconds", 0) for e in ledger.done.values())
        final = ledger.operation(f"final:{scope}", {"scope": scope, "ids": ids},
                                 lambda: {"scores": scores, "coverage": coverage, "costs": accounting})
        # Scope-specific immutable artifacts allow an inspected pilot to extend to full without replacement.
        for name, value in final.items():
            path = output / f"verif1_{scope}_{name}.json"
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous != value:
                    raise ValueError("existing final artifact differs from ledger")
            else:
                write_new(path, value)
        return final["scores"], final["coverage"], final["costs"]


def counter_status(author_results):
    counts = defaultdict(int)
    for families in author_results.values():
        for results in families.values():
            for result in results:
                counts[result["status"]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--simulate", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_bundle(bundle, plan)
    if args.dry_run:
        print(canonical({"status": "ready_dev_no_execution", "questions": len(bundle["questions"]),
                         "plan_sha256": file_sha(args.plan), "cost_pilot": cost(64), "cost_full": cost(270)}))
        return
    if not args.output or (args.run and not args.models):
        parser.error("--output required; real execution additionally requires --models")
    from essaim.verif1_runtime import GGUFEngine, LinuxSandbox, SimulatedEngine, SimulatedSandbox
    engine = SimulatedEngine() if args.simulate else GGUFEngine(bundle["catalogue"], args.models)
    sandbox = SimulatedSandbox() if args.simulate else LinuxSandbox()
    result = run_pipeline(bundle, plan, args.output, engine, sandbox, args.scope, args.simulate)
    print(canonical(result[1]))


if __name__ == "__main__":
    main()
