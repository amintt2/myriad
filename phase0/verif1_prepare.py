"""CPU-only DEV snapshot projection. Never calls dataset loaders, models, network or hidden grading."""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
from pathlib import Path

from essaim import code
from essaim.sandbox import enc_text
from verif1_protocol import (AUTHOR_DECODER, AUTHOR_INSTRUCTION, AUTHORS, FAMILIES, JUDGE_DECODER,
                             JUDGE_INSTRUCTION, MODELS, VERSION,
                             cost, file_sha, sha, utc, write_new)

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"humanevalplus": 82, "mbppplus": 189}
# Required operational provenance only; private research notes are not runtime dependencies.
SOURCE_FILES = (
    "phase0/verif1_protocol.py", "phase0/verif1_prepare.py", "phase0/verif1_run.py",
    "phase0/essaim/verif1_runtime.py", "phase0/colab/verif1_bootstrap.py",
    "phase0/colab/verif1_package.py", "phase0/colab/verif1_campaign.py", "phase0/verif1_analyze.py",
    "phase0/analyze_percees_code.py", "phase0/essaim/models.py", "phase0/colab/safe_cli.py",
    "phase0/colab/transfer.py", "phase0/colab/reconstruct.py", "phase0/colab/session_json.py",
    "phase0/colab/read_timeout.py", "phase0/colab/colab_phase0.sh",
    "phase0/colab/budget.py", "phase0/colab/verif1_supervisor.py",
    "phase0/essaim/sandbox.py", "phase0/essaim/code.py", "phase0/run_solo.py",
    "phase0/essaim/llamacpp.py", "phase0/essaim/common.py", "phase0/essaim/results.py",
    "phase0/essaim/answers.py", "phase0/essaim/data.py", "phase0/colab_jobs.py",
    "phase0/essaim/__init__.py", "phase0/pyproject.toml", "phase0/uv.lock",
    "phase0/README.md",
    "docs/12_percees_verif1_colab.md",
)


def source_hashes():
    return {name: file_sha(ROOT / name) for name in SOURCE_FILES}


def read_dev_prefix(path: Path, count: int):
    """Unbuffered readline reads precisely the authorized prefix, never the next byte of TEST."""
    digest, rows = hashlib.sha256(), []
    if path.name not in {f"{b}_all_s0_v2.jsonl" for b in COUNTS}:
        raise ValueError("unknown mixed cache; no test file may be opened")
    with path.open("rb", buffering=0) as stream:
        for _ in range(count):
            line = stream.readline()
            if not line.endswith(b"\n"):
                raise ValueError("missing or incomplete DEV prefix")
            digest.update(line)
            row = json.loads(line)
            rows.append({k: row[k] for k in ("id", "prompt", "entry_point", "test_list", "test_imports")
                         if k in row})
    return rows, digest.hexdigest()


def public_contract(bench: str, item: dict) -> dict:
    if bench == "humanevalplus":
        entry = item["entry_point"]
        tree = ast.parse(item["prompt"])
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry]
        if len(funcs) != 1:
            raise ValueError("ambiguous public HumanEval entry")
        function = funcs[0]
        doc = ast.get_docstring(function, clean=False)
        function.body = ([ast.Expr(ast.Constant(doc))] if doc is not None else []) + [ast.Pass()]
        function.decorator_list = []
        sanitized = ast.unparse(ast.fix_missing_locations(ast.Module(body=imports + [function], type_ignores=[])))
        public_item = {"id": item["id"], "entry_point": entry, "prompt": sanitized}
        inputs = code.visible_inputs(bench, public_item)
        text = sanitized
        header = "\n".join(ast.unparse(n) for n in imports)
    elif bench == "mbppplus":
        calls = []
        for test in item["test_list"]:
            tree = ast.parse(test)
            # Unwrap public truthiness, negation and equality predicates; ignore builtin wrappers.
            for node in tree.body:
                if not isinstance(node, ast.Assert):
                    continue
                predicate = node.test.left if isinstance(node.test, ast.Compare) else node.test
                calls.extend(n for n in ast.walk(predicate) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Name) and not hasattr(builtins, n.func.id))
        names = {node.func.id for node in calls}
        if len(names) != 1:
            raise ValueError(f"ambiguous MBPP public entry: {item['id']}")
        entry = names.pop()
        inputs = []
        for node in calls:
            try:
                if node.keywords:
                    continue
                # literal_eval interprets data-only AST nodes, never executable Python expressions.
                inputs.append(tuple(ast.literal_eval(arg) for arg in node.args))
            except (ValueError, TypeError):
                continue
        import_tree = ast.parse("\n".join(item["test_imports"]))
        if any(not isinstance(n, (ast.Import, ast.ImportFrom)) for n in import_tree.body):
            raise ValueError("non-import public preamble")
        header = ast.unparse(import_tree)
        text = item["prompt"] + "\nPublic examples:\n" + "\n".join(item["test_list"])
    else:
        raise ValueError("unknown benchmark")
    if not entry.isidentifier():
        raise ValueError("invalid public entry")
    encoded = [enc_text(v) for v in inputs]
    if any(v is None for v in encoded):
        raise ValueError("unrepresentable public input")
    result = {"id": item["id"], "bench": bench, "text": text, "entry": entry, "header": header,
              "public_inputs": list(dict.fromkeys(encoded)), "public_only": True}
    result["sha256"] = sha(result)
    return result


def prepare(snapshot: Path, cache: Path, output: Path):
    from colab_jobs import SOLO
    expected = json.loads((ROOT / "phase0/results/percees_code_snapshot_copy.json").read_text(encoding="utf-8"))
    expected = expected["source_and_snapshot_sha256"]
    if len(expected) != 56 or any(not name.endswith(("_dev.jsonl", "_dev.jsonl.meta.json")) for name in expected):
        raise ValueError("snapshot registry must be the frozen 56 DEV files")
    hashes = {}
    for name, wanted in expected.items():
        hashes[name] = file_sha(snapshot / name)
        if hashes[name] != wanted:
            raise ValueError(f"frozen snapshot differs: {name}")
    questions, cache_hashes, catalogue = [], {}, {}
    for bench, count in COUNTS.items():
        rows, prefix_hash = read_dev_prefix(cache / f"{bench}_all_s0_v2.jsonl", count)
        cache_hashes[bench] = {"dev_prefix_sha256": prefix_hash, "lines": count, "test_suffix_read": False}
        public_rows = {}
        with (snapshot / f"codeexec_e11_colab_{bench}_dev.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                raw = json.loads(line)
                public_rows[(raw["id"], raw["prog"])] = {
                    "load": raw["load"], "visible": bool(raw["visible"]) and all(raw["visible"]),
                    "visible_cases": raw["visible"], "visible_errors": raw["visible_err"], "extra": raw["extra"],
                    "status": {k: raw["status"][k] for k in ("visible", "extra")}}
        candidates = {}
        for model in MODELS:
            path = snapshot / f"code_{model.replace('/', '__')}_colab_{bench}_dev.jsonl"
            meta = json.loads(path.with_name(path.name + ".meta.json").read_text(encoding="utf-8"))
            if meta["split"] != "dev" or meta["model"] != model or meta["bench"] != bench:
                raise ValueError("unexpected generation manifest")
            if FAMILIES[model] in AUTHORS:
                entry = {"model": model, "family": FAMILIES[model], "revision": meta["revision"],
                         "gguf": meta["gguf"], "weights": meta["weights"], "engine": meta["engine"],
                         "gguf_repository": SOLO[model][0],
                         "license": "MIT" if FAMILIES[model] == "Phi" else "Apache-2.0",
                         "license_evidence": "https://huggingface.co/" + model,
                         "license_checked": "official model cards 2026-10-10; Phi/OLMo also parent checked"}
                if catalogue.setdefault(FAMILIES[model], entry) != entry:
                    raise ValueError("cross-benchmark model provenance differs")
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    raw = json.loads(line)
                    if raw["model"] != model or raw["bench"] != bench or raw["prog"] != code.prog_id(raw["code"]):
                        raise ValueError("candidate identity/code mismatch")
                    c = {"model": model, "family": FAMILIES[model], "sample": raw["sample"], "code": raw["code"],
                         "code_sha256": hashlib.sha256(raw["code"].encode()).hexdigest(), "prog": raw["prog"],
                         "cost": {k: raw[k] for k in ("n_tokens", "prompt_tokens", "ms")},
                         "public": public_rows[(raw["id"], raw["prog"])]}
                    candidates.setdefault(raw["id"], []).append(c)
        if set(candidates) != {r["id"] for r in rows}:
            raise ValueError("DEV contracts and candidates do not align")
        for row in rows:
            if row["id"] == "HumanEval/32":
                continue
            fixed = candidates[row["id"]]
            if (len(fixed) != 35 or {(c["model"], c["sample"]) for c in fixed}
                    != {(m, s) for m in MODELS for s in range(5)}):
                raise ValueError("exactly 35 fixed E11 candidates required")
            fixed.sort(key=lambda c: (MODELS.index(c["model"]), c["sample"]))
            questions.append({"contract": public_contract(bench, row), "candidates": fixed})
    questions.sort(key=lambda q: (q["contract"]["bench"], q["contract"]["id"]))
    if len(questions) != 270:
        raise ValueError("frozen 270 DEV questions required")
    bundle = {"version": VERSION, "split": "dev", "questions": questions, "catalogue": catalogue,
              "snapshot": hashes, "cache_prefixes": cache_hashes, "excluded_historical": ["HumanEval/32"]}
    plan = {"version": VERSION, "split": "dev", "bundle_sha256": sha(bundle), "sources": source_hashes(),
            "catalogue_sha256": sha(catalogue), "prompts": {"author": AUTHOR_INSTRUCTION, "judge": JUDGE_INSTRUCTION},
            "config": {"seed": 20261010, "author_calls": 2, "author_tokens": 512, "judge_tokens": 256,
                       "support_thresholds": [1, 2], "mutations": [16, 64], "shortlist": 3,
                       "author_decoder": AUTHOR_DECODER, "judge_decoder": JUDGE_DECODER,
                       "authors": list(AUTHORS), "seeds": [20261010, 20261011],
                       "shortlist_order": "family_count_then_catalogue_sample_no_K",
                       "context": 8192, "parallel": 1}, "cost_pilot": cost(64), "cost_full": cost(270),
            "created_utc": utc(), "candidate_regeneration": False, "network_during_prepare": False}
    write_new(output / "verif1_dev_bundle.json", bundle)
    write_new(output / "verif1_dev_plan.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("C:/tmp/percees/e11-dev"))
    parser.add_argument("--cache", type=Path, default=Path("C:/Decentralised LLM/phase0/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.snapshot, args.cache, args.output), indent=2))


if __name__ == "__main__":
    main()
