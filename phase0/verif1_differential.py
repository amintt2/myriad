"""V07: offline public-input differential execution, then separate complete DEV analysis."""
from __future__ import annotations

import argparse
import ast
import builtins
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import analyze_percees_code as analysis
from essaim import code

ROOT = Path(__file__).resolve().parent.parent
PREFIX = ROOT / "phase0/results/verif1_differential_"
CONFIG = {
    "variant": "V07", "registration_commit": "cb07e83", "split": "dev", "test_consumed": 0,
    "campaign_variants": 7, "future_alpha": 0.05 / 28, "sizes": [16, 64], "max_attempts": 40 * 64,
    "seed": "int(SHA256(id + ':verif1:20261010'), 16)", "outer_folds": 5, "inner_folds": 3,
    "fold": "int(SHA256('percees-v1:' + id),16) % n", "workers": 4,
    "representative": "visible only; sample then fixed model order; fixed for all inputs",
    "score": "mean over target normal results of matches / all other visible represented families",
    "fallback": "zero positive agreement: visible plus fold-train K weights",
    "eligibility": "visible first, then loaded; fallback if neither",
    "bootstrap": {"draws": 2000, "seed": 20261010, "descriptive_only": True},
}
MODELS = analysis.SWARM
FAMILIES = dict(zip(MODELS, ("Qwen", "Gemma", "Granite", "SmolLM", "Mistral", "Phi", "OLMo")))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(suffix):
    return Path(str(PREFIX) + suffix)


def attempt_prefix(attempt):
    require(re.fullmatch(r"[a-z][a-z0-9_]{0,31}", attempt) is not None, "invalid attempt name")
    suffix = "" if attempt == "original" else attempt + "_"
    return ROOT / ("phase0/results/verif1_differential_" + suffix)


def write_once(target, value):
    payload = canonical(value) + "\n"
    if target.exists():
        require(target.read_text(encoding="utf-8") == payload, f"immutable artifact changed: {target}")
        return
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def preparation_manifest(target, proposed):
    """Preserve first timing and commit provenance on an unchanged deterministic preparation."""
    if not target.exists():
        return proposed
    original = json.loads(target.read_text(encoding="utf-8"))
    variable = {"preparation_cost", "code_commit"}
    require({k: val for k, val in original.items() if k not in variable}
            == {k: val for k, val in proposed.items() if k not in variable}, "preparation inputs/code changed")
    return original


def safe(value, depth=0, budget=None):
    budget = [4096] if budget is None else budget
    budget[0] -= 1
    require(depth <= 12 and budget[0] >= 0, "input structural budget")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        require(value.bit_length() <= 256, "large integer")
        return
    if type(value) is float:
        require(math.isfinite(value), "nonfinite input")
        return
    if type(value) is str:
        require(len(value) <= 4096, "large string")
        return
    require(type(value) in (list, tuple, dict, set, frozenset), "unsupported input")
    for item in (list(value.keys()) + list(value.values()) if type(value) is dict else value):
        safe(item, depth + 1, budget)


def mutate(qid, examples):
    """E11's public-example operator, full SHA seed, bounded attempts and no annotation fallback."""
    rng = random.Random(int(hashlib.sha256((qid + ":verif1:20261010").encode()).hexdigest(), 16))
    bases = []
    for args in examples:
        try:
            safe(args)
            require(code._roundtrips(args), "input does not roundtrip")
            bases.append(args)
        except ValueError:
            continue
    seen, outputs, attempts = {repr(a) for a in bases}, [], 0
    if not bases:
        return outputs, attempts
    for _ in range(CONFIG["max_attempts"]):
        if len(outputs) == 64:
            break
        attempts += 1
        args = list(rng.choice(bases))
        if not args:
            break
        for _ in range(1 if rng.random() < 0.5 else 2):
            index = rng.randrange(len(args))
            args[index] = code._mutate(args[index], rng)
        result = tuple(args)
        try:
            safe(result)
        except ValueError:
            continue
        text = repr(result)
        if text not in seen and len(text) < 20000 and code._roundtrips(result):
            seen.add(text)
            outputs.append(text)
    return outputs, attempts


def project_public(bench, row):
    """All raw reference/hidden fields are discarded before any contract preparation."""
    keys = ("id", "prompt", "entry_point") if bench == "humanevalplus" else (
        "id", "prompt", "test_list", "test_imports")
    return {k: row[k] for k in keys}


def public_contract(bench, item):
    expected = {"id", "prompt", "entry_point"} if bench == "humanevalplus" else {
        "id", "prompt", "test_list", "test_imports"}
    require(set(item) == expected, "nonpublic contract fields")
    if bench == "humanevalplus":
        entry = item["entry_point"]
        examples = [a for a, _ in code.humaneval_examples(item)]
        header = item["prompt"]
    else:
        calls = []
        excluded = set(vars(builtins))
        for source in item["test_imports"]:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    excluded.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        for test in item["test_list"]:
            tree = ast.parse(test)
            require(len(tree.body) == 1 and isinstance(tree.body[0], ast.Assert), "nonpublic test statement")
            # Walk through builtin/imported wrappers without using hidden tests to resolve the entry.
            for node in ast.walk(tree.body[0].test):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in excluded:
                    calls.append(node.func.id)
        require(bool(calls) and len(set(calls)) == 1, "ambiguous public entry point")
        entry = calls[0]
        examples = []
        for test in item["test_list"]:
            for node in ast.walk(ast.parse(test)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry:
                    args = code._call_args(ast.unparse(node), entry)
                    if args is not None and args not in examples:
                        examples.append(args)
        header = "\n".join(item["test_imports"])
    require(entry.isidentifier(), "invalid entry")
    mutations, attempts = mutate(item["id"], examples)
    return {"bench": bench, "id": item["id"], "entry": entry, "header": header,
            "contract": code.prompt(bench, item), "inputs": mutations, "mutation_attempts": attempts,
            "public_examples": [repr(a) for a in examples]}


def read_prefix(cache, bench, identity):
    n = analysis.EXPECTED_N[bench]
    target = cache / f"{bench}_all_s0_v2.jsonl"
    meta = cache / f"{bench}_all_s0_v2.meta.json"
    require(json.loads(meta.read_text(encoding="utf-8")) == identity, "cache metadata identity mismatch")
    rows, h = [], hashlib.sha256()
    # Unbuffered, bounded readline: never prefetch the held-out suffix.
    with target.open("rb", buffering=0) as stream:
        for _ in range(n):
            raw = stream.readline()
            require(raw.endswith(b"\n"), "incomplete dev prefix")
            h.update(raw)
            rows.append(project_public(bench, json.loads(raw)))
    require(len({r["id"] for r in rows}) == n, "duplicate dev cache IDs")
    return rows, {"prefix_rows": n, "dev_prefix_sha256": h.hexdigest(), "metadata_sha256": file_sha(meta)}


def validate_snapshot(snapshot):
    record = ROOT / "phase0/results/percees_code_snapshot_copy.json"
    expected = json.loads(record.read_text(encoding="utf-8"))["source_and_snapshot_sha256"]
    require(len(expected) == 56 and all("_dev.jsonl" in name for name in expected), "nondev inventory")
    require({p.name for p in snapshot.iterdir() if p.is_file()} == set(expected), "snapshot inventory")
    for name, digest in expected.items():
        require(file_sha(snapshot / name) == digest, f"snapshot hash mismatch: {name}")
    return expected


def prepare(snapshot, cache):
    started, cpu_started = time.monotonic(), time.process_time()
    hashes = validate_snapshot(snapshot)
    questions, cache_hashes = [], {}
    for bench in analysis.BENCHES:
        generation_meta = analysis.manifest(snapshot / f"code_{MODELS[0].replace('/', '__')}_colab_{bench}_dev.jsonl")
        rows, cache_hashes[bench] = read_prefix(cache, bench, generation_meta["data"])
        require({r["id"] for r in rows} == {r["id"] for r in analysis.read_jsonl(
            snapshot / f"code_{MODELS[0].replace('/', '__')}_colab_{bench}_dev.jsonl", ("id", "sample"))},
            "cache/generation IDs differ")
        executions = analysis.read_jsonl(snapshot / f"codeexec_e11_colab_{bench}_dev.jsonl", ("id", "prog"))
        # No reference or label reaches the public preparation path. The exclusion is historically fixed.
        public = {(r["id"], r["prog"]): {"load": r["load"] is None,
                  "visible": r["load"] is None and all(r["visible"]), "visible_count": sum(r["visible"])}
                  for r in executions if r["prog"] != "reference"}
        generated = {}
        for model in MODELS:
            target = snapshot / f"code_{model.replace('/', '__')}_colab_{bench}_dev.jsonl"
            man = analysis.manifest(target)
            generated[model] = analysis.read_jsonl(target, ("id", "sample"))
            for row in generated[model]:
                analysis.check_generation(row, man, bench, model)
        for item in sorted(rows, key=lambda r: r["id"]):
            if bench == "humanevalplus" and item["id"] == "HumanEval/32":
                continue
            question = public_contract(bench, item)
            candidates = []
            for model in MODELS:
                for row in sorted((r for r in generated[model] if r["id"] == item["id"]), key=lambda r: r["sample"]):
                    candidates.append({"model": model, "sample": row["sample"], "prog": row["prog"],
                                       "code": row["code"], "public": public[item["id"], row["prog"]]})
            require(len(candidates) == 35, "incomplete fixed pool")
            question["candidates"] = candidates
            questions.append(question)
    require(Counter(q["bench"] for q in questions) == {"humanevalplus": 81, "mbppplus": 189}, "dev coverage")
    manifest = {"config": CONFIG, "snapshot_sha256": hashes, "cache": cache_hashes,
                "source_sha256": source_hashes(), "public_sha256": sha(questions),
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "preparation_cost": {"wall_s": time.monotonic() - started,
                                     "controller_cpu_s": time.process_time() - cpu_started}}
    manifest_target = path("inputs_manifest.json")
    preparation_cost = manifest["preparation_cost"]
    manifest = preparation_manifest(manifest_target, manifest)
    write_once(path("public.json"), questions)
    write_once(manifest_target, manifest)
    with path("preparation_segments.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(canonical({"inputs_manifest_sha256": sha(manifest), "cost": preparation_cost}) + "\n")
    print(f"Prepared {len(questions)} public contracts; no scoring.", flush=True)


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): file_sha(p) for p in (
        Path(__file__), ROOT / "phase0/analyze_percees_code.py", ROOT / "phase0/essaim/code.py",
        ROOT / "phase0/essaim/sandbox.py", ROOT / "phase0/essaim/models.py",
        ROOT / "tests/test_verif1_differential.py",
        ROOT / "docs/12_percees_v2_differential.md")}


class Ledger:
    """Hash chained attempts/results. A missing terminal operation may be replayed, never a completed one."""
    def __init__(self, target, binding):
        self.target, self.binding, self.head = target, binding, "0" * 64
        self.done, self.attempts, self.events, self.inputs = {}, Counter(), [], {}
        if target.exists():
            with target.open("rb") as stream:
                for raw in stream:
                    require(raw.endswith(b"\n"), "partial ledger tail: preserve and repair before resuming")
                    event = json.loads(raw)
                    digest = event.pop("sha256")
                    require(sha(event) == digest and event["previous"] == self.head
                            and event["binding"] == binding, "ledger provenance mismatch")
                    self.accept(event)
                    self.head = digest

    def accept(self, event):
        key = event["key"]
        require(event["kind"] in ("start", "result"), "unknown event")
        require(key not in self.inputs or self.inputs[key] == event["input_sha256"], "operation input changed")
        if event["kind"] == "start":
            require(key not in self.done, "completed operation restarted")
            self.attempts[key] += 1
            require(event["attempt"] == self.attempts[key], "attempt sequence")
        else:
            require(key not in self.done and self.attempts[key] == event["attempt"], "duplicate/unstarted result")
            require(sha(event["output"]) == event["output_sha256"], "output hash")
            self.done[key] = event
        self.events.append(event)
        self.inputs[key] = event["input_sha256"]

    def append(self, kind, key, input_hash, **fields):
        event = {"kind": kind, "key": key, "input_sha256": input_hash, "binding": self.binding,
                 "previous": self.head, "attempt": self.attempts[key] + (kind == "start"), **fields}
        require(key not in self.done and (key not in self.inputs or self.inputs[key] == input_hash),
                "completed operation or changed input")
        digest = sha(event)
        with self.target.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical({**event, "sha256": digest}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.accept(event)
        self.head = digest


def jobs(questions):
    operations = {}
    for q in questions:
        for c in q["candidates"]:
            key = f"{q['bench']}:{q['id']}:{c['prog']}"
            job = {"code": c["code"], "entry": q["entry"], "header": q["header"],
                   "inputs": q["inputs"], "phase": "extra"}
            require(key not in operations or operations[key] == job, "hash dedup source conflict")
            operations[key] = job
    return operations


def load_public():
    questions = json.loads(path("public.json").read_text(encoding="utf-8"))
    manifest = json.loads(path("inputs_manifest.json").read_text(encoding="utf-8"))
    require(manifest["config"] == CONFIG and manifest["source_sha256"] == source_hashes(), "code/config changed")
    require(manifest["public_sha256"] == sha(questions), "public payload changed")
    return questions, manifest


def child_script(sandbox):
    # Trusted harness extension only; candidate compilation remains inside sandbox.CHILD.
    old = '_rec = {"sig": _call(lambda: signature(_r), cfg["encode_s"])}'
    new = ('_txt = _call(lambda: enc_text(_r), cfg["encode_s"])\n'
           '                _rec = {"out": _txt} if _txt is not None else '
           '{"digest": _call(lambda: digest(_r), cfg["encode_s"])}')
    require(sandbox.CHILD.count(old) == 1, "sandbox extension anchor changed")
    result = sandbox.CHILD.replace(old, new)
    result = result.replace('_emit(kind="done", ms=round((time.perf_counter() - _t0) * 1000))',
                            '_usage = resource.getrusage(resource.RUSAGE_SELF)\n'
                            '_emit(kind="done", cpu_s=_usage.ru_utime + _usage.ru_stime)')
    return result


def typed_output(text):
    """Canonicalize plain codec trees without constructing any candidate-defined Python objects."""
    def normalize(node):
        require(type(node) is list and node and type(node[0]) is str, "malformed typed output")
        tag = node[0]
        require(tag != "r", "opaque candidate output")
        if tag in ("l", "t", "S", "F"):
            values = [normalize(x) for x in node[1]]
            return [tag, sorted(values, key=canonical) if tag in ("S", "F") else values]
        if tag == "d":
            values = [[normalize(k), normalize(val)] for k, val in node[1]]
            return [tag, sorted(values, key=lambda pair: canonical(pair[0]))]
        if tag == "c":
            return [tag, normalize(node[1]), normalize(node[2])]
        if tag == "a":
            return [tag, node[1], node[2], normalize(node[3])]
        require(tag in ("N", "b", "i", "f", "s", "y", "Y"), "unsupported codec tag")
        return node
    return canonical(normalize(json.loads(text)))


def validate_campaign(questions, manifest, script):
    """Read-only complete validation, shared by finished execution replay and analysis."""
    execution = json.loads(path("execution_manifest.json").read_text(encoding="utf-8"))
    require(set(execution) == {"inputs_manifest_sha256", "script_sha256", "isolation", "python",
                               "job_keys_sha256"}, "execution manifest schema")
    operations = jobs(questions)
    require(execution["inputs_manifest_sha256"] == sha(manifest), "execution input manifest changed")
    require(execution["job_keys_sha256"] == sha(sorted(operations)), "execution job inventory changed")
    require(execution["script_sha256"] == sha(script), "execution child script changed")
    require(execution["isolation"]["landlock"].startswith("on (ABI "), "recorded Landlock missing")
    ledger = Ledger(path("execution.jsonl"), sha(execution))
    completion = json.loads(path("complete.json").read_text(encoding="utf-8"))
    require(completion == {"binding": ledger.binding, "head": ledger.head, "operations": len(operations)}
            and set(ledger.done) == set(operations), "campaign not complete")
    for event in ledger.events:
        key = event["key"]
        require(key in operations and event["input_sha256"] == sha(operations[key]), "campaign input mismatch")
        if event["kind"] == "result":
            require(len(event["output"]["cells"]) == len(operations[key]["inputs"]), "partial matrix")
    segments = [json.loads(line) for line in path("segments.jsonl").read_text(encoding="utf-8").splitlines()]
    require(bool(segments) and segments[-1]["complete"] is True, "completion costs missing")
    require(all(segment["binding"] == ledger.binding for segment in segments), "segment binding changed")
    return execution, ledger, operations


def execute_job(job, script, sandbox):
    recs, status, ms = sandbox._spawn(script, job, "extra", sandboxed=True)
    iso = next((r.get("landlock") for r in recs if r.get("kind") == "iso"), None)
    require(isinstance(iso, str) and iso.startswith("on (ABI "), f"child Landlock not on: {iso}")
    load = next((r.get("err") for r in recs if r.get("kind") == "load"), "MissingLoad")
    cases = {r["i"]: r for r in recs if r.get("kind") == "case"}
    cells = []
    for i in range(len(job["inputs"])):
        r = cases.get(i, {})
        output = {k: r[k] for k in ("out", "digest") if k in r}
        error = r.get("err", None if output else status + ":missing")
        if "out" in output:
            try:
                output["out"] = typed_output(output["out"])
            except (ValueError, TypeError, IndexError, RecursionError):
                output, error = {}, "UnsupportedTypedOutput"
        cells.append({"ok": load is None and bool(output) and not r.get("err"),
                      "output": output, "error": error})
    return {"status": status, "load": load, "cells": cells, "wall_ms": ms, "landlock": iso,
            "cpu_s": next((r["cpu_s"] for r in recs if r.get("kind") == "done"), None)}


def execute(smoke=False):
    require(sys.platform.startswith("linux"), "execution requires Linux")
    from essaim import sandbox
    script = child_script(sandbox)
    if not smoke and path("complete.json").exists():
        questions, manifest = load_public()
        validate_campaign(questions, manifest, script)
        print("Completed campaign fully validated; no operations or artifacts changed.", flush=True)
        return
    import fcntl
    import resource
    started = time.monotonic()
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    iso = sandbox.isolation()
    require(iso["landlock"].startswith("on (ABI "), "Linux Landlock required")
    if smoke:
        job = {"code": "def f(x):\n    return sorted(x)", "entry": "f", "header": "",
               "inputs": ["([3, 1, 2],)", "([],)", "(1,)"], "phase": "extra"}
        result = execute_job(job, script, sandbox)
        require([c["ok"] for c in result["cells"]] == [True, True, False], "synthetic smoke failed")
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        write_once(path("smoke.json"), {"result": result, "isolation": iso, "script_sha256": sha(script),
            "cost": {"wall_s": time.monotonic() - started,
                     "child_cpu_s": usage.ru_utime + usage.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime,
                     "controller_cpu_s": time.process_time()}})
        print("Synthetic smoke passed.", flush=True)
        return
    questions, manifest = load_public()
    operations = jobs(questions)
    binding_data = {"inputs_manifest_sha256": sha(manifest), "script_sha256": sha(script),
                    "isolation": iso, "python": sys.version, "job_keys_sha256": sha(sorted(operations))}
    write_once(path("execution_manifest.json"), binding_data)
    binding = sha(binding_data)
    with path("lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = Ledger(path("execution.jsonl"), binding)
        require(set(ledger.done) <= set(operations), "unexpected journal operations")
        for event in ledger.events:
            require(event["key"] in operations and event["input_sha256"] == sha(operations[event["key"]]),
                    "job input provenance changed")
        missing = [key for key in operations if key not in ledger.done]
        if path("complete.json").exists():
            validate_campaign(questions, manifest, script)
            print("Completed campaign fully validated; no operations or artifacts changed.", flush=True)
            return
        require(bool(missing), "all results exist but completion marker is missing; technical recovery required")
        def stop(*_):
            sandbox.shutdown()
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(f"CPU campaign: {len(ledger.done)}/{len(operations)} complete; {len(missing)} missing.", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                pending, iterator = {}, iter(missing)
                while True:
                    while len(pending) < 4:
                        key = next(iterator, None)
                        if key is None:
                            break
                        ledger.append("start", key, sha(operations[key]), utc=time.time())
                        pending[executor.submit(execute_job, operations[key], script, sandbox)] = key
                    if not pending:
                        break
                    complete, _ = concurrent.futures.wait(pending, timeout=30,
                                                          return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in complete:
                        key = pending.pop(future)
                        output = future.result()
                        ledger.append("result", key, sha(operations[key]), output=output,
                                      output_sha256=sha(output), utc=time.time())
                    if complete and (len(ledger.done) % 50 < 4 or not pending):
                        print(f"Progress {len(ledger.done)}/{len(operations)}; elapsed {time.monotonic()-started:.0f}s",
                              flush=True)
        finally:
            cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            segment = {"wall_s": time.monotonic() - started,
                       "child_cpu_s": cpu_after.ru_utime + cpu_after.ru_stime
                       - cpu_before.ru_utime - cpu_before.ru_stime,
                       "controller_cpu_s": time.process_time(), "complete": len(ledger.done) == len(operations),
                       "started_operations": len(missing), "binding": binding}
            with path("segments.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical(segment) + "\n")
        require(set(ledger.done) == set(operations), "incomplete campaign; no scoring")
        write_once(path("complete.json"), {"binding": binding, "head": ledger.head, "operations": len(operations)})
        print("All CPU operations complete. Analysis is now permitted.", flush=True)


def differential_scores(candidates, matrix, features, size):
    """Label-free score. Missing peers stay in the denominator; target exceptions are omitted from the mean."""
    representatives = {}
    for c in sorted(candidates, key=lambda c: (c.sample, analysis.MODELS.index(c.model))):
        if features[c].visible:
            representatives.setdefault(FAMILIES[c.model], c)
    result = {}
    for c in candidates:
        peers = [p for family, p in representatives.items() if family != FAMILIES[c.model]]
        rates = []
        if peers:
            for i, cell in enumerate(matrix[c][:size]):
                if cell["ok"]:
                    matches = sum(matrix[p][i]["ok"] and matrix[p][i]["output"] == cell["output"] for p in peers)
                    rates.append(matches / len(peers))
        result[c] = statistics.mean(rates) if rates else None
    return result


def select(candidates, features, scores, weights):
    eligible = [c for c in candidates if features[c].visible] or [c for c in candidates if features[c].load]
    if not eligible or not any(scores[c] is not None and scores[c] > 0 for c in eligible):
        return analysis.select(candidates, features, weights, "visible")
    return max(eligible, key=lambda c: (scores[c] if scores[c] is not None else -1, analysis.rank(c, weights)))


def analyze(snapshot):
    """Labels exist only here and in analyze_percees_code; no partial campaign can enter this function."""
    questions, manifest = load_public()
    require(validate_snapshot(snapshot) == manifest["snapshot_sha256"], "snapshot changed")
    from essaim import sandbox
    execution, ledger, operations = validate_campaign(questions, manifest, child_script(sandbox))
    datasets = {b: analysis.load_bench(snapshot, b) for b in analysis.BENCHES}
    require([len(datasets[b].candidates) for b in analysis.BENCHES] == [81, 189], "analysis coverage")
    for m in analysis.MODELS:
        a, b = (datasets[x].validation["generation_manifests"][m] for x in analysis.BENCHES)
        require(all(a[k] == b[k] for k in ("model", "revision", "gguf", "weights", "engine", "backend")),
                "cross-bench model identity")
    public_features, matrices, signals = {}, {}, {}
    for q in questions:
        b, qid = q["bench"], q["id"]
        data = datasets[b]
        candidates = analysis.pool(data, qid, MODELS, 5)
        require(len(candidates) == 35, "analysis fixed pool")
        by_identity = {(c["model"], c["sample"], c["prog"]): c for c in q["candidates"]}
        features, matrix = {}, {}
        for c in candidates:
            source = by_identity[c.model, c.sample, c.prog]
            require(code.normalized_text(source["code"]) == c.text, "program projection mismatch")
            p = source["public"]
            features[c] = analysis.PublicFeatures(p["load"], p["visible"], p["visible_count"], ())
            matrix[c] = ledger.done[f"{b}:{qid}:{c.prog}"]["output"]["cells"]
        public_features[b, qid], matrices[b, qid] = features, matrix
        for size in CONFIG["sizes"]:
            signals[b, qid, size] = differential_scores(candidates, matrix, features, size)

    def predictions(b, ids, size, weights):
        return {q: select(analysis.pool(datasets[b], q, MODELS, 5), public_features[b, q],
                          signals[b, q, size], weights) for q in ids}

    journal, records, choices = [], [], []
    for outer in range(5):
        train = {b: [q for q in d.candidates if analysis.fold(q) != outer] for b, d in datasets.items()}
        evaluate = {b: [q for q in d.candidates if analysis.fold(q) == outer] for b, d in datasets.items()}
        fits = {b: analysis.fit(d, train[b], MODELS) for b, d in datasets.items()}
        internal = {size: {b: {} for b in datasets} for size in CONFIG["sizes"]}
        for inner in range(3):
            itrain = {b: [q for q in train[b] if analysis.fold(q, 3) != inner] for b in datasets}
            ival = {b: [q for q in train[b] if analysis.fold(q, 3) == inner] for b in datasets}
            ifits = {b: analysis.fit(d, itrain[b], MODELS) for b, d in datasets.items()}
            for size in CONFIG["sizes"]:
                scores = {}
                for b, d in datasets.items():
                    pred = predictions(b, ival[b], size, ifits[b]["weights"])
                    internal[size][b].update(pred)
                    scores[b] = analysis.score(d, pred)
                journal.append({"type": "inner_trial", "experiment": "V07", "outer": outer, "inner": inner,
                                "size": size, "scores": scores, "train": itrain, "validation": ival, "fits": ifits})
        internal_scores = {size: {b: analysis.score(d, internal[size][b]) for b, d in datasets.items()}
                           for size in CONFIG["sizes"]}
        size = max(CONFIG["sizes"], key=lambda n: (statistics.mean(internal_scores[n].values()), -n))
        journal.append({"type": "decision", "experiment": "V07", "outer": outer, "size": size,
                        "internal_scores": internal_scores})
        solo_variants = {m: analysis.choose_variant(datasets, train, (m,), outer, f"multi:{m}", journal)
                         for m in ("Qwen/Qwen3.8-27B", MODELS[0])}
        best = max(MODELS, key=lambda m: (statistics.mean(fits[b]["greedy_accuracy"][m] for b in datasets),
                                        -MODELS.index(m)))
        choices.append({"outer": outer, "size": size, "train": train, "evaluation": evaluate,
                        "fits": fits, "solo_variants": solo_variants, "best_small_greedy": best})
        for b, data in datasets.items():
            methods = {"V07": predictions(b, evaluate[b], size, fits[b]["weights"]),
                       "visible_k5": analysis.predict(data, evaluate[b], MODELS, (5, "visible"), fits[b]["weights"]),
                       "best_small_greedy": analysis.predict(data, evaluate[b], (best,), (1, "visible"), {best: 1})}
            for model, variant in solo_variants.items():
                methods[f"multi:{model}"] = analysis.predict(data, evaluate[b], (model,), variant,
                                                             analysis.fit(data, train[b], (model,))["weights"])
            for method, preds in methods.items():
                for qid, c in preds.items():
                    models = MODELS if method in ("V07", "visible_k5") else (
                        (best,) if method == "best_small_greedy" else (method.removeprefix("multi:"),))
                    k = 5 if method in ("V07", "visible_k5") else (
                        1 if method == "best_small_greedy" else solo_variants[models[0]][0])
                    records.append({"bench": b, "id": qid, "outer": outer, "method": method,
                                    "size": size if method == "V07" else None,
                                    "selected": {"model": c.model, "sample": c.sample, "prog": c.prog},
                                    "correct": data.labels[c], "cost": analysis.costs(
                                        analysis.pool(data, qid, models, k), "visible")})
    results, paired = {}, {}
    for method in sorted({r["method"] for r in records}):
        rows = sorted((r for r in records if r["method"] == method), key=lambda r: (r["bench"], r["id"]))
        require(len(rows) == 270 and len({(r["bench"], r["id"]) for r in rows}) == 270, "OOF coverage")
        paired[method] = {b: [r["correct"] for r in rows if r["bench"] == b] for b in datasets}
        scores = {b: statistics.mean(paired[method][b]) for b in datasets}
        costs = {b: {k: sum(r["cost"][k] for r in rows if r["bench"] == b) for k in rows[0]["cost"]}
                 for b in datasets}
        results[method] = {"scores": scores, "macro": statistics.mean(scores.values()), "cost_total_by_bench": costs}
    comparisons = {}
    for method in results:
        if method == "V07":
            continue
        comparison = analysis.bootstrap_difference(paired["V07"], paired[method])
        comparison["by_bench"] = {}
        for b in datasets:
            a, other = paired["V07"][b], paired[method][b]
            differences = [int(x)-int(y) for x, y in zip(a, other, strict=True)]
            rng = random.Random(20261010)
            draws = sorted(100 * sum(rng.choice(differences) for _ in differences) / len(differences)
                           for _ in range(2000))
            comparison["by_bench"][b] = {"repairs": sum(x and not y for x, y in zip(a, other)),
                "breaks": sum(y and not x for x, y in zip(a, other)),
                "difference_pp": 100 * statistics.mean(differences), "ci95_descriptive_pp": [draws[49], draws[1949]]}
        comparisons[method] = comparison
    oracle = {b: statistics.mean(any(d.labels[c] for c in analysis.pool(d, q, MODELS, 5)) for q in d.candidates)
              for b, d in datasets.items()}
    gap = {}
    for b in (*datasets, "macro"):
        o = statistics.mean(oracle.values()) if b == "macro" else oracle[b]
        baseline = results["visible_k5"]["macro"] if b == "macro" else results["visible_k5"]["scores"][b]
        score = results["V07"]["macro"] if b == "macro" else results["V07"]["scores"][b]
        gap[b] = (score-baseline)/(o-baseline) if o != baseline else None
    cells = [cell for event in ledger.done.values() for cell in event["output"]["cells"]]
    segments = [json.loads(line) for line in path("segments.jsonl").read_text().splitlines()]
    smoke_cost = json.loads(path("smoke.json").read_text(encoding="utf-8"))["cost"]
    preparation_segments = [json.loads(line) for line in path("preparation_segments.jsonl").read_text().splitlines()]
    require(all(s["inputs_manifest_sha256"] == sha(manifest) for s in preparation_segments), "preparation costs binding")
    diagnostics = {}
    for b in datasets:
        qs = [q for q in questions if q["bench"] == b]
        diagnostics[b] = {"mutation_counts": dict(Counter(len(q["inputs"]) for q in qs)),
            "no_public_examples": sum(not q["public_examples"] for q in qs),
            "mutation_attempts": sum(q["mutation_attempts"] for q in qs),
            "coverage_by_size": {str(n): {"no_positive_signal_questions": sum(
                not any(v is not None and v > 0 for v in signals[b, q["id"], n].values()) for q in qs),
                "candidate_no_normal_result": sum(v is None for q in qs for v in signals[b, q["id"], n].values())}
                for n in CONFIG["sizes"]}}
    summary = {"status": "complete_dev_exploratory", "config": CONFIG, "results": results,
        "comparisons_V07_minus": comparisons, "oracle_fixed35": oracle, "gap_recovered": gap, "choices": choices,
        "diagnostics": diagnostics, "validation": {b: d.validation for b, d in datasets.items()},
        "cost": {"cpu_child_s": sum(s["child_cpu_s"] for s in segments),
                 "wall_campaign_s": sum(s["wall_s"] for s in segments), "segments": segments,
                 "preparation": preparation_segments, "synthetic_smoke": smoke_cost,
                 "operations": len(ledger.done), "attempts": sum(ledger.attempts.values()),
                 "normal_cells": sum(c["ok"] for c in cells), "total_cells": len(cells),
                 "exception_or_absence_rate": sum(not c["ok"] for c in cells)/len(cells) if cells else None,
                 "statuses": dict(Counter(e["output"]["status"] for e in ledger.done.values())),
                 "errors": dict(Counter(c["error"] for c in cells if not c["ok"])),
                 "new_llm_tokens": 0, "new_authors": 0, "new_judges": 0, "gpu_hours": 0},
        "registry_counts": {"inner_trials": sum(r["type"] == "inner_trial" for r in journal),
                            "decisions": sum(r["type"] == "decision" for r in journal)},
        "bindings": {"inputs": sha(manifest), "execution": sha(execution), "ledger_head": ledger.head}}
    write_once(path("summary.json"), summary)
    write_once(path("registry.json"), journal)
    write_once(path("predictions.json"), records)
    rendered = report(summary)
    target = path("report.md")
    if target.exists():
        require(target.read_text(encoding="utf-8") == rendered, "report changed")
    else:
        target.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, flush=True)


def report(summary):
    lines = ["# V07 : différentiel nouveau, DEV complet", "",
        "270 questions, 35 candidats fixes, 16/64 mutations publiques. CPU Linux WSL seulement. "
        "Accord ≠ correction ; entrées potentiellement hors contrat. Aucun p-value, test consommé : 0/1. "
        "Sept variantes ; alpha futur 0,05/28. Bootstrap descriptif apparié, 2 000 tirages, graine 20261010.", "",
        "| Méthode | HumanEval+ (%) | MBPP+ (%) | Macro (%) |", "|---|---:|---:|---:|"]
    for method, r in summary["results"].items():
        lines.append(f"| {method} | {100*r['scores']['humanevalplus']:.2f} | "
                     f"{100*r['scores']['mbppplus']:.2f} | {100*r['macro']:.2f} |")
    lines += ["", "| Comparateur | Gain macro (pp) | IC descriptif 95 % (pp) |", "|---|---:|---|"]
    for method, r in summary["comparisons_V07_minus"].items():
        lines.append(f"| {method} | {r['difference_pp']:+.2f} | {r['ci95_descriptive_pp']} |")
        for b, detail in r["by_bench"].items():
            lines.append(f"\n{method}, {b} : réparations {detail['repairs']}, cassures {detail['breaks']}, "
                         f"gain {detail['difference_pp']:+.2f} pp, IC {detail['ci95_descriptive_pp']}.")
    lines += ["", f"Oracle fixe 35 : {summary['oracle_fixed35']}. Fraction du gap récupérée : "
              f"{summary['gap_recovered']}.", "", f"Coûts complets CPU et échecs : `{canonical(summary['cost'])}`.",
              "", f"Couverture : `{canonical(summary['diagnostics'])}`.", "",
              "Les coûts de génération historiques (prompt/sortie et paramètres × tokens) figurent par méthode "
              "et banc dans summary.json. Ils incluent tous les candidats sollicités ; CPU nouveau distinct. "
              "L'exploration exécute 64 entrées même si un pli choisit 16 : tout ce coût est compté. "
              "Aucun coût GPU historique non journalisé n'est inventé.", "",
              "Les comparateurs utilisent exactement les fonctions historiques analyze_percees_code. "
              "V07 utilise les seules observations visibles, sans référence ; la baseline historique écarte "
              "les assertions dont la référence échouait. Cette différence est conservée et documentée. "
              "Poids K et choix internes sont appris seulement sur train (5 plis externes/3 internes).", "",
              "Limites : provenance historique sans empreinte officielle indépendante des IDs du split ; "
              "accords exacts via codec typé, sans tolérance flottante ; dépendances entre familles ; "
              "bootstrap descriptif ne réapprend pas les plis. Les exceptions/absences ne votent jamais. "
              "Les entrées dérivent seulement des exemples publics et peuvent manquer ou épuiser leur domaine. "
              "Une campagne interrompue n'est jamais notée ; seules opérations sans résultat terminal sont reprises.", ""]
    return "\n".join(lines)


def main():
    global PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "execute", "analyze"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--attempt", default="original", help="Distinct artifact namespace; original is immutable")
    args = parser.parse_args()
    if args.mode == "prepare" and (args.snapshot is None or args.cache is None):
        parser.error("prepare requires explicit --snapshot and --cache DEV paths")
    if args.mode == "analyze" and args.snapshot is None:
        parser.error("analyze requires an explicit --snapshot DEV path")
    PREFIX = attempt_prefix(args.attempt)
    path("placeholder").parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args.snapshot, args.cache)
    elif args.mode == "analyze":
        analyze(args.snapshot)
    else:
        execute(args.mode == "smoke")


if __name__ == "__main__":
    main()
