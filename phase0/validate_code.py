"""Strict, offline validation of the canonical E11 campaign; never executes candidate or dataset code."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from analyze_e4 import EXTRA, FAMILIES, REFS
from colab_jobs import SOLO
from essaim import code, data

MODELS = list(FAMILIES.values()) + EXTRA + REFS
DATA_HASHES = {"humanevalplus": "786d49ec2612b2b7fee5b4e964c2f84622edad0a9039b851bd5e75ff54832231",
               "mbppplus": "489c1c0a334ba129144e360490c3e9e7ed4eef7b70501a9aa5c5d740fe8227ea"}
IMPLEMENTATION = ("analyze_code.py", "validate_code.py", "e11_costs.py", "analyze_e4.py", "colab_jobs.py",
                  "essaim/code.py", "essaim/data.py", "essaim/stats.py", "essaim/sandbox.py", "essaim/results.py",
                  "essaim/models.py", "essaim/sot.py", "run_solo.py", "run_code.py", "exec_code.py", "uv.lock")


def fail(path, message):
    raise SystemExit(f"E11 : {Path(path).name} : {message}")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path, key):
    if not path.is_file():
        fail(path, "source absente")
    try:
        result = [json.loads(line) for line in path.read_bytes().decode("utf-8").splitlines()]
        keys = [tuple(r[k] for k in key) for r in result]
    except (ValueError, KeyError, TypeError, UnicodeError):
        fail(path, "source corrompue ou tronquée")
    if len(set(keys)) != len(keys):
        fail(path, "clés dupliquées")
    return result


def manifest(path):
    meta = path.with_name(path.name + ".meta.json")
    if not meta.is_file():
        fail(meta, "manifeste absent")
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        fail(meta, "manifeste corrompu")
    if not isinstance(m, dict):
        fail(meta, "manifeste invalide")
    return m


def require(path, actual, expected):
    for k, v in expected.items():
        if actual.get(k) != v:
            fail(path, f"{k} divergent")


def cache(bench):
    """Read an existing cache only: a missing cache must never trigger a download."""
    path = data.DATA / f"{bench}_all_s0_v2.jsonl"
    meta = path.with_suffix(".meta.json")
    if not path.is_file() or not meta.is_file():
        fail(path, "cache offline absent")
    m = json.loads(meta.read_text(encoding="utf-8"))
    require(meta, m, {"dataset": data.REPOS[bench], "revision": data.REVISIONS[bench],
                      "license": data.LICENSES[bench], "sha256": digest(path)})
    if m["sha256"] != DATA_HASHES[bench]:
        fail(meta, "empreinte du jeu canonique divergente")
    items = rows(path, ("id",))
    if len(items) != 2 * data.PART[bench]:
        fail(path, "couverture du cache divergente")
    if bench == "humanevalplus" and {x["id"] for x in items} != {f"HumanEval/{i}" for i in range(164)}:
        fail(path, "IDs HumanEval divergents")
    return items, m


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def hidden_count(bench, item):
    """Count literal stored cases from the AST, without executing even trusted dataset source."""
    spec = code.hidden_spec(bench, item)
    assignments = [ast.parse(s).body[0] for s in spec["setup"]]
    lengths = {n.targets[0].id: len(n.value.elts) for n in assignments
               if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.List)}
    if len(lengths) != len(assignments) or "inputs" not in lengths:
        fail(item["id"], "cas cachés non littéraux")
    if spec["iter"] == "enumerate(zip(inputs, results))" and lengths.get("results") == lengths["inputs"]:
        return lengths["inputs"]
    if spec["iter"] == "enumerate(inputs)":
        return lengths["inputs"]
    fail(item["id"], "boucle de cas cachés divergente")


def validate_partition(bench, split, items, identity, suffix, tag, models=MODELS):
    """Exact generation/execution coverage, including problems later excluded by the reference oracle."""
    expected = {(it["id"], s) for it in items for s in range(5)}
    wanted = {(it["id"], "reference") for it in items}
    by_id = {it["id"]: it for it in items}
    hashes, mans, full_sources = {}, {}, {}
    counts = Counter()
    for model in models:
        path = code.gen_path(model, suffix, bench, split)
        man = manifest(path)
        if model in SOLO:
            require(path, man, {"gguf": SOLO[model][1]})
        require(path, man, {"model": model, "bench": bench, "split": split, "n": len(items), "data": identity,
                            "prompt": code.PROMPT_VERSION, "max_tokens": code.MAX_TOKENS, "thinking": False,
                            "ctx_per_slot": 3072, "samples": 4,
                            "greedy": {"temperature": 0.0, "seed": code.GREEDY_SEED},
                            "sampling": {**code.SAMPLING, "seeds": [code.sample_seed(s) for s in range(1, 5)]}})
        weights = man.get("weights", {})
        if (not re.fullmatch(r"[0-9a-f]{40}", man.get("revision", ""))
                or not re.fullmatch(r"[0-9a-f]{64}", weights.get("sha256", ""))
                or not finite_number(weights.get("size")) or weights["size"] == 0
                or weights.get("name") != man.get("gguf")
                or not man.get("engine") or man.get("backend") != "llama.cpp (--jinja, OpenAI chat)"):
            fail(path, "identité modèle invalide")
        gs = rows(path, ("id", "sample"))
        if {(r["id"], r["sample"]) for r in gs} != expected:
            fail(path, "couverture IDs/samples divergente")
        for i, g in enumerate(gs, 1):
            require(path, g, {"model": model, "bench": bench, "seed": code.sample_seed(g["sample"]),
                              "temperature": 0.0 if g["sample"] == 0 else code.SAMPLING["temperature"]})
            if type(g["sample"]) is not int or not isinstance(g.get("text"), str):
                fail(path, f"ligne {i} : forme génération invalide")
            src, status = code.extract_code(g["text"], code.entry_point(bench, by_id[g["id"]]))
            require(path, g, {"code": src, "extract": status, "prog": code.prog_id(src)})
            key = (g["id"], g["prog"])
            if key in full_sources and full_sources[key] != src:
                fail(path, "collision du hash de programme")
            full_sources[key] = src
            wanted.add(key)
            if (g.get("finish") not in ("stop", "length", None)
                    or type(g.get("reasoning")) is not bool or g["reasoning"]):
                fail(path, f"ligne {i} : finish/reasoning invalide")
            for field in ("n_tokens", "prompt_tokens", "ms"):
                if g.get(field) is not None and not finite_number(g[field]):
                    fail(path, f"ligne {i} : {field} invalide")
                counts[f"missing_{field}"] += g.get(field) is None
            counts["generations"] += 1
            counts["length"] += g.get("finish") == "length"
            counts["missing_finish"] += g.get("finish") is None
        mans[model] = man
        for p in (path, path.with_name(path.name + ".meta.json")):
            hashes[p.name] = digest(p)
    path = code.exec_path(tag, suffix, bench, split)
    em = manifest(path)
    require(path, em, {"bench": bench, "split": split, "n": len(items), "data": identity,
                       "tests": code.TESTS_VERSION, "extract": code.EXTRACT_VERSION, "extra_n": 16,
                       "models": sorted(models), "reference_only": False})
    iso = em.get("isolation", {})
    require(path, iso, {"version": "sandbox-v3", "platform": "linux", "landlock": "unavailable",
                        "prefix": ["/usr/bin/unshare", "--net", "--"], "per_case_timeout": True})
    if not em.get("python") or not em.get("numpy"):
        fail(path, "versions d'exécution absentes")
    rs = rows(path, ("id", "prog"))
    if {(r["id"], r["prog"]) for r in rs} != wanted:
        fail(path, "couverture exec divergente (référence/programme absent ou étranger)")
    refs = {r["id"]: r for r in rs if r["prog"] == "reference"}
    sizes = {q: (len(code.visible_tests(bench, it)), len(code.extra_inputs(bench, it, em["extra_n"])),
                 hidden_count(bench, it))
             for q, it in by_id.items()}
    statuses = Counter()
    for i, r in enumerate(rs, 1):
        it, h = by_id[r["id"]], r.get("hidden", {})
        nv, nx, nh = sizes[r["id"]]
        if (r.get("bench") != bench or not finite_number(r.get("ms"))
                or r.get("load") is not None and not isinstance(r["load"], str)
                or len(r.get("visible", [])) != nv or any(type(v) is not bool for v in r["visible"])
                or len(r.get("visible_err", [])) != nv
                or any(e is not None and not isinstance(e, str) for e in r["visible_err"])
                or any(ok != (err is None) for ok, err in zip(r["visible"], r["visible_err"]))
                or len(r.get("extra", [])) != nx
                or any(not isinstance(s, str) or not re.fullmatch(r"!|[0-9a-f]{16}|D[0-9a-f]{15}", s) for s in r["extra"])
                or type(h.get("pass")) is not bool
                or type(h.get("n")) is not int or type(h.get("n_pass")) is not int
                or not 0 <= h["n_pass"] <= h["n"] or h["n"] < 1
                or h["n"] != nh
                or h.get("err") is not None and not isinstance(h["err"], str)
                or h["pass"] and (h["n_pass"] != h["n"] or h.get("err") is not None)
                or set(r.get("status", {})) != {"visible", "extra", "hidden"}
                or any(s not in ("ok", "timeout", "overflow", "crash") for s in r["status"].values())
                or h["pass"] and r["status"]["hidden"] != "ok"):
            fail(path, f"ligne {i} : forme/statut exec invalide")
        for phase, status in r["status"].items():
            statuses[f"{phase}|{status}"] += 1
    for p in (path, path.with_name(path.name + ".meta.json")):
        hashes[p.name] = digest(p)
    excluded = {q: r["hidden"] for q, r in refs.items() if not r["hidden"]["pass"]}
    rejected = {q: r["visible_err"] for q, r in refs.items() if not all(r["visible"])}
    return {"ids": list(by_id), "generations": dict(counts), "exec_rows": len(rs), "statuses": dict(statuses),
            "excluded": excluded, "visible_rejected": rejected, "manifests": mans, "exec": em}, hashes


def campaign(suffix="_colab", tag="e11"):
    partitions, hashes, identities = {}, {}, {}
    for bench in code.BENCHES:
        items, identity = cache(bench)
        identities[bench] = identity
        for split in data.SPLITS:
            subset = data._split(items, data.PART[bench], split, bench)
            result, sources = validate_partition(bench, split, subset, identity, suffix, tag)
            partitions[f"{bench}|{split}"] = result
            hashes.update(sources)
    # All four partitions use the same weights, engine and generation/execution protocol.
    first = next(iter(partitions.values()))
    for p in partitions.values():
        for m in MODELS:
            require(m, p["manifests"][m], {k: first["manifests"][m][k]
                                          for k in ("revision", "gguf", "weights", "engine", "backend")})
        require("exec", p["exec"], {k: first["exec"][k] for k in ("isolation", "python", "numpy")})
    root = Path(__file__).resolve().parent
    return {"version": "e11-validation-v1", "complete": True, "suffix": suffix, "tag": tag,
            "models": MODELS, "datasets": identities, "partitions": partitions, "sources_sha256": hashes,
            "implementation_sha256": {name: digest(root / name) for name in IMPLEMENTATION}}


def verify_provenance(summary, root):
    """The paper requires every canonical source and the implementation that produced the summary."""
    v = summary.get("validation", {})
    expected = set()
    for bench in code.BENCHES:
        for split in data.SPLITS:
            names = [code.gen_path(m, "_colab", bench, split).name for m in MODELS]
            names += [code.exec_path("e11", "_colab", bench, split).name]
            expected.update(names + [n + ".meta.json" for n in names])
    if (not v.get("complete") or v.get("version") != "e11-validation-v1" or v.get("models") != MODELS
            or v.get("suffix") != "_colab" or v.get("tag") != "e11"
            or set(v.get("sources_sha256", {})) != expected
            or set(v.get("implementation_sha256", {})) != set(IMPLEMENTATION)):
        fail("summary", "provenance canonique incomplète")
    for directory, entries in ((root / "results", v["sources_sha256"]), (root, v["implementation_sha256"])):
        for name, sha in entries.items():
            path = directory / name
            if not path.is_file() or digest(path) != sha:
                fail(path, "source absente ou modifiée")
