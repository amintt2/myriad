"""Strict offline E12 gate. The parent's hash-pinned delivery is the external trust anchor."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

import aa_timing
from analyze_e4 import EXTRA, FAMILIES, REFS
from colab_jobs import SOLO
from essaim import answers, data, gpqa, scicode
from exec_scicode import HARNESS_VERSION, gen_digest, jobs_for_model, jobs_for_oracle
from run_aa import CTX_PER_SLOT, MAX_TOKENS, MEASUREMENTS, SEED, MC_PROMPT

MODELS = list(FAMILIES.values()) + EXTRA + REFS
IDENTITY = ("model", "revision", "gguf", "weights", "engine", "backend")
RUNTIME = ("run_aa.py", "exec_scicode.py", "aa_timing.py", "colab_jobs.py", "run_solo.py",
           "essaim/scicode.py", "essaim/sandbox.py", "essaim/data.py", "essaim/code.py", "essaim/sot.py",
           "essaim/results.py", "essaim/llamacpp.py", "essaim/common.py", "pyproject.toml", "uv.lock")
GPQA_RUNTIME = ("essaim/gpqa.py", "essaim/answers.py", "analyze_e4.py")


def fail(message):
    raise SystemExit(f"E12 : {message}")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def decode(raw):
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def read(path, lines=False):
    try:
        raw = Path(path).read_bytes()
        value = [decode(line) for line in raw.splitlines()] if lines else decode(raw)
        if lines and (not raw.endswith(b"\n") or not value or any(type(r) is not dict for r in value)):
            raise ValueError("incomplete JSONL")
        return value
    except (OSError, ValueError, UnicodeError):
        fail(f"{Path(path).name} absent, invalide ou tronqué")


def equal(actual, expected, where):
    # JSON equality alone accepts True == 1, including inside nested protocol objects.
    if type(actual) is not type(expected):
        fail(f"{where} : type divergent")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            fail(f"{where} : clés divergentes")
        for key in expected:
            equal(actual[key], expected[key], f"{where}/{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            fail(f"{where} : longueur divergente")
        for a, b in zip(actual, expected):
            equal(a, b, where)
    elif actual != expected:
        fail(f"{where} : valeur divergente")


def require(actual, expected, where):
    if type(actual) is not dict:
        fail(f"{where} : objet attendu")
    for key, value in expected.items():
        if key not in actual:
            fail(f"{where}/{key} absent")
        equal(actual[key], value, f"{where}/{key}")


def number(value, nullable=False, integer=False):
    try:
        return (nullable and value is None) or (type(value) in ((int,) if integer else (int, float))
                                                and math.isfinite(value) and value >= 0)
    except OverflowError:
        return False


def hash_value(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def relative(name):
    if type(name) is not str or not name:
        return False
    p = PurePosixPath(name)
    return type(name) is str and not p.is_absolute() and ".." not in p.parts and ":" not in name and "\\" not in name


def indexed(path, expected):
    rows = read(path, lines=True)
    if any(type(r.get("id")) is not str for r in rows):
        fail(f"{path.name} : IDs non textuels")
    out = {r["id"]: r for r in rows}
    if len(out) != len(rows) or set(out) != set(expected):
        fail(f"{path.name} : doublons ou couverture non canonique")
    return out


def fields(row, required, optional=()):
    if not set(required) <= set(row) or set(row) - set(required) - set(optional):
        fail("champs JSON absents ou étrangers")


def official(dataset_dir):
    """Pinned raw files, not an unchecked cache; only parsing and script assembly, never exec."""
    probs = {}
    all_problems = []
    for split, name in scicode.FILES.items():
        path = Path(dataset_dir) / name
        raw = path.read_bytes()
        if scicode.git_blob_sha1(raw) != scicode.BLOB_SHA1[name]:
            fail(f"{name} : source officielle divergente")
        probs[split] = [scicode._problem(r, split) for r in read(path, lines=True)]
        if (len(probs[split]), sum(len(p["steps"]) for p in probs[split])) != scicode.COUNTS[split]:
            fail(f"{split} : taille non canonique")
        all_problems += probs[split]
    ids = [{p["id"] for p in probs[sp]} for sp in ("dev", "test")]
    if ids[0] & ids[1]:
        fail("dev/test non disjoints")
    ident = {"dataset": data.REPOS["scicode"], "revision": data.REVISIONS["scicode"],
             "license": data.LICENSES["scicode"], "sha256": sha(data._canonical(all_problems))}
    return probs, ident


def timing(path, manifest, rows, problems):
    trace = read(Path(str(path) + ".timing.jsonl"), lines=True)
    equal(trace[0], aa_timing.header(path, manifest), "timing header")
    calls = {}
    for row in trace[1:]:
        if not number(row.get("wall_s")):
            fail("timing : durée invalide")
        kind = row.get("kind")
        if kind == "call":
            fields(row, ("kind", "bench", "problem", "id", "returned", "error_type", "n_tokens",
                         "prompt_tokens", "wall_s"))
            rid = row.get("id")
            if type(rid) is not str or rid not in rows:
                fail("timing : appel étranger")
            require(row, {"bench": "scicode", "problem": rows[rid]["problem"]}, "timing call")
            if type(row.get("returned")) is not bool or not all(number(row.get(k), True, True)
                                                                for k in ("n_tokens", "prompt_tokens")):
                fail("timing : compteurs/types invalides")
            error = row.get("error_type")
            if error is not None and (type(error) is not str or not error):
                fail("timing : erreur invalide")
            if row["returned"] != (error is None):
                fail("timing : statut incohérent")
            calls.setdefault(rid, []).append(row)
        elif kind == "problem":
            fields(row, ("kind", "problem", "complete", "resumed_steps", "wall_s"))
            if row.get("problem") not in problems or type(row.get("complete")) is not bool or \
                    not number(row.get("resumed_steps"), integer=True):
                fail("timing : problème invalide")
        elif kind == "batch":
            fields(row, ("kind", "bench", "split", "new", "wall_s"))
            require(row, {"bench": "scicode", "split": manifest["split"]}, "timing batch")
            if not number(row.get("new"), integer=True):
                fail("timing : batch invalide")
        else:
            fail("timing : événement inconnu")
    if set(calls) != set(rows) or not any(r["kind"] == "batch" for r in trace[1:]):
        fail("timing : couverture incomplète")
    for rid, row in rows.items():
        matches = [c for c in calls[rid] if c["returned"] == (row["finish"] != "error") and
                   c["n_tokens"] == row["n_tokens"] and c["prompt_tokens"] == row["prompt_tokens"]]
        if not matches or (row["finish"] != "error" and not any(c["wall_s"] == row["wall_s"] for c in matches)):
            fail("timing : réponse non liée à un appel")
    return trace[1:]


def identity(ident, model):
    require(ident, {"model": model, "gguf": SOLO[model][1],
                    "backend": "llama.cpp (--jinja, OpenAI chat)"}, model)
    if set(ident) != set(IDENTITY) or type(ident.get("revision")) is not str or \
            not re.fullmatch(r"[0-9a-f]{40}", ident["revision"]) or \
            type(ident.get("engine")) is not str or not ident["engine"]:
        fail("identité modèle invalide")
    weight = ident.get("weights")
    if type(weight) is not dict or set(weight) != {"name", "size", "sha256"} or \
            weight["name"] != SOLO[model][1] or not number(weight["size"], integer=True) or \
            not weight["size"] or not hash_value(weight["sha256"]):
        fail("identité des poids invalide")


def validate_gpqa(root, suffix, inventory, identities, csv_path):
    # Parse only an already admitted local copy; never call the downloader or unchecked cache.
    if csv_path is None:
        fail("CSV GPQA local admis requis")
    raw = Path(csv_path).read_bytes()
    gpqa.check_file(raw)
    items = gpqa.parse(raw)
    if len(items) != gpqa.N_QUESTIONS or len({x["id"] for x in items}) != gpqa.N_QUESTIONS:
        fail("couverture GPQA non canonique")
    items = {x["id"]: x for x in items}
    data_ident = {"dataset": data.REPOS["gpqa"], "revision": data.REVISIONS["gpqa"],
                  "license": data.LICENSES["gpqa"], "sha256": sha(data._canonical(list(items.values())))}
    frozen_name = f"e4_summary{suffix}.json"
    if frozen_name not in inventory:
        fail("poids et meilleur pair E4 dev non épinglés")
    frozen = read(root / frozen_name)["benches"]["mmlupro"]
    for key in ("weights", "p_dev"):
        values = frozen[key]
        if type(values) is not dict or set(values) != set(FAMILIES) or \
                any(not number(v) or (key == "p_dev" and v > 1) for v in values.values()):
            fail("paramètres E4 dev invalides")
    best = FAMILIES[max(FAMILIES, key=lambda f: frozen["p_dev"][f])]
    equal(frozen["best_dev"], best, "meilleur pair E4 dev")
    loaded = {}
    for model in MODELS:
        name = f"aa_{model.replace('/', '__')}{suffix}_gpqa_all.jsonl"
        for part in (name, name + ".meta.json", name + ".timing.jsonl"):
            if part not in inventory:
                fail("livraison GPQA incomplète (références comprises)")
        path = root / name
        man = read(root / (name + ".meta.json"))
        require(man, {**identities[model], "bench": "gpqa", "split": "all", "data": data_ident,
                      "n": gpqa.N_QUESTIONS, "prompt": MC_PROMPT, "max_tokens": MAX_TOKENS["gpqa"],
                      "temperature": 0.0, "seed": SEED, "thinking": False, "ctx_per_slot": CTX_PER_SLOT,
                      "logprobs": False, "measurements": MEASUREMENTS}, name)
        rows = indexed(path, items)
        for rid, row in rows.items():
            fields(row, ("id", "model", "bench", "text", "answer", "gold", "finish", "n_tokens",
                         "ms", "wall_s", "reasoning"))
            require(row, {"model": model, "bench": "gpqa"}, "GPQA row")
            if type(row["text"]) is not str or type(row["reasoning"]) is not bool or \
                    row["finish"] not in ("stop", "length") or not number(row["wall_s"]) or \
                    not number(row["n_tokens"], True, True) or not number(row["ms"], True):
                fail("types ou statut GPQA invalides")
            equal(row["answer"], answers.extract("gpqa", row["text"], items[rid], row["finish"] == "stop"),
                  "GPQA answer")
            equal(row["gold"], answers.gold("gpqa", items[rid]), "GPQA gold")
        trace = read(root / (name + ".timing.jsonl"), lines=True)
        equal(trace[0], aa_timing.header(path, man), "GPQA timing header")
        calls, batches = {}, []
        for event in trace[1:]:
            if not number(event.get("wall_s")):
                fail("durée GPQA invalide")
            if event.get("kind") == "batch":
                fields(event, ("kind", "bench", "new", "wall_s"))
                require(event, {"bench": "gpqa"}, "GPQA batch")
                if not number(event["new"], integer=True):
                    fail("batch GPQA invalide")
                batches.append(event)
            else:
                fields(event, ("kind", "bench", "id", "returned", "error_type", "n_tokens",
                               "prompt_tokens", "wall_s"))
                require(event, {"kind": "call", "bench": "gpqa"}, "GPQA call")
                if type(event["id"]) is not str or event["id"] not in rows or \
                        type(event["returned"]) is not bool or \
                        any(not number(event[k], True, True) for k in ("n_tokens", "prompt_tokens")) or \
                        (event["error_type"] is not None and
                         (type(event["error_type"]) is not str or not event["error_type"])) or \
                        event["returned"] != (event["error_type"] is None):
                    fail("appel GPQA invalide")
                calls.setdefault(event["id"], []).append(event)
        if set(calls) != set(rows) or not batches:
            fail("timing GPQA incomplet")
        for rid, row in rows.items():
            if not any(c["returned"] and c["n_tokens"] == row["n_tokens"] and
                       c["wall_s"] == row["wall_s"] for c in calls[rid]):
                fail("réponse GPQA non liée au timing")
        loaded[model] = rows, man
    return {"items": items, "loaded": loaded, "frozen": {
        "weights": frozen["weights"], "p_dev": frozen["p_dev"], "best": best}}


def _validate(root, suffix, delivery_path, delivery_sha256, dataset_dir, selected, gpqa_path):
    if not delivery_path or not hash_value(delivery_sha256):
        fail("livraison complète explicite du parent et SHA-256 requis")
    root = Path(root)
    if sha(Path(delivery_path).read_bytes()) != delivery_sha256:
        fail("livraison parent modifiée")
    delivery = read(delivery_path)
    require(delivery, {"schema": 1, "complete": True, "plan": "aa-1"}, "delivery")
    benches = delivery.get("benches")
    if type(benches) is not list or any(type(b) is not str for b in benches) or \
            benches not in (["scicode"], ["gpqa", "scicode"]):
        fail("disponibilité des bancs non canonique")
    if type(selected) is not list or not selected or any(type(b) is not str for b in selected) or \
            len(set(selected)) != len(selected) or not set(selected) <= set(benches):
        fail("sélection requise : sous-ensemble non vide, sans doublons, des bancs attestés")
    # Validate every attested benchmark, including those hidden by the report filter.
    selected = [b for b in benches if b in selected]
    attempts = delivery.get("attempts")
    if type(attempts) is not list or not attempts:
        fail("provenance des tentatives absente")
    seen = set()
    for attempt in attempts:
        cid = attempt.get("campaign_id")
        if type(cid) is not str or not re.fullmatch(r"[0-9a-f]{32}", cid) or cid in seen:
            fail("identité de tentative invalide")
        seen.add(cid)
        if type(attempt.get("interrupted")) is not bool or type(attempt.get("resumed")) is not bool:
            fail("statut de tentative invalide")
        evidence = attempt.get("evidence")
        if type(evidence) is not dict or not evidence or any(not relative(k) or not hash_value(v)
                                                          for k, v in evidence.items()):
            fail("empreintes immuables des tentatives requises (noms relatifs)")
    inventory = delivery.get("files")
    if type(inventory) is not dict or not inventory:
        fail("inventaire immuable absent")
    for name, digest in inventory.items():
        if not relative(name) or not hash_value(digest) or sha((root / name).read_bytes()) != digest:
            fail("inventaire de livraison divergent")
    if any(("_gpqa_" in name and "gpqa" not in benches) or
           ("_scicode_" in name and "scicode" not in benches) or
           name == f"sciexec_oracle{suffix}_scicode_test.jsonl" for name in inventory):
        fail("banc étranger à la livraison ou oracle test interdit")
    for name in ("e12_campaign_sources.json", "e12_campaign_sources.local.json"):
        if name not in inventory:
            fail("provenance upload/récupération non épinglée")
    source = read(root / "e12_campaign_sources.json")
    equal(source, read(root / "e12_campaign_sources.local.json"), "sources upload/récupération")
    require(source, {"schema": 1, "plan": "aa-1", "benches": benches}, "sources")
    sources = source.get("sources")
    if type(sources) is not dict or any(not relative(k) for k in sources):
        fail("sources invalides")
    for value in sources.values():
        if type(value) is not dict or set(value) != {"size", "sha256"} or \
                not number(value["size"], integer=True) or not hash_value(value["sha256"]):
            fail("empreinte source invalide")
    for name in RUNTIME + (GPQA_RUNTIME if "gpqa" in benches else ()):
        path = Path(__file__).parent / name
        require(sources.get(name), {"size": path.stat().st_size, "sha256": sha(path.read_bytes())}, name)
    identities = delivery.get("identities")
    if type(identities) is not dict or set(identities) != set(MODELS):
        fail("treize identités canoniques requises")
    for model in MODELS:
        identity(identities[model], model)
    common = {"delivery_sha256": delivery_sha256, "attempts": attempts,
              "source_sha256": sha((root / "e12_campaign_sources.json").read_bytes()),
              "available_benches": benches, "report_benches": selected}
    if "gpqa" in benches:
        common["gpqa"] = validate_gpqa(root, suffix, inventory, identities, gpqa_path)
    if "scicode" not in benches:
        return common
    probs, data_ident = official(dataset_dir)
    environment = delivery.get("execution_environment")
    if type(environment) is not dict or set(environment) != {"isolation", "python", "numpy", "scipy"} or \
            type(environment["isolation"]) is not dict or not environment["isolation"] or \
            any(type(environment[k]) is not str or not environment[k] for k in ("python", "numpy", "scipy")):
        fail("environnement de notation absent ou invalide")
    loaded, traces, oracle = {"dev": {}, "test": {}}, {"dev": {}, "test": {}}, {}

    def artifact(name):
        for part in (name, name + ".meta.json"):
            if part not in inventory:
                fail(f"{part} absent de la livraison")
        return root / name, read(root / (name + ".meta.json"))

    for split in ("dev", "test"):
        need = {scicode.row_id(s): p["id"] for p in probs[split] for s in p["steps"] if not scicode.is_skipped(p, s)}
        for model in MODELS + (["oracle"] if split == "dev" else []):
            gen, gman = None, None
            if model != "oracle":
                ident = identities[model]
                path, gman = artifact(f"aa_{model.replace('/', '__')}{suffix}_scicode_{split}.jsonl")
                require(gman, {**ident, "bench": "scicode", "split": split, "data": data_ident,
                               "n": len(probs[split]), "n_steps": scicode.COUNTS[split][1],
                               "prompt": scicode.PROMPT_VERSION, "max_tokens": MAX_TOKENS["scicode"],
                               "temperature": 0.0, "seed": SEED, "thinking": False, "ctx_per_slot": CTX_PER_SLOT,
                               "background": True, "official_commit": scicode.OFFICIAL_COMMIT,
                               "skipped_steps": sorted(scicode.SKIPPED_SHA256), "skipped_code": scicode.SKIPPED_SHA256,
                               "measurements": MEASUREMENTS}, path.name)
                gen = indexed(path, need)
                steps = {scicode.row_id(s): s for p in probs[split] for s in p["steps"]}
                for rid, row in gen.items():
                    fields(row, ("id", "problem", "step", "model", "bench", "text", "extract", "finish",
                                 "n_tokens", "prompt_tokens", "ms", "wall_s", "reasoning"), ("error",))
                    require(row, {"model": model, "bench": "scicode", "problem": need[rid],
                                  "step": scicode.step_number(steps[rid])}, rid)
                    if type(row.get("text")) is not str or type(row.get("reasoning")) is not bool or \
                            row.get("finish") not in ("stop", "length", "error") or not number(row.get("wall_s")) or \
                            not all(number(row.get(k), True, k != "ms") for k in ("n_tokens", "prompt_tokens", "ms")):
                        fail("génération : types ou statut invalides")
                    equal(row.get("extract"), scicode.extract(row["text"], scicode.def_name(steps[rid]["header"]))[1], rid)
                    if "error" in row and (type(row["error"]) is not str or row["finish"] != "error"):
                        fail("génération : erreur invalide")
                    if row["finish"] == "error" and (row["text"] or any(row[k] is not None for k in
                                                                        ("n_tokens", "prompt_tokens", "ms"))):
                        fail("génération : erreur incohérente")
                if path.name + ".timing.jsonl" not in inventory:
                    fail("timing non épinglé")
                traces[split][model] = timing(path, gman, gen, {p["id"] for p in probs[split]})
            path, man = artifact(f"sciexec_{model.replace('/', '__')}{suffix}_scicode_{split}.jsonl")
            require(man, {**environment, "bench": "scicode", "split": split, "model": model,
                          "n": len(need), "data": data_ident,
                          "harness": HARNESS_VERSION, "timeout_s": scicode.DEFAULT_TIMEOUT_S,
                          "official_commit": scicode.OFFICIAL_COMMIT, "skipped_code": scicode.SKIPPED_SHA256,
                          "oracle": model == "oracle", "generation": {**gman, "rows_sha256": gen_digest(gen)}
                          if gman else None}, path.name)
            h5 = man.get("h5")
            if type(h5) is not dict or h5.get("sha256") != scicode.H5_SHA256 or \
                    type(h5.get("size")) is not int or h5["size"] != 1049345865:
                fail("cibles HDF5 divergentes")
            rows = indexed(path, need)
            jobs = jobs_for_model(probs[split], gen) if gen is not None else jobs_for_oracle(probs[split])
            for rid, pid, script in jobs:
                row = rows[rid]
                fields(row, ("id", "problem", "model", "ok", "err", "msg", "status", "ms"), ("script",))
                require(row, {"problem": pid, "model": model}, rid)
                if type(row.get("ok")) is not bool or \
                        row.get("status") not in ("ok", "none", "timeout", "overflow", "crash") or \
                        not number(row.get("ms")) or type(row.get("msg")) is not str:
                    fail("notation : types/statut invalides")
                if script is None:
                    require(row, {"ok": False, "status": "none", "err": "NoCode", "ms": 0}, rid)
                    if "script" in row:
                        fail("NoCode avec programme")
                else:
                    equal(row.get("script"), sha(script.encode())[:16], rid)
                    if row["status"] == "none" or (row["ok"] and (row["status"] != "ok" or row.get("err") is not None)) or \
                            (not row["ok"] and (type(row.get("err")) is not str or not row["err"])):
                        fail("notation incohérente")
            value = {"rows": rows, "man": man, "generation_rows": gen}
            if model == "oracle":
                if sum(r["ok"] for r in rows.values()) / len(rows) < 0.9:
                    fail("validité du harnais dev inférieure à 90 %")
                oracle[split] = value
            else:
                loaded[split][model] = value
    return {"data_by": loaded, "oracle": oracle, "probs": probs, "traces": traces,
            **common}


def validate(root, suffix, delivery_path, delivery_sha256, dataset_dir, benches=None, gpqa_path=None):
    try:
        benches = ["scicode"] if benches is None else benches
        return _validate(root, suffix, delivery_path, delivery_sha256, dataset_dir, benches, gpqa_path)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        fail("livraison invalide, incomplète ou non canonique")
