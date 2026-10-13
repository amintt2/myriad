"""Dev-only CPU replay of PERC-20261010-v1. No benchmark loaders or inference are called.

Run with the existing phase0 Python environment: python phase0/analyze_percees.py --split dev.
All learned quantities are fitted inside the training partition, including each inner candidate's weights.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_e4 import BENCHES, EXTRA, FAMILIES, PROTOCOL, REFS
from essaim import answers, models, sot

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE = Path("C:/Decentralised LLM/phase0/data")
HF = Path.home() / ".cache/huggingface/hub"
SEED, BOOT = 20261010, 2000
SOLO = "Qwen/Qwen3.5-4B"
JUDGE = "Qwen/Qwen3.8-27B"
FINGERPRINTS: dict[str, str] = {}
VALIDATION: list[dict] = []


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fold(identifier: str, internal: bool = False) -> int:
    prefix, count = ("percees-inner-v1:", 3) if internal else ("percees-v1:", 5)
    return int(sha((prefix + identifier).encode()), 16) % count


def dev_path(path: Path) -> Path:
    """Reject non-dev response paths before opening either responses or manifests."""
    path = path.resolve()
    base = path.name.removesuffix(".meta.json")
    if path.parent != RESULTS.resolve() or not base.endswith("_dev.jsonl"):
        raise ValueError(f"Lecture non-dev interdite : {path}")
    return path


def fingerprint(path: Path) -> bytes:
    raw = path.read_bytes()
    FINGERPRINTS[str(path)] = sha(raw)
    return raw


def read_dev(path: Path, key: tuple[str, ...] = ("id",)) -> tuple[dict, list[dict]]:
    path = dev_path(path)
    man = json.loads(fingerprint(path.with_name(path.name + ".meta.json")))
    if man.get("split") != "dev" or not isinstance(man.get("n"), int) or man["n"] <= 0:
        raise ValueError(f"Manifeste dev invalide : {path.name}")
    rows = [json.loads(line) for line in fingerprint(path).decode("utf-8").splitlines()]
    keys = [tuple(r[k] for k in key) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Doublons : {path.name}")
    return man, rows


def cache_dev(bench: str, n: int, identity: dict) -> dict[str, dict] | None:
    """Read only the dev prefix of the local seeded cache, never its held-out suffix.

    The complete-cache hash is attested by its metadata, not recomputed by reading non-dev questions.
    The bytes actually consumed receive their own hash, distinct from that attestation.
    """
    limits = {"gsm8k": 300, "math500": 250, "arc": 300, "mmlupro": 300}
    if bench not in limits or not 0 < n <= limits[bench]:
        raise ValueError("Taille hors partition dev")
    for root in (HERE / "data", CACHE):
        path = root / f"{bench}_all_s0_v2.jsonl"
        meta = root / f"{bench}_all_s0_v2.meta.json"
        if not path.exists() or not meta.exists():
            continue
        if json.loads(fingerprint(meta)) != identity:
            raise ValueError(f"Identité du cache incompatible : {meta}")
        with path.open("rb") as stream:
            raw = b"".join(next(stream) for _ in range(n))
        FINGERPRINTS[str(path) + f"#dev-prefix-{n}"] = sha(raw)
        items = [json.loads(line) for line in raw.decode().splitlines()]
        if len({r["id"] for r in items}) != n:
            raise ValueError("Doublons dans le cache dev")
        return {r["id"]: r for r in items}
    return None


def validate_ids(rows: dict, items: dict | None, n: int) -> None:
    if len(rows) != n or (items is not None and set(rows) != set(items)):
        raise ValueError("Identifiants dev incomplets, étrangers ou surnuméraires")


def freeze_math_golds(items: dict, identity: dict) -> None:
    """Keep raw dev golds as an independent, versioned fallback; never use predictions as labels."""
    rows = [{"id": i, "answer": r["answer"]} for i, r in items.items()]
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    prefixes = {k: v for k, v in FINGERPRINTS.items() if k.endswith(f"math500_all_s0_v2.jsonl#dev-prefix-{len(rows)}")}
    if not prefixes:
        raise ValueError("MATH : aucune empreinte du préfixe dev source à conserver")
    write_json(RESULTS / "percees_math_gold_dev.json", {"split": "dev", "n": len(rows), "data": identity,
                                                       "source_dev_prefix_sha256": prefixes,
                                                       "gold_rows_sha256": sha(canonical), "rows": rows})


def load_math_golds(n: int, identity: dict) -> dict:
    """Without questions, use only the frozen raw gold source and the strict boxed grader."""
    path = RESULTS / "percees_math_gold_dev.json"
    if not path.exists():
        raise ValueError("MATH : cache dev et source de golds bruts corrigés absents ; scoring refusé")
    source = json.loads(fingerprint(path))
    rows = source["rows"]
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if (source.get("split") != "dev" or source.get("data") != identity or source.get("n") != len(rows)
            or not 0 < n <= len(rows) <= 250 or not source.get("source_dev_prefix_sha256")
            or source.get("gold_rows_sha256") != sha(canonical) or len({r["id"] for r in rows}) != len(rows)):
        raise ValueError("MATH : source de golds dev corrigés incohérente")
    return {r["id"]: r for r in rows[:n]}


def token_proxy(peer: dict, identity: dict) -> float | None:
    """Recorded output tokens, otherwise locally retokenized text; never estimate missing costs as zero."""
    if "n_tokens" in peer:
        return float(peer["n_tokens"])
    path = HF / ("models--" + peer["model"].replace("/", "--")) / "snapshots" / identity["revision"]
    path /= "tokenizer.json"
    if not path.exists() or peer.get("text") is None:
        return None
    from tokenizers import Tokenizer
    if str(path) not in TOKENIZERS:
        fingerprint(path)
        TOKENIZERS[str(path)] = Tokenizer.from_file(str(path))
    return float(len(TOKENIZERS[str(path)].encode(peer["text"], add_special_tokens=False).ids))


TOKENIZERS: dict = {}


@dataclass
class Bank:
    name: str
    names: list[str]
    ids: list[str]
    gold: np.ndarray
    answer: np.ndarray
    confidence: np.ndarray
    tokens: np.ndarray
    references: dict[str, dict]
    e3: bool = False


def load_e4(bench: str, strict_math: bool) -> Bank:
    names = list(FAMILIES)
    loaded, protocol, items, gold_items = {}, None, None, None
    for model in [*FAMILIES.values(), *EXTRA, *REFS]:
        path = RESULTS / f"solo_{sot.slug(model)}_colab_{bench}_dev.jsonl"
        man, raw = read_dev(path)
        proto = {k: man.get(k) for k in PROTOCOL}
        if protocol is None:
            protocol = proto
            items = cache_dev(bench, man["n"], man["data"])
            if items is None and bench != "math500":
                raise ValueError(f"Cache dev nécessaire pour valider {bench}")
            gold_items = items if items is not None else load_math_golds(man["n"], man["data"])
        elif proto != protocol:
            raise ValueError(f"Protocoles différents : {path.name}")
        if man["model"] != model or man["bench"] != bench or not all(man.get(k) for k in
                ("revision", "weights", "gguf", "engine", "backend")):
            raise ValueError(f"Provenance invalide : {path.name}")
        if items is None and bench != "math500":
            raise ValueError(f"Cache dev nécessaire pour valider {bench}")
        rows = {r["id"]: dict(r) for r in raw}
        validate_ids(rows, gold_items, man["n"])
        for i, r in rows.items():
            if r["model"] != model or r["bench"] != bench or r["finish"] not in ("stop", "length"):
                raise ValueError(f"Ligne incohérente : {path.name} {i}")
            if not isinstance(r["n_tokens"], int) or not 0 <= r["n_tokens"] <= man["max_tokens"]:
                raise ValueError(f"Jetons incohérents : {path.name} {i}")
            item = gold_items[i]
            gold = answers.gold(bench, item)
            stored_gold = answers.norm_math(str(r["gold"]))
            legacy_subscript = (bench == "math500" and
                                stored_gold == answers.norm_math(str(item["answer"]).replace("_", "")))
            if gold is None or (stored_gold != gold and not legacy_subscript):
                raise ValueError(f"Gold incohérent : {path.name} {i}")
            r["stored_gold"] = stored_gold
            r["gold_changed"] = stored_gold != gold
            old = r["answer"]
            answers.MATH_FALLBACK = not (strict_math or items is None)
            r.update(gold=gold, answer=answers.extract(bench, r["text"], item, ended=r["finish"] == "stop"))
            r["changed"] = old != r["answer"]
        loaded[model] = rows
        VALIDATION.append({"file": path.name, "n": len(rows), "protocol": proto,
                           "identity": {k: man[k] for k in ("model", "revision", "weights", "engine")},
                           "reextractions_changed": sum(r["changed"] for r in rows.values()),
                           "gold_reextractions_changed": sum(r["gold_changed"] for r in rows.values()),
                           "gold_source": "validated_question_cache" if items is not None else "frozen_raw_dev_golds"})
    ids = sorted(loaded[next(iter(FAMILIES.values()))])
    gold = np.array([loaded[next(iter(FAMILIES.values()))][i]["gold"] for i in ids], dtype=object)
    for model, rows in loaded.items():
        first = loaded[next(iter(FAMILIES.values()))]
        if set(rows) != set(ids) or any(rows[i]["gold"] != gold[j] or
                                       rows[i]["stored_gold"] != first[i]["stored_gold"] for j, i in enumerate(ids)):
            raise ValueError(f"Alignement ID/gold invalide : {model}")
    def array(key, dtype):
        return np.array([[loaded[m][i][key] for m in FAMILIES.values()] for i in ids], dtype=dtype)
    refs = {m: {"answer": [loaded[m][i]["answer"] for i in ids],
                "tokens": [loaded[m][i]["n_tokens"] for i in ids], "model": m} for m in [*EXTRA, *REFS]}
    return Bank("E4/" + bench, names, ids, gold, array("answer", object), array("mean_logp", float),
                array("n_tokens", float), refs)


def load_e3() -> Bank:
    sources = ["essaim4_colab", "ref-Qwen3-4B_gpu"]
    loaded, items, proto = [], None, None
    for tag in sources:
        man, raw = read_dev(RESULTS / f"gen_{tag}_dev.jsonl", ("mode", "id"))
        current = {k: man.get(k) for k in ("data", "n", "prompt", "protocol", "template_date", "solo_tokens",
                                         "block", "k", "max_rounds", "tau")}
        if proto is None:
            proto = current
            items = cache_dev("gsm8k", man["n"], man["data"])
        elif proto != current:
            raise ValueError("Protocoles E3 incompatibles")
        if items is None:
            raise ValueError("Cache dev GSM8K absent")
        rows = {r["id"]: r for r in raw if r["mode"] == "solo"}
        validate_ids(rows, items, man["n"])
        peers = [p["model"] for p in man["peers"]]
        if len(peers) != len(set(peers)) or not all(p.get("revision") and p.get("weights") for p in man["peers"]):
            raise ValueError("Provenance des pairs E3 invalide")
        changed = 0
        for i, r in rows.items():
            if r["peers"] != peers or [p["model"] for p in r["per_peer"]] != peers:
                raise ValueError("Ordre des pairs E3 incohérent")
            if answers.norm_math(str(r["gold"])) != answers.gold("gsm8k", items[i]):
                raise ValueError("Gold E3 incohérent")
            r["gold"] = answers.gold("gsm8k", items[i])
            for p, identity in zip(r["per_peer"], man["peers"]):
                old = p["answer"]
                p["answer"] = answers.extract("gsm8k", p.get("text") or "", items[i], ended=bool(p.get("eos")))
                changed += old != p["answer"]
                p["token_proxy"] = token_proxy(p, identity)
        VALIDATION.append({"file": f"gen_{tag}_dev.jsonl", "n": len(rows), "protocol": current,
                           "peers": man["peers"], "reextractions_changed": changed})
        loaded.append((peers, rows))
    peers, rows = loaded[0]
    if len(peers) != 4:
        raise ValueError("E3 attend quatre pairs")
    ids = sorted(rows)
    def array(key, dtype):
        return np.array([[p[key] for p in rows[i]["per_peer"]] for i in ids], dtype=dtype)
    ref_peers, ref_rows = loaded[1]
    if ref_peers != ["Qwen/Qwen3-4B"] or set(ref_rows) != set(ids):
        raise ValueError("Référence E3 incohérente")
    refs = {ref_peers[0]: {"model": ref_peers[0],
                          "answer": [ref_rows[i]["per_peer"][0]["answer"] for i in ids],
                          "tokens": [ref_rows[i]["per_peer"][0]["token_proxy"] for i in ids]}}
    return Bank("E3/gsm8k", peers, ids, np.array([rows[i]["gold"] for i in ids], object),
                array("answer", object), array("conf", float), array("token_proxy", float), refs, True)


def weight(p, collision):
    p = np.clip(p, .02, .98)
    return np.maximum(0, np.log(p / (1 - p)) - math.log(collision))


def vote(a, w, confidence, e3=False):
    scores = defaultdict(float)
    for x, wi in zip(a, w):
        if x is not None and wi > 0:
            scores[x] += wi
    if not scores:
        return None, 0.0
    top = max(scores.values())
    tied = {x for x, s in scores.items() if s >= top - 1e-12}
    eligible = [j for j, x in enumerate(a) if x in tied]
    def tie(j):
        conf = confidence[j] if np.isfinite(confidence[j]) else -1e9
        return (conf,) if e3 else (w[j], conf)
    result = a[max(eligible, key=tie)]
    ranked = sorted(scores.values(), reverse=True) + [0.0]
    # Abstentions reduce the margin; an empty vote always has zero margin.
    return result, (ranked[0] - ranked[1]) / max(float(sum(w)), 1e-12)


def agreement(bank, idx, subset):
    a = bank.answer[np.ix_(idx, subset)]
    majority = [vote(x, np.ones(len(subset)), bank.confidence[i, subset], bank.e3)[0]
                for x, i in zip(a, idx)]
    return a == np.asarray(majority, object)[:, None]


def fit(bank: Bank, train, config: dict) -> dict:
    subset = config["subset"]
    a = bank.answer[np.ix_(train, subset)]
    g = bank.gold[train, None]
    given, correct = a != None, a == g  # noqa: E711
    p = np.divide(correct.sum(0), given.sum(0), out=np.zeros(len(subset)), where=given.sum(0) > 0)
    same = both = 0
    for row, gold in zip(a, bank.gold[train]):
        wrong = [x for x in row if x is not None and x != gold]
        for x, y in itertools.combinations(wrong, 2):
            both += 1
            same += x == y
    c = max(same, .5) / max(both, 1)
    state = {"subset": subset, "p": p.tolist(), "collision": c, "same": same, "both": both,
             "plain_accuracy": correct.mean(0).tolist(), "weights": weight(p, c).tolist()}
    if config["kind"] == "P02":
        agree = agreement(bank, train, subset)
        probabilities, counts = [], []
        for flag in (False, True):
            mask = given & (agree == flag)
            n = mask.sum(0)
            probabilities.append(((correct * mask).sum(0) + config["pseudo"] * p) / (n + config["pseudo"]))
            counts.append(n.tolist())
        state.update(conditional_counts=counts, conditional_p=np.asarray(probabilities).tolist(),
                     conditional_weights=weight(np.asarray(probabilities), c).tolist())
    elif config["kind"] == "P03":
        errors = (~correct).astype(float)
        centered = errors - errors.mean(0)
        norm = np.sqrt((centered ** 2).sum(0))
        denominator = norm[:, None] * norm[None, :]
        corr = np.divide(centered.T @ centered, denominator, out=np.zeros_like(denominator), where=denominator > 0)
        np.fill_diagonal(corr, 0)
        penalty = np.maximum(corr, 0).sum(0)
        state.update(error_correlations=corr.tolist(), penalty=penalty.tolist(),
                     weights=(weight(p, c) / (1 + config["lambda"] * penalty)).tolist())
    elif config["kind"] == "best":
        # Family order is fixed; no evaluated gold is used to choose the solo.
        state["best"] = int(np.argmax(correct.mean(0)))
    return state


def predict(bank: Bank, idx, config: dict, state: dict):
    subset = state["subset"]
    weights = np.asarray(state["weights"])
    agrees = agreement(bank, idx, subset) if config["kind"] == "P02" else None
    out, margins = [], []
    for k, i in enumerate(idx):
        if config["kind"] == "best":
            answer, margin = bank.answer[i, subset[state["best"]]], 0.0
        else:
            w = np.ones(len(subset)) if config["kind"] == "vote" else weights
            if agrees is not None:
                w = np.asarray(state["conditional_weights"])[agrees[k].astype(int), np.arange(len(subset))]
            answer, margin = vote(bank.answer[i, subset], w, bank.confidence[i, subset], bank.e3)
        out.append(answer)
        margins.append(margin)
    return np.array(out, object), np.array(margins)


def catalog(n: int):
    full = list(range(n))
    out = [{"kind": k, "subset": full} for k in ("best", "vote", "K")]
    out += [{"kind": "P01", "subset": list(s)} for k in range(1, n + 1)
            for s in itertools.combinations(full, k)]
    out += [{"kind": "P02", "subset": full, "pseudo": p} for p in (10, 30, 100)]
    out += [{"kind": "P03", "subset": full, "lambda": p} for p in (0, .5, 1, 2)]
    return out


def config_key(config, bank):
    return (len(config["subset"]), tuple(sorted(bank.names[j] for j in config["subset"])),
            config.get("pseudo", 0), config.get("lambda", 0))


def threshold(margins, predictions, q):
    """Conservative empirical quantile: ties never make the training referral share exceed q.

    Empty votes are mandatory referrals, included in the budget. Return None if they alone exceed q.
    """
    empty = predictions == None  # noqa: E711
    if empty.mean() > q + 1e-12:
        return None
    candidates = [-1.0, *sorted(set(float(x) for x in margins))]
    return max(t for t in candidates if np.mean(empty | (margins <= t)) <= q + 1e-12)


def referred(pred, margins, t):
    return (pred == None) | (margins <= t)  # noqa: E711


def cost(bank, i, subset, reference=None, call=False):
    ts = list(bank.tokens[i, subset])
    ms = [bank.names[j] if bank.e3 else FAMILIES[bank.names[j]] for j in subset]
    if reference is not None and call:
        ts.append(bank.references[reference]["tokens"][i])
        ms.append(reference)
    if any(t is None or not np.isfinite(t) for t in ts):
        return {"generated_tokens_proxy": None, "billion_parameters_times_generated_tokens_proxy": None}
    return {"generated_tokens_proxy": float(sum(ts)),
            "billion_parameters_times_generated_tokens_proxy": float(sum(models.total_b(m) * t for m, t in zip(ms, ts)))}


def ci(difference):
    v = np.asarray(difference, float)
    idx = np.random.default_rng(SEED).integers(0, len(v), size=(BOOT, len(v)))
    draws = v[idx].mean(1)
    return {"mean": float(v.mean()), "lo95": float(np.percentile(draws, 2.5)),
            "hi95": float(np.percentile(draws, 97.5)), "bootstrap_draws": BOOT, "seed": SEED}


def replay(bank: Bank, log, output, decisions):
    configs = catalog(len(bank.names))
    inner_candidates = []
    partitions = np.array([fold(i) for i in bank.ids])
    internal = np.array([fold(i, True) for i in bank.ids])
    for outer in range(5):
        train, held = np.flatnonzero(partitions != outer), np.flatnonzero(partitions == outer)
        if not len(held):
            raise ValueError("Pli externe vide")
        evaluations = []
        k_inner = []
        for config in configs:
            hits = total = 0
            for inner in range(3):
                fit_ids, valid = train[internal[train] != inner], train[internal[train] == inner]
                if not len(fit_ids) or not len(valid):
                    raise ValueError("Pli interne vide")
                state = fit(bank, fit_ids, config)
                pred, margins = predict(bank, valid, config, state)
                correct = pred == bank.gold[valid]
                hits += int(correct.sum())
                total += len(valid)
                log.append({"bank": bank.name, "outer": outer, "inner": inner, "config": config,
                            "train_ids": [bank.ids[i] for i in fit_ids], "valid_ids": [bank.ids[i] for i in valid],
                            "state": state, "correct": correct.astype(int).tolist(), "hits": int(correct.sum()),
                            "n": len(valid), "score": float(correct.mean())})
                if config["kind"] == "K":
                    train_pred, train_margins = predict(bank, fit_ids, config, state)
                    k_inner.append((inner, fit_ids, valid, state, pred, margins, train_pred, train_margins))
            evaluations.append((hits / total, config))
        inner_candidates.append({"outer": outer, "candidates": [{"score": s, "config": c} for s, c in evaluations]})
        selected = {}
        for kind in ("best", "vote", "K", "P01", "P02", "P03"):
            candidates = [x for x in evaluations if x[1]["kind"] == kind]
            score, config = min(candidates, key=lambda x: (-x[0], config_key(x[1], bank)))
            state = fit(bank, train, config)
            selected[kind] = (config, state)
            pred, _ = predict(bank, held, config, state)
            decisions.append({"bank": bank.name, "outer": outer, "kind": kind, "config": config,
                              "inner_score": score, "train_ids": [bank.ids[i] for i in train], "state": state})
            for i, p in zip(held, pred):
                subset = [state["subset"][state["best"]]] if kind == "best" else state["subset"]
                output.append({"bank": bank.name, "id": bank.ids[i], "outer": outer, "system": kind,
                               "prediction": p, "gold": bank.gold[i], "correct": int(p == bank.gold[i]),
                               "config": config, "models_called": [bank.names[j] for j in subset], **cost(bank, i, subset)})
        if bank.e3:
            continue
        kconfig, kstate = selected["K"]
        for reference in REFS:
            for budget in (.1, .2):
                trials = []
                for q in [0.0, *[x / 100 for x in range(1, round(100 * budget) + 1)]]:
                    hits = calls = count = 0
                    feasible = True
                    for inner, fit_ids, valid, state, pred, margins, tr_pred, tr_margin in k_inner:
                        t = threshold(tr_margin, tr_pred, q)
                        record = {"bank": bank.name, "outer": outer, "inner": inner, "config":
                                  {"kind": "P04", "reference": reference, "budget": budget, "quantile": q},
                                  "train_ids": [bank.ids[i] for i in fit_ids],
                                  "valid_ids": [bank.ids[i] for i in valid], "threshold": t, "state": state}
                        if t is None:
                            feasible = False
                            record.update(feasible=False, reason="abstentions au-delà du budget")
                        else:
                            route = referred(pred, margins, t)
                            answer = np.where(route, np.asarray(bank.references[reference]["answer"], object)[valid], pred)
                            correct = answer == bank.gold[valid]
                            hits += int(correct.sum())
                            calls += int(route.sum())
                            count += len(valid)
                            record.update(feasible=True, hits=int(correct.sum()), n=len(valid),
                                          score=float(correct.mean()), referrals=int(route.sum()),
                                          train_referral_rate=float(referred(tr_pred, tr_margin, t).mean()),
                                          correct=correct.astype(int).tolist())
                        log.append(record)
                    pooled = calls / count if count else None
                    eligible = feasible and pooled <= budget + 1e-12
                    trials.append({"quantile": q, "score": hits / count if feasible else None,
                                   "referral_rate": pooled, "eligible": eligible})
                candidates = [t for t in trials if t["eligible"]]
                system = f"P04/{reference}/{budget:.0%}"
                if not candidates:
                    decisions.append({"bank": bank.name, "outer": outer, "kind": system,
                                      "status": "infeasible", "trials": trials})
                    continue
                chosen = min(candidates, key=lambda t: (-t["score"], t["quantile"]))
                tr_pred, tr_margin = predict(bank, train, kconfig, kstate)
                t = threshold(tr_margin, tr_pred, chosen["quantile"])
                # A quantile feasible on inner training may be infeasible on the final training partition.
                if t is None:
                    decisions.append({"bank": bank.name, "outer": outer, "kind": system,
                                      "status": "infeasible_final_fit", "chosen": chosen, "trials": trials})
                    continue
                pred, margins = predict(bank, held, kconfig, kstate)
                route = referred(pred, margins, t)
                decisions.append({"bank": bank.name, "outer": outer, "kind": system, "chosen": chosen,
                                  "threshold": t, "trials": trials, "state": kstate,
                                  "train_ids": [bank.ids[i] for i in train],
                                  "train_referral_rate": float(referred(tr_pred, tr_margin, t).mean())})
                for i, p, call, margin in zip(held, pred, route, margins):
                    p = bank.references[reference]["answer"][i] if call else p
                    output.append({"bank": bank.name, "id": bank.ids[i], "outer": outer, "system": system,
                                   "prediction": p, "gold": bank.gold[i], "correct": int(p == bank.gold[i]),
                                   "reference": reference, "budget": budget, "threshold": t, "margin": float(margin),
                                   "referral": bool(call), **cost(bank, i, kstate["subset"], reference, call)})
    for i, identifier in enumerate(bank.ids):
        for j, name in enumerate(bank.names):
            p = bank.answer[i, j]
            output.append({"bank": bank.name, "id": identifier, "outer": int(partitions[i]),
                           "system": "solo/" + name, "prediction": p, "gold": bank.gold[i],
                           "correct": int(p == bank.gold[i]), **cost(bank, i, [j])})
        for name, ref in bank.references.items():
            p = ref["answer"][i]
            tokens = ref["tokens"][i]
            output.append({"bank": bank.name, "id": identifier, "outer": int(partitions[i]),
                           "system": "ref/" + name, "prediction": p, "gold": bank.gold[i],
                           "correct": int(p == bank.gold[i]), "generated_tokens_proxy": tokens,
                           "billion_parameters_times_generated_tokens_proxy":
                           None if tokens is None else models.total_b(name) * tokens})
        output.append({"bank": bank.name, "id": identifier, "outer": int(partitions[i]), "system": "oracle",
                       "correct": int(any(a == bank.gold[i] for a in bank.answer[i])), "deployable": False})
    correct = bank.answer == bank.gold[:, None]
    return {"catalog": configs, "inner_candidates": inner_candidates, "n": len(bank.ids), "families": bank.names,
            "oracle": float(correct.any(1).mean()),
            "oracle_with_each_reference": {m: float((correct.any(1) |
                                                     (np.array(ref["answer"], object) == bank.gold)).mean())
                                           for m, ref in bank.references.items()},
            "oracle_marginal_by_family": {name: float((correct.any(1) & ~np.delete(correct, j, 1).any(1)).mean())
                                         for j, name in enumerate(bank.names)},
            "folds": {i: {"outer": fold(i), "inner": fold(i, True)} for i in bank.ids}}


def category_guard(ids, categories, scores):
    """Leave one prompt out within its category; activate only for a strictly positive training gain."""
    out = []
    for i in ids:
        train = [j for j in ids if j != i and categories[j] == categories[i]]
        gain = float(np.mean([scores[j] - .5 for j in train])) if train else 0.0
        out.append({"id": i, "category": categories[i], "train_ids": train, "train_gain": gain,
                    "activated": bool(train and gain > 0), "score": scores[i] if train and gain > 0 else .5})
    return out


def validate_judged_keys(judged, ids, baselines):
    wanted = {(i, m, order) for i in ids for m in baselines for order in ("sot_first", "sot_second")}
    keys = [(r["id"], r["vs"], r["order"]) for r in judged]
    if set(keys) != wanted or len(keys) != len(set(keys)):
        raise ValueError("Les deux ordres E10 sont obligatoires et uniques pour chaque comparaison")


def replay_e10(log, output, decisions):
    path = RESULTS / f"sot_judge_sot1_{sot.slug(JUDGE)}_colab_dev.jsonl"
    man, judged = read_dev(path, ("id", "vs", "order"))
    expected = {"n": 16, "judge": JUDGE, "outline_model": SOLO, "prompt": sot.PROMPT_VERSION,
                "judge_prompt": sot.JUDGE_VERSION, "tag": "sot1", "temperature": 0.0, "thinking": False,
                "grammar": sot.JUDGE_GRAMMAR, "top_logprobs": sot.JUDGE_TOP}
    if any(man.get(k) != v for k, v in expected.items()):
        raise ValueError("Protocole juge E10 invalide")
    peers, baselines = man["peers"], man["baselines"]
    if SOLO not in baselines or len(peers) != len(set(peers)) or len(baselines) != len(set(baselines)):
        raise ValueError("Pairs/baselines E10 invalides")
    # Source questions only, pinned local HF snapshot. No hub API or benchmark loader is invoked.
    source = HF / "datasets--HuggingFaceH4--mt_bench_prompts/snapshots" / man["data"]["revision"]
    source /= "raw/question.jsonl"
    prompts = [json.loads(line) for line in fingerprint(source).decode().splitlines()]
    prompts.sort(key=lambda r: r["question_id"])
    prompts = [r for r in prompts if r["category"] in ("writing", "roleplay", "stem", "humanities")]
    random.Random(0).shuffle(prompts)
    items = {f"mtbench-{r['question_id']}": r for r in prompts[:16]}
    ids = list(items)
    categories = {i: items[i]["category"] for i in ids}
    stages, identities = {}, {}
    specs = [(RESULTS / f"sot_outline_{sot.slug(SOLO)}_colab_dev.jsonl", "outline", SOLO)]
    specs += [(RESULTS / f"sot_base_{sot.slug(m)}_colab_dev.jsonl", "baseline", m) for m in baselines]
    specs += [(RESULTS / f"sot_expand_sot1_{sot.slug(m)}_colab_dev.jsonl", "expand", m) for m in peers]
    for path, stage, model in specs:
        sm, rows = read_dev(path, ("id", "point") if stage == "expand" else ("id",))
        want = {"data": man["data"], "n": 16, "stage": stage, "model": model, "temperature": 0.0,
                "thinking": False, "seed": sot.SEED, "prompt": sot.PROMPT_VERSION,
                "max_tokens": {"outline": 256, "baseline": 1024, "expand": 256}[stage]}
        if stage == "expand":
            want.update(tag="sot1", peers=peers, outline_model=SOLO)
        if any(sm.get(k) != v for k, v in want.items()):
            raise ValueError(f"Protocole E10 invalide : {path.name}")
        identity = {k: sm.get(k) for k in ("revision", "weights", "engine", "gguf", "backend")}
        if not all(identity.values()) or (model in identities and identity != identities[model]):
            raise ValueError("Provenance E10 incohérente")
        identities[model] = identity
        outlines = stages.get(("outline", SOLO), {})
        wanted = {(i, k) for i, r in outlines.items() for k, m in enumerate(sot.assign(len(r["points"]), peers))
                  if m == model} if stage == "expand" else {(i,) for i in ids}
        keys = {(r["id"], r["point"]) if stage == "expand" else (r["id"],) for r in rows}
        if keys != wanted:
            raise ValueError(f"Complétude E10 invalide : {path.name}")
        for r in rows:
            if r["model"] != model or r["category"] != categories[r["id"]] or not isinstance(r["n_tokens"], int):
                raise ValueError("Ligne E10 incohérente")
            if not 0 <= r["n_tokens"] <= sm["max_tokens"] or r["finish"] not in ("stop", "length"):
                raise ValueError("Génération E10 invalide")
            if stage == "outline" and any(r.get(k) != v for k, v in sot.parse_skeleton(r["text"]).items()):
                raise ValueError("Squelette E10 incohérent")
            if stage == "expand" and (r["point_text"] != outlines[r["id"]]["points"][r["point"]]
                                      or r["n_points"] != len(outlines[r["id"]]["points"])):
                raise ValueError("Point E10 incohérent")
        if stage == "expand":
            outline_sha = sot.canonical_sha({i: {"points": r["points"], "status": r["status"]}
                                            for i, r in outlines.items()})
            if sm.get("outline_sha256") != outline_sha:
                raise ValueError("Empreinte squelette E10 invalide")
        stages[(stage, model)] = {(r["id"], r["point"]) if stage == "expand" else r["id"]: r for r in rows}
        VALIDATION.append({"file": path.name, "n_rows": len(rows), "manifest": sm})
    par, parallel_cost = {}, {}
    for i in ids:
        o = stages[("outline", SOLO)][i]
        exp = [stages[("expand", m)][(i, k)] for k, m in enumerate(sot.assign(len(o["points"]), peers))]
        base = stages[("baseline", SOLO)][i]
        par[i] = base["text"] if o["status"] == "fallback" else sot.assemble(o["points"], [r["text"] for r in exp])
        spent = [o, base] if o["status"] == "fallback" else [o, *exp]
        parallel_cost[i] = {"generated_tokens_proxy": sum(r["n_tokens"] for r in spent),
                            "billion_parameters_times_generated_tokens_proxy":
                            sum(models.total_b(r["model"]) * r["n_tokens"] for r in spent)}
    inputs = {i: {"sot": par[i], **{m: stages[("baseline", m)][i]["text"] for m in baselines}} for i in ids}
    if man["inputs_sha256"] != sot.canonical_sha(inputs):
        raise ValueError("Empreinte des réponses jugées E10 invalide")
    validate_judged_keys(judged, ids, baselines)
    scores, soft = defaultdict(list), defaultdict(list)
    for r in judged:
        if r["category"] != categories[r["id"]] or r["sot_is"] != ("A" if r["order"] == "sot_first" else "B"):
            raise ValueError("Ordre/catégorie juge invalide")
        if r["verdict"] not in ("A", "B", "C"):
            raise ValueError("Verdict E10 invalide")
        if bool(r.get("fallback")) != (stages[("outline", SOLO)][r["id"]]["status"] == "fallback"):
            raise ValueError("Fallback juge incohérent")
        identical = inputs[r["id"]]["sot"] == inputs[r["id"]][r["vs"]]
        if bool(r.get("identical")) != identical or (identical and r["verdict"] != "C"):
            raise ValueError("Égalité juge incohérente")
        p = r.get("probs")
        if not identical and (not p or set(p) != {"A", "B", "C", "mass"} or
                              any(not math.isfinite(v) or v < 0 for v in p.values()) or
                              not 0 < p["mass"] <= 1 + 1e-6 or
                              abs(sum(p[k] for k in ("A", "B", "C")) - 1) > 1e-6):
            raise ValueError("Probabilités juge invalides")
        scores[(r["id"], r["vs"])].append(sot.sot_outcome(r["verdict"], r["sot_is"]))
        soft[(r["id"], r["vs"])].append(.5 if identical else p[r["sot_is"]] + .5 * p["C"])
    raw = {i: float(np.mean(scores[(i, SOLO)])) for i in ids}
    guarded = category_guard(ids, categories, raw)
    for row in guarded:
        i = row["id"]
        base = stages[("baseline", SOLO)][i]
        baseline_cost = {"generated_tokens_proxy": base["n_tokens"],
                         "billion_parameters_times_generated_tokens_proxy": models.total_b(SOLO) * base["n_tokens"]}
        log.append({"bank": "E10", "config": {"kind": "P12", "reference": SOLO}, **row})
        decisions.append({"bank": "E10", "kind": "P12", **row})
        output.append({"bank": "E10", "system": "P12", **row,
                       **(parallel_cost[i] if row["activated"] else baseline_cost)})
        for system, score, costs in (("solo", .5, baseline_cost), ("parallel", raw[i], parallel_cost[i])):
            output.append({"bank": "E10", "id": i, "category": categories[i], "system": system,
                           "score": score, **costs})
    VALIDATION.append({"file": "sot_judge_dev", "manifest": man, "both_orders_validated": True})
    return {"reference": SOLO, "judge": JUDGE, "n": 16, "activated": sum(r["activated"] for r in guarded),
            "raw_vs_every_baseline": {m: {"hard": ci([np.mean(scores[(i, m)]) for i in ids]),
                                               "soft": ci([np.mean(soft[(i, m)]) for i in ids]),
                                               "wins": int(sum(np.mean(scores[(i, m)]) > .5 for i in ids)),
                                               "losses": int(sum(np.mean(scores[(i, m)]) < .5 for i in ids))}
                                      for m in baselines},
            "category_counts": {c: list(categories.values()).count(c) for c in sorted(set(categories.values()))}}


def summarize(output):
    grouped = defaultdict(dict)
    for row in output:
        group = grouped[(row["bank"], row["system"])]
        if row["id"] in group:
            raise ValueError("Sortie hors pli dupliquée")
        group[row["id"]] = row
    summary = {}
    for (bank, system), by_id in grouped.items():
        key = "score" if bank == "E10" else "correct"
        primary = ("solo" if bank == "E10" else "ref/" + next(iter(by_id.values()))["reference"]
                   if system.startswith("P04/") else "K")
        reference = grouped[(bank, primary)]
        ids = sorted(by_id)
        difference = [by_id[i][key] - reference[i][key] for i in ids]
        costs = {}
        for field in ("generated_tokens_proxy", "billion_parameters_times_generated_tokens_proxy"):
            values = [by_id[i].get(field) for i in ids]
            costs[field] = float(np.mean(values)) if all(v is not None for v in values) else None
            costs[field + "_available_n"] = sum(v is not None for v in values)
        summary.setdefault(bank, {})[system] = {"n": len(ids), "mean": float(np.mean([by_id[i][key] for i in ids])),
                                              "paired_comparator": primary, "paired_difference": ci(difference),
                                              "referral_rate": float(np.mean([by_id[i].get("referral", False)
                                                                              for i in ids])),
                                              **costs}
        if bank != "E10":
            summary[bank][system]["paired_vs_K"] = ci([by_id[i][key] - grouped[(bank, "K")][i][key] for i in ids])
    e4 = [b for b in summary if b.startswith("E4/")]
    macro = {}
    if len(e4) == 4:
        common = set.intersection(*(set(summary[b]) for b in e4))
        for system in sorted(common):
            if any(summary[b][system]["n"] != summary[b]["K"]["n"] for b in e4):
                continue
            draws = []
            rng = np.random.default_rng(SEED)
            for b in e4:
                primary = summary[b][system]["paired_comparator"]
                rows, ref = grouped[(b, system)], grouped[(b, primary)]
                ids = sorted(rows)
                v = np.array([rows[i]["correct"] - ref[i]["correct"] for i in ids])
                draws.append(v[rng.integers(0, len(v), (BOOT, len(v)))].mean(1))
            d = np.mean(draws, axis=0)
            macro[system] = {"mean": float(np.mean([summary[b][system]["mean"] for b in e4])),
                             "paired_difference": {"mean": float(np.mean([summary[b][system]["paired_difference"]["mean"]
                                                                           for b in e4])),
                                                   "lo95": float(np.percentile(d, 2.5)),
                                                   "hi95": float(np.percentile(d, 97.5))},
                             **{f: float(np.mean([summary[b][system][f] for b in e4]))
                                if all(summary[b][system][f] is not None for b in e4) else None
                                for f in ("generated_tokens_proxy", "billion_parameters_times_generated_tokens_proxy")}}
    return summary, macro


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def report(summary, macro, banks, e10, decisions, amendment):
    lines = ["# Percées Myriad : rejeu CPU dev du 10 octobre 2026", "",
             "Pré-enregistrement : f771fcc, PERC-20261010-v1. Analyse exploratoire exclusivement dev. "
             "Les tests déjà exposés historiquement ne constituent plus une confirmation vierge. "
             "Aucun fichier de réponses test ni résumé mêlant les partitions n'a été ouvert. Aucun GPU, "
             "Colab, téléchargement ni nouvelle génération.", "", "## Protocole et détails fixés avant calcul", ""]
    findings = ["## Lecture exploratoire des résultats", ""]
    for kind in ("P01", "P02", "P03"):
        s = macro[kind]
        d = s["paired_difference"]
        reduction = 1 - s["generated_tokens_proxy"] / macro["K"]["generated_tokens_proxy"]
        findings.append(f"- {kind} : macro E4 {100*s['mean']:.2f} %, gain sur K {100*d['mean']:+.2f} pp "
                        f"[{100*d['lo95']:+.2f} ; {100*d['hi95']:+.2f}], variation des jetons générés proxy "
                        f"{-100*reduction:+.1f} %. Ce classement descriptif ne fige aucune configuration future.")
    findings += ["", "P01 montre surtout un compromis coût/score à explorer ; son petit gain macro ne fournit "
                 "pas une confirmation de supériorité. P02 ne bat pas K ici. P03 conserve presque toutes les "
                 "décisions de K et son très petit gain dev reste exploratoire.", ""]
    e3 = summary["E3/gsm8k"]
    findings.append(f"E3 : K {100*e3['K']['mean']:.1f} %, P01 {100*e3['P01']['mean']:.1f} %, "
                    f"P02 {100*e3['P02']['mean']:.1f} %, P03 {100*e3['P03']['mean']:.1f} %. "
                    "Aucune réplication indépendante de GSM8K revendiquée.")
    e10s = summary["E10"]
    findings += ["", f"P12 : {e10['activated']}/16 activations ; la garde revient au solo. Décomposition brute "
                 f": score juge {100*e10s['parallel']['mean']:.2f} % contre ce solo, "
                 f"{e10s['parallel']['generated_tokens_proxy'] / e10s['solo']['generated_tokens_proxy']:.2f}× "
                 "les jetons générés proxy. Le retour au solo évite une perte observée ; ce n'est pas une "
                 "amélioration qualitative démontrée.", ""]
    lines[3:3] = findings
    lines += ["- " + s for s in amendment["details"]]
    lines += ["", "IC descriptifs appariés par question : bootstrap percentile, 2 000 tirages, graine 20261010. "
              "Pour la macro E4, rééchantillonnage séparé dans chacun des quatre bancs fixes, puis moyenne. "
              "Les plis d'apprentissage se recouvrent : ces IC ne couvrent pas toute l'incertitude de sélection. "
              "Aucune p-value ni conclusion confirmatoire. E3/GSM8K et E4/GSM8K ne sont pas indépendants.", "",
              "## Résultats hors pli", ""]
    for bank, systems in summary.items():
        lines += [f"### {bank}", "", "| système | n | score % | différence pp [IC 95 %] | comparateur | "
                  "jetons générés proxy | G-paramètres × jetons proxy | appels réf. % |",
                  "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |"]
        for system, s in systems.items():
            d = s["paired_difference"]
            fmt = lambda x: "indisponible" if x is None else f"{x:.1f}"
            lines.append(f"| {system} | {s['n']} | {100*s['mean']:.2f} | {100*d['mean']:+.2f} "
                         f"[{100*d['lo95']:+.2f} ; {100*d['hi95']:+.2f}] | {s['paired_comparator']} | "
                         f"{fmt(s['generated_tokens_proxy'])} | "
                         f"{fmt(s['billion_parameters_times_generated_tokens_proxy'])} | "
                         f"{100*s['referral_rate']:.1f} |")
        lines += [""]
        cascades = {k: v for k, v in systems.items() if k.startswith("P04/")}
        if cascades:
            lines += ["P04 : comparaison supplémentaire au sélecteur primaire K, avec toutes les références fixes.", "",
                      "| variante | gain sur K pp [IC descriptif] |", "| --- | --- |"]
            for name, s in cascades.items():
                d = s["paired_vs_K"]
                lines.append(f"| {name} | {100*d['mean']:+.2f} [{100*d['lo95']:+.2f} ; {100*d['hi95']:+.2f}] |")
            lines += [""]
    lines += ["## Moyenne macro des quatre bancs E4", "", "| système | score % | différence pp [IC descriptif] |",
              "| --- | ---: | --- |"]
    for system, s in macro.items():
        d = s["paired_difference"]
        lines.append(f"| {system} | {100*s['mean']:.2f} | {100*d['mean']:+.2f} "
                     f"[{100*d['lo95']:+.2f} ; {100*d['hi95']:+.2f}] |")
    lines += ["", "## Décisions internes sélectionnées", "",
              "Les listes ci-dessous sont ordonnées par pli externe 0–4. Les états finaux, chaque score "
              "de chaque configuration dans chaque pli interne, les identifiants d'apprentissage et de "
              "validation figurent dans les artefacts JSON/JSONL ; aucune variante négative n'est retirée.", ""]
    for bank in banks:
        lines += [f"### {bank}", ""]
        for kind in ("best", "P01", "P02", "P03"):
            ds = [d for d in decisions if d["bank"] == bank and d["kind"] == kind]
            text = []
            for d in ds:
                config = d["config"]
                if kind == "best":
                    choice = banks[bank]["families"][config["subset"][d["state"]["best"]]]
                elif kind == "P01":
                    choice = ", ".join(banks[bank]["families"][j] for j in config["subset"])
                else:
                    choice = str(config.get("pseudo", config.get("lambda")))
                text.append(f"pli {d['outer']}: {choice} (interne {100*d['inner_score']:.2f} %)")
            lines.append(f"- {kind} : " + " ; ".join(text))
        for d in [d for d in decisions if d["bank"] == bank and d["kind"].startswith("P04/")]:
            lines.append(f"- {d['kind']}, pli {d['outer']} : " +
                         (d["status"] if "status" in d else f"q={d['chosen']['quantile']:.2f}, "
                          f"seuil={d['threshold']:.6f}, appels apprentissage={100*d['train_referral_rate']:.1f} %"))
        lines += ["", "Plafond oracle (non déployable) et contribution exclusive à ce plafond : " +
                  json.dumps({k: v for k, v in banks[bank].items() if k.startswith("oracle")}, ensure_ascii=False), ""]
    lines += ["## P12 : décomposition qualitative et garde par catégorie", "",
              f"Référence fixée avant rejeu : {e10['reference']}. Juge fixé : {e10['judge']}. "
              f"{e10['activated']}/16 activations leave-one-out. Effectifs : {e10['category_counts']}. "
              "Un score de 0,5 signifie la parité avec le solo ; il ne s'agit pas d'une exactitude factuelle.", "",
              "| solo comparé à la décomposition brute | score juge % [IC] | victoires | pertes | score soft % |",
              "| --- | --- | ---: | ---: | ---: |"]
    for m, s in e10["raw_vs_every_baseline"].items():
        d = s["hard"]
        lines.append(f"| {m} | {100*d['mean']:.1f} [{100*d['lo95']:.1f} ; {100*d['hi95']:.1f}] | "
                     f"{s['wins']} | {s['losses']} | {100*s['soft']['mean']:.1f} |")
    lines += ["", "La garde sélectionne sur quelques voisins de catégorie seulement. Les verdicts des deux ordres "
              "sont moyennés par prompt. Une économie de jetons ne prouve pas une amélioration qualitative, "
              "ni un gain de latence réelle ; le juge est unique et les 16 prompts restent exploratoires.", ""]
    for d in [d for d in decisions if d["bank"] == "E10"]:
        lines.append(f"- {d['id']} ({d['category']}) : {len(d['train_ids'])} voisins, "
                     f"gain apprentissage {d['train_gain']:+.3f}, activation {d['activated']}, score {d['score']:.2f}.")
    lines += ["", "## Limites, statut des douze idées et suite", "",
              "P01/P02/P03 sont des sélecteurs exploratoires, comparés principalement à K. P04 doit aussi être "
              "comparée à chacune de ses références fixes. Le budget est une contrainte interne d'apprentissage : "
              "le taux hors pli peut dépasser 10/20 % ; il est publié sans réajustement. Le coût P04 additionne "
              "tous les pairs puis la référence appelée. P01 compte seulement les pairs sélectionnés. Les coûts "
              "excluent les prompts, le réseau et l'apprentissage CPU : ce sont des proxies, pas des FLOPs, "
              "des joules ou des latences mesurées. Les tailles sont les paramètres totaux, embeddings compris.", "",
              "P05 est couvert par P01 et les plafonds oracle ; aucune découverte indépendante revendiquée. "
              "P06 : exécutions E11 non analysées. P07/P08 : protocoles GPU sans lancement. "
              "P09 : logs spéculatifs absents. P10 : logits absents. P11 : conditionnel, aucune adaptation entraînée. "
              "P12 conserve les pertes qualitatives et les coûts de la décomposition brute ainsi que de la garde.", "",
              "L'identité complète des caches de questions est attestée par leur manifeste local ; seules les "
              "lignes dev réellement lues ont une empreinte recalculée. Le texte source local MT-Bench est lu "
              "pour reconstruire la partition déterministe des questions ; aucune réponse non-dev n'est lue. "
              "Les jetons E3 proviennent, lorsque possible, d'une retokenisation CPU du texte via tokenizer.json "
              "local fixé à la révision du manifeste ; le texte ne restitue pas forcément les tokens spéciaux "
              "ou cachés de la génération. Une valeur indisponible reste nulle, jamais remplacée par zéro.", "",
              "Le repli MATH sans questions utilise exclusivement percees_math_gold_dev.json, qui conserve "
              "les golds bruts dev du cache validé, leur identité et leurs empreintes, indépendamment des "
              "prédictions. Son grader est boxed strict pour tous les modèles. Si cette source et le cache "
              "manquent, le scoring est refusé ; les golds historiques abîmés ne sont pas réutilisés.", "",
              "Une confirmation exigera de nouvelles tâches, une configuration figée par idée avant ouverture, "
              "les comparateurs/bancs déclarés et le budget familial du registre. Ces résultats dev ne justifient "
              "ni une supériorité confirmée ni une équivalence.", "",
              "Artefacts : percees_amendment_dev.json (avant calcul), percees_internal_dev.jsonl, "
              "percees_oof_dev.jsonl, percees_decisions_dev.json, percees_summary_dev.json, "
              "percees_inputs_dev.json, percees_math_gold_dev.json, percees_validation_dev.json. "
              "Le registre docs/12_percees.md reste inchangé.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["dev"], default="dev")
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args()
    if a.results.resolve() != RESULTS.resolve():
        ap.error("Seul le répertoire results du présent worktree est autorisé ; chemins non-dev refusés")
    math_man = json.loads(fingerprint(dev_path(RESULTS / "solo_Qwen__Qwen3.5-4B_colab_math500_dev.jsonl")
                                      .with_name("solo_Qwen__Qwen3.5-4B_colab_math500_dev.jsonl.meta.json")))
    math_items = cache_dev("math500", math_man["n"], math_man["data"])
    if math_items is None:
        load_math_golds(math_man["n"], math_man["data"])
    amendment = {"date": "2026-10-10", "preregistration": "f771fcc", "split": "dev",
                 "written_before_calculation": True, "math_strict_boxed": math_items is None,
                 "details": [
                     "SHA256('percees-v1:'+id) modulo 5 ; internes SHA256('percees-inner-v1:'+id) modulo 3, par question.",
                     "Ré-extraction MATH via answers.py et cache dev local ; sinon boxed strict pour tous les modèles, "
                     "avec amendement préalable. Mode de cette passe : " +
                     ("boxed strict (cache absent)." if math_items is None else "grader courant avec contexte question dev."),
                     "Validation préalable MATH : trois golds stockés ont perdu les underscores des bases numériques "
                     "(204_5, 40_9, 52_8). Gold ré-extrait du cache dev avec answers.gold ; ancienne transformation "
                     "identifiée explicitement, alignement des golds stockés et des golds corrigés vérifié entre modèles. "
                     "Aucune autre divergence gold tolérée.",
                     "P01 : collision et poids K refaits pour chaque sous-ensemble sur train interne ; score interne "
                     "agrégé par question, départage taille puis familles lexicographiques ; refit train externe.",
                     "P02 : majorité uniforme comprenant le modèle, départage existant, abstention hors p ; "
                     "pseudo-comptes 10/30/100 vers p global et collision globale du train.",
                     "P03 : corrélations Pearson des erreurs (abstention = erreur), constantes à zéro, diagonale exclue ; "
                     "lambda 0/0,5/1/2, tous calculs sur train seulement.",
                     "P04 : marge K=(premier-second)/somme de tous les poids, abstention marge 0. Grille quantile "
                     "0,01 par pas jusqu'au budget 10/20 %, avec q=0. Seuil conservateur appris sur train interne, "
                     "abstentions forcées incluses ; éligibilité taux interne agrégé <= budget. Départage plus petit q. "
                     "Refit du quantile choisi sur train externe uniquement. "
                     "Variante infaisable déclarée sans assouplissement.",
                     "E10/P12 : solo Qwen/Qwen3.5-4B du squelette fixé avant rejeu, juge Qwen/Qwen3.8-27B, "
                     "deux ordres obligatoires. Leave-one-out dans la catégorie ; activation si moyenne(score-0,5)>0, "
                     "solo sinon, y compris catégorie sans voisin.",
                     "Échec technique préalable : référence E3 ref-Qwen3-4B_colab_dev incomplète (83/100), "
                     "aucun score calculé. Référence E3 fixée à ref-Qwen3-4B_gpu_dev (100/100), protocole et "
                     "questions identiques ; poids GGUF et build du moteur distincts, provenance publiée. "
                     "Lecture de générations historiques uniquement, aucun lancement GPU.",
                     "Proxies de coût : jetons de sortie enregistrés E4/E10 ; retokenisation locale E3 si tokenizer "
                     "disponible, sinon coût indisponible. G-paramètres totaux × jetons générés ; "
                     "aucun coût matériel revendiqué.",
                     "Caches : lecture préfixe dev uniquement ; identité totale attestée par métadonnées, "
                     "empreinte dev recalculée. "
                     "Questions MT-Bench reconstruites depuis la source HF locale épinglée, sans téléchargement."]}
    amendment["details"].append("Échec technique de validation E10 : champ mass distingué des probabilités "
                     "normalisées A/B/C, conformément à sot.letter_probs ; contrôle corrigé, "
                                "aucun choix de configuration modifié à partir de scores.")
    amendment["details"].append("Échec technique d'export : compteurs victoires/pertes NumPy convertis en int "
                                "Python pour le JSON strict ; aucun score ou choix modifié.")
    amendment["details"].append("Correctif de revue b025bb2 : le repli MATH sans questions exige la source "
                                "percees_math_gold_dev.json de golds bruts dev figés depuis le cache validé, "
                                "avec identité dataset, empreinte des lignes dev sources et intégrité des golds. "
                                "Boxed strict reste fixé pour tous les modèles dans ce repli. Si cache et source "
                                "de golds corrigés sont absents, le scoring est refusé ; les golds historiques "
                                "abîmés ne sont jamais utilisés comme vérité. Résultats actuels avec cache inchangés.")
    write_json(RESULTS / "percees_amendment_dev.json", amendment)
    if math_items is not None:
        freeze_math_golds(math_items, math_man["data"])
    for path in (Path(__file__), HERE / "essaim/answers.py", HERE / "essaim/sot.py", HERE / "essaim/models.py",
                 HERE / "analyze_e4.py", HERE.parent / "docs/12_percees.md"):
        fingerprint(path)
    incomplete_man, incomplete_raw = read_dev(RESULTS / "gen_ref-Qwen3-4B_colab_dev.jsonl", ("mode", "id"))
    VALIDATION.append({"file": "gen_ref-Qwen3-4B_colab_dev.jsonl", "status": "not_scored_incomplete_run",
                       "n_solo": sum(r["mode"] == "solo" for r in incomplete_raw), "manifest": incomplete_man})
    log, output, decisions, banks = [], [], [], {}
    for bank in [load_e3(), *[load_e4(b, math_items is None) for b in BENCHES]]:
        print(f"Rejeu CPU {bank.name} ({len(bank.ids)} questions)", flush=True)
        banks[bank.name] = replay(bank, log, output, decisions)
    e10 = replay_e10(log, output, decisions)
    summary, macro = summarize(output)
    write_jsonl(RESULTS / "percees_internal_dev.jsonl", log)
    write_jsonl(RESULTS / "percees_oof_dev.jsonl", output)
    write_json(RESULTS / "percees_decisions_dev.json", decisions)
    write_json(RESULTS / "percees_inputs_dev.json", {"sha256": FINGERPRINTS, "validation": VALIDATION})
    write_json(RESULTS / "percees_summary_dev.json", {"banks": banks, "results": summary, "e4_macro": macro, "e10": e10,
                                                     "bootstrap": {"draws": BOOT, "seed": SEED}, "confirmatory": False})
    rendered_report = report(summary, macro, banks, e10, decisions, amendment)
    (RESULTS / "percees_report_dev.md").write_text(rendered_report, encoding="utf-8")
    print("Rapport : phase0/results/percees_report_dev.md", flush=True)


if __name__ == "__main__":
    main()
