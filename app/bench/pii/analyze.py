"""Metrics, combinations and conformal calibration of the PII detectors (after run_bench).

    uv run --group pii-bench python -m bench.pii.analyze [--alpha 0.01] [--delta 0.05]

Protocol: train split -> probe training and lambda_high; cal split -> conformal lambda_low; test split
-> every reported number. Writes bench/results/pii_report.json and pii_report.md."""
from __future__ import annotations

import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from myriad.privacy_model import decide, merge_tokens, residual_tokens

from ..common import pct
from .conformal import clopper_pearson, marginal_threshold, min_n, pac_threshold
from .data import TYPES, load
from .registry import MODELS
from .run_bench import RUNS

OUT = Path(__file__).resolve().parent.parent / "results"
SPAN_MODELS = ["nym-small-edge", "nym-small-int8", "bert-small-pii", "ettin-32m", "minilm-nemotron",
               "gliner-pii-small"]
PROBES = ["e5-small-probe", "mminilm-probe", "gte-small-probe", "embeddinggemma2-probe"]
DOC_MODELS = ["bunker-laya"]
LAT_SHORT = 600  # characters: a typical typed prompt
REDRAWS = 500


_DOCS: list[dict] | None = None


def read_run(name: str) -> dict:
    """Records of a finished run keyed by document id. Records are in load() order: they are aligned
    by position (ids written by older runs may lack the suffix that made Nemotron ids unique)."""
    global _DOCS
    p = RUNS / f"{name}.jsonl.gz"
    if not p.exists() or not (RUNS / f"{name}.meta.json").exists():  # missing or still running
        return {}
    if _DOCS is None:
        _DOCS = load()
    with gzip.open(p, "rt", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    if len(recs) != len(_DOCS):  # a subset run (PII_BENCH_SUBSET), written with the current ids
        ids = {d["id"] for d in _DOCS}
        assert all(r["id"] in ids for r in recs), name
        return {r["id"]: r for r in recs}
    out = {}
    for d, r in zip(_DOCS, recs):
        assert r["id"] == d["id"] or d["id"].startswith(r["id"] + "-"), (name, r["id"], d["id"])
        out[d["id"]] = r
    return out


def meta(name: str) -> dict:
    out = {}
    for suffix in (".meta.json", ".mem.json", ".mem2.json"):
        p = RUNS / f"{name}{suffix}"
        if p.exists():
            out[suffix.strip(".").replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
    return out


def overlap(a, z, s, e) -> bool:
    return a < e and s < z


def scores_tokens(doc: dict, tokens: list) -> tuple[float, float, list[float]]:
    """(gate score, coverage score, per-gold-span best score) from token scores."""
    gate = max((t[2] for t in tokens), default=0.0)
    per_span = []
    for g in doc["spans"]:
        per_span.append(max((t[2] for t in tokens if overlap(t[0], t[1], g["start"], g["end"])), default=0.0))
    cov = min(per_span) if per_span else gate
    return gate, cov, per_span


def train_probe(name: str, docs: list[dict], kind: str = "lr"):
    """Probe on chunk embeddings; label = chunk overlaps an in-scope gold span. Doc score = max chunk."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    p = RUNS / f"{name}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    vecs, owners = z["vecs"], z["owners"]
    run = read_run(name)
    docs = [d for d in docs if d["id"] in run]
    ranges = []
    for i, d in enumerate(docs):
        ranges += [(i, a, b) for a, b in run[d["id"]]["chunks"]]
    assert len(ranges) == len(vecs)
    y = np.array([int(any(overlap(a, b, g["start"], g["end"]) for g in docs[i]["spans"])) for i, a, b in ranges])
    tr = np.array([docs[i]["split"] == "train" for i, _, _ in ranges])
    if kind == "lr":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=300, early_stopping=True, random_state=0)
    clf.fit(vecs[tr], y[tr])
    prob = clf.predict_proba(vecs)[:, 1]
    doc = defaultdict(float)
    for (i, _, _), pr in zip(ranges, prob):
        doc[i] = max(doc[i], float(pr))
    return {docs[i]["id"]: doc[i] for i in range(len(docs))}


def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    ranks, i = {}, 0
    r = np.empty(len(allv))
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    lab = np.array([l for _, l in allv])
    rp = r[lab == 1].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(name: str, docs: list[dict], S: dict, alpha: float, delta: float, has_spans: bool) -> dict:
    """S[id] = {"gate": float, "cov": float, "per_span": [...]}. Thresholds from train/cal, metrics on test."""
    all_docs = [d for d in docs if d["id"] in S]  # a slow detector may have run on a subset
    docs = [d for d in all_docs if d["guarantee"]]  # gretel (noisy labels) is reported apart, see data.py
    by = defaultdict(list)
    for d in docs:
        by[d["split"]].append(d)
    cal_pos = [S[d["id"]]["gate"] for d in by["cal"] if d["label"]]
    lam = marginal_threshold(cal_pos, alpha)
    lam_pac = pac_threshold(cal_pos, alpha, delta)
    res = {"detector": name, "n_cal_pos": len(cal_pos), "lambda_low": lam, "lambda_low_pac": lam_pac}
    # the trade-off at other targets (same protocol)
    res["alphas"] = {}
    test_pos_ = [d for d in by["test"] if d["label"]]
    test_neg_ = [d for d in by["test"] if not d["label"]]
    for a in (0.01, 0.02, 0.05, 0.10):
        t = marginal_threshold(cal_pos, a)
        m = sum(S[d["id"]]["gate"] < t for d in test_pos_)
        res["alphas"][str(a)] = {"lambda": t, "test_miss_rate": m / max(1, len(test_pos_)),
                                 "test_miss_ci95": list(clopper_pearson(m, len(test_pos_))),
                                 "fpr": sum(S[d["id"]]["gate"] >= t for d in test_neg_) / max(1, len(test_neg_))}
    if has_spans:
        cal_cov = [S[d["id"]]["cov"] for d in by["cal"] if d["label"]]
        res["lambda_cov"] = marginal_threshold(cal_cov, alpha)
    # lambda_high on train: just above s_(N - floor(0.05 N)) (1-based) of the N train negatives, so that at
    # most 5 % of them reach it (no cap at 1: with ties at 1 the threshold may exceed 1)
    tr_neg = sorted(S[d["id"]]["gate"] for d in by["train"] if not d["label"])
    hi = tr_neg[len(tr_neg) - math.floor(0.05 * len(tr_neg)) - 1] if tr_neg else 1.0
    res["lambda_high"] = max(lam, hi + 1e-6)
    test = by["test"]
    pos = [d for d in test if d["label"]]
    neg = [d for d in test if not d["label"]]
    # validity check: re-draw the cal/test partition of the positives many times (same sizes)
    pool = [S[d["id"]]["gate"] for d in by["cal"] + by["test"] if d["label"]]
    rng = np.random.default_rng(0)
    rates = []
    for _ in range(REDRAWS):
        perm = rng.permutation(len(pool))
        c = [pool[i] for i in perm[:len(cal_pos)]]
        t = marginal_threshold(c, alpha)
        rates.append(float(np.mean([pool[i] < t for i in perm[len(cal_pos):]])))
    res["redraws"] = {"n": REDRAWS, "mean_miss": float(np.mean(rates)), "p90_miss": float(np.quantile(rates, 0.9)),
                      "share_above_alpha": float(np.mean([r > alpha for r in rates]))}

    def recall_at(ds, t, key="gate"):
        if not ds:
            return None
        return sum(S[d["id"]][key] >= t for d in ds) / len(ds)

    miss = sum(S[d["id"]]["gate"] < lam for d in pos)
    lo, hi_ci = clopper_pearson(miss, len(pos))
    miss_pac = sum(S[d["id"]]["gate"] < lam_pac for d in pos)
    res.update(
        n_test_pos=len(pos), n_test_neg=len(neg),
        test_miss=miss, test_miss_rate=miss / len(pos), test_miss_ci95=[lo, hi_ci],
        test_miss_rate_pac=miss_pac / len(pos), test_miss_ci95_pac=list(clopper_pearson(miss_pac, len(pos))),
        fpr=recall_at(neg, lam), fpr_pac=recall_at(neg, lam_pac), fpr_high=recall_at(neg, res["lambda_high"]),
        auroc=auroc([S[d["id"]]["gate"] for d in pos], [S[d["id"]]["gate"] for d in neg]),
        recall_at_05=recall_at(pos, 0.5), fpr_at_05=recall_at(neg, 0.5),
    )
    flagged = [d for d in test if S[d["id"]]["gate"] >= lam]
    res["precision"] = (sum(d["label"] for d in flagged) / len(flagged)) if flagged else None
    # decisions of the scanner on test (grey zone = ask)
    dec = defaultdict(lambda: defaultdict(int))
    for d in test:
        g = S[d["id"]]["gate"]
        k = "send" if g < lam else "ask" if g < res["lambda_high"] else "mask/local"
        dec["pos" if d["label"] else "neg"][k] += 1
    res["decisions"] = {k: dict(v) for k, v in dec.items()}
    # per source / language
    res["by_source"] = {}
    all_test = [d for d in all_docs if d["split"] == "test"]
    for src in sorted({d["source"] for d in all_test}):
        ps = [d for d in all_test if d["source"] == src and d["label"]]
        ns = [d for d in all_test if d["source"] == src and not d["label"]]
        res["by_source"][src] = {"n_pos": len(ps), "miss_rate": (1 - recall_at(ps, lam)) if ps else None,
                                 "miss_rate_05": (1 - recall_at(ps, 0.5)) if ps else None,
                                 "n_neg": len(ns), "fpr": recall_at(ns, lam), "fpr_05": recall_at(ns, 0.5)}
    res["by_lang"] = {}
    for lang in ("en", "fr", "de", "es", "it", "nl", "pt", "sv", "code"):
        ps = [d for d in pos if d["lang"] == lang]
        ns = [d for d in neg if d["lang"] == lang]
        if ps or ns:
            m = sum(S[d["id"]]["gate"] < lam for d in ps)
            res["by_lang"][lang] = {"n_pos": len(ps), "miss_rate": m / len(ps) if ps else None,
                                    "miss_ci95": list(clopper_pearson(m, len(ps))) if ps else None,
                                    "miss_rate_05": (1 - recall_at(ps, 0.5)) if ps else None,
                                    "n_neg": len(ns), "fpr": recall_at(ns, lam), "fpr_05": recall_at(ns, 0.5)}
    other = [d for d in pos if d["lang"] not in res["by_lang"]]
    if other:
        res["by_lang"]["other"] = {"n_pos": len(other), "miss_rate": 1 - recall_at(other, lam)}
    if has_spans:
        lc = res["lambda_cov"]
        leak = sum(S[d["id"]]["cov"] < lc for d in pos)
        res["coverage"] = {"lambda": lc, "test_leak_rate": leak / len(pos),
                           "test_leak_ci95": list(clopper_pearson(leak, len(pos))),
                           "fpr": recall_at(neg, lc), "alphas": {}}
        cal_cov = [S[d["id"]]["cov"] for d in by["cal"] if d["label"]]
        for a in (0.01, 0.02, 0.05, 0.10, 0.20):
            t = marginal_threshold(cal_cov, a)
            m = sum(S[d["id"]]["cov"] < t for d in pos)
            res["coverage"]["alphas"][str(a)] = {"lambda": t, "test_leak_rate": m / len(pos),
                                                 "test_leak_ci95": list(clopper_pearson(m, len(pos))),
                                                 "fpr": recall_at(neg, t)}
        per_type = defaultdict(lambda: [0, 0, 0])
        for d in pos:
            for g, sc in zip(d["spans"], S[d["id"]]["per_span"]):
                per_type[g["type"]][0] += sc >= lam
                per_type[g["type"]][1] += sc >= 0.5
                per_type[g["type"]][2] += 1
        res["span_recall_by_type"] = {t: {"recall": a / n, "recall_05": b / n, "n": n}
                                      for t, (a, b, n) in sorted(per_type.items())}
        n = sum(v[2] for v in per_type.values())
        res["span_recall"] = sum(v[0] for v in per_type.values()) / n if n else None
        res["span_recall_05"] = sum(v[1] for v in per_type.values()) / n if n else None
    return res


def residual_calibration(docs: list[dict], regex_run: dict, run: dict, alpha: float, delta: float) -> dict:
    """Policy "mask": rule spans are masked; the model gates what they leave. A positive document leaks
    iff one of its gold spans is not entirely masked by the rules and no model token outside the rule
    spans reaches lambda. Critical score: max residual token score, or +inf (2.0) when the rules mask every character
    of every gold span (a partial overlap does not count: the rest of the span would be sent)."""
    def masked(g, rules, text):  # every non-space character of the gold span lies in a rule span
        return all(text[i].isspace() or any(r["start"] <= i < r["end"] for r in rules) for i in range(g["start"], g["end"]))

    def crit(d):
        rules = [{"start": t[0], "end": t[1]} for t in regex_run[d["id"]]["tokens"]]
        uncovered = [g for g in d["spans"] if not masked(g, rules, d["text"])]
        if not uncovered:
            return 2.0
        res = residual_tokens([tuple(t) for t in run[d["id"]]["tokens"]], rules)
        return max((t[2] for t in res), default=0.0)

    def held(d, lam):  # the document is not sent automatically (ask or local)
        rules = [{"start": t[0], "end": t[1]} for t in regex_run[d["id"]]["tokens"]]
        res = residual_tokens([tuple(t) for t in run[d["id"]]["tokens"]], rules)
        return max((t[2] for t in res), default=0.0) >= lam

    g = [d for d in docs if d["guarantee"] and d["id"] in run and d["id"] in regex_run]
    cal = [crit(d) for d in g if d["split"] == "cal" and d["label"]]
    lam = marginal_threshold(cal, alpha)
    lam_pac = pac_threshold(cal, alpha, delta)
    tpos = [d for d in g if d["split"] == "test" and d["label"]]
    tneg = [d for d in g if d["split"] == "test" and not d["label"]]
    leak = sum(crit(d) < lam for d in tpos)
    leak_pac = sum(crit(d) < lam_pac for d in tpos)
    return {"lambda_mask": lam, "lambda_mask_pac": lam_pac, "n_cal_pos": len(cal),
            "rules_cover_all_share_cal": sum(c == 2.0 for c in cal) / max(1, len(cal)),
            "test_leak_rate": leak / max(1, len(tpos)), "test_leak_ci95": list(clopper_pearson(leak, len(tpos))),
            "test_leak_rate_pac": leak_pac / max(1, len(tpos)),
            "test_leak_ci95_pac": list(clopper_pearson(leak_pac, len(tpos))),
            "fpr_held": sum(held(d, lam) for d in tneg) / max(1, len(tneg)),
            "fpr_held_pac": sum(held(d, lam_pac) for d in tneg) / max(1, len(tneg)),
            "neg_masked_by_rules": sum(bool(regex_run[d["id"]]["tokens"]) for d in tneg) / max(1, len(tneg))}


def latency(run: dict, docs: list[dict]) -> dict:
    short = [run[d["id"]]["ms"] for d in docs if d["id"] in run and len(d["text"]) <= LAT_SHORT]
    allv = [run[d["id"]]["ms"] for d in docs if d["id"] in run]
    return {"short_p50_ms": pct(short, 50), "short_p95_ms": pct(short, 95), "all_p50_ms": pct(allv, 50),
            "all_p95_ms": pct(allv, 95), "n_short": len(short)}


def combine(*parts):
    """Element-wise combination of score dicts: max of gate, max of per-span (coverage recomputed)."""
    out = {}
    for k in set(parts[0]).intersection(*parts[1:]):
        gate = max(p[k]["gate"] for p in parts)
        per = [max(vals) for vals in zip(*[p[k]["per_span"] for p in parts if p[k]["per_span"] is not None])] \
            if all(p[k]["per_span"] is not None for p in parts) else None
        out[k] = {"gate": gate, "per_span": per, "cov": (min(per) if per else gate) if per is not None else gate}
    return out


def main(argv: list[str]) -> None:
    alpha = float(next((a.split("=")[1] for a in argv if a.startswith("--alpha=")), 0.01))
    delta = float(next((a.split("=")[1] for a in argv if a.startswith("--delta=")), 0.05))
    global _DOCS
    docs = _DOCS = load()
    S: dict[str, dict] = {}
    reports, lat, mem = {}, {}, {}
    for name in ["regex"] + SPAN_MODELS:
        run = read_run(name)
        if not run:
            continue
        S[name] = {}
        for d in docs:
            if d["id"] not in run:
                continue
            g, c, per = scores_tokens(d, run[d["id"]]["tokens"])
            S[name][d["id"]] = {"gate": g, "cov": c, "per_span": per}
        lat[name], mem[name] = latency(run, docs), meta(name)
    for name in DOC_MODELS:
        run = read_run(name)
        if run:
            S[name] = {d["id"]: {"gate": run[d["id"]]["doc"], "cov": run[d["id"]]["doc"], "per_span": None}
                       for d in docs if d["id"] in run}
            lat[name], mem[name] = latency(run, docs), meta(name)
    for name in PROBES:
        for kind in ("lr", "mlp"):
            sc = train_probe(name, docs, kind)
            if sc:
                S[f"{name}-{kind}"] = {k: {"gate": v, "cov": v, "per_span": None} for k, v in sc.items()}
        run = read_run(name)
        if run:
            lat[name], mem[name] = latency(run, docs), meta(name)
    # combinations: regex ∪ model, regex ∪ probe, regex ∪ probe ∪ span model
    if "regex" in S:
        for name in list(S):
            if name != "regex":
                S[f"regex+{name}"] = combine(S["regex"], S[name]) if next(iter(S[name].values()))["per_span"] is not None \
                    else combine({k: dict(v, per_span=None) for k, v in S["regex"].items()}, S[name])
        for p in [n for n in S if n.endswith("-lr") and not n.startswith("regex")]:
            for m in ("nym-small-edge", "bert-small-pii"):
                if m in S:
                    S[f"regex+{m}+{p}"] = combine({k: dict(v, per_span=None) for k, v in S[f"regex+{m}"].items()}, S[p])
    for name, sc in S.items():
        if not sc:
            continue
        spans = next(iter(sc.values()))["per_span"] is not None
        reports[name] = evaluate(name, docs, sc, alpha, delta, spans)
    # the scanner's own decision rule (myriad.privacy_model.decide) replayed on the stored tokens
    regex_run = read_run("regex")
    for m in SPAN_MODELS:
        name = f"regex+{m}"
        if name in reports:
            run = read_run(m)
            r = reports[name]
            r["mask_policy"] = residual_calibration(docs, regex_run, run, alpha, delta)
            r["scanner"] = {}
            mp = r["mask_policy"]
            for key, policy, low in (("block", "block", r["lambda_low"]), ("block-pac", "block", r["lambda_low_pac"]),
                                     ("mask", "mask", mp["lambda_mask"]), ("mask-pac", "mask", mp["lambda_mask_pac"])):
                out = defaultdict(lambda: defaultdict(int))
                for d in docs:
                    if d["split"] != "test" or d["id"] not in run or d["id"] not in regex_run:
                        continue
                    rules = [{"start": t[0], "end": t[1]} for t in regex_run[d["id"]]["tokens"]]
                    toks = [tuple(t) for t in run[d["id"]]["tokens"]]
                    if policy == "mask":
                        toks = residual_tokens(toks, rules)
                    ms = merge_tokens(toks, low, d["text"])
                    if low <= 0:  # PrivacyScanner.scan: no useful threshold, everything is held
                        dec = "local" if rules and policy == "block" else "ask"
                    else:
                        dec = decide(rules, ms, r["lambda_high"], policy)
                    group = d["source"] if not d["guarantee"] else ("pos" if d["label"] else "neg")
                    out[group][dec] += 1
                    if d["source"] == "oasst":
                        out["oasst"][dec] += 1
                r["scanner"][key] = {"low": low, "high": r["lambda_high"], **{k: dict(v) for k, v in out.items()}}
    summary = {
        "alpha": alpha, "delta": delta, "min_cal_pos_marginal": min_n(alpha), "min_cal_pos_pac": min_n(alpha, delta),
        "n_docs": len(docs), "splits": {s: sum(d["split"] == s for d in docs) for s in ("train", "cal", "test")},
        "sources": {s: {"pos": sum(1 for d in docs if d["source"] == s and d["label"]),
                        "neg": sum(1 for d in docs if d["source"] == s and not d["label"])}
                    for s in sorted({d["source"] for d in docs})},
        "detectors": reports, "latency": lat, "memory": mem,
        "licences": {n: MODELS[n]["licence"] for n in MODELS},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "pii_report.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "pii_report.md").write_text(render(summary), encoding="utf-8")
    print(render(summary))


def _p(x, nd=1):
    return "–" if x is None else f"{100 * x:.{nd}f} %"


def _f3(x):
    return "–" if x is None else f"{x:.3f}"


def _ms(x):
    return "–" if x is None else f"{x:.1f}"


def _ci(r, key="test_miss_rate", ci="test_miss_ci95"):
    return f"{_p(r[key], 2)} ({_p(r[ci][0], 2)} – {_p(r[ci][1], 2)})"


def render(s: dict) -> str:
    D = s["detectors"]
    alpha = s["alpha"]
    L = [f"# Détecteurs de données personnelles : résultats sur `test` (α = {alpha:g})", "",
         f"{s['n_docs']} documents ; découpage {s['splits']}.", "",
         "Sources (positifs / négatifs) : " + ", ".join(f"{k} {v['pos']}/{v['neg']}" for k, v in s["sources"].items())
         + ". gretel (étiquettes bruitées) est hors calibration et rapporté à part.", "",
         f"## 1. Politique « bloquer » : seuil conforme à α = {alpha:g} (calibré sur `cal`, mesuré sur `test`)", "",
         "Ratés : documents avec données personnelles dont aucun jeton n'atteint λ (IC 95 % de Clopper-Pearson). "
         "FPR : documents sans donnée personnelle retenus au seuil λ (négatifs du banc : fait main, oasst, "
         "HumanEval) ; « oasst » : vraies premières questions d'utilisateurs, négatifs présumés.", "",
         "| détecteur | λ | ratés test (IC 95 %) | ratés, moyenne de 500 re-tirages cal/test | ratés FR | ratés EN | FPR | "
         "FPR oasst | AUROC | ratés gretel |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in sorted(D.items(), key=lambda kv: (kv[1]["fpr"] if kv[1]["lambda_low"] > 0 else 9, kv[0])):
        bl, bs = r["by_lang"], r["by_source"]
        L.append(f"| {name} | {r['lambda_low']:.3g} | {_ci(r)} | {_p(r['redraws']['mean_miss'], 2)} | "
                 f"{_p((bl.get('fr') or {}).get('miss_rate'))} | "
                 f"{_p((bl.get('en') or {}).get('miss_rate'))} | {_p(r['fpr'])} | {_p((bs.get('oasst') or {}).get('fpr'))} | "
                 f"{_f3(r['auroc'])} | {_p((bs.get('gretel') or {}).get('miss_rate'))} |")
    L += ["", f"### 1 bis. Variante PAC : taux de ratés ≤ α avec probabilité ≥ {1 - s['delta']:g} sur le tirage de `cal`", "",
          "| détecteur | λ PAC | ratés test (IC 95 %) | FPR |", "| --- | --- | --- | --- |"]
    for name, r in sorted(D.items()):
        if name.count("+") <= 1 and r["lambda_low_pac"] > 0:
            L.append(f"| {name} | {r['lambda_low_pac']:.3g} | {_ci(r, 'test_miss_rate_pac', 'test_miss_ci95_pac')} | "
                     f"{_p(r['fpr_pac'])} |")
    L += ["", "## 2. Compromis selon α (même protocole)", "",
          "| détecteur | " + " | ".join(f"α = {a} : λ / ratés / FPR" for a in ("1 %", "2 %", "5 %", "10 %")) + " |",
          "| --- | --- | --- | --- | --- |"]
    for name, r in sorted(D.items()):
        if name.startswith("regex+") and name.count("+") == 1 or name == "nym-small-edge":
            cells = [f"{v['lambda']:.3g} / {_p(v['test_miss_rate'], 2)} / {_p(v['fpr'])}" for v in r["alphas"].values()]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## 3. Au seuil fixe 0,5 (sans calibration)", "",
          "| détecteur | ratés | FPR | ratés FR | ratés EN | FPR oasst | rappel des passages |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in sorted(D.items(), key=lambda kv: kv[0]):
        if "+" in name and name.count("+") > 1:
            continue
        bl, bs = r["by_lang"], r["by_source"]
        L.append(f"| {name} | {_p(1 - r['recall_at_05'], 2)} | {_p(r['fpr_at_05'])} | "
                 f"{_p((bl.get('fr') or {}).get('miss_rate_05'))} | {_p((bl.get('en') or {}).get('miss_rate_05'))} | "
                 f"{_p((bs.get('oasst') or {}).get('fpr_05'))} | {_p(r.get('span_recall_05'))} |")
    types = sorted({t for r in D.values() for t in r.get("span_recall_by_type", {})})
    L += ["", "## 4. Rappel des passages par type (seuil 0,5)", "",
          "| détecteur | " + " | ".join(types) + " |", "| --- |" + " --- |" * len(types)]
    for name, r in sorted(D.items()):
        if "span_recall_by_type" in r and name.count("+") <= 1:
            t = r["span_recall_by_type"]
            L.append(f"| {name} | " + " | ".join(_p(t[x]["recall_05"], 0) if x in t else "–" for x in types) + " |")
    L += ["", "## 5. Politique « masquer » : couverture de tous les passages", "",
          "Raté = au moins un passage personnel sans jeton ≥ λ (on masque tout ce qui dépasse λ).", "",
          "| détecteur | " + " | ".join(f"α = {a} : λ / fuites / FPR" for a in ("1 %", "5 %", "10 %", "20 %")) + " |",
          "| --- | --- | --- | --- | --- |"]
    for name, r in sorted(D.items()):
        if "coverage" in r and name.startswith("regex+") and name.count("+") == 1:
            a = r["coverage"]["alphas"]
            cells = [f"{a[k]['lambda']:.3g} / {_p(a[k]['test_leak_rate'], 2)} / {_p(a[k]['fpr'])}"
                     for k in ("0.01", "0.05", "0.1", "0.2")]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", f"## 5 bis. Politique « masquer » retenue : règles masquées, modèle en porte sur le reste (α = {alpha:g})", "",
          "Fuite = un passage personnel que les règles ne masquent pas entièrement, et aucun jeton du modèle hors "
          "des passages des règles n'atteint λ. « Retenu » : négatifs envoyés ni automatiquement ni masqués "
          "(demander ou local).", "",
          "| combinaison | λ masque | positifs entièrement masqués par les règles (cal) | fuites test (IC 95 %) | "
          "négatifs retenus | λ PAC | fuites PAC | négatifs retenus PAC | négatifs masqués par les règles |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in sorted(D.items()):
        if "mask_policy" in r:
            m = r["mask_policy"]
            L.append(f"| {name} | {m['lambda_mask']:.3g} | {_p(m['rules_cover_all_share_cal'])} | "
                     f"{_ci(m, 'test_leak_rate', 'test_leak_ci95')} | {_p(m['fpr_held'])} | {m['lambda_mask_pac']:.3g} | "
                     f"{_ci(m, 'test_leak_rate_pac', 'test_leak_ci95_pac')} | {_p(m['fpr_held_pac'])} | "
                     f"{_p(m['neg_masked_by_rules'])} |")
    L += ["", "## 6. Décisions du scanneur sur `test` (règle de `privacy_model.decide`)", "",
          "| combinaison | politique | λ bas | λ haut | positifs : envoyer / demander / masquer-local | "
          "négatifs : envoyer / demander / masquer-local | oasst : envoyer / demander / masquer-local |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in sorted(D.items()):
        for pol, v in r.get("scanner", {}).items():
            def trio(g):
                g = v.get(g, {})
                return " / ".join(str(g.get(k, 0)) for k in ("send", "ask")) + f" / {g.get('mask', 0) + g.get('local', 0)}"
            L.append(f"| {name} | {pol} | {v['low']:.3g} | {v['high']:.3g} | {trio('pos')} | {trio('neg')} | {trio('oasst')} |")
    L += ["", "## 7. Coût : mémoire et latence (i5-10400F, 6 cœurs, onnxruntime CPU)", "",
          "RSS mesurée par le système dans un processus neuf : « exécution » = Python + numpy + onnxruntime + "
          "tokenizers importés ; « chargé » = après chargement du modèle ; « pic » = pic de l'ensemble de travail "
          "après les questions faites main et un texte long. Latence : questions faites main (`short`, "
          "≤ 600 caractères) et tous les documents du banc.", "",
          "| détecteur | licence | RSS exécution | RSS chargé | pic | p50 / p95 court (ms) | p50 / p95 tous (ms) | "
          "chargement (s) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in sorted(s["latency"]):
        m = s["memory"].get(name, {})
        mm, me = m.get("mem", {}), m.get("meta", {})
        lt = s["latency"][name]
        L.append(f"| {name} | {s['licences'].get(name, '–')} | {mm.get('rss_runtime_mb', '–')} | "
                 f"{mm.get('rss_loaded_mb', '–')} | {mm.get('peak_mb', '–')} | "
                 f"{_ms(mm.get('short_p50_ms'))} / {_ms(mm.get('short_p95_ms'))} | "
                 f"{_ms(lt.get('all_p50_ms'))} / {_ms(lt.get('all_p95_ms'))} | {me.get('load_s', '–')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(sys.argv[1:])
