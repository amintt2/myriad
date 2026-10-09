"""Fuse the per-model letter probabilities and compare with each model alone.

Two protocols, both reported:
  1 ordre   each pass (one option order) is scored on its own; mean, worst and best pass.
  5 ordres  per question, the passes' distributions are averaged into one decision.
            This "permutation ensemble" is given to single models too, so it is not
            mistaken for a gain of the swarm.

Fusion with learned temperatures uses two-fold cross-fitting by QUESTION (all passes of a
question stay on the same side). Uncertainty: paired bootstrap over questions that refits
the temperatures in every replicate (the whole procedure is resampled).

    uv run python analyze_mc.py --split dev --experts Qwen/Qwen3-1.7B@_gpu google/gemma-4-E2B-it --reference Qwen/Qwen3-4B@_gpu

A spec is MODEL[@SUFFIX]; the file read is results/mc_<MODEL><SUFFIX>_<SPLIT>.jsonl.
Writes results/mc_report_<split>.md and results/mc_summary_<split>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from essaim.results import read_manifest, read_rows

RESULTS = Path(__file__).resolve().parent / "results"
TITLES = {"arc": "ARC-Challenge (4 choix)", "mmlupro": "MMLU-Pro (10 choix)"}
TGRID = np.exp(np.linspace(np.log(0.25), np.log(20), 60))


def spec_path(spec: str, split: str) -> Path:
    model, _, suffix = spec.partition("@")
    return RESULTS / f"mc_{model.replace('/', '__')}{suffix}_{split}.jsonl"


# Manifest fields that must be identical across all files fused together (same protocol and data).
PROTOCOL = ("split", "n", "runs", "repetition", "prompt", "template_date", "data", "budget", "temperature")


def load(spec: str, split: str) -> dict:
    f = spec_path(spec, split)
    if not f.exists():
        raise SystemExit(f"fichier absent : {f.name}")
    if read_manifest(f) is None:
        raise SystemExit(f"{f.name} n'a pas de manifeste : provenance inconnue, analyse refusée")
    return {(r["bench"], r["id"], r.get("run", 0)): r for r in read_rows(f, key=("bench", "id", "run"))}


IDENTITY = ("model", "revision", "backend", "gguf", "weights", "engine", "device", "dtype")
N_OPTIONS = {"arc": 4, "mmlupro": 10}


def check_protocols(specs: list[str], split: str) -> dict:
    """All fused files must share the same protocol and data identity. Returns the common manifest."""
    mans = {s: read_manifest(spec_path(s, split)) for s in specs}
    ref = mans[specs[0]]
    for s in specs[1:]:
        diff = {k: (ref.get(k), mans[s].get(k)) for k in PROTOCOL if ref.get(k) != mans[s].get(k)}
        if diff:
            raise SystemExit(f"protocoles incompatibles entre {specs[0]} et {s} : {diff}")
    return ref


def check_across_splits(specs: list[str], fit_split: str, split: str):
    """Temperatures learned on fit_split may only be applied to the same experts under the same protocol."""
    for s in specs:
        a, b = read_manifest(spec_path(s, fit_split)), read_manifest(spec_path(s, split))
        # "data" identifies the whole cache before partitioning, so it must match between splits too
        keys = IDENTITY + tuple(k for k in PROTOCOL if k not in ("split", "n"))
        diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
        if diff:
            raise SystemExit(f"{s} : {fit_split} et {split} ne sont pas mesurés de la même façon : {diff}")


def repetition(specs: list[str], split: str) -> str:
    kinds = {(read_manifest(spec_path(s, split)) or {}).get("repetition", "inconnu") for s in specs}
    return " / ".join(sorted(kinds))


def logsoftmax(x: np.ndarray) -> np.ndarray:
    return x - np.logaddexp.reduce(x, axis=-1, keepdims=True)


def fit_temperature(logp: np.ndarray, gold: np.ndarray) -> float:
    """logp: (Q, R, M) log-probs of training questions over all passes; gold: (Q,)."""
    q = logsoftmax(logp[None] / TGRID[:, None, None, None])           # (T, Q, R, M)
    nll = -np.take_along_axis(q, gold[None, :, None, None], -1).mean(axis=(1, 2, 3))
    return float(TGRID[nll.argmin()])


def entropy_weight(p: np.ndarray) -> np.ndarray:
    h = -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1)
    return 1 - h / np.log(p.shape[-1])


def fuse_fixed(name: str, P: np.ndarray) -> np.ndarray:
    """Untrained fusions. P: (K, ..., M) -> scores (..., M)."""
    if name == "moyenne":
        return P.mean(0)
    if name == "produit":
        return np.log(np.clip(P, 1e-12, 1)).sum(0)
    if name == "pondérée par la confiance":
        return (entropy_weight(P)[..., None] * P).sum(0)
    if name == "vote":
        votes = (P == P.max(-1, keepdims=True)).astype(float).sum(0)
        return votes + 1e-3 * P.mean(0)
    raise KeyError(name)


FIXED = ["moyenne", "produit", "pondérée par la confiance", "vote"]
LEARNED = ["produit calibré", "mélange calibré"]


class FoldError(ValueError):
    pass


def _apply(LP: np.ndarray, temps: list[float]) -> dict[str, np.ndarray]:
    cal = np.stack([logsoftmax(LP[k] / temps[k]) for k in range(LP.shape[0])])
    return {"produit calibré": cal.sum(0).argmax(-1), "mélange calibré": np.exp(cal).mean(0).argmax(-1)}


def predict_learned(LP: np.ndarray, gold: np.ndarray, fold: np.ndarray, frozen: list[float] | None = None) -> dict[str, np.ndarray]:
    """LP: (K, Q, R, M) log-probs; returns predictions (Q, R) for each learned fusion.
    `frozen`: temperatures fitted elsewhere (dev) and applied as is. Otherwise two-fold
    cross-fitting: temperatures fitted on the other fold of questions (all their passes)."""
    if frozen is not None:
        return _apply(LP, frozen)
    K, Q, R, M = LP.shape
    out = {name: np.zeros((Q, R), dtype=int) for name in LEARNED}
    for f in (0, 1):
        train, test = fold != f, fold == f
        if not test.any() or not train.any():
            raise FoldError("un pli est vide")
        temps = [fit_temperature(LP[k][train], gold[train]) for k in range(K)]
        for name, pred in _apply(LP[:, test], temps).items():
            out[name][test] = pred
    return out


def evaluate(LP_exp, LP_ref, gold, fold, frozen=None):
    """Accuracies for one (possibly bootstrapped) set of questions.
    LP_exp: (K, Q, R, M); returns dict system -> {'pass': (R,) acc per pass, 'avg': acc of averaged decision}."""
    P_exp = np.exp(LP_exp)
    res = {}
    for i in range(LP_exp.shape[0]):
        res[("seul", i)] = P_exp[i]
    for j in range(LP_ref.shape[0]):
        res[("ref", j)] = np.exp(LP_ref[j])
    acc = {}
    for key, P in res.items():  # single models
        acc[key] = {"pass": (P.argmax(-1) == gold[:, None]).mean(0), "avg": (P.mean(1).argmax(-1) == gold).mean()}
    for name in FIXED:
        acc[name] = {"pass": (fuse_fixed(name, P_exp).argmax(-1) == gold[:, None]).mean(0),
                     "avg": (fuse_fixed(name, P_exp.mean(2)).argmax(-1) == gold).mean()}
    frozen = frozen or {}
    learned_pass = predict_learned(LP_exp, gold, fold, frozen.get("pass"))
    LP_avg = avg_passes(LP_exp)  # averaged passes as one pass
    learned_avg = predict_learned(LP_avg, gold, fold, frozen.get("avg"))
    for name in LEARNED:
        acc[name] = {"pass": (learned_pass[name] == gold[:, None]).mean(0),
                     "avg": (learned_avg[name][:, 0] == gold).mean()}
    top1 = np.stack([P_exp[i].argmax(-1) == gold[:, None] for i in range(P_exp.shape[0])]).any(0)
    acc["plafond top-1"] = {"pass": top1.mean(0), "avg": np.nan}
    return acc


def avg_passes(LP: np.ndarray) -> np.ndarray:
    """(K, Q, R, M) log-probs -> (K, Q, 1, M): one decision per question from the averaged distributions."""
    return np.log(np.clip(np.exp(LP).mean(2, keepdims=True), 1e-12, 1))


def fmt_pass(v: np.ndarray) -> str:
    return f"{v.mean() * 100:.1f} ({v.min() * 100:.1f}–{v.max() * 100:.1f})"


def build(runs: dict, specs: list[str], bench: str, manifest: dict):
    """Requires every spec to hold exactly the manifest's n questions x runs passes for this bench.
    Returns (ids, passes, gold, {spec: LP (Q, R, M)}, dropped=0)."""
    expected_runs = list(range(int(manifest["runs"])))
    for s in specs:
        own = {k for k in runs[s] if k[0] == bench}
        qids = {k[1] for k in own}
        if len(qids) != int(manifest["n"]) or own != {(bench, q, p) for q in qids for p in expected_runs}:
            raise SystemExit(f"{s} {bench} : {len(own)} lignes pour {len(qids)} questions, attendu "
                             f"{manifest['n']} questions x {len(expected_runs)} passages complets")
    keys = [set(k for k in r if k[0] == bench) for r in runs.values()]
    common = set.intersection(*keys) if keys else set()
    if len(common) != int(manifest["n"]) * len(expected_runs):
        raise SystemExit(f"{bench} : les fichiers ne portent pas sur les mêmes questions")
    passes = expected_runs
    ids = sorted({k[1] for k in common})
    dropped = 0
    for s in specs:
        for i in ids:
            for p in passes:
                r = runs[s][(bench, i, p)]
                if len(r["probs"]) != N_OPTIONS[bench] or not 0 <= int(r["answer"]) < N_OPTIONS[bench]:
                    raise SystemExit(f"{s} {bench} {i} : {len(r['probs'])} options ou réponse {r['answer']} invalides")
    for i in ids:  # every file must agree on the gold answer of every question
        answers = {runs[s][(bench, i, p)]["answer"] for s in specs for p in passes}
        if len(answers) != 1:
            raise SystemExit(f"{bench} {i} : bonnes réponses différentes selon les fichiers {answers}")
    gold = np.array([runs[specs[0]][(bench, i, passes[0])]["answer"] for i in ids])
    LP = {}
    for s in specs:
        P = np.array([[runs[s][(bench, i, p)]["probs"] for p in passes] for i in ids], dtype=float)
        if P.ndim != 3 or not np.isfinite(P).all() or (P < 0).any() or np.abs(P.sum(-1) - 1).max() > 1e-3:
            raise SystemExit(f"{s} {bench} : distributions invalides (forme {P.shape}, non finies ou non normalisées)")
        LP[s] = np.log(np.clip(P, 1e-12, 1))
    return ids, passes, gold, LP, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--fit-split", default=None,
                    help="learn the temperatures once on this split (e.g. dev) and apply them frozen to --split")
    ap.add_argument("--experts", nargs="+", required=True)
    ap.add_argument("--reference", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-questions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    specs = a.experts + a.reference
    if a.fit_split == a.split:
        raise SystemExit("--fit-split doit être un autre jeu que --split")
    runs = {s: load(s, a.split) for s in specs}
    manifest = check_protocols(specs, a.split)
    fit_runs = fit_manifest = None
    if a.fit_split:
        fit_runs = {s: load(s, a.fit_split) for s in a.experts}
        fit_manifest = check_protocols(a.experts, a.fit_split)
        check_across_splits(a.experts, a.fit_split, a.split)
        fit_ids = {k[:2] for r in fit_runs.values() for k in r}
        eval_ids = {k[:2] for r in runs.values() for k in r}
        if fit_ids & eval_ids:
            raise SystemExit(f"{len(fit_ids & eval_ids)} questions communes entre {a.fit_split} et {a.split} : "
                             f"les températures ne seraient pas apprises sur des données séparées")
    rep = repetition(specs, a.split)
    summary = {"split": a.split, "fit_split": a.fit_split, "experts": a.experts, "reference": a.reference,
               "repetition": rep, "benches": {}}
    how_fit = (f"apprennent une température par modèle sur le jeu **{a.fit_split}**, puis l'appliquent telle quelle "
               f"(méthode figée)" if a.fit_split else
               "apprennent une température par modèle sur une moitié des questions et sont évaluées sur l'autre "
               "(validation croisée par question)")
    rng = np.random.default_rng(a.seed)
    lines = [f"# Phase 0, mode B : fusion de familles différentes (QCM, jeu {a.split})", "",
             f"Exactitude en %. Répétitions : {rep}. **1 passage** : chaque passage noté séparément, moyenne "
             "(pire–meilleur passage). **Moyenne des passages** : les distributions des passages sont moyennées en "
             "une seule décision par question, pour chaque système, modèles seuls compris (cet ensemble gratuit "
             f"est donc aussi donné aux modèles seuls). Les fusions « calibrées » {how_fit}. "
             "IC à 95 % : bootstrap apparié par question" +
             ("." if a.fit_split else ", procédure entière réajustée à chaque tirage.") +
             " Le meilleur expert est choisi séparément pour chaque protocole, sur ces mêmes données.", ""]
    for bench, title in TITLES.items():
        ids, passes, gold, LP, dropped = build(runs, specs, bench, manifest)
        if len(ids) < a.min_questions:
            if ids:
                lines += [f"## {title}", "", f"Seulement {len(ids)} questions complètes : analyse non faite.", ""]
            continue
        R = len(passes)
        LP_exp = np.stack([LP[s] for s in a.experts])
        LP_ref = np.stack([LP[s] for s in a.reference]) if a.reference else np.zeros((0,) + LP_exp.shape[1:])
        fold = np.arange(len(ids)) % 2
        frozen = None
        if fit_runs is not None:
            fids, fpasses, fgold, fLP, _ = build(fit_runs, a.experts, bench, fit_manifest)
            if len(fids) < a.min_questions:
                raise SystemExit(f"jeu {a.fit_split} insuffisant pour {bench}")
            fLP_exp = np.stack([fLP[s] for s in a.experts])
            fLP_avg = avg_passes(fLP_exp)
            frozen = {"pass": [fit_temperature(fLP_exp[k], fgold) for k in range(len(a.experts))],
                      "avg": [fit_temperature(fLP_avg[k], fgold) for k in range(len(a.experts))]}
        acc = evaluate(LP_exp, LP_ref, gold, fold, frozen)
        K = len(a.experts)
        best = {p: int(np.argmax([np.mean(acc[("seul", i)][p]) for i in range(K)])) for p in ("pass", "avg")}

        # paired bootstrap over questions; folds follow the original question, so copies stay together
        gains = {name: {"pass": [], "avg": []} for name in FIXED + LEARNED}
        gains_each = {name: {p: [[] for _ in range(K)] for p in ("pass", "avg")} for name in LEARNED}
        gains_ref = {name: {"pass": [], "avg": []} for name in LEARNED}
        done = 0
        while done < a.boot:
            idx = rng.integers(0, len(ids), len(ids))
            try:
                ab = evaluate(LP_exp[:, idx], LP_ref[:, idx], gold[idx], fold[idx], frozen)
            except FoldError:
                continue
            done += 1
            for name in FIXED + LEARNED:
                for p in ("pass", "avg"):
                    gains[name][p].append(np.mean(ab[name][p]) - np.mean(ab[("seul", best[p])][p]))
            for name in LEARNED:
                for p in ("pass", "avg"):
                    for i in range(K):
                        gains_each[name][p][i].append(np.mean(ab[name][p]) - np.mean(ab[("seul", i)][p]))
                    if a.reference:
                        gains_ref[name][p].append(np.mean(ab[name][p]) - np.mean(ab[("ref", 0)][p]))
        ci = lambda xs: (float(np.percentile(xs, 2.5)) * 100, float(np.percentile(xs, 97.5)) * 100)
        pt = lambda name, p, base: (np.mean(acc[name][p]) - np.mean(acc[base][p])) * 100

        lab = lambda s: s.replace("@", " ")
        lines += [f"## {title} : {len(ids)} questions × {R} passages" +
                  (f" ({dropped} questions incomplètes écartées)" if dropped else ""), "",
                  f"| système | 1 passage | moyenne des {R} passages |", "| --- | --- | --- |"]
        for i, s in enumerate(a.experts):
            tags = [n for n, p in (("meilleur, 1 passage", "pass"), ("meilleur, moyenne", "avg")) if best[p] == i]
            lines.append(f"| {lab(s)} seul{' (' + ' ; '.join(tags) + ')' if tags else ''} | "
                         f"{fmt_pass(acc[('seul', i)]['pass'])} | {acc[('seul', i)]['avg'] * 100:.1f} |")
        for j, s in enumerate(a.reference):
            lines.append(f"| {lab(s)} seul (référence locale) | {fmt_pass(acc[('ref', j)]['pass'])} | {acc[('ref', j)]['avg'] * 100:.1f} |")
        for name in FIXED + LEARNED:
            lines.append(f"| fusion : {name} | {fmt_pass(acc[name]['pass'])} | {acc[name]['avg'] * 100:.1f} |")
        lines.append(f"| plafond top-1 (au moins un expert a raison) | {fmt_pass(acc['plafond top-1']['pass'])} | — |")
        lines += ["", f"Gain sur le meilleur expert de chaque protocole (1 passage : {lab(a.experts[best['pass']])} ; "
                      f"moyenne : {lab(a.experts[best['avg']])}), en points, IC 95 % :", "",
                  "| fusion | 1 passage | moyenne des passages |", "| --- | --- | --- |"]
        point = {name: {p: pt(name, p, ("seul", best[p])) for p in ("pass", "avg")} for name in FIXED + LEARNED}
        for name in FIXED + LEARNED:
            c1, c2 = ci(gains[name]["pass"]), ci(gains[name]["avg"])
            lines.append(f"| {name} | {point[name]['pass']:+.1f} [{c1[0]:+.1f} ; {c1[1]:+.1f}] | "
                         f"{point[name]['avg']:+.1f} [{c2[0]:+.1f} ; {c2[1]:+.1f}] |")
        lines += ["", "Gain sur chaque expert (fusions calibrées), borne basse de l'IC à 95 % unilatéral "
                      "(percentile 5 %) : « bat tous les experts » n'est soutenu que si toutes sont positives.", ""]
        for name in LEARNED:
            for p, pl in (("pass", "1 passage"), ("avg", "moyenne")):
                lows = [float(np.percentile(gains_each[name][p][i], 5)) * 100 for i in range(K)]
                lines.append(f"- {name}, {pl} : " + ", ".join(f"{lab(s).split('/')[-1]} {lo:+.1f}"
                                                               for s, lo in zip(a.experts, lows)))
        if a.reference:
            lines += ["", f"Gain sur la référence locale ({lab(a.reference[0])}) :"]
            for name in LEARNED:
                c1, c2 = ci(gains_ref[name]["pass"]), ci(gains_ref[name]["avg"])
                lines.append(f"- {name} : 1 passage {pt(name, 'pass', ('ref', 0)):+.1f} [{c1[0]:+.1f} ; {c1[1]:+.1f}], "
                             f"moyenne {pt(name, 'avg', ('ref', 0)):+.1f} [{c2[0]:+.1f} ; {c2[1]:+.1f}]")
        if frozen:
            lines += ["", "Températures figées (apprises sur " + a.fit_split + ") : " +
                      ", ".join(f"{lab(s).split('/')[-1]} {t:.2f}" for s, t in zip(a.experts, frozen["pass"]))]
        lines.append("")

        summary["benches"][bench] = {
            "questions": len(ids), "passes": R, "dropped_incomplete": dropped,
            "best_expert": {p: a.experts[best[p]] for p in best},
            "systems": {**{f"seul:{s}": {"pass": acc[("seul", i)]["pass"].tolist(), "avg": float(acc[("seul", i)]["avg"])}
                           for i, s in enumerate(a.experts)},
                        **{f"ref:{s}": {"pass": acc[("ref", j)]["pass"].tolist(), "avg": float(acc[("ref", j)]["avg"])}
                           for j, s in enumerate(a.reference)},
                        **{f"fusion:{n}": {"pass": acc[n]["pass"].tolist(), "avg": float(acc[n]["avg"])} for n in FIXED + LEARNED}},
            "gain_vs_best": {n: {p: {"points": float(point[n][p]), "ci95": ci(gains[n][p])} for p in ("pass", "avg")}
                             for n in FIXED + LEARNED},
            "frozen_temperatures": frozen,
        }

    (RESULTS / f"mc_summary_{a.split}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / f"mc_report_{a.split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
