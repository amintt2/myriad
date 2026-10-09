"""E4 cascade replay: the swarm answers first, a bigger reference is called only when the swarm is unsure.

    uv run python analyze_cascade_e4.py --suffix _colab

Offline, from the same answers as analyze_e4.py (every model answered every question alone, once, greedy).
The swarm is the 7-family weighted vote of analyze_e4.py (K-class weights fitted on DEV). For each
reference model R (Qwen3.5-9B, Gemma-4-12B, Ministral-3-14B, Qwen3.8-27B), the cascade returns the
swarm's decision, or R's own answer when the swarm "calls" R. Two signals decide the call:
  support  the number of voting peers behind the most-voted answer (unweighted); call when support < m.
           m = 2 ("no answer has two votes") is fixed in advance (docs/07_idees_codex_2.md, E3 replay):
           nothing is tuned for it.
  margin   the weighted score of the decision minus the runner-up's (0 without any answer). Calling when
           margin <= t is calling when the stop certificate with a reserve t fails: the decision would not
           be certified if peers holding a total weight t were still missing (Theorem 1 of the paper).
For each signal and each call BUDGET b (fraction of questions), the threshold (m or t) is chosen on DEV:
the best dev accuracy among the thresholds whose dev call rate is <= b (ties: fewer calls). It is then
frozen and applied to TEST, where accuracy, call rate and paired comparisons (essaim/stats.py: Tango score
interval, exact unconditional tests, margin DELTA points) against the swarm alone and R alone are reported.
Compute proxy: generated tokens times TOTAL parameters (essaim/models.py), summed over the peers that ran,
relative to R alone on the same questions (prompt processing ignored).
Writes results/e4_cascade<suffix>.md and results/e4_cascade_summary<suffix>.json (read by the paper).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import analyze_e4
from analyze_e4 import BENCHES, DELTA, FAMILIES, PARAMS_B, REFS, check_manifests, decide, fit_weights, load
from essaim.stats import compare

RESULTS = Path(__file__).resolve().parent / "results"
BUDGETS = (0.1, 0.2, 0.3)
FIXED_M = 2  # "no answer has two votes": the rule fixed before E4


def signals(answers: list[tuple[str | None, float]]) -> tuple[str | None, int, float]:
    """(decision, support, margin) of a weighted vote. answers: (answer, weight) per peer, in family order.
    Peers without an answer or with a zero weight cast no vote. The decision is analyze_e4.decide's."""
    votes = [(a, w) for a, w in answers if a is not None and w > 0]
    dec = decide([(a, w, None) for a, w in answers])
    count: dict[str, int] = defaultdict(int)
    score: dict[str, float] = defaultdict(float)
    for a, w in votes:
        count[a] += 1
        score[a] += w
    support = max(count.values(), default=0)
    if dec is None:
        return None, support, 0.0
    runner = max((s for a, s in score.items() if a != dec), default=0.0)
    return dec, support, score[dec] - runner


def calls_for(rule: str, threshold: float, support: list[int], margin: list[float]) -> list[bool]:
    if rule == "support":
        return [s < threshold for s in support]
    if rule == "margin":
        return [m <= threshold for m in margin]
    raise ValueError(rule)


def choose(rule: str, sig: dict, swarm_ok: list[float], ref_ok: list[float], budget: float) -> float:
    """Threshold of `rule` with the best accuracy on these questions among those calling at most a
    fraction `budget` of them (ties: fewer calls, then the smaller threshold)."""
    n = len(swarm_ok)
    if rule == "support":
        cands = [float(m) for m in range(0, 9)]  # support < 0 never calls
    else:
        cands = [-1.0] + sorted(set(sig["margin"]))  # margin <= -1 never calls (margins are >= 0)
    best = None
    for t in cands:
        call = calls_for(rule, t, sig["support"], sig["margin"])
        k = sum(call)
        if k > budget * n + 1e-9:
            continue
        acc = sum(r if c else s for c, s, r in zip(call, swarm_ok, ref_ok)) / n
        key = (acc, -k, -t)
        if best is None or key > best[0]:
            best = (key, t)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_colab")
    ap.add_argument("--strict-math", action="store_true",
                    help="MATH-500: \\boxed{} only, as analyze_e4.py --strict-math (weights and thresholds refitted); "
                         "writes *_strictmath files")
    a = ap.parse_args()
    out_suffix = a.suffix + ("_strictmath" if a.strict_math else "")
    if a.strict_math:
        analyze_e4.answers.MATH_FALLBACK = False
    fams = list(FAMILIES)
    out: dict = {"families": FAMILIES, "refs": REFS, "budgets": list(BUDGETS), "fixed_m": FIXED_M, "delta": DELTA,
                 "math_grader": "boxed only" if a.strict_math else "boxed, else final expression (complete answers)",
                 "benches": {}}
    L = ["# E4 : cascade (l'essaim d'abord, un gros modèle seulement en cas de doute)", "",
         "Rejeu hors ligne des réponses de E4. L'essaim est le vote pondéré des 7 familles (poids appris sur dev). "
         "La cascade renvoie la décision de l'essaim, ou la réponse de la référence R quand l'essaim l'appelle. "
         "Signaux : **soutien** (nombre de pairs derrière la réponse la plus votée ; appel si < m, m = 2 fixé "
         "d'avance) et **marge** (score pondéré de la décision moins celui de la deuxième ; appel si marge ≤ t, "
         "c'est-à-dire si le certificat d'arrêt avec une réserve t échoue). Seuils choisis sur **dev** sous un "
         "budget d'appels, appliqués à **test**. Calcul : jetons générés × paramètres totaux, rapporté à R seul.",
         ""]
    # per-question right/wrong vectors of every system, concatenated over the benchmarks (pooled tests)
    pool: dict = {"swarm": [], "refs": {r: {"alone": [], "rules": defaultdict(list)} for r in REFS}}
    if a.strict_math:
        L += ["**Variante MATH-500 stricte** : seul \\boxed{} compte (poids et seuils réappris sur dev).", ""]
    for bench in BENCHES:
        models = list(FAMILIES.values()) + REFS
        dev = {m: load(m, a.suffix, bench, "dev") for m in models}
        test = {m: load(m, a.suffix, bench, "test") for m in models}
        missing = [m for m in models if dev[m] is None or test[m] is None]
        if missing:
            raise SystemExit(f"{bench} : runs manquants {missing}")
        check_manifests(bench, models)
        ids = {"dev": sorted(dev[FAMILIES[fams[0]]]), "test": sorted(test[FAMILIES[fams[0]]])}
        rows = {"dev": dev, "test": test}
        for split in ("dev", "test"):
            for m in models:
                if sorted(rows[split][m]) != ids[split]:
                    raise SystemExit(f"{m} {bench} {split} : pas les mêmes questions")
        _, _, c, w = fit_weights({f: dev[FAMILIES[f]] for f in fams}, ids["dev"])
        sig, swarm_ok, gold = {}, {}, {}
        for split in ("dev", "test"):
            R = rows[split]
            gold[split] = [R[FAMILIES[fams[0]]][i]["gold"] for i in ids[split]]
            s = [signals([(R[FAMILIES[f]][i]["answer"], w[f]) for f in fams]) for i in ids[split]]
            sig[split] = {"support": [x[1] for x in s], "margin": [x[2] for x in s]}
            swarm_ok[split] = [float(x[0] == g) for x, g in zip(s, gold[split])]
        swarm_cost = [sum(PARAMS_B[FAMILIES[f]] * (test[FAMILIES[f]][i]["n_tokens"] or 0) for f in fams)
                      for i in ids["test"]]
        n = len(ids["test"])
        pool["swarm"] += swarm_ok["test"]
        b_out = {"n": n, "swarm": 100 * st.mean(swarm_ok["test"]), "refs": {}}
        L += [f"## {bench} ({n} questions test)", "", f"Essaim seul (vote pondéré) : {b_out['swarm']:.1f} %.", "",
              "| référence | règle (seuil choisi sur dev) | appels dev (%) | appels test (%) | exactitude test (%) | "
              "réparées / cassées | moins l'essaim [IC 95 %] | moins R seul [IC 95 %] | verdict contre R | calcul / R |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for ref in REFS:
            ref_ok = {sp: [float(rows[sp][ref][i]["answer"] == g) for i, g in zip(ids[sp], gold[sp])]
                      for sp in ("dev", "test")}
            pool["refs"][ref]["alone"] += ref_ok["test"]
            ref_cost = [PARAMS_B[ref] * (test[ref][i]["n_tokens"] or 0) for i in ids["test"]]
            rules = [("support", f"support<{FIXED_M}", float(FIXED_M), None)]
            for b in BUDGETS:
                for rule in ("support", "margin"):
                    t = choose(rule, sig["dev"], swarm_ok["dev"], ref_ok["dev"], b)
                    rules.append((rule, f"{rule}@{int(100 * b)}%", t, b))
            r_out = {"alone": 100 * st.mean(ref_ok["test"]), "rules": {}}
            for rule, name, t, b in rules:
                call = {sp: calls_for(rule, t, sig[sp]["support"], sig[sp]["margin"]) for sp in ("dev", "test")}
                ok = [r if k else s for k, s, r in zip(call["test"], swarm_ok["test"], ref_ok["test"])]
                pool["refs"][ref]["rules"][name] += ok
                rescued = sum(1 for k, s, r in zip(call["test"], swarm_ok["test"], ref_ok["test"]) if k and r and not s)
                broken = sum(1 for k, s, r in zip(call["test"], swarm_ok["test"], ref_ok["test"]) if k and s and not r)
                g_swarm, g_ref = compare(ok, swarm_ok["test"], DELTA), compare(ok, ref_ok["test"], DELTA)
                cost = (sum(swarm_cost) + sum(rc for k, rc in zip(call["test"], ref_cost) if k)) / sum(ref_cost)
                r_out["rules"][name] = {
                    "signal": rule, "threshold": t, "budget": b, "calls_dev": 100 * st.mean(call["dev"]),
                    "calls_test": 100 * st.mean(call["test"]), "accuracy": 100 * st.mean(ok), "rescued": rescued,
                    "broken": broken, "vs_swarm": g_swarm, "vs_ref": g_ref, "compute_vs_ref": cost}
                L.append(f"| {ref} ({r_out['alone']:.1f} %) | {name} ({t:g}) | {100 * st.mean(call['dev']):.1f} | "
                         f"{100 * st.mean(call['test']):.1f} | {100 * st.mean(ok):.1f} | {rescued} / {broken} | "
                         f"{g_swarm['mean']:+.1f} [{g_swarm['lo95']:+.1f} ; {g_swarm['hi95']:+.1f}] | "
                         f"{g_ref['mean']:+.1f} [{g_ref['lo95']:+.1f} ; {g_ref['hi95']:+.1f}] | {g_ref['verdict']} | "
                         f"{cost:.2f} |")
            r_out["swarm_compute_vs_ref"] = sum(swarm_cost) / sum(ref_cost)
            b_out["refs"][ref] = r_out
        L.append("")
        out["benches"][bench] = b_out
    # Macro averages over the four benchmarks (equal weight per benchmark, as in analyze_e4.py).
    out["mean"] = {"swarm": st.mean(out["benches"][b]["swarm"] for b in BENCHES), "refs": {}}
    L += ["## Moyenne des quatre benchmarks", "", "| référence | règle | appels test (%) | exactitude test (%) | "
          "R seul (%) | calcul / R |", "| --- | --- | --- | --- | --- | --- |"]
    for ref in REFS:
        names = list(out["benches"][BENCHES[0]]["refs"][ref]["rules"])
        m_ref = {"alone": st.mean(out["benches"][b]["refs"][ref]["alone"] for b in BENCHES),
                 "swarm_compute_vs_ref": st.mean(out["benches"][b]["refs"][ref]["swarm_compute_vs_ref"] for b in BENCHES),
                 "rules": {}}
        for name in names:
            rs = [out["benches"][b]["refs"][ref]["rules"][name] for b in BENCHES]
            m_ref["rules"][name] = {k: st.mean(r[k] for r in rs) for k in ("calls_test", "accuracy", "compute_vs_ref")}
            m = m_ref["rules"][name]
            L.append(f"| {ref} | {name} | {m['calls_test']:.1f} | {m['accuracy']:.1f} | {m_ref['alone']:.1f} | "
                     f"{m['compute_vs_ref']:.2f} |")
        out["mean"]["refs"][ref] = m_ref
    L.append(f"\nEssaim seul : {out['mean']['swarm']:.1f} % en moyenne.")
    # Pooled paired tests over all test questions of the four benchmarks (each question counts once, so the
    # pooled accuracy weights the benchmarks by their sizes, unlike the macro average above).
    sw = pool["swarm"]
    out["pooled"] = {"n": len(sw), "swarm": 100 * st.mean(sw), "refs": {}}
    L += ["", f"## Tests appariés sur les {len(sw)} questions test réunies", "",
          "| référence | système | exactitude (%) | moins R seul [IC 95 %] | verdict | moins l'essaim [IC 95 %] |",
          "| --- | --- | --- | --- | --- | --- |"]
    for ref in REFS:
        alone = pool["refs"][ref]["alone"]
        g = compare(sw, alone, DELTA)
        p_ref = {"alone": 100 * st.mean(alone), "swarm_vs_ref": g, "rules": {}}
        L.append(f"| {ref} ({p_ref['alone']:.1f} %) | essaim seul | {100 * st.mean(sw):.1f} | "
                 f"{g['mean']:+.1f} [{g['lo95']:+.1f} ; {g['hi95']:+.1f}] | {g['verdict']} | |")
        for name, ok in pool["refs"][ref]["rules"].items():
            gr, gs = compare(ok, alone, DELTA), compare(ok, sw, DELTA)
            p_ref["rules"][name] = {"accuracy": 100 * st.mean(ok), "vs_ref": gr, "vs_swarm": gs}
            L.append(f"| {ref} | cascade {name} | {100 * st.mean(ok):.1f} | {gr['mean']:+.1f} [{gr['lo95']:+.1f} ; "
                     f"{gr['hi95']:+.1f}] | {gr['verdict']} | {gs['mean']:+.1f} [{gs['lo95']:+.1f} ; {gs['hi95']:+.1f}] |")
        out["pooled"]["refs"][ref] = p_ref
    (RESULTS / f"e4_cascade{out_suffix}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / f"e4_cascade_summary{out_suffix}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                                  encoding="utf-8")
    print(f"results/e4_cascade{out_suffix}.md, results/e4_cascade_summary{out_suffix}.json")


if __name__ == "__main__":
    analyze_e4.PARTIAL = False
    main()
