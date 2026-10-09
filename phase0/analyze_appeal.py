"""E5 analysis: does the minority appeal recover correct runners-up without breaking correct answers?

    uv run python analyze_appeal.py --bench arc

The appeal replaces the fused answer a by the runner-up b when the MEDIAN of the judges' order-averaged
log-ratios ell = log P(b)/P(a) exceeds tau. tau is chosen on dev (the value maximising the dev gain,
"never replace" included), then applied unchanged on test. By the appeal identity the accuracy gain over
ALL questions is exactly B*r - C*h, with B = P(appeal, b correct), C = P(appeal, a correct), r and h the
replacement rates in those two cases. Paired bootstrap 95 % intervals over all questions of the split.
Writes results/appeal_report_<bench>.md and results/appeal_summary_<bench>.json.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from essaim.results import read_manifest, read_rows
from run_appeal import appeal_set

RESULTS = Path(__file__).resolve().parent / "results"


def judged(bench: str, split: str) -> tuple[list[dict], int]:
    cases, meta = appeal_set(bench, split)
    ells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for j in sorted({j for c in cases for j in c["judges"]}):
        path = RESULTS / f"appeal_{j.replace('/', '__')}_{bench}_{split}.jsonl"
        man = read_manifest(path)
        if man is None:
            raise SystemExit(f"{path.name} : absent")
        stale = {k: (man.get(k), meta.get(k)) for k in meta if man.get(k) != meta.get(k)}
        if stale:  # judged on another appeal population (other MC results or temperatures)
            raise SystemExit(f"{path.name} : jugé sur une autre configuration d'appel : {stale}")
    by_judge = {}
    for c in cases:
        for j in c["judges"]:
            if j not in by_judge:
                by_judge[j] = {(r["id"], r["order"]): r for r in read_rows(
                    RESULTS / f"appeal_{j.replace('/', '__')}_{bench}_{split}.jsonl", key=("id", "order"))}
            rows = by_judge[j]
            if (c["id"], "ab") not in rows or (c["id"], "ba") not in rows:
                raise SystemExit(f"{j} n'a pas jugé {c['id']} dans les deux ordres")
            for o in ("ab", "ba"):  # the judged pair must be the reconstructed one
                if (rows[(c["id"], o)]["a"], rows[(c["id"], o)]["b"]) != (c["a"], c["b"]):
                    raise SystemExit(f"{j} {c['id']} : paire jugée différente de la paire reconstruite")
            ells[c["id"]][j] = [rows[(c["id"], "ab")]["ell"], rows[(c["id"], "ba")]["ell"]]
    for c in cases:
        c["score"] = st.median(sum(v) / 2 for v in ells[c["id"]].values())
    return cases, meta["questions_total"]


def effect(cases: list[dict], n: int, tau: float) -> dict:
    flip = [c for c in cases if c["score"] > tau]
    B = sum(c["b"] == c["gold"] for c in cases)
    C = sum(c["a"] == c["gold"] for c in cases)
    rb = sum(c["b"] == c["gold"] for c in flip)
    ha = sum(c["a"] == c["gold"] for c in flip)
    return {"tau": tau, "appeals": len(cases), "flips": len(flip), "B": B / n, "C": C / n,
            "r": rb / B if B else 0.0, "h": ha / C if C else 0.0, "gain_points": 100 * (rb - ha) / n}


def boot_gain(cases: list[dict], n: int, tau: float, b: int, rng: random.Random) -> list[float]:
    per_q = [0.0] * n  # questions without an appeal contribute 0
    for k, c in enumerate(cases):
        if c["score"] > tau:
            per_q[k] = float(c["b"] == c["gold"]) - float(c["a"] == c["gold"])
    means = sorted(sum(per_q[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return [100 * means[int(0.025 * b)], 100 * means[int(0.975 * b) - 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    dev, n_dev = judged(a.bench, "dev")
    test, n_test = judged(a.bench, "test")
    grid = [float("-inf")] + sorted({c["score"] for c in dev}) + [float("inf")]  # replace-all ... never
    best = max(grid, key=lambda t: (effect(dev, n_dev, t)["gain_points"], t))  # ties: replace less
    e_dev, e_test = effect(dev, n_dev, best), effect(test, n_test, best)
    ci = boot_gain(test, n_test, best, a.boot, random.Random(0))
    L = [f"# E5 : appel du minoritaire ({a.bench})", "",
         "Seuil choisi sur dev, appliqué tel quel sur test. Gain exact = B·r − C·h (en points, sur toutes les questions).", "",
         "| jeu | appels | renversements | B | C | r | h | gain (points) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, e in (("dev", e_dev), ("test", e_test)):
        L.append(f"| {name} | {e['appeals']} | {e['flips']} | {100 * e['B']:.1f} % | "
                 f"{100 * e['C']:.1f} % | {100 * e['r']:.1f} % | {100 * e['h']:.1f} % | {e['gain_points']:+.1f} |")
    L += ["", f"Seuil τ = {best:.3f} ; gain sur test {e_test['gain_points']:+.1f} points, IC 95 % [{ci[0]:+.1f} ; {ci[1]:+.1f}]."]
    curve = [effect(test, n_test, t) for t in sorted({c["score"] for c in test})[::max(1, len(test) // 15)]]
    L += ["", "Courbe sur test (exploratoire) :", "", "| τ | renversements | r | h | gain |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {e['tau']:.2f} | {e['flips']} | {100 * e['r']:.0f} % | {100 * e['h']:.0f} % | {e['gain_points']:+.1f} |" for e in curve]
    (RESULTS / f"appeal_report_{a.bench}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / f"appeal_summary_{a.bench}.json").write_text(json.dumps(
        {"tau": best, "dev": e_dev, "test": e_test, "test_ci95": ci}, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
