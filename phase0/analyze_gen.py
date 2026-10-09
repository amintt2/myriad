"""Experiment 2 analysis (GSM8K): accuracy of each way of answering, its cost in round trips and time.

    uv run python analyze_gen.py --split dev --tag essaim4_colab --reference ref-Qwen3-4B_colab

Reads results/gen_<tag>_<split>.jsonl (written by run_gen.py) and, optionally, the reference
peer's solo answers on the same questions. Every question must have every mode, and the reference
must cover exactly the same questions. Gains are paired bootstrap 95 % intervals over questions.
The best solo peer is picked on these same data, which favours it (the gains are conservative).

Round trips are counted per question; the projected time on a wide-area network adds one RTT per
round trip to the measured decision time (the peers ran on one machine, so the measured network
time is close to zero).
Writes results/gen_report_<tag>_<split>.md and results/gen_summary_<tag>_<split>.json.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from essaim.answers import regrade_gen
from essaim.results import read_manifest, read_rows

GRADER = "v2"

RESULTS = Path(__file__).resolve().parent / "results"
TOGETHER = ("vote", "accord", "croise")
LABEL = {"vote": "vote (majorité des réponses seules)", "accord": "accord (préfixe commun, 1 aller-retour par tour)",
         "croise": "croisé (brouillons notés par tous, 2 allers-retours par tour)"}
RTTS = (50, 100, 150)


def load(tag: str, split: str) -> tuple[dict, dict[tuple[str, str], dict]]:
    path = RESULTS / f"gen_{tag}_{split}.jsonl"
    manifest = read_manifest(path)
    if manifest is None:
        raise SystemExit(f"{path.name} : manifeste absent")
    if manifest.get("split") != split:
        raise SystemExit(f"{path.name} : le manifeste dit split={manifest.get('split')}")
    rows = read_rows(path, key=("mode", "id"))
    if GRADER == "v2":  # re-grade from the stored texts (essaim/answers.regrade_gen)
        rows = regrade_gen(rows, manifest.get("max_rounds"), manifest.get("k"))
    return manifest, {(r["mode"], r["id"]): r for r in rows}


def boot(diff: list[float], b: int, rng: random.Random) -> tuple[float, float, float]:
    """Mean of paired differences (in points) and its 95 % percentile interval."""
    n = len(diff)
    means = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return 100 * sum(diff) / n, 100 * means[int(0.025 * b)], 100 * means[int(0.975 * b) - 1]


def fmt(g: tuple[float, float, float]) -> str:
    return f"{g[0]:+.1f} [{g[1]:+.1f} ; {g[2]:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reference", default=None, help="tag of a run with one peer, mode solo")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grader", choices=("v2", "strict"), default="v2",
                    help="v2: re-grade the stored texts with essaim/answers.py; strict: as recorded in phase 0")
    a = ap.parse_args()
    global GRADER
    GRADER = a.grader

    man, rows = load(a.tag, a.split)
    ids = sorted({i for (_, i) in rows})
    modes = ["solo"] + [m for m in TOGETHER if any(k[0] == m for k in rows)]
    missing = [(m, i) for i in ids for m in modes if (m, i) not in rows]
    if missing or len(ids) != man["n"]:
        raise SystemExit(f"run incomplet : {len(ids)} questions sur {man['n']}, {len(missing)} lignes manquantes "
                         f"(ex. {missing[:3]})")
    peers = rows[("solo", ids[0])]["peers"]
    gold = {i: rows[("solo", i)]["gold"] for i in ids}
    for (m, i), r in rows.items():
        if r["gold"] != gold[i] or r["peers"] != peers:
            raise SystemExit(f"{m} {i} : réponse attendue ou pairs différents d'un mode à l'autre")

    correct: dict[str, list[float]] = {}
    errors = {p: 0 for p in peers}
    for pi, p in enumerate(peers):
        col = []
        for i in ids:
            pp = rows[("solo", i)]["per_peer"][pi]
            if pp["model"] != p:
                raise SystemExit(f"solo {i} : pairs dans le désordre")
            errors[p] += pp.get("error") is not None
            col.append(float(pp["answer"] == gold[i]))  # a failed peer counts as wrong (reported below)
        correct[p] = col
    for m in modes[1:]:
        correct[m] = [float(rows[(m, i)].get("answer") == gold[i]) for i in ids]
    ceiling = [float(any(rows[("solo", i)]["per_peer"][pi]["answer"] == gold[i] for pi in range(len(peers)))) for i in ids]

    ref = None
    if a.reference:
        rman, rrows = load(a.reference, a.split)
        same = ("data", "n", "prompt", "protocol", "template_date", "solo_tokens")
        diff = {k: (man.get(k), rman.get(k)) for k in same if man.get(k) != rman.get(k)}
        if diff:
            raise SystemExit(f"la référence n'a pas été mesurée sur les mêmes questions ou avec le même protocole : {diff}")
        rids = sorted(i for (m, i) in rrows if m == "solo")
        if rids != ids:
            raise SystemExit(f"la référence ne couvre pas les mêmes questions ({len(rids)} contre {len(ids)})")
        if any(rrows[("solo", i)]["gold"] != gold[i] for i in ids):
            raise SystemExit("la référence n'a pas les mêmes réponses attendues")
        ref_name = rrows[("solo", ids[0])]["peers"][0]
        ref = "référence : " + ref_name  # its own key, never an expert's (same model id, other quantisation...)
        correct[ref] = [float(rrows[("solo", i)]["per_peer"][0]["answer"] == gold[i]) for i in ids]
        errors[ref] = sum(rrows[("solo", i)]["per_peer"][0].get("error") is not None for i in ids)

    best = max(peers, key=lambda p: sum(correct[p]))
    rng = random.Random(a.seed)
    acc = {k: 100 * sum(v) / len(v) for k, v in correct.items()}
    gains = {m: boot([x - y for x, y in zip(correct[m], correct[best])], a.boot, rng) for m in modes[1:]}
    gains_ref = {m: boot([x - y for x, y in zip(correct[m], correct[ref])], a.boot, rng) for m in modes[1:]} if ref else {}

    def cost(m: str) -> dict:
        rs = [rows[(m, i)] for i in ids]
        trips = [1 if m in ("solo", "vote") else r["trips"] for r in rs]
        wall = [r["cost_from_solo_s"] if m == "vote" else r["wall_s"] for r in rs]
        calls = [c for r in rs for c in r.get("calls", [])]
        if m == "vote":  # its calls are those of the solo answers
            calls = [c for r in (rows[("solo", i)] for i in ids) for c in r.get("calls", [])]
        failed = sum(c["status"] != "ok" for c in calls)
        out = {"trips_mean": st.mean(trips), "trips_max": max(trips), "wall_s_mean": st.mean(wall),
               "wall_s_median": st.median(wall), "failed_calls": failed, "calls": len(calls),
               "net_ms_median": st.median([c["net_ms"] for c in calls if "net_ms" in c] or [0.0]),
               "wan_s_mean": {rtt: st.mean(w + t * rtt / 1000 for w, t in zip(wall, trips)) for rtt in RTTS}}
        if m in ("accord", "croise"):
            out["rounds_mean"] = st.mean(r["rounds"] for r in rs)
            out["words_mean"] = st.mean(len(r["text"].split()) for r in rs)
        return out

    costs = {m: cost(m) for m in modes}
    n = len(ids)
    L = [f"# Expérience 2 : écrire ensemble (GSM8K, jeu {a.split})", "",
         f"{n} questions, une réponse par question et par mode. Pairs : {', '.join(peers)}. "
         f"Blocs de {man['block']} jetons, au plus {man['max_rounds']} tours, k = {man['k'] or 'tous'}, "
         f"seuil du mode croisé tau = {man['tau']}, {man['solo_tokens']} jetons au plus pour une réponse seule. "
         "IC à 95 % : bootstrap apparié par question. Le meilleur pair seul est choisi sur ces mêmes données.", "",
         "## Exactitude", "", "| système | exactitude (%) |", "| --- | --- |"]
    for p in peers:
        L.append(f"| {p} seul{' (meilleur)' if p == best else ''} | {acc[p]:.1f} |")
    for m in modes[1:]:
        L.append(f"| **essaim : {LABEL[m]}** | **{acc[m]:.1f}** |")
    if ref:
        L.append(f"| {ref_name} seul (référence locale) | {acc[ref]:.1f} |")
    L += [f"| plafond (au moins un pair seul a raison) | {100 * sum(ceiling) / n:.1f} |", "",
          f"Gain sur le meilleur pair seul ({best}), en points :", ""]
    L += [f"- {m} : {fmt(gains[m])}" for m in modes[1:]]
    if ref:
        L += ["", f"Gain sur la référence ({ref_name}) :", ""] + [f"- {m} : {fmt(gains_ref[m])}" for m in modes[1:]]
    if any(errors.values()):
        L += ["", "Pairs en échec (comptés faux) : " + ", ".join(f"{p} {e}" for p, e in errors.items() if e)]
    L += ["", "## Coût", "",
          "Temps de décision mesuré (pairs sur une seule machine), puis temps projeté sur un réseau étendu : "
          "temps mesuré + allers-retours × RTT.", "",
          "| mode | allers-retours (moy. / max) | tours | temps mesuré (moy. / méd., s) | "
          + " | ".join(f"projeté RTT {r} ms (s)" for r in RTTS) + " | appels en échec |",
          "| --- | --- | --- | --- | " + " | ".join("---" for _ in RTTS) + " | --- |"]
    for m in modes:
        c = costs[m]
        L.append(f"| {m} | {c['trips_mean']:.1f} / {c['trips_max']} | {c.get('rounds_mean', 1):.1f} | "
                 f"{c['wall_s_mean']:.1f} / {c['wall_s_median']:.1f} | "
                 + " | ".join(f"{c['wan_s_mean'][r]:.1f}" for r in RTTS) + f" | {c['failed_calls']} / {c['calls']} |")
    L += ["", "Le vote ne coûte rien de plus que les réponses seules : il les réutilise (1 aller-retour au total)."]
    out = RESULTS / f"gen_report_{a.tag}_{a.split}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / f"gen_summary_{a.tag}_{a.split}.json").write_text(json.dumps(
        {"split": a.split, "n": n, "peers": peers, "best_peer": best, "reference": ref, "accuracy": acc,
         "ceiling": 100 * sum(ceiling) / n, "gain_vs_best": gains, "gain_vs_reference": gains_ref,
         "cost": costs, "peer_errors": errors}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
