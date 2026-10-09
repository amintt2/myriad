"""E3 (replay, no new inference): reliability-weighted vote and the exact stop certificate on GSM8K.

    uv run python analyze_e3.py --tag essaim4_colab --reference ref-Qwen3-4B_gpu

Weights are fixed on dev and applied unchanged to test (dev chooses, test reports):
  vote      plain majority, ties broken by the peers' mean log-probability (as in run_gen.py)
  logodds   each peer votes with w_i = log(p_i / (1 - p_i)) where p_i is its dev accuracy AMONG THE
            QUESTIONS IT ANSWERED, clipped to [0.02, 0.98] (Nitzan-Paroush weights; peers with p_i < 0.5 get
            a negative weight and are dropped, w_i = 0)
  logodds_K the K-class weights w_i = log(p_i / (1 - p_i)) - log(c), c the dev probability that two wrong
            answers that were both GIVEN are equal
Abstentions (a failed peer, or no extractable answer) cast no vote. Under the model, where an abstention
does not depend on the true answer, it multiplies the likelihood of every candidate answer by the same
factor: it carries no evidence, and the weight of a cast vote involves P(right | answered) and the
collisions between wrong answers actually given. Counting abstentions as wrong answers would lower p_i and,
since they never collide, inflate the denominator of c (audit 2026-10-09, phase0 5).
  best      the peer with the best dev accuracy, alone (what a single strong node would give)
Stop certificate (docs/03_idees_codex.md, idea 2): answers arrive in the order of the peers' measured
compute times; the decision is taken as soon as the leading answer's weight exceeds the runner-up's
plus all the weight still missing (ties broken against the leader, so the certificate is exact).
Reports accuracy with paired bootstrap 95 % intervals, and the number of peers waited for and the
decision time with and without the certificate. Writes results/e3_report_<tag>.md.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from essaim import data
from essaim.answers import regrade_gen
from essaim.results import read_manifest, read_rows

GRADER = "v2"

RESULTS = Path(__file__).resolve().parent / "results"


def solo(tag: str, split: str) -> tuple[list[str], dict[str, dict]]:
    path = RESULTS / f"gen_{tag}_{split}.jsonl"
    if read_manifest(path) is None:
        raise SystemExit(f"{path.name} : manifeste absent")
    man = read_manifest(path)
    if man.get("split") != split:
        raise SystemExit(f"{path.name} : manifeste de la partition {man.get('split')!r}, {split!r} attendue")
    raw = read_rows(path, key=("mode", "id"))
    if GRADER == "v2":  # re-grade from the stored texts (essaim/answers.regrade_gen)
        raw = regrade_gen(raw, read_manifest(path).get("max_rounds"), read_manifest(path).get("k"))
    rows = {r["id"]: r for r in raw if r["mode"] == "solo"}
    if len(rows) != man["n"]:  # never fit or report on an interrupted run
        raise SystemExit(f"{path.name} : {len(rows)} réponses solo sur {man['n']}, run incomplet")
    # The rows must be exactly the questions of this partition (essaim/data.py: dev and test are disjoint
    # by construction), so that a copy of one partition under the other's name is refused.
    want = {it["id"] for it in data.gsm8k(man["n"], split=split)}
    if set(rows) != want:
        raise SystemExit(f"{path.name} : les questions ne sont pas celles de la partition {split} de GSM8K")
    peers = next(iter(rows.values()))["peers"]
    return peers, rows


# Everything that must be identical between the run the weights are fitted on (dev) and the run they are
# applied to (test): the peers' model files and engines, the data and the generation protocol.
SAME_FIT_TEST = ("peers", "data", "prompt", "protocol", "template_date", "solo_tokens", "block", "k", "max_rounds", "tau")


def check_fit_test(tag: str, dev: dict, test: dict):
    """Dev and test of the same peers and protocol, on disjoint questions (audit 2026-10-09, phase0 6)."""
    mdev, mtest = read_manifest(RESULTS / f"gen_{tag}_dev.jsonl"), read_manifest(RESULTS / f"gen_{tag}_test.jsonl")
    diff = {k: "différent" for k in SAME_FIT_TEST if mdev.get(k) != mtest.get(k)}
    if diff:
        raise SystemExit(f"dev et test ne viennent pas des mêmes pairs ou du même protocole : {sorted(diff)}")
    common = set(dev) & set(test)
    if common:
        raise SystemExit(f"dev et test partagent {len(common)} questions (ex. {sorted(common)[:3]}) : refusé")


def accuracies(peers: list[str], rows: dict[str, dict]) -> list[float]:
    """Accuracy of each peer alone (an abstention counts as wrong): used to choose the best single peer."""
    return [sum(r["per_peer"][i]["answer"] == r["gold"] for r in rows.values()) / len(rows) for i in range(len(peers))]


def vote_accuracies(peers: list[str], rows: dict[str, dict]) -> list[float]:
    """P(right | the peer gave an answer): the p_i of the vote weights."""
    out = []
    for i in range(len(peers)):
        given = [r for r in rows.values() if r["per_peer"][i]["answer"] is not None]
        out.append(sum(r["per_peer"][i]["answer"] == r["gold"] for r in given) / len(given) if given else 0.0)
    return out


def collision(rows: dict[str, dict]) -> tuple[int, int, float]:
    """(equal pairs, pairs, c): pairs of wrong answers GIVEN by two peers on the same question, and how
    often they are equal; one half-collision as a floor so that c is never 0."""
    both = same = 0
    for r in rows.values():
        wrong = [p["answer"] for p in r["per_peer"] if p["answer"] is not None and p["answer"] != r["gold"]]
        for x in range(len(wrong)):
            for y in range(x + 1, len(wrong)):
                both += 1
                same += wrong[x] == wrong[y]
    return same, both, max(same, 0.5) / max(both, 1)


def decide(per_peer: list[dict], weights: list[float]) -> str | None:
    """Weighted vote over the peers that answered; ties broken by the best mean log-probability."""
    score: dict[str, float] = defaultdict(float)
    for p, w in zip(per_peer, weights):
        if p["answer"] is not None and w > 0:
            score[p["answer"]] += w
    if not score:
        return None
    top = max(score.values())
    tied = [a for a, s in score.items() if s == top]
    if len(tied) == 1:
        return tied[0]
    return max((p for p in per_peer if p["answer"] in tied), key=lambda p: p["conf"] if p["conf"] is not None else -1e9)["answer"]


def certified(per_peer: list[dict], weights: list[float], order: list[int]) -> tuple[str | None, int, float]:
    """Replay arrivals in `order`; return (decision, peers waited for, decision time in s)."""
    score: dict[str, float] = defaultdict(float)
    eps = 1e-9 * (1 + sum(abs(w) for w in weights))  # rounding margin: never certify a possible tie
    for k, i in enumerate(order, 1):
        p, w = per_peer[i], max(weights[i], 0.0)
        left = sum(max(weights[j], 0.0) for j in order[k:])  # recomputed, not accumulated by subtraction
        if p["answer"] is not None and w > 0:
            score[p["answer"]] += w
        ranked = sorted(score.values(), reverse=True) + [0.0, 0.0]
        # Exact: no completion of the missing answers can overtake or tie the leader.
        if score and ranked[0] > ranked[1] + left + eps:
            return max(score, key=score.get), k, per_peer[i]["compute_ms"] / 1000
    full = decide(per_peer, weights)
    return full, len(order), max(p["compute_ms"] for p in per_peer if p["compute_ms"] is not None) / 1000


def boot(diff: list[float], b: int, rng: random.Random) -> str:
    n = len(diff)
    means = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return f"{100 * sum(diff) / n:+.1f} [{100 * means[int(0.025 * b)]:+.1f} ; {100 * means[int(0.975 * b) - 1]:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--grader", choices=("v2", "strict"), default="v2",
                    help="v2: re-grade the stored texts with essaim/answers.py; strict: as recorded in phase 0")
    a = ap.parse_args()
    global GRADER
    GRADER = a.grader
    rng = random.Random(0)

    peers, dev = solo(a.tag, "dev")
    peers_t, test = solo(a.tag, "test")
    if peers != peers_t:
        raise SystemExit("pas les mêmes pairs sur dev et test")
    check_fit_test(a.tag, dev, test)
    if any(p["compute_ms"] is None for r in test.values() for p in r["per_peer"]):
        raise SystemExit("un pair a échoué sur test : le rejeu des temps n'est pas défini")
    p_dev = accuracies(peers, dev)
    p_vote = vote_accuracies(peers, dev)
    clip = [min(max(p, 0.02), 0.98) for p in p_vote]
    logodds = [max(0.0, math.log(p / (1 - p))) for p in clip]
    # K-class Nitzan-Paroush: w_i = log(p_i (K-1) / (1 - p_i)), with errors spread uniformly over K-1 wrong
    # answers. K is estimated on dev from c, the probability that two peers that both give a wrong answer
    # give the SAME one (c = 1/(K-1) under that model). Open answers (numbers) make c small: agreement
    # between weak peers is strong evidence, which the binary weights ignore.
    same, both, c = collision(dev)
    logodds_k = [max(0.0, math.log(p / (1 - p)) - math.log(c)) for p in clip]
    rules = {"vote": [1.0] * len(peers), "logodds": logodds, "logodds_K": logodds_k}
    best = max(range(len(peers)), key=lambda i: p_dev[i])

    ids = sorted(test)
    correct = {name: [float(decide(test[i]["per_peer"], w) == test[i]["gold"]) for i in ids] for name, w in rules.items()}
    correct["best"] = [float(test[i]["per_peer"][best]["answer"] == test[i]["gold"]) for i in ids]
    ref_name = None
    if a.reference:
        rman = read_manifest(RESULTS / f"gen_{a.reference}_test.jsonl")
        man = read_manifest(RESULTS / f"gen_{a.tag}_test.jsonl")
        _, ref = solo(a.reference, "test")
        fields = ("data", "n", "prompt", "protocol", "template_date", "solo_tokens")
        diff = {k: (man.get(k), rman.get(k)) for k in fields if man.get(k) != rman.get(k)}
        if diff or sorted(ref) != ids or any(ref[i]["gold"] != test[i]["gold"] for i in ids):
            raise SystemExit(f"la référence n'a pas été mesurée sur les mêmes questions ou avec le même protocole : {diff}")
        ref_name = next(iter(ref.values()))["peers"][0]
        correct["référence"] = [float(ref[i]["per_peer"][0]["answer"] == test[i]["gold"]) for i in ids]

    summary = {"peers": peers, "p_dev": p_dev, "p_vote": p_vote, "collision_pairs": [same, both], "collision": c, "weights": {"logodds": logodds, "logodds_K": logodds_k},
               "rules": {}, "certificate": {}}
    L = [f"# E3 : vote pondéré et certificat d'arrêt (GSM8K, rejeu, {a.tag})", "",
         "Poids appris sur **dev** (exactitude de chaque pair parmi les questions où il a donné une réponse : "
         "une abstention ne vote pas), appliqués tels quels sur **test** "
         f"({len(ids)} questions, disjointes de dev, mêmes pairs et même protocole : vérifié). "
         "IC 95 % : bootstrap apparié par question.", "",
         f"Deux pairs qui donnent chacun une réponse fausse donnent la même avec une probabilité c = {same}/{both} = "
         f"{c:.3f} sur dev, soit K - 1 = {1 / c:.0f} réponses fausses « équiprobables ».", "",
         "| pair | exactitude dev | exactitude dev parmi ses réponses | poids log-odds (2 classes) | "
         "poids log-odds (K classes) |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {p} | {100 * pd:.1f} | {100 * pv:.1f} | {w:.2f} | {wk:.2f} |"
          for p, pd, pv, w, wk in zip(peers, p_dev, p_vote, logodds, logodds_k)]
    L += ["", "| règle (test) | exactitude | gain sur le meilleur pair de dev | gain sur la référence |",
          "| --- | --- | --- | --- |"]
    for name in ("best", "vote", "logodds", "logodds_K"):
        label = {"best": f"meilleur pair de dev seul ({peers[best]})", "vote": "vote simple",
                 "logodds": "vote pondéré log-odds (2 classes)", "logodds_K": "vote pondéré log-odds (K classes)"}[name]
        g_best = "—" if name == "best" else boot([x - y for x, y in zip(correct[name], correct["best"])], a.boot, rng)
        g_ref = boot([x - y for x, y in zip(correct[name], correct["référence"])], a.boot, rng) if ref_name else "—"
        L.append(f"| {label} | {100 * st.mean(correct[name]):.1f} | {g_best} | {g_ref} |")
        summary["rules"][name] = {"accuracy": 100 * st.mean(correct[name]), "gain_vs_best": g_best, "gain_vs_ref": g_ref}
    if ref_name:
        L.append(f"| {ref_name} seul (référence) | {100 * st.mean(correct['référence']):.1f} | | |")

    L += ["", "## Certificat d'arrêt (rejeu des temps de calcul mesurés)", "",
          "Les réponses arrivent dans l'ordre des temps de calcul mesurés de chaque pair. On décide dès que la "
          "réponse en tête ne peut plus être rattrapée par les pairs manquants. La décision est identique "
          "à celle de l'attente de tous (vérifié ci-dessous).", "",
          "| règle | pairs attendus (moy.) | temps de décision moy. (s) | attente de tous moy. (s) | décisions identiques |",
          "| --- | --- | --- | --- | --- |"]
    for name, w in rules.items():
        waited, t_cert, t_all, same = [], [], [], 0
        for i in ids:
            pp = test[i]["per_peer"]
            order = sorted(range(len(peers)), key=lambda j: pp[j]["compute_ms"])
            d, k, t = certified(pp, w, order)
            waited.append(k)
            t_cert.append(t)
            t_all.append(max(p["compute_ms"] for p in pp) / 1000)
            same += d == decide(pp, w)
        summary["certificate"][name] = {"waited": st.mean(waited), "peers": len(peers), "t_cert_s": st.mean(t_cert),
                                        "t_all_s": st.mean(t_all), "same": same, "n": len(ids)}
        L.append(f"| {name} | {st.mean(waited):.2f} / {len(peers)} | {st.mean(t_cert):.2f} | {st.mean(t_all):.2f} | "
                 f"{same}/{len(ids)} |")
    if ref_name:
        summary["reference"] = {"model": ref_name, "accuracy": 100 * st.mean(correct["référence"])}
    (RESULTS / f"e3_summary_{a.tag}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    out = RESULTS / f"e3_report_{a.tag}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
