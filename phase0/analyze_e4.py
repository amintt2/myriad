"""E4 analysis: a swarm of small heterogeneous models against much bigger ones (offline, from run_solo.py).

    uv run python analyze_e4.py --suffix _colab

Every model answered every question alone, once (greedy). A swarm decision is a vote over the peers'
normalised answers, computed here for any set of peers:
  vote      plurality, ties broken by the higher mean log-probability
  wvote     K-class Nitzan-Paroush weights w_i = log(p_i / (1 - p_i)) - log(c), where p_i is the peer's
            DEV accuracy among the questions it answered (an abstention, no extractable answer, casts no vote
            and is uninformative under the model) and c the DEV probability that two wrong ANSWERS (both
            given) are the same (docs/03_idees_codex.md, analyze_e3.py); fitted on dev, applied to test
Reported on TEST, per benchmark and macro-averaged:
  * accuracy of each model alone, of the swarm of the 7 families (one small model per family), of the
    best dev peer alone, and the oracle (some peer is right);
  * swarm minus each reference: Tango's score 95 % interval for the paired difference, and verdicts from
    exact unconditional tests (essaim/stats.py): equivalence = two one-sided tests at 5 % with margin
    DELTA points (TOST), superiority/inferiority = one-sided tests at 2.5 %;
  * scaling: mean accuracy over ALL subsets of k families, k = 1..7, and the best-k-by-dev subset;
  * the exact stop certificate: peers waited for (arrival order = the peers' measured times).
Writes results/e4_report<suffix>.md and results/e4_summary<suffix>.json (used by the paper figures).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from essaim import answers, data, models
from essaim.stats import compare
from essaim.results import read_manifest, read_rows

RESULTS = Path(__file__).resolve().parent / "results"
BENCHES = ("gsm8k", "math500", "arc", "mmlupro")
LOADERS = {"gsm8k": data.gsm8k, "math500": data.math500, "arc": data.arc, "mmlupro": data.mmlu_pro}
FAMILIES = {  # one small model (<= 4B) per family: the swarm
    "Qwen": "Qwen/Qwen3.5-4B", "Google": "google/gemma-4-E4B-it", "IBM": "ibm-granite/granite-4.2-3b",
    "Hugging Face": "HuggingFaceTB/SmolLM3-3B", "Mistral": "mistralai/Ministral-3-3B-Instruct-2512",
    "Microsoft": "microsoft/Phi-4-mini-instruct", "AllenAI": "allenai/OLMo-2-0425-1B-Instruct"}
EXTRA = ["Qwen/Qwen3.5-2B", "google/gemma-4-E2B-it"]  # smaller siblings (not in the 7-family swarm)
REFS = ["Qwen/Qwen3.5-9B", "google/gemma-4-12B-it", "mistralai/Ministral-3-14B-Instruct-2512", "Qwen/Qwen3.8-27B"]
PARAMS_B = {m: models.total_b(m) for m in list(FAMILIES.values()) + EXTRA + REFS}  # essaim/models.py: total
DELTA = 2.0  # equivalence margin, points
PARTIAL = False  # --partial: incomplete reference runs are skipped (preview only, never for the paper)


def load(model: str, suffix: str, bench: str, split: str) -> dict[str, dict] | None:
    path = RESULTS / f"solo_{model.replace('/', '__')}{suffix}_{bench}_{split}.jsonl"
    man = read_manifest(path)
    if man is None:
        return None
    rows = {r["id"]: r for r in read_rows(path, key=("id",))}
    if len(rows) != man["n"]:
        if PARTIAL and model in REFS:  # preview while references are still running: treat as missing
            return None
        raise SystemExit(f"{path.name} : {len(rows)} réponses sur {man['n']}, run incomplet")
    # Answers are re-extracted from the raw text with the current grader (essaim/answers.py), so a grader
    # fix never needs new inference; the stored "answer" is only what the grader said at run time.
    items = {it["id"]: it for it in LOADERS[bench](man["n"], split=split)}
    for i, r in rows.items():
        r["answer"] = answers.extract(bench, r["text"], items[i], ended=r["finish"] == "stop")
        r["gold"] = answers.gold(bench, items[i])
    MANIFESTS[(model, bench, split)] = man
    return rows


MANIFESTS: dict[tuple[str, str, str], dict] = {}
PROTOCOL = ("data", "prompt", "max_tokens", "temperature", "thinking", "ctx_per_slot", "bench", "logprobs")
IDENTITY = ("model", "revision", "gguf", "weights", "engine", "backend")


def check_manifests(bench: str, models: list[str]):
    """Same protocol and questions for every model and split of a benchmark; the same model files and
    engine for a model's dev and test runs (weights are fitted on one and applied to the other)."""
    ref = None
    for m in models:
        for split in ("dev", "test"):
            man = MANIFESTS[(m, bench, split)]
            proto = {k: man.get(k) for k in PROTOCOL if k != "data"}
            if ref is None:
                ref = proto
            elif proto != ref:
                raise SystemExit(f"{m} {bench} {split} : protocole différent {proto} != {ref}")
        dv, te = MANIFESTS[(m, bench, "dev")], MANIFESTS[(m, bench, "test")]
        diff = {k: (dv.get(k), te.get(k)) for k in IDENTITY if dv.get(k) != te.get(k)}
        if diff:
            raise SystemExit(f"{m} {bench} : dev et test ne viennent pas du même modèle ou moteur : {diff}")
    for split in ("dev", "test"):
        datas = {json.dumps(MANIFESTS[(m, bench, split)].get("data"), sort_keys=True) for m in models}
        if len(datas) != 1:
            raise SystemExit(f"{bench} {split} : jeux de données différents d'un modèle à l'autre")


def collision(rows_by_model: dict[str, dict], ids: list[str]) -> float:
    """P(two peers that both GIVE a wrong answer give the same one), on the given questions. Abstentions
    (no extractable answer) cast no vote, so they are not wrong answers here (see valid_accuracy)."""
    both = same = 0
    for i in ids:
        wrong = [r[i]["answer"] for r in rows_by_model.values()
                 if r[i]["answer"] is not None and r[i]["answer"] != r[i]["gold"]]
        for x, y in itertools.combinations(wrong, 2):
            both += 1
            same += x == y
    return max(same, 0.5) / max(both, 1)


def valid_accuracy(rows: dict, ids: list[str]) -> float:
    """P(right | the peer gave an answer): the p_i of the K-class weights. Under the model (abstention
    independent of the true answer), an abstention multiplies the likelihood of every candidate answer by
    the same factor, so the weight of a cast vote is log(P(right | answered) / P(wrong | answered)) - log c."""
    given = [i for i in ids if rows[i]["answer"] is not None]
    return sum(rows[i]["answer"] == rows[i]["gold"] for i in given) / len(given) if given else 0.0


def decide(answers: list[tuple[str | None, float, float | None]]) -> str | None:
    """answers: (answer, weight, mean_logp); weighted plurality, ties broken by mean log-probability."""
    score: dict[str, float] = defaultdict(float)
    for a, w, _ in answers:
        if a is not None and w > 0:
            score[a] += w
    if not score:
        return None
    top = max(score.values())
    tied = {a for a, s in score.items() if s >= top - 1e-12}
    if len(tied) == 1:
        return next(iter(tied))
    # Ties: the answer of the heaviest supporting peer, then its mean log-probability when recorded,
    # then the first peer in family order (deterministic).
    return max((x for x in answers if x[0] in tied), key=lambda x: (x[1], x[2] if x[2] is not None else -1e9))[0]


def certificate_waits(answers: list[tuple[str | None, float, float | None]], times: list[float]) -> int:
    order = sorted(range(len(answers)), key=lambda j: times[j])
    score: dict[str, float] = defaultdict(float)
    eps = 1e-9 * (1 + sum(abs(w) for _, w, _ in answers))
    for k, j in enumerate(order, 1):
        a, w, _ = answers[j]
        if a is not None and w > 0:
            score[a] += w
        left = sum(max(answers[o][1], 0.0) for o in order[k:])
        ranked = sorted(score.values(), reverse=True) + [0.0, 0.0]
        if score and ranked[0] > ranked[1] + left + eps:
            return k
    return len(answers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_colab")
    ap.add_argument("--partial", action="store_true", help="preview: skip incomplete reference runs")
    ap.add_argument("--strict-math", action="store_true",
                    help="MATH-500: \boxed{} only (no fallback to the final expression); writes *_strictmath files")
    a = ap.parse_args()
    if a.strict_math:
        answers.MATH_FALLBACK = False
        a.suffix_out = "_strictmath"
    else:
        a.suffix_out = ""
    if a.partial:  # previews never share the names the paper reads
        a.suffix_out += "_preview"
    global PARTIAL
    PARTIAL = a.partial
    fams = list(FAMILIES)
    summary: dict = {"benches": {}, "families": FAMILIES, "refs": REFS, "delta": DELTA,
                     "math_grader": "boxed only" if a.strict_math else "boxed, else final expression (complete answers)",
                     "preview": a.partial}
    L = ["# E4 : un essaim de petits modèles de familles différentes contre des modèles bien plus gros", "",
         "Chaque modèle répond seul, une fois (glouton), via llama-server comme dans l'app. Les décisions de "
         "l'essaim sont calculées hors ligne (un aller-retour par requête). Poids appris sur **dev**, résultats "
         "sur **test**. IC 95 % : intervalle du score de Tango pour la différence appariée. Verdicts : tests exacts "
         f"non conditionnels (essaim/stats.py) ; équivalence (TOST) : marge ±{DELTA} points, deux tests "
         "unilatéraux à 5 % ; supériorité ou infériorité : test unilatéral à 2,5 %. Poids : exactitude parmi "
         "les réponses données (une abstention ne vote pas), collision entre réponses fausses données.", ""]
    for bench in BENCHES:
        models = list(FAMILIES.values()) + EXTRA + REFS
        dev = {m: load(m, a.suffix, bench, "dev") for m in models}
        test = {m: load(m, a.suffix, bench, "test") for m in models}
        missing = [m for m in models if dev[m] is None or test[m] is None]
        if any(FAMILIES[f] in missing for f in fams):
            L += [f"## {bench}", "", f"incomplet : {missing}", ""]
            continue
        check_manifests(bench, [m for m in models if m not in missing])
        ids_dev, ids = sorted(dev[FAMILIES[fams[0]]]), sorted(test[FAMILIES[fams[0]]])
        for m in models:
            if m in missing:
                continue
            if sorted(dev[m]) != ids_dev or sorted(test[m]) != ids:
                raise SystemExit(f"{m} {bench} : pas les mêmes questions")
        acc = lambda rows, qs: sum(rows[i]["answer"] == rows[i]["gold"] for i in qs) / len(qs)
        p_dev = {f: acc(dev[FAMILIES[f]], ids_dev) for f in fams}  # accuracy (abstentions count as wrong)
        p_vote = {f: valid_accuracy(dev[FAMILIES[f]], ids_dev) for f in fams}  # among the answers given
        c = collision({f: dev[FAMILIES[f]] for f in fams}, ids_dev)
        w = {f: max(0.0, math.log(min(max(p_vote[f], .02), .98) / (1 - min(max(p_vote[f], .02), .98))) - math.log(c))
             for f in fams}

        def swarm(subset: list[str], rule: str, i: str) -> str | None:
            return decide([(test[FAMILIES[f]][i]["answer"], 1.0 if rule == "vote" else w[f], test[FAMILIES[f]][i]["mean_logp"])
                           for f in subset])

        correct = {}
        for rule in ("vote", "wvote"):
            correct[rule] = [float(swarm(fams, rule, i) == test[FAMILIES[fams[0]]][i]["gold"]) for i in ids]
        best_f = max(fams, key=lambda f: p_dev[f])
        correct["best"] = [float(test[FAMILIES[best_f]][i]["answer"] == test[FAMILIES[best_f]][i]["gold"]) for i in ids]
        oracle = st.mean(float(any(test[FAMILIES[f]][i]["answer"] == test[FAMILIES[f]][i]["gold"] for f in fams)) for i in ids)
        alone = {m: acc(test[m], ids) for m in models if m not in missing}
        L += [f"## {bench} ({len(ids)} questions test)", "",
              f"Collision des réponses fausses sur dev : c = {c:.3f}.", "",
              "| système | exactitude test (%) | poids (wvote) |", "| --- | --- | --- |"]
        for f in fams:
            L.append(f"| {FAMILIES[f]} seul ({f}) | {100 * alone[FAMILIES[f]]:.1f} | {w[f]:.2f} |")
        for m in EXTRA:
            if m in alone:
                L.append(f"| {m} seul (hors essaim) | {100 * alone[m]:.1f} | |")
        L += [f"| **essaim 7 familles, vote** | **{100 * st.mean(correct['vote']):.1f}** | |",
              f"| **essaim 7 familles, vote pondéré** | **{100 * st.mean(correct['wvote']):.1f}** | |"]
        for m in REFS:
            if m in alone:
                L.append(f"| {m} seul (référence, {PARAMS_B[m]} G) | {100 * alone[m]:.1f} | |")
        L += [f"| oracle (au moins un pair juste) | {100 * oracle:.1f} | |", ""]

        comps = {}
        L += ["| essaim (wvote) moins | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |",
              "| --- | --- | --- | --- |"]
        for name, other in [(f"meilleur pair de dev ({FAMILIES[best_f]})", correct["best"])] + \
                [(m, [float(test[m][i]["answer"] == test[m][i]["gold"]) for i in ids]) for m in REFS if m in alone]:
            for rule in ("wvote", "vote"):
                g = compare(correct[rule], other, DELTA)
                comps[f"{rule}|{name}"] = g
                if rule == "wvote":
                    L.append(f"| {name} | {g['mean']:+.1f} [{g['lo95']:+.1f} ; {g['hi95']:+.1f}] | "
                             f"{g['p_low_margin']:.3f} ; {g['p_high_margin']:.3f} | {g['verdict']} |")

        scale = {}
        for k in range(1, len(fams) + 1):
            subs = list(itertools.combinations(fams, k))
            mean_k = st.mean(st.mean(float(swarm(list(s), "wvote", i) == test[FAMILIES[fams[0]]][i]["gold"]) for i in ids)
                             for s in subs)
            best_k = sorted(fams, key=lambda f: -p_dev[f])[:k]
            bk = st.mean(float(swarm(best_k, "wvote", i) == test[FAMILIES[fams[0]]][i]["gold"]) for i in ids)
            scale[k] = {"mean_all_subsets": 100 * mean_k, "best_k_by_dev": 100 * bk, "n_subsets": len(subs)}
        L += ["", "Passage à l'échelle (vote pondéré) : moyenne sur tous les sous-ensembles de k familles, et "
              "les k meilleures familles de dev.", "", "| k | moyenne | k meilleures |", "| --- | --- | --- |"]
        L += [f"| {k} | {v['mean_all_subsets']:.1f} | {v['best_k_by_dev']:.1f} |" for k, v in scale.items()]

        times = {f: [test[FAMILIES[f]][i]["ms"] for i in ids] for f in fams}
        waits = [certificate_waits([(test[FAMILIES[f]][i]["answer"], w[f], test[FAMILIES[f]][i]["mean_logp"]) for f in fams],
                                   [times[f][n] for f in fams]) for n, i in enumerate(ids)]
        toks = {m: st.mean((test[m][i]["n_tokens"] or 0) for i in ids) for m in models if m not in missing}
        L += ["", f"Certificat d'arrêt (ordre d'arrivée = temps mesurés) : {st.mean(waits):.2f} pairs attendus sur "
              f"{len(fams)} en moyenne.", ""]
        summary["benches"][bench] = {"n": len(ids), "collision": c, "weights": w, "p_dev": p_dev, "p_vote": p_vote,
                                     "alone": alone,
                                     "swarm": {r: 100 * st.mean(correct[r]) for r in ("vote", "wvote")},
                                     "best_dev": FAMILIES[best_f], "oracle": 100 * oracle, "comparisons": comps,
                                     "scaling": scale, "certificate_waits": st.mean(waits), "mean_tokens": toks}
    done = [b for b in BENCHES if b in summary["benches"]]
    if done:
        L += ["## Moyenne des benchmarks complets (" + ", ".join(done) + ")", "", "| système | exactitude moyenne (%) |",
              "| --- | --- |"]
        for r in ("vote", "wvote"):
            L.append(f"| essaim 7 familles, {r} | {st.mean(summary['benches'][b]['swarm'][r] for b in done):.1f} |")
        for m in list(FAMILIES.values()) + REFS:
            if all(m in summary["benches"][b]["alone"] for b in done):
                L.append(f"| {m} | {100 * st.mean(summary['benches'][b]['alone'][m] for b in done):.1f} |")
    (RESULTS / f"e4_report{a.suffix}{a.suffix_out}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / f"e4_summary{a.suffix}{a.suffix_out}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
