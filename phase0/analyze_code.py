"""E11 analysis: on code, does a swarm of small models of different families, selecting by EXECUTION, rival
much bigger models? (offline, from run_code.py and exec_code.py)

    uv run python analyze_code.py --suffix _colab

Candidates: for each problem, the greedy solution (sample 0) and the sampled ones (1..n) of each model.
Each distinct program was run on the visible tests (the prompt's examples; those the reference solution fails
are ignored), on extra inputs (output signatures) and on the hidden EvalPlus tests (grading only).
Selection rules, for a set of families (one small model per family, as in E4), over a pool of candidates
(greedy only, or greedy + samples):
  (a)  best single model: the greedy program of the family with the best DEV pass@1
  (b)  text vote: plurality of the normalised program text (essaim/code.normalized_text)
  (c)  visible + weight: among candidates passing every visible test, the one of the heaviest family
       (greedy first); if none passes, the one passing the most visible tests
  (d)  functional clustering (CodeT / AlphaCode style): candidates passing the visible tests (all loadable
       ones if none does) are grouped by identical outputs on the extra inputs; the cluster with the best
       score wins: number of programs (count), of distinct families (families), or sum of the weights of
       its distinct families (wfamilies); the variant is chosen on DEV
  (e)  cascade: the reference model's greedy program is taken when no candidate passes the visible tests,
       or when the winning cluster has fewer than m distinct families (m chosen on DEV, ties -> fewer calls)
Family weights (fitted on DEV): w_f = max(0, logit(p_f) - log(c)), p_f the family's greedy pass@1 and c the
functional collision rate on DEV (below), as for answers in E4 (docs/03_idees_codex.md).
Functional collision rate c: probability that two WRONG programs of different families, both passing the
visible tests, give the same outputs on every extra input; the analogue of the answer collision c of the
math vote theory. Its counterpart a: probability that two CORRECT programs agree on every extra input.
Agreement needs common VALID observations: a program's signature is "!" on an input where it raised, timed out
or returned nothing; a program with no valid output at all is never grouped with another (no functional
evidence, so a cascade calls the reference), and such programs are left out of the rates c and a.
Reported on TEST, per benchmark and pooled: accuracy (pass@1 of the selected program), Tango's score 95 %
intervals and exact TOST verdicts (margin DELTA points, essaim/stats.py) against the best single model and each reference
(greedy, and with the same selection over its own samples), scaling over all subsets of k families, and the
number of programs executed per problem.
Writes results/e11_report<suffix>.md and results/e11_summary<suffix>.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from analyze_e4 import DELTA, EXTRA, FAMILIES, PARAMS_B, REFS
from essaim import code, data
from essaim.stats import compare
from essaim.results import read_manifest, read_rows

RESULTS = Path(__file__).resolve().parent / "results"
PROTOCOL = ("prompt", "max_tokens", "thinking", "ctx_per_slot", "samples", "greedy", "sampling", "bench", "n")
IDENTITY = ("model", "revision", "gguf", "weights", "engine", "backend")
SCORES = ("count", "families", "wfamilies")
CASCADE_M = (1, 2, 3)


@dataclass(frozen=True)
class Cand:
    model: str
    fam: str
    sample: int
    prog: str
    text: str  # normalised text, for the text vote
    qid: str   # the problem: features are keyed by Cand, and one program can answer two problems


def has_obs(sig: tuple) -> bool:
    """At least one valid output on the extra inputs ("!" = exception, timeout or no output)."""
    return any(s != "!" for s in sig)


@dataclass(frozen=True)
class Feat:
    load: bool       # the program loads and defines the entry point
    vis: bool        # passes every (valid) visible test
    vis_n: int       # number of visible tests passed
    sig: tuple       # output signature on the extra inputs
    correct: bool    # passes the hidden tests


# ---------- selection rules (pure: candidates of one problem, their features, family weights) ----------

def rank(c: Cand, w: dict[str, float], order: dict[str, int]):
    """Preference between candidates: heavier family, greedy first, lower sample index, then family order
    (order[f] is larger for families listed earlier)."""
    return (w.get(c.fam, 0.0), c.sample == 0, -c.sample, order.get(c.fam, -10**6))


def pick(cs: list[Cand], w, order) -> Cand | None:
    return max(cs, key=lambda c: rank(c, w, order)) if cs else None


def text_vote(cs: list[Cand], w, order) -> Cand | None:
    groups: dict[str, list[Cand]] = defaultdict(list)
    for c in cs:
        if c.text:
            groups[c.text].append(c)
    if not groups:
        return pick(cs, w, order)
    best = max(groups.values(), key=lambda g: (len(g), rank(pick(g, w, order), w, order)))
    return pick(best, w, order)


def visible_weight(cs: list[Cand], F: dict[Cand, Feat], w, order) -> Cand | None:
    passing = [c for c in cs if F[c].vis]
    if passing:
        return pick(passing, w, order)
    return max(cs, key=lambda c: (F[c].vis_n, rank(c, w, order))) if cs else None


def cluster_score(g: list[Cand], score: str, w) -> float:
    fams = {c.fam for c in g}
    if score == "count":
        return len(g)
    if score == "families":
        return len(fams)
    return sum(w.get(f, 0.0) for f in fams)


def clusters(cs: list[Cand], F: dict[Cand, Feat]) -> tuple[list[list[Cand]], bool, list[Cand]]:
    """Groups of candidates with identical output signatures, among those passing the visible tests (or, if
    none does, among the loadable ones) that have at least one valid output; whether some candidate passed
    the visible tests; and that pool (programs run on the extra inputs)."""
    passing = [c for c in cs if F[c].vis]
    pool = passing or [c for c in cs if F[c].load]
    groups: dict[tuple, list[Cand]] = defaultdict(list)
    for c in pool:
        if has_obs(F[c].sig):  # failing everywhere is not an output two programs can agree on
            groups[F[c].sig].append(c)
    return list(groups.values()), bool(passing), pool


def functional(cs: list[Cand], F: dict[Cand, Feat], w, order, score: str = "wfamilies") -> tuple[Cand | None, dict]:
    groups, any_pass, pool = clusters(cs, F)
    if not pool:
        return pick(cs, w, order), {"any_pass": False, "top_fams": 0, "n_clusters": 0}
    if not groups:
        # No valid output on any input (no readable example and no usable annotation, or every program fails
        # on every input): no functional evidence at all. The heaviest candidate (as rule c), and no agreement
        # to report, so a cascade calls the reference.
        return pick(pool, w, order), {"any_pass": any_pass, "top_fams": 0, "n_clusters": 0}
    key = lambda g: (cluster_score(g, score, w), len({c.fam for c in g}), len(g), rank(pick(g, w, order), w, order))
    best = max(groups, key=key)
    return pick(best, w, order), {"any_pass": any_pass, "top_fams": len({c.fam for c in best}),
                                  "n_clusters": len(groups)}


def cascade(cs, F, w, order, score: str, ref: Cand | None, m: int) -> tuple[Cand | None, bool]:
    sel, info = functional(cs, F, w, order, score)
    call = ref is not None and (not info["any_pass"] or info["top_fams"] < m)
    return (ref if call else sel), call


def pair_rates(groups_by_q: list[list[tuple[Cand, Feat]]]) -> dict:
    """Over pairs of candidates of DIFFERENT families on the same problem:
      c      P(same outputs on every extra input | both wrong, both pass the visible tests)
      a      P(same outputs on every extra input | both correct)
      c_text P(same normalised text | both wrong)
      vis_fa P(both pass the visible tests | both wrong)  (false acceptance of the visible tests, in pairs)"""
    n = defaultdict(int)
    for cands in groups_by_q:
        for (x, fx), (y, fy) in itertools.combinations(cands, 2):
            if x.fam == y.fam:
                continue
            if not fx.correct and not fy.correct:
                n["wrong"] += 1
                n["text_same"] += bool(x.text) and x.text == y.text
                n["wrong_both_vis"] += fx.vis and fy.vis
                # no valid output (no extra input, or failures everywhere): no evidence, left out
                if fx.vis and fy.vis and has_obs(fx.sig) and has_obs(fy.sig):
                    n["wrong_vis"] += 1
                    n["wrong_vis_same"] += fx.sig == fy.sig
            if fx.correct and fy.correct and has_obs(fx.sig) and has_obs(fy.sig):
                n["right"] += 1
                n["right_same"] += fx.sig == fy.sig
    rate = lambda a, b: (n[a] / n[b]) if n[b] else None
    return {"c": rate("wrong_vis_same", "wrong_vis"), "a": rate("right_same", "right"),
            "c_text": rate("text_same", "wrong"), "vis_fa": rate("wrong_both_vis", "wrong"),
            "pairs_wrong_vis": n["wrong_vis"], "pairs_wrong": n["wrong"], "pairs_right": n["right"]}


def weights(p: dict[str, float], c: float | None) -> dict[str, float]:
    cc = min(max(c if c is not None else 0.5, 1e-3), 0.999)
    lo = lambda x: math.log(min(max(x, .02), .98) / (1 - min(max(x, .02), .98)))
    return {f: max(0.0, lo(p[f]) - math.log(cc)) for f in p}


# ---------- loading ----------

def load(bench: str, split: str, suffix: str, tag: str, models: dict[str, str]):
    """models: {model: family label}. Returns (ids, cands {qid: [Cand]}, feats {Cand: Feat}, info)."""
    gens, mans = {}, {}
    for m in models:
        path = code.gen_path(m, suffix, bench, split)
        man = read_manifest(path)
        if man is None:
            return None
        rows = read_rows(path, key=("id", "sample"))
        if len(rows) != man["n"] * (man["samples"] + 1):
            raise SystemExit(f"{path.name} : {len(rows)} solutions sur {man['n'] * (man['samples'] + 1)}, incomplet")
        gens[m], mans[m] = rows, man
    proto = {json.dumps({k: man.get(k) for k in PROTOCOL + ("data",)}, sort_keys=True) for man in mans.values()}
    if len(proto) != 1:
        raise SystemExit(f"{bench} {split} : protocole ou données différents d'un modèle à l'autre")
    man0 = next(iter(mans.values()))
    epath = code.exec_path(tag, suffix, bench, split)
    eman = read_manifest(epath)
    if eman is None:
        raise SystemExit(f"{epath.name} : absent, lancer exec_code.py")
    if eman["data"] != man0["data"] or eman["tests"] != code.TESTS_VERSION or eman["extract"] != code.EXTRACT_VERSION:
        raise SystemExit(f"{epath.name} : exécuté avec d'autres données ou une autre version des tests/de l'extraction")
    ex = {(r["id"], r["prog"]): r for r in read_rows(epath, key=("id", "prog"))}
    items = data.CODE_LOADERS[bench](man0["n"], split=split)
    if data.dataset_identity(bench) != man0["data"]:
        raise SystemExit(f"{bench} {split} : données locales différentes de celles des générations")
    ids, cands, feats, excluded, missing = [], {}, {}, [], 0
    for it in items:
        qid, entry = it["id"], code.entry_point(bench, it)
        ref = ex.get((qid, "reference"))
        if ref is None:
            raise SystemExit(f"{epath.name} : référence absente pour {qid}")
        if not ref["hidden"]["pass"]:  # the harness cannot grade this problem: left out, and reported
            excluded.append(qid)
            continue
        valid = [i for i, ok in enumerate(ref["visible"]) if ok]
        ids.append(qid)
        cands[qid] = []
        for m, fam in models.items():
            for g in gens[m]:
                if g["id"] != qid:
                    continue
                src, _ = code.extract_code(g["text"], entry)
                prog = code.prog_id(src)
                r = ex.get((qid, prog))
                if r is None:
                    missing += 1
                    continue
                c = Cand(m, fam, g["sample"], prog, code.normalized_text(src) if src.strip() else "", qid)
                load_ok = r["load"] is None
                vis_n = sum(1 for i in valid if r["visible"][i])
                feats[c] = Feat(load_ok, load_ok and vis_n == len(valid), vis_n, tuple(r["extra"]),
                                bool(r["hidden"]["pass"]))
                cands[qid].append(c)
    if missing:
        raise SystemExit(f"{epath.name} : {missing} programmes jamais exécutés (extraction changée ?), relancer exec_code.py")
    info = {"manifests": mans, "exec": eman, "excluded": excluded,
            "visible_counts": {it["id"]: sum(ex[(it["id"], "reference")]["visible"]) for it in items},
            "n_visible_rejected": sum(len(ex[(it["id"], "reference")]["visible"]) - sum(ex[(it["id"], "reference")]["visible"])
                                      for it in items)}
    return ids, cands, feats, info


EXEC_SETTINGS = ("tests", "extract", "extra_n", "isolation", "python", "numpy")


def check_splits(bench: str, fit: dict, test: dict):
    """Weights and rules fitted on one split are applied to the other: same model files and engine, same
    generation protocol, same execution settings, for every model present in both."""
    for m in set(fit["manifests"]) & set(test["manifests"]):
        a, b = fit["manifests"][m], test["manifests"][m]
        diff = {k: (a.get(k), b.get(k)) for k in IDENTITY + PROTOCOL if k != "n" and a.get(k) != b.get(k)}
        if diff:
            raise SystemExit(f"{m} {bench} : les deux partitions ne viennent pas du même modèle ou protocole : {diff}")
    diff = {k: (fit["exec"].get(k), test["exec"].get(k)) for k in EXEC_SETTINGS if fit["exec"].get(k) != test["exec"].get(k)}
    if diff:
        raise SystemExit(f"{bench} : exécutions des deux partitions faites avec d'autres réglages : {diff}")


# ---------- report ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_colab")
    ap.add_argument("--tag", default="e11")
    ap.add_argument("--benches", nargs="+", default=list(code.BENCHES), choices=list(code.BENCHES))
    ap.add_argument("--swarm", nargs="+", default=None, help="small models (default: the 7 families of E4)")
    ap.add_argument("--refs", nargs="*", default=None, help="reference models (default: those of E4)")
    ap.add_argument("--fit-split", default="dev", choices=list(data.SPLITS),
                    help="split used to fit weights and choose rules (test only for a smoke run)")
    a = ap.parse_args()
    swarm = dict(FAMILIES) if a.swarm is None else {m: m for m in a.swarm}  # label -> model
    refs = REFS if a.refs is None else a.refs
    extra = EXTRA if a.swarm is None else []
    fams = list(swarm)
    order = {f: -i for i, f in enumerate(fams)}  # earlier families win the last ties
    smoke = a.fit_split == "test"
    out_suffix = a.suffix + ("_fittest" if smoke else "")
    L = ["# E11 : un essaim de petits modèles qui choisit un programme en l'EXÉCUTANT, contre des modèles plus gros",
         ""]
    if smoke:
        L += ["**Essai de fumée : poids et règles choisis sur les mêmes problèmes que ceux rapportés (test). "
              "Aucune conclusion à en tirer.**", ""]
    L += [f"Candidats : la solution gloutonne de chaque modèle et ses solutions tirées (température "
          f"{code.SAMPLING['temperature']}). Règles et poids choisis sur **{a.fit_split}**, rapportés sur **test**. "
          f"Exactitude = pass@1 du programme choisi sur les tests cachés EvalPlus. IC 95 % : score de Tango pour la "
          f"différence appariée ; verdicts : tests exacts non conditionnels (essaim/stats.py), équivalence (TOST) "
          f"avec une marge de ±{DELTA} points, deux tests unilatéraux à 5 %.", ""]
    summary = {"benches": {}, "families": swarm, "refs": refs, "delta": DELTA, "fit_split": a.fit_split}
    pooled: dict[str, list[float]] = defaultdict(list)
    for bench in a.benches:
        models = {swarm[f]: f for f in fams}
        models.update({m: m for m in extra + list(refs)})
        got = {}
        for split in {a.fit_split, "test"}:
            avail = {m: f for m, f in models.items()
                     if read_manifest(code.gen_path(m, a.suffix, bench, split)) is not None}
            got[split] = load(bench, split, a.suffix, a.tag, avail) if all(swarm[f] in avail for f in fams) else None
        if got[a.fit_split] is None or got["test"] is None:
            L += [f"## {bench}", "", "incomplet (générations de l'essaim absentes)", ""]
            continue
        check_splits(bench, got[a.fit_split][3], got["test"][3])
        # references must exist in both splits (the cascade threshold is chosen on the fit split)
        present = [m for m in models if all(m in got[s][3]["manifests"] for s in (a.fit_split, "test"))]
        R = [m for m in refs if m in present]

        def solo(split, model, sample=0):
            ids, cands, F, _ = got[split]
            return {q: next((c for c in cands[q] if c.model == model and c.sample == sample), None) for q in ids}

        def acc_of(split, sel: dict) -> float:
            ids, _, F, _ = got[split]
            return 100 * st.mean(float(sel[q] is not None and F[sel[q]].correct) for q in ids)

        # --- fitted on the fit split ---
        fids, fc, fF, _ = got[a.fit_split]
        p_fit = {f: acc_of(a.fit_split, solo(a.fit_split, swarm[f])) / 100 for f in fams}
        rates_fit = pair_rates([[(c, fF[c]) for c in fc[q] if c.sample == 0 and c.model in swarm.values()] for q in fids])
        w = weights(p_fit, rates_fit["c"])
        best_f = max(fams, key=lambda f: (p_fit[f], order[f]))

        def pool(split, q, subset, greedy_only):
            return [c for c in got[split][1][q] if c.fam in subset and c.model == swarm[c.fam] and
                    (c.sample == 0 or not greedy_only)]

        def rule(split, name, subset=None, greedy_only=False, score="wfamilies", m=1, ref=None):
            subset = set(subset or fams)
            ids, cands, F, _ = got[split]
            sel, calls = {}, 0
            ref_sel = solo(split, ref) if ref else None
            for q in ids:
                cs = pool(split, q, subset, greedy_only)
                if name == "best":
                    bf = max(subset, key=lambda f: (p_fit[f], order[f]))
                    sel[q] = next((c for c in cs if c.fam == bf and c.sample == 0), None)
                elif name == "text":
                    sel[q] = text_vote(cs, w, order)
                elif name == "visible":
                    sel[q] = visible_weight(cs, F, w, order)
                elif name == "cluster":
                    sel[q] = functional(cs, F, w, order, score)[0]
                elif name == "cascade":
                    sel[q], called = cascade(cs, F, w, order, score, ref_sel[q], m)
                    calls += called
            return sel, calls

        score_best = max(SCORES, key=lambda s: (acc_of(a.fit_split, rule(a.fit_split, "cluster", score=s)[0]),
                                                -SCORES.index(s)))
        primary = max([("visible", None), ("cluster", score_best)],
                      key=lambda r: acc_of(a.fit_split, rule(a.fit_split, r[0], score=r[1] or "wfamilies")[0]))
        casc_ref = max(R, key=lambda m: PARAMS_B.get(m, 0)) if R else None
        m_best = None
        if casc_ref:
            m_best = max(CASCADE_M, key=lambda m: (acc_of(a.fit_split, rule(a.fit_split, "cascade", score=score_best,
                                                                             m=m, ref=casc_ref)[0]), -m))

        # --- reported on test ---
        ids, cands, F, info = got["test"]
        corr = lambda sel: [float(sel[q] is not None and F[sel[q]].correct) for q in ids]
        rows, systems = [], {}

        def add(label, sel, calls=None, key=None, pool_note=""):
            v = corr(sel)
            systems[key or label] = v
            rows.append((label, 100 * st.mean(v), calls, pool_note))

        for f in fams:
            add(f"{swarm[f]} seul ({f}), glouton", solo("test", swarm[f]), key=f"alone|{swarm[f]}")
        for m in extra + R:
            if m in present:
                add(f"{m} seul" + (f" (référence, {PARAMS_B.get(m, '?')} G)" if m in R else " (hors essaim)") + ", glouton",
                    solo("test", m), key=f"alone|{m}")
        add(f"(a) meilleur pair de {a.fit_split} ({swarm[best_f]}), glouton", rule("test", "best")[0], key="best")
        own = lambda model: {q: functional([c for c in cands[q] if c.model == model], F, w, order, score_best)[0] for q in ids}
        add(f"(a') meilleur pair, sélection sur ses {1 + info['manifests'][swarm[best_f]]['samples']} solutions",
            own(swarm[best_f]), key="best_self")
        for greedy_only, note in ((True, "gloutons"), (False, "gloutons + tirages")):
            g = "g" if greedy_only else "all"
            add(f"(b) vote sur le texte, {note}", rule("test", "text", greedy_only=greedy_only)[0], key=f"text|{g}")
            add(f"(c) tests visibles + poids, {note}", rule("test", "visible", greedy_only=greedy_only)[0], key=f"visible|{g}")
            for s in SCORES:
                add(f"(d) regroupement fonctionnel ({s}), {note}",
                    rule("test", "cluster", greedy_only=greedy_only, score=s)[0], key=f"cluster-{s}|{g}")
        primary_key = "visible|all" if primary[0] == "visible" else f"cluster-{score_best}|all"
        for m in R:
            add(f"{m} + sélection sur ses propres solutions", own(m), key=f"refself|{m}")
        casc_calls = None
        if casc_ref:
            for mm in CASCADE_M:
                sel, calls = rule("test", "cascade", score=score_best, m=mm, ref=casc_ref)
                add(f"(e) cascade vers {casc_ref}, m = {mm}" + (" (choisi)" if mm == m_best else ""), sel,
                    calls=calls, key=f"cascade|{mm}")
                if mm == m_best:
                    casc_calls = calls
        oracle = [float(any(F[c].correct for c in pool("test", q, fams, False))) for q in ids]
        rows.append(("oracle (au moins un candidat de l'essaim juste)", 100 * st.mean(oracle), None, ""))

        rates_test = {"greedy": pair_rates([[(c, F[c]) for c in cands[q] if c.sample == 0 and c.model in swarm.values()]
                                           for q in ids]),
                      "all": pair_rates([[(c, F[c]) for c in cands[q] if c.model in swarm.values()] for q in ids])}
        L += [f"## {bench} ({len(ids)} problèmes test" +
              (f", {len(info['excluded'])} exclus car la référence échoue : {info['excluded']}" if info["excluded"] else "") + ")",
              "", f"Poids ({a.fit_split}) : " + ", ".join(f"{f} {w[f]:.2f}" for f in fams) +
              f". Variante de regroupement choisie : **{score_best}** ; règle principale : **{primary_key}**" +
              (f" ; cascade : m = {m_best}." if casc_ref else "."), "",
              "| système | pass@1 test (%) | appels au gros modèle |", "| --- | --- | --- |"]
        L += [f"| {lab} | {acc:.1f} | {'' if calls is None else calls} |" for lab, acc, calls, _ in rows]

        comps = {}
        L += ["", f"Comparaisons appariées (test) : différence en points [IC 95 %], p exacts des deux tests de marge, "
              "verdict.", "",
              "| système | contre | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |",
              "| --- | --- | --- | --- | --- |"]
        others = [("meilleur pair (a)", "best")] + [(m, f"alone|{m}") for m in R] + [(f"{m} + sélection", f"refself|{m}") for m in R]
        mains = [("règle principale", primary_key)] + ([("cascade (m choisi)", f"cascade|{m_best}")] if casc_ref else [])
        for mname, mk in mains:
            for oname, ok in others:
                g = compare(systems[mk], systems[ok], DELTA)
                comps[f"{mk}|{ok}"] = g
                L.append(f"| {mname} | {oname} | {g['mean']:+.1f} [{g['lo95']:+.1f} ; {g['hi95']:+.1f}] | "
                         f"{g['p_low_margin']:.3f} ; {g['p_high_margin']:.3f} | {g['verdict']} |")
                pooled[f"{mname} moins {oname}"].append((systems[mk], systems[ok]))

        scale = {}
        for k in range(1, len(fams) + 1):
            subs = list(itertools.combinations(fams, k))
            res = defaultdict(list)
            for s in subs:
                for name, kw in (("text", {}), ("visible", {}), ("cluster", {"score": score_best})):
                    res[name].append(acc_of("test", rule("test", name, subset=s, **kw)[0]))
                res["oracle"].append(100 * st.mean(float(any(F[c].correct for c in pool("test", q, s, False))) for q in ids))
            scale[k] = {n: st.mean(v) for n, v in res.items()} | {"n_subsets": len(subs)}
        L += ["", "Passage à l'échelle (gloutons + tirages) : moyenne sur tous les sous-ensembles de k familles.", "",
              "| k | (b) texte | (c) visibles | (d) regroupement | oracle |", "| --- | --- | --- | --- | --- |"]
        L += [f"| {k} | {v['text']:.1f} | {v['visible']:.1f} | {v['cluster']:.1f} | {v['oracle']:.1f} |" for k, v in scale.items()]

        n_vis = st.mean(len({c.prog for c in pool("test", q, fams, False)}) for q in ids)
        n_ext = st.mean(len({c.prog for c in clusters(pool("test", q, fams, False), F)[2]}) for q in ids)
        per_model = {}
        for m in present:
            mine = [c for q in ids for c in cands[q] if c.model == m]
            vis = [c for c in mine if F[c].vis]
            per_model[m] = {"greedy": 100 * st.mean(float(F[c].correct) for c in mine if c.sample == 0),
                            "all_samples": 100 * st.mean(float(F[c].correct) for c in mine),
                            "visible_pass": 100 * len(vis) / len(mine),
                            "precision_visible": (100 * st.mean(float(F[c].correct) for c in vis)) if vis else None}
        L += ["", f"Programmes exécutés par problème (essaim de {len(fams)} familles, gloutons + tirages, programmes "
              f"distincts) : {n_vis:.1f} sur les tests visibles, {n_ext:.1f} sur les entrées supplémentaires "
              f"({info['exec']['extra_n']} entrées dérivées + les entrées visibles).", "",
              "Collisions fonctionnelles (paires de familles différentes) :", "",
              "| pool | c (2 faux d'accord sur tout) | a (2 justes d'accord) | c texte | faux acceptés par les tests visibles (paires) |",
              "| --- | --- | --- | --- | --- |"]
        fmt = lambda x: "—" if x is None else f"{x:.3f}"
        for nm, r in rates_test.items():
            L.append(f"| {nm} | {fmt(r['c'])} ({r['pairs_wrong_vis']} paires) | {fmt(r['a'])} ({r['pairs_right']}) | "
                     f"{fmt(r['c_text'])} | {fmt(r['vis_fa'])} |")
        L += ["", "| modèle | glouton | moyenne des tirages et du glouton | passe les tests visibles | juste sachant visibles OK |",
              "| --- | --- | --- | --- | --- |"]
        pct = lambda x: "—" if x is None else f"{x:.1f}"
        L += [f"| {m} | {pct(v['greedy'])} | {pct(v['all_samples'])} | {pct(v['visible_pass'])} | "
              f"{pct(v['precision_visible'])} |" for m, v in per_model.items()]
        L += ["", f"Tests visibles : {sum(1 for q in ids if info['visible_counts'][q] == 0)} problèmes sans aucun test "
              f"visible valide ; {info['n_visible_rejected']} tests visibles ignorés (la référence y échoue).", ""]
        summary["benches"][bench] = {
            "n": len(ids), "excluded": info["excluded"], "weights": w, "p_fit": p_fit, "best_fit": swarm[best_f],
            "cluster_score": score_best, "primary": primary_key, "cascade": {"ref": casc_ref, "m": m_best, "calls": casc_calls},
            "systems": {k: 100 * st.mean(v) for k, v in systems.items()}, "oracle": 100 * st.mean(oracle),
            "comparisons": comps, "scaling": scale, "collisions_fit": rates_fit, "collisions_test": rates_test,
            "programs_per_problem": {"visible": n_vis, "extra": n_ext}, "per_model": per_model,
            "isolation": info["exec"]["isolation"], "extra_n": info["exec"]["extra_n"]}
    if len(summary["benches"]) > 1:
        L += ["## Les deux benchmarks réunis (problèmes test mis bout à bout)", "",
              "| comparaison | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |", "| --- | --- | --- | --- |"]
        summary["pooled"] = {}
        for k, parts in pooled.items():
            if len(parts) != len(summary["benches"]):  # only comparisons available on every benchmark
                continue
            g = compare([x for p in parts for x in p[0]], [y for p in parts for y in p[1]], DELTA)
            summary["pooled"][k] = g
            L.append(f"| {k} | {g['mean']:+.1f} [{g['lo95']:+.1f} ; {g['hi95']:+.1f}] | {g['p_low_margin']:.3f} ; "
                     f"{g['p_high_margin']:.3f} | {g['verdict']} |")
    e4 = RESULTS / f"e4_summary{a.suffix}.json"
    if e4.exists():
        cs = {b: v.get("collision") for b, v in json.loads(e4.read_text(encoding="utf-8"))["benches"].items()}
        L += ["", "Pour comparaison, collision des réponses fausses en mathématiques et QCM (E4, dev) : " +
              ", ".join(f"{b} {c:.3f}" for b, c in cs.items() if c is not None) + "."]
    (RESULTS / f"e11_report{out_suffix}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / f"e11_summary{out_suffix}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str),
                                                           encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
