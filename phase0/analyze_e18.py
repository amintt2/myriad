"""E18 analysis: Lean-checked proofs from a swarm of small provers, against the best of them and a bigger prover.

    uv run python analyze_e18.py --suffix _colab

Reads results/e18_<model><suffix>_minif2f_<split>.jsonl (run_e18.py) and the oracle files (e18_setup.py). Offline,
no Lean: every verdict is recomputed from the Lean messages stored with the attempt (essaim/lean.verdict), and the
checked file is rebuilt from the raw completion with the current screening (essaim/lean.build); any difference with
what the VM recorded stops the analysis. The choice of the best small prover is made on DEV only (one attempt per
problem: pass@1); the confirmatory comparisons of the pre-registered protocol (docs/14_e18_lean.md) are on TEST:

  P1  swarm: B attempts taken in turn from the K provers   vs  the best small prover of dev, B attempts
  P2  swarm, B attempts                                    vs  the reference prover, B attempts
  P3  swarm: chain 0 of each prover, revised from Lean's   vs  the same provers, R+1 independent samples each
      errors (at most R+1 generations per prover)
B = the number of independent samples per prover on test (manifest `samples`, B = K in e18-1). P1-P3: exact
unconditional paired tests (essaim/stats.py) and Holm's correction over their three two-sided p-values; everything
else (token-matched versions, every prover's own revision, the whole swarm, complementarity) is exploratory and
listed as such in the multiplicity register written with the report.
Writes results/e18_report<suffix>.md and results/e18_summary<suffix>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import e18_lean as E
from essaim import lean
from essaim.results import read_manifest, read_rows
from essaim.stats import ALPHA, compare

RESULTS = Path(__file__).resolve().parent / "results"
DELTA = 2.0  # equivalence margin in points (as E4 and E11): reported, never part of the confirmatory family
COMMON = ("data", "bench", "prompt", "max_tokens", "lean", "checker", "screen", "samples", "problems", "n")
SWARM_ONLY = ("chains", "revisions", "ctx_per_slot")  # the reference is never revised (its own protocol)
IDENTITY = ("model", "revision", "gguf", "weights", "engine", "backend", "style", "thinking", "sampling")

Attempts = list  # [(verified, completion tokens)] in the order the system uses them


def fail(msg: str):
    raise SystemExit(f"E18 : {msg}")


def load(model: str, suffix: str, split: str, items: dict[str, dict], results: Path) -> tuple[dict, dict]:
    """{(id, sample, round): row} with `ok` recomputed, and the manifest."""
    path = E.result_path(model, suffix, split, results)
    man = read_manifest(path)
    if man is None:
        fail(f"{path.name} absent")
    rows = {(r["id"], r["sample"], r["round"]): r for r in read_rows(path, key=("id", "sample", "round"))}
    for r in rows.values():
        if r["id"] not in items:
            fail(f"{path.name} : problème hors de la partition {r['id']}")
        r["ok"] = recheck(r, items[r["id"]])
    check_complete(rows, man, items, path.name)
    return rows, man


def recheck(r: dict, item: dict) -> bool:
    """The attempt's verdict, from the raw completion and the stored Lean messages only."""
    where = f"{r['model']} {r['id']} {r['sample']}/{r['round']}"
    if r["build"] == "prompt_too_long":
        if r["verified"] or r.get("n_tokens") != 0:
            fail(f"{where} : tentative sans génération incohérente")
        return False
    built = lean.build(item, E.candidate(r["text"], r.get("reasoning_block")))
    if built["status"] != r["build"] or built.get("code") != r.get("code"):
        fail(f"{where} : fichier reconstruit différent de celui vérifié")
    if built["status"] != "ok":
        if r["verified"]:
            fail(f"{where} : vérifié sans fichier")
        return False
    if r.get("axioms_line") != built["axioms_line"] or not isinstance(r.get("lean"), dict):
        fail(f"{where} : vérification Lean absente")
    v = lean.verdict(r["lean"], built["name"], built["axioms_line"])
    if v["verified"] != r["verified"] or v["status"] != r["status"]:
        fail(f"{where} : verdict recalculé {v['status']} != {r['status']}")
    return v["verified"]


def check_complete(rows: dict, man: dict, items: dict, name: str):
    """Every problem of the partition, every sample's round 0; a chain stops exactly at its first success or after
    R revisions; nothing else."""
    s, c, rr = man["samples"], man["chains"], man["revisions"]
    if len(items) != man["n"]:
        fail(f"{name} : {man['n']} problèmes au manifeste, {len(items)} utilisables")
    for i in items:
        for k in range(s):
            last = rr if k < c else 0
            done = [x for x in range(last + 1) if (i, k, x) in rows]
            if done != list(range(len(done))) or not done:
                fail(f"{name} : {i} échantillon {k} incomplet")
            stop = next((x for x in done if rows[(i, k, x)]["ok"]), None)
            if (stop is not None and done[-1] != stop) or (stop is None and done[-1] != last):
                fail(f"{name} : {i} chaîne {k} incomplète ou prolongée après un succès")
    if any(k[1] >= s or k[2] > (rr if k[1] < c else 0) for k in rows):
        fail(f"{name} : tentatives hors protocole")


def check_manifests(mans: dict[tuple[str, str], dict]):
    """Same protocol for every model of a split (revision settings: the small provers only); the same model files,
    engine and sampling for a model's dev and test runs."""
    for split in E.SPLITS:
        here = {m: man for (m, s), man in mans.items() if s == split}
        for keys, models in ((COMMON, list(here)), (SWARM_ONLY, [m for m in here if m in E.SWARM])):
            ref = None
            for m in models:
                proto = {k: json.dumps(here[m].get(k), sort_keys=True) for k in keys}
                if ref is None:
                    ref = proto
                elif proto != ref:
                    fail(f"{m} {split} : protocole différent de celui des autres modèles")
    for m in E.SWARM:
        dv, te = mans[(m, "dev")], mans[(m, "test")]
        diff = [k for k in IDENTITY if dv.get(k) != te.get(k)]
        if diff:
            fail(f"{m} : dev et test ne viennent pas du même modèle, moteur ou réglage : {diff}")
    ref = mans[(E.REFERENCE, "test")]
    if ref["chains"] != 0:
        fail("la référence n'est pas révisée dans le protocole")


# ---------- systems ----------

def tokens(r: dict) -> int:
    if r.get("n_tokens") is None:
        fail(f"{r['model']} {r['id']} : nombre de jetons inconnu")
    return int(r["n_tokens"])


def independent(rows: dict, i: str, n: int) -> Attempts:
    """Samples 0..n-1, round 0."""
    return [(rows[(i, k, 0)]["ok"], tokens(rows[(i, k, 0)])) for k in range(n)]


def chain(rows: dict, i: str, sample: int = 0) -> Attempts:
    out, rnd = [], 0
    while (i, sample, rnd) in rows:
        out.append((rows[(i, sample, rnd)]["ok"], tokens(rows[(i, sample, rnd)])))
        rnd += 1
    return out


def round_robin(lists: list[Attempts], budget: int) -> Attempts:
    """Attempts interleaved in prover order (p1 a0, p2 a0, ..., p1 a1, ...), the first `budget` of them."""
    out, k = [], 0
    while len(out) < budget and any(k < len(l) for l in lists):
        out += [l[k] for l in lists if k < len(l)]
        k += 1
    return out[:budget]


def solved(att: Attempts) -> float:
    return float(any(ok for ok, _ in att))


def solved_within(att: Attempts, budget: int) -> float:
    total = 0
    for ok, t in att:
        total += t
        if total > budget:
            return 0.0
        if ok:
            return 1.0
    return 0.0


def spent(att: Attempts) -> int:
    return sum(t for _, t in att)


def token_matched(a: list[Attempts], b: list[Attempts]) -> tuple[list[float], list[float]]:
    """Per problem, both systems cut at T = the smaller of the completion tokens they spent on it."""
    xa, xb = [], []
    for u, v in zip(a, b):
        t = min(spent(u), spent(v))
        xa.append(solved_within(u, t))
        xb.append(solved_within(v, t))
    return xa, xb


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    out, run = {}, 0.0
    for k, name in enumerate(order):
        run = max(run, min(1.0, (len(order) - k) * pvals[name]))
        out[name] = run
    return out


def two_sided(g: dict) -> float:
    return min(1.0, 2 * min(g["p_superior"], g["p_inferior"]))


PLAN = {  # name: (A, B, family); A minus B
    "P1 essaim vs meilleur petit prouveur (dev)": ("swarm", "best", "primaire"),
    "P2 essaim vs référence": ("swarm", "reference", "primaire"),
    "P3 essaim révisé vs rééchantillonné": ("swarm_revise", "swarm_resample", "primaire"),
    "S1 essaim vs meilleur, à jetons égaux": ("swarm", "best", "exploratoire (jetons)"),
    "S2 essaim vs référence, à jetons égaux": ("swarm", "reference", "exploratoire (jetons)"),
    "S3 essaim révisé vs rééchantillonné, à jetons égaux": ("swarm_revise", "swarm_resample", "exploratoire (jetons)"),
    "S4 essaim complet (K x B tentatives) vs référence": ("swarm_all", "reference", "exploratoire"),
}


def compare_all(sy: dict[str, list[Attempts]]) -> dict:
    plan = dict(PLAN)
    plan.update({f"S5 révision vs rééchantillonnage, {m}": (f"revise:{m}", f"resample:{m}", "exploratoire")
                 for m in E.SWARM})
    comps = {}
    for name, (x, y, fam) in plan.items():
        if "jetons" in fam:
            va, vb = token_matched(sy[x], sy[y])
        else:
            va, vb = [solved(u) for u in sy[x]], [solved(u) for u in sy[y]]
        g = compare(va, vb, DELTA)
        g.update({"family": fam, "a": x, "b": y, "p_two_sided": two_sided(g)})
        comps[name] = g
    adj = holm({k: g["p_two_sided"] for k, g in comps.items() if g["family"] == "primaire"})
    for k, p in adj.items():
        comps[k]["p_holm"] = p
        comps[k]["verdict_holm"] = ("supérieur" if comps[k]["mean"] > 0 else "inférieur") if p < ALPHA else "non concluant"
    return comps


def analyse(suffix: str, results: Path = RESULTS) -> tuple[dict, list[str]]:
    rows: dict[str, dict[str, dict]] = {s: {} for s in E.SPLITS}
    mans: dict[tuple[str, str], dict] = {}
    ids: dict[str, list[str]] = {}
    for split in E.SPLITS:
        items = {it["id"]: it for it in E.usable(split, suffix, results=results)}
        ids[split] = sorted(items)
        for m in E.SWARM + ([E.REFERENCE] if split == "test" else []):
            rows[split][m], mans[(m, split)] = load(m, suffix, split, items, results)
    check_manifests(mans)
    te = mans[(E.SWARM[0], "test")]
    S, R = te["samples"], te["revisions"]
    if te["chains"] < 1 or S < R + 1 or S < len(E.SWARM):
        fail("le protocole exige une chaîne révisée, R+1 <= B et K <= B")
    dev_n = mans[(E.SWARM[0], "dev")]["samples"]
    dev_score = {m: st.mean(solved(independent(rows["dev"][m], i, dev_n)) for i in ids["dev"]) for m in E.SWARM}
    best = max(E.SWARM, key=lambda m: (dev_score[m], -E.SWARM.index(m)))  # ties: table order

    summary: dict = {"suffix": suffix, "swarm": E.SWARM, "reference": E.REFERENCE, "samples": S, "revisions": R,
                     "best_dev": best, "dev_score": dev_score, "delta": DELTA, "splits": {}}
    L = ["# E18 : preuves Lean vérifiées, un essaim de petits prouveurs contre le meilleur d'entre eux et un plus gros",
         "", f"miniF2F (dev : {len(ids['dev'])} problèmes de valid, test : {len(ids['test'])} de test, après l'oracle). "
         f"Test : {S} tentatives indépendantes par prouveur et par problème, chaîne 0 révisée au plus {R} fois depuis "
         "les erreurs de Lean ; la référence n'est pas révisée. Une preuve compte si Lean l'accepte (aucune erreur, "
         f"aucun `sorry`, axiomes standard seulement). Meilleur petit prouveur choisi sur dev (pass@{dev_n}) : "
         f"**{best}**. Tests exacts non conditionnels (essaim/stats.py), Holm sur P1-P3.", ""]
    for split in E.SPLITS:
        r, I = rows[split], ids[split]
        models = list(r)
        n_ind = S if split == "test" else dev_n
        sy = {f"seul:{m}": [independent(r[m], i, n_ind) for i in I] for m in models}
        sy["swarm_all"] = [round_robin([independent(r[m], i, n_ind) for m in E.SWARM], n_ind * len(E.SWARM)) for i in I]
        if split == "test":
            sy["swarm"] = [round_robin([independent(r[m], i, S) for m in E.SWARM], S) for i in I]
            sy["best"] = sy[f"seul:{best}"]
            sy["reference"] = sy[f"seul:{E.REFERENCE}"]
            for m in E.SWARM:
                sy[f"revise:{m}"] = [chain(r[m], i) for i in I]
                sy[f"resample:{m}"] = [independent(r[m], i, R + 1) for i in I]
            sy["swarm_revise"] = [round_robin([chain(r[m], i) for m in E.SWARM], (R + 1) * len(E.SWARM)) for i in I]
            sy["swarm_resample"] = [round_robin([independent(r[m], i, R + 1) for m in E.SWARM], (R + 1) * len(E.SWARM))
                                    for i in I]
        acc = {k: 100 * st.mean(solved(x) for x in v) for k, v in sy.items()}
        toks = {k: st.mean(spent(x) for x in v) for k, v in sy.items()}
        L += [f"## {split} ({len(I)} problèmes)", "", "| système | réussite (%) | jetons générés par problème |",
              "| --- | --- | --- |"] + [f"| {k} | {acc[k]:.1f} | {toks[k]:.0f} |" for k in sorted(acc)]
        status = {m: {} for m in models}
        for m in models:
            for row in r[m].values():
                status[m][row["status"]] = status[m].get(row["status"], 0) + 1
        L += ["", "Statuts de toutes les tentatives : " + " ; ".join(
            f"{m.split('/')[-1]} {dict(sorted(c.items()))}" for m, c in status.items()), ""]
        by = {m: {i for i, x in zip(I, sy[f"seul:{m}"]) if solved(x)} for m in E.SWARM}
        rec = {"n": len(I), "accuracy": acc, "mean_tokens": toks, "status": status,
               "complementarity": {"union": len(set().union(*by.values())),
                                   "only_this_prover": {m: len(s - set().union(*(by[o] for o in E.SWARM if o != m)))
                                                        for m, s in by.items()}}}
        if split == "test":
            comps = compare_all(sy)
            L += ["| comparaison (A moins B) | famille | différence [IC 95 %] | p bilatéral | p Holm | verdict |",
                  "| --- | --- | --- | --- | --- | --- |"]
            for k, g in comps.items():
                holm_p = f"{g['p_holm']:.3f}" if "p_holm" in g else "-"
                L.append(f"| {k} | {g['family']} | {g['mean']:+.1f} [{g['lo95']:+.1f} ; {g['hi95']:+.1f}] | "
                         f"{g['p_two_sided']:.3f} | {holm_p} | {g.get('verdict_holm', g['verdict'] + ' (non corrigé)')} |")
            L += ["", "Registre de multiplicité : trois comparaisons primaires (P1-P3), corrigées par Holm à 5 % "
                  "bilatéral ; toutes les autres sont exploratoires, sans correction, et ne fondent aucune conclusion.",
                  ""]
            rec["comparisons"] = comps
            rec["multiplicity"] = {"primary": [k for k, g in comps.items() if g["family"] == "primaire"],
                                   "method": "Holm, two-sided exact unconditional p-values", "alpha": ALPHA,
                                   "exploratory": [k for k, g in comps.items() if g["family"] != "primaire"]}
        summary["splits"][split] = rec
    return summary, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_colab")
    a = ap.parse_args()
    summary, L = analyse(a.suffix)
    (RESULTS / f"e18_report{a.suffix}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / f"e18_summary{a.suffix}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
