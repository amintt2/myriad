"""E12 analysis: the E4 swarm (7 small models of 7 families) against the big single references on GPQA Diamond
and SciCode, the benchmarks Artificial Analysis uses (offline, from run_aa.py and exec_scicode.py).

    uv run python analyze_e12.py --suffix _colab

GPQA Diamond (198 questions, 4 options, chance 25 %): no dev/test split (a split of 198 questions would
leave 99 for the test: no power). Nothing is fitted on GPQA. The weighted vote uses the weights frozen from E4's
MMLU-Pro DEV run (results/e4_summary<suffix>.json: the hardest multiple-choice benchmark of E4), and the "best
peer" is the one best on E4's MMLU-Pro dev; the plain vote has no parameter. Per system: accuracy with its Wilson
95 % interval and the exact one-sided binomial test against chance; swarm minus each reference with the same
paired tests as E4 (essaim/stats.py, margin +-DELTA points: with 198 questions the equivalence test has little
power, which the verdicts say honestly).

SciCode (Artificial Analysis' "scientific coding"; essaim/scicode.py): each model writes the sub-problems of a
problem in order, seeing its own earlier code, and the official tests grade them (exec_scicode.py). Sub-problems
the dev reference code fails in our harness (--oracle run) are left out of dev for everyone.
Test has no references: every officially graded test step is kept. dev (15 problems,
50 sub-problems) picks the best of the 7 peers; test (65 problems, 288 graded, 3 skipped) is reported once.
There is no swarm DECISION rule for code without execution of candidates against each other (E11 needed a
shared set of inputs; SciCode tests build their own inputs), so the swarm row is an upper bound only: "at least
one of the 7 peers passes" (oracle), clearly labelled as such, next to the best-dev peer and the references.
Writes results/e12_report<suffix>.md and results/e12_summary<suffix>.json (aggregates only: no question text).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from analyze_e4 import DELTA, EXTRA, FAMILIES, REFS, decide
from essaim import answers, gpqa, scicode
from essaim.results import read_manifest, read_rows
from essaim.stats import _norm_ppf, compare
from exec_scicode import HARNESS_VERSION, exec_path, gen_digest
from run_aa import path_for

RESULTS = Path(__file__).resolve().parent / "results"
CHANCE = 0.25
GPQA_PROTOCOL = ("data", "prompt", "max_tokens", "temperature", "seed", "thinking", "ctx_per_slot", "bench", "logprobs")
SCI_PROTOCOL = ("data", "prompt", "max_tokens", "temperature", "seed", "thinking", "ctx_per_slot", "background",
                "official_commit", "skipped_steps", "skipped_code", "backend", "bench")
SCI_EXEC_PROTOCOL = ("h5", "official_commit", "harness", "timeout_s", "data", "skipped_code")
IDENTITY = ("model", "revision", "gguf", "weights", "engine", "backend")


# ---------- statistics ----------

def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion (points)."""
    if n == 0:
        return 0.0, 100.0
    z, p = _norm_ppf(0.5 + level / 2), k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def binom_tail(k: int, n: int, p: float) -> float:
    """Exact P(X >= k), X ~ Binomial(n, p): one-sided test of "better than chance"."""
    return min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def check_same(manifests: dict[str, dict], keys: tuple[str, ...], what: str):
    ref_name, ref = next(iter(manifests.items()))
    for name, man in manifests.items():
        diff = {k: (ref.get(k), man.get(k)) for k in keys if ref.get(k) != man.get(k)}
        if diff:
            raise SystemExit(f"{what} : {name} et {ref_name} n'ont pas le même protocole : {diff}")


# ---------- GPQA ----------

def load_gpqa(model: str, suffix: str, items: dict[str, dict]) -> tuple[dict, dict] | None:
    path = path_for(model, suffix, "gpqa", "all")
    man = read_manifest(path)
    if man is None:
        return None
    rows = {r["id"]: r for r in read_rows(path, key=("id",))}
    if len(rows) != man["n"] or set(rows) != set(items):
        raise SystemExit(f"{path.name} : {len(rows)} réponses sur {man['n']}, run incomplet ou autres questions")
    for i, r in rows.items():  # re-extracted from the raw text with the current grader
        r["answer"] = answers.extract("gpqa", r["text"], items[i], ended=r["finish"] == "stop")
        r["gold"] = answers.gold("gpqa", items[i])
    return rows, man


def e4_frozen(suffix: str) -> dict | None:
    p = RESULTS / f"e4_summary{suffix}.json"
    if not p.exists():
        return None
    b = json.loads(p.read_text(encoding="utf-8"))["benches"].get("mmlupro")
    return {"weights": b["weights"], "p_dev": b["p_dev"], "best": b["best_dev"]} if b else None


def system_line(name: str, correct: list[float]) -> tuple[str, dict]:
    n, k = len(correct), int(sum(correct))
    lo, hi = wilson(k, n)
    p = binom_tail(k, n, CHANCE)
    return (f"| {name} | {100 * k / n:.1f} | [{lo:.1f} ; {hi:.1f}] | {p:.3f} |",
            {"acc": 100 * k / n, "lo95": lo, "hi95": hi, "p_chance": p, "n": n})


def gpqa_section(suffix: str, summary: dict) -> list[str]:
    items = {x["id"]: x for x in gpqa.diamond()}
    ids = sorted(items)
    fams = list(FAMILIES)
    models = list(FAMILIES.values()) + EXTRA + REFS
    loaded = {m: load_gpqa(m, suffix, items) for m in models}
    L = ["## GPQA Diamond (198 questions, 4 options, hasard 25 %)", ""]
    absent = [m for m in models if loaded[m] is None]
    if any(FAMILIES[f] in absent for f in fams):
        return L + [f"incomplet : {absent}", ""]
    present = {m: v for m, v in loaded.items() if v is not None}
    check_same({m: v[1] for m, v in present.items()}, GPQA_PROTOCOL, "gpqa")
    frozen = e4_frozen(suffix)
    w = {f: frozen["weights"][f] for f in fams} if frozen else None
    correct = {m: [float(v[0][i]["answer"] == v[0][i]["gold"]) for i in ids] for m, v in present.items()}

    def swarm(rule: str, i: str) -> str | None:
        return decide([(present[FAMILIES[f]][0][i]["answer"], 1.0 if rule == "vote" else w[f], None) for f in fams])

    sw = {"vote": [float(swarm("vote", i) == present[FAMILIES[fams[0]]][0][i]["gold"]) for i in ids]}
    if w:
        sw["wvote"] = [float(swarm("wvote", i) == present[FAMILIES[fams[0]]][0][i]["gold"]) for i in ids]
    best_m = frozen["best"] if frozen else None  # never chosen on GPQA itself (that would be selection on the test)
    oracle = [float(any(correct[FAMILIES[f]][n] for f in fams)) for n in range(len(ids))]
    abst = {m: sum(v[0][i]["answer"] is None for i in ids) for m, v in present.items()}
    L += ["Chaque modèle répond seul, une fois (glouton), même prompt que E4. **Aucun paramètre n'est ajusté sur "
          "GPQA** : poids du vote pondéré et « meilleur pair » viennent du dev de MMLU-Pro (E4)"
          + ("" if frozen else " ; ABSENTS ici (pas de résumé E4 pour ce suffixe) : seuls le vote simple et les "
             "références sont rapportés") + ". Test exact unilatéral contre le hasard (25 %).", "",
          "| système | exactitude (%) | IC 95 % (Wilson) | p exact contre le hasard |", "| --- | --- | --- | --- |"]
    out = {}
    for f in fams:
        line, out[FAMILIES[f]] = system_line(f"{FAMILIES[f]} seul ({f})", correct[FAMILIES[f]])
        L.append(line)
    for m in EXTRA:
        if m in correct:
            line, out[m] = system_line(f"{m} seul (hors essaim)", correct[m])
            L.append(line)
    for rule, label in (("vote", "essaim 7 familles, vote"), ("wvote", "essaim 7 familles, vote pondéré (poids de E4)")):
        if rule in sw:
            line, out[f"swarm_{rule}"] = system_line(f"**{label}**", sw[rule])
            L.append(line)
    for m in REFS:
        if m in correct:
            line, out[m] = system_line(f"{m} seul (référence)", correct[m])
            L.append(line)
    line, out["oracle"] = system_line("oracle (au moins un pair juste)", oracle)
    L += [line, ""]
    comps = {}
    rule = "wvote" if "wvote" in sw else "vote"
    L += [f"Essaim ({rule}) moins chaque système, paires par question (marge ±{DELTA} points) :", "",
          "| essaim moins | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |", "| --- | --- | --- | --- |"]
    for name, other in ([(f"meilleur pair de E4 ({best_m})", correct.get(best_m))] if best_m else []) + \
            [(m, correct[m]) for m in REFS if m in correct]:
        if other is None:
            continue
        g = compare(sw[rule], other, DELTA)
        comps[name] = g
        L.append(f"| {name} | {g['mean']:+.1f} [{g['lo95']:+.1f} ; {g['hi95']:+.1f}] | "
                 f"{g['p_low_margin']:.3f} ; {g['p_high_margin']:.3f} | {g['verdict']} |")
    L += ["", "Réponses sans lettre extractible (comptées fausses) : "
          + ", ".join(f"{m.split('/')[-1]} {k}" for m, k in abst.items() if k) + "." if any(abst.values()) else
          "Toutes les réponses contiennent une lettre extractible.", ""]
    summary["gpqa"] = {"n": len(ids), "systems": out, "comparisons": comps, "frozen_from": "e4 mmlupro dev" if frozen else None,
                       "weights": w, "best_peer": best_m, "rule": rule}
    return L


# ---------- SciCode ----------

def load_sci(tag: str, suffix: str, split: str, oracle: bool = False) -> dict | None:
    path = exec_path(tag, suffix, split)
    man = read_manifest(path)
    if man is None:
        return None
    rows = {r["id"]: r for r in read_rows(path, key=("id",))}
    if len(rows) != man["n"]:
        raise SystemExit(f"{path.name} : {len(rows)} résultats sur {man['n']}, exécution incomplète")
    if man.get("oracle") != oracle or man.get("harness") != HARNESS_VERSION or \
            man.get("skipped_code") != scicode.SKIPPED_SHA256:
        raise SystemExit(f"{path.name} : oracle ou version du harnais incompatible")
    need = {scicode.row_id(s) for p in scicode.problems(split) for s in p["steps"] if not scicode.is_skipped(p, s)}
    if set(rows) != need:
        raise SystemExit(f"{path.name} : sous-problèmes différents du split {split}")
    return {"rows": rows, "man": man}


def sci_vectors(probs: list[dict], res: dict, excluded: set[str]) -> tuple[list[str], dict[str, bool], dict[str, bool]]:
    """(graded step ids kept, step id -> passes, problem id -> all its kept steps pass)."""
    ids, ok, solved = [], {}, {}
    for p in probs:
        keep = [scicode.row_id(s) for s in p["steps"] if not scicode.is_skipped(p, s) and scicode.row_id(s) not in excluded]
        ids += keep
        for i in keep:
            ok[i] = bool(res[i]["ok"])
        if keep:
            solved[p["id"]] = all(ok[i] for i in keep)
    return ids, ok, solved


def sci_section(suffix: str, summary: dict) -> list[str]:
    L = ["## SciCode (sous-problèmes, tests officiels, glouton, un seul passage)", ""]
    models = list(FAMILIES.values()) + EXTRA + REFS
    data_by = {sp: {m: load_sci(m, suffix, sp) for m in models} for sp in ("dev", "test")}
    oracle = {sp: load_sci("oracle", suffix, sp, oracle=True) for sp in scicode.ORACLE_SPLITS}
    fams = list(FAMILIES)
    if any(oracle[sp] is None for sp in oracle) or any(data_by[sp][FAMILIES[f]] is None for sp in data_by for f in fams):
        return L + ["incomplet : exécution de l'oracle ou d'un pair de l'essaim absente.", ""]
    present = {sp: {m: v for m, v in data_by[sp].items() if v is not None} for sp in data_by}
    check_same({f"{sp}/{m}": v["man"] for sp in present for m, v in present[sp].items()},
               SCI_EXEC_PROTOCOL, "scicode dev/test (exécution)")
    for sp in present:
        check_same({m: v["man"] for m, v in present[sp].items()}, SCI_EXEC_PROTOCOL,
                   f"scicode {sp} (exécution)")
        if sp in oracle:
            check_same({"oracle": oracle[sp]["man"], **{m: v["man"] for m, v in present[sp].items()}},
                       SCI_EXEC_PROTOCOL, f"scicode {sp} (oracle et modèles)")
        gens = {m: read_manifest(path_for(m, suffix, "scicode", sp)) for m in present[sp]}
        for m, g in gens.items():
            if g is None or g["prompt"] != scicode.PROMPT_VERSION:
                raise SystemExit(f"scicode : {m} {sp} absent ou généré avec un autre prompt")
            check_same({"generation": g, "execution": present[sp][m]["man"]["generation"]},
                       SCI_PROTOCOL + IDENTITY + ("n", "n_steps"), f"scicode {m} {sp} (protocole noté)")
            rows = {r["id"]: r for r in read_rows(path_for(m, suffix, "scicode", sp), key=("id",))}
            if present[sp][m]["man"]["generation"].get("rows_sha256") != gen_digest(rows):
                raise SystemExit(f"scicode : {m} {sp} : les notes ne correspondent plus aux textes générés (régénéré ?)")
        check_same(gens, SCI_PROTOCOL + ("n", "n_steps"), f"scicode {sp} (génération)")
    for m in present["test"]:  # the same model files for a model's dev and test runs
        if m in present["dev"]:
            a, b = (read_manifest(path_for(m, suffix, "scicode", sp)) for sp in ("dev", "test"))
            diff = {k: (a.get(k), b.get(k)) for k in IDENTITY if a.get(k) != b.get(k)}
            if diff:
                raise SystemExit(f"scicode {m} : dev et test ne viennent pas du même modèle ou moteur : {diff}")
    probs = {sp: scicode.problems(sp) for sp in ("dev", "test")}
    excluded = {sp: {i for i, r in oracle[sp]["rows"].items() if not r["ok"]} if sp in oracle else set() for sp in probs}
    if not any(scicode.row_id(s) not in excluded["dev"] for p in probs["dev"] for s in p["steps"]
               if not scicode.is_skipped(p, s)):
        raise SystemExit("scicode : aucun sous-problème dev utilisable après contrôle du banc")
    L += [f"Sous-problèmes dev écartés parce que **le code de référence les échoue dans notre banc** : "
          f"{len(excluded['dev'])}. Test sans référence : **aucun filtrage par oracle**, tests officiels uniquement. "
          "Étapes à code officiel fourni (non générées, non notées) : "
          + ", ".join(f"{p}.{s}" for p, s in sorted(scicode.SKIPPED)) + ".", ""]
    vec = {sp: {m: sci_vectors(probs[sp], v["rows"], excluded[sp]) for m, v in present[sp].items()} for sp in present}
    ids_dev, ids_test = vec["dev"][FAMILIES[fams[0]]][0], vec["test"][FAMILIES[fams[0]]][0]
    rate = lambda sp, m: st.mean(float(vec[sp][m][1][i]) for i in vec[sp][m][0])
    best = max(fams, key=lambda f: (rate("dev", FAMILIES[f]), -fams.index(f)))
    bm = FAMILIES[best]
    L += [f"dev ({len(ids_dev)} sous-problèmes) ne sert qu'à choisir le meilleur pair : **{bm}** "
          f"({100 * rate('dev', bm):.1f} %). Résultats sur test ({len(ids_test)} sous-problèmes, "
          f"{len(vec['test'][bm][2])} problèmes) :", "",
          "| système | sous-problèmes réussis (%) | IC 95 % (Wilson) | problèmes entièrement résolus |",
          "| --- | --- | --- | --- |"]
    out, tests = {}, {}

    def line(name, key, flags, solved):
        k, n = int(sum(flags)), len(flags)
        lo, hi = wilson(k, n)
        s, t = sum(solved), len(solved)
        out[key] = {"steps_pct": 100 * k / n, "lo95": lo, "hi95": hi, "steps": k, "n_steps": n,
                    "problems_solved": s, "n_problems": t}
        L.append(f"| {name} | {100 * k / n:.1f} ({k}/{n}) | [{lo:.1f} ; {hi:.1f}] | {s}/{t} |")

    for f in fams:
        m = FAMILIES[f]
        line(f"{m} seul ({f})", m, [float(vec["test"][m][1][i]) for i in ids_test], list(vec["test"][m][2].values()))
        tests[m] = [float(vec["test"][m][1][i]) for i in ids_test]
    for m in EXTRA:
        if m in vec["test"]:
            line(f"{m} seul (hors essaim)", m, [float(vec["test"][m][1][i]) for i in ids_test], list(vec["test"][m][2].values()))
    for m in REFS:
        if m in vec["test"]:
            tests[m] = [float(vec["test"][m][1][i]) for i in ids_test]
            line(f"{m} seul (référence)", m, tests[m], list(vec["test"][m][2].values()))
    orc = [float(any(vec["test"][FAMILIES[f]][1][i] for f in fams)) for i in ids_test]
    orc_solved = [all(any(vec["test"][FAMILIES[f]][1][i] for f in fams)
                      for i in ids_test if i.split(".")[0] == pid) for pid in vec["test"][bm][2]]
    line("*plafond : au moins un pair juste par sous-problème (n'est pas une décision d'essaim)*", "oracle_upper_bound",
         orc, orc_solved)
    L.append("")
    comps = {}
    L += [f"Meilleur pair de dev ({bm}) moins chaque référence (paires par sous-problème, marge ±{DELTA} points) :", "",
          "| moins | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |", "| --- | --- | --- | --- |"]
    for m in REFS:
        if m in tests:
            g = compare(tests[bm], tests[m], DELTA)
            comps[m] = g
            L.append(f"| {m} | {g['mean']:+.1f} [{g['lo95']:+.1f} ; {g['hi95']:+.1f}] | "
                     f"{g['p_low_margin']:.3f} ; {g['p_high_margin']:.3f} | {g['verdict']} |")
    L += ["", "Aucune décision d'essaim n'est évaluée sur SciCode (voir l'en-tête) : le plafond dit seulement ce qu'une "
          "sélection parfaite pourrait espérer.", ""]
    summary["scicode"] = {"n_steps_test": len(ids_test), "excluded": {k: sorted(v) for k, v in excluded.items()},
                          "best_dev_peer": bm, "systems": out, "comparisons": comps}
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_colab")
    ap.add_argument("--benches", nargs="+", default=None, choices=["gpqa", "scicode"])
    a = ap.parse_args()
    gpqa_path = gpqa.local_path()
    if a.benches is None:
        a.benches = ["gpqa", "scicode"] if gpqa_path is not None else ["scicode"]
    summary: dict = {"swarm": FAMILIES, "refs": REFS, "delta": DELTA}
    L = ["# E12 : l'essaim E4 sur des bancs plus réels (GPQA Diamond, SciCode)", "",
         "Mêmes 7 petits modèles de 7 familles et mêmes références qu'E4. Tests exacts non conditionnels "
         f"(essaim/stats.py) ; marge d'équivalence ±{DELTA} points. Rapport d'agrégats : aucune question de GPQA "
         "n'y figure (conditions d'accès de GPQA).", ""]
    if "gpqa" in a.benches:
        if gpqa_path is not None:
            gpqa.check_file(gpqa_path.read_bytes())
        L += gpqa_section(a.suffix, summary)
    if "scicode" in a.benches:
        L += sci_section(a.suffix, summary)
    (RESULTS / f"e12_report{a.suffix}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / f"e12_summary{a.suffix}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
