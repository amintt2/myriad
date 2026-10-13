"""E16 analysis: a bigger model in low-bit GGUF, or a small model in Q8_0, at the same memory per peer (offline).

    uv run python analyze_e16.py            # reads results/solo_*, code_*, codeexec_e16_*, solo_e16speed_*

Systems (docs/13_e16_quantification.md has the pre-registered protocol; everything below follows it):
  small   the 7 families of E4 (analyze_e4.FAMILIES), Q8_0, results suffix _colab  (E4, E11)
  big     Qwen3.5-9B, Gemma-4-12B, Ministral-3-14B in the quantizations of e16_quant.QUANT (suffix _<quant>_colab)
  refs    the same three in Q8_0 (E4's references): the cost of the quantization itself
A peer is "in budget" when its weights file is at most BUDGET_GIB (6.5 GiB: the 8 GB consumer card minus KV cache and
buffers). Every choice is made on DEV: the best small peer, the best quantization of each big model, the best big
peer, the swarm weights (analyze_e4.fit_weights, refitted for each swarm on its own members), the selection rule
of the code swarm (frozen from E11). TEST is only read for the frozen choices, and every number on TEST is reported
(no search on test). Comparisons use the same exact tests as E4 and E11 (essaim/stats.py: Tango score interval,
TOST with DELTA points, superiority at 2.5 %), per benchmark and on the questions put end to end.
Writes results/e16_report.md and results/e16_summary.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

import e16_quant as Q
from analyze_e4 import BENCHES, DELTA, EXTRA, FAMILIES, IDENTITY, MANIFESTS, PROTOCOL, REFS, accuracy, decide, fit_weights
from analyze_e4 import load as load_e4
from essaim import models
from essaim.results import read_manifest, read_rows
from essaim.stats import compare

RESULTS = Path(__file__).resolve().parent / "results"
SMALL_SUFFIX = "_colab"
SPLITS = ("dev", "test")


@dataclass
class Sys:
    """One peer configuration: a model file, with its answers on every benchmark and split."""
    label: str
    model: str
    tag: str  # quantization type, "Q8_0" for E4's own files
    size: int  # bytes of the weights file (from the manifest)
    rows: dict = field(default_factory=dict)  # (bench, split) -> {id: row}
    mans: dict = field(default_factory=dict)  # (bench, split) -> manifest

    @property
    def gib(self) -> float:
        return self.size / Q.GIB

    @property
    def bpw(self) -> float:
        """File bits per parameter (embeddings and metadata included), not the nominal bits of the quantization type."""
        return self.size * 8 / (models.total_b(self.model) * 1e9)

    @property
    def quantized(self) -> bool:
        return self.tag != "Q8_0"


def load_system(model: str, tag: str, suffix: str) -> Sys | None:
    """All benchmarks and splits of one configuration, or None if its files are absent (analyze_e4.load stops on a
    file that exists but is incomplete)."""
    s = Sys(Q.label(model, tag), model, tag, 0)
    for b in BENCHES:
        for sp in SPLITS:
            r = load_e4(model, suffix, b, sp)
            if r is None:
                return None
            s.rows[(b, sp)] = r
            s.mans[(b, sp)] = dict(MANIFESTS[(model, b, sp)])  # MANIFESTS is keyed by model: copy before the next load
    s.size = s.mans[(BENCHES[0], "dev")]["weights"]["size"]
    return s


def check_protocol(systems: list[Sys]):
    """Same questions, prompts and generation settings for every system; for each system, dev and test from the same
    file and engine (weights are fitted on one split and applied to the other); quantized files are the pinned ones."""
    for b in BENCHES:
        ref = None
        for s in systems:
            for sp in SPLITS:
                proto = {k: s.mans[(b, sp)].get(k) for k in PROTOCOL if k != "data"}
                if ref is None:
                    ref = proto
                elif proto != ref:
                    raise SystemExit(f"{s.label} {b} {sp} : protocole différent {proto} != {ref}")
            diff = {k: (s.mans[(b, "dev")].get(k), s.mans[(b, "test")].get(k)) for k in IDENTITY
                    if s.mans[(b, "dev")].get(k) != s.mans[(b, "test")].get(k)}
            if diff:
                raise SystemExit(f"{s.label} {b} : dev et test ne viennent pas du même modèle ou moteur : {diff}")
        for sp in SPLITS:
            datas = {json.dumps(s.mans[(b, sp)].get("data"), sort_keys=True) for s in systems}
            if len(datas) != 1:
                raise SystemExit(f"{b} {sp} : jeux de données différents d'un système à l'autre")
    for s in systems:
        if s.quantized:
            q = Q.get(s.model, s.tag)
            for (b, sp), man in s.mans.items():
                w = man.get("weights") or {}
                if w.get("sha256") != q.sha256 or w.get("size") != q.size:
                    raise SystemExit(f"{s.label} {b} {sp} : le fichier utilisé n'est pas celui épinglé dans e16_quant.QUANT")
        for b in BENCHES:
            for sp in SPLITS:
                if sorted(s.rows[(b, sp)]) != sorted(systems[0].rows[(b, sp)]):
                    raise SystemExit(f"{s.label} {b} {sp} : pas les mêmes questions")


def engines(systems: list[Sys]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in systems:
        out.setdefault(str(s.mans[(BENCHES[0], "dev")].get("engine")), []).append(s.label)
    return out


# ---------- pure helpers on loaded systems ----------

def ids_of(s: Sys, b: str, sp: str) -> list[str]:
    return sorted(s.rows[(b, sp)])


def acc(s: Sys, b: str, sp: str) -> float:
    return accuracy(s.rows[(b, sp)], ids_of(s, b, sp))


def macro(s: Sys, sp: str) -> float:
    """Mean accuracy over the four benchmarks, in points."""
    return 100 * st.mean(acc(s, b, sp) for b in BENCHES)


def correct(s: Sys, b: str, sp: str = "test") -> list[float]:
    r = s.rows[(b, sp)]
    return [float(r[i]["answer"] == r[i]["gold"]) for i in ids_of(s, b, sp)]


def best_by_dev(cands: list[Sys]) -> Sys:
    """Highest macro DEV accuracy; ties go to the smaller file, then to the label (deterministic)."""
    return max(cands, key=lambda s: (round(macro(s, "dev"), 9), -s.size, s.label))


def in_budget(cands: list[Sys], gib: float | None = None) -> list[Sys]:
    limit = Q.BUDGET_GIB if gib is None else gib
    return [s for s in cands if s.gib <= limit]


def vote_fit(peers: list[Sys], b: str):
    """Weights of the swarm of `peers` on benchmark b, fitted on dev: returns (w by label, c, vote function)."""
    dev = {p.label: p.rows[(b, "dev")] for p in peers}
    ids_dev = ids_of(peers[0], b, "dev")
    _, _, c, w = fit_weights(dev, ids_dev)
    ids = ids_of(peers[0], b, "test")

    def vote(subset: list[Sys]) -> list[float]:
        out = []
        for i in ids:
            ans = [(p.rows[(b, "test")][i]["answer"], w[p.label], p.rows[(b, "test")][i]["mean_logp"]) for p in subset]
            out.append(float(decide(ans) == peers[0].rows[(b, "test")][i]["gold"]))
        return out
    return w, c, vote


def swarm_vectors(peers: list[Sys]) -> dict[str, list[float]]:
    """Per benchmark, right/wrong on TEST of the weighted vote of `peers` (weights fitted on their own dev answers)."""
    peers = sorted(peers, key=lambda p: p.label)
    return {b: vote_fit(peers, b)[2](peers) for b in BENCHES}


def scaling(peers: list[Sys]) -> dict[int, dict[str, float]]:
    """Mean TEST accuracy (macro over benchmarks, points) of the weighted vote over all subsets of k peers, k = 1..n,
    and over the k best peers by dev macro. Weights are fitted once, on all the peers, as in E4."""
    peers = sorted(peers, key=lambda p: p.label)
    fits = {b: vote_fit(peers, b) for b in BENCHES}
    best_order = sorted(peers, key=lambda p: (-macro(p, "dev"), p.label))
    out = {}
    for k in range(1, len(peers) + 1):
        subs = list(itertools.combinations(peers, k))
        mean_all = st.mean(st.mean(st.mean(fits[b][2](list(sub))) for b in BENCHES) for sub in subs)
        best = st.mean(st.mean(fits[b][2](best_order[:k])) for b in BENCHES)
        out[k] = {"mean_all_subsets": 100 * mean_all, "best_k_by_dev": 100 * best, "n_subsets": len(subs)}
    return out


def pooled(vecs: dict[str, list[float]]) -> list[float]:
    return [x for b in BENCHES for x in vecs[b]]


def alone_vectors(s: Sys) -> dict[str, list[float]]:
    return {b: correct(s, b) for b in BENCHES}


def macro_of(vecs: dict[str, list[float]]) -> float:
    return 100 * st.mean(st.mean(vecs[b]) for b in BENCHES)


def versus(a: dict[str, list[float]], b: dict[str, list[float]]) -> dict:
    """Paired comparison A minus B, on the pooled questions and per benchmark."""
    return {"pooled": compare(pooled(a), pooled(b), DELTA), **{bench: compare(a[bench], b[bench], DELTA) for bench in BENCHES}}


def cell(g: dict) -> str:
    return f"{g['mean']:+.1f} [{g['lo95']:+.1f} ; {g['hi95']:+.1f}] ; {g['verdict']}"


def abstain_rate(s: Sys, sp: str = "test") -> float:
    rows = [r for b in BENCHES for r in s.rows[(b, sp)].values()]
    return 100 * sum(r["answer"] is None for r in rows) / len(rows)


def truncated_rate(s: Sys, sp: str = "test") -> float:
    rows = [r for b in BENCHES for r in s.rows[(b, sp)].values()]
    return 100 * sum(r["finish"] != "stop" for r in rows) / len(rows)


def mean_tokens(s: Sys, sp: str = "test") -> float:
    rows = [r for b in BENCHES for r in s.rows[(b, sp)].values()]
    return st.mean((r["n_tokens"] or 0) for r in rows)


# ---------- speed ----------

def read_speed() -> dict[str, dict]:
    path = RESULTS / f"solo_{Q.SPEED_STEM}_colab_bench_all.jsonl"
    if read_manifest(path) is None:
        return {}
    return {r["label"]: r for r in read_rows(path, key=("label",))}


def consumer_calibration() -> list[tuple[float, int]]:
    speeds = json.loads((Path(__file__).resolve().parent / "speeds_consumer.json").read_text(encoding="utf-8"))
    return [(speeds[m], size) for m, size in Q.CALIBRATION_DENSE if m in speeds]


def fmt_consumer(size: int, calib: list[tuple[float, int]]) -> str:
    lo, mid, hi = Q.estimate_consumer_tps(size, calib)
    return f"{mid:.0f} ({lo:.0f}-{hi:.0f})"


# ---------- report ----------

def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def build_mc(systems: list[Sys], L: list[str], summary: dict, speed: dict[str, dict]):
    small = [s for s in systems if s.tag == "Q8_0" and s.model in FAMILIES.values()]
    extra = [s for s in systems if s.tag == "Q8_0" and s.model in EXTRA]
    refs = [s for s in systems if s.tag == "Q8_0" and s.model in REFS]
    big = [s for s in systems if s.quantized]
    calib = consumer_calibration()
    gpus = sorted({str(sp.get("gpu")) for sp in speed.values()})
    if len(gpus) > 1:
        raise SystemExit(f"vitesses mesurées sur plusieurs GPU ({gpus}) : table refusée")
    gpu = gpus[0] if gpus else "non mesuré"
    L += ["## 1. Chaque système seul", "",
          f"Budget mémoire d'un pair : fichier d'au plus {Q.BUDGET_GIB} Gio. Exactitude TEST en %, moyenne des quatre "
          "benchmarks (macro) ; DEV sert aux choix. Fichier en Gio, bits par poids du fichier (plongements inclus), "
          "abstention = pas de réponse extractible, tronqué = pas de fin naturelle (limite de jetons), jetons = moyenne "
          f"par réponse. Vitesse sur le GPU de la VM ({gpu}) : llama-bench seul sur le GPU, génération tg128 puis traitement "
          "du prompt pp512 (jetons/s) ; estimation grand public (RX 6650 XT) : modèle de bande passante calibré sur "
          "`speeds_consumer.json`, intervalle = bande passante minimale et maximale observée, **non mesurée**.", "",
          "| système | Gio | bits/poids | " + " | ".join(BENCHES) + " | macro TEST | macro DEV | abst. % | tronqué % | jetons "
          "| VM tg128 / pp512 | RX 6650 XT estimé tok/s |", "| --- | --- | --- | " + " | ".join("---" for _ in BENCHES)
          + " | --- | --- | --- | --- | --- | --- | --- |"]
    rows_summary = {}
    for group, items in (("petit Q8_0", small + extra), ("référence Q8_0", refs), ("gros quantifié", big)):
        for s in sorted(items, key=lambda s: (s.model, s.gib)):
            sp = speed.get(s.label)
            sps = f"{sp['tg']['tps']:.0f} / {sp['pp']['tps']:.0f}" if sp else "—"
            L.append(f"| {s.label} ({group}){'' if s.gib <= Q.BUDGET_GIB else ' hors budget'} | {s.gib:.2f} | {s.bpw:.2f} | "
                     + " | ".join(f"{100 * acc(s, b, 'test'):.1f}" for b in BENCHES)
                     + f" | **{macro(s, 'test'):.1f}** | {macro(s, 'dev'):.1f} | {abstain_rate(s):.1f} | "
                       f"{truncated_rate(s):.1f} | {mean_tokens(s):.0f} | {sps} | {fmt_consumer(s.size, calib)} |")
            rows_summary[s.label] = {"gib": s.gib, "bpw": s.bpw, "group": group, "in_budget": s.gib <= Q.BUDGET_GIB,
                                     "test": {b: 100 * acc(s, b, "test") for b in BENCHES}, "macro_test": macro(s, "test"),
                                     "macro_dev": macro(s, "dev"), "abstain": abstain_rate(s), "truncated": truncated_rate(s),
                                     "tokens": mean_tokens(s), "speed": sp}
    summary["systems"] = rows_summary
    L += ["", "Une quantification qui casse le modèle se voit ici avant tout test : une abstention ou un taux de "
          "troncature très supérieurs à ceux du Q8_0 du même modèle indiquent un gabarit de conversation ou un fichier "
          "défectueux, pas une perte de précision (règle de garde du protocole).", ""]

    # ---- choices on dev ----
    sb, bb = in_budget(small), in_budget(big)
    if not sb or not bb:
        L += ["## 2. Choix sur DEV", "", "Pas de petit pair ou pas de gros pair dans le budget : comparaison impossible.", ""]
        return
    best_small, best_big = best_by_dev(sb), best_by_dev(bb)
    per_model = {m: best_by_dev(in_budget([s for s in big if s.model == m])) for m in dict.fromkeys(s.model for s in big)
                 if in_budget([s for s in big if s.model == m])}
    tiers = {}
    for t in Q.TIERS_GIB:
        tb, ts = in_budget(big, t), in_budget(small, t)
        if tb and ts:
            tiers[t] = (best_by_dev(ts), best_by_dev(tb))
    L += ["## 2. Choix faits sur DEV", "",
          f"- meilleur petit pair du budget : **{best_small.label}** (macro DEV {macro(best_small, 'dev'):.1f})",
          f"- meilleur gros pair du budget : **{best_big.label}** (macro DEV {macro(best_big, 'dev'):.1f}, "
          f"{best_big.gib:.2f} Gio)"] + \
         [f"- meilleure quantification de {m} : {s.tag} (macro DEV {macro(s, 'dev'):.1f}, {s.gib:.2f} Gio)"
          for m, s in per_model.items()] + [""]
    summary["choices"] = {"best_small": best_small.label, "best_big": best_big.label,
                          "per_model": {m: s.label for m, s in per_model.items()},
                          "tiers": {str(t): [a.label, b.label] for t, (a, b) in tiers.items()}}

    # ---- single peer comparisons ----
    comps = {}
    L += ["## 3. Un pair seul : gros quantifié contre petit en Q8_0 (TEST)", "",
          "Différence appariée A moins B en points [IC 95 % Tango] ; verdict (marge ±" + f"{DELTA:g}" + ").", "",
          "| A | B | mémoire A / B (Gio) | questions mises bout à bout | " + " | ".join(BENCHES) + " |",
          "| --- | --- | --- | --- | " + " | ".join("---" for _ in BENCHES) + " |"]
    pairs = [("meilleur gros du budget", best_big, "meilleur petit du budget", best_small)]
    pairs += [("quantification choisie sur DEV pour ce modèle", s, "meilleur petit du budget", best_small) for m, s in per_model.items()
              if s is not best_big]
    pairs += [(f"meilleur gros <= {t:g} Gio", b, f"meilleur petit <= {t:g} Gio", a) for t, (a, b) in tiers.items()
              if t != Q.BUDGET_GIB]
    for na, a, nb, b in pairs:
        g = versus(alone_vectors(a), alone_vectors(b))
        comps[f"{a.label}|{b.label}"] = g
        L.append(f"| {na} : {a.label} | {nb} : {b.label} | {a.gib:.2f} / {b.gib:.2f} | {cell(g['pooled'])} | "
                 + " | ".join(cell(g[bn]) for bn in BENCHES) + " |")
    L += ["", "### Coût de la quantification : chaque fichier quantifié contre le Q8_0 du même modèle", "",
          "| quantifié | Q8_0 | Gio | bits/poids | macro TEST quantifié | macro TEST Q8_0 | différence appariée, bout à bout |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    cost = {}
    for s in sorted(big, key=lambda s: (s.model, s.gib)):
        ref = next((r for r in refs if r.model == s.model), None)
        if ref is None:
            continue
        g = compare(pooled(alone_vectors(s)), pooled(alone_vectors(ref)), DELTA)
        cost[s.label] = g
        L.append(f"| {s.label} | {ref.label} | {s.gib:.2f} (Q8_0 {ref.gib:.2f}) | {s.bpw:.2f} | {macro(s, 'test'):.1f} | "
                 f"{macro(ref, 'test'):.1f} | {cell(g)} |")
    L.append("")
    summary["single_comparisons"], summary["quantization_cost"] = comps, cost

    # ---- swarms ----
    mixes = {
        "essaim de 7 familles Q8_0 (E4)": small,
        f"essaim de {len(sb)} petits du budget": sb,
        f"les {min(3, len(sb))} meilleurs petits du budget (DEV)": sorted(sb, key=lambda s: (-macro(s, 'dev'), s.label))[:3],
        f"essaim de {len(per_model)} gros (une quantification par modèle, choisie sur DEV)": list(per_model.values()),
    }
    by_tag = {}
    for s in big:
        by_tag.setdefault(s.tag, []).append(s)
    for tag, g in by_tag.items():
        if len(g) >= 2 and all(x in bb for x in g):
            mixes[f"essaim de {len(g)} gros, tous en {tag}"] = g
    mixes[f"essaim mixte : {len(sb)} petits + {len(per_model)} gros (budget)"] = sb + list(per_model.values())
    vecs = {name: swarm_vectors(peers) for name, peers in mixes.items() if len(peers) >= 2}
    L += ["## 4. Essaims (vote pondéré, poids appris sur DEV pour chaque essaim, TEST)", "",
          "| essaim | pairs | mémoire totale (Gio) | plus gros fichier (Gio) | macro TEST | " + " | ".join(BENCHES) + " |",
          "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in BENCHES) + " |"]
    for name, v in vecs.items():
        peers = mixes[name]
        L.append(f"| {name} | {len(peers)} | {sum(p.gib for p in peers):.1f} | {max(p.gib for p in peers):.2f} | "
                 f"**{macro_of(v):.1f}** | " + " | ".join(f"{100 * st.mean(v[b]):.1f}" for b in BENCHES) + " |")
    summary["swarms"] = {n: {"members": [p.label for p in mixes[n]], "macro_test": macro_of(v),
                             "per_bench": {b: 100 * st.mean(v[b]) for b in BENCHES}} for n, v in vecs.items()}
    big_name = next((n for n in vecs if n.startswith(f"essaim de {len(per_model)} gros (une")), None)
    small_name = next((n for n in vecs if n.startswith(f"essaim de {len(sb)} petits")), None)
    top3_name = next((n for n in vecs if n.startswith("les ")), None)
    swarm_cmp = {}
    if big_name:
        L += ["", "Différences appariées entre essaims (TEST, A moins B).", "",
              "| A | B | questions mises bout à bout | " + " | ".join(BENCHES) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in BENCHES) + " |"]
        targets = [(big_name, n) for n in (small_name, top3_name, "essaim de 7 familles Q8_0 (E4)") if n in vecs]
        mixed = next((n for n in vecs if n.startswith("essaim mixte")), None)
        if mixed and small_name:
            targets.append((mixed, small_name))
        targets.append((big_name, ("seul", best_small)))
        for na, nb in targets:
            vb = alone_vectors(nb[1]) if isinstance(nb, tuple) else vecs[nb]
            lab_b = f"{nb[1].label} seul" if isinstance(nb, tuple) else nb
            g = versus(vecs[na], vb)
            swarm_cmp[f"{na}|{lab_b}"] = g
            L.append(f"| {na} | {lab_b} | {cell(g['pooled'])} | " + " | ".join(cell(g[bn]) for bn in BENCHES) + " |")
    summary["swarm_comparisons"] = swarm_cmp

    # ---- scaling ----
    sc = {}
    L += ["", "## 5. Passage à l'échelle : moyenne sur tous les sous-ensembles de k pairs (macro TEST)", "",
          "Les pairs d'un essaim sont de familles différentes ; k = 1 est la moyenne des pairs seuls.", "",
          "| k | petits du budget (tous / k meilleurs DEV) | gros (tous / k meilleurs DEV) |", "| --- | --- | --- |"]
    sc["small"] = scaling(sb) if len(sb) >= 2 else {}
    sc["big"] = scaling(list(per_model.values())) if len(per_model) >= 2 else {}
    for k in range(1, max(len(sc["small"]), len(sc["big"])) + 1):
        f = lambda d: f"{d[k]['mean_all_subsets']:.1f} / {d[k]['best_k_by_dev']:.1f}" if k in d else "—"
        L.append(f"| {k} | {f(sc['small'])} | {f(sc['big'])} |")
    L.append("")
    summary["scaling"] = sc


# ---------- code (E11) ----------

def check_code_provenance(AC, groups, loaded, quants):
    """Before the candidates of several groups are merged into one vote: every model generated with the same protocol
    and data (sample count, seeds, prompts, ...), dev and test of one model from the same file and engine, and each
    quantized file is the pinned one in every benchmark and split (analyze_code.load checks one group at a time)."""
    from essaim import code
    for b in code.BENCHES:
        for key in groups:
            AC.check_splits(b, loaded[(key, b, "dev")][3], loaded[(key, b, "test")][3])
        for sp in SPLITS:
            protos = {json.dumps({k: man.get(k) for k in AC.PROTOCOL + ("data",)}, sort_keys=True)
                      for key in groups for man in loaded[(key, b, sp)][3]["manifests"].values()}
            if len(protos) != 1:
                raise SystemExit(f"code {b} {sp} : protocole de génération ou données différents entre quantifications")
    for q in quants:
        key = (q.suffix, q.exec_tag)
        for b in code.BENCHES:
            for sp in SPLITS:
                w = loaded[(key, b, sp)][3]["manifests"][q.model].get("weights") or {}
                if w.get("sha256") != q.sha256 or w.get("size") != q.size:
                    raise SystemExit(f"code {Q.label(q.model, q.tag)} {b} {sp} : pas le fichier épinglé dans e16_quant.QUANT")


def build_code(L: list[str], summary: dict, speed: dict[str, dict]):
    """E11 code on the quantized files that have generations and execution results (e16-1: the bounds), against the
    Q8_0 small peers of E11. The selection rule is the one E11 chose on its own DEV (cluster, count, all samples)."""
    import analyze_code as AC
    from essaim import code, data

    rule_by_bench = {}
    p11 = RESULTS / "e11_summary_colab.json"
    if p11.exists():
        rule_by_bench = {b: v["primary"] for b, v in json.loads(p11.read_text(encoding="utf-8"))["benches"].items()}
    quants = [q for q in Q.QUANT
              if all(read_manifest(code.gen_path(q.model, q.suffix, b, sp)) is not None for b in code.BENCHES for sp in SPLITS)]
    L += ["## 6. Code (E11) : programme choisi par exécution", ""]
    if not quants:
        L += ["Pas de génération de code E16 (plan e16-1 pas encore lancé ou récupéré).", ""]
        return
    small_models = {Q.label(m, "Q8_0"): m for m in FAMILIES.values()}
    # One group per (suffix, execution tag): E11's small peers, then one group per quantized file (exec_e16.py).
    groups: dict[tuple[str, str], dict[str, str]] = {(SMALL_SUFFIX, "e11"): {m: lab for lab, m in small_models.items()}}
    for q in quants:
        groups[(q.suffix, q.exec_tag)] = {q.model: Q.label(q.model, q.tag)}
    sizes = {Q.label(q.model, q.tag): q.size for q in quants}
    loaded = {}
    for b in code.BENCHES:
        for sp in SPLITS:
            for key, mm in groups.items():
                r = AC.load(b, sp, key[0], key[1], mm)
                if r is None:
                    raise SystemExit(f"code {b} {sp} {key[0]} : générations absentes")
                loaded[(key, b, sp)] = r
    check_code_provenance(AC, groups, loaded, quants)
    got: dict[tuple[str, str], tuple] = {}
    for b in code.BENCHES:
        for sp in SPLITS:
            merged = None
            for key in groups:
                r = loaded[(key, b, sp)]
                if merged is None:
                    merged = [r[0], {q: list(c) for q, c in r[1].items()}, dict(r[2]), [r[3]]]
                else:
                    if r[0] != merged[0]:
                        raise SystemExit(f"code {b} {sp} : problèmes exclus ou ordre différents entre quantifications")
                    for q, c in r[1].items():
                        merged[1][q] += c
                    merged[2].update(r[2])
                    merged[3].append(r[3])
            got[(b, sp)] = tuple(merged)
    labels = [lab for mm in groups.values() for lab in mm.values()]
    order = {lab: -i for i, lab in enumerate(labels)}

    def greedy_acc(sp, lab, b):
        ids, cands, F, _ = got[(b, sp)]
        return st.mean(float(F[c].correct) for q in ids for c in cands[q] if c.fam == lab and c.sample == 0)

    def dev_code(lab):
        return 100 * st.mean(greedy_acc("dev", lab, b) for b in code.BENCHES)

    in_b = lambda lab: (sizes.get(lab) or small_size(lab)) / Q.GIB <= Q.BUDGET_GIB
    small_size = lambda lab: read_manifest(code.gen_path(small_models[lab], SMALL_SUFFIX, code.BENCHES[0], "dev"))["weights"]["size"]
    small_labels = [lab for lab in small_models if in_b(lab)]
    big_by_model = {}
    for q in quants:
        lab = Q.label(q.model, q.tag)
        if q.fits():
            big_by_model.setdefault(q.model, []).append(lab)
    if not big_by_model or not small_labels:
        L += ["Pas de gros pair du budget avec du code, ou pas de petit pair : comparaison impossible.", ""]
        return
    big_pick = {m: max(labs, key=lambda l: (round(dev_code(l), 9), -sizes[l], l)) for m, labs in big_by_model.items()}
    best_small = max(small_labels, key=lambda l: (round(dev_code(l), 9), -small_size(l), l))
    best_big = max(big_pick.values(), key=lambda l: (round(dev_code(l), 9), -sizes[l], l))

    def select(b, sp, subset, w, rule):
        ids, cands, F, _ = got[(b, sp)]
        out = []
        for q in ids:
            cs = [c for c in cands[q] if c.fam in subset]
            if rule.startswith("visible"):
                sel = AC.visible_weight(cs, F, w, order)
            else:
                sel = AC.functional(cs, F, w, order, rule.split("-")[1].split("|")[0])[0]
            out.append(float(sel is not None and F[sel].correct))
        return out

    def swarm(subset: list[str]) -> dict[str, list[float]]:
        out = {}
        for b in code.BENCHES:
            ids, cands, F, _ = got[(b, "dev")]
            p = {lab: greedy_acc("dev", lab, b) for lab in subset}
            rates = AC.pair_rates([[(c, F[c]) for c in cands[q] if c.sample == 0 and c.fam in subset] for q in ids])
            w = AC.weights(p, rates["c"])
            out[b] = select(b, "test", set(subset), w, rule_by_bench.get(b, "cluster-count|all"))
        return out

    def alone(lab):
        return {b: [float(got[(b, "test")][2][c].correct) for q in got[(b, "test")][0]
                    for c in got[(b, "test")][1][q] if c.fam == lab and c.sample == 0] for b in code.BENCHES}

    pool_ = lambda v: [x for b in code.BENCHES for x in v[b]]
    L += ["Règle du vote gelée d'après E11 (choisie sur le DEV d'E11) : " +
          ", ".join(f"{b} {rule_by_bench.get(b, 'cluster-count|all')}" for b in code.BENCHES) +
          ". Candidats : programme glouton et tirages de chaque pair. Les quantifications avec code sont celles du plan "
          "demandé (e16-1 : bornes basse et haute de chaque modèle).", "",
          "| système | Gio | " + " | ".join(f"{b} DEV | {b} TEST" for b in code.BENCHES) + " | pass@1 glouton TEST, bout à bout |",
          "| --- | --- | " + " | ".join("--- | ---" for _ in code.BENCHES) + " | --- |"]
    summ = {"systems": {}}
    for lab in labels:
        if not (lab in sizes or lab in small_labels):
            continue
        g = alone(lab)
        sz = (sizes.get(lab) or small_size(lab)) / Q.GIB
        L.append(f"| {lab} | {sz:.2f} | " + " | ".join(f"{100 * greedy_acc('dev', lab, b):.1f} | {100 * st.mean(g[b]):.1f}"
                                                    for b in code.BENCHES) + f" | **{100 * st.mean(pool_(g)):.1f}** |")
        summ["systems"][lab] = {"gib": sz, "test_pooled": 100 * st.mean(pool_(g))}
    L += ["", f"Choix sur DEV : meilleur petit **{best_small}**, meilleur gros **{best_big}**, quantification retenue par "
          "modèle : " + ", ".join(f"{m} -> {l.rpartition('@')[2]}" for m, l in big_pick.items()) + ".", ""]
    cmp_rows = [("meilleur gros seul", alone(best_big), "meilleur petit seul", alone(best_small))]
    sets = {f"essaim de {len(small_labels)} petits du budget": small_labels,
            f"essaim de {len(big_pick)} gros (choisis sur DEV)": list(big_pick.values())}
    sw = {n: swarm(s) for n, s in sets.items() if len(s) >= 2}
    L += ["| essaim | pass@1 TEST, bout à bout | " + " | ".join(code.BENCHES) + " |", "| --- | --- | " +
          " | ".join("---" for _ in code.BENCHES) + " |"]
    for n, v in sw.items():
        L.append(f"| {n} | **{100 * st.mean(pool_(v)):.1f}** | " + " | ".join(f"{100 * st.mean(v[b]):.1f}" for b in code.BENCHES) + " |")
    names = list(sw)
    if len(names) == 2:
        cmp_rows.append((names[1], sw[names[1]], names[0], sw[names[0]]))
        cmp_rows.append((names[1], sw[names[1]], f"{best_small} seul", alone(best_small)))
    L += ["", "| A | B | différence appariée, bout à bout | " + " | ".join(code.BENCHES) + " |",
          "| --- | --- | --- | " + " | ".join("---" for _ in code.BENCHES) + " |"]
    summ["comparisons"] = {}
    for na, a, nb, b in cmp_rows:
        g = {"pooled": compare(pool_(a), pool_(b), DELTA), **{bn: compare(a[bn], b[bn], DELTA) for bn in code.BENCHES}}
        summ["comparisons"][f"{na}|{nb}"] = g
        L.append(f"| {na} | {nb} | {cell(g['pooled'])} | " + " | ".join(cell(g[bn]) for bn in code.BENCHES) + " |")
    L += ["", "Même bac à sable que E11 : les réglages d'exécution (tests, extraction, entrées supplémentaires, "
          "versions de Python et NumPy) sont comparés entre les groupes ; un écart arrête l'analyse.", ""]
    ex = [m["exec"] for (b, sp), t in got.items() for m in t[3]]
    keys = AC.EXEC_SETTINGS
    base = {k: ex[0].get(k) for k in keys if k != "isolation"}
    for e in ex:
        diff = {k: (base[k], e.get(k)) for k in base if e.get(k) != base[k]}
        if diff:
            raise SystemExit(f"réglages d'exécution différents entre groupes : {diff}")
    summary["code"] = summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-code", action="store_true", help="E4 benchmarks only")
    a = ap.parse_args()
    systems: list[Sys] = []
    absent = []
    for m in list(FAMILIES.values()) + EXTRA + REFS:
        s = load_system(m, "Q8_0", SMALL_SUFFIX)
        (systems.append(s) if s else absent.append(Q.label(m, "Q8_0")))
    for q in Q.QUANT:
        s = load_system(q.model, q.tag, q.suffix)
        (systems.append(s) if s else absent.append(Q.label(q.model, q.tag)))
    L = ["# E16 : un plus gros modèle quantifié par pair, contre un petit modèle en Q8_0", "",
         "Protocole préenregistré dans docs/13_e16_quantification.md. Mémoire égale : un pair tient dans "
         f"{Q.BUDGET_GIB} Gio de poids. Mêmes questions, mêmes invites, mêmes réglages que E4 (et E11 pour le code) ; seul "
         "le fichier de poids change. Choix sur DEV, rapport sur TEST ; tests exacts d'E4 et d'E11 (essaim/stats.py), "
         f"marge d'équivalence ±{DELTA:g} point(s). Les comparaisons sont exploratoires, sans correction de multiplicité ; "
         "seule la comparaison « meilleur gros du budget contre meilleur petit du budget » est la comparaison principale.", ""]
    summary: dict = {"budget_gib": Q.BUDGET_GIB, "delta": DELTA, "absent": absent}
    if absent:
        L += [f"Systèmes absents ou non lancés ({len(absent)}) : " + ", ".join(absent), ""]
    have_small = [s for s in systems if s.model in FAMILIES.values() and s.tag == "Q8_0"]
    if len(have_small) < len(FAMILIES) or not any(s.quantized for s in systems):
        L += ["Données insuffisantes : il faut les 7 petits pairs d'E4 et au moins un fichier quantifié.", ""]
    else:
        check_protocol(systems)
        eng = engines(systems)
        summary["engines"] = eng
        if len(eng) > 1:
            L += ["**Avertissement : versions de llama.cpp différentes** : " + " ; ".join(f"{k} -> {len(v)} systèmes"
                                                                                     for k, v in eng.items()), ""]
        speed = read_speed()
        build_mc(systems, L, summary, speed)
        if not a.no_code:
            build_code(L, summary, speed)
    (RESULTS / "e16_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / "e16_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
