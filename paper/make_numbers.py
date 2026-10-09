"""Write paper/numbers.tex: every number quoted in the paper, as LaTeX macros, read from phase0/results.

    uv run --project ../phase0 python make_numbers.py

No number in main.tex is typed by hand: each one is a macro defined here from a results file, so the
paper always matches the data. Missing results produce a visible "??" instead of a stale value, except for
E4 (the main result) and E10: missing, preview or incomplete sources, or undefined macros used by
main.tex, stop the script with an error.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "phase0" / "results"
OUT = HERE / "numbers.tex"
sys.path.insert(0, str(HERE.parent / "phase0"))
from essaim import models  # noqa: E402  (model sizes: one convention, total parameters)

macros: dict[str, str] = {}


def load(name: str) -> dict | None:
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def put(name: str, value, fmt: str = "{:.1f}"):
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(name)
    macros[name] = "??" if value is None else (fmt.format(value) if isinstance(value, (int, float)) else str(value))


def signed(x: float) -> str:
    return f"{x:+.1f}".replace("-", "$-$")


def ci(lo: float, hi: float) -> str:
    return f"[{signed(lo)}, {signed(hi)}]"


def parse_gain(s: str) -> tuple[float, float, float] | None:
    m = re.match(r"([+-][\d.]+) \[([+-][\d.]+) ; ([+-][\d.]+)\]", s or "")
    return tuple(float(x) for x in m.groups()) if m else None


# --- E1: multiple choice, calibrated fusion of 4 families (phase 0) --------------------------------
mc = load("mc_summary_test.json")
for bench, tag in (("arc", "Arc"), ("mmlupro", "Mmlu")):
    b = (mc or {}).get("benches", {}).get(bench)
    if not b:
        continue
    sysd = b["systems"]
    mean_pass = lambda k: 100 * sum(sysd[k]["pass"]) / len(sysd[k]["pass"])
    best = b["best_expert"]["pass"]
    put(f"eOne{tag}Fusion", mean_pass("fusion:mélange calibré"))
    put(f"eOne{tag}Best", mean_pass(f"seul:{best}"))
    ref = next(k for k in sysd if k.startswith("ref:"))
    put(f"eOne{tag}Ref", mean_pass(ref))
    g = b["gain_vs_best"]["mélange calibré"]["pass"]
    put(f"eOne{tag}Gain", signed(g["points"]))
    put(f"eOne{tag}GainCI", ci(*g["ci95"]))
think = load("think_summary_test.json")
for bench, tag in (("arc", "Arc"), ("mmlupro", "Mmlu")):
    b = (think or {}).get("benches", {}).get(bench)
    if not b:
        continue
    sysd = b["systems"]
    mean_pass = lambda k: 100 * sum(sysd[k]["pass"]) / len(sysd[k]["pass"])
    put(f"eOneThink{tag}Fusion", mean_pass("fusion:mélange calibré"))
    put(f"eOneThink{tag}Best", mean_pass(f"seul:{b['best_expert']['pass']}"))
    put(f"eOneThink{tag}Ref", mean_pass(next(k for k in sysd if k.startswith("ref:"))))

# --- E2: writing together vs voting (GSM8K, 4 families) --------------------------------------------
gen = load("gen_summary_essaim4_colab_test.json")
if gen:
    acc = gen["accuracy"]
    put("eTwoN", gen["n"], "{}")
    put("eTwoBest", acc[gen["best_peer"]])
    put("eTwoVote", acc["vote"])
    put("eTwoAccord", acc["accord"])
    put("eTwoCroise", acc["croise"])
    put("eTwoRef", acc[gen["reference"]] if gen.get("reference") else None)
    put("eTwoCeiling", gen["ceiling"])
    for k, tag in (("vote", "Vote"), ("accord", "Accord"), ("croise", "Croise")):
        g = gen["gain_vs_best"][k]
        put(f"eTwo{tag}Gain", signed(g[0]))
        put(f"eTwo{tag}GainCI", ci(g[1], g[2]))
        if gen.get("gain_vs_reference"):
            r = gen["gain_vs_reference"][k]
            put(f"eTwo{tag}RefGain", signed(r[0]))
            put(f"eTwo{tag}RefGainCI", ci(r[1], r[2]))
        c = gen["cost"][k if k != "vote" else "vote"]
        put(f"eTwo{tag}Trips", c["trips_mean"])
        put(f"eTwo{tag}Wan", c["wan_s_mean"]["100"])

# --- Model sizes (essaim/models.py: total parameters, embeddings included) --------------------------


def size_range(ms) -> str:
    sizes = [models.total_b(m) for m in ms]
    lo, hi = min(sizes), max(sizes)
    return models.fmt_b(lo) if lo == hi else f"{models.fmt_b(lo)}--{models.fmt_b(hi)}"


if mc:
    put("eOneSwarmSize", size_range(e.split("@")[0] for e in mc["experts"]))
if gen:
    put("eTwoSwarmSize", size_range(gen["peers"]))
    if gen.get("reference"):
        put("eTwoRefSize", size_range([gen["reference"].split(":", 1)[-1].strip()]))
from analyze_e4 import FAMILIES, REFS  # noqa: E402  (E4: the swarm and the references)

put("eFourSwarmSize", size_range(FAMILIES.values()))
put("eFourRefSizes", size_range(REFS))
for m, tag in (("google/gemma-4-E2B-it", "Two"), ("google/gemma-4-E4B-it", "Four")):
    put(f"eGemma{tag}Effective", models.fmt_b(models.effective_b(m)))
    put(f"eGemma{tag}Total", models.fmt_b(models.total_b(m)))
small = [e.split("@")[0] for e in (mc or {}).get("experts", [])] + list((gen or {}).get("peers", [])) + list(FAMILIES.values())
put("eSmallSizes", size_range(small))

# --- E2 dispersion: who wins when the peers split (re-graded texts) --------------------------------
gen_path = RES / "gen_essaim4_colab_test.jsonl"
if gen_path.exists():
    import itertools
    from essaim.answers import regrade_gen  # noqa: E402
    from essaim.results import read_rows  # noqa: E402
    from essaim.results import read_manifest  # noqa: E402
    rows = regrade_gen(read_rows(gen_path, key=("mode", "id")), read_manifest(gen_path).get("max_rounds"), read_manifest(gen_path).get("k"))
    solo = {r["id"]: r for r in rows if r["mode"] == "solo"}
    vote = {r["id"]: r for r in rows if r["mode"] == "vote"}
    split = {k: [0, 0] for k in range(5)}
    for i, r in solo.items():
        k = sum(p["answer"] == r["gold"] for p in r["per_peer"])
        split[k][0] += 1
        split[k][1] += vote[i]["answer"] == r["gold"]
    for k, tag in ((1, "One"), (2, "Two")):
        put(f"eTwoSplit{tag}N", split[k][0], "{}")
        put(f"eTwoSplit{tag}Won", split[k][1], "{}")
    p = [sum(r["per_peer"][j]["answer"] == r["gold"] for r in solo.values()) / len(solo) for j in range(4)]
    acc = 0.0
    for x in itertools.product([0, 1], repeat=4):  # independent binary majority, ties split evenly
        pr = 1.0
        for xi, pi in zip(x, p):
            pr *= pi if xi else 1 - pi
        acc += pr * (1.0 if sum(x) > 2 else 0.5 if sum(x) == 2 else 0.0)
    put("eTwoBinaryPred", 100 * acc)

# --- E3: K-class weights and the stop certificate (replay) -----------------------------------------
e3 = load("e3_summary_essaim4_colab.json")
if e3:
    put("eThreeCollision", e3["collision"], "{:.3f}")
    for rule, tag in (("logodds", "Binary"), ("logodds_K", "Kclass"), ("vote", "Vote")):
        r = e3["rules"][rule]
        put(f"eThree{tag}Acc", r["accuracy"])
        g = parse_gain(r["gain_vs_best"])
        if g:
            put(f"eThree{tag}Gain", signed(g[0]))
            put(f"eThree{tag}GainCI", ci(g[1], g[2]))
        c = e3["certificate"].get(rule)
        if c:
            put(f"eThree{tag}Waited", c["waited"], "{:.2f}")
            put(f"eThree{tag}Same", f"{c['same']}/{c['n']}")
    put("eThreeBestAcc", e3["rules"]["best"]["accuracy"])

# --- E7: scalability of the protocol and the tracker (app/bench, report generated from its JSON) ----
rep7 = HERE.parent / "app" / "bench" / "results" / "e7_report.md"
if rep7.exists():
    text = rep7.read_text(encoding="utf-8")

    def table_after(heading: str) -> list[list[str]]:
        i = text.find(heading)
        if i < 0:
            return []
        rows = []
        for line in text[i:].splitlines()[1:]:
            if line.startswith("## ") or line.startswith("### "):
                break
            if line.startswith("|") and not line.startswith("| ---"):
                rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        return rows[1:]  # without the header row

    def cell(v: str):
        """A report cell: a number, a lower bound "≥ x" (kept as such), or unavailable ("–")."""
        v = v.strip()
        if v in ("–", "-", ""):
            return None
        if v.startswith("≥"):
            return r"$\geq$" + f"{float(v.lstrip('≥ ')):.1f}"
        return float(v)

    for r in table_after("### V1. Débit soutenu"):
        n, rtt, before, after = r[0], r[1], r[2], r[5]
        words = {"64": "SixtyFour", "256": "TwoFiftySix", "1024": "OneK", "4096": "FourK"}
        if rtt == "100" and n in words:
            put(f"eSevenRps{words[n]}", cell(after))
            put(f"eSevenPfifty{words[n]}", cell(r[6]))
            put(f"eSevenPninetyfive{words[n]}", cell(r[7]))
            put(f"eSevenRpsBefore{words[n]}", cell(before))
    gains = [r for r in table_after("Gain du certificat") if r and r[0].startswith("lat_rtt")]
    for r in gains:
        if r[0] == "lat_rtt100_k4":
            put("eSevenCertWaitAll", float(r[1]), "{:.2f}")
            put("eSevenCertCert", float(r[2]), "{:.2f}")
            put("eSevenCertGain", float(r[3].rstrip(" %")), "{:.0f}")
        if r[0] == "lat_rtt100_k4_hetero":
            put("eSevenCertGainHetero", float(r[3].rstrip(" %")), "{:.0f}")

# --- E5: minority appeal ---------------------------------------------------------------------------
for bench, tag in (("arc", "Arc"), ("mmlupro", "Mmlu")):
    a5 = load(f"appeal_summary_{bench}.json")
    if not a5:
        continue
    t = a5["test"]
    put(f"eFive{tag}Appeals", t["appeals"], "{}")
    put(f"eFive{tag}Flips", t["flips"], "{}")
    put(f"eFive{tag}B", 100 * t["B"])
    put(f"eFive{tag}C", 100 * t["C"])
    put(f"eFive{tag}R", 100 * t["r"], "{:.0f}")
    put(f"eFive{tag}H", 100 * t["h"], "{:.0f}")
    put(f"eFive{tag}Gain", signed(t["gain_points"]))
    put(f"eFive{tag}GainCI", ci(*a5["test_ci95"]))

# --- E4: scaling to 7 families against much bigger models ------------------------------------------
# E4 is the main result of the paper: unlike the other blocks, a missing or incomplete source is an error
# (never a "??" in the compiled paper, never a stale number).
from analyze_e4 import EXTRA  # noqa: E402
from essaim.results import read_manifest  # noqa: E402

E4_BENCHES = {"gsm8k": "Gsm", "math500": "Math", "arc": "Arc", "mmlupro": "Mmlu"}
E4_MODELS = {"Qwen/Qwen3.5-4B": "QwenFour", "google/gemma-4-E4B-it": "GemmaEFour",
             "ibm-granite/granite-4.2-3b": "GraniteThree", "HuggingFaceTB/SmolLM3-3B": "SmolThree",
             "mistralai/Ministral-3-3B-Instruct-2512": "MinistralThree", "microsoft/Phi-4-mini-instruct": "PhiMini",
             "allenai/OLMo-2-0425-1B-Instruct": "OlmoOne", "Qwen/Qwen3.5-2B": "QwenTwo",
             "google/gemma-4-E2B-it": "GemmaETwo", "Qwen/Qwen3.5-9B": "QwenNine",
             "google/gemma-4-12B-it": "GemmaTwelve", "mistralai/Ministral-3-14B-Instruct-2512": "MinistralFourteen",
             "Qwen/Qwen3.8-27B": "QwenTwentySeven"}
DISPLAY = {"Qwen/Qwen3.5-4B": "Qwen3.5-4B", "google/gemma-4-E4B-it": "Gemma-4-E4B",
           "ibm-granite/granite-4.2-3b": "Granite-4.2-3B", "HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B",
           "mistralai/Ministral-3-3B-Instruct-2512": "Ministral-3-3B", "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
           "allenai/OLMo-2-0425-1B-Instruct": "OLMo-2-1B"}  # names used in the paper's tables
if set(E4_MODELS) != set(FAMILIES.values()) | set(EXTRA) | set(REFS) or set(DISPLAY) != set(FAMILIES.values()):
    raise SystemExit("E4 : la liste des modèles de make_numbers.py ne suit plus analyze_e4.py")
E4_REFS = {m: E4_MODELS[m] for m in REFS}
VERDICTS = {"supérieur": "superior", "inférieur": "inferior", "indéterminé": "inconclusive",
            "équivalent (±2)": r"equivalent ($\pm$2)", "non inférieur (-2)": r"non-inferior ($-$2)"}
CASCADE_RULES = {"support<2": "SupTwo", "margin@10%": "MarTen", "margin@20%": "MarTwenty"}


def need(name: str) -> dict:
    """A results file the E4 section cannot do without: absent, preview or incomplete is fatal."""
    d = load(name)
    if d is None:
        raise SystemExit(f"E4 : {name} absent (lancer phase0/analyze_e4.py et analyze_cascade_e4.py)")
    if d.get("preview"):
        raise SystemExit(f"E4 : {name} est un aperçu (--partial), pas un résultat final")
    if set(d.get("benches", {})) != set(E4_BENCHES):
        raise SystemExit(f"E4 : {name} incomplet ({sorted(d.get('benches', {}))})")
    return d


def verdict_en(g: dict) -> str:
    if g["verdict"] not in VERDICTS:
        raise SystemExit(f"E4 : verdict inconnu {g['verdict']!r}")
    return VERDICTS[g["verdict"]]


SHORT = {"superior": r"\textsc{sup}", "inferior": r"\textsc{inf}", "inconclusive": "--",
         r"equivalent ($\pm$2)": r"\textsc{eq}", r"non-inferior ($-$2)": r"\textsc{ni}"}


def put_cmp(prefix: str, g: dict):
    put(f"{prefix}Diff", signed(g["mean"]))
    put(f"{prefix}CI", ci(g["lo95"], g["hi95"]))
    put(f"{prefix}Verdict", verdict_en(g))
    put(f"{prefix}V", SHORT[verdict_en(g)])
    put(f"{prefix}PSup", g["p_superior"], "{:.3f}")


def put_e4(e: dict, pre: str):
    """Per-benchmark and macro-averaged macros of an analyze_e4.py summary, under the prefix `pre`."""
    if set(e["families"].values()) != set(FAMILIES.values()) or list(e["refs"]) != list(REFS) or e["delta"] != 2.0:
        raise SystemExit("E4 : le résumé ne correspond pas à l'essaim et aux références d'analyze_e4.py")
    for bench, tag in E4_BENCHES.items():
        b = e["benches"][bench]
        put(f"{pre}{tag}N", b["n"], "{}")
        put(f"{pre}{tag}Swarm", b["swarm"]["wvote"])
        put(f"{pre}{tag}Vote", b["swarm"]["vote"])
        put(f"{pre}{tag}Oracle", b["oracle"])
        put(f"{pre}{tag}Waits", b["certificate_waits"], "{:.2f}")
        put(f"{pre}{tag}Collision", b["collision"], "{:.2f}")
        for m, mtag in E4_MODELS.items():
            put(f"{pre}{tag}{mtag}", 100 * b["alone"][m])
        for ref, rtag in E4_REFS.items():
            put_cmp(f"{pre}{tag}{rtag}", b["comparisons"][f"wvote|{ref}"])
        best = b["best_dev"]
        put(f"{pre}{tag}BestName", DISPLAY[best])
        put_cmp(f"{pre}{tag}Best", b["comparisons"][f"wvote|meilleur pair de dev ({best})"])
        for k in range(1, len(FAMILIES) + 1):
            put(f"{pre}{tag}Scale{['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven'][k - 1]}",
                b["scaling"][str(k)]["mean_all_subsets"])
    mean = lambda f: sum(f(e["benches"][b]) for b in E4_BENCHES) / len(E4_BENCHES)
    put(f"{pre}MeanSwarm", mean(lambda b: b["swarm"]["wvote"]))
    put(f"{pre}MeanVote", mean(lambda b: b["swarm"]["vote"]))
    put(f"{pre}MeanOracle", mean(lambda b: b["oracle"]))
    for m, mtag in E4_MODELS.items():
        put(f"{pre}Mean{mtag}", mean(lambda b: 100 * b["alone"][m]))
    for k, word in enumerate(["One", "Two", "Three", "Four", "Five", "Six", "Seven"], 1):
        put(f"{pre}MeanScale{word}", mean(lambda b: b["scaling"][str(k)]["mean_all_subsets"]))
        put(f"{pre}MeanBest{word}", mean(lambda b: b["scaling"][str(k)]["best_k_by_dev"]))
    waits = [e["benches"][b]["certificate_waits"] for b in E4_BENCHES]
    put(f"{pre}WaitsMin", min(waits), "{:.2f}")
    put(f"{pre}WaitsMax", max(waits), "{:.2f}")
    put(f"{pre}AvoidedMin", len(FAMILIES) - max(waits), "{:.2f}")
    put(f"{pre}AvoidedMax", len(FAMILIES) - min(waits), "{:.2f}")
    # Comparisons: computed (weighted and plurality votes) and displayed (weighted vote, Table e4tests).
    shown = [b["comparisons"][k] for b in e["benches"].values() for k in b["comparisons"] if k.startswith("wvote|")]
    put(f"{pre}NComputed", sum(len(b["comparisons"]) for b in e["benches"].values()), "{}")
    put(f"{pre}NShown", len(shown), "{}")
    put(f"{pre}NInconclusive", sum(g["verdict"] == "indéterminé" for g in shown), "{}")
    put(f"{pre}NSuperior", sum(g["verdict"] == "supérieur" for g in shown), "{}")
    put(f"{pre}NNonInferior", sum(g["verdict"].startswith("non inférieur") for g in shown), "{}")
    put(f"{pre}NInferior", sum(g["verdict"] == "inférieur" for g in shown), "{}")
    put(f"{pre}BonferroniAlpha", 0.025 / len(shown), "{:.5f}")


e4 = need("e4_summary_colab.json")
e4s = need("e4_summary_colab_strictmath.json")
if not e4s["math_grader"].startswith("boxed only") or e4["math_grader"].startswith("boxed only"):
    raise SystemExit("E4 : les résumés normal et strict (MATH) sont inversés ou mal étiquetés")
put_e4(e4, "eFour")
put_e4(e4s, "eFourStrict")
# The paper says the strict MATH grader leaves the references' scores unchanged: checked, not assumed.
for ref in REFS:
    if e4["benches"]["math500"]["alone"][ref] != e4s["benches"]["math500"]["alone"][ref]:
        raise SystemExit(f"E4 : le correcteur strict change le score de {ref} sur MATH-500 (corriger le texte)")
put("eFourFamilies", len(FAMILIES), "{}")

# Model files and engine, from the run manifests (every model, every benchmark and split).
gguf_gb, quants, engines, max_tokens = {}, set(), set(), {}
for m in E4_MODELS:
    for bench in E4_BENCHES:
        for split in ("dev", "test"):
            man = read_manifest(RES / f"solo_{m.replace('/', '__')}_colab_{bench}_{split}.jsonl")
            if man is None:
                raise SystemExit(f"E4 : manifeste absent pour {m} {bench} {split}")
            gguf_gb[m] = man["weights"]["size"] / 1e9
            quants.add(re.search(r"(Q\d_\d|F16|BF16)", man["gguf"]).group(1))
            engines.add(re.search(r"build (\d+)", man["engine"]).group(1))
            max_tokens.setdefault(bench, set()).add(man["max_tokens"])
            if man["temperature"] != 0.0 or man["thinking"]:
                raise SystemExit(f"E4 : {m} {bench} {split} n'est pas glouton sans réflexion")
if len(quants) != 1 or len(engines) != 1 or any(len(v) != 1 for v in max_tokens.values()):
    raise SystemExit(f"E4 : quantification, moteur ou budget de jetons non uniformes : {quants} {engines} {max_tokens}")
put("eFourQuant", quants.pop().replace("_", r"\_"))
put("eFourBuild", engines.pop())
for bench, tag in E4_BENCHES.items():
    put(f"eFour{tag}MaxTokens", max_tokens[bench].pop(), "{}")
swarm = list(FAMILIES.values())
for m, mtag in E4_MODELS.items():
    put(f"eFourParams{mtag}", models.fmt_b(models.total_b(m)))
put("eFourSwarmParams", sum(models.total_b(m) for m in swarm), "{:.1f}")
put("eFourSwarmGB", sum(gguf_gb[m] for m in swarm), "{:.1f}")
put("eFourLargestPeerGB", max(gguf_gb[m] for m in swarm), "{:.1f}")
put("eFourLargestPeerParams", models.fmt_b(max(models.total_b(m) for m in swarm)))
for ref, rtag in E4_REFS.items():
    put(f"eFour{rtag}GB", gguf_gb[ref], "{:.1f}")
    put(f"eFour{rtag}Params", models.fmt_b(models.total_b(ref)))

# The caption of fig_scaling names the only zero-weight member (OLMo-2-1B on MMLU-Pro): checked.
_zero = {(b, f) for b, x in e4["benches"].items() for f, w in x["weights"].items() if w <= 0}
if _zero != {("mmlupro", "AllenAI")}:
    raise SystemExit(f"E4 : membres de poids nul {_zero} (corriger la légende de fig_scaling)")

# The paper says the MMLU-Pro superiority over the 27B does not survive a Bonferroni correction over the
# displayed comparisons: checked here, so the sentence cannot outlive the data.
_g = e4["benches"]["mmlupro"]["comparisons"]["wvote|Qwen/Qwen3.8-27B"]
if not (_g["p_superior"] < 0.025 and _g["p_superior"] > 0.025 / 20):
    raise SystemExit("E4 : la phrase sur la correction de Bonferroni (MMLU-Pro contre le 27B) ne tient plus")

# Cascade replay (analyze_cascade_e4.py): macro averages, pooled paired tests, per-benchmark details.
cas = need("e4_cascade_summary_colab.json")
if list(cas["refs"]) != list(REFS) or cas["fixed_m"] != 2 or cas["delta"] != 2.0:
    raise SystemExit("E4 : la cascade ne correspond pas aux références d'analyze_e4.py")
if abs(cas["mean"]["swarm"] - sum(e4["benches"][b]["swarm"]["wvote"] for b in E4_BENCHES) / 4) > 1e-9:
    raise SystemExit("E4 : l'essaim de la cascade n'est pas celui d'analyze_e4.py")
put("eFourPooledN", cas["pooled"]["n"], "{}")
put("eFourPooledSwarm", cas["pooled"]["swarm"])
for ref, rtag in E4_REFS.items():
    mr, pr = cas["mean"]["refs"][ref], cas["pooled"]["refs"][ref]
    put(f"eFourPooled{rtag}", pr["alone"])
    put_cmp(f"eFourPooled{rtag}", pr["swarm_vs_ref"])
    put(f"eFourCompute{rtag}", mr["swarm_compute_vs_ref"], "{:.2f}")
    for rule, utag in CASCADE_RULES.items():
        p = f"eFourCasc{rtag}{utag}"
        put(f"{p}Calls", mr["rules"][rule]["calls_test"])
        put(f"{p}Acc", mr["rules"][rule]["accuracy"])
        put(f"{p}Compute", mr["rules"][rule]["compute_vs_ref"], "{:.2f}")
        put(f"{p}PooledAcc", pr["rules"][rule]["accuracy"])
        put_cmp(p, pr["rules"][rule]["vs_ref"])
        put_cmp(f"{p}Swarm", pr["rules"][rule]["vs_swarm"])
        for bench, tag in E4_BENCHES.items():
            r = cas["benches"][bench]["refs"][ref]["rules"][rule]
            put(f"{p}{tag}Acc", r["accuracy"])
            put(f"{p}{tag}Calls", r["calls_test"])
            put(f"{p}{tag}Rescued", r["rescued"], "{}")
            put(f"{p}{tag}Broken", r["broken"], "{}")
            put_cmp(f"{p}{tag}", r["vs_ref"])
            put_cmp(f"{p}{tag}Swarm", r["vs_swarm"])

# The same cascade under the strict MATH-500 grader (weights and thresholds refitted on dev).
cass = need("e4_cascade_summary_colab_strictmath.json")
if not cass.get("math_grader", "").startswith("boxed only") or cas.get("math_grader", "boxed only").startswith("boxed only"):
    raise SystemExit("E4 : cascades normale et stricte inversées ou mal étiquetées")
for ref, rtag in E4_REFS.items():
    for rule, utag in CASCADE_RULES.items():
        p = f"eFourStrictCasc{rtag}{utag}"
        put(f"{p}Calls", cass["mean"]["refs"][ref]["rules"][rule]["calls_test"])
        put(f"{p}Acc", cass["mean"]["refs"][ref]["rules"][rule]["accuracy"])
        put_cmp(p, cass["pooled"]["refs"][ref]["rules"][rule]["vs_ref"])

# --- E10: parallel sections, judged quality and modelled latency ----------------------------------
from analyze_sot import pick_best  # noqa: E402
from colab_jobs import SOT  # noqa: E402


def put_e10(e: dict):
    """Require the complete frozen campaign; never publish numbers from a partial summary."""
    protocol = e.get("protocol", {})
    expected = {"tag": "sot1", "outline_model": SOT["outline"], "judge": SOT["judge"],
                "peers": SOT["peers"], "baselines": SOT["peers"], "suffix": "_colab",
                "select_split": "dev", "forced_best": None, "n": None, "boot": 10000,
                "budgets": {"baseline": 1024, "outline": 256, "expand": 256},
                "rtt_ms": [50, 100, 150], "uniform_speed": 50.0, "default_speed": 40.0, "pp_speed": None}
    if not e.get("complete") or any(protocol.get(k) != v for k, v in expected.items()):
        raise SystemExit("E10 : résumé incomplet ou protocole différent de la campagne sot1")
    # Freeze the rates described in main.tex, including which models use the assumed rate.
    measured = {"ibm-granite/granite-4.2-3b": 56.0, "HuggingFaceTB/SmolLM3-3B": 37.0}
    table = json.loads((RES.parent / "speeds_consumer.json").read_text(encoding="utf-8"))
    rates = {m: measured.get(m, 40.0) for m in SOT["peers"]}
    defaulted = [m for m in SOT["peers"] if m not in measured]
    if ({m: table.get(m, 40.0) for m in SOT["peers"]} != rates
            or [m for m in SOT["peers"] if m not in table] != defaulted):
        raise SystemExit("E10 : table de débits différente des hypothèses figées de l'article")
    if e.get("speeds") != rates or e.get("defaulted") != defaulted:
        raise SystemExit("E10 : débits effectifs ou modèles par défaut différents des hypothèses de l'article")
    splits = e.get("splits", {})
    if set(splits) != {"dev", "test", "other"}:
        raise SystemExit("E10 : partitions dev/test/other requises")
    for split, n in (("dev", 16), ("test", 24), ("other", 40)):
        b = splits[split]
        st_keys = {"n", "fallback", "long_points", "expansions_cut", "expansions",
                   "mean_tokens_parallel", "mean_tokens_single", "single_cut"}
        if not st_keys <= set(b.get("structure", {})) or b["structure"]["n"] != n:
            raise SystemExit(f"E10 : structure {split} incomplète")
        if any(set(b["structure"][k]) != set(SOT["peers"]) for k in ("mean_tokens_single", "single_cut")):
            raise SystemExit(f"E10 : longueurs des solos {split} incomplètes")
        if set(b.get("quality", {})) != set(SOT["peers"]):
            raise SystemExit(f"E10 : partition {split} incomplète")
        for q in b["quality"].values():
            q_keys = {"n", "win", "tie", "loss", "score", "score_ci", "soft", "consistency"}
            if not q_keys <= set(q) or q["n"] != n or sum(q[k] for k in ("win", "tie", "loss")) != n:
                raise SystemExit(f"E10 : jugements {split} incomplets")
        keys = {f"{label}|{rtt}" for label in ("mono-flux + défauts", "toutes à 50 tok/s") for rtt in (50, 100, 150)}
        if set(b.get("speed", {})) != keys or any(set(t) != {"parallel", *SOT["peers"]} for t in b["speed"].values()):
            raise SystemExit(f"E10 : scénarios de latence {split} incomplets")
        for t in b["speed"].values():
            if "mean_s" not in t["parallel"] or any(not {"mean_s", "speedup_latency"} <= set(t[m]) for m in SOT["peers"]):
                raise SystemExit(f"E10 : temps de latence {split} incomplets")
    best = e.get("best_single")
    if best != pick_best(splits["dev"]["quality"]):
        raise SystemExit("E10 : meilleur solo différent du choix sur dev")
    sources = e.get("sources_sha256", {})
    wanted = set()
    for split in splits:
        names = [f"sot_outline_{SOT['outline'].replace('/', '__')}_colab_{split}.jsonl",
                 f"sot_judge_sot1_{SOT['judge'].replace('/', '__')}_colab_{split}.jsonl"]
        names += [f"sot_{stage}_{'sot1_' if stage == 'expand' else ''}{m.replace('/', '__')}_colab_{split}.jsonl"
                  for stage in ("base", "expand") for m in SOT["peers"]]
        wanted.update(names)
        wanted.update(n + ".meta.json" for n in names)
    if set(sources) != wanted:
        raise SystemExit("E10 : empreintes de la campagne incomplètes")
    for name, sha in sources.items():
        path = RES / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"E10 : source absente ou modifiée : {name}")
    put("eTenBestName", best.split("/")[-1])
    put("eTenPeers", len(SOT["peers"]), "{}")
    put("eTenDefaulted", len(e["defaulted"]), "{}")
    put("eTenDefaultSpeed", protocol["default_speed"], "{:.0f}")
    put("eTenUniformSpeed", protocol["uniform_speed"], "{:.0f}")
    put("eTenBoot", protocol["boot"], "{}")
    put("eTenCILevel", 95, "{}")
    for budget, value in protocol["budgets"].items():
        put(f"eTen{budget.title()}Budget", value, "{}")
    for rtt, word in ((50, "Low"), (100, "Mid"), (150, "High")):
        put(f"eTenRtt{word}", rtt, "{}")
    for split, word in (("dev", "Dev"), ("test", "Test"), ("other", "Other")):
        b, pre = splits[split], f"eTen{word}"
        st, q = b["structure"], b["quality"][best]
        for key, tag in (("n", "N"), ("fallback", "Fallback"), ("long_points", "LongPoints"),
                         ("expansions_cut", "Cut"), ("expansions", "Expansions")):
            put(pre + tag, st[key], "{}")
        for key, tag in (("win", "Win"), ("tie", "Tie"), ("loss", "Loss")):
            put(pre + tag, q[key], "{}")
        put(pre + "Score", q["score"], "{:.3f}")
        put(pre + "ScoreCI", "[{:.3f}, {:.3f}]".format(*q["score_ci"]))
        put(pre + "Consistency", 100 * q["consistency"], "{:.1f}")
        put(pre + "Tokens", st["mean_tokens_parallel"], "{:.0f}")
        put(pre + "SoloTokens", st["mean_tokens_single"][best], "{:.0f}")
        put(pre + "SoloCut", st["single_cut"][best], "{}")
        t = b["speed"]["mono-flux + défauts|100"]
        put(pre + "Seconds", t["parallel"]["mean_s"])
        put(pre + "SoloSeconds", t[best]["mean_s"])
        put(pre + "Speedup", t[best]["speedup_latency"], "{:.2f}")
        put(pre + "UniformSpeedup", b["speed"]["toutes à 50 tok/s|100"][best]["speedup_latency"], "{:.2f}")


e10 = load("sot_summary_colab.json")
if e10 is None:
    raise SystemExit("E10 : sot_summary_colab.json absent (lancer phase0/analyze_sot.py)")
put_e10(e10)

# Every result macro used by the paper is defined: a missing results file shows "??" instead of
# breaking the compilation.
used = set(re.findall(r"\\(e(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Gemma|Small)[A-Za-z]*)",
                      (HERE / "main.tex").read_text(encoding="utf-8")))
for name in sorted(used - set(macros)):
    macros[name] = "??"
broken = sorted(n for n in used if n.startswith(("eFour", "eTen")) and macros[n] == "??")
if broken:  # E4 and E10 never compile with a placeholder.
    raise SystemExit(f"E4/E10 : macros utilisées par main.tex mais non définies : {broken}")
OUT.write_text("% Generated by make_numbers.py from phase0/results -- do not edit.\n" +
               "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())), encoding="utf-8")
print(f"{len(macros)} macros -> {OUT.name}")
