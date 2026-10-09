"""Write paper/numbers.tex: every number quoted in the paper, as LaTeX macros, read from phase0/results.

    uv run --project ../phase0 python make_numbers.py

No number in main.tex is typed by hand: each one is a macro defined here from a results file, so the
paper always matches the data. Missing results produce a visible "??" instead of a stale value.
"""
from __future__ import annotations

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
e4 = load("e4_summary_colab.json")
if e4:
    names = {"gsm8k": "Gsm", "math500": "Math", "arc": "ArcGen", "mmlupro": "MmluGen"}
    for bench, tag in names.items():
        b = e4["benches"].get(bench)
        if not b:
            continue
        put(f"eFour{tag}Swarm", b["swarm"]["wvote"])
        put(f"eFour{tag}Vote", b["swarm"]["vote"])
        put(f"eFour{tag}Oracle", b["oracle"])
        put(f"eFour{tag}Waits", b["certificate_waits"], "{:.2f}")
        for ref, rtag in (("Qwen/Qwen3.5-9B", "QwenNine"), ("google/gemma-4-12B-it", "GemmaTwelve"),
                          ("mistralai/Ministral-3-14B-Instruct-2512", "MinistralFourteen"),
                          ("Qwen/Qwen3.8-27B", "QwenTwentySeven")):
            if ref in b["alone"]:
                put(f"eFour{tag}{rtag}", 100 * b["alone"][ref])
                g = b["comparisons"].get(f"wvote|{ref}")
                if g:
                    put(f"eFour{tag}{rtag}Diff", signed(g["mean"]))
                    put(f"eFour{tag}{rtag}CI", ci(g["lo95"], g["hi95"]))

# Every result macro used by the paper is defined: a missing results file shows "??" instead of
# breaking the compilation.
used = set(re.findall(r"\\(e(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Gemma|Small)[A-Za-z]*)",
                      (HERE / "main.tex").read_text(encoding="utf-8")))
for name in sorted(used - set(macros)):
    macros[name] = "??"
OUT.write_text("% Generated by make_numbers.py from phase0/results -- do not edit.\n" +
               "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())), encoding="utf-8")
print(f"{len(macros)} macros -> {OUT.name}")
