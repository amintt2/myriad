"""Paper figures from phase0/results (no number typed by hand).

    uv run --project ../phase0 --group paper python make_figures.py

fig_scaling.pdf  E4: accuracy of the swarm vs number of families (mean over all subsets, and best-k by
                 dev), one panel per benchmark (all four required) and one for their mean, with the bigger
                 references as horizontal lines; sizes are total parameter counts (essaim/models.py).
fig_protocol.pdf E2: accuracy vs round trips for vote / cross-scored / majority-prefix.

Every figure is independent: one whose source is missing or incomplete is skipped (and its old PDF deleted,
so that a stale figure is never compiled into the paper), the others are still drawn. The exit code is 1 when
a figure that main.tex includes could not be drawn, so the paper cannot silently show old results.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "phase0" / "results"
sys.path.insert(0, str(HERE.parent / "phase0"))
from essaim import models  # noqa: E402  (model sizes: one convention, total parameters)

NAMES = {"gsm8k": "GSM8K", "math500": "MATH-500", "arc": "ARC-Challenge", "mmlupro": "MMLU-Pro"}
REF_STYLE = {"Qwen/Qwen3.5-9B": ("Qwen3.5-9B", ":", "tab:green"), "google/gemma-4-12B-it": ("Gemma-4-12B", "--", "tab:red"),
             "mistralai/Ministral-3-14B-Instruct-2512": ("Ministral-3-14B", "-.", "tab:purple"),
             "Qwen/Qwen3.8-27B": ("Qwen3.8-27B", "-", "black")}


class Missing(Exception):
    """The source of a figure is absent or incomplete."""


def scaling() -> None:
    p = RES / "e4_summary_colab.json"
    if not p.exists():
        raise Missing("e4_summary_colab.json absent")
    e4 = json.loads(p.read_text(encoding="utf-8"))
    if e4.get("preview"):
        raise Missing("e4_summary_colab.json est un aperçu (--partial)")
    benches = [b for b in NAMES if b in e4.get("benches", {})]
    if benches != list(NAMES):  # analyze_e4.py writes complete benchmarks only; the paper needs all four
        raise Missing(f"benchmarks complets dans e4_summary_colab.json : {benches}")
    biggest = max(models.total_b(m) for m in e4["families"].values())
    ks = sorted(int(k) for k in e4["benches"][benches[0]]["scaling"])
    # One panel per benchmark, then their macro average (equal weight per benchmark, as in analyze_e4.py).
    curves = {b: {"mean": [e4["benches"][b]["scaling"][str(k)]["mean_all_subsets"] for k in ks],
                  "best": [e4["benches"][b]["scaling"][str(k)]["best_k_by_dev"] for k in ks],
                  "refs": {r: 100 * e4["benches"][b]["alone"][r] for r in REF_STYLE if r in e4["benches"][b]["alone"]}}
              for b in benches}
    avg = lambda xs: sum(xs) / len(xs)
    curves["mean"] = {"mean": [avg([curves[b]["mean"][j] for b in benches]) for j in range(len(ks))],
                      "best": [avg([curves[b]["best"][j] for b in benches]) for j in range(len(ks))],
                      "refs": {r: avg([curves[b]["refs"][r] for b in benches]) for r in REF_STYLE
                               if all(r in curves[b]["refs"] for b in benches)}}
    titles = {**NAMES, "mean": "Mean of the four"}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.9))
    flat = [ax for row in axes for ax in row]
    for ax, b in zip(flat, benches + ["mean"]):
        cv = curves[b]
        ax.plot(ks, cv["mean"], "o-", color="C0", ms=3, label="swarm, mean over all subsets of $k$ families")
        ax.plot(ks, cv["best"], "s--", color="C1", ms=3, label="swarm, $k$ best families on dev")
        for ref, acc in cv["refs"].items():
            lab, ls, col = REF_STYLE[ref]
            ax.axhline(acc, color=col, ls=ls, lw=0.9, alpha=0.8, label=f"{lab} alone ({models.fmt_b(models.total_b(ref))}B total)")
        ax.set_title(titles[b], fontsize=9)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=7)
    for ax in axes[1][:2]:
        ax.set_xlabel("families $k$", fontsize=8)
    for row in axes:
        row[0].set_ylabel("test accuracy (%)", fontsize=8)
    handles, labels = flat[0].get_legend_handles_labels()
    flat[-1].axis("off")
    flat[-1].legend(handles, labels, fontsize=7, loc="center", frameon=False,
                    title=f"each family: one model $\\leq${models.fmt_b(biggest)}B total", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fig_scaling.pdf")
    plt.close(fig)


def protocol() -> None:
    p = RES / "gen_summary_essaim4_colab_test.json"
    if not p.exists():
        raise Missing("gen_summary_essaim4_colab_test.json absent")
    g = json.loads(p.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for mode, lab in (("vote", "vote"), ("croise", "cross-scored blocks"), ("accord", "majority prefix")):
        ax.scatter(g["cost"][mode]["trips_mean"], g["accuracy"][mode], s=40)
        ax.annotate(lab, (g["cost"][mode]["trips_mean"], g["accuracy"][mode]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(g["accuracy"][g["best_peer"]], color="gray", ls="--", lw=1)
    ax.annotate("best single peer", (1, g["accuracy"][g["best_peer"]]), fontsize=7, xytext=(2, -10), textcoords="offset points")
    ref = next((k for k in g["accuracy"] if k.startswith("référence")), None)
    if ref:
        ax.axhline(g["accuracy"][ref], color="gray", ls=":", lw=1)
        ax.annotate(ref.split(":", 1)[-1].strip().split("/")[-1], (1, g["accuracy"][ref]), fontsize=7, xytext=(2, 3),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("WAN round trips per question")
    ax.set_ylabel("GSM8K accuracy (%)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_protocol.pdf")
    plt.close(fig)


FIGURES = {"fig_scaling.pdf": scaling, "fig_protocol.pdf": protocol}


def main() -> int:
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    needed = {Path(m).name if m.endswith(".pdf") else Path(m).name + ".pdf"
              for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)}
    failed = []
    for name, draw in FIGURES.items():
        try:
            draw()
            print(name)
        except Missing as e:
            (HERE / name).unlink(missing_ok=True)  # never leave a figure of older results behind
            print(f"{name} non générée : {e}" + (" (figure incluse dans main.tex)" if name in needed else ""))
            if name in needed:
                failed.append(name)
    unknown = sorted(needed - set(FIGURES))
    if unknown:
        print(f"figures incluses dans main.tex sans générateur : {unknown}")
    if failed:
        print(f"ÉCHEC : {failed} manquante(s), main.tex ne compilera pas avec des résultats périmés", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
