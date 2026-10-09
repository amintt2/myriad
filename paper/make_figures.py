"""Paper figures from phase0/results (no number typed by hand).

    uv run --project ../phase0 --group paper python make_figures.py

fig_scaling.pdf  E4: accuracy of the swarm vs number of families (mean over all subsets, and best-k by
                 dev), one panel per benchmark, with the bigger references as horizontal lines.
fig_protocol.pdf E2: accuracy vs round trips for vote / cross-scored / majority-prefix.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "phase0" / "results"
NAMES = {"gsm8k": "GSM8K", "math500": "MATH-500", "arc": "ARC-Challenge", "mmlupro": "MMLU-Pro"}
REF_STYLE = {"Qwen/Qwen3.5-9B": ("Qwen3.5-9B", ":"), "google/gemma-4-12B-it": ("Gemma-4-12B", "--"),
             "mistralai/Ministral-3-14B-Instruct-2512": ("Ministral-3-14B", "-."),
             "Qwen/Qwen3.8-27B": ("Qwen3.8-27B", (0, (1, 1)))}


def scaling():
    p = RES / "e4_summary_colab.json"
    if not p.exists():
        print("e4_summary_colab.json absent : figure d'échelle non générée")
        return
    e4 = json.loads(p.read_text(encoding="utf-8"))
    benches = [b for b in NAMES if b in e4["benches"]]
    fig, axes = plt.subplots(1, len(benches), figsize=(3.2 * len(benches), 3.0), squeeze=False)
    for ax, b in zip(axes[0], benches):
        sc = e4["benches"][b]["scaling"]
        ks = sorted(int(k) for k in sc)
        ax.plot(ks, [sc[str(k)]["mean_all_subsets"] for k in ks], "o-", color="C0", label="swarm, mean over subsets")
        ax.plot(ks, [sc[str(k)]["best_k_by_dev"] for k in ks], "s--", color="C1", label="swarm, best $k$ by dev")
        for ref, (lab, ls) in REF_STYLE.items():
            if ref in e4["benches"][b]["alone"]:
                ax.axhline(100 * e4["benches"][b]["alone"][ref], color="gray", ls=ls, lw=1, label=lab)
        ax.set_title(NAMES[b])
        ax.set_xlabel("families in the swarm ($\\leq$4B each)")
        ax.set_xticks(ks)
    axes[0][0].set_ylabel("accuracy (%)")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "fig_scaling.pdf")
    print("fig_scaling.pdf")


def protocol():
    p = RES / "gen_summary_essaim4_colab_test.json"
    if not p.exists():
        return
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
        ax.annotate("Qwen3-4B", (1, g["accuracy"][ref]), fontsize=7, xytext=(2, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("WAN round trips per question")
    ax.set_ylabel("GSM8K accuracy (%)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_protocol.pdf")
    print("fig_protocol.pdf")


if __name__ == "__main__":
    scaling()
    protocol()
