"""Paper figures from phase0/results (no number typed by hand).

    uv run --project ../phase0 --group paper python make_figures.py

fig_scaling.pdf  E4: accuracy of the swarm vs number of families (mean over all subsets, and best-k by
                 dev), one panel per complete benchmark, with the bigger references as horizontal lines.
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
REF_STYLE = {"Qwen/Qwen3.5-9B": ("Qwen3.5-9B", ":"), "google/gemma-4-12B-it": ("Gemma-4-12B", "--"),
             "mistralai/Ministral-3-14B-Instruct-2512": ("Ministral-3-14B", "-."),
             "Qwen/Qwen3.8-27B": ("Qwen3.8-27B", (0, (1, 1)))}


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
    if not benches:  # analyze_e4.py run before any benchmark was complete
        raise Missing("aucun benchmark complet dans e4_summary_colab.json")
    biggest = max(models.total_b(m) for m in e4["families"].values())
    fig, axes = plt.subplots(1, len(benches), figsize=(3.2 * len(benches), 3.0), squeeze=False)
    for ax, b in zip(axes[0], benches):
        sc = e4["benches"][b]["scaling"]
        ks = sorted(int(k) for k in sc)
        ax.plot(ks, [sc[str(k)]["mean_all_subsets"] for k in ks], "o-", color="C0", label="swarm, mean over subsets")
        ax.plot(ks, [sc[str(k)]["best_k_by_dev"] for k in ks], "s--", color="C1", label="swarm, best $k$ by dev")
        for ref, (lab, ls) in REF_STYLE.items():
            if ref in e4["benches"][b]["alone"]:
                ax.axhline(100 * e4["benches"][b]["alone"][ref], color="gray", ls=ls, lw=1,
                           label=f"{lab} ({models.fmt_b(models.total_b(ref))}B)")
        ax.set_title(NAMES[b])
        ax.set_xlabel(f"families in the swarm ($\\leq${models.fmt_b(biggest)}B total each)")
        ax.set_xticks(ks)
    axes[0][0].set_ylabel("accuracy (%)")
    axes[0][-1].legend(fontsize=7, loc="lower right")
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
