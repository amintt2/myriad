"""Observable generation costs, with per-problem attribution and explicitly labelled duration proxies."""
from __future__ import annotations

from collections import Counter
import statistics as st


def measure(rows):
    result = {"calls": len(rows), "length": sum(r.get("finish") == "length" for r in rows),
              "missing_finish": sum(r.get("finish") is None for r in rows)}
    for field in ("n_tokens", "prompt_tokens", "ms"):
        values = [r[field] for r in rows if r.get(field) is not None]
        result[field] = {"known_sum": sum(values), "missing": len(rows) - len(values),
                         "total": sum(values) if len(values) == len(rows) else None}
    return result


def generation_observations(gens):
    return {m: {"all": measure(gs), "greedy": measure([g for g in gs if g["sample"] == 0]),
                "sampled": measure([g for g in gs if g["sample"] != 0]),
                "finish": dict(Counter(g.get("finish") for g in gs)),
                "extract": dict(Counter(g["extract"] for g in gs))} for m, gs in gens.items()}


def observable_costs(ids, gens, systems, selections, swarm, best, ref, called):
    indexed = {(m, g["id"], g["sample"]): g for m, gs in gens.items() for g in gs}
    by_problem = {q: [(m, s, g) for (m, qid, s), g in indexed.items() if qid == q] for q in ids}
    out = {}
    for key, accuracy in systems.items():
        if key.startswith("alone|"):
            models, samples = [key.split("|", 1)[1]], [0]
        elif key.startswith("refself|"):
            models, samples = [key.split("|", 1)[1]], None
        elif key in ("best", "best_self"):
            models, samples = [best], [0] if key == "best" else None
        else:
            models, samples = list(swarm.values()), [0] if key.endswith("|g") else None
        per_problem = []
        for q, correct in zip(ids, accuracy):
            rs = [g for m, s, g in by_problem[q] if m in models and (samples is None or s in samples)]
            call = key.startswith("cascade|") and called(q, int(key.split("|")[1]))
            if call:
                rs.append(indexed[ref, q, 0])
            selected = selections[key][q]
            per_problem.append({"id": q, "correct": bool(correct), "reference_called": bool(call),
                                "selected": {"model": selected.model, "sample": selected.sample,
                                             "prog": selected.prog} if selected else None,
                                **measure(rs)})
        means = {}
        for field in ("n_tokens", "prompt_tokens", "ms"):
            vals = [p[field]["total"] for p in per_problem]
            means[field] = {"known_sum": sum(p[field]["known_sum"] for p in per_problem),
                            "missing_calls": sum(p[field]["missing"] for p in per_problem),
                            "mean_per_problem": st.mean(vals) if all(v is not None for v in vals) else None}
        out[key] = {"n": len(ids), "accuracy": 100 * st.mean(accuracy), "metrics": means,
                    "calls": sum(p["calls"] for p in per_problem),
                    "length": sum(p["length"] for p in per_problem),
                    "missing_finish": sum(p["missing_finish"] for p in per_problem), "problems": per_problem}
    return out


def cost_report(costs, primary, threshold, best, refs):
    keys = ["best", "best_self", primary] + ([f"cascade|{threshold}"] if threshold else [])
    keys += [f"{kind}|{m}" for m in refs for kind in ("alone", "refself")]
    lines = ["", "### Coûts observables de génération", "",
             "Sommes des compteurs des appels nécessaires à chaque mode, par problème test retenu. "
             "Les compteurs de jetons utilisent les tokenizers des producteurs. La somme des ms est un proxy "
             "de durées d'appels, **pas le temps mur** : les requêtes de run_code.py sont parallèles "
             "(défaut 16), sur un GPU partagé. Même leur maximum serait une projection, sans WAN ni essaim réel. "
             "Une cascade est un replay offline : seuls les appels de référence déclenchés sont attribués. "
             "Le temps exec (visible + extra + HIDDEN) n'est pas un temps de sélection en ligne mesuré. "
             "Énergie, coût monétaire réel et coût total restent inconnus ; aucune comparaison à calcul égal.", "",
             "| mode | jetons sortie/problème | jetons prompt/problème | somme durées appels (s)/problème, proxy | "
             "length/appels |", "| --- | --- | --- | --- | --- |"]
    for k in keys:
        c = costs[k]
        vals = [c["metrics"][f]["mean_per_problem"] for f in ("n_tokens", "prompt_tokens", "ms")]
        shown = ["inconnu" if x is None else f"{x / (1000 if i == 2 else 1):.2f}" for i, x in enumerate(vals)]
        lines.append(f"| {k.replace('|', ' / ')} | " + " | ".join(shown) + f" | {c['length']}/{c['calls']} |")
    return lines + ["", "Détails par problème/mode, compteurs connus et manquants, troncatures et observations "
                    "par modèle/split : e11_summary_colab.json. Le coût des problèmes exclus figure dans "
                    "generation_observations ; il ne disparaît pas des totaux de campagne.", ""]


def plot_costs(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analyze_code import RESULTS
    with plt.rc_context({"svg.hashsalt": "e11", "font.size": 8}):
        for field, label, scale in (("n_tokens", "Output tokens/problem (producer tokenizers)", 1),
                                    ("ms", "Sum of generation call durations/problem (s; proxy, not wall time)", 1000)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            for ax, (bench, b) in zip(axes, summary["benches"].items()):
                keys = ["best", "best_self", b["primary"], f"cascade|{b['cascade']['m']}"]
                keys += [f"{kind}|{m}" for m in summary["refs"] for kind in ("alone", "refself")]
                for key in keys:
                    c = b["costs"][key]
                    x = c["metrics"][field]["mean_per_problem"]
                    if x is None:
                        continue
                    if key.startswith(("alone|", "refself|")):
                        name = key.split("/")[-1] + (" (own selection)" if key.startswith("refself|") else " (greedy)")
                        marker = "x" if key.startswith("refself|") else "o"
                    else:
                        name = ("Best dev peer (own selection)" if key == "best_self" else
                                "Best dev peer (greedy)" if key == "best" else
                                "Cascade (dev threshold)" if key.startswith("cascade|") else "Primary swarm (dev rule)")
                        marker = "D" if key == "best_self" else "s"
                    ax.scatter(x / scale, c["accuracy"], s=24, marker=marker, label=name)
                ax.set(title=bench, xlabel=label, ylabel="Selected-program test accuracy (%)", ylim=(0, 103))
                ax.grid(alpha=.2)
                ax.margins(x=.2)
                ax.legend(loc="upper center", bbox_to_anchor=(.5, -.20), ncol=2, fontsize=6, frameon=False)
            for ext in ("png", "svg", "pdf"):
                metadata = {"Date": None} if ext == "svg" else (
                    {"CreationDate": None, "ModDate": None} if ext == "pdf" else {})
                fig.savefig(RESULTS / f"e11_accuracy_{field}_colab.{ext}", dpi=160, metadata=metadata)
            plt.close(fig)
