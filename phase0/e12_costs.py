"""Problem-cluster statistics and explicitly scoped observed generation accounting."""
from __future__ import annotations

import random
from pathlib import Path


def cluster_interval(groups, other=None, seed=12, repeats=10000):
    """Resample whole problems; ratio of summed successes to summed steps, paired when requested."""
    keys = sorted(groups)
    if not keys or any(not groups[k] for k in keys):
        raise ValueError("nonempty problem clusters required")
    if other is not None and (set(other) != set(groups) or any(len(other[k]) != len(groups[k]) for k in keys)):
        raise ValueError("paired clusters must have identical steps")
    counts = [(sum(groups[k]) - (sum(other[k]) if other is not None else 0), len(groups[k])) for k in keys]
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        sample = [counts[rng.randrange(len(keys))] for _ in keys]
        draws.append(100 * sum(s for s, n in sample) / sum(n for s, n in sample))
    draws.sort()

    def quantile(q):
        x = (len(draws) - 1) * q
        i = int(x)
        return draws[i] + (draws[min(i + 1, len(draws) - 1)] - draws[i]) * (x - i)

    return {"mean": 100 * sum(s for s, n in counts) / sum(n for s, n in counts),
            "lo95": quantile(.025), "hi95": quantile(.975), "seed": seed, "replicates": repeats,
            "unit": "problem", "weighting": "ratio of summed steps", "exploratory": True}


def oracle_chains(groups_by_model):
    """Step coverage and coherent single-model chains are separate descriptive objects."""
    models = list(groups_by_model)
    keys = sorted(groups_by_model[models[0]])
    coverage = {k: [any(groups_by_model[m][k][j] for m in models)
                    for j in range(len(groups_by_model[models[0]][k]))] for k in keys}
    chains = {k: any(all(groups_by_model[m][k]) for m in models) for k in keys}
    return coverage, chains


def accounting(events, attempts, n_problems, parallel):
    calls = [r for r in events if r["kind"] == "call"]
    problems = [r for r in events if r["kind"] == "problem"]
    batches = [r for r in events if r["kind"] == "batch"]
    resumed = len(attempts) > 1 or any(a["interrupted"] or a["resumed"] for a in attempts) or \
        any(r["resumed_steps"] for r in problems) or len(batches) != 1
    known = lambda key: sum(r[key] for r in calls if r[key] is not None)
    unknown = lambda key: sum(r[key] is None for r in calls)
    duration_known = not resumed and len(problems) == n_problems and all(r["complete"] for r in problems) and \
        len({r["problem"] for r in problems}) == n_problems and batches[0]["new"] == len(calls)
    contradictions = []
    if duration_known:
        # Calls within one problem are sequential; problem workers may overlap.
        for worker in problems:
            seconds = sum(r["wall_s"] for r in calls if r["problem"] == worker["problem"])
            if worker["wall_s"] + 1e-6 * max(1, seconds) < seconds:
                contradictions.append("worker shorter than its sequential calls")
        lower = max(r["wall_s"] for r in problems)
        if batches[0]["wall_s"] + 1e-6 * max(1, lower) < lower:
            contradictions.append("batch shorter than concurrent worker lower bound")
        duration_known = not contradictions
    tokens_known = duration_known and not unknown("n_tokens")
    prompt_known = duration_known and not unknown("prompt_tokens")
    problem_out = {}
    for pid in sorted({r["problem"] for r in calls}):
        pc = [r for r in calls if r["problem"] == pid]
        problem_out[pid] = {"observed_calls": len(pc), "observed_call_seconds_sum": sum(r["wall_s"] for r in pc),
                            "observed_completion_tokens_sum": sum(r["n_tokens"] for r in pc if r["n_tokens"] is not None),
                            "unknown_completion_token_calls": sum(r["n_tokens"] is None for r in pc),
                            "observed_prompt_tokens_sum": sum(r["prompt_tokens"] for r in pc
                                                              if r["prompt_tokens"] is not None),
                            "unknown_prompt_token_calls": sum(r["prompt_tokens"] is None for r in pc),
                            "errors": sum(r["error_type"] is not None for r in pc),
                            "observed_worker_seconds": [r["wall_s"] for r in problems if r["problem"] == pid],
                            "total_tokens": sum(r["n_tokens"] for r in pc) if tokens_known else None}
    return {"parallel_problem_workers": parallel, "shared_gpu_concurrency": "models may overlap; not attributed",
            "observed_calls": len(calls), "observed_call_seconds_sum": sum(r["wall_s"] for r in calls),
            "total_calls": len(calls) if duration_known else None,
            "observed_completion_tokens_sum": known("n_tokens"), "observed_prompt_tokens_sum": known("prompt_tokens"),
            "unknown_completion_token_calls": unknown("n_tokens"), "unknown_prompt_token_calls": unknown("prompt_tokens"),
            "errors": sum(r["error_type"] is not None for r in calls), "problems": problem_out,
            "observed_batch_seconds": [r["wall_s"] for r in batches],
            "attempt_total_known": duration_known and tokens_known and prompt_known,
            "duration_total_known": duration_known, "duration_contradictions": contradictions,
            "completion_token_total_known": tokens_known,
            "prompt_token_total_known": prompt_known,
            "total_completion_tokens": known("n_tokens") if tokens_known else None,
            "total_prompt_tokens": known("prompt_tokens") if prompt_known else None,
            "total_generation_wall_s": batches[0]["wall_s"] if duration_known else None,
            "batch_amortized_seconds_per_problem": batches[0]["wall_s"] / n_problems if duration_known else None,
            "campaign_bill": None, "pc_energy": None, "wan_seconds": None,
            "scope": "generation workers only; excludes grading, server startup, download and preflight"}


def energy_scenario(seconds, watts, eur_kwh):
    if seconds is None:
        return None
    if min(seconds, watts, eur_kwh) < 0:
        raise ValueError("nonnegative scenario assumptions required")
    return {"estimated_wh": watts * seconds / 3600, "estimated_eur": watts * seconds / 3600000 * eur_kwh,
            "assumed_watts": watts, "assumed_eur_kwh": eur_kwh,
            "duration_scope": "observed generation batch amortized per problem", "measured_energy": False,
            "allocation_assumption": "hypothetical full device power during this batch; shared GPU not attributed"}


def figures(systems, costs, destination, watts=None, eur_kwh=None):
    """Only interpretable durations; no oracle or purported distributed selector on these axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    scenarios = watts is not None and eur_kwh is not None
    for axis in (["time", "cost"] if scenarios else ["time"]):
        points = []
        for model, c in costs.items():
            seconds = c["batch_amortized_seconds_per_problem"]
            if seconds is None:
                continue
            x = seconds if axis == "time" else energy_scenario(seconds, watts, eur_kwh)["estimated_eur"]
            points.append((model, x, systems[model]["steps_pct"]))
        if not points:
            continue
        with plt.rc_context({"svg.hashsalt": "e12", "font.family": "DejaVu Sans"}):
            fig, ax = plt.subplots(figsize=(9, 6))
            for model, x, y in points:
                ax.scatter(x, y)
                ax.annotate(model.split("/")[-1], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.set_ylabel("SciCode test: graded steps passed (%)")
            label = "Observed generation batch seconds / problem (amortized)" if axis == "time" else \
                f"Estimated EUR / problem ({watts:g} W, {eur_kwh:g} EUR/kWh assumed)"
            ax.set_xlabel(label)
            ax.set_ylim(0, 100)
            ax.grid(alpha=.2)
            fig.tight_layout()
            for ext in ("svg", "png", "pdf"):
                path = Path(str(destination) + f"_{axis}.{ext}")
                metadata = {"Date": None} if ext == "svg" else \
                    {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
                fig.savefig(path, metadata=metadata, dpi=180)
                out.append(path.name)
            plt.close(fig)
    return out
