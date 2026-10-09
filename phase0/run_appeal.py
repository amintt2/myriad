"""E5: minority appeal (docs/03_idees_codex.md, idea 1; paper, Theorem "appeal identity").

For each multiple-choice question, the calibrated mixture of the 4 experts (pass 0, temperatures frozen on
dev) gives its answer a and its runner-up b. The appeal is triggered when b is the FIRST choice of at least
one expert. The most confident proposer of b is set aside; every other expert judges "a or b?" without
seeing any vote, in both display orders, and returns ell = mean over the two orders of log P(b) / P(a).
analyze_appeal.py then replaces a by b when the median ell exceeds a threshold chosen on dev.

One judge per call (the PC GPU holds one model at a time):
    uv run python run_appeal.py --judge HuggingFaceTB/SmolLM3-3B --gguf ../models/SmolLM3-Q8_0.gguf --bench arc --split dev
Writes results/appeal_<judge>_<bench>_<split>.jsonl (+ manifest), resumable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from essaim import data
from essaim.common import FIXED_DATE, resolve_revision
from essaim.results import ResultsFile, file_identity, read_manifest, read_rows

RESULTS = Path(__file__).resolve().parent / "results"
PROMPT_VERSION = "appeal-v1"
EXPERTS = ["Qwen/Qwen3-1.7B@_gpu", "ibm-granite/granite-3.3-2b-instruct@_gpu", "HuggingFaceTB/SmolLM3-3B@_colab",
           "google/gemma-4-E2B-it@_gpu"]
LOADERS = {"arc": data.arc, "mmlupro": data.mmlu_pro}


def logsoftmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def appeal_set(bench: str, split: str, run: int = 0) -> tuple[list[dict], dict]:
    """Questions where the runner-up of the calibrated mixture is some expert's first choice."""
    summ = json.loads((RESULTS / "mc_summary_test.json").read_text(encoding="utf-8"))
    if summ["experts"] != EXPERTS or summ.get("fit_split") != "dev":
        raise SystemExit("mc_summary_test.json : experts ou températures inattendus")
    temps = summ["benches"][bench]["frozen_temperatures"]["pass"]  # fitted on dev
    rows = []
    for spec in EXPERTS:
        model, suffix = spec.split("@")
        path = RESULTS / f"mc_{model.replace('/', '__')}{suffix}_{split}.jsonl"
        man = read_manifest(path)
        all_rows = read_rows(path, key=("bench", "id", "run"))  # duplicates raise
        mine = {r["id"]: r for r in all_rows if r["bench"] == bench and r["run"] == run}
        if man is None or len(mine) != man["n"]:  # a complete source run only
            raise SystemExit(f"{path.name} : {len(mine)} questions {bench} au passage {run}, "
                             f"{man['n'] if man else '?'} attendues")
        rows.append(mine)
    ids = sorted(rows[0])
    if any(sorted(r) != ids for r in rows):
        raise SystemExit("les experts n'ont pas les mêmes questions")
    out = []
    for i in ids:
        lp = np.log(np.clip(np.array([r[i]["probs"] for r in rows]), 1e-300, None))
        cal = np.exp(np.stack([logsoftmax(lp[k] / temps[k]) for k in range(len(rows))]))
        q = cal.mean(0)
        order = np.argsort(-q, kind="stable")
        a, b = int(order[0]), int(order[1])
        tops = [int(lp[k].argmax()) for k in range(len(rows))]
        proposers = [k for k in range(len(rows)) if tops[k] == b]
        if not proposers:
            continue
        lead = max(proposers, key=lambda k: cal[k, b])  # set aside: the most confident proposer of b
        out.append({"id": i, "a": a, "b": b, "gold": rows[0][i]["answer"], "proposer": EXPERTS[lead].split("@")[0],
                    "judges": [EXPERTS[k].split("@")[0] for k in range(len(rows)) if k != lead]})
    return out, {"temperatures": temps, "temperatures_from": "mc_summary_test.json (fitted on dev)", "pass": run,
                 "experts": EXPERTS, "questions_total": len(ids)}


def judge_prompt(item: dict, first: str, second: str) -> str:
    return (f"{item['question']}\n\nTwo candidate answers:\nA. {first}\nB. {second}\n\n"
            "Which candidate answer is correct? Reply with the letter only (A or B).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--bench", choices=list(LOADERS), required=True)
    ap.add_argument("--split", choices=data.SPLITS, required=True)
    a = ap.parse_args()

    cases, meta = appeal_set(a.bench, a.split)
    mine = [c for c in cases if a.judge in c["judges"]]
    items = {it["id"]: it for it in LOADERS[a.bench](split=a.split)}
    from essaim.llamacpp import LlamaCppLM
    lm = LlamaCppLM(a.judge, a.gguf, ctx=4096, revision=resolve_revision(a.judge))
    manifest = {**lm.identity(), "weights": file_identity(a.gguf), "data": data.dataset_identity(a.bench),
                "bench": a.bench, "split": a.split, "prompt": PROMPT_VERSION, "template_date": str(FIXED_DATE), **meta}
    path = RESULTS / f"appeal_{a.judge.replace('/', '__')}_{a.bench}_{a.split}.jsonl"
    out = ResultsFile(path, manifest, key=("id", "order"))
    for c in mine:
        it = items[c["id"]]
        opt_a, opt_b = it["options"][c["a"]], it["options"][c["b"]]
        for order, (first, second) in (("ab", (opt_a, opt_b)), ("ba", (opt_b, opt_a))):
            if (c["id"], order) in out.done:
                continue
            p = lm.letter_probs(judge_prompt(it, first, second), ["A", "B"])
            p_b, p_a = (p[1], p[0]) if order == "ab" else (p[0], p[1])
            out.write({"id": c["id"], "order": order, "judge": a.judge, "a": c["a"], "b": c["b"], "gold": c["gold"],
                       "p_a": p_a, "p_b": p_b, "ell": math.log(max(p_b, 1e-300)) - math.log(max(p_a, 1e-300))})
    out.release()
    print(f"{a.judge} {a.bench} {a.split} : {len(mine)} appels sur {len(cases)} ({meta['questions_total']} questions)",
          flush=True)


if __name__ == "__main__":
    main()
