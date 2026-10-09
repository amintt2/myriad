"""E12 execution: grade the SciCode sub-problems written by run_aa.py (CPU only), or check the harness.

For every model and every graded sub-problem, the script `dependencies + the model's earlier code + its code +
the official tests` runs in a limited child process (essaim/scicode.run_script) and passes only if it reaches
its end. `--oracle` runs the dataset's reference code through the same path instead of a model: if the
dev reference fails a sub-problem, that dev sub-problem is a harness problem (missing package, wrong h5 file...),
and the analysis leaves it out of dev for everyone. Test has no references and is never oracle-filtered.
With `--oracle` the exit code is 1 when fewer than ORACLE_MIN of the graded dev sub-problems pass:
colab_jobs.py runs it first, so that a broken harness stops the plan
before any GPU time is spent.

    uv run python exec_scicode.py --oracle --suffix _colab           # the harness alone
    uv run python exec_scicode.py --suffix _colab --workers 8        # every aa_*_scicode_* generation file
Writes results/sciexec_<model|oracle><suffix>_scicode_<split>.jsonl (+ manifest), resumable, key (id).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from essaim import data, sandbox, scicode
from essaim.results import ResultsFile, read_manifest, read_rows

RESULTS = Path(__file__).resolve().parent / "results"
ORACLE_MIN = 0.9
HARNESS_VERSION = "scicode-exec-v2"


def exec_path(tag: str, suffix: str, split: str) -> Path:
    return RESULTS / f"sciexec_{tag.replace('/', '__')}{suffix}_scicode_{split}.jsonl"


def gen_path(model: str, suffix: str, split: str) -> Path:
    return RESULTS / f"aa_{model.replace('/', '__')}{suffix}_scicode_{split}.jsonl"


def discover(suffix: str, split: str) -> list[str]:
    pat = re.compile(rf"^aa_(.+){re.escape(suffix)}_scicode_{split}\.jsonl$")
    return sorted(m.group(1).replace("__", "/") for p in RESULTS.glob(f"aa_*{suffix}_scicode_{split}.jsonl")
                  if (m := pat.match(p.name)))


def gen_digest(rows: dict[str, dict]) -> str:
    """SHA-256 of the generated texts (id and text of every row): binds the grades to the code they graded."""
    return hashlib.sha256(json.dumps(sorted((i, r["text"]) for i, r in rows.items()), ensure_ascii=False).encode()).hexdigest()


def jobs_for_model(problems: list[dict], rows: dict[str, dict]) -> list[tuple]:
    """(row id, problem id, assembled script or None when the model wrote no code) for every graded step."""
    out = []
    for p in problems:
        chain = [scicode.chain_code(p, k, rows[scicode.row_id(s)]["text"]) if not scicode.is_skipped(p, s) else
                 scicode.chain_code(p, k, None) for k, s in enumerate(p["steps"])]
        for k, s in enumerate(p["steps"]):
            if scicode.is_skipped(p, s):
                continue
            text = rows[scicode.row_id(s)]["text"]
            code_text = scicode.extract(text, scicode.def_name(s["header"]))[0]
            out.append((scicode.row_id(s), p["id"], scicode.script(p, k, chain, code_text) if code_text.strip() else None))
    return out


def jobs_for_oracle(problems: list[dict]) -> list[tuple]:
    return [(scicode.row_id(s), p["id"], scicode.reference_script(p, k)) for p in problems
            for k, s in enumerate(p["steps"]) if not scicode.is_skipped(p, s)]


def run_one(model: str, jid: str, pid: str, src: str | None, h5: str) -> dict:
    if src is None:
        return {"id": jid, "problem": pid, "model": model, "ok": False, "err": "NoCode", "msg": "", "status": "none", "ms": 0}
    r = scicode.run_script(src, h5)
    return {"id": jid, "problem": pid, "model": model, "script": hashlib.sha256(src.encode()).hexdigest()[:16], **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--models", nargs="+", default=None, help="default: every generation file with this suffix")
    ap.add_argument("--splits", nargs="+", default=None, choices=["dev", "test"])
    ap.add_argument("--oracle", action="store_true", help="the reference code instead of a model (checks the harness)")
    ap.add_argument("--n", type=int, default=None, help="first n problems (smoke tests)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args()
    a.splits = a.splits if a.splits is not None else (list(scicode.ORACLE_SPLITS) if a.oracle else list(scicode.FILES))
    if a.oracle and any(sp not in scicode.ORACLE_SPLITS for sp in a.splits):
        ap.error("--oracle requires a split with reference code (dev)")

    def on_term(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, on_term)
    missing = scicode.preflight()
    if missing:
        raise SystemExit("environnement SciCode incomplet : " + " ; ".join(missing))
    h5 = scicode.h5_path()
    ident = scicode.h5_identity(h5)
    iso = sandbox.isolation()
    import numpy
    import scipy
    print("isolation :", {k: v for k, v in iso.items() if k != "limits_s"}, "| cibles :", ident["sha256"][:12], flush=True)
    ok_all = True
    for split in a.splits:
        probs = scicode.problems(split)[: a.n or None]
        tags = ["oracle"] if a.oracle else (a.models or discover(a.suffix, split))
        if not tags:
            raise SystemExit(f"aucune génération scicode {split} pour le suffixe {a.suffix!r}")
        for tag in tags:
            if a.oracle:
                jobs = jobs_for_oracle(probs)
                gen = None
            else:
                gpath = gen_path(tag, a.suffix, split)
                gman = read_manifest(gpath)
                if gman is None:
                    raise SystemExit(f"{gpath.name} : absent")
                if gman.get("prompt") != scicode.PROMPT_VERSION or \
                        gman.get("skipped_code") != scicode.SKIPPED_SHA256:
                    raise SystemExit(f"{gpath.name} : protocole SciCode incompatible, régénérer les réponses")
                if gman["n"] != len(probs) and not a.n:
                    raise SystemExit(f"{gpath.name} : {gman['n']} problèmes, {len(probs)} attendus")
                rows = {r["id"]: r for r in read_rows(gpath, key=("id",))}
                graded = sum(1 for p in probs for s in p["steps"] if not scicode.is_skipped(p, s))
                need = {scicode.row_id(s) for p in probs for s in p["steps"] if not scicode.is_skipped(p, s)}
                if not need <= set(rows):
                    raise SystemExit(f"{gpath.name} : {len(need & set(rows))} sous-problèmes sur {graded}, génération incomplète")
                jobs = jobs_for_model(probs, rows)
                gen = {**gman, "rows_sha256": gen_digest(rows)}  # the whole generation protocol, and the texts graded
            manifest = {"bench": "scicode", "split": split, "model": tag, "n": len(jobs), "data": data.dataset_identity("scicode"),
                        "h5": ident, "official_commit": scicode.OFFICIAL_COMMIT, "harness": HARNESS_VERSION,
                        "timeout_s": scicode.DEFAULT_TIMEOUT_S, "isolation": iso, "python": platform.python_version(),
                        "numpy": numpy.__version__, "scipy": scipy.__version__, "generation": gen, "oracle": a.oracle,
                        "skipped_code": scicode.SKIPPED_SHA256}
            out = ResultsFile(exec_path(tag, a.suffix, split), manifest, key=("id",))
            todo = [j for j in jobs if (j[0],) not in out.done]
            t0, k = time.perf_counter(), 0
            ex = ThreadPoolExecutor(max_workers=a.workers)
            try:
                futs = [ex.submit(run_one, tag, jid, pid, src, str(h5)) for jid, pid, src in todo]
                for fut in as_completed(futs):
                    out.write(fut.result())
                    k += 1
                    if k % 50 == 0:
                        print(f"{tag} {split} : {k}/{len(todo)} sous-problèmes en {time.perf_counter() - t0:.0f} s", flush=True)
            except BaseException:
                sandbox.shutdown()
                ex.shutdown(wait=False, cancel_futures=True)
                out.release()
                raise
            ex.shutdown()
            out.release()
            res = read_rows(exec_path(tag, a.suffix, split), key=("id",))
            n_ok = sum(bool(r["ok"]) for r in res)
            print(f"{tag} {split} : {n_ok}/{len(res)} sous-problèmes passent ({k} exécutés en "
                  f"{time.perf_counter() - t0:.0f} s)", flush=True)
            if a.oracle:
                for r in res:
                    if not r["ok"]:
                        print(f"  oracle échoue : {r['id']} {r['err']} {r['msg'][:120]!r} ({r['status']})", flush=True)
                ok_all &= bool(res) and n_ok / len(res) >= ORACLE_MIN
    if a.oracle and not ok_all:
        print(f"ÉCHEC : moins de {100 * ORACLE_MIN:.0f} % des références passent : le banc est cassé", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
