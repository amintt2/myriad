"""E11 execution: run every distinct program of the generation files in the sandbox (essaim/sandbox.py).

For each problem, the code of every answer (all models, greedy and samples) is re-extracted from the raw text
with the current extractor, and each distinct program is run three times, each in its own child process:
  visible  the tests shown in the prompt (pass/fail per test)
  extra    the visible inputs, then --extra inputs derived from them (essaim/code.extra_inputs): one output
           signature per input, for functional clustering
  hidden   the EvalPlus tests (grading only; stops at the first failure)
The dataset's reference solution goes through the same three runs (prog "reference"): visible tests that it
fails are not used by the analysis, and a hidden failure of the reference flags a harness problem.

    uv run python exec_code.py --suffix _colab --workers 12
Writes results/codeexec_<tag><suffix>_<bench>_<split>.jsonl (+ manifest), resumable, key (id, prog).
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from essaim import code, data, sandbox
from essaim.results import ResultsFile, read_manifest, read_rows


def generations(bench: str, split: str, suffix: str, models: list[str] | None) -> tuple[dict, dict, dict]:
    """{model: rows}, {model: manifest} and the shared data identity, for complete generation files only."""
    if models is None:
        pat = re.compile(rf"^code_(.+){re.escape(suffix)}_{bench}_{split}\.jsonl$")
        models = sorted(m.group(1).replace("__", "/") for p in code.RESULTS.glob(f"code_*{suffix}_{bench}_{split}.jsonl")
                        if (m := pat.match(p.name)))
    rows, mans = {}, {}
    for m in models:
        path = code.gen_path(m, suffix, bench, split)
        man = read_manifest(path)
        if man is None:
            raise SystemExit(f"{path.name} : absent")
        r = read_rows(path, key=("id", "sample"))
        want = man["n"] * (man["samples"] + 1)
        if len(r) != want:
            raise SystemExit(f"{path.name} : {len(r)} solutions sur {want}, génération incomplète")
        rows[m], mans[m] = r, man
    if not rows:
        raise SystemExit(f"aucune génération {bench} {split} pour le suffixe {suffix!r}")
    datas = {repr(sorted(man["data"].items())) for man in mans.values()}
    ns = {man["n"] for man in mans.values()}
    if len(datas) != 1 or len(ns) != 1:
        raise SystemExit(f"{bench} {split} : jeux de données différents d'un modèle à l'autre")
    return rows, mans, next(iter(mans.values()))["data"]


def run_program(bench: str, item: dict, prog: str, src: str, tests: dict) -> dict:
    entry, head = tests["entry"], tests["header"]
    t0 = time.perf_counter()
    v = sandbox.run_visible(src, entry, head, tests["visible"])
    x = sandbox.run_extra(src, entry, head, tests["extra"])
    h = sandbox.run_hidden(src, entry, head, tests["hidden"])
    return {"id": item["id"], "prog": prog, "bench": bench, "load": v["load"],
            "visible": [c["ok"] for c in v["cases"]], "visible_err": [c["err"] for c in v["cases"]],
            "extra": x["sigs"], "hidden": {k: h[k] for k in ("pass", "n_pass", "n", "err")},
            "status": {"visible": v["status"], "extra": x["status"], "hidden": h["status"]},
            "ms": round((time.perf_counter() - t0) * 1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--tag", default="e11")
    ap.add_argument("--models", nargs="+", default=None, help="default: every generation file with this suffix")
    ap.add_argument("--benches", nargs="+", default=list(code.BENCHES), choices=list(code.BENCHES))
    ap.add_argument("--splits", nargs="+", default=list(data.SPLITS), choices=list(data.SPLITS))
    ap.add_argument("--extra", type=int, default=16, help="extra inputs per problem, besides the visible ones")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--reference-only", action="store_true",
                    help="only the reference solutions of the first --n problems (checks the harness, no model)")
    ap.add_argument("--n", type=int, default=None, help="with --reference-only: first n problems (default: all)")
    a = ap.parse_args()
    if "_" in a.tag:
        ap.error("--tag sans « _ » (il fait partie du nom des fichiers)")
    iso = sandbox.isolation()
    print("isolation :", {k: v for k, v in iso.items() if k != "limits_s"}, flush=True)
    import numpy
    for bench in a.benches:
        for split in a.splits:
            if a.reference_only:
                items = data.CODE_LOADERS[bench](a.n or data.PART[bench], split=split)
                rows, ident = {}, data.dataset_identity(bench)
            else:
                rows, mans, ident = generations(bench, split, a.suffix, a.models)
                items = data.CODE_LOADERS[bench](next(iter(mans.values()))["n"], split=split)
                if data.dataset_identity(bench) != ident:
                    raise SystemExit(f"{bench} : les générations ne viennent pas des données chargées ici")
            manifest = {"bench": bench, "split": split, "n": len(items), "data": ident, "tests": code.TESTS_VERSION,
                        "extract": code.EXTRACT_VERSION, "extra_n": a.extra, "isolation": iso,
                        "python": platform.python_version(), "numpy": numpy.__version__,
                        "models": sorted(rows), "reference_only": a.reference_only}
            path = code.exec_path(a.tag, a.suffix + ("_refcheck" if a.reference_only else ""), bench, split)
            out = ResultsFile(path, manifest, key=("id", "prog"))
            jobs, seen = [], set()
            for it in items:
                entry = code.entry_point(bench, it)
                tests = {"entry": entry, "header": code.header(bench, it), "visible": code.visible_tests(bench, it),
                         "extra": code.extra_inputs(bench, it, a.extra), "hidden": code.hidden_spec(bench, it)}
                progs = {"reference": it["reference"]}
                for r in rows.values():
                    for g in r:
                        if g["id"] == it["id"]:
                            src, _ = code.extract_code(g["text"], entry)
                            progs.setdefault(code.prog_id(src), src)
                for prog, src in progs.items():
                    if (it["id"], prog) not in out.done and (it["id"], prog) not in seen:
                        seen.add((it["id"], prog))
                        jobs.append((it, prog, src, tests))
            t0, k = time.perf_counter(), 0
            ex = ThreadPoolExecutor(max_workers=a.workers)
            try:
                futs = [ex.submit(run_program, bench, it, prog, src, tests) for it, prog, src, tests in jobs]
                for fut in as_completed(futs):
                    row = fut.result()
                    out.write(row)
                    k += 1
                    if row["prog"] == "reference" and not row["hidden"]["pass"]:
                        print(f"ATTENTION : la solution de référence de {row['id']} échoue aux tests cachés "
                              f"({row['hidden']}, {row['status']})", flush=True)
                    if k % 200 == 0:
                        print(f"{bench} {split} : {k}/{len(jobs)} programmes en {time.perf_counter() - t0:.0f} s", flush=True)
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)
                out.release()
                raise
            ex.shutdown()
            out.release()
            print(f"{bench} {split} : {k} programmes exécutés en {time.perf_counter() - t0:.0f} s "
                  f"({len(items)} problèmes)", flush=True)


if __name__ == "__main__":
    main()
