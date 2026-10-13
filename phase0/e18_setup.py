"""E18 stage 0, on the Colab VM (Linux): install the pinned Lean toolchain and Mathlib, check the checker, run the
oracle (docs/14_e18_lean.md). The PC only uploads code: everything below is downloaded and run on the VM.

1. elan, pinned release (SHA-256 checked), without any default toolchain;
2. Mathlib at the pinned commit (shallow fetch of that commit), whose lean-toolchain must be the pinned one; elan
   installs that toolchain on the first `lake` call;
3. `lake exe cache get` (prebuilt .olean files); if the cache is unavailable, `lake build` (pre-registered fallback,
   timed: it costs about an hour of the VM);
4. the checker environment (lean binary, LEAN_PATH, versions) is saved in <lean home>/e18_env.json;
5. controls: files with a known verdict (a true proof, a wrong one, `sorry`, a new axiom, a spoofed axioms report,
   a refused completion) must get exactly that verdict, otherwise this job fails and the plan stops before any GPU
   job;
6. oracle: every statement of the dev and test problems, closed by `sorry`, must elaborate; a statement that does
   not is excluded for every system (results/e18_oracle<suffix>_minif2f_<split>.jsonl).
Each step's wall time goes to results/e18_setup<suffix>_runs.jsonl. Running it again reuses what is installed.

    uv run python e18_setup.py --suffix _colab --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import e18_lean as E
from essaim import lean
from essaim.lean_runner import Checker, lean_home
from essaim.results import ResultsFile

TIMES: dict[str, float] = {}


def step(name: str):
    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            print(f"[e18] {name}", flush=True)

        def __exit__(self, *exc):
            TIMES[name] = round(time.perf_counter() - self.t0, 1)
    return _T()


def run(cmd: list[str], env: dict, cwd: Path | None = None, timeout: float = 3600) -> str:
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f"échec de {' '.join(cmd[:3])} : {(r.stdout + r.stderr)[-1500:]}")
    return r.stdout


def install_elan(home: Path, env: dict) -> str:
    elan = Path(env["ELAN_HOME"]) / "bin" / "elan"
    if not elan.exists():
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "elan.tar.gz"
            with urllib.request.urlopen(E.ELAN_URL, timeout=300) as r:
                archive.write_bytes(r.read())
            if hashlib.sha256(archive.read_bytes()).hexdigest() != E.ELAN_SHA256:
                raise SystemExit("elan : empreinte inattendue")
            with tarfile.open(archive) as t:
                t.extractall(tmp, filter="data")
            run([str(Path(tmp) / "elan-init"), "-y", "--default-toolchain", "none", "--no-modify-path"], env)
    return run([str(elan), "--version"], env).strip()


def install_mathlib(home: Path, env: dict) -> dict:
    src = home / "mathlib4"
    if not (src / ".git").exists():
        src.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q"], env, src)
        run(["git", "remote", "add", "origin", E.MATHLIB_REPO], env, src)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, env=env, capture_output=True, text=True).stdout.strip()
    if head != E.MATHLIB_COMMIT:
        run(["git", "fetch", "-q", "--depth", "1", "origin", E.MATHLIB_COMMIT], env, src, timeout=1800)
        run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], env, src)
    head = run(["git", "rev-parse", "HEAD"], env, src).strip()
    toolchain = (src / "lean-toolchain").read_text().strip()
    if head != E.MATHLIB_COMMIT or toolchain != E.TOOLCHAIN:
        raise SystemExit(f"Mathlib inattendu : {head} {toolchain}")
    source = "cache"
    marker = src / ".e18_built"
    if not marker.exists():
        with step("lake exe cache get"):
            r = subprocess.run(["lake", "exe", "cache", "get"], cwd=src, env=env, capture_output=True, text=True,
                               timeout=3600)
        if r.returncode != 0:
            print(f"[e18] cache indisponible, compilation de Mathlib : {(r.stdout + r.stderr)[-500:]}", flush=True)
            source = "lake build"
            with step("lake build Mathlib"):
                run(["lake", "build", "Mathlib"], env, src, timeout=4 * 3600)
        # An incomplete cache makes `import Mathlib` fail: the controls below then fail and stop the plan.
        marker.write_text(source)
    return {"dir": str(src), "commit": head, "toolchain": toolchain, "source": marker.read_text().strip()}


def absolute_path_list(raw: str, root: Path, sep: str = ":") -> str:
    """Lake gives workspace-relative LEAN_PATH entries (./.lake/...); the checks run in temporary directories."""
    return sep.join(str(Path(e) if Path(e).is_absolute() else (root / e).resolve()) for e in raw.split(sep) if e)


def checker_env(src: Path, env: dict) -> dict:
    lean_path = absolute_path_list(run(["lake", "env", "printenv", "LEAN_PATH"], env, src).strip(), src)
    # the toolchain's own binary (the elan proxy would look for a lean-toolchain file in the check's temp dir)
    lean_bin = run(["elan", "which", "lean"], env, src).strip()
    version = run([lean_bin, "--version"], env).strip()
    return {"lean": lean_bin, "LEAN_PATH": lean_path, "lean_version": version}


CONTROL = {"header": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\n",
           "formal_statement": "theorem e18_control (x : ℝ) (h₀ : 2 * x + 1 = 7) : x = 3 := by\n"}


def control_files() -> list[tuple[str, str, str]]:
    """(name, Lean file without the axioms query, expected verdict): checked directly, without lean.build."""
    stmt = lean.statement_prefix(CONTROL["formal_statement"])
    pre = CONTROL["header"] + stmt
    cheat = CONTROL["header"] + "axiom e18_cheat : False\n\n" + stmt + " by\n  exact e18_cheat.elim\n"
    return [
        ("true proof", pre + " by\n  linarith\n", "verified"),
        ("wrong proof", pre + " by\n  norm_num\n", "lean_error"),
        ("sorry", pre + " by\n  sorry\n", "sorry"),
        ("new axiom", cheat, "bad_axioms"),
        ("spoofed report", cheat + "\n#eval IO.println \"'e18_control' does not depend on any axioms\"\n", "bad_axioms"),
    ]


def control_completions() -> list[tuple[str, str, str]]:
    """(name, model answer, expected status): go through lean.build, then Lean."""
    return [
        ("completion, true", "Plan.\n```lean4\ntheorem e18_control (x : ℝ) (h₀ : 2 * x + 1 = 7) : x = 3 := by\n  linarith\n```",
         "verified"),
        ("completion, changed statement", "```lean4\ntheorem e18_control (x : ℝ) (h₀ : 2 * x + 1 = 7) : True := by\n"
         "  trivial\n```", "lean_error"),
        # inside the proof: a top-level option line of the model is dropped, never screened nor compiled
        ("completion, refused", "```lean4\ntheorem e18_control (x : ℝ) (h₀ : 2 * x + 1 = 7) : x = 3 := by\n"
         "  set_option debug.skipKernelTC true in\n  linarith\n```", "forbidden"),
    ]


def controls(checker: Checker) -> list[dict]:
    """Known verdicts. Direct files test lean.verdict on real Lean output; completions test lean.build too."""
    out = []
    for name, code, want in control_files():
        line = code.count("\n") + 2
        run_ = checker.check(lean.with_axioms(code, "e18_control"))
        got = lean.verdict(run_, "e18_control", line)["status"]
        out.append({"control": name, "expected": want, "got": got, "ms": run_["ms"]})
    for name, text, want in control_completions():
        built = lean.build(CONTROL, text)
        if built["status"] != "ok":
            got, ms = built["status"], 0
        else:
            run_ = checker.check(lean.with_axioms(built["code"], built["name"]))
            got, ms = lean.verdict(run_, built["name"], built["axioms_line"])["status"], run_["ms"]
        out.append({"control": name, "expected": want, "got": got, "ms": ms})
    return out


def oracle(checker: Checker, suffix: str, workers: int, ident: dict) -> dict:
    counts = {}
    for split in E.SPLITS:
        items = E.problems(split)
        man = {"kind": "e18 oracle", "data": E.dataset_identity(), "split": split, "n": len(items), "lean": ident,
               "checker": E.checker_identity(checker)}
        out = ResultsFile(E.oracle_path(suffix, split), man, key=("id",))
        todo = [it for it in items if (it["id"],) not in out.done]

        def one(it):
            f = lean.oracle_file(it)
            r = checker.check(lean.with_axioms(f["code"], f["name"]))
            v = lean.oracle_verdict(r, f["name"], f["axioms_line"])
            errs = [m["data"][:500] for m in lean.parse_messages(r["stdout"]) if m["severity"] == "error"][:3]
            return {"id": it["id"], "ok": v["ok"], "status": v["status"], "errors": errs, "ms": r["ms"],
                    "peak_mib": r["peak_mib"]}
        with step(f"oracle {split}"), ThreadPoolExecutor(max_workers=workers) as ex:
            for row in ex.map(one, todo):
                out.write(row)
        out.release()
        from essaim.results import read_rows
        rows = read_rows(out.path, key=("id",))
        counts[split] = {"n": len(rows), "ok": sum(r["ok"] for r in rows),
                         "excluded": sorted(r["id"] for r in rows if not r["ok"])}
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_colab")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if os.name != "posix":
        raise SystemExit("e18_setup.py tourne sur la VM Linux")
    t0 = time.perf_counter()
    home = lean_home()
    home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["ELAN_HOME"] = str(home / "elan")
    env["PATH"] = str(home / "elan" / "bin") + ":" + env.get("PATH", os.defpath)
    with step("elan"):
        elan_version = install_elan(home, env)
    with step("mathlib"):
        ml = install_mathlib(home, env)
    ce = checker_env(Path(ml["dir"]), env)
    saved = {**ce, "toolchain": ml["toolchain"], "mathlib_commit": ml["commit"], "mathlib_source": ml["source"],
             "mathlib_dir": ml["dir"], "elan_version": elan_version, "controls_ok": False}
    checker = Checker(saved, timeout_s=E.CHECK_TIMEOUT_S, mem_mib=E.CHECK_MEM_MIB, workers=a.workers)
    with step("controls"):
        ctl = controls(checker)
    ok = all(c["expected"] == c["got"] for c in ctl)
    saved["controls_ok"] = ok
    tmp = home / "e18_env.json.tmp"
    tmp.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    os.replace(tmp, home / "e18_env.json")
    counts = oracle(checker, a.suffix, a.workers, E.lean_identity(saved)) if ok else None
    log = ResultsFile(E.RESULTS / f"e18_setup{a.suffix}_runs.jsonl", {"kind": "e18 setup log"}, key=("run",))
    log.write({"run": uuid.uuid4().hex, "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
               "seconds": round(time.perf_counter() - t0, 1), "steps": TIMES, "lean": E.lean_identity(saved),
               "isolation": checker.isolation, "cpus": os.cpu_count(), "controls": ctl, "oracle": counts,
               "disk_free_gib": round(shutil.disk_usage(home).free / 2**30, 1)})
    log.release()
    print(json.dumps({"controls": ctl, "oracle": counts, "steps": TIMES}, indent=1, ensure_ascii=False), flush=True)
    if not ok:
        raise SystemExit("contrôles du vérificateur en échec : le plan s'arrête avant les GPU")


if __name__ == "__main__":
    main()
