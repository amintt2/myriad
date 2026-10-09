"""E12: SciCode (Tian et al. 2024; Artificial Analysis' "Scientific Coding" benchmark).

Official source, pinned: dataset SciCode1/SciCode (Apache-2.0) at data.REVISIONS["scicode"], files
problems_dev.jsonl (15 problems, 50 sub-problems) and problems_test.jsonl (65 problems, 291 steps, 288 graded);
code of the benchmark scicode-bench/SciCode at OFFICIAL_COMMIT (Apache-2.0). The numeric targets of the tests
are NOT in the dataset: they are in one HDF5 file (test_data.h5) that the authors publish on Google Drive
(a folder, not a pinned file). The owner downloads it once into phase0/data/scicode_test_data.h5; its SHA-256
is recorded in every manifest and the analysis refuses a mix of two files (see H5_SHA256).

Protocol (the official one, eval/scripts/gencode.py and test_generated_code.py at OFFICIAL_COMMIT):
  * a problem is a chain of sub-problems; the model writes sub-problem k given the main description, the
    descriptions (and scientist-written background, as Artificial Analysis does) of the previous sub-problems
    with the code THE MODEL itself wrote for them, and the function header of sub-problem k;
  * three sub-problems use the official eval/data code instead of a model answer, in the chain and in the
    score (they are not generated, not graded): SKIPPED below;
  * sub-problem k is graded by running `dependencies + previous code + code k + tests`; the test cases compare
    with targets read from the HDF5 file; it passes only if every test case passes (exit without error);
    a main problem is solved when all its graded sub-problems pass. Artificial Analysis reports the
    sub-problem rate (pass@1, 300 s timeout); both are reported here.
Differences from the official/AA run, all deliberate and recorded in the manifest: one greedy generation per
sub-problem instead of 3 repeats; the prompt is our own wording of the official one (PROMPT_VERSION), not
its bytes; the model is served by llama-server (--jinja, thinking off) with a 2048-token answer limit.

This module is pure parsing and script assembly: nothing here executes model code (see run_script).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from . import code as codemod
from . import data

PROMPT_VERSION = "scicode-ours-v2"
ORACLE_SPLITS = ("dev",)
SKIPPED_DIR = Path(__file__).with_name("scicode_skipped")
OFFICIAL_COMMIT = "e3158ea011d4235245a547460d3688d7ccbf9900"  # scicode-bench/SciCode, main on 2025-10-07
FILES = {"dev": "problems_dev.jsonl", "test": "problems_test.jsonl"}
# git blob ids of the two files at data.REVISIONS["scicode"] (from the public metadata of the dataset)
BLOB_SHA1 = {"problems_dev.jsonl": "ce8984e15447ef0876e696906d100f3eadf941d0",
             "problems_test.jsonl": "b72b96897f0b72ee560b588b7b80106e00a93d29"}
COUNTS = {"dev": (15, 50), "test": (65, 291)}  # raw steps; the official score counts 288 after the 3 skipped
SKIPPED = {("13", 6), ("62", 1), ("76", 3)}  # (problem id, 1-based step): official supplied code, neither generated nor graded
SKIPPED_SHA256 = {
    "13.6": "795a2b57c2d9bb12ca4eaf16d6b8e1f202015a89a886628858abf42a1b18a94e",
    "62.1": "bc9931d88a7d5950091b72a996a25b8be6c936fd136b01005e22c3d45b0008a2",
    "76.3": "4758300d96ea726cdc0fbf749f1bc437030d2a3e8b43bde232ecdeaf636cd367",
}
H5_NAME = "scicode_test_data.h5"
H5_SHA256: str | None = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"  # test_data.h5, 1 049 345 865 bytes, Drive folder of the SciCode README (2026-10-09)
H5_HELP = (
    "SciCode a besoin du fichier de cibles numériques test_data.h5, publié par les auteurs sur Google Drive "
    "(https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR). Étape manuelle pour le "
    f"propriétaire : le télécharger et l'enregistrer sous phase0/data/{H5_NAME} (ou pointer SCICODE_H5 dessus), "
    "puis lancer `uv run python -m essaim.scicode` : il affiche son SHA-256, à reporter dans H5_SHA256.")
DEFAULT_TIMEOUT_S = 300  # as Artificial Analysis (dataset v1.0.1); sandbox phase "hidden" has a 300 s child limit
_DEF = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)", re.M)


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


# ---------- data ----------

def _problem(r: dict, split: str) -> dict:
    steps = []
    for st in r["sub_steps"]:
        steps.append({"number": st["step_number"], "description": st["step_description_prompt"],
                      "background": st["step_background"], "header": st["function_header"],
                      "return_line": st["return_line"], "tests": list(st["test_cases"]),
                      "reference": st.get("ground_truth_code")})
    return {"id": str(r["problem_id"]), "split": split, "name": r["problem_name"],
            "description": r["problem_description_main"], "background": r["problem_background_main"],
            "io": r["problem_io"], "dependencies": r["required_dependencies"], "steps": steps}


def problems(split: str) -> list[dict]:
    """The problems of a split ("dev" = the dataset's validation file, "test"), in file order."""
    if split not in FILES:
        raise ValueError(f"split must be one of {tuple(FILES)}")

    def build():
        from huggingface_hub import hf_hub_download
        out = []
        for sp, fname in FILES.items():
            path = hf_hub_download(data.REPOS["scicode"], fname, repo_type="dataset",
                                   revision=data.REVISIONS["scicode"])
            raw = Path(path).read_bytes()
            if git_blob_sha1(raw) != BLOB_SHA1[fname]:
                raise SystemExit(f"{fname} ne correspond pas à la version fixée du jeu de données")
            out += [_problem(json.loads(l), sp) for l in raw.decode("utf-8").splitlines() if l.strip()]
        for sp, (n_p, n_s) in COUNTS.items():
            got = [p for p in out if p["split"] == sp]
            if (len(got), sum(len(p["steps"]) for p in got)) != (n_p, n_s):
                raise SystemExit(f"SciCode {sp} : {len(got)} problèmes, "
                                 f"{sum(len(p['steps']) for p in got)} sous-problèmes ; attendus {n_p} et {n_s}")
        return out
    out = [p for p in data._cached("scicode_all_optional_reference", build, "scicode") if p["split"] == split]
    for p in out:
        for k, step in enumerate(p["steps"]):
            if is_skipped(p, step):
                chain_code(p, k, None)  # validate the supplied code before any GPU job
    return out


def h5_path() -> Path:
    p = Path(os.environ.get("SCICODE_H5") or data.DATA / H5_NAME)
    if not p.is_file():
        raise SystemExit(H5_HELP)
    return p.resolve()  # absolute: the sandbox children run in another working directory


def h5_identity(path: Path | None = None) -> dict:
    """Size and SHA-256 of the targets file, cached next to it; refuses a file other than the pinned one."""
    from .results import file_identity
    ident = file_identity(path or h5_path())
    if H5_SHA256 and ident["sha256"] != H5_SHA256:
        raise SystemExit(f"{ident['name']} : SHA-256 {ident['sha256'][:12]} au lieu de {H5_SHA256[:12]}")
    return ident


# ---------- steps and prompts ----------

def step_number(step: dict) -> int:
    """1-based position of a sub-problem in its problem ("13.6" -> 6)."""
    return int(str(step["number"]).split(".")[-1])


def is_skipped(problem: dict, step: dict) -> bool:
    return (problem["id"], step_number(step)) in SKIPPED


def def_name(header: str) -> str:
    m = _DEF.search(header)
    if not m:
        raise ValueError(f"no def or class in the header {header[:60]!r}")
    return m.group(1)


def row_id(step: dict) -> str:
    return str(step["number"])


def _step_text(step: dict, background: bool) -> str:
    s = step["description"].strip()
    if background and step["background"].strip():
        s += "\n" + step["background"].strip()
    return s


def prompt(problem: dict, k: int, chain: list[str], background: bool = True) -> str:
    """Prompt for step k (0-based). `chain[j]` is the code kept for step j < k (the model's own answers)."""
    parts = []
    for j in range(k):
        st = problem["steps"][j]
        parts.append(f"{_step_text(st, background)}\n\n{chain[j].strip()}\n\n------")
    step = problem["steps"][k]
    nxt = f"{_step_text(step, background)}\n\n{step['header'].strip()}\n\n{step['return_line'].strip()}"
    return (
        "PROBLEM DESCRIPTION\n"
        "You will be provided with the description of a scientific problem, the previous steps already solved, "
        "and the next step. Your task is to write the Python code of the next step"
        + (", after writing the disciplinary knowledge it needs as a short comment" if background else "")
        + ".\n\n"
        f"{problem['description'].strip()}\n\n"
        + (("PREVIOUS STEPS AND THEIR CODE\n" + "\n\n".join(parts) + "\n\n") if parts else "")
        + "NEXT STEP - DESCRIPTION AND FUNCTION HEADER\n"
        f"{nxt}\n\n"
        "DEPENDENCIES\n"
        "Use only the following dependencies. Do not write them at the top of your code again.\n"
        f"{problem['dependencies'].strip()}\n\n"
        "RESPONSE GUIDELINES\n"
        "Write one complete, executable Python function (or class) that follows the function header exactly, in a "
        "single ```python code block. Do not repeat the code of the previous steps, and do not write example usage "
        "or tests.")


# ---------- code extraction ----------

def extract(text: str, name: str) -> tuple[str, str]:
    """(code, status): the last fenced Python block that defines `name`; else the last Python block; else
    the whole text trimmed from its first line of code. Status: fenced, unclosed, no_entry, unfenced, empty."""
    blocks = [(m.group(1).lower(), m.group(2), m.group(3) == "```") for m in codemod._FENCE.finditer(text)]
    py = [b for b in blocks if b[0] in codemod._PY]
    if py:
        withd = [b for b in py if re.search(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b", b[1], re.M)]
        _, body, closed = (withd or py)[-1]
        status = ("fenced" if closed else "unclosed") if withd else "no_entry"
        return codemod.sanitize(body, name), status
    code_text, status = codemod.extract_code(text, name)  # unfenced text (its entry test is for def only)
    if status == "empty" and re.search(rf"^\s*class\s+{re.escape(name)}\b", text, re.M):
        start = re.search(r"^\s*class\s", text, re.M).start()
        return codemod.sanitize(text[start:], name), "unfenced"
    return code_text, status


def named_definition(code_text: str, name: str) -> str:
    """The source of the function or class `name` (what the official chain keeps from earlier steps); the whole
    code when it cannot be parsed or does not define it."""
    try:
        tree = ast.parse(code_text)
    except (SyntaxError, ValueError):
        return code_text
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            lines = code_text.splitlines()
            first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            return "\n".join(lines[first:node.end_lineno])
    return code_text


def chain_code(problem: dict, k: int, text: str | None) -> str:
    """Code kept in the chain for step k: the official code for skipped steps, else the named definition of
    the model's answer (empty if there is no answer)."""
    step = problem["steps"][k]
    if is_skipped(problem, step):
        path = SKIPPED_DIR / f"{row_id(step)}.txt"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SKIPPED_SHA256[row_id(step)]:
            raise SystemExit(f"SciCode {row_id(step)} : code officiel modifié")
        return named_definition(raw.decode("utf-8"), def_name(step["header"]))
    if not text:
        return ""
    name = def_name(step["header"])
    return named_definition(extract(text, name)[0], name)


# ---------- test scripts ----------

def test_suffix(step: dict) -> str:
    """The official test harness: targets from the HDF5 file, then each test case with its own target."""
    n = len(step["tests"])
    lines = ["", "from scicode.parse.parse import process_hdf5_to_tuple",
             f"targets = process_hdf5_to_tuple({row_id(step)!r}, {n})"]
    for i, t in enumerate(step["tests"]):
        lines += [f"target = targets[{i}]", t.strip("\n")]
    return "\n".join(lines) + "\n"


def script(problem: dict, k: int, chain: list[str], step_code: str) -> str:
    """dependencies + code of steps < k + code k + tests of step k."""
    step = problem["steps"][k]
    return ("\n".join([problem["dependencies"].strip(), *[c.strip() for c in chain[:k] if c.strip()], step_code.strip()])
            + "\n" + test_suffix(step))


def reference_script(problem: dict, k: int) -> str:
    """The same with the dataset's reference code everywhere: must pass (checks the harness, not a model)."""
    if problem["split"] not in ORACLE_SPLITS or any(not s.get("reference") for s in problem["steps"]):
        raise ValueError("SciCode oracle requires a split with reference code")
    ref = [s["reference"] for s in problem["steps"]]
    return script(problem, k, ref, ref[k])


# ---------- running a script ----------

CHILD_BODY = r'''
import types
_TMP = os.path.realpath(cfg["tmp"])
if cfg.get("h5"):
    os.makedirs(os.path.join(_TMP, "eval", "data"), exist_ok=True)
    _dst = os.path.join(_TMP, "eval", "data", "test_data.h5")  # the default path of the official parser
    try:
        os.symlink(cfg["h5"], _dst)
    except (OSError, NotImplementedError):
        import shutil
        shutil.copyfile(cfg["h5"], _dst)
try:
    import datasets  # noqa: F401  (scicode.parse imports it for a function we never call)
except Exception:
    _m = types.ModuleType("datasets")
    _m.load_dataset = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network"))
    sys.modules["datasets"] = _m
_g = {"__name__": "__main__"}
_res = {"ok": False, "err": "Crash"}
try:
    _call(lambda: exec(compile(cfg["script"], "<step>", "exec"), _g), max(1.0, cfg["cpu_s"] - 12.0))
    _res = {"ok": True}
except BaseException as _e:
    _res = {"ok": False, "err": _err(_e), "msg": str(_e)[:300]}
_emit(kind="result", **_res)
_emit(kind="done")
_out.close()
os._exit(0)
'''


def run_script(src: str, h5: str | None) -> dict:
    """Run one assembled script in a limited child process (essaim/sandbox.py: rlimits, no network where
    `unshare --net` works, a clean environment and temporary directory, memory and wall-clock limits, killed
    on shutdown). It passes only if the script reaches its end: any exception, a timeout, an exit or a crash
    fails. Unlike E11, model code and the expected targets live in ONE process (the official harness does the
    same); the audit hook and Landlock of the E11 children are not used, so run it on a throwaway VM.
    Returns {ok, err, msg, status, ms}."""
    from . import sandbox
    script_src = sandbox._PROLOGUE + CHILD_BODY
    recs, status, ms = sandbox._spawn(script_src, {"script": src, "h5": h5}, "hidden", sandboxed=True)
    r = next((x for x in recs if x.get("kind") == "result"), None)
    if r is None:
        return {"ok": False, "err": "Timeout" if status == "timeout" else "Crash", "msg": "", "status": status, "ms": ms}
    return {"ok": bool(r.get("ok")) and status == "ok", "err": r.get("err"), "msg": r.get("msg", ""),
            "status": status, "ms": ms}


def preflight() -> list[str]:
    """Problems that would make every SciCode test fail (missing packages), as messages; empty when fine."""
    out = []
    for mod in ("numpy", "scipy", "sympy", "h5py"):
        try:
            __import__(mod)
        except ImportError:
            out.append(f"module Python absent : {mod} (uv sync --group scicode)")
    try:
        import importlib.util
        if importlib.util.find_spec("scicode") is None:
            out.append("paquet scicode absent : uv pip install --no-deps "
                       f"'scicode @ git+https://github.com/scicode-bench/SciCode@{OFFICIAL_COMMIT}'")
    except (ImportError, ValueError):
        out.append("paquet scicode absent")
    return out


if __name__ == "__main__":  # python -m essaim.scicode : loads the data, prints the identity of the targets file
    for sp in FILES:
        ps = problems(sp)
        print(sp, len(ps), "problèmes,", sum(len(p["steps"]) for p in ps), "sous-problèmes")
    ident = h5_identity()
    print(json.dumps(ident))
    if H5_SHA256 is None:
        print("H5_SHA256 n'est pas fixé : reporter la valeur ci-dessus dans essaim/scicode.py", file=sys.stderr)
    for m in preflight():
        print("MANQUE :", m, file=sys.stderr)
