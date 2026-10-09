"""E11 (code generation with selection by execution): prompts, code extraction, visible tests, hidden tests,
extra inputs and output signatures. Pure parsing: nothing here executes model code (see essaim/sandbox.py).

Visible tests are the examples shown in the prompt: the docstring examples of HumanEval (parsed below,
several formats) and the asserts of MBPP. Hidden tests are the EvalPlus "plus" tests (base + plus inputs),
used only for grading. Extra inputs, for functional clustering, are derived deterministically from the
visible inputs by small type-preserving mutations (seeded by the problem id).
"""
from __future__ import annotations

import ast
import hashlib
import random
import re
import symtable
import textwrap
from pathlib import Path

PROMPT_VERSION = "code-v1"
EXTRACT_VERSION = "extract-v2"  # v2: initialisations the solution uses (solver = Solver()) are kept
# v4: visible tests replayed by the trusted grader on the candidate's outputs, typed output signatures
TESTS_VERSION = "tests-v4"  # visible-test parsing, hidden-test parsing, extra-input generation, signatures
BENCHES = ("humanevalplus", "mbppplus")
RESULTS = Path(__file__).resolve().parent.parent / "results"
MAX_TOKENS = 1024
GREEDY_SEED = 7
SAMPLING = {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.05}  # llama.cpp's defaults, written out


def sample_seed(sample: int) -> int:
    """Sample 0 is greedy; sample s >= 1 is drawn at SAMPLING with this fixed seed."""
    return GREEDY_SEED if sample == 0 else 1000 + sample


def gen_path(model: str, suffix: str, bench: str, split: str) -> Path:
    return RESULTS / f"code_{model.replace('/', '__')}{suffix}_{bench}_{split}.jsonl"


def exec_path(tag: str, suffix: str, bench: str, split: str) -> Path:
    return RESULTS / f"codeexec_{tag}{suffix}_{bench}_{split}.jsonl"


# ---------- prompts ----------

def prompt(bench: str, item: dict) -> str:
    if bench == "humanevalplus":
        return ("Complete the following Python function.\n\n```python\n" + item["prompt"].rstrip() + "\n```\n\n"
                "Write the complete function (signature included), with the imports and helper functions it "
                "needs, in a single ```python code block. Do not include tests or example usage.")
    tests = "\n".join(item["test_list"])
    return (item["prompt"].strip() + "\n\nYour code should pass these tests:\n\n```python\n" + tests + "\n```\n\n"
            "Write the Python function, with the imports it needs, in a single ```python code block. "
            "Do not include tests or example usage.")


def entry_point(bench: str, item: dict) -> str:
    if bench == "humanevalplus":
        return item["entry_point"]
    loop = ast.parse(item["test"]).body[-1]
    for node in ast.walk(loop):  # assertion(NAME(*inp), exp, atol): the first call taking *inp
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in ("assertion", "ref_func") \
                and any(isinstance(a, ast.Starred) for a in node.args):
            return node.func.id
    raise ValueError(f"{item['id']}: entry point not found in the hidden tests")


def header(bench: str, item: dict) -> str:
    """Code run before the candidate. HumanEval: the whole prompt (imports, helper functions, and a stub of the
    entry point that the candidate redefines; a docstring-only function is valid Python). MBPP: the test
    imports."""
    if bench == "humanevalplus":
        return item["prompt"]
    return "\n".join(item["test_imports"])


# ---------- code extraction ----------

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.\-]*)[^\n]*\n(.*?)(```|\Z)", re.S)
_PY = {"", "python", "py", "python3", "py3"}


def _defines(code: str, entry: str) -> bool:
    return re.search(rf"^\s*(async\s+)?def\s+{re.escape(entry)}\s*\(", code, re.M) is not None


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def extract_code(text: str, entry: str) -> tuple[str, str]:
    """(code, status). The last fenced Python block that defines the entry point; else the first Python block;
    else, without fences, the text from its first line of code, trimmed until it parses. Status: fenced,
    unclosed (truncated block), unfenced, no_entry (no block defines the entry point), empty."""
    blocks = [(m.group(1).lower(), m.group(2), m.group(3) == "```") for m in _FENCE.finditer(text)]
    py = [b for b in blocks if b[0] in _PY]
    if py:
        with_entry = [b for b in py if _defines(b[1], entry)]
        lang, code, closed = with_entry[-1] if with_entry else py[0]
        status = ("fenced" if closed else "unclosed") if with_entry else "no_entry"
        return sanitize(code, entry), status
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"\s*(def |async def |class |import |from \S+ import )", l)), None)
    if start is None:
        return "", "empty"
    body = lines[start:]
    for end in range(len(body), 0, -1):  # drop trailing prose until the rest parses
        code = "\n".join(body[:end])
        if _parses(code):
            break
    else:
        code = "\n".join(body)
    return sanitize(code, entry), "unfenced" if _defines(code, entry) else "no_entry"


def sanitize(code: str, entry: str) -> str:
    """Keep what defines the solution: imports, functions, classes, constant assignments, try-imports and
    sys.setrecursionlimit; drop top-level tests and example usage (prints, asserts, loops, __main__ blocks,
    assignments that call a function of the code). An assignment that calls a function of the code is kept
    when what it assigns may be read by the code that is kept (`solver = Solver()` read by the entry point:
    an initialisation, not an example), inside `try: ... except Exception: pass` so that an example kept by
    a name collision cannot break the program. Unparseable code is returned unchanged (it will fail)."""
    code = textwrap.dedent(code).strip("\n")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

    def calls_defined(node) -> bool:
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in defined
                   for c in ast.walk(node))

    def assigned(node) -> set[str]:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name)}

    def class_reads(cls: ast.ClassDef) -> set[str]:
        """Names a class may read from the globals, outside its methods' bodies (left to the symbol table):
        statement by statement, a name counts unless an earlier UNCONDITIONAL statement of the class body
        bound it (assignment, def, class, import; `del` unbinds); conditional bindings do not count. Inside a
        comprehension (but its first iterable) and a lambda body, class names are invisible: only their own
        targets and parameters are local. Decorators, bases, default values and annotations are read where
        they appear. When in doubt a name counts: kept initialisations are guarded (see below)."""
        out = set()

        def targets(t) -> set[str]:
            return {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}

        def same_scope(stmt):
            """The nodes of a class-body statement that run in the class scope (not inside a def, a lambda, a
            nested class or a comprehension)."""
            todo = [stmt]
            while todo:
                n = todo.pop()
                yield n
                if n is not stmt and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                                                    ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                    continue
                if n is stmt and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                todo.extend(ast.iter_child_nodes(n))

        def loads(n, bound: set[str]):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):  # body: the symbol table
                a = n.args
                args = a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]
                for x in n.decorator_list + a.defaults + [d for d in a.kw_defaults if d] + \
                        [x.annotation for x in args if x.annotation] + ([n.returns] if n.returns else []):
                    loads(x, bound)
                return
            if isinstance(n, ast.ClassDef):  # a nested class: its own body is read on its own
                for x in n.decorator_list + n.bases + [k.value for k in n.keywords]:
                    loads(x, bound)
                out.update(class_reads(n))
                return
            if isinstance(n, ast.Lambda):
                for d in n.args.defaults + [d for d in n.args.kw_defaults if d]:
                    loads(d, bound)
                a = n.args
                loads(n.body, {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]})
                return
            if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                gens = n.generators
                loads(gens[0].iter, bound)  # evaluated in the class scope
                inner = set().union(*(targets(g.target) for g in gens))
                for k, g in enumerate(gens):
                    if k:
                        loads(g.iter, inner)
                    for c in g.ifs:
                        loads(c, inner)
                for e in ([n.key, n.value] if isinstance(n, ast.DictComp) else [n.elt]):
                    loads(e, inner)
                return
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound:
                out.add(n.id)
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id not in bound:
                out.add(n.target.id)  # x += 1 reads x
            for c in ast.iter_child_nodes(n):
                loads(c, bound)

        bound: set[str] = set()
        for stmt in cls.body:
            loads(stmt, bound)
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    bound |= targets(t)
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) and (getattr(stmt, "value", None) is not None):
                bound |= targets(stmt.target)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(stmt.name)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                bound |= {(al.asname or al.name).split(".")[0] for al in stmt.names}
            # whatever a statement may unbind in the class scope, at any depth of its blocks (`if ...: del x`,
            # `except E as x`), is no longer bound; deletions inside a method or a nested scope do not count
            for x in same_scope(stmt):
                if isinstance(x, ast.Delete):
                    for t in x.targets:
                        bound -= targets(t)
                elif isinstance(x, ast.ExceptHandler) and x.name:
                    bound.discard(x.name)
        return out

    def used(nodes) -> set[str]:
        """Module-level names the nodes may read: references at top level, the GLOBAL references of their
        functions (a local variable of a function that happens to share a name is not one), and every name
        their classes read outside method bodies."""
        out = set()

        def scan(t):
            for s in t.get_symbols():
                if s.is_referenced() and (t.get_type() == "module" or s.is_global()):
                    out.add(s.get_name())
            for c in t.get_children():
                scan(c)

        for node in nodes:
            try:
                scan(symtable.symtable(ast.unparse(node), "<sanitize>", "exec"))
            except SyntaxError:  # cannot happen for parsed code; conservative fallback
                out |= {x.id for x in ast.walk(node) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
            for c in ast.walk(node):
                if isinstance(c, ast.ClassDef):
                    out |= class_reads(c)
        return out

    keep, maybe = set(), []
    for i, n in enumerate(tree.body):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ok = True
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            ok = not calls_defined(n)
            if not ok:
                maybe.append(i)
        elif isinstance(n, ast.Try):
            ok = all(isinstance(s, (ast.Import, ast.ImportFrom, ast.Pass)) for s in n.body)
        elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            ok = ast.unparse(n.value.func) == "sys.setrecursionlimit"
        else:
            ok = False
        if ok:
            keep.add(i)
    changed = True
    while changed:  # an initialisation can itself be needed by another one
        need = used(tree.body[i] for i in keep)
        changed = False
        for i in maybe:
            if i not in keep and assigned(tree.body[i]) & need:
                keep.add(i)
                changed = True
    lines = code.splitlines()
    out = []
    for i in sorted(keep):
        n = tree.body[i]
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        src = "\n".join(lines[start - 1: n.end_lineno])
        if i in maybe:  # a kept call of the code's own functions: an initialisation, or an example whose name
            # collides with one the code may read; guarded, so that a failing example never breaks the load
            # (built from the AST: re-indenting the text would change multi-line string literals)
            guard = ast.Try(body=[n], handlers=[ast.ExceptHandler(type=ast.Name("Exception", ast.Load()), name=None,
                                                                  body=[ast.Pass()])], orelse=[], finalbody=[])
            src = ast.unparse(ast.fix_missing_locations(guard))
        out.append(src)
    return "\n\n".join(out)


def prog_id(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def normalized_text(code: str) -> str:
    """Text identity for the text-vote baseline: the AST without docstrings, comments or formatting."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return " ".join(code.split())
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            n.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


# ---------- literals ----------

def _literal(src: str):
    """A Python literal parsed safely; raises ValueError if it is not one, or does not round-trip."""
    v = ast.literal_eval(src.strip())
    if not _roundtrips(v):
        raise ValueError("not a round-trippable literal")
    return v


def _roundtrips(v) -> bool:
    try:
        return ast.literal_eval(repr(v)) == v and repr(ast.literal_eval(repr(v))) == repr(v)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False


def _leading_literal(s: str):
    """The longest prefix of s that is a literal (expected values are often followed by a comment)."""
    s = s.strip().rstrip(".").strip()
    first = s.split()[0] if s.split() else ""
    if first.rstrip(".,") in ("true", "false"):  # prose spelling of booleans (HumanEval/76)
        return first.rstrip(".,") == "true"
    cuts = sorted({len(s)} | {i for i, ch in enumerate(s) if ch in " #(,;"}, reverse=True)
    for k in cuts:
        try:
            return _literal(s[:k].rstrip().rstrip(".").rstrip())
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    raise ValueError("no literal")


def _call_args(call_src: str, entry: str) -> tuple | None:
    """Literal positional arguments of `entry(...)`, or None."""
    try:
        node = ast.parse(call_src.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry) or node.keywords:
        return None
    try:
        args = tuple(ast.literal_eval(a) for a in node.args)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return args if _roundtrips(args) else None


def _balanced_call(line: str, start: int) -> int | None:
    """Index just after the parenthesis closing the call that opens at line[start:] (quotes respected)."""
    depth, quote, i = 0, None, start
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


_SEPS = ("==>", "==", "➞", "=>", "->", "→", "# returns", "# =>", "#=>", "# ->", "should return", "returns", "return",
         "=", "#")


# ---------- visible tests ----------

def _params(src: str, entry: str) -> list[str] | None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry:
            return [a.arg for a in n.args.posonlyargs + n.args.args]
    return None


def _docstring(src: str, entry: str) -> str:
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry:
            return ast.get_docstring(n, clean=True) or ""
    return ""


def _named_args(s: str, params: list[str]) -> tuple | None:
    """'lst = [1,2], k = 3' (or a bare value for a one-parameter function) -> arguments in parameter order."""
    s = s.strip().rstrip(",").strip()
    parts = re.split(r",\s*(?=[A-Za-z_]\w*\s*[=:](?!=))", s)
    named = {}
    for p in parts:
        m = re.match(r"([A-Za-z_]\w*)\s*[=:](?!=)\s*(.+)$", p.strip(), re.S)
        if not m:
            named = None
            break
        try:
            named[m.group(1)] = _literal(m.group(2))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None
    if named is not None and set(named) == set(params):
        return tuple(named[p] for p in params)
    if named is None and len(params) == 1:
        try:
            return (_literal(s),)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None
    return None


def humaneval_examples(item: dict) -> list[tuple[tuple, object]]:
    """(arguments, expected output) pairs read from the entry point's docstring. Formats: doctest
    (`>>> f(x)` then the output), `f(x) == y` and its variants (➞, =>, ->, =, # returns...), `For x = ... the
    output should be y`, and `Input: x = ...` / `Output: y` blocks. Only literal arguments and outputs."""
    entry = item["entry_point"]
    params = _params(item["prompt"], entry) or []
    doc = _docstring(item["prompt"], entry)
    lines = doc.splitlines()
    out: list[tuple[tuple, object]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        doctest = line.startswith(">>>")
        if doctest:
            line = line[3:].strip()
        k = line.find(entry + "(")
        if k >= 0 and (k == 0 or not (line[k - 1].isalnum() or line[k - 1] == "_")) and (not doctest or k == 0):
            end = _balanced_call(line, k + len(entry))
            args = _call_args(line[k:end], entry) if end else None
            rest = line[end:].strip() if end else ""
            j = i + 1
            if doctest and not rest:  # doctest: the output is on the next lines
                exp_lines = []
                while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(">>>"):
                    exp_lines.append(lines[j].strip())
                    j += 1
                rest = " ".join(exp_lines)
            elif rest:
                sep = next((s for s in _SEPS if rest.startswith(s)), None)
                rest = rest[len(sep):] if sep else (rest if doctest else "")  # `>>> f(x) [0, 1]`: no separator
            if args is not None and rest:
                try:
                    out.append((args, _leading_literal(rest)))
                except ValueError:
                    pass
            i = j
            continue
        m = re.match(r"For (.+?),? the (?:output|result) should be (.+)$", line)
        if m and params:
            args = _named_args(m.group(1), params)
            if args is not None:
                try:
                    out.append((args, _leading_literal(m.group(2))))
                except ValueError:
                    pass
            i += 1
            continue
        m = re.match(r"Input\s*:\s*(.*)$", line)
        if m and params:
            j, inp = i + 1, m.group(1)
            while j < len(lines) and not lines[j].strip().startswith("Output"):
                if not lines[j].strip():
                    break
                inp += " " + lines[j].strip()
                j += 1
            if j < len(lines) and lines[j].strip().startswith("Output"):
                args = _named_args(inp, params) if inp.strip() else None
                mo = re.match(r"Output\s*:\s*(.*)$", lines[j].strip())
                if args is not None and mo:
                    try:
                        out.append((args, _leading_literal(mo.group(1))))
                    except ValueError:
                        pass
                i = j + 1
                continue
        i += 1
    seen, uniq = set(), []
    for a, e in out:
        key = repr((a, e))
        if key not in seen:
            seen.add(key)
            uniq.append((a, e))
    return uniq


def visible_tests(bench: str, item: dict) -> list[str]:
    """Assert statements, run in the candidate's namespace (the sandbox provides `_close`, a comparison with
    float tolerance, for the HumanEval examples)."""
    if bench == "humanevalplus":
        entry = item["entry_point"]
        return [f"assert _close({entry}(*{a!r}), {e!r})" for a, e in humaneval_examples(item)]
    return list(item["test_list"])


def visible_inputs(bench: str, item: dict) -> list[tuple]:
    if bench == "humanevalplus":
        return [a for a, _ in humaneval_examples(item)]
    entry = entry_point(bench, item)
    out = []
    for t in item["test_list"]:
        try:
            tree = ast.parse(t)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == entry and not n.keywords:
                args = _call_args(ast.unparse(n), entry)
                if args is not None and repr(args) not in {repr(x) for x in out}:
                    out.append(args)
    return out


# ---------- hidden tests ----------

def hidden_spec(bench: str, item: dict) -> dict:
    """The EvalPlus test file split into: prelude (helpers: assertion, ref_func, imports), setup (the `inputs`
    and `results` assignments), and the final loop (target, iterable, body), so the sandbox can run each case
    separately (deep-copied inputs, one timeout per case) and stop at the first failure."""
    src = item["test"]
    tree = ast.parse(src)
    seg = lambda n: ast.get_source_segment(src, n)
    if bench == "humanevalplus":
        check = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"]
        if len(check) != 1:
            raise ValueError(f"{item['id']}: no check()")
        prelude = [n for n in tree.body if n is not check[0]]
        setup, loop = check[0].body[:-1], check[0].body[-1]
    else:
        prelude = [n for n in tree.body[:-1] if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and
                   isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("inputs", "results"))]
        setup = [n for n in tree.body[:-1] if n not in prelude]
        loop = tree.body[-1]
    if not isinstance(loop, ast.For) or not all(isinstance(s, ast.Assign) for s in setup):
        raise ValueError(f"{item['id']}: unexpected hidden-test layout")
    body = ast.unparse(loop.body)
    parts = []
    for n in prelude:
        if isinstance(n, ast.FunctionDef) and n.name == "assertion" and \
                not any(isinstance(x, ast.Assert) for x in ast.walk(n)) and "exact_match" in ast.unparse(n):
            # Mbpp/737, 787, 794: the exported helper computes exact_match but never asserts it, so every program
            # that returns would pass; EvalPlus asserts it. Repaired here.
            n = ast.parse(ast.unparse(n) + "\n    assert exact_match").body[0]
            parts.append(ast.unparse(n))
        else:
            parts.append(seg(n))
    if item["id"] == "HumanEval/32":
        # The exported test calls _poly(*candidate(*inp), inp), which raises for every program (the reference
        # included). The intended oracle is a root check, |poly(xs, x)| <= 1e-4; but neither the reference nor
        # even the stored expected outputs pass it on every input (badly conditioned polynomials), so the
        # reference fails here and the analysis leaves this problem out (reported), like any problem whose
        # reference solution fails our grading.
        body = "assert abs(_poly(*inp, candidate(*inp))) <= 0.0001"
    return {"prelude": "\n".join(parts), "setup": [seg(n) for n in setup],
            "target": seg(loop.target), "iter": seg(loop.iter), "body": body}


# ---------- extra inputs ----------

def _mutate(v, rng: random.Random, depth: int = 0):
    """One small change of a literal value that keeps its type AND its apparent domain, so that the new input
    most likely still meets the problem's preconditions (an input outside them splits correct programs, which
    may legitimately differ there): numbers keep their sign (zero stays non-negative), strings keep their
    alphabet (characters are only moved, dropped or repeated) and are never emptied, sequences are never
    emptied, dicts keep their keys."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        if v == 0:
            return rng.choice([1, 2])
        s, m = (1 if v > 0 else -1), abs(v)
        return s * rng.choice([m + 1, m + 2, max(1, m - 1), m + rng.randrange(3, 10), max(1, m // 2)])
    if isinstance(v, float):
        if v == 0.0:
            return rng.choice([0.5, 1.5])
        return rng.choice([v * 1.5, v / 2, round(v * 0.7, 3), v + (0.5 if v > 0 else -0.5)])
    if isinstance(v, str):
        if not v:
            return v
        op = rng.choice(["rev", "dup", "repl", "shuffle"] + (["drop"] if len(v) > 1 else []))
        i = rng.randrange(len(v))
        if op == "rev":
            return v[::-1]
        if op == "drop":
            return v[:i] + v[i + 1:]
        if op == "dup":
            return v[:i] + v[i] + v[i:]
        if op == "shuffle":
            chars = list(v)
            rng.shuffle(chars)
            return "".join(chars)
        return v[:i] + rng.choice(v) + v[i + 1:]
    if isinstance(v, (list, tuple)):
        seq = list(v)
        if not seq:
            return v
        op = rng.choice(["rev", "dup", "elem", "swap", "sort", "append"] + (["drop"] if len(seq) > 1 else []))
        i = rng.randrange(len(seq))
        if op == "rev":
            seq = seq[::-1]
        elif op == "drop":
            seq = seq[:i] + seq[i + 1:]
        elif op == "dup":
            seq = seq[:i] + [seq[i]] + seq[i:]
        elif op == "elem" and depth < 3:
            seq[i] = _mutate(seq[i], rng, depth + 1)
        elif op == "swap":
            j = rng.randrange(len(seq))
            seq[i], seq[j] = seq[j], seq[i]
        elif op == "sort":
            try:
                seq = sorted(seq)
            except TypeError:
                seq = seq[::-1]
        elif op == "append" and depth < 3:
            seq = seq + [_mutate(seq[i], rng, depth + 1)]
        return tuple(seq) if isinstance(v, tuple) else seq
    if isinstance(v, dict) and v and depth < 3:
        d = dict(v)
        k = rng.choice(sorted(d, key=repr))
        d[k] = _mutate(d[k], rng, depth + 1)
        return d
    if isinstance(v, (set, frozenset)) and len(v) > 1:
        s = set(v)
        s.discard(rng.choice(sorted(s, key=repr)))
        return type(v)(s)
    return v


_TYPED = {"str": lambda r: "".join(r.choice("abcdefgh") for _ in range(r.randrange(1, 8))),
          "int": lambda r: r.randrange(1, 30), "float": lambda r: round(r.uniform(0.1, 10), 2),
          "bool": lambda r: r.random() < 0.5}


def _typed_value(ann: str, rng: random.Random):
    ann = ann.replace("typing.", "").replace(" ", "")
    if ann in _TYPED:
        return _TYPED[ann](rng)
    m = re.match(r"(?:List|list)\[(\w+)\]$", ann)
    if m and m.group(1) in _TYPED:
        return [_TYPED[m.group(1)](rng) for _ in range(rng.randrange(1, 7))]
    raise KeyError(ann)


def _typed_inputs(item: dict, n: int, rng: random.Random) -> list[tuple]:
    """No example to mutate (a few HumanEval problems): values drawn from the entry point's annotations."""
    try:
        tree = ast.parse(item.get("prompt", ""))
    except SyntaxError:
        return []
    fn = next((x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
               and x.name == item.get("entry_point")), None)
    if fn is None:
        return []
    anns = [ast.unparse(a.annotation) if a.annotation is not None else "" for a in fn.args.posonlyargs + fn.args.args]
    out = []
    try:
        for _ in range(n):
            out.append(tuple(_typed_value(a, rng) for a in anns))
    except KeyError:
        return []
    return out


def extra_inputs(bench: str, item: dict, n: int = 16) -> list[tuple]:
    """The visible inputs, then n new argument tuples derived from them by one or two small mutations
    (deterministic: seeded by the problem id). Every value is a literal that round-trips through repr."""
    rng = random.Random(int(hashlib.sha256(item["id"].encode()).hexdigest()[:16], 16))
    base = visible_inputs(bench, item)
    seen = {repr(a) for a in base}
    out = list(base)
    pool = base or _typed_inputs(item, 4 * n, rng)
    if not pool:
        return out
    if not base:  # drawn from annotations: already new inputs
        for a in pool:
            if repr(a) not in seen and _roundtrips(a) and len(out) < n:
                seen.add(repr(a))
                out.append(a)
        return out
    for _ in range(40 * n):
        if len(out) >= len(base) + n:
            break
        args = list(rng.choice(pool))
        if not args:
            break
        for _k in range(1 if rng.random() < 0.5 else 2):
            p = rng.randrange(len(args))
            args[p] = _mutate(args[p], rng)
        t = tuple(args)
        r = repr(t)
        if r not in seen and len(r) < 20000 and _roundtrips(t):
            seen.add(r)
            out.append(t)
    return out
