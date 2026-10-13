"""E18: Lean 4 proofs written by models, checked by Lean (docs/14_e18_lean.md). Pure functions, no Lean needed.

A candidate proof is trusted only through Lean: the file that Lean compiles is built HERE, never taken as is from
the model, so that the model cannot change what is proved:

    <dataset header (imports, opens, options)>
    <the model's auxiliary lemmas, if any: `theorem` / `lemma` only>
    <the dataset's formal statement, verbatim, up to its `:=`><the model's text after the `:=` of its own copy>
    #print axioms <theorem name>

Before compiling, `screen` refuses text that could change the environment or the checker itself (notations,
macros, instances, attributes, options other than a few resource limits, `#eval` and every other way to run code,
`sorry`, `axiom`...). After compiling, `verdict` accepts the proof only if Lean exited normally with no error, no
`sorry`, and the `#print axioms` line (matched by its position, so the model cannot print a fake one) lists only
the three standard axioms. The model's own copy of the statement is never compiled: if it differs from the dataset's
(beyond whitespace) the attempt is flagged `statement_rewritten` and the dataset's statement is used anyway.
"""
from __future__ import annotations

import json
import re

ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
# Options that only change resource limits (a model may write `set_option maxHeartbeats 400000 in`). Any other
# option is refused: some disable checks (e.g. debug.skipKernelTC skips the kernel).
ALLOWED_OPTIONS = frozenset({"maxHeartbeats", "maxRecDepth", "synthInstance.maxHeartbeats", "synthInstance.maxSize"})
# Whole words refused anywhere in the model's Lean code (comments and strings removed first).
FORBIDDEN_WORDS = ("sorry", "admit", "axiom", "axioms", "native_decide", "implemented_by", "extern", "unsafe",
                   "opaque", "macro", "macro_rules", "syntax", "notation", "infix", "infixl", "infixr", "prefix",
                   "postfix", "elab", "elab_rules", "instance", "attribute", "import", "namespace", "section",
                   "local", "scoped", "run_cmd", "run_tac", "run_elab", "run_meta", "initialize", "builtin_initialize",
                   "partial", "export", "universe", "variable", "open", "by_elab", "include_str", "dbg_trace",
                   "dbgTrace", "unsafeCast", "unsafeIO", "unsafeBaseIO", "unsafeEIO", "IO", "EIO", "BaseIO", "System")
FORBIDDEN_TOKENS = ("#eval", "#exit", "#print", "#reduce", "#check", "#synth", "#guard", "#help", "#lint", "#find",
                    "#where", "@[")
FORBIDDEN_RE = re.compile(r"(?<![\w.'])Lean\.")  # the metaprogramming API (Lean.Elab, Lean.Meta...)
HEADER_LINE = re.compile(r"(import|open|set_option)\s")  # dropped from the prelude: the header already has them
SORRY_WARNING = "declaration uses 'sorry'"
FENCE = re.compile(r"```[ \t]*(lean4|lean|Lean4|Lean)?[ \t]*\n(.*?)```", re.S)
DECL = re.compile(r"^(?:theorem|lemma)\s+([^\s(:{\[⦃]+)", re.M)


def strip_comments(code: str) -> str:
    """Lean source with comments (`--` to end of line, nested `/- -/`) and string literals blanked out (replaced by
    spaces, newlines kept) so that a forbidden word inside a comment is not a match and positions are unchanged."""
    out, i, n, depth = [], 0, len(code), 0
    while i < n:
        c = code[i]
        if depth:
            if code.startswith("/-", i):
                depth, step = depth + 1, 2
            elif code.startswith("-/", i):
                depth, step = depth - 1, 2
            else:
                step = 1
            out.append("".join(ch if ch == "\n" else " " for ch in code[i:i + step]))
            i += step
        elif code.startswith("/-", i):
            depth = 1
            out.append("  ")
            i += 2
        elif code.startswith("--", i):
            j = code.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == '"':
            j, esc = i + 1, False
            while j < n and (esc or code[j] != '"'):
                esc = code[j] == "\\" and not esc
                j += 1
            j = min(j + 1, n)
            out.append("".join(ch if ch == "\n" else " " for ch in code[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def theorem_name(statement: str) -> str:
    m = DECL.search(statement)
    if not m:
        raise ValueError("énoncé sans `theorem` ni `lemma`")
    return m.group(1)


def assign_index(code: str, start: int = 0) -> int:
    """Index of the first `:=` at bracket depth 0 from `start` (comments and strings ignored), or -1."""
    clean = strip_comments(code)
    depth = 0
    for i in range(start, len(clean) - 1):
        c = clean[i]
        if c in "([{⦃⟨":
            depth += 1
        elif c in ")]}⦄⟩":
            depth = max(0, depth - 1)
        elif depth == 0 and clean.startswith(":=", i):
            return i
    return -1


def statement_prefix(statement: str) -> str:
    """The dataset statement up to and including its `:=` (a trailing `by sorry`, `sorry` or `by` is dropped)."""
    k = assign_index(statement)
    if k < 0:
        raise ValueError("énoncé sans `:=`")
    return statement[:k + 2]


def normalize_ws(text: str) -> str:
    return " ".join(strip_comments(text).split())


def last_code_block(text: str) -> str | None:
    """The last ```lean4 / ```lean block of a completion (provers end with the final code), else the last untagged
    block, or None."""
    blocks = FENCE.findall(text or "")  # (language, body)
    tagged = [b for lang, b in blocks if lang]
    if tagged:
        return tagged[-1]
    return blocks[-1][1] if blocks else None


def split_candidate(block: str, name: str) -> dict:
    """Split the model's code block into auxiliary lemmas (`prelude`), its copy of the statement and the text after
    the statement's `:=` (`body`). The LAST declaration of the theorem's name is the candidate (an earlier one stays
    in the prelude, where `build` refuses it)."""
    clean = strip_comments(block)
    found = list(re.finditer(rf"(?m)^[ \t]*(?:theorem|lemma)\s+{re.escape(name)}(?=[\s(:{{\[⦃])", clean))
    m = found[-1] if found else None
    if not m:
        return {"status": "no_theorem"}
    k = assign_index(block, m.end())
    if k < 0:
        return {"status": "no_theorem"}
    # The header's imports, opens and options are ours: the model's own lines of that kind are dropped (dropping one
    # can only make the file fail, never make a false statement pass).
    prelude_lines = [l for l in block[:m.start()].splitlines() if not HEADER_LINE.match(l)]
    return {"status": "ok", "prelude": "\n".join(prelude_lines).strip("\n"), "statement": block[m.start():k + 2],
            "body": block[k + 2:].rstrip()}


def _top_level_ok(prelude: str) -> bool:
    """Restrict the prelude to auxiliary proofs and their continuations, regardless of command indentation."""
    clean = strip_comments(prelude)
    # These declaration keywords cannot occur in a proof continuation. Lean commands may be indented even
    # further than the preceding proof, so indentation alone cannot distinguish a continuation from a command.
    if re.search(r"(?<![\w.'])(?:def|abbrev|constant|constants|structure|class|inductive|coinductive|"
                 r"noncomputable|mutual|end|example)(?![\w'])", clean):
        return False
    seen_proof = False
    for line in clean.splitlines():
        if not line.strip():
            continue
        if re.match(r"(theorem|lemma)\s", line.lstrip()):
            seen_proof = True
        elif not seen_proof or not line[0].isspace():
            return False
    return True


def screen(code: str) -> str | None:
    """Reason to refuse the model's Lean text before compiling it, or None. Applied to prelude and body."""
    clean = strip_comments(code)
    for m in re.finditer(r"set_option\s+([\w.]+)", clean):
        if m.group(1) not in ALLOWED_OPTIONS:
            return f"option interdite : {m.group(1)}"
    for tok in FORBIDDEN_TOKENS:
        if tok in clean:
            return f"interdit : {tok}"
    if FORBIDDEN_RE.search(clean):
        return "interdit : Lean."
    for w in FORBIDDEN_WORDS:
        if re.search(rf"(?<![\w.']){re.escape(w)}(?![\w'])", clean):
            return f"interdit : {w}"
    return None


def build(problem: dict, completion: str) -> dict:
    """The file to compile for one completion, or why there is none. Returns status (ok, no_code, no_theorem,
    forbidden), code (what Lean compiles, without the axioms line), axioms_line (1-based line of `#print axioms`),
    statement_rewritten, reason."""
    block = last_code_block(completion)
    if block is None:
        return {"status": "no_code"}
    name = theorem_name(problem["formal_statement"])
    part = split_candidate(block, name)
    if part["status"] != "ok":
        return {"status": part["status"]}
    for piece in (part["prelude"], part["body"]):
        reason = screen(piece)
        if reason:
            return {"status": "forbidden", "reason": reason}
    if part["prelude"] and not _top_level_ok(part["prelude"]):
        return {"status": "forbidden", "reason": "commande auxiliaire autre que theorem/lemma"}
    if re.search(rf"(?<![\w.']){re.escape(name)}(?![\w'])", strip_comments(part["prelude"])):
        return {"status": "forbidden", "reason": "le prélude redéfinit le théorème"}
    prefix = statement_prefix(problem["formal_statement"])
    code = problem["header"].rstrip("\n") + "\n\n"
    if part["prelude"]:
        code += part["prelude"] + "\n\n"
    code += prefix + part["body"] + "\n"
    rewritten = normalize_ws(part["statement"]) != normalize_ws(prefix)
    return {"status": "ok", "code": code, "axioms_line": code.count("\n") + 2, "statement_rewritten": rewritten,
            "name": name}


def oracle_file(problem: dict) -> dict:
    """The dataset statement closed by `sorry` (setup check: every statement must elaborate)."""
    code = problem["header"].rstrip("\n") + "\n\n" + statement_prefix(problem["formal_statement"]) + " by\n  sorry\n"
    return {"code": code, "axioms_line": code.count("\n") + 2, "name": theorem_name(problem["formal_statement"])}


def with_axioms(code: str, name: str) -> str:
    """The compiled text: `code` then, on the line `axioms_line` of `build`, the axioms query."""
    return code + "\n#print axioms " + name + "\n"


def parse_messages(stdout: str) -> list[dict]:
    """Messages of `lean --json` (one JSON object per line); a non-JSON line becomes a pseudo error so that nothing
    unexpected is ever ignored."""
    out = []
    for line in (stdout or "").splitlines():
        if not line.strip():
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            out.append({"severity": "error", "line": None, "col": None, "end_line": None, "end_col": None,
                        "data": "sortie non JSON : " + line[:300]})
            continue
        if not isinstance(m, dict):
            out.append({"severity": "error", "line": None, "col": None, "end_line": None, "end_col": None,
                        "data": "message inattendu"})
            continue
        pos, end = m.get("pos") or {}, m.get("endPos") or {}
        out.append({"severity": str(m.get("severity", "error")), "line": pos.get("line"), "col": pos.get("column"),
                    "end_line": end.get("line"), "end_col": end.get("column"), "data": str(m.get("data", ""))})
    return out


AXIOMS_RE = re.compile(r"^'(?P<name>[^']+)' (?:depends on axioms: \[(?P<list>[^\]]*)\]|does not depend on any axioms)")


def axioms_of(messages: list[dict], name: str, line: int) -> list[str] | None:
    """Axioms reported by OUR `#print axioms` (the info message at `line`), or None if it is missing."""
    for m in messages:
        if m["line"] == line and m["severity"] == "information":
            a = AXIOMS_RE.match(m["data"].strip())
            if a and a.group("name") == name:
                return [x.strip() for x in (a.group("list") or "").split(",") if x.strip()]
    return None


def messages_of(run: dict) -> list[dict]:
    """The messages of a run: parsed from `stdout`, or already parsed (`messages`, as stored by run_e18.py)."""
    return run["messages"] if "messages" in run else parse_messages(run.get("stdout", ""))


def verdict(run: dict, name: str, axioms_line: int) -> dict:
    """Decide from one Lean run: {"status", "verified", "axioms"}. `run`: returncode, timeout (bool), memory (bool),
    overflow (bool), and stdout or messages. Order: resource failures, errors, exit code, sorry, then the axioms."""
    if run.get("timeout"):
        return {"status": "timeout", "verified": False, "axioms": None}
    if run.get("memory"):
        return {"status": "memory", "verified": False, "axioms": None}
    if run.get("overflow"):
        return {"status": "overflow", "verified": False, "axioms": None}
    msgs = messages_of(run)
    if any(m["severity"] == "error" for m in msgs):
        return {"status": "lean_error", "verified": False, "axioms": None}
    if run.get("returncode") != 0:
        return {"status": "crash", "verified": False, "axioms": None}
    if any(SORRY_WARNING in m["data"] for m in msgs):
        return {"status": "sorry", "verified": False, "axioms": None}
    ax = axioms_of(msgs, name, axioms_line)
    if ax is None:
        return {"status": "no_axioms_report", "verified": False, "axioms": None}
    if not set(ax) <= ALLOWED_AXIOMS:
        return {"status": "bad_axioms", "verified": False, "axioms": ax}
    return {"status": "verified", "verified": True, "axioms": ax}


def oracle_verdict(run: dict, name: str, axioms_line: int) -> dict:
    """The statement-with-sorry check: elaborates (no error) and depends on sorryAx (so the statement is used)."""
    if run.get("timeout") or run.get("memory") or run.get("overflow"):
        return {"ok": False, "status": "resources"}
    msgs = messages_of(run)
    if any(m["severity"] == "error" for m in msgs) or run.get("returncode") != 0:
        return {"ok": False, "status": "lean_error"}
    ax = axioms_of(msgs, name, axioms_line)
    ok = ax is not None and "sorryAx" in ax
    return {"ok": ok, "status": "ok" if ok else "no_sorry"}


def feedback(run: dict, code: str, max_errors: int = 8, max_chars: int = 6000) -> str:
    """Lean's errors for the revision prompt, in the format of Goedel-Prover-V2's self-correction (its
    src/utils.py get_error_str, Apache-2.0, re-implemented): per error, up to four lines before it, the error span
    between <error></error>, the next line, then the message; at most `max_errors` errors, `max_chars` in all."""
    lines = code.split("\n")
    errors = [m for m in messages_of(run) if m["severity"] == "error"]
    out = []
    for k, m in enumerate(errors[:max_errors], 1):
        line, col = m.get("line"), m.get("col")
        if not isinstance(line, int) or not 0 < line <= len(lines) or not isinstance(col, int):
            out.append(f"\nError {k}:\n\nError Message: {m['data'].strip()}\n")
            continue
        end_line, end_col = m.get("end_line"), m.get("end_col")
        if not isinstance(end_line, int) or not line <= end_line <= len(lines) or not isinstance(end_col, int):
            end_line, end_col = line, len(lines[line - 1])
        a, b = line - 1, end_line - 1
        snippet = [lines[j] for j in range(max(0, a - 4), a)]
        if a == b:
            t = lines[a]
            snippet.append(t[:col] + "<error>" + t[col:end_col] + "</error>" + t[end_col:])
        else:
            snippet.append(lines[a][:col] + "<error>" + lines[a][col:])
            snippet += lines[a + 1:min(b, a + 6)]
            if b > a + 6:
                snippet.append("... --[Truncated]-- ...")
            snippet.append(lines[b][:end_col] + "</error>" + lines[b][end_col:])
        if b + 1 < len(lines):
            snippet.append(lines[b + 1])
        body = "\n".join(snippet)
        out.append(f"\nError {k}:\n\nCorresponding Code:\n```lean4\n{body}\n```\n\nError Message: {m['data'].strip()}\n")
    if len(errors) > max_errors:
        out.append(f"\n... [Omitted {len(errors) - max_errors} more errors] ...\n")
    text = "".join(out) if out else "The proof was not accepted (no error message)."
    return text if len(text) <= max_chars else text[:max_chars] + "\n[...]"
