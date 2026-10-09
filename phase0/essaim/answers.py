"""Answer extraction and grading for generated (not scored) answers: GSM8K, MATH-500, multiple choice.

Votes compare the NORMALISED answers, so two peers agree when their normalised answers are equal.
MATH normalisation follows Hendrycks et al.'s `strip_string` (the usual MATH grader), plus a numeric
comparison when both sides parse as numbers.
"""
from __future__ import annotations

import re

from .common import _END_RE, FINAL_RE, visible_answer_text


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}), with nested braces."""
    i = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    out = parts[0]
    for p in parts[1:]:
        out += "\\frac"
        if p.startswith("{") or len(p) < 2:
            out += p
            continue
        a, b, rest = p[0], p[1], p[2:]
        out += f"{{{a}}}{{{b}}}{rest}" if b != "{" else f"{{{a}}}{b}{rest}"
    return out


def _fix_a_slash_b(s: str) -> str:
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}" if m else s


def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def norm_math(s: str | None) -> str | None:
    """Hendrycks' strip_string, slightly hardened; None stays None."""
    if s is None:
        return None
    s = s.strip().replace("\n", "").rstrip(".")
    s = s.replace("\\!", "").replace("\\\\", "\\").replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "").replace("$", "")
    # Trailing units after a number (Hendrycks' _remove_right_units): "5.4 \text{ cents}" -> "5.4".
    # Only a final text block after something holding a digit: "\text{ Evelyn}" or "1\text{ or }2" stay.
    m = re.fullmatch(r"(.*\d.*?)\s*\\text\{\s+[A-Za-z][A-Za-z .]*\}\s*", s)
    if m:
        s = m.group(1)
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if "=" in s and len(s.split("=")[0]) <= 2:  # "x = 5" -> "5"
        s = s.split("=")[-1]
    s = _fix_sqrt(s).replace(" ", "")
    s = _fix_fracs(s)
    s = _fix_a_slash_b(s)
    s = s.replace(",\\!", "").replace("{,}", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):  # thousands separators
        s = s.replace(",", "")
    try:  # one canonical spelling for plain numbers: 5, 5.0, 5.00 -> 5
        x = float(s)
        s = str(int(x)) if x == int(x) and abs(x) < 1e15 else repr(x)
    except ValueError:
        pass
    if s == "0.5":  # Hendrycks' special case, after canonicalisation so that 0.50 and .5 map here too
        s = "\\frac{1}{2}"
    return s


# MATH answers without \boxed{} (some families ignore the instruction): when True, a COMPLETE answer
# falls back to the final math expression of its conclusion (final_math). Same rule for every model;
# set to False for the strict grader (\boxed{} only). Both are reported.
MATH_FALLBACK = True


_MATH_SEG = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\$([^$]+?)\$", re.DOTALL)  # each bounded by its own delimiters
_NUM = r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?"
# A value stated in the prose, never a word: "... is 24," / "... equals 3.5" / "27 is the smallest ..."
_PLAIN = re.compile(rf"(?:\bis|\bequals|=)\s*:?\s*({_NUM})(?!\d|\.\d)|({_NUM})\s+is\s+the\b")
_LONE_VAR = re.compile(r"\s*\\?[A-Za-z](?:_\{?\w+\}?)?\s*")  # "$n$", "$x_1$": a mention, not an answer


def _variables(side: str) -> set[str]:
    """Single-letter variables of an expression, ignoring LaTeX commands and \\text{...} labels."""
    side = re.sub(r"\\text\{[^{}]*\}|\\mathrm\{[^{}]*\}", " ", side)
    side = re.sub(r"\\[A-Za-z]+", " ", side)
    return set(re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", side))


def _value(expr: str, equation_asked: bool = False) -> str | None:
    """The answer held by a math expression.
    - When the question asks for an equation, an equation is the answer itself and is kept whole
      ("5x - 7y + 11z = 4"); a chain still gives its last member.
    - Otherwise a chain "a = b = c" gives its last member, and one "=" gives its right-hand side
      ("x = 7", "b+c = 3.21", "\\dbinom{31}{28} = 4495", "\\text{Constant Term} = -125")."""
    parts = [p.strip() for p in re.split(r"(?<![<>!\\])=", expr.strip())]
    if len(parts) >= 3:
        return parts[-1] or None
    if len(parts) == 2:
        return (expr.strip() if equation_asked else parts[1]) or None
    return parts[0] or None


_ASKS_EQUATION = re.compile(
    r"\b(?:find|enter|determine|give|write|compute|what\s+is)\s+(?:the|an|a)\s+equations?\b"
    r"|\bin\s+the\s+form\b[^.?!]*=", re.IGNORECASE)


def equation_asked(question: str) -> bool:
    """True only when the question REQUESTS an equation ("Find the equation of the plane ...", "Enter
    your answer in the form Ax + By = C"); merely mentioning one ("the equation of the line is ... ;
    find a+b+m") does not count."""
    return bool(_ASKS_EQUATION.search(question))


_STATED = re.compile(r"(?:\bis|=|:)\s*$")  # "... is $i$", "= $x_1$", "answer: $k$"


def final_math(text: str, equation_asked: bool = False) -> str | None:
    """Final answer of a solution written without \\boxed{}, read in its last non-empty paragraph:
    the LAST of its math segments ($$...$$, \\[...\\], $...$, each bounded by its own delimiters) and
    of the values stated in its prose after "is"/"equals"/"=" (numbers only, never words), read with
    _value; else the last number of the paragraph. A lone symbol ("$n$") is only a mention unless it
    is stated as the answer ("The answer is $i$."). None if nothing qualifies."""
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not paras:
        return None
    last = paras[-1].strip()
    cands = [(m.start(), g) for m in _MATH_SEG.finditer(last)
             for g in [next(x for x in m.groups() if x)]
             if not _LONE_VAR.fullmatch(g) or _STATED.search(last[:m.start()])]
    masked = _MATH_SEG.sub(lambda m: " " * len(m.group(0)), last)  # prose only, positions kept
    cands += [(m.start(), m.group(1) or m.group(2)) for m in _PLAIN.finditer(masked)]
    if not cands:  # no math segment: an undelimited final assignment, "x = \sqrt{2}"
        m = re.search(r"=\s*([^=\n$]+?)\s*\.?\s*$", masked)
        if m:
            cands.append((m.start(1), m.group(1)))
    if cands:
        return _value(max(cands)[1], equation_asked)
    nums = re.findall(_NUM, last)
    return nums[-1] if nums else None


def mc_letter(text: str, letters: list[str]) -> str | None:
    """The chosen option letter: last "answer is (X)" / "answer: X", else a lone letter on the last line."""
    L = "".join(letters)
    m = re.findall(rf"answer\s*(?:is|:)?\s*[:\-]?\s*\**\(?\s*([{L}])\s*\)?(?![A-Za-z])", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    # Fallback: a last line that is ONLY a letter, e.g. "(C)" or "**C**" -- never a letter inside prose
    # ("I do not know." must not become a vote for option I).
    last = text.strip().splitlines()[-1] if text.strip() else ""
    m = re.fullmatch(rf"[\s*_(\[]*([{L}])[\s*_)\].]*", last)
    return m.group(1) if m else None


def extract(bench: str, text: str, item: dict, ended: bool = True) -> str | None:
    """Normalised answer of a generated text (None when the model gave none)."""
    if bench == "gsm8k":
        # The LAST answer statement wins, whatever its form: the strict forms of common (needed to stop
        # a generation in time) and, for complete answers only, a lenient form that also accepts
        # "**The answer is 16.**" at the very end and "The answer is $N = 80$.".
        text = visible_answer_text(text)
        found = [(m.start(), m.group(1)) for m in FINAL_RE.finditer(text)]
        if ended:
            found += [(m.start(), m.group(1)) for m in _END_RE.finditer(text)]
            found += [(m.start(), m.group(1)) for m in re.finditer(
                r"answer is[\s:$*]*(?:[A-Za-z]\s*=\s*)?\$?\s*(-?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)]
        a = max(found)[1] if found else None
        return norm_math(a) if a is not None else None
    if bench == "math500":
        boxed = last_boxed(text)
        if boxed is not None or not MATH_FALLBACK or not ended:
            return norm_math(boxed)
        asked = equation_asked(str(item.get("question", "")))
        return norm_math(final_math(text, equation_asked=asked))
    if bench in ("arc", "mmlupro"):
        letters = [chr(65 + i) for i in range(len(item["options"]))]
        return mc_letter(text, letters)
    raise ValueError(bench)


def _collab_ended(r: dict, max_rounds: int | None, k: int | None = None) -> bool:
    """A text written together counts as complete only if the run stopped by itself: before the round
    limit, and not because no peer answered or no draft was admissible."""
    if "complete" in r:  # recorded by run_gen since gen-v3
        return bool(r["complete"])
    last = (r.get("log") or [{}])[-1].get("how", "") or ""
    # run_gen.solve_together: "aucun brouillon (pairs terminés)" means every peer had finished its answer,
    # even at the last allowed round; "aucune réponse" (every peer failed) and "aucun brouillon admissible"
    # are failures; any other run that used all its rounds may have been cut.
    if k is not None:  # k-of-n runs before gen-v3 did not record completion: never guess
        raise ValueError(f"{r.get('id')} : run en mode k sans indicateur de fin (antérieur à gen-v3), "
                         "ré-évaluation impossible sans deviner")
    if "pairs terminés" in last:  # every peer was asked, and every one had finished
        return True
    if "aucune réponse" in last or "admissible" in last:
        return False
    return max_rounds is not None and r.get("rounds", max_rounds) < max_rounds


def regrade_gen(rows: list[dict], max_rounds: int | None = None, k: int | None = None) -> list[dict]:
    """Re-grade experiment-2 rows (run_gen.py) from their stored texts with the current grader.

    The strict grader of phase 0 (common.FINAL_RE) missed final answers such as "**The answer is 16.**"
    or "The answer is N = 195."; the texts are complete, so re-grading needs no new inference. Solo
    answers are re-extracted (ended = the peer stopped by itself), the vote is recomputed from them with
    the same rule as run_gen.vote (plurality, ties to the most confident), and the answers written
    together (accord, croise) are re-extracted from their final text. Gold answers are normalised too."""
    out, solo = [], {}
    for r in rows:
        r = dict(r)
        r["gold"] = norm_math(str(r["gold"]))
        if r["mode"] == "solo":
            r["per_peer"] = [dict(p, answer=extract("gsm8k", p["text"] or "", {}, ended=bool(p.get("eos"))))
                             if p.get("text") is not None else dict(p) for p in r["per_peer"]]
            solo[r["id"]] = r
        elif r["mode"] in ("accord", "croise"):  # unfinished texts: strict extraction only
            r["answer"] = extract("gsm8k", r.get("text") or "", {}, ended=_collab_ended(r, max_rounds, k))
        out.append(r)
    for r in out:
        if r["mode"] == "vote" and r["id"] in solo:
            per = [p for p in solo[r["id"]]["per_peer"] if p["answer"] is not None]
            if not per:
                r["answer"] = None
                continue
            counts: dict[str, int] = {}
            for p in per:
                counts[p["answer"]] = counts.get(p["answer"], 0) + 1
            top = max(counts.values())
            r["answer"] = max((p for p in per if counts[p["answer"]] == top),
                              key=lambda p: p["conf"] if p["conf"] is not None else -1e9)["answer"]
    return out


def gold(bench: str, item: dict) -> str:
    if bench in ("gsm8k", "math500"):
        return norm_math(str(item["answer"]))
    return chr(65 + item["answer"])
