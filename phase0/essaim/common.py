"""Pieces both backends (PyTorch lm.py, llama.cpp llamacpp.py) must share exactly."""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

WORD_RE = re.compile(r"\S+\s*")
PROMPT_VERSION = "mc-v3"  # bump whenever a prompt, the template rendering or the letter reading changes
# Some chat templates (Granite, SmolLM3) insert today's date: freeze it so prompts are reproducible.
FIXED_DATE = _dt.date(2026, 10, 8)
_STRFTIME = re.compile(r"strftime_now\(\s*(['\"])(.*?)\1\s*\)")


def fixed_date_template(template: str) -> str:
    return _STRFTIME.sub(lambda m: repr(FIXED_DATE.strftime(m.group(2))), template)


def render_chat(tok, user: str, system: str | None = None, think: bool = False) -> str:
    """Chat-template text with the generation prompt, date frozen, thinking switched on or off."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    tpl = tok.chat_template if isinstance(tok.chat_template, str) else None
    kw = {"chat_template": fixed_date_template(tpl)} if tpl else {}
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=think, thinking=think, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def hf_revision(model_id: str) -> str | None:
    """Commit hash of the locally cached Hugging Face snapshot (tokenizer and template)."""
    ref = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def resolve_revision(model_id: str) -> str:
    """The commit to pin for this run: the cached one, or the current one after fetching the
    small tokenizer/config files. Models are then loaded with revision=<this commit>."""
    rev = hf_revision(model_id)
    if rev:
        return rev
    from huggingface_hub import snapshot_download
    p = snapshot_download(model_id, allow_patterns=["*.json", "*.jinja", "*.model", "*.txt"])
    return Path(p).name


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of words, each word keeping its trailing whitespace."""
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    if spans and spans[0][0] > 0:  # leading whitespace belongs to the first word
        spans[0] = (0, spans[0][1])
    return spans


def letter_token_ids(tok, letters: list[str]) -> list[list[int]]:
    """Single-token spellings (" A", "A") of each answer letter, disjoint across letters.

    Only spellings that the tokenizer encodes as exactly one token are kept, so the
    next-token probability is the probability of the whole letter."""
    out, seen = [], {}
    for L in letters:
        ids = []
        for spelling in (" " + L, L):
            enc = tok(spelling, add_special_tokens=False)["input_ids"]
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        if not ids:
            raise ValueError(f"letter {L!r} has no single-token spelling in {tok.name_or_path}")
        for t in ids:
            if t in seen and seen[t] != L:
                raise ValueError(f"token {t} spells both {seen[t]!r} and {L!r}")
            seen[t] = L
        out.append(ids)
    return out


def think_spec(template: str) -> tuple[str, str, str]:
    """(open tag, close tag, text that opens the answer after the close tag), per family.

    Gemma 4: "<|channel>thought ... <channel|>answer".
    Granite 3.3: "<think>...</think><response>...</response>".
    Qwen3 / SmolLM3: "<think>...</think>\\n\\nanswer"."""
    if "<channel|>" in template:
        return "<|channel>", "<channel|>", ""
    if "<response>" in template:
        return "<think>", "</think>", "<response>"
    return "<think>", "</think>", "\n\n"


def close_thought(thought: str, spec: tuple[str, str, str], generation_opened: bool) -> str:
    """Text to append after the (possibly cut) reasoning so that the answer can start.
    `generation_opened`: the prompt itself ends inside an open thinking block."""
    open_tag, close_tag, after = spec
    if generation_opened or open_tag in thought:
        return f"\n{close_tag}{after}"
    return "\n\n"


# ---------- GSM8K answers ----------
# "The answer is 42." then whitespace, or "...is 42" then a newline / bold marker: the number is
# followed by a delimiter that has actually been received, so neither "The answer is 4" (before "2")
# nor "The answer is 3." (before "75") counts as final while text is still arriving.
FINAL_RE = re.compile(r"answer is\s*\$?\s*\**\s*(-?\d[\d,]*(?:\.\d+)?)\s*\**\s*(?:\.\s|\n|\*\*)", re.I)
_END_RE = re.compile(r"answer is\s*\$?\s*\**\s*(-?\d[\d,]*(?:\.\d+)?)\s*\**\s*\.?\s*$", re.I)
_OPEN_TAGS, _CLOSE_TAGS = ("<think>", "<|channel>"), ("</think>", "<channel|>")


def visible_answer_text(text: str) -> str:
    """The text after the last closed thinking block, minus any thinking block still open."""
    for tag in _CLOSE_TAGS:
        if tag in text:
            text = text.split(tag)[-1]
    for tag in _OPEN_TAGS:
        if tag in text:
            text = text.split(tag)[0]
    return text


def final_answer(text: str, ended: bool = False) -> str | None:
    """Last explicitly finalised answer outside thinking; None if there is none.
    With `ended` (the generation stopped by itself), an answer at the very end also counts."""
    text = visible_answer_text(text)
    found = [(m.start(), m.group(1)) for m in FINAL_RE.finditer(text)]
    if ended:
        m = _END_RE.search(text)
        if m:
            found.append((m.start(), m.group(1)))
    if not found:
        return None
    s = max(found)[1].replace(",", "")
    v = float(s)
    return str(int(v)) if v == int(v) else str(v)


def answer_finalised(text: str) -> bool:
    return FINAL_RE.search(visible_answer_text(text)) is not None


def final_answer_end(text: str) -> int | None:
    """Character position just after the first finalised answer outside thinking, or None.
    Used as the named stop "final_answer": generation stops there, and only text up to there is kept."""
    start = 0
    for tag in _CLOSE_TAGS:
        i = text.rfind(tag)
        if i >= 0:
            start = max(start, i + len(tag))
    end = len(text)
    for tag in _OPEN_TAGS:
        j = text.find(tag, start)
        if j >= 0:
            end = min(end, j)
    m = FINAL_RE.search(text, start, end)
    return m.end() if m else None


# Named stop conditions a peer accepts (clients never send regular expressions).
STOPS = {"final_answer": final_answer_end}
GEN_PROTOCOL_VERSION = "gen-v3"  # GSM8K coordination protocol (stops, quorum, k answers); independent of PROMPT_VERSION


def stop_point(prefix: str, continuation: str, names: list[str]) -> int | None:
    """Stop position inside `continuation`, judged on prefix + continuation (so a thinking block
    opened in the assistant prefix is respected); None if no stop applies to the new text."""
    full = prefix + continuation
    cuts = [c - len(prefix) for c in (STOPS[n](full) for n in names) if c is not None and c > len(prefix)]
    return min(cuts) if cuts else None
