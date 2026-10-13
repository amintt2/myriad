"""E17 Aider loop, isolated from the historical agent and other benchmark adapters."""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Callable

from essaim.agent import (ActionError, Chat, Env, GenerationDeadline, GenerationLimit, Step, SUBMIT, clip,
                          is_submit, normalise, parse_action, vote)
from essaim.local_linux import IsolationError

TEST = "MYRIAD_RUN_OFFICIAL_TESTS"
SYSTEM = (
    "You are a careful software engineer working in a Linux shell. Reply with a short thought, then EXACTLY ONE "
    "bash command in a single ```bash code block. Implement the sources, test, and correct failures. "
    f"To test, use a block containing only: {TEST}. This is a controller action, not a shell executable. "
    "The controller tests a snapshot of your sources against the protected original assertions and returns feedback. "
    "Running a Python file or your own tests does not authorize submission. After a complete green official test "
    f"of your current sources, submit with a block containing only: echo {SUBMIT}. "
    "Editing sources invalidates the green test; run the official test again before submitting.")


class Reply(str):
    def __new__(cls, text: str, finish_reason=None):
        value = super().__new__(cls, text)
        value.finish_reason = finish_reason
        return value


def parse_reply(reply: str) -> str:
    if isinstance(reply, Reply) and reply.finish_reason != "stop":
        raise ActionError("incomplete reply: finish_reason must be stop; no command was executed")
    return parse_action(reply)


def source_hashes(sources: dict[str, str]) -> dict:
    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}


def complete_green(grade: dict) -> bool:
    calls = grade.get("calls")
    expected = grade.get("tests_expected")
    return (grade.get("status") == "graded" and grade.get("passed") is True
            and type(expected) is int and expected > 0 and grade.get("tests") == expected
            and grade.get("tests_complete") is True and grade.get("failures") == 0 and grade.get("errors") == 0
            and not grade.get("broken") and isinstance(calls, list) and bool(calls)
            and all(c.get("complete") is True and "candidate_failure" not in c for c in calls))


def test_feedback(grade: dict, green: bool) -> str:
    details = {k: grade.get(k) for k in ("tests", "tests_expected", "tests_complete", "failures", "errors", "feedback")}
    keys = ("test", "function", "candidate_failure", "program_exception", "diagnostics")
    failures = [{k: c[k] for k in keys if k in c}
                for c in grade.get("calls", []) if "candidate_failure" in c or "program_exception" in c]
    details["candidate_failures"] = failures
    state = "GREEN: current sources may be submitted." if green else "FAILED: correct the sources and test again."
    return f"Official test {state}\n" + clip(json.dumps(details, ensure_ascii=True))


def run_episode(chat: Chat, env: Env, task: str, families: list[str], strategy: str = "single", max_steps: int = 50,
                cmd_timeout_s: float = 60.0, priority: list[str] | None = None, timeout_s: float | None = None,
                on_step: Callable[[Step], None] | None = None, reference: str | None = None, *,
                snapshot, test_sources) -> dict:
    """Aider-only loop: official tests authorize submission of an unchanged source snapshot."""
    if strategy not in ("single", "vote", "cascade"):
        raise ValueError("strategy must be 'single', 'vote' or 'cascade'")
    if strategy == "cascade" and not reference:
        raise ValueError("cascade requires a reference model")
    if not families or len(set(families)) != len(families) or max_steps <= 0 or cmd_timeout_s <= 0:
        raise ValueError("nonempty distinct models and positive limits required")
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("episode timeout must be positive and finite")
    deadline = time.monotonic() + timeout_s if timeout_s is not None else math.inf
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    steps: list[Step] = []
    green = None

    def record(step):
        steps.append(step)
        if on_step is not None:
            on_step(step)

    for _ in range(max_steps):
        asks = families[:1] if strategy == "single" else families
        props, replies = [], {}
        for fam in asks:
            if time.monotonic() >= deadline:
                return {"steps": steps, "stopped": "timeout"}
            try:
                reply = chat(fam, messages)
            except GenerationDeadline:
                return {"steps": steps, "stopped": "timeout"}
            except GenerationLimit:
                return {"steps": steps, "stopped": "generation_limit"}
            replies[fam] = reply
            try:
                props.append((fam, parse_reply(reply)))
            except ActionError as e:
                replies[fam] = reply + f"\n[format error: {e}]"
        cascade = strategy == "cascade" and (len(props) != len(asks)
                   or len({normalise(c) for _, c in props}) != 1)
        peer_vote = vote(props, priority or list(asks))[1] if props and strategy == "cascade" else None
        if cascade:
            if time.monotonic() >= deadline:
                return {"steps": steps, "stopped": "timeout"}
            try:
                reply = chat(reference, messages)
            except GenerationDeadline:
                return {"steps": steps, "stopped": "timeout"}
            except GenerationLimit:
                return {"steps": steps, "stopped": "generation_limit"}
            replies[reference] = reply
            try:
                props = [(reference, parse_reply(reply))]
            except ActionError as e:
                props = []
                replies[reference] = reply + f"\n[format error: {e}]"
        if not props:  # nobody produced a command: tell the model, count the turn
            messages += [{"role": "assistant", "content": next(iter(replies.values()))},
                         {"role": "user", "content": "No command executed: return a complete response with exactly "
                          "one ```bash block. If the reply was truncated, use a shorter edit."}]
            record(Step(next(iter(replies.values())), None, None, "", {
                "format_error": True, "replies": replies, "cascade": cascade}))
            continue
        cmd, note = vote(props, priority or list(asks)) if strategy in ("vote", "cascade") else (props[0][1], {})
        note["cascade"] = cascade
        if peer_vote is not None:
            note["peer_vote"] = peer_vote
        note["replies"] = replies
        chosen = next(f for f, c in props if c == cmd)
        messages.append({"role": "assistant", "content": replies[chosen]})
        if time.monotonic() >= deadline:
            return {"steps": steps, "stopped": "timeout"}
        if is_submit(cmd):
            current = snapshot()
            if time.monotonic() >= deadline:
                return {"steps": steps, "stopped": "timeout"}
            if green is not None and current == green:
                record(Step(replies[chosen], cmd, 0, "", {**note, "submitted": True}))
                return {"steps": steps, "stopped": "submitted", "tested_sources": green}
            out = f"Submission refused: current sources need a complete green official test. Run {TEST}."
            record(Step(replies[chosen], cmd, 1, out, {**note, "submission_refused": True}))
            messages.append({"role": "user", "content": out})
            continue
        if cmd == TEST:
            sources = snapshot()
            green = None
            grade = test_sources(dict(sources), deadline - time.monotonic())
            if grade.get("status") != "graded":
                record(Step(replies[chosen], cmd, None, "Official verifier infrastructure error.",
                            {**note, "official_test": grade}))
                raise IsolationError("official verifier infrastructure error")
            passed = complete_green(grade)
            if time.monotonic() >= deadline:
                record(Step(replies[chosen], cmd, 124, "Official test exceeded episode deadline.",
                            {**note, "official_test": grade}))
                return {"steps": steps, "stopped": "timeout"}
            if passed and snapshot() == sources:
                green = sources
            out = test_feedback(grade, green is not None)
            record(Step(replies[chosen], cmd, 0 if green is not None else 1, out,
                        {**note, "official_test": grade, "source_hashes": source_hashes(sources)}))
            messages.append({"role": "user", "content": out})
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"steps": steps, "stopped": "timeout"}
        code, out = env.run(cmd, min(cmd_timeout_s, remaining))
        record(Step(replies[chosen], cmd, code, out, note))
        if green is not None and snapshot() != green:
            green = None
        if code in (124, 125) or time.monotonic() >= deadline:
            return {"steps": steps, "stopped": "output_limit" if code == 125 else "timeout"}
        messages.append({"role": "user", "content": f"exit code: {code}\n{clip(out)}"})
    return {"steps": steps, "stopped": "max_steps"}
