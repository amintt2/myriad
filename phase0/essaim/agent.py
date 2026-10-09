"""E12, part 3: an agent loop for Terminal-Bench behind the Myriad gateway.

Terminal-Bench tasks are scored by Harbor's official verifier after the agent stops, in the environment mode
declared by the task. The loop follows mini-swe-agent: one shell command per turn, in a single ```bash block,
the observation (exit code and output) fed back, the run ended by a command that prints a sentinel. This module
reproduces that loop against any OpenAI-compatible endpoint (our gateway: app/myriad/gateway.py, default
http://127.0.0.1:8400/v1) and adds the swarm as a per-step strategy:

  single    one model answers each turn (`myriad:family=<f>` pins a family; a plain model name pins a model):
            the baseline, and what the references (one big model) do;
  vote      k proposals per turn, one per family (`myriad:family=<f>` in k separate requests), the commands are
            normalised and the plurality wins; ties go to the family with the best dev record. This is the
            "proposer + vote on the next command" strategy: it only helps when peers often propose the SAME command,
            which for exploratory shell work they rarely do, hence
  cascade   the same peers; ask a reference on disagreement or any malformed peer reply, without consulting
            the final verifier (used by the Docker-free Aider Python pilot);
  verify    (design only, see the E12 section of the phase-0 README) best-of-k with execution feedback: run each distinct
            proposal in a COPY of the container (docker commit / snapshot), keep the one whose output looks
            productive (exit 0, new files, tests moving from fail to pass). Needs snapshot support in the environment.

HarborEnv borrows Harbor's environment; Harbor owns setup, verification and teardown. The loop also runs
without Harbor installed, using a fake chat and environment (tests/test_agent.py).
"""
from __future__ import annotations

import asyncio
import math
import re
import shlex
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Protocol

SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MAX_OBS_CHARS = 4000  # observation fed back to the model (head and tail kept)
SYSTEM = (
    "You are a careful software engineer working in a Linux shell. Reply with a short thought, then EXACTLY ONE "
    "bash command in a single ```bash code block. You see the exit code and output of each command. When the "
    f"task is done, reply with a block containing only: echo {SUBMIT}")
_BLOCK = re.compile(r"```(?:bash|sh)[ \t]*\n(.*?)```", re.S)


class ActionError(ValueError):
    """The reply does not hold exactly one bash block: the model is told so and asked again (mini-swe-agent's rule)."""


class GenerationLimit(RuntimeError):
    """The shared episode generation budget is exhausted."""


class GenerationDeadline(GenerationLimit):
    """The episode deadline expired during generation; grade the sources already produced."""


def parse_action(reply: str) -> str:
    blocks = _BLOCK.findall(reply)
    if len(blocks) != 1:
        raise ActionError(f"expected exactly one ```bash block, found {len(blocks)}")
    cmd = blocks[0].strip()
    if not cmd:
        raise ActionError("empty command")
    return cmd


def is_submit(cmd: str) -> bool:
    return cmd.strip() == f"echo {SUBMIT}"


def normalise(cmd: str) -> str:
    """Key under which two proposals count as the same command. Conservative: only what the shell ignores is
    ignored. A one-line command is cut into words with their quotes kept (shlex, non-POSIX mode), so
    `ls  -la  # list` and `ls -la` agree but `echo "$HOME"` and `echo '$HOME'`, or `ls *.py` and `ls '*.py'`,
    do not. A command of several lines (heredocs, where indentation is content) is compared exactly, apart
    from blank edges and trailing spaces."""
    lines = [l.rstrip() for l in cmd.strip().splitlines()]
    if len(lines) != 1:
        return "\n".join(lines)
    lex = shlex.shlex(lines[0], posix=False)
    lex.whitespace_split = True
    lex.commenters = "#"
    try:
        return " ".join(lex)
    except ValueError:  # unbalanced quote: the line as written
        return lines[0].strip()


def vote(proposals: list[tuple[str, str]], priority: list[str]) -> tuple[str, dict]:
    """proposals: (family, command); plurality over normalised commands, ties to the family earliest in
    `priority`. Returns the chosen command (as proposed) and the tally for the log."""
    if not proposals:
        raise ActionError("no proposal")
    keyed = [(f, c, normalise(c)) for f, c in proposals]
    count = Counter(k for _, _, k in keyed)
    top = max(count.values())
    rank = {f: i for i, f in enumerate(priority)}
    best = min((x for x in keyed if count[x[2]] == top), key=lambda x: rank.get(x[0], len(rank)))
    return best[1], {"tally": dict(count), "chosen_votes": top, "of": len(keyed), "unanimous": top == len(keyed)}


class Env(Protocol):
    def run(self, cmd: str, timeout_s: float) -> tuple[int, str]: ...  # (exit code, combined output)
    def close(self) -> None: ...


class HarborEnv:
    """Sync bridge to a borrowed Harbor BaseEnvironment, called from the episode's worker thread.

    The event loop remains available to Harbor's own timeouts. Closing cancels outstanding calls, but never
    stops the container: Trial must first collect artifacts and run the official verifier, then delete it.
    GNU timeout inside the container kills the command group; an exec-client timeout alone cannot do that.
    """

    def __init__(self, environment, loop: asyncio.AbstractEventLoop):
        self.environment, self.loop = environment, loop
        self._lock, self._pending, self._closed = threading.Lock(), set(), False

    def call(self, coroutine):
        with self._lock:
            if self._closed:
                coroutine.close()
                raise RuntimeError("Harbor environment bridge is closed")
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
            self._pending.add(future)
        try:
            return future.result()
        finally:
            with self._lock:
                self._pending.discard(future)

    def run(self, cmd: str, timeout_s: float) -> tuple[int, str]:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("command timeout must be positive and finite")
        command = f"timeout --signal=KILL {timeout_s:.6f}s bash -lc {shlex.quote(cmd)}"
        result = self.call(self.environment.exec(command=command, timeout_sec=math.ceil(timeout_s) + 5))
        output = (result.stdout or "") + (result.stderr or "")
        return result.return_code, output

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for future in self._pending:
                future.cancel()


Chat = Callable[[str, list[dict]], str]  # (model name, messages) -> reply text


@dataclass
class Step:
    reply: str
    command: str | None
    exit_code: int | None
    output: str
    note: dict = field(default_factory=dict)


def clip(text: str, limit: int = MAX_OBS_CHARS) -> str:
    return text if len(text) <= limit else text[: limit // 2] + "\n[... output cut ...]\n" + text[-limit // 2:]


def run_episode(chat: Chat, env: Env, task: str, families: list[str], strategy: str = "single", max_steps: int = 50,
                cmd_timeout_s: float = 60.0, priority: list[str] | None = None, timeout_s: float | None = None,
                on_step: Callable[[Step], None] | None = None, reference: str | None = None) -> dict:
    """One task. `families`: model names to ask (one for `single`, the k peers for `vote`). Returns the transcript
    and why it stopped ("submitted", "max_steps"); the verdict comes from the benchmark's verifier, not from here."""
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
                props.append((fam, parse_action(reply)))
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
                props = [(reference, parse_action(reply))]
            except ActionError as e:
                props = []
                replies[reference] = reply + f"\n[format error: {e}]"
        if not props:  # nobody produced a command: tell the model, count the turn
            messages += [{"role": "assistant", "content": next(iter(replies.values()))},
                         {"role": "user", "content": "Format error: reply with exactly one ```bash block."}]
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
            record(Step(replies[chosen], cmd, 0, "", {**note, "submitted": True}))
            return {"steps": steps, "stopped": "submitted"}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"steps": steps, "stopped": "timeout"}
        code, out = env.run(cmd, min(cmd_timeout_s, remaining))
        record(Step(replies[chosen], cmd, code, out, note))
        messages.append({"role": "user", "content": f"exit code: {code}\n{clip(out)}"})
    return {"steps": steps, "stopped": "max_steps"}


def gateway_chat(base_url: str = "http://127.0.0.1:8400/v1", api_key: str = "unused", max_tokens: int = 1024,
                 timeout_s: float = 600.0) -> Chat:
    """A Chat over the gateway's OpenAI-compatible endpoint (the gateway ignores `tools`: the command format
    above is plain text, as app/README.md advises for coding agents)."""
    import httpx
    http = httpx.Client(base_url=base_url, timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"})

    def chat(model: str, messages: list[dict]) -> str:
        r = http.post("/chat/completions", json={"model": model, "messages": messages, "max_tokens": max_tokens,
                                                  "temperature": 0.0})
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""
    return chat
