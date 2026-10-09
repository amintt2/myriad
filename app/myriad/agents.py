"""Sub-agents: one task split into sub-tasks that run IN PARALLEL on different peers, each with its own
small context, then verified locally, escalated once if they fail, and combined.

    POST /v1/agents/run      (gateway, 127.0.0.1:8400; `"stream": true` for server-sent events)
    GET  /v1/agents/schema   (JSON schemas of the request and of an automatic plan)

A run takes a `task` and either an explicit `plan` (a list of sub-tasks: prompt, role, skill or family
or model, context texts or local files, dependencies, k candidates, an optional verification) or
`"auto"`: an orchestrator peer (skill tag `orchestrator`, any peer otherwise) writes the plan as JSON,
validated against a strict schema (unknown fields are refused: a peer can neither name a local file
nor a command).

Scheduling. Sub-tasks whose dependencies are done leave at once, each in ONE gateway round trip (k
candidates from k peers of distinct families); a sub-task waits for its dependencies and receives their
results in its prompt. A sub-task whose dependency failed is skipped. Peers already working for the run
are avoided when another peer can take the job, so parallel sub-tasks land on different machines.

Verification. A sub-task may name a verification command: the requester (this machine) writes each
candidate's code into a temporary copy of a local directory and runs the command there, with a
timeout. Only commands the USER allowed, by name, run: the configuration's `verify_commands` and the
request's `allow_commands`. Commands are argument lists (no shell); `{file}` and `{workdir}` are
replaced by the candidate file and the temporary directory. Nothing a peer writes is ever executed
except through such a user-chosen command. Candidates that pass are kept; several passing candidates
are separated by weighted agreement (identical code adds up its peers' weights, then the weighted
medoid).

Cascade. A sub-task that fails (no answer, or no candidate passes verification) is retried once on a
stronger route: by default the fused swarm with more candidates (most reliable families first).

Combination: the results concatenated in plan order, or a final `merge` call that writes one answer.

Budgets: number of sub-tasks, parallel sub-tasks, completion tokens (reserved before each call,
counted after), and a deadline for the whole run; size limits on every text.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Callable, Literal

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .fusion import medoid
from .protocol import MAX_CONTENT_CHARS, MAX_TOKENS, TAG_PATTERN, Assigned, JobError, JobFrame, ResultFrame

log = logging.getLogger("myriad.agents")

MAX_SUBTASKS = 32
MAX_PARALLEL = 16
MAX_CANDIDATES = 4
MAX_RUN_TOKENS = 200_000
MAX_DEADLINE_S = 1800.0
MAX_TASK_CHARS = 16_000
MAX_PROMPT_CHARS = 16_000
MAX_CONTEXT_ITEMS = 16
MAX_CONTEXT_CHARS = 30_000  # one context item, and all the explicit context of one sub-task
MAX_FILE_BYTES = 256 * 1024
MAX_DEPENDENCIES = 16
MAX_COMMANDS = 16
MAX_OUTPUT_CHARS = 4000  # verification output kept
MAX_RESULT_CHARS = 64_000
MAX_COPY_FILES = 5000
MAX_COPY_BYTES = 100 * 1024 * 1024
VERIFY_CONCURRENCY = 4
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
             ".ruff_cache", ".tox", "dist", "build", ".idea", ".vscode"}
RESERVED_IDS = {"plan", "merge"}

Id = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")]
Tag = Annotated[str, Field(pattern=TAG_PATTERN)]
Short = Annotated[str, Field(max_length=200)]
CmdName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")]
Argv = Annotated[list[Annotated[str, Field(min_length=1, max_length=1000)]], Field(min_length=1, max_length=64)]
Hint = Literal["math", "mc", "free"]

SUBAGENT_SYSTEM = ("You are a sub-agent of Myriad working on ONE part of a larger task. Do only your sub-task, "
                   "completely and concisely. When code is asked for, give the complete code in one fenced code "
                   "block.")
PLANNER_SYSTEM = """You are the orchestrator of Myriad, a network of small language models. Split the user's task into \
independent sub-tasks that small models can each solve with a short, self-contained prompt. Sub-tasks run in \
parallel unless one needs the result of another (depends_on). Answer with ONE JSON object and nothing else:
{"subtasks": [{"id": "short-id", "role": "what this agent is", "skill": "one tag or null", "prompt": "complete \
instructions for this sub-task", "depends_on": ["ids of sub-tasks whose results it needs"], "uses_context": \
[indices of the context items it needs], "kind": "code or text"}], "combine": "concat or merge"}
Rules: at most {max} sub-tasks; ids are letters, digits, - or _; no other field; "kind" is "code" when the \
sub-task must produce code; skills available on the network: {skills}."""
MERGE_SYSTEM = ("You are the final agent of Myriad. Sub-agents solved parts of a task; combine their results into "
                "one coherent, complete answer to the task. Keep code blocks intact; resolve contradictions.")

_SUB = contextvars.ContextVar("myriad_subagent", default=None)  # (run id, sub-task id, attempt)
_VERIFY_SEM: asyncio.Semaphore | None = None


class AgentsError(Exception):
    def __init__(self, status: int, message: str, code: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


# ======================================================================== schema
class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextItem(_M):
    """Context given to a sub-task: a text, or a LOCAL file read by the requester (never by a peer)."""
    text: Annotated[str, Field(max_length=MAX_CONTEXT_CHARS)] | None = None
    file: Annotated[str, Field(min_length=1, max_length=1024)] | None = None
    label: Annotated[str, Field(max_length=120)] | None = None

    @model_validator(mode="after")
    def _one(self):
        if (self.text is None) == (self.file is None):
            raise ValueError("un élément de contexte a soit `text`, soit `file`")
        return self


class VerifySpec(_M):
    """Run the allowed command `run` on each candidate: its code goes to `target` (relative path) in a
    temporary copy of `workdir` (a local directory; an empty directory if none)."""
    run: CmdName
    target: Annotated[str, Field(min_length=1, max_length=260)] | None = None
    workdir: Annotated[str, Field(min_length=1, max_length=1024)] | None = None
    timeout_s: Annotated[float, Field(ge=1, le=600, allow_inf_nan=False)] = 60.0
    extract: Literal["code", "text"] = "code"

    @field_validator("target")
    @classmethod
    def _relative(cls, v):
        if v is None:
            return v
        p = v.replace("\\", "/")
        if p.startswith("/") or re.match(r"^[A-Za-z]:", p) or ".." in p.split("/"):
            raise ValueError("target doit être un chemin relatif sans ..")
        return p


class RouteChoice(_M):
    skill: Tag | None = None
    family: Annotated[str, Field(max_length=64)] | None = None
    model: Short | None = None
    k: Annotated[int, Field(ge=1, le=MAX_CANDIDATES)] | None = None


class SubTask(_M):
    id: Id
    prompt: Annotated[str, Field(min_length=1, max_length=MAX_PROMPT_CHARS)]
    role: Annotated[str, Field(max_length=60)] | None = None
    skill: Tag | None = None  # routed like the model name "myriad:<skill>"
    family: Annotated[str, Field(max_length=64)] | None = None  # "myriad:<family>"
    model: Short | None = None  # one exact network model
    system: Annotated[str, Field(max_length=4000)] | None = None
    context: Annotated[list[ContextItem], Field(max_length=MAX_CONTEXT_ITEMS)] = []
    uses_context: Annotated[list[Annotated[int, Field(ge=0, lt=MAX_CONTEXT_ITEMS)]], Field(max_length=MAX_CONTEXT_ITEMS)] = []
    depends_on: Annotated[list[Id], Field(max_length=MAX_DEPENDENCIES)] = []
    k: Annotated[int, Field(ge=1, le=MAX_CANDIDATES)] = 1
    verify: VerifySpec | None = None
    max_tokens: Annotated[int, Field(ge=1, le=MAX_TOKENS)] = 768
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] | None = None
    task_hint: Hint | None = None
    escalate: bool = True
    timeout_s: Annotated[float, Field(ge=5, le=600, allow_inf_nan=False)] | None = None


class MergeSpec(_M):
    prompt: Annotated[str, Field(max_length=4000)] | None = None
    skill: Tag | None = "orchestrator"
    family: Annotated[str, Field(max_length=64)] | None = None
    model: Short | None = None
    max_tokens: Annotated[int, Field(ge=1, le=MAX_TOKENS)] = 1024


class Budget(_M):
    max_subtasks: Annotated[int, Field(ge=1, le=MAX_SUBTASKS)] = 12
    max_parallel: Annotated[int, Field(ge=1, le=MAX_PARALLEL)] = 8
    max_tokens: Annotated[int, Field(ge=64, le=MAX_RUN_TOKENS)] = 32_000
    deadline_s: Annotated[float, Field(ge=1, le=MAX_DEADLINE_S, allow_inf_nan=False)] = 300.0


class RunRequest(_M):
    task: Annotated[str, Field(max_length=MAX_TASK_CHARS)] = ""
    plan: Annotated[list[SubTask], Field(min_length=1, max_length=MAX_SUBTASKS)] | Literal["auto"] = "auto"
    context: Annotated[list[ContextItem], Field(max_length=MAX_CONTEXT_ITEMS)] = []
    combine: Literal["concat", "merge"] | None = None  # None: the plan's choice (auto), else concat
    merge: MergeSpec | None = None
    budget: Budget = Budget()
    allow_commands: Annotated[dict[CmdName, Argv], Field(max_length=MAX_COMMANDS)] = {}
    planner: RouteChoice = RouteChoice(skill="orchestrator")
    verify: VerifySpec | None = None  # auto plans: applied to the sub-tasks of kind "code"
    auto_k: Annotated[int, Field(ge=1, le=MAX_CANDIDATES)] = 1  # auto plans: candidates per sub-task
    escalate_to: RouteChoice = RouteChoice(k=3)  # default: the fused swarm, 3 candidates
    stream: bool = False

    @model_validator(mode="after")
    def _task(self):
        if self.plan == "auto" and not self.task.strip():
            raise ValueError("`task` est requis pour un plan automatique")
        return self


class AutoSubTask(_M):
    """What an orchestrator PEER may write: no file, no command, no route other than a skill tag."""
    id: Id
    prompt: Annotated[str, Field(min_length=1, max_length=MAX_PROMPT_CHARS)]
    role: Annotated[str, Field(max_length=60)] | None = None
    skill: Tag | None = None
    depends_on: Annotated[list[Id], Field(max_length=MAX_DEPENDENCIES)] = []
    uses_context: Annotated[list[Annotated[int, Field(ge=0, lt=MAX_CONTEXT_ITEMS)]], Field(max_length=MAX_CONTEXT_ITEMS)] = []
    kind: Literal["code", "text"] = "text"


class AutoPlan(_M):
    subtasks: Annotated[list[AutoSubTask], Field(min_length=1, max_length=MAX_SUBTASKS)]
    combine: Literal["concat", "merge"] | None = None


def schemas() -> dict:
    return {"request": RunRequest.model_json_schema(), "auto_plan": AutoPlan.model_json_schema()}


# ======================================================================== validation
def _err(e: ValidationError) -> str:
    x = e.errors()[0]
    loc = ".".join(str(p) for p in x.get("loc", ()))
    return f"{loc} : {x.get('msg')}" if loc else str(x.get("msg"))


def parse_request(body: dict, config_commands: dict | None = None) -> RunRequest:
    """JSON body -> validated RunRequest (raises AgentsError 400)."""
    if isinstance(body, list):  # a bare list of sub-tasks
        body = {"plan": body}
    if not isinstance(body, dict):
        raise AgentsError(400, "objet JSON attendu")
    try:
        req = RunRequest.model_validate(body)
    except ValidationError as e:
        raise AgentsError(400, f"requête invalide : {_err(e)}") from e
    if req.plan != "auto":
        validate_plan(req.plan, req, allowed_commands(config_commands, req.allow_commands))
    elif req.verify is not None and req.verify.run not in allowed_commands(config_commands, req.allow_commands):
        raise AgentsError(400, f"commande de vérification non autorisée : {req.verify.run}")
    for i, c in enumerate(req.context):
        load_context(c, f"context[{i}]")
    return req


def allowed_commands(config_commands: dict | None, request_commands: dict | None) -> dict[str, list[str]]:
    """The user's allow-list: the configuration's commands, then the request's (same name: the
    request's wins). Malformed configuration entries are ignored."""
    out: dict[str, list[str]] = {}
    for src in (config_commands or {}, request_commands or {}):
        for name, argv in src.items():
            if (isinstance(name, str) and re.match(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$", name)
                    and isinstance(argv, list) and argv and all(isinstance(a, str) and a for a in argv)):
                out[name] = list(argv)
    return out


def validate_plan(subtasks: list[SubTask], req: RunRequest, allowed: dict) -> list[str]:
    """Ids unique, dependencies known and acyclic, budget respected, verification commands allowed,
    context sizes within limits. Returns the ids in a topological order (plan order kept)."""
    if len(subtasks) > req.budget.max_subtasks:
        raise AgentsError(400, f"trop de sous-tâches : {len(subtasks)} > {req.budget.max_subtasks} (budget)")
    ids = [s.id for s in subtasks]
    if len(set(ids)) != len(ids):
        raise AgentsError(400, "identifiants de sous-tâches en double")
    for s in subtasks:
        if s.id.lower() in RESERVED_IDS:
            raise AgentsError(400, f"identifiant réservé : {s.id}")
        for d in s.depends_on:
            if d == s.id:
                raise AgentsError(400, f"{s.id} dépend d'elle-même")
            if d not in ids:
                raise AgentsError(400, f"{s.id} dépend d'une sous-tâche inconnue : {d}")
        for i in s.uses_context:
            if i >= len(req.context):
                raise AgentsError(400, f"{s.id} : uses_context {i} hors de la liste `context`")
        if s.verify is not None and s.verify.run not in allowed:
            raise AgentsError(400, f"{s.id} : commande de vérification non autorisée : {s.verify.run} "
                                   "(à déclarer dans verify_commands ou allow_commands)")
        if sum(1 for _ in [s.model, s.family, s.skill] if _) > 1:
            raise AgentsError(400, f"{s.id} : une seule route parmi skill, family, model")
        size = len(s.prompt) + sum(len(load_context(c, f"{s.id}.context")) for c in s.context)
        if size > MAX_CONTEXT_CHARS:
            raise AgentsError(400, f"{s.id} : consigne et contexte trop longs ({size} > {MAX_CONTEXT_CHARS} caractères)")
    order = topological(subtasks)
    if order is None:
        raise AgentsError(400, "les dépendances forment un cycle")
    return order


def topological(subtasks: list[SubTask]) -> list[str] | None:
    """A topological order, as close to the plan order as possible (the plan order itself when it
    already lists every dependency first); None if the dependencies form a cycle."""
    deps = {s.id: set(s.depends_on) for s in subtasks}
    done: list[str] = []
    seen: set[str] = set()
    left = [s.id for s in subtasks]
    while left:
        nxt = next((i for i in left if deps[i] <= seen), None)
        if nxt is None:
            return None
        done.append(nxt)
        seen.add(nxt)
        left.remove(nxt)
    return done


def levels(subtasks: list[SubTask]) -> dict[str, int]:
    """Depth of each sub-task in the dependency graph (0: no dependency)."""
    by = {s.id: s for s in subtasks}
    memo: dict[str, int] = {}

    def lv(i: str) -> int:
        if i not in memo:
            memo[i] = 0 if not by[i].depends_on else 1 + max(lv(d) for d in by[i].depends_on)
        return memo[i]

    return {s.id: lv(s.id) for s in subtasks}


def load_context(c: ContextItem, where: str = "context") -> str:
    """The text of a context item; a file is read here, by the requester (size-limited, UTF-8)."""
    if c.text is not None:
        return c.text
    p = Path(c.file).expanduser()
    try:
        if not p.is_file():
            raise AgentsError(400, f"{where} : fichier introuvable : {c.file}")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise AgentsError(400, f"{where} : fichier trop gros (> {MAX_FILE_BYTES // 1024} Kio) : {c.file}")
        return p.read_bytes().decode("utf-8", errors="replace")
    except OSError as e:
        raise AgentsError(400, f"{where} : lecture impossible : {e}") from e


def parse_auto_plan(text: str, max_subtasks: int) -> AutoPlan:
    """Orchestrator output -> AutoPlan: the JSON object (fenced or bare), strictly validated."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""
    if not raw:
        raise ValueError("aucun objet JSON dans la réponse")
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ValueError(f"JSON invalide : {e}") from e
    try:
        plan = AutoPlan.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"plan non conforme : {_err(e)}") from e
    if len(plan.subtasks) > max_subtasks:
        raise ValueError(f"trop de sous-tâches : {len(plan.subtasks)} > {max_subtasks}")
    return plan


# ======================================================================== verification
def extract_code(text: str) -> str:
    """The longest fenced code block of an answer, else the whole answer."""
    blocks = re.findall(r"```[^\n`]*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy a directory for verification: no symbolic links followed, VCS/dependency folders skipped,
    bounded in files and bytes."""
    n = size = 0
    for root, dirs, files in os.walk(src, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not os.path.islink(os.path.join(root, d))]
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            sp = Path(root) / f
            if sp.is_symlink() or not sp.is_file():
                continue
            n += 1
            size += sp.stat().st_size
            if n > MAX_COPY_FILES or size > MAX_COPY_BYTES:
                raise AgentsError(400, f"dossier de vérification trop gros (> {MAX_COPY_FILES} fichiers "
                                       f"ou {MAX_COPY_BYTES >> 20} Mio) : {src}")
            shutil.copy2(sp, dst / rel / f)


def _default_target(text: str) -> str:
    m = re.search(r"```([A-Za-z0-9+#-]*)", text)
    lang = (m.group(1) if m else "").lower()
    ext = {"python": "py", "py": "py", "typescript": "ts", "ts": "ts", "javascript": "js", "js": "js",
           "json": "json", "bash": "sh", "sh": "sh", "go": "go", "rust": "rs", "c": "c", "cpp": "cpp",
           "java": "java", "html": "html", "css": "css", "sql": "sql", "yaml": "yaml", "toml": "toml"}.get(lang, "txt")
    return f"candidate.{ext}"


def run_verification(argv: list[str], text: str, spec: VerifySpec, cancel=None) -> dict:
    """Blocking: write the candidate into a temporary copy of the work directory and run the allowed
    command there (no shell, timeout, stdin closed, bounded output; `cancel`: a threading.Event that
    kills it). The work directory itself is never touched."""
    t0 = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix="myriad-verify-"))
    try:
        if spec.workdir:
            src = Path(spec.workdir).expanduser()
            if not src.is_dir():
                return {"passed": False, "exit_code": None, "error": f"dossier introuvable : {spec.workdir}",
                        "ms": 0.0, "output": ""}
            _copy_tree(src, tmp)
        target = (tmp / (spec.target or _default_target(text))).resolve()
        if tmp.resolve() not in target.parents:
            return {"passed": False, "exit_code": None, "error": "cible hors du dossier", "ms": 0.0, "output": ""}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(extract_code(text) if spec.extract == "code" else text, encoding="utf-8")
        cmd = [a.replace("{file}", str(target)).replace("{workdir}", str(tmp)) for a in argv]
        ms = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731
        try:
            code, out, why = _run_bounded(cmd, tmp, spec.timeout_s, cancel)
        except OSError as e:
            return {"passed": False, "exit_code": None, "error": f"commande impossible à lancer : {e}",
                    "ms": ms(), "output": ""}
        if why == "timeout":
            return {"passed": False, "exit_code": None, "error": f"délai dépassé ({spec.timeout_s:g} s)",
                    "ms": ms(), "output": out}
        if why == "cancelled":
            return {"passed": False, "exit_code": None, "error": "annulé", "ms": ms(), "output": out}
        return {"passed": code == 0, "exit_code": code, "error": None, "ms": ms(), "output": out}
    except AgentsError as e:
        return {"passed": False, "exit_code": None, "error": e.message, "ms": 0.0, "output": ""}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the verification command and the processes it started."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                           timeout=10)
        else:
            os.killpg(proc.pid, 9)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_bounded(cmd: list[str], cwd: Path, timeout_s: float, cancel=None) -> tuple[int | None, str, str | None]:
    """Run without a shell; keep only the last MAX_OUTPUT_CHARS of stdout+stderr (a noisy command
    cannot fill the memory); kill the process tree at the timeout or when `cancel` is set.
    Returns (exit code, output tail, None | "timeout" | "cancelled")."""
    import collections
    import threading

    kw: dict = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
    tail: collections.deque = collections.deque()
    size = [0]

    def drain() -> None:
        while True:
            chunk = proc.stdout.read1(8192) if hasattr(proc.stdout, "read1") else proc.stdout.read(8192)
            if not chunk:
                return
            tail.append(chunk)
            size[0] += len(chunk)
            while size[0] - len(tail[0]) > 4 * MAX_OUTPUT_CHARS:
                size[0] -= len(tail.popleft())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    end = time.monotonic() + timeout_s
    why = None
    # Done when the command has exited AND its output is closed: a child it left running (and that
    # still holds the output) keeps the deadline and the cancellation in force.
    while proc.poll() is None or reader.is_alive():
        if cancel is not None and cancel.is_set():
            why = "cancelled"
        elif time.monotonic() > end:
            why = "timeout"
        if why:
            _kill_tree(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            break
        if proc.poll() is None:
            try:
                proc.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        else:
            reader.join(timeout=0.1)
    # A leftover child that survived the kill (Windows: once its parent is gone, it is out of reach of
    # the tree kill) may still hold the pipe: its reader, a daemon thread, is left to finish on its own.
    reader.join(timeout=2)
    out = b"".join(tail).decode("utf-8", "replace")[-MAX_OUTPUT_CHARS:]
    return (None if why else proc.returncode), out, why


async def verify_async(argv: list[str], text: str, spec: VerifySpec) -> dict:
    """Run one verification in a worker thread. The concurrency slot is held until the worker has
    really finished: on cancellation the command is killed first."""
    import threading

    global _VERIFY_SEM
    if _VERIFY_SEM is None:
        _VERIFY_SEM = asyncio.Semaphore(VERIFY_CONCURRENCY)
    async with _VERIFY_SEM:
        cancel = threading.Event()
        work = asyncio.ensure_future(asyncio.to_thread(run_verification, argv, text, spec, cancel))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            cancel.set()
            try:
                await asyncio.wait_for(asyncio.shield(work), 30)
            except BaseException:  # noqa: BLE001 - still cancelling: the slot is released anyway
                pass
            raise


def pick_candidate(cands: list[dict]) -> int:
    """Among passing candidates: identical code (only leading/trailing blank space ignored: inner
    whitespace can be meaningful) pools its peers' weights; the heaviest group wins; ties go to the
    weighted medoid."""
    if len(cands) == 1:
        return 0
    groups: dict[str, float] = {}
    keys = [extract_code(c["text"]).strip() for c in cands]
    for key, c in zip(keys, cands):
        groups[key] = groups.get(key, 0.0) + float(c.get("weight") or 0.0)
    best = max(groups.values())
    tied = [i for i, key in enumerate(keys) if groups[key] == best]
    if len({keys[i] for i in tied}) == 1:
        return max(tied, key=lambda i: float(cands[i].get("weight") or 0.0))
    sub = [cands[i] for i in tied]
    m = medoid([c["text"] for c in sub], [float(c.get("weight") or 0.0) or 1e-6 for c in sub])
    return tied[m if m is not None else 0]


# ======================================================================== execution
@dataclass
class SubState:
    spec: SubTask
    level: int = 0
    status: str = "pending"  # pending, running, verifying, escalating, ok, failed, skipped, cancelled
    text: str | None = None
    peer: dict | None = None
    route: dict | None = None
    tokens: int = 0
    started: float | None = None
    ended: float | None = None
    attempts: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    verify: dict | None = None
    escalated: bool = False
    error: str | None = None
    inputs: list | None = None  # the merge: the sub-tasks it combines (any number)

    def public(self, t0: float) -> dict:
        s = self.spec
        ms = lambda t: None if t is None else round((t - t0) * 1000, 1)  # noqa: E731
        return {"id": s.id, "role": s.role, "skill": s.skill, "family": s.family, "model": s.model,
                "depends_on": list(self.inputs) if self.inputs is not None else list(s.depends_on), "level": self.level, "k": s.k,
                "verify_command": s.verify.run if s.verify else None, "status": self.status,
                "text": (self.text or "")[:MAX_RESULT_CHARS] if self.text is not None else None, "peer": self.peer,
                "route": self.route, "tokens": self.tokens, "started_ms": ms(self.started), "ended_ms": ms(self.ended),
                "ms": round((self.ended - self.started) * 1000, 1) if self.started and self.ended else None,
                "attempts": self.attempts, "candidates": self.candidates, "verify": self.verify,
                "escalated": self.escalated, "error": self.error}


class AgentRun:
    """One run: plan (given or written by an orchestrator), parallel dispatch with dependencies,
    verification, cascade, combination. `emit` receives every progress event (a dict)."""

    def __init__(self, gateway, req: RunRequest, emit: Callable[[dict], None] | None = None,
                 config_commands: dict | None = None):
        self.gw, self.req = gateway, req
        self.id = uuid.uuid4().hex
        self._emit = emit
        self.allowed = allowed_commands(config_commands if config_commands is not None
                                        else getattr(gateway, "verify_commands", {}), req.allow_commands)
        self.contexts = [load_context(c, f"context[{i}]") for i, c in enumerate(req.context)]
        self.subs: dict[str, SubState] = {}
        self.order: list[str] = []
        self.t0 = time.perf_counter()
        self.tokens_used = 0
        self.tokens_reserved = 0
        self.plan_source = "auto" if req.plan == "auto" else "explicit"
        self.planner: dict | None = None
        self.merge_state: SubState | None = None
        self.combine = req.combine or "concat"
        self._jobs: dict[str, tuple] = {}  # job id -> (sub-task id, attempt)
        self._job_node: dict[str, str] = {}  # job id -> its peer, while the job is in flight
        self.deadline = time.monotonic() + req.budget.deadline_s

    # ---------- events ----------
    def ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)

    def emit(self, kind: str, **kw) -> None:
        if self._emit is not None:
            try:
                self._emit({"type": kind, "run_id": self.id, "t_ms": self.ms(), **kw})
            except Exception as e:  # a slow or gone listener never breaks the run
                log.debug("agent event listener failed: %s", e)

    def _observe(self, direction: str, frame) -> None:
        """Node frame tap: which peer got which sub-task's job, live."""
        if direction == "out" and isinstance(frame, JobFrame):
            tag = _SUB.get()
            if tag is not None and tag[0] == self.id:
                self._jobs[frame.job.job_id] = tag[1:]
                if frame.target:
                    self._job_node[frame.job.job_id] = frame.target
                    p = self._peer_info(frame.target)
                    self.emit("job", subtask=tag[1], attempt=tag[2], job_id=frame.job.job_id, node_id=frame.target,
                              model=p.get("model"), family=p.get("family"))
            return
        if direction != "in":
            return
        res = getattr(frame, "result", None)
        jid = getattr(res, "job_id", None) or getattr(frame, "job_id", None)
        tag = self._jobs.get(jid)
        if tag is None:
            return
        if isinstance(frame, Assigned):
            c = frame.peer
            self._job_node[jid] = c.node_id
            self.emit("job", subtask=tag[0], attempt=tag[1], job_id=jid, node_id=c.node_id, model=c.model,
                      family=c.family, tag_match=frame.tag_match)
        elif isinstance(frame, ResultFrame):
            self._job_node.pop(jid, None)
            self.emit("answered", subtask=tag[0], attempt=tag[1], job_id=jid, node_id=res.node_id, model=res.model,
                      tokens=res.completion_tokens, compute_ms=res.compute_ms)
        elif isinstance(frame, JobError):
            self._job_node.pop(jid, None)
            self.emit("job_failed", subtask=tag[0], attempt=tag[1], job_id=jid, error=frame.error[:200])

    def _forget(self, sid: str, attempt: int) -> None:
        """A call is over: its stragglers (cancelled by the gateway) no longer occupy their peers."""
        for jid, tag in list(self._jobs.items()):
            if tag == (sid, attempt):
                self._job_node.pop(jid, None)

    def _peer_info(self, node_id: str) -> dict:
        d = getattr(self.gw, "_dir", None)
        for p in (d[1] if d else []):
            if p.get("node_id") == node_id:
                return p
        return {}

    # ---------- run ----------
    async def execute(self) -> dict:
        node = self.gw.node
        node.add_observer(self._observe)
        status = "ok"
        error = None
        try:
            try:
                await asyncio.wait_for(self._execute(), max(1.0, self.deadline - time.monotonic()))
            except asyncio.TimeoutError:
                status, error = "timeout", f"délai de la tâche dépassé ({self.req.budget.deadline_s:g} s)"
                for st in self.subs.values():
                    if st.status in ("pending", "running", "verifying", "escalating"):
                        st.status = "cancelled" if st.status != "pending" else "skipped"
                        st.error = st.error or "délai dépassé"
                        st.ended = st.ended or time.perf_counter()
            except AgentsError as e:
                status, error = "failed", e.message
        finally:
            node.remove_observer(self._observe)
        result = self.result(status, error)
        self.emit("done", result=result)
        return result

    async def _execute(self) -> None:
        if self.req.plan == "auto":
            subtasks = await self._plan()
        else:
            subtasks = list(self.req.plan)
        self.order = validate_plan(subtasks, self.req, self.allowed)
        lv = levels(subtasks)
        for s in subtasks:
            self.subs[s.id] = SubState(spec=s, level=lv[s.id])
        self.emit("plan", source=self.plan_source, planner=self.planner, combine=self.combine,
                  subtasks=[self.subs[i].public(self.t0) for i in self.order], budget=self.req.budget.model_dump())
        await self._schedule()
        if self.combine == "merge" and any(st.status == "ok" for st in self.subs.values()):
            await self._merge()

    async def _schedule(self) -> None:
        running: dict[asyncio.Task, str] = {}
        try:
            while True:
                for i in self.order:
                    st = self.subs[i]
                    if st.status != "pending":
                        continue
                    deps = [self.subs[d] for d in st.spec.depends_on]
                    if any(d.status in ("failed", "skipped", "cancelled") for d in deps):
                        st.status, st.error = "skipped", "une dépendance a échoué"
                        self.emit("subtask", **self._brief(st))
                    elif all(d.status == "ok" for d in deps) and len(running) < self.req.budget.max_parallel:
                        st.status, st.started = "running", time.perf_counter()
                        running[asyncio.create_task(self._run_sub(st))] = i
                if not running:
                    return
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    sid = running.pop(t)
                    st = self.subs[sid]
                    exc = t.exception()
                    if exc is not None:
                        st.status, st.error = "failed", f"{type(exc).__name__}: {exc}"[:300]
                        st.ended = time.perf_counter()
                        self.emit("subtask", **self._brief(st))
        finally:
            for t in running:
                t.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

    def _brief(self, st: SubState) -> dict:
        d = st.public(self.t0)
        d["text"] = (st.text or "")[:4000] if st.text is not None else None
        return {"subtask": d}

    def _messages(self, st: SubState) -> list[dict]:
        s = st.spec
        parts = []
        if self.req.task.strip():
            task = self.req.task.strip()
            parts.append(f"Overall task (for context only):\n{task[:2000]}{'…' if len(task) > 2000 else ''}")
        parts.append(f"Your sub-task{f' ({s.role})' if s.role else ''}:\n{s.prompt}")
        for i in s.uses_context:
            label = self.req.context[i].label or self.req.context[i].file or f"context {i}"
            parts.append(f"## Context: {label}\n{self.contexts[i]}")
        for c in s.context:
            label = c.label or c.file or "context"
            parts.append(f"## Context: {label}\n{load_context(c, s.id)}")
        fixed = sum(len(p) + 2 for p in parts)
        deps = [self.subs[d] for d in s.depends_on]
        room = max(0, MAX_CONTENT_CHARS - 200 - fixed)
        each = room // max(1, len(deps))
        for d in deps:
            text = d.text or ""
            if len(text) > each:
                text = text[:max(0, each - 40)] + "\n…[tronqué]"
            parts.append(f"## Result of sub-task {d.spec.id}{f' ({d.spec.role})' if d.spec.role else ''}:\n{text}")
        user = "\n\n".join(parts)[:MAX_CONTENT_CHARS]
        system = s.system or (SUBAGENT_SYSTEM + (f" Your role: {s.role}." if s.role else ""))
        return [{"role": "system", "content": system[:MAX_CONTENT_CHARS]}, {"role": "user", "content": user}]

    def _reserve(self, want: int) -> int:
        """Reserve up to `want` completion tokens; returns what was granted (0: budget spent)."""
        left = self.req.budget.max_tokens - self.tokens_used - self.tokens_reserved
        got = max(0, min(want, left))
        self.tokens_reserved += got
        return got

    async def _call(self, sid: str, attempt: int, messages: list[dict], max_tokens: int, k: int, temperature: float,
                    hint: str, skill=None, family=None, model=None, timeout_s: float | None = None):
        """One gateway round trip for a sub-task (k candidates). Tokens reserved before, counted after."""
        want = max_tokens * k
        granted = self._reserve(want)
        if granted < min(want, 32):
            self.tokens_reserved -= granted
            raise AgentsError(429, "budget de jetons épuisé", "budget")
        if granted < want:  # fewer candidates, or shorter answers, within what is left
            k = max(1, min(k, granted // max_tokens)) if granted >= max_tokens else 1
            max_tokens = min(max_tokens, granted // k)
        left = self.deadline - time.monotonic()
        if left < 1.0:  # the gateway's shortest request timeout
            self.tokens_reserved -= granted
            raise AgentsError(504, "délai de la tâche dépassé", "deadline")
        timeout = min(timeout_s or self.gw.timeout_s, left)
        avoid = sorted(set(self._job_node.values()))[:64]
        token = _SUB.set((self.id, sid, attempt))
        try:
            from .gateway import GatewayError
            pause = 0.5
            while True:
                try:
                    ans = await self.gw.ask(messages, max_tokens=max_tokens, temperature=temperature, k=k,
                                            task_hint=hint, format_instruction=False, timeout_s=timeout, tag=skill,
                                            family=family, model=model, avoid=avoid)
                    break
                except GatewayError as e:
                    if e.code != "no_peers":
                        raise
                    if avoid:  # every fitting peer is already busy with this run: any peer then
                        avoid = []
                        continue
                    # Every peer busy (more parallel sub-tasks than free peers): wait for one, a few times.
                    # A model or family nobody serves fails at once.
                    if model or family or pause > 4 or self.deadline - time.monotonic() < pause + 1.0:
                        raise
                    await asyncio.sleep(pause)
                    pause *= 2
                    timeout = min(timeout_s or self.gw.timeout_s, self.deadline - time.monotonic())
        finally:
            _SUB.reset(token)
            self.tokens_reserved -= granted
            self._forget(sid, attempt)
        used = sum(int(c.get("completion_tokens") or 0) for c in ans.candidates) or ans.completion_tokens
        self.tokens_used += used
        return ans, used, k

    async def _run_sub(self, st: SubState) -> None:
        s = st.spec
        self.emit("subtask", **self._brief(st))
        messages = self._messages(st)
        temp = s.temperature if s.temperature is not None else (0.7 if s.k > 1 else 0.2)
        routes = [("primary", {"skill": s.skill, "family": s.family, "model": s.model}, s.k)]
        if s.escalate:
            esc = self.req.escalate_to
            routes.append(("escalation", {"skill": esc.skill, "family": esc.family, "model": esc.model},
                           esc.k or max(3, s.k + 1)))
        for attempt, (kind, route, k) in enumerate(routes, start=1):
            if attempt > 1:
                st.status, st.escalated = "escalating", True
                self.emit("escalate", subtask=s.id, reason=st.error, route=route, k=k)
            a = {"attempt": attempt, "kind": kind, "route": route, "k": k, "status": "running",
                 "started_ms": self.ms()}
            st.attempts.append(a)
            st.route = {**route, "k": k}
            try:
                ans, used, k = await self._call(s.id, attempt, messages, s.max_tokens, k,
                                                temp if attempt == 1 or k > 1 else 0.2, s.task_hint or "free",
                                                timeout_s=s.timeout_s, **route)
            except AgentsError as e:
                a.update(status="failed", error=e.message, ended_ms=self.ms())
                st.error = e.message
                if e.code in ("budget", "deadline"):
                    break
                continue
            except Exception as e:  # GatewayError and the like: this attempt failed
                msg = getattr(e, "message", None) or f"{type(e).__name__}: {e}"
                a.update(status="failed", error=msg[:300], ended_ms=self.ms())
                st.error = msg[:300]
                continue
            st.tokens += used
            a["tokens"] = used
            a["fallback"] = bool(ans.meta.get("route", {}).get("fallback"))
            cands = ans.candidates or [{"text": ans.text, "weight": 1.0, "node_id": None, "model": None,
                                        "family": None, "completion_tokens": ans.completion_tokens}]
            chosen, verdicts = await self._choose(st, ans, cands, attempt)
            st.candidates = [{"node_id": c.get("node_id"), "model": c.get("model"), "family": c.get("family"),
                              "tokens": c.get("completion_tokens"), "weight": round(float(c.get("weight") or 0), 4),
                              "tag_match": c.get("tag_match"), "verify": v, "chosen": i == chosen, "attempt": attempt}
                             for i, (c, v) in enumerate(zip(cands, verdicts))]
            if chosen is None:
                a.update(status="failed", error=st.error, ended_ms=self.ms())
                continue
            c = cands[chosen]
            st.text = c["text"]
            st.peer = {"node_id": c.get("node_id"), "model": c.get("model"), "family": c.get("family"),
                       "tag_match": c.get("tag_match")}
            st.verify = verdicts[chosen]
            st.status, st.error = "ok", None
            a.update(status="ok", ended_ms=self.ms())
            break
        if st.status != "ok":
            st.status = "failed"
        st.ended = time.perf_counter()
        self.emit("subtask", **self._brief(st))

    async def _choose(self, st: SubState, ans, cands: list[dict], attempt: int) -> tuple[int | None, list]:
        """Index of the kept candidate (None: none acceptable) and each candidate's verification."""
        spec = st.spec.verify
        if spec is None:
            texts = [c["text"] for c in cands]
            idx = texts.index(ans.text) if ans.text in texts else 0
            if not (cands[idx]["text"] or "").strip():
                st.error = "réponse vide"
                return None, [None] * len(cands)
            return idx, [None] * len(cands)
        argv = self.allowed.get(spec.run)
        if argv is None:  # validated before: only a changed allow-list gets here
            st.error = f"commande non autorisée : {spec.run}"
            return None, [None] * len(cands)
        st.status = "verifying"
        self.emit("verifying", subtask=st.spec.id, attempt=attempt, command=spec.run, candidates=len(cands))
        verdicts = await asyncio.gather(*(verify_async(argv, c["text"] or "", spec) for c in cands))
        for i, (c, v) in enumerate(zip(cands, verdicts)):
            self.emit("verified", subtask=st.spec.id, attempt=attempt, candidate=i, node_id=c.get("node_id"),
                      model=c.get("model"), passed=v["passed"], exit_code=v["exit_code"], ms=v["ms"],
                      error=v["error"], output=v["output"][-600:])
        passing = [i for i, v in enumerate(verdicts) if v["passed"]]
        if not passing:
            st.error = f"aucun candidat ne passe la vérification ({spec.run})"
            st.verify = verdicts[0] if verdicts else None
            return None, list(verdicts)
        best = passing[pick_candidate([cands[i] for i in passing])]
        return best, list(verdicts)

    async def _plan(self) -> list[SubTask]:
        """The orchestrator writes the plan; one retry with the validation error if it is not valid."""
        r = self.req
        try:
            peers = (await self.gw.directory())[0]
        except Exception:
            peers = []
        skills = sorted({t for p in peers for t in (p.get("tags") or [])}) or ["(none)"]
        system = PLANNER_SYSTEM.replace("{max}", str(r.budget.max_subtasks)).replace("{skills}", ", ".join(skills))
        ctx = "".join(f"\n\n## Context item {i}: {c.label or c.file or 'text'}\n{self.contexts[i][:3000]}"
                      for i, c in enumerate(r.context))
        user = f"Task:\n{r.task}{ctx}"[:MAX_CONTENT_CHARS]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        state = SubState(spec=SubTask(id="plan", prompt="plan", role="orchestrator", skill=r.planner.skill,
                                      family=r.planner.family, model=r.planner.model), level=-1)
        state.status, state.started = "running", time.perf_counter()
        self.planner = {"status": "running"}
        self.emit("planning", planner=state.public(self.t0))
        last_err = None
        unreachable = False
        for attempt in (1, 2):
            try:
                ans, used, _ = await self._call("plan", attempt, messages, 1024, 1, 0.0, "free",
                                                skill=r.planner.skill, family=r.planner.family, model=r.planner.model)
            except Exception as e:
                last_err, unreachable = getattr(e, "message", None) or str(e), True
                break
            state.tokens += used
            c = ans.candidates[0] if ans.candidates else {}
            state.peer = {"node_id": c.get("node_id"), "model": c.get("model"), "family": c.get("family")}
            try:
                plan = parse_auto_plan(ans.text, r.budget.max_subtasks)
                if topological([SubTask(id=x.id, prompt=x.prompt, depends_on=x.depends_on) for x in plan.subtasks]) is None:
                    raise ValueError("les dépendances forment un cycle")
                ids = {x.id for x in plan.subtasks}
                if len(ids) != len(plan.subtasks) or any(d not in ids for x in plan.subtasks for d in x.depends_on):
                    raise ValueError("identifiants en double ou dépendance inconnue")
            except ValueError as e:
                last_err = str(e)
                messages = messages + [{"role": "assistant", "content": ans.text[:MAX_CONTENT_CHARS]},
                                       {"role": "user", "content": f"Invalid plan: {last_err}. Answer again with "
                                                                   "the JSON object only, following the rules."}]
                continue
            state.status, state.ended, state.text = "ok", time.perf_counter(), ans.text
            self.planner = state.public(self.t0)
            if r.combine is None and plan.combine:
                self.combine = plan.combine
            self.emit("planning", planner=self.planner)
            return [SubTask(id=x.id, prompt=x.prompt, role=x.role, skill=x.skill, depends_on=x.depends_on,
                            uses_context=[i for i in x.uses_context if i < len(r.context)],
                            verify=r.verify if x.kind == "code" else None, k=r.auto_k) for x in plan.subtasks]
        state.status, state.ended, state.error = "failed", time.perf_counter(), last_err
        self.planner = state.public(self.t0)
        self.emit("planning", planner=self.planner)
        if unreachable:
            raise AgentsError(502, f"l'orchestrateur n'a pas répondu : {last_err}", "planner_failed")
        raise AgentsError(502, f"plan automatique invalide : {last_err}", "plan_invalid")

    async def _merge(self) -> None:
        r = self.req
        m = r.merge or MergeSpec()
        st = SubState(spec=SubTask(id="merge", prompt=m.prompt or "merge", role="merge", skill=m.skill,
                                   family=m.family, model=m.model, max_tokens=m.max_tokens, escalate=False),
                      level=1 + max((x.level for x in self.subs.values()), default=0),
                      inputs=[i for i in self.order if self.subs[i].status == "ok"])
        self.merge_state = st
        st.status, st.started = "running", time.perf_counter()
        self.emit("subtask", **self._brief(st))
        done = [self.subs[i] for i in self.order if self.subs[i].status == "ok"]
        room = (MAX_CONTENT_CHARS - 3000) // max(1, len(done))
        parts = [f"Task:\n{r.task[:2000]}"] if r.task.strip() else []
        if m.prompt:
            parts.append(f"Instructions:\n{m.prompt}")
        for d in done:
            text = d.text or ""
            parts.append(f"## Result of {d.spec.id}{f' ({d.spec.role})' if d.spec.role else ''}:\n"
                         f"{text[:room]}{'…' if len(text) > room else ''}")
        messages = [{"role": "system", "content": MERGE_SYSTEM},
                    {"role": "user", "content": "\n\n".join(parts)[:MAX_CONTENT_CHARS]}]
        try:
            ans, used, _ = await self._call("merge", 1, messages, m.max_tokens, 1, 0.2, "free", skill=m.skill,
                                            family=m.family, model=m.model)
            st.tokens = used
            c = ans.candidates[0] if ans.candidates else {}
            st.peer = {"node_id": c.get("node_id"), "model": c.get("model"), "family": c.get("family"),
                       "tag_match": c.get("tag_match")}
            st.text, st.status = ans.text, "ok" if ans.text.strip() else "failed"
            if st.status == "failed":
                st.error = "réponse vide"
        except Exception as e:
            st.status, st.error = "failed", (getattr(e, "message", None) or str(e))[:300]
        st.ended = time.perf_counter()
        self.emit("subtask", **self._brief(st))

    # ---------- result ----------
    def concat(self) -> str:
        out = []
        for i in self.order:
            st = self.subs[i]
            if st.status == "ok":
                out.append(f"## {st.spec.id}{f' — {st.spec.role}' if st.spec.role else ''}\n\n{st.text}")
        return "\n\n".join(out)

    def result(self, status: str, error: str | None) -> dict:
        subs = [self.subs[i] for i in self.order] if self.order else list(self.subs.values())
        n_ok = sum(1 for s in subs if s.status == "ok")
        merged = self.merge_state is not None and self.merge_state.status == "ok"
        if status == "ok":
            status = "ok" if subs and n_ok == len(subs) else "partial" if n_ok else "failed"
            if status == "ok" and self.merge_state is not None and not merged:  # results kept, concatenated
                status, error = "partial", f"fusion finale échouée : {self.merge_state.error or '?'}"
        text = self.merge_state.text if merged else self.concat()
        durations = [(s.ended - s.started) for s in subs if s.started and s.ended]
        wall = (time.perf_counter() - self.t0) * 1000
        sum_ms = sum(durations) * 1000
        return {"id": self.id, "object": "myriad.agents.run", "status": status, "error": error, "task": self.req.task,
                "plan_source": self.plan_source, "planner": self.planner, "combine": self.combine,
                "merged": merged, "result": text[:MAX_RESULT_CHARS],
                "subtasks": [s.public(self.t0) for s in subs],
                "merge": self.merge_state.public(self.t0) if self.merge_state else None,
                "usage": {"completion_tokens": self.tokens_used, "budget_tokens": self.req.budget.max_tokens,
                          "subtasks": len(subs), "ok": n_ok},
                "timing": {"wall_ms": round(wall, 1), "subtasks_sum_ms": round(sum_ms, 1),
                           "parallel_speedup": round(sum_ms / wall, 2) if wall > 0 and sum_ms else None}}


async def run_agents(gateway, body: dict, emit: Callable[[dict], None] | None = None) -> dict:
    """Validate a request body and run it (raises AgentsError 400 on an invalid request)."""
    cmds = getattr(gateway, "verify_commands", {})
    req = parse_request(body, cmds)
    return await AgentRun(gateway, req, emit, cmds).execute()


# ======================================================================== HTTP
def sse_events(gateway, req: RunRequest):
    """Server-sent events of one run: every progress event, then `done` with the result. The run is
    cancelled if the client goes away."""
    q: asyncio.Queue = asyncio.Queue()
    run = AgentRun(gateway, req, q.put_nowait)

    async def go() -> None:
        try:
            await run.execute()
        except Exception as e:  # never leave the stream hanging
            q.put_nowait({"type": "error", "run_id": run.id, "message": f"{type(e).__name__}: {e}"[:300]})
        finally:
            q.put_nowait(None)

    async def events():
        task = asyncio.create_task(go())
        try:
            yield f"data: {json.dumps({'type': 'start', 'run_id': run.id, 'task': req.task[:2000]}, ensure_ascii=False)}\n\n"
            while True:
                ev = await q.get()
                if ev is None:
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    return events()


def install(app, gateway) -> None:
    """POST /v1/agents/run and GET /v1/agents/schema on the gateway's app (local-only middleware)."""
    from .gateway import GatewayError, read_json

    @app.post("/v1/agents/run")
    async def agents_run(request: Request):
        try:
            body = await read_json(request)
            req = parse_request(body, gateway.verify_commands)
        except GatewayError as e:
            return JSONResponse({"error": {"message": e.message, "type": e.code}}, status_code=e.status)
        except AgentsError as e:
            return JSONResponse({"error": {"message": e.message, "type": e.code}}, status_code=e.status)
        if not gateway.node.connected.is_set():
            return JSONResponse({"error": {"message": "le nœud n'est pas connecté au traqueur", "type": "not_connected"}},
                                status_code=503)
        if req.stream:
            return StreamingResponse(sse_events(gateway, req), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        return await AgentRun(gateway, req, None, gateway.verify_commands).execute()

    @app.get("/v1/agents/schema")
    async def agents_schema():
        return {**schemas(), "verify_commands": sorted(gateway.verify_commands)}


def install_ui(app, runtime) -> None:
    """The interface's endpoints (same-origin, token-checked by the UI middleware): the run as
    server-sent events, and the demo plan."""
    from .gateway import GatewayError, read_json

    @app.post("/api/agents/run")
    async def ui_agents_run(request: Request):
        gw = runtime.gateway
        if gw is None:
            return JSONResponse({"error": "le nœud démarre"}, status_code=503)
        try:
            body = await read_json(request)
            req = parse_request(body, gw.verify_commands)
        except GatewayError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        except AgentsError as e:
            return JSONResponse({"error": e.message}, status_code=e.status)
        return StreamingResponse(sse_events(gw, req), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/api/agents/demo")
    async def ui_agents_demo(lang: str = "fr"):
        gw = runtime.gateway
        return demo_plan(set(getattr(gw, "verify_commands", {}) or {}), lang)


# ======================================================================== demo, client
DEMO_VERIFY = "python-syntax"


def demo_verify_commands() -> dict[str, list[str]]:
    """The allow-list of the demo network: a syntax check of the candidate (nothing else runs)."""
    return {DEMO_VERIFY: [sys.executable, "-I", "-m", "py_compile", "{file}"]}


def demo_plan(allowed: set[str] | None = None, lang: str = "fr") -> dict:
    """A 6-agent plan (3 parallel, then 3 depending on them) and a final merge. Python sub-tasks are
    verified by a syntax check when the allow-list has `python-syntax`."""
    v = {"run": DEMO_VERIFY, "target": "candidate.py", "timeout_s": 20} if DEMO_VERIFY in (allowed or set()) else None
    fr = lang != "en"
    task = ("Construire une petite application de liste de tâches : module Python, analyse des dates, client "
            "TypeScript, tests, revue de code et documentation." if fr else
            "Build a small to-do list app: Python module, due-date parsing, TypeScript client, tests, code review "
            "and documentation.")
    plan = [
        {"id": "api", "role": "Python developer", "skill": "python", "k": 2, "verify": v,
         "prompt": "Write a Python module todo.py with a Todo dataclass (id, title, done, due) and functions "
                   "add(title, due=None), complete(todo_id) and open_items() over an in-memory list."},
        {"id": "dates", "role": "Python developer", "skill": "python", "verify": v,
         "prompt": "Write a Python function parse_due(text) that turns 'today', 'tomorrow', 'in 3 days' or an ISO "
                   "date into a datetime.date, and raises ValueError otherwise."},
        {"id": "client", "role": "TypeScript developer", "skill": "typescript",
         "prompt": "Write a TypeScript client class TodoClient with typed methods add, complete and list calling a "
                   "REST API at /todos (fetch, async/await)."},
        {"id": "tests", "role": "test writer", "skill": "python", "depends_on": ["api", "dates"], "verify": v,
         "prompt": "Write pytest tests for the todo module and for parse_due, using the code given in context."},
        {"id": "review", "role": "code reviewer", "skill": "review", "depends_on": ["api", "client"],
         "prompt": "Review the Python module and the TypeScript client: list concrete bugs and risks, most "
                   "important first, in at most 8 bullet points."},
        {"id": "docs", "role": "technical writer", "skill": "docs", "depends_on": ["client"],
         "prompt": "Write a short README section explaining how to use the TypeScript client, with one example."},
    ]
    for s in plan:
        if s.get("verify") is None:
            s.pop("verify", None)
    return {"task": task, "plan": plan, "combine": "merge",
            "merge": {"prompt": "Assemble a short delivery note: what was built, the files, open review points."},
            "budget": {"max_subtasks": 8, "max_tokens": 24000, "deadline_s": 120}}


def run_remote(body: dict, base_url: str = "http://127.0.0.1:8400", on_event: Callable[[dict], None] | None = None,
               timeout: float = 1900.0) -> dict:
    """Python helper: run a request on a local gateway, streaming progress events to `on_event`.

        from myriad.agents import run_remote
        result = run_remote({"task": "...", "plan": "auto"}, on_event=print)
    """
    import httpx

    body = {**body, "stream": True}
    result: dict | None = None
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=10)) as c:
        with c.stream("POST", "/v1/agents/run", json=body) as r:
            if r.status_code != 200:
                r.read()
                try:
                    msg = r.json().get("error", {}).get("message")
                except ValueError:
                    msg = r.text
                raise AgentsError(r.status_code, msg or f"HTTP {r.status_code}", "http")
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if on_event is not None:
                    on_event(ev)
                if ev.get("type") == "done":
                    result = ev["result"]
                elif ev.get("type") == "error":
                    raise AgentsError(500, ev.get("message", "erreur"), "run_error")
    if result is None:
        raise AgentsError(502, "flux interrompu avant la fin", "stream")
    return result
