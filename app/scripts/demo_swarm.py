"""A local demo network for trying the Myriad interface without a GPU (and for the screenshots).

It starts a real tracker, ~16 simulated peers of six model families (scripted engines with realistic
delays), background traffic from two other users, your own simulated node with the full interface,
and a second interface in first-run mode (setup wizard).

    uv run python scripts/demo_swarm.py            # prints the two URLs, Ctrl+C to stop

Nothing is downloaded and nothing leaves 127.0.0.1."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import secrets
import tempfile
from pathlib import Path

import uvicorn

from myriad.agents import demo_verify_commands
from myriad.config import Config
from myriad.crypto import Identity
from myriad.engine import FakeEngine, free_port
from myriad.gateway import Gateway
from myriad.node import NodeClient
from myriad.priors import family_of, params_of
from myriad.protocol import MAX_FRAME_BYTES
from myriad.routing import node_tags
from myriad.runtime import NodeRuntime
from myriad.tracker import Tracker
from myriad.ui import make_ui_app
from myriad.wizard import SetupWizard

PEERS = [
    ("unsloth/Qwen3.5-4B-GGUF", 0.90, 1.6), ("unsloth/Qwen3.5-2B-GGUF", 0.70, 0.9), ("Qwen/Qwen3-1.7B-GGUF", 0.55, 0.8),
    ("unsloth/gemma-4-E4B-it-GGUF", 0.82, 1.8), ("unsloth/gemma-4-E2B-it-GGUF", 0.74, 1.1), ("ggml-org/gemma-4-E2B-it-GGUF", 0.74, 1.3),
    ("ibm-granite/granite-4.2-3b-GGUF", 0.72, 1.2), ("ibm-granite/granite-3.3-2b-instruct-GGUF", 0.50, 1.0),
    ("ggml-org/SmolLM3-3B-GGUF", 0.86, 1.3), ("HuggingFaceTB/SmolLM3-3B-GGUF", 0.86, 1.5),
    ("mistralai/Ministral-3-3B-Instruct-2512-GGUF", 0.78, 1.4), ("mistralai/Ministral-3-3B-Instruct-2512-GGUF", 0.78, 2.0),
    ("unsloth/Phi-4-mini-instruct-GGUF", 0.80, 1.7), ("unsloth/Phi-4-mini-instruct-GGUF", 0.80, 1.2),
    ("unsloth/Qwen3.5-4B-GGUF", 0.90, 2.2), ("ibm-granite/granite-4.2-3b-GGUF", 0.72, 1.6),
]
FREE = {
    "qwen": "The sky looks blue because air molecules scatter short (blue) wavelengths of sunlight much more than long "
            "(red) ones — Rayleigh scattering. That scattered blue light reaches our eyes from every direction of the sky.",
    "gemma": "Sunlight is scattered by the molecules of the atmosphere, and blue light, with its shorter wavelength, is "
             "scattered the most. So wherever we look in the sky, we see that scattered blue light.",
    "granite": "Rayleigh scattering: gas molecules scatter blue light more strongly than red light, so the sky appears blue.",
    "smollm": "Because of Rayleigh scattering: shorter blue wavelengths are scattered far more by air molecules than red "
              "ones, filling the sky with blue light coming from all directions.",
    "mistral": "The atmosphere scatters the blue part of sunlight more than the red part (Rayleigh scattering), so blue "
               "light reaches us from all over the sky.",
    "phi": "Air molecules scatter shorter wavelengths more efficiently; blue light is scattered across the sky and that "
           "is the colour we perceive.",
}


# Skill tags of the demo peers (sub-agents, "myriad:<tag>"), on top of those read from the model name
# (orchestrator and review for the 4B-class models).
DEMO_TAGS = {"Qwen3-1.7B": ["python"], "granite-3.3": ["python"], "Ministral": ["typescript"],
             "Phi-4": ["typescript"], "SmolLM3": ["docs"]}

AGENT_CODE = {
    "todo.py": "```python\nfrom dataclasses import dataclass\nfrom datetime import date\n\n\n@dataclass\nclass Todo:\n"
               "    id: int\n    title: str\n    done: bool = False\n    due: date | None = None\n\n\n_items: list[Todo] = []\n\n\n"
               "def add(title: str, due: date | None = None) -> Todo:\n    todo = Todo(len(_items) + 1, title, due=due)\n"
               "    _items.append(todo)\n    return todo\n\n\ndef complete(todo_id: int) -> None:\n    for t in _items:\n"
               "        if t.id == todo_id:\n            t.done = True\n            return\n    raise KeyError(todo_id)\n\n\n"
               "def open_items() -> list[Todo]:\n    return [t for t in _items if not t.done]\n```",
    "parse_due": "```python\nimport re\nfrom datetime import date, timedelta\n\n\ndef parse_due(text: str) -> date:\n"
                 "    s = text.strip().lower()\n    if s == \"today\":\n        return date.today()\n    if s == \"tomorrow\":\n"
                 "        return date.today() + timedelta(days=1)\n    m = re.fullmatch(r\"in (\\d+) days?\", s)\n    if m:\n"
                 "        return date.today() + timedelta(days=int(m.group(1)))\n    return date.fromisoformat(s)\n```",
    "parse_due_bad": "```python\nfrom datetime import date, timedelta\n\ndef parse_due(text)\n    if text == 'today':\n"
                     "        return date.today()\n    return date.fromisoformat(text)\n```",
    "TodoClient": "```typescript\nexport interface Todo { id: number; title: string; done: boolean; due?: string }\n\n"
                  "export class TodoClient {\n  constructor(private base = \"/todos\") {}\n\n"
                  "  async add(title: string, due?: string): Promise<Todo> {\n    const r = await fetch(this.base, { method: \"POST\", "
                  "headers: { \"Content-Type\": \"application/json\" }, body: JSON.stringify({ title, due }) });\n"
                  "    return r.json();\n  }\n\n  async complete(id: number): Promise<void> {\n"
                  "    await fetch(`${this.base}/${id}/done`, { method: \"POST\" });\n  }\n\n"
                  "  async list(): Promise<Todo[]> {\n    return (await fetch(this.base)).json();\n  }\n}\n```",
    "pytest tests": "```python\nimport pytest\n\nfrom todo import add, complete, open_items\nfrom dates import parse_due\n\n\n"
                    "def test_add_and_complete():\n    t = add(\"write docs\")\n    complete(t.id)\n"
                    "    assert t not in open_items()\n\n\ndef test_parse_due_rejects_garbage():\n"
                    "    with pytest.raises(ValueError):\n        parse_due(\"someday\")\n```",
    "Review": "- `complete()` raises KeyError for an unknown id: the client expects a 404, map it in the API layer.\n"
              "- `_items` is a module global: not thread-safe, and ids are reused after a restart.\n"
              "- `TodoClient.add` ignores HTTP errors: check `r.ok` before `r.json()`.\n"
              "- `due` is a string on the TypeScript side and a `date` in Python: document the ISO format.\n"
              "- `list()` loads every item: add paging before the list grows.",
    "README section": "## Using the TypeScript client\n\n```ts\nconst todos = new TodoClient();\n"
                      "const t = await todos.add(\"Buy milk\", \"2026-10-12\");\nawait todos.complete(t.id);\n"
                      "console.log(await todos.list());\n```\n\nEvery method returns a promise; `due` is an ISO date (YYYY-MM-DD).",
}


def agent_reply(messages, model: str) -> str | None:
    """Scripted sub-agent answers for the Agents view (None: not a sub-agent call)."""
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    user = messages[-1]["content"]
    if "orchestrator of Myriad" in system:
        task = user.split("Task:\n", 1)[-1].split("\n\n## ", 1)[0][:300]
        return json.dumps({"subtasks": [
            {"id": "outline", "role": "analyst", "skill": "review", "prompt": f"List the parts and risks of: {task}"},
            {"id": "core", "role": "Python developer", "skill": "python", "kind": "code",
             "prompt": f"Write the core Python module for: {task}"},
            {"id": "client", "role": "TypeScript developer", "skill": "typescript", "kind": "code",
             "prompt": f"Write the TypeScript client for: {task}"},
            {"id": "review", "role": "reviewer", "skill": "review", "depends_on": ["core", "client"],
             "prompt": "Review the code of the other sub-agents."}], "combine": "merge"})
    if "final agent of Myriad" in system:
        parts = re.findall(r"## Result of (\w+)", user)
        return ("Delivery note: a Python to-do module with due-date parsing (todo.py, dates.py), a typed TypeScript "
                "client (client.ts), pytest tests and a README section. Assembled from " + ", ".join(parts) + ".\n\n"
                "Open review points: map unknown ids to 404, check HTTP errors in the client, document ISO dates.")
    if "sub-agent of Myriad" not in system:
        return None
    sub = user.split("Your sub-task", 1)[-1].split("\n\n## ", 1)[0]
    if "parse_due" in sub and "pytest" not in sub:
        bad = any(k in model for k in ("Qwen3-1.7B", "granite-3.3"))  # the small python specialists slip here
        return AGENT_CODE["parse_due_bad" if bad else "parse_due"]
    for needle, key in (("pytest", "pytest tests"), ("todo.py", "todo.py"), ("TodoClient", "TodoClient"),
                        ("Review", "Review"), ("README", "README section")):
        if needle in sub:
            return AGENT_CODE[key]
    if "Python" in sub or "python" in sub:
        return AGENT_CODE["todo.py"]
    return f"Done: {sub.strip().splitlines()[-1][:200] if sub.strip() else 'ok'}"


def make_reply(p: float, family: str, rng: random.Random, model: str = ""):
    def reply(messages):
        a = agent_reply(messages, model)
        if a is not None:
            return a
        q = messages[-1]["content"]
        right = rng.random() < p
        if re.search(r"\bA\)", q):
            letter = "B" if right else rng.choice("ACD")
            return f"Canberra is the capital. The answer is {letter}."
        nums = [int(x) for x in re.findall(r"\d+", q)]
        if len(nums) >= 2 and ("answer is" in q or "?" in q) and not "sky" in q.lower() and "ciel" not in q.lower():
            good = 29 if {3, 7, 50} <= set(nums) else 42
            ans = good if right else rng.choice([good + 1, good - 3, 21, good * 2])
            return f"Let me work it out step by step... The answer is {ans}."
        return FREE.get(family, FREE["qwen"])
    return reply


async def main(args) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="myriad-demo-"))
    rng = random.Random(7)
    tracker = Tracker(db_path=tmp / "tracker.sqlite", starter_credit=1000.0, spot_rate=0.0, sweep_s=0.1)
    tport = args.tracker_port or free_port()
    tserver = uvicorn.Server(uvicorn.Config(tracker.app, host="127.0.0.1", port=tport, log_level="warning",
                                            ws_max_size=MAX_FRAME_BYTES))
    tasks = [asyncio.create_task(tserver.serve())]
    while not tserver.started:
        await asyncio.sleep(0.05)
    url = f"http://127.0.0.1:{tport}"

    def node(model, engine, mp=2):
        extra = [t for key, tags in DEMO_TAGS.items() if model and key in model for t in tags]
        return NodeClient(Identity.generate(), url, engine=engine, model=model, family=family_of(model) if model else None,
                          gguf="model.gguf" if model else None, params_b=params_of(model) if model else None, ctx=4096,
                          max_parallel=mp, tags=node_tags(model, extra))

    for i, (model, p, d) in enumerate(PEERS):
        fam = family_of(model)
        eng = FakeEngine(model, make_reply(p, fam, rng, model),
                         delay_s=lambda m, d=d: max(0.3, rng.lognormvariate(0, 0.35) * d), tokens=rng.randint(60, 220))
        n = node(model, eng, mp=rng.choice([1, 2, 2, 4]))
        if i % 7 == 6:
            n.accepting = False
        tasks.append(asyncio.create_task(n.run()))
    me_model = "ggml-org/SmolLM3-3B-GGUF"
    me_engine = FakeEngine(me_model, make_reply(0.86, "smollm", rng, me_model), delay_s=lambda m: rng.uniform(0.8, 2.0),
                           tokens=140)
    me = node(me_model, me_engine, mp=2)
    tasks.append(asyncio.create_task(me.run()))
    gw = Gateway(me, default_k=4, timeout_s=30)
    gw.verify_commands = demo_verify_commands()  # the Agents view's demo plan checks Python syntax locally
    home = tmp / "me"
    home.mkdir()
    cfg = Config(tracker_url=url, model=f"{me_model}:SmolLM3-Q4_K_M.gguf", family="smollm", params_b=3.1, max_parallel=2)
    ui = make_ui_app(me, gw, cfg, home, secrets.token_urlsafe(16))
    uport = args.port or free_port()
    servers = [uvicorn.Server(uvicorn.Config(ui, host="127.0.0.1", port=uport, log_level="warning"))]

    # First-run interface: an empty home, the real wizard (hardware detection is real, nothing is installed
    # unless you click "Install").
    home2 = tmp / "newcomer"
    home2.mkdir()
    rt2 = NodeRuntime(home2)
    rt2.config.tracker_url = url
    wiz = SetupWizard(home2, rt2)
    import myriad.wizard as wmod
    wmod.default_tracker = lambda: url  # the demo tracker as the proposed default
    ui2 = make_ui_app(config=rt2.config, home=home2, token=secrets.token_urlsafe(16), runtime=rt2, wizard=wiz)
    wport = args.wizard_port or free_port()
    servers.append(uvicorn.Server(uvicorn.Config(ui2, host="127.0.0.1", port=wport, log_level="warning")))
    tasks += [asyncio.create_task(s.serve()) for s in servers]

    # Background traffic from two other users of the network.
    clients = []
    for _ in range(2):
        c = node(None, None)
        tasks.append(asyncio.create_task(c.run()))
        clients.append(Gateway(c, default_k=4, timeout_s=30))
    questions = ["Tom has 40 apples and buys 2 more. How many apples does he have?",
                 "A pen costs 3 euros. Ana buys 7 and pays with 50 euros. How much change?",
                 "What is the capital of Australia? A) Sydney B) Canberra C) Melbourne D) Perth"]

    async def traffic(g: Gateway):
        await asyncio.sleep(2)
        while True:
            try:
                await g.ask([{"role": "user", "content": rng.choice(questions)}], max_tokens=256)
            except Exception:
                pass
            await asyncio.sleep(rng.uniform(0.3, 1.5) / max(args.load, 0.1))

    async def top_up():  # the two simulated users never run out of credits
        while True:
            for g in clients:
                tracker.ledger.db.execute("UPDATE accounts SET balance = ? WHERE node_id = ?",
                                          (10**12, g.node.node_id))
            # your own node: never below 5000 credits (agent runs spend a lot in a demo)
            tracker.ledger.db.execute("UPDATE accounts SET balance = MAX(balance, 5000000) WHERE node_id = ?",
                                      (me.node_id,))
            await asyncio.sleep(2)

    for g in clients:
        tasks.append(asyncio.create_task(traffic(g)))
    tasks.append(asyncio.create_task(top_up()))
    print(f"Tracker      : {url}  (it serves the landing page on /)")
    print(f"Dashboard    : http://127.0.0.1:{uport}/")
    print(f"Setup wizard : http://127.0.0.1:{wport}/", flush=True)
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--wizard-port", type=int, default=0)
    ap.add_argument("--tracker-port", type=int, default=0, help="port of the demo tracker (0: any free port)")
    ap.add_argument("--load", type=float, default=1.0, help="traffic multiplier")
    try:
        asyncio.run(main(ap.parse_args()))
    except KeyboardInterrupt:
        pass
