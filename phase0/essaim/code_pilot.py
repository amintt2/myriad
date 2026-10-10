"""Reproducible descriptive code-agent pilot, without Docker or model downloads."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from essaim.agent import GenerationDeadline, GenerationLimit, SYSTEM, run_episode
from essaim.aider_python import (PILOT, ROOT, collect_sources, copy_task, prepare, provenance, smoke, task_config,
                                SourceArtifactError, verify)
from essaim.local_linux import LocalLinuxEnv, VERSION
from essaim.code_provenance import declarations
from essaim.terminal_bench import check_tasks, summarize, validate_endpoint

VERSION_PILOT = "code-pilot-v3"
RESERVATION_NOTE = "Les plafonds réservés sont des réservations de budget, ni consommation de jetons ni facture."
PUBLICATION_NOTE = ("Dérivation publique distincte : instantanés inchangés remplacés par références épinglées ; "
                    "chemins home masqués dans les champs textuels. Les commandes ne sont pas un rejeu textuel exact.")


def hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def campaign_lock(output: Path):
    lock = output / ".campaign.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "version": VERSION_PILOT}).encode())
    finally:
        os.close(fd)
    try:
        yield
    finally:
        lock.unlink()


class DeadlineClient:
    def __init__(self, url: str, key: str):
        self.url, self.key = url.rstrip("/") + "/", key

    def post(self, path, json, timeout):
        import httpx
        async def request():
            async with httpx.AsyncClient(base_url=self.url, headers={"Authorization": f"Bearer {self.key}"},
                                         trust_env=False, follow_redirects=False) as client:
                return await asyncio.wait_for(client.post(path, json=json, timeout=timeout), timeout)
        return asyncio.run(request())


class BudgetChat:
    """Reserve each request ceiling, including failed calls and the reference; never call it observed usage."""
    def __init__(self, client, max_tokens: int, total_tokens: int, seed: int, deadline: float, record):
        self.client, self.max_tokens, self.remaining = client, max_tokens, total_tokens
        self.seed, self.deadline, self.record = seed, deadline, record
        self.requests = []

    def __call__(self, model: str, messages: list[dict]) -> str:
        remaining_s = self.deadline - time.monotonic()
        if remaining_s <= 0:
            raise GenerationDeadline("episode deadline exhausted")
        if self.remaining <= 0:
            raise GenerationLimit("episode allowance exhausted")
        ceiling = min(self.max_tokens, self.remaining)
        self.remaining -= ceiling
        payload = {"model": model, "messages": messages, "max_tokens": ceiling, "seed": self.seed,
                   "temperature": 0.0, "myriad": {"k": 1, "early_stop": False, "format_instruction": False}}
        started = time.monotonic()
        row = {"kind": "request", "model": model, "parameters": payload, "ceiling": ceiling,
               "completion_tokens": None}
        try:
            response = self.client.post("chat/completions", json=payload, timeout=remaining_s)
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage")
            count = usage.get("completion_tokens") if isinstance(usage, dict) else None
            meta = data.get("myriad", data.get("essaim"))
            known_count = count
            if isinstance(meta, dict):
                known_count = meta.get("total_completion_tokens")
                count = known_count
                # Gateway usage describes the chosen answer; failed/replaced peers may have unreported work.
                if meta.get("replacements", 0) or meta.get("peers_asked", 0) != meta.get("peers_answered", 0):
                    count = None
            if any(v is not None and (type(v) is not int or v < 0) for v in (count, known_count)):
                raise ValueError("invalid completion usage")
            if known_count is not None and known_count > ceiling:
                raise ValueError("server exceeded the reserved completion ceiling")
            choice = data["choices"][0]
            message = choice["message"]
            reply = message.get("content") or ""
            if not isinstance(reply, str):
                raise ValueError("reply must be text")
            row.update(served_model=data.get("model"), usage=usage, myriad=meta, completion_tokens=count,
                       known_completion_tokens=known_count, reply=reply, message=message,
                       finish_reason=choice.get("finish_reason"),
                       server_headers={k: response.headers[k] for k in ("server", "x-server-version")
                                       if k in response.headers})
            return reply
        except Exception as error:
            row["exception_type"] = type(error).__name__
            import httpx
            if isinstance(error, (TimeoutError, httpx.TimeoutException)) and time.monotonic() >= self.deadline:
                row["stop_reason"] = "episode_deadline"
                raise GenerationDeadline("episode deadline expired during generation") from error
            raise
        finally:
            row["seconds"] = time.monotonic() - started
            self.requests.append(row)
            self.record(row)

    def accounting(self) -> dict:
        counts = [row["completion_tokens"] for row in self.requests]
        return {"completion_tokens": sum(counts) if all(v is not None for v in counts) else None,
                "known_completion_tokens": sum(row.get("known_completion_tokens") or 0 for row in self.requests),
                "unknown_usage_requests": sum(v is None for v in counts),
                "charged_request_ceilings": sum(row["ceiling"] for row in self.requests),
                "requests": len(counts), "request_seconds": sum(row["seconds"] for row in self.requests)}


def instructions(cache: Path, task: str) -> str:
    base, files = task_config(cache, task)
    docs = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted((base / ".docs").glob("instructions*.md")))
    return (f"Implement the Python exercise {task}. Working directory is the exercise copy. "
            f"Only these source files are submitted: {', '.join(files['solution'])}. "
            "Official tests are visible. You may run python -m pytest -q *_test.py. Do not alter tests. "
            "The final verifier restores the original tests and runs outside your interpreter.\n\n" + docs)


def run_one(args, task: str, mode: str, models: list[str], client, output: Path, campaign_hash: str,
            backend=None) -> dict:
    copy = backend.copy_task if backend else copy_task
    collect = backend.collect_sources if backend else collect_sources
    instruction_for = backend.instructions if backend else instructions
    grade_sources = backend.verify if backend else verify
    started = time.monotonic()
    with (output / f"{mode}__{task}.jsonl").open("w", encoding="utf-8") as log:
        def record(value):
            log.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
        row = {"task": task, "mode": mode, "campaign_hash": campaign_hash, "status": "error", "passed": None}
        chat = BudgetChat(client, args.max_tokens, args.total_tokens, args.seed, started + args.timeout, record)
        try:
            with tempfile.TemporaryDirectory(prefix="myriad-agent-") as temporary:
                work = Path(temporary)
                allowed = copy(args.tasks_dir, task, work)
                initial = collect(work, allowed)
                instruction = instruction_for(args.tasks_dir, task)
                record({"kind": "start", "task": task, "mode": mode, "instruction": instruction, "system": SYSTEM})
                env = LocalLinuxEnv(work, args.runtime)
                try:
                    strategy = "single" if mode == "reference" else mode
                    episode = run_episode(chat, env, instruction, models, strategy, args.max_steps, args.cmd_timeout,
                                          models, max(0.001, chat.deadline - time.monotonic()),
                                          lambda step: record({"kind": "step", **asdict(step)}),
                                          reference=args.reference if mode == "cascade" else None)
                finally:
                    env.close()
                    record({"kind": "isolation", "receipt": env.isolation})
                row.update(stopped=episode["stopped"], steps=len(episode["steps"]))
                try:
                    sources = collect(work, allowed)
                except SourceArtifactError:
                    # Only post-episode artifacts are candidate failures; initial preparation stays infrastructure.
                    grade = {"status": "graded", "passed": False, "candidate_failure": "invalid_source_artifact",
                             "tests": 0, "tests_expected": None, "tests_complete": False, "seconds": 0}
                    record({"kind": "changes", "initial": initial, "sources": None,
                            "candidate_failure": grade["candidate_failure"]})
                else:
                    row["source_hashes"] = {name: hashlib.sha256(text.encode()).hexdigest()
                                            for name, text in sources.items()}
                    record({"kind": "changes", "initial": initial, "sources": sources})
                    # Tests and any agent-written scores/artifacts are discarded, never trusted by the grader.
                    grade = grade_sources(args.tasks_dir, task, sources, args.runtime, args.verify_timeout)
                row.update(status=grade["status"], passed=grade["passed"], verifier=grade)
                row["episode_seconds"] = time.monotonic() - started - grade["seconds"]
        except Exception as error:
            row.update(status="error", passed=None, exception_type=type(error).__name__)
        row.update(chat.accounting(), seconds=time.monotonic() - started)
        row.setdefault("episode_seconds", row["seconds"])
        record({"kind": "end", **row})
    row["transcript_sha256"] = hashlib.sha256((output / f"{mode}__{task}.jsonl").read_bytes()).hexdigest()
    return row


def report(rows: list[dict], campaign: dict, output: Path, backend=None) -> dict:
    pilot = backend.PILOT if backend else PILOT
    modes = list(campaign["models"])
    expected = {(mode, task) for mode in modes for task in pilot}
    if (len(rows) != len(expected) or {(r["mode"], r["task"]) for r in rows} != expected
            or any(r["status"] != "graded" or type(r["passed"]) is not bool for r in rows)):
        raise ValueError("final report requires all predeclared episodes and valid verdicts")
    if any(r["campaign_hash"] != hash_json(campaign) for r in rows):
        raise ValueError("campaign provenance mismatch")
    for row in rows:
        transcript = output / f"{row['mode']}__{row['task']}.jsonl"
        if hashlib.sha256(transcript.read_bytes()).hexdigest() != row["transcript_sha256"]:
            raise ValueError("transcript provenance mismatch")
    # Reuse the existing completeness/Wilson helper; paired testing adds no useful claim with n=3.
    summary = {"scope": "three preselected tasks, not the full benchmark", "paired": {},
               "modes": {mode: summarize([r for r in rows if r["mode"] == mode], [mode], pilot)["modes"][mode]
                         for mode in modes}}
    lines = ["# Pilote descriptif Aider Python", "", "Trois exercices présélectionnés ; aucune généralisation aux 225 "
             "exercices, aux six langages ou à un dépôt logiciel réel.", "",
             "L'énergie et le coût ci-dessous sont ESTIMÉS, pas mesurés ; ils couvrent le temps mur séquentiel "
             "agent + vérification. Les appels distants demandent une hypothèse de puissance agrégée appropriée.", "",
             f"Hypothèses : {campaign['watts']} W, {campaign['eur_kwh']} €/kWh.", "",
             "| stratégie | réussites | Wilson 95 % | jetons réels | plafonds réservés | s/tâche "
             "| Wh/tâche estimés | €/tâche estimés |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    if backend:
        lines[:4] = ["# Pilote descriptif SymPy sur dépôt réel", "", "Trois réparations historiques présélectionnées ; "
                     "assertions de chaînes seulement, aucun score SWE-bench officiel ni couverture PASS_TO_PASS.", ""]
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        metric = summary["modes"][mode]
        seconds = sum(r["seconds"] for r in selected) / len(pilot)
        wh = campaign["watts"] * seconds / 3600
        counts = [r["completion_tokens"] for r in selected]
        tokens = sum(counts) if all(v is not None for v in counts) else None
        metric.update(seconds_per_task=seconds, estimated_wh_per_task=wh,
                      estimated_eur_per_task=wh * campaign["eur_kwh"] / 1000, completion_tokens=tokens,
                      charged_request_ceilings=sum(r["charged_request_ceilings"] for r in selected))
        low, high = metric["wilson_95"]
        lines.append(f"| {mode} | {metric['passed']}/3 | [{low:.3f}, {high:.3f}] | "
                     f"{tokens if tokens is not None else 'inconnu'} | {metric['charged_request_ceilings']} | "
                     f"{seconds:.6f} | {wh:.9f} | {metric['estimated_eur_per_task']:.12f} |")
    lines.extend(["", RESERVATION_NOTE])
    if (output / "publication.json").is_file():
        lines.extend(["", PUBLICATION_NOTE])
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures(summary, output)
    return summary


def figures(summary: dict, output: Path) -> None:
    """Standard scientific plots from complete results only; import plotting dependencies on demand."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    for key, title in (("seconds_per_task", "Temps mur par tâche (s)"),
                       ("estimated_eur_per_task", "Coût énergétique ESTIMÉ par tâche (€)")):
        fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
        try:
            for mode, metric in summary["modes"].items():
                low, high = metric["wilson_95"]
                score = metric["pass_at_1"]
                ax.errorbar(metric[key], score, yerr=[[max(0, score - low)], [max(0, high - score)]],
                            fmt="o", capsize=4, label=mode)
            ax.set(xlabel=title, ylabel="Exactitude", ylim=(-0.03, 1.03),
                   title="Pilote descriptif présélectionné, n=3 ; Wilson 95 %")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
            ax.grid(alpha=0.25)
            ax.legend()
            metadata = {"Description": json.dumps(summary["modes"], ensure_ascii=False)}
            fig.savefig(output / f"accuracy_{key}.svg", metadata=metadata)
            fig.savefig(output / f"accuracy_{key}.png", dpi=180)
            fig.savefig(output / f"accuracy_{key}.pdf")
        finally:
            plt.close(fig)


def model_modes(args) -> dict:
    if not args.single:
        raise ValueError("--single is required")
    if args.vote and not args.reference:
        raise ValueError("--vote requires --reference for the predeclared comparison")
    if args.vote and (len(args.vote) < 2 or len(set(args.vote)) != len(args.vote) or args.reference in args.vote):
        raise ValueError("distinct peers and a separate reference are required")
    models = {"single": [args.single]}
    if args.vote:
        models.update(vote=args.vote, cascade=args.vote)
    if args.reference:
        models["reference"] = [args.reference]
    return models


def parser(backend=None) -> argparse.ArgumentParser:
    name = "SymPy sur dépôt réel" if backend else "Aider Python"
    directory = "repo-pilot" if backend else "code-pilot"
    cli = argparse.ArgumentParser(description=f"Pilote reproductible {name} sans Docker, trois tâches fixes")
    action = cli.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true", help="Télécharger les sources publiques épinglées")
    action.add_argument("--check", action="store_true", help="Vérifier les fichiers sans inférence")
    action.add_argument("--smoke", action="store_true", help="Références officielles et solutions incorrectes, sans modèle")
    action.add_argument("--report", action="store_true", help="Rapport uniquement sur campagne complète")
    cli.add_argument("--tasks-dir", type=Path, default=ROOT / "data" / directory)
    runtime = Path.home() / ".local/share" / ("myriad-" + directory) / "venv"
    cli.add_argument("--runtime", type=Path, default=runtime)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--single")
    cli.add_argument("--vote", nargs="+")
    cli.add_argument("--reference")
    cli.add_argument("--provenance", type=Path,
                     help="JSON nonsecret modèle/serveur/matériel/raisonnement avant inférence")
    cli.add_argument("--max-steps", type=int, default=20)
    cli.add_argument("--max-tokens", type=int, default=1024)
    cli.add_argument("--total-tokens", type=int, default=12288)
    cli.add_argument("--timeout", type=float, default=900)
    cli.add_argument("--cmd-timeout", type=float, default=30)
    cli.add_argument("--verify-timeout", type=float, default=120)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--watts", type=float, default=100)
    cli.add_argument("--eur-kwh", type=float, default=0.25)
    return cli


def execute(args, backend=None) -> int:
    manifest = backend.provenance() if backend else provenance()
    pilot = backend.PILOT if backend else PILOT
    prepare_tasks = backend.prepare if backend else prepare
    check_smoke = backend.smoke if backend else smoke
    instruction_for = backend.instructions if backend else instructions
    resource = "repo_pilot" if backend else "code_pilot"
    if args.prepare:
        prepare_tasks(args.tasks_dir, manifest)
        print("Sources vérifiées ; pilote fixe : " + ", ".join(pilot))
        return 0
    check_tasks(args.tasks_dir, manifest)
    if args.check:
        if args.provenance:
            modes = model_modes(args)
            declarations(args.provenance, sorted({m for peers in modes.values() for m in peers}),
                         args.watts, args.eur_kwh)
        print(f"Provenance vérifiée ({len(manifest['files'])} fichiers), aucune inférence.")
        return 0
    if not sys.platform.startswith("linux"):
        raise ValueError("run execution from WSL with its separate frozen uv environment")
    args.runtime = args.runtime.resolve()
    if Path(sys.prefix).resolve() != args.runtime:
        raise ValueError("use the dedicated runtime interpreter for both orchestration and execution")
    if backend:
        backend.check_runtime()
    if args.smoke:
        result = check_smoke(args.tasks_dir, args.runtime)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / "smoke.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if not args.output:
        raise ValueError("--output is required")
    if args.report:
        campaign = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        rows = json.loads((args.output / "verdicts.json").read_text(encoding="utf-8"))
        report(rows, campaign, args.output, backend)
        return 0
    models = model_modes(args)
    if not args.provenance:
        raise ValueError("--provenance is required before inference")
    if any(not math.isfinite(v) or v <= 0 for v in (args.max_steps, args.max_tokens, args.total_tokens,
                                                  args.timeout, args.cmd_timeout, args.verify_timeout, args.watts)):
        raise ValueError("positive finite limits required")
    if not math.isfinite(args.eur_kwh) or args.eur_kwh < 0:
        raise ValueError("nonnegative finite electricity price required")
    url, key = os.environ.get("MYRIAD_BENCH_URL", "http://127.0.0.1:8400/v1"), os.environ.get("MYRIAD_BENCH_TOKEN", "")
    validate_endpoint(url, key)
    execution_provenance = declarations(args.provenance, sorted({m for peers in models.values() for m in peers}),
                                        args.watts, args.eur_kwh)
    campaign = {"version": backend.VERSION if backend else VERSION_PILOT, "sandbox": VERSION,
                "dataset_hash": hash_json(manifest), "tasks": pilot, "models": models,
                "reference": args.reference,
                "execution_provenance": execution_provenance,
                "endpoint": url, "system": SYSTEM, "instructions": {t: instruction_for(args.tasks_dir, t) for t in pilot},
                "limits": {k: getattr(args, k) for k in ("max_steps", "max_tokens", "total_tokens", "timeout",
                           "cmd_timeout", "verify_timeout", "seed")}, "watts": args.watts, "eur_kwh": args.eur_kwh,
                "platform": platform.platform(), "python": platform.python_version(),
                "environment_lock_sha256": hashlib.sha256((ROOT / resource / "uv.lock").read_bytes()).hexdigest(),
                "runtime_versions": {p: importlib.metadata.version(p) for p in ("httpx", "pytest", "matplotlib")},
                "code_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), ROOT / "essaim/agent.py", ROOT / "essaim/local_linux.py",
                                 ROOT / "essaim/aider_python.py", ROOT / "essaim/code_provenance.py"]}}
    if backend:
        campaign["code_hashes"].update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                      [ROOT / "essaim/repo_pilot.py", ROOT / "essaim/repo_tasks.py"]})
        campaign["runtime_versions"]["mpmath"] = importlib.metadata.version("mpmath")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text(encoding="utf-8")) != campaign:
            raise ValueError("existing campaign requires --resume and identical provenance")
    elif any(p.name != ".campaign.lock" for p in args.output.iterdir()):
        raise ValueError("new campaign output must be empty")
    else:
        write_json(manifest_path, campaign)
    # Smoke is mandatory before every invocation that may issue inference requests, including resume.
    write_json(args.output / "smoke.json", check_smoke(args.tasks_dir, args.runtime))
    rows_path = args.output / "verdicts.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    completed = set()
    for row in rows:
        identity = (row["mode"], row["task"])
        if identity in completed or identity[0] not in models or identity[1] not in pilot:
            raise ValueError("duplicate or undeclared resumed episode")
        if row["campaign_hash"] != hash_json(campaign) or row["status"] != "graded":
            raise ValueError("resume refuses incomplete/error episodes; use a fresh campaign")
        transcript = args.output / f"{identity[0]}__{identity[1]}.jsonl"
        if hashlib.sha256(transcript.read_bytes()).hexdigest() != row["transcript_sha256"]:
            raise ValueError("resumed transcript changed")
        completed.add(identity)
    client = DeadlineClient(url, key)
    for mode, peers in models.items():
        for task in pilot:
            if (mode, task) in completed:
                continue
            transcript = args.output / f"{mode}__{task}.jsonl"
            if transcript.exists():
                raise ValueError("orphan transcript: interrupted episode cannot be silently rerun")
            row = run_one(args, task, mode, peers, client, args.output, hash_json(campaign), backend)
            rows.append(row)
            write_json(rows_path, rows)
            print(f"{mode}/{task}: {row['status']}, passed={row['passed']}", flush=True)
            if row["status"] != "graded":
                return 2
    report(rows, campaign, args.output, backend)
    return 0


def cli_main(argv=None, backend=None) -> int:
    args = parser(backend).parse_args(argv)
    if args.output and not (args.prepare or args.check or args.smoke):
        args.output.mkdir(parents=True, exist_ok=True)
        with campaign_lock(args.output):
            return execute(args, backend)
    return execute(args, backend)
