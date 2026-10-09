"""Pinned three-task preparation and sequential trials through the official Harbor API."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "harbor" / "tasks.json"


def validate_endpoint(url: str, token: str) -> None:
    parsed = urlsplit(url)
    local = parsed.hostname in ("127.0.0.1", "localhost", "::1")
    if (parsed.scheme not in ("http", "https") or (parsed.scheme == "http" and not local)
            or parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname):
        raise ValueError("use a loopback HTTP or authenticated HTTPS endpoint without embedded credentials")
    if not token.strip():
        raise ValueError("MYRIAD_BENCH_TOKEN is required (environment only)")


def provenance() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def task_path(directory: Path, relative: str) -> Path:
    path = directory / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("task paths must stay within the cache")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("symlinks are not allowed in the task cache")
        if parent == directory:
            break
    return path


def check_tasks(directory: Path, manifest: dict) -> None:
    expected = set(manifest["files"])
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("task files differ from the pinned selection (missing or extra files)")
    for relative, info in manifest["files"].items():
        path = task_path(directory, relative)
        data = path.read_bytes()
        if len(data) != info["size"] or digest(data) != info["sha256"]:
            raise ValueError(f"task provenance mismatch: {relative}")


def prepare(directory: Path, manifest: dict) -> None:
    for relative, info in manifest["files"].items():
        path = task_path(directory, relative)
        if path.exists() and digest(path.read_bytes()) == info["sha256"]:
            continue
        url = ("https://raw.githubusercontent.com/harbor-framework/terminal-bench/"
               f"{manifest['commit']}/{relative}")
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(info["size"] + 1)
        if len(data) != info["size"] or digest(data) != info["sha256"]:
            raise ValueError(f"download provenance mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    check_tasks(directory, manifest)


def check_harbor(manifest: dict) -> None:
    distribution = importlib.metadata.distribution("harbor")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if (distribution.version != manifest["harbor_version"]
            or direct.get("vcs_info", {}).get("commit_id") != manifest["harbor_commit"]):
        raise ValueError("use the dedicated Harbor environment and its frozen lock")


def verdict(result: dict) -> dict:
    rewards = (result.get("verifier_result") or {}).get("rewards")
    exception = result.get("exception_info") or {}
    kind = exception.get("exception_type")
    valid = isinstance(rewards, dict) and set(rewards) == {"reward"} and rewards["reward"] in (0, 1)
    agent_stop = kind in ("AgentTimeoutError", "NonZeroAgentExitCodeError")
    status = "graded" if valid and (not kind or agent_stop) else "error"
    return {"status": status, "passed": status == "graded" and rewards["reward"] == 1,
            "official_rewards": rewards, "exception_type": kind}


def summarize(rows: list[dict], modes: list[str], tasks: list[str]) -> dict:
    report = {"scope": "three preselected tasks, not the full benchmark", "modes": {}, "paired": {}}
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        complete = (len(selected) == len(tasks) and {r["task"] for r in selected} == set(tasks)
                    and all(r["status"] == "graded" for r in selected))
        wins, n = sum(r["passed"] for r in selected), len(tasks)
        interval = None
        if complete:
            z, p = 1.959963984540054, wins / n
            center = (p + z * z / (2 * n)) / (1 + z * z / n)
            radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            interval = [max(0.0, center - radius), min(1.0, center + radius)]
        report["modes"][mode] = {"complete": complete, "passed": wins, "expected": n,
                                  "pass_at_1": wins / n if complete else None, "wilson_95": interval}
    if "vote" in modes and all(value["complete"] for value in report["modes"].values()):
        from essaim.stats import compare
        vectors = {mode: [next(r["passed"] for r in rows if r["mode"] == mode and r["task"] == task)
                          for task in tasks] for mode in modes}
        report["paired"] = {mode: compare(vectors["vote"], vectors[mode], delta_points=2.0)
                            for mode in modes if mode != "vote"}
    return report


def trial_config(args, task: Path, mode: str, families: list[str], output: Path, run_id: str):
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig

    return TrialConfig(
        task=TaskConfig(path=task), trial_name=f"{run_id}__{mode}__{task.name}", trials_dir=output,
        agent=AgentConfig(import_path="essaim.harbor_agent:MyriadAgent", model_name=families[0],
                          override_timeout_sec=args.timeout, kwargs={
                              "families": families, "strategy": "vote" if mode == "vote" else "single",
                              "max_steps": args.max_steps, "timeout_s": args.timeout,
                              "cmd_timeout_s": args.cmd_timeout, "max_tokens": args.max_tokens,
                              "total_tokens": args.total_tokens, "seed": args.seed}),
        environment=EnvironmentConfig(type="docker", delete=True, force_build=True))


async def run_trials(args, manifest: dict, output: Path, run_id: str) -> bool:
    from harbor.trial.trial import Trial

    modes = {"single": [args.single]}
    if args.compare:
        modes.update(vote=args.vote, reference=[args.reference])
    rows = []
    for mode, families in modes.items():
        for task in manifest["tasks"]:
            config = trial_config(args, args.tasks_dir / "tasks" / task, mode, families, output / "trials", run_id)
            try:
                trial = await Trial.create(config)
                result = (await trial.run()).model_dump(mode="json")
                grade = verdict(result)
            except Exception as error:
                grade = {"status": "error", "passed": False, "official_rewards": None,
                         "exception_type": type(error).__name__}
            row = {"mode": mode, "task": task, **grade, "trial_name": config.trial_name}
            rows.append(row)
            (output / "verdicts.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            print(f"{mode} / {task}: {row['status']}, verifier={row['official_rewards']}", flush=True)
    report = summarize(rows, list(modes), manifest["tasks"])
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return all(value["complete"] for value in report["modes"].values())


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Essai Terminal-Bench 4.0 : trois tâches épinglées, Harbor officiel")
    action = cli.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true", help="Télécharger uniquement les trois tâches publiques")
    action.add_argument("--check", action="store_true", help="Vérifier provenance et API Harbor sans Docker ni modèle")
    cli.add_argument("--tasks-dir", type=Path, default=ROOT / "data" / "terminal-bench" / "selected")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--single")
    cli.add_argument("--vote", nargs="+")
    cli.add_argument("--reference")
    cli.add_argument("--compare", action="store_true")
    cli.add_argument("--max-steps", type=int, default=100)
    cli.add_argument("--timeout", type=float, default=1800)
    cli.add_argument("--cmd-timeout", type=float, default=60)
    cli.add_argument("--max-tokens", type=int, default=1024)
    cli.add_argument("--total-tokens", type=int, default=102400)
    cli.add_argument("--seed", type=int, default=0)
    return cli


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    manifest = provenance()
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2), flush=True)
    if args.prepare:
        prepare(args.tasks_dir, manifest)
        print("Provenance vérifiée ; aucun conteneur ni modèle lancé.")
        return 0
    check_tasks(args.tasks_dir, manifest)
    check_harbor(manifest)
    from harbor.models.task.task import Task
    for task in manifest["tasks"]:
        Task(args.tasks_dir / "tasks" / task)
    if args.check:
        from essaim.harbor_agent import MyriadAgent
        print(f"API Harbor et agent {MyriadAgent.name()} chargés ; aucun résultat mesuré.")
        return 0
    if not args.single or not args.output:
        raise ValueError("--single and a new --output directory are required")
    if args.compare and (not args.vote or len(set(args.vote)) != len(args.vote) or len(args.vote) < 2
                         or args.single not in args.vote or not args.reference):
        raise ValueError("comparison requires distinct vote models including single, plus a reference")
    for limit in (args.max_steps, args.timeout, args.cmd_timeout, args.max_tokens, args.total_tokens):
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("all limits must be positive and finite")
    validate_endpoint(os.environ.get("MYRIAD_BENCH_URL", "http://127.0.0.1:8400/v1"),
                      os.environ.get("MYRIAD_BENCH_TOKEN", ""))
    subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=20)
    args.output.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    code = {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in (
        Path(__file__), ROOT / "essaim" / "agent.py", ROOT / "essaim" / "harbor_agent.py",
        ROOT / "essaim" / "stats.py", ROOT / "run_terminal_bench.py", ROOT / "harbor" / "uv.lock")}
    record = {"run_id": run_id, "provenance": manifest, "code_sha256": code, "configuration": vars(args).copy(),
              "generation_budget": "sum of requested max_tokens, reserved before each call"}
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return 0 if asyncio.run(run_trials(args, manifest, args.output, run_id)) else 1


def cli_main() -> int:
    try:
        return main()
    except (ValueError, OSError, subprocess.SubprocessError, importlib.metadata.PackageNotFoundError) as error:
        print(f"Essai bloqué : {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())
