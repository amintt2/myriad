"""Hard Instruct pilot orchestration, reusing the frozen agent budgets, provenance and accounting."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from essaim import aider_agent, bigcode_tasks as tasks, code_pilot as shared
from essaim.code_provenance import declarations
from essaim.local_linux import VERSION, LocalLinuxEnv
from essaim.terminal_bench import summarize, validate_endpoint


def parser():
    cli = shared.parser()
    cli.description = "E17 BigCodeBench DEV 120/184, unittest isolé, aucun score officiel"
    cache = Path.home() / ".local/share/myriad-bigcode-pilot"
    cli.set_defaults(tasks_dir=cache / "data", runtime=cache / "venv")
    cli.add_argument("--source-cache", type=Path, help="Cache existant, copie DEV sans réseau")
    return cli


def run_one(args, task: str, mode: str, models: list[str], client, output: Path, campaign_hash: str) -> dict:
    copy = tasks.copy_task
    collect = tasks.collect_sources
    instruction_for = tasks.instructions
    grade_sources = tasks.verify
    sandbox, environment = "strict", LocalLinuxEnv
    started = time.monotonic()
    with (output / f"{mode}__{task}.jsonl").open("w", encoding="utf-8") as log:
        step_count = 0
        def record(value):
            nonlocal step_count
            step_count += value.get("kind") == "step"
            value = {"sandbox_backend": sandbox,
                     "sandbox": environment.version if sandbox != "strict" else VERSION, **value}
            log.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
        def official_test(sources, remaining):
            limit = min(args.verify_timeout, remaining)
            test_start = time.monotonic()
            grade = grade_sources(args.tasks_dir, task, sources, args.runtime, limit)
            if grade["status"] == "graded" and time.monotonic() - test_start >= limit:
                grade = {**grade, "passed": False, "feedback": [{"test": "controller",
                         "traceback": "Official verification exceeded its time budget; no green test recorded."}]}
            return grade
        row = {"task": task, "mode": mode, "campaign_hash": campaign_hash, "status": "error", "passed": None,
               "trust_scope": tasks.TRUST_SCOPE, "adversarial_attestation": False,
               "sandbox_backend": sandbox, "sandbox": environment.version if sandbox != "strict" else VERSION}
        chat = shared.BudgetChat(client, args.max_tokens, args.total_tokens, args.seed, started + args.timeout, record,
                                 strict=True)
        try:
            with tempfile.TemporaryDirectory(prefix="myriad-agent-") as temporary:
                work = Path(temporary)
                allowed = copy(args.tasks_dir, task, work)
                initial = collect(work, allowed)
                instruction = instruction_for(args.tasks_dir, task)
                system = aider_agent.SYSTEM
                record({"kind": "start", "task": task, "mode": mode, "instruction": instruction, "system": system})
                env = environment(work, args.runtime)
                try:
                    strategy = "single" if mode == "reference" else mode
                    loop = aider_agent.run_episode
                    options = {
                        "snapshot": lambda: collect(work, allowed),
                        "test_sources": official_test}
                    episode = loop(chat, env, instruction, models, strategy, args.max_steps, args.cmd_timeout,
                                          models, max(0.001, chat.deadline - time.monotonic()),
                                          lambda step: record({"kind": "step", "isolation": env.isolation,
                                                               **asdict(step)}),
                                          reference=args.reference if mode == "cascade" else None, **options)
                except tasks.SourceArtifactError:
                    episode = {"stopped": "invalid_source_artifact", "steps": []}
                finally:
                    env.close()
                    row["isolation"] = env.isolation
                    record({"kind": "isolation", "receipt": env.isolation})
                row.update(stopped=episode["stopped"], steps=step_count)
                sources = None
                try:
                    sources = collect(work, allowed)
                except tasks.SourceArtifactError:
                    # Only post-episode artifacts are candidate failures; initial preparation stays infrastructure.
                    grade = {"status": "graded", "passed": False, "candidate_failure": "invalid_source_artifact",
                             "tests": 0, "tests_expected": None, "tests_complete": False, "seconds": 0,
                             "trust_scope": tasks.TRUST_SCOPE, "adversarial_attestation": False,
                             "sandbox_backend": "strict", "sandbox": VERSION, "isolation": None, "scorable": True}
                    record({"kind": "changes", "initial": initial, "sources": None,
                            "candidate_failure": grade["candidate_failure"]})
                else:
                    row["source_hashes"] = {name: hashlib.sha256(text.encode()).hexdigest()
                                            for name, text in sources.items()}
                    record({"kind": "changes", "initial": initial, "sources": sources})
                    # Tests and any agent-written scores/artifacts are discarded, never trusted by the grader.
                    grade = grade_sources(args.tasks_dir, task, sources, args.runtime, args.verify_timeout)
                row.update(status=grade["status"], passed=grade["passed"], verifier=grade)
                if grade["status"] == "graded":
                    accepted = (episode["stopped"] == "submitted" and sources is not None
                                and sources == episode.get("tested_sources") and aider_agent.complete_green(grade))
                    accepted = accepted and grade["seconds"] < args.verify_timeout
                    row["passed"] = accepted
                    row["submission_accepted"] = accepted
                row["episode_seconds"] = time.monotonic() - started - grade["seconds"]
        except Exception as error:
            row.update(status="error", passed=None, exception_type=type(error).__name__)
        row.update(chat.accounting(), seconds=time.monotonic() - started)
        row.setdefault("episode_seconds", row["seconds"])
        record({"kind": "end", **row})
    row["transcript_sha256"] = hashlib.sha256((output / f"{mode}__{task}.jsonl").read_bytes()).hexdigest()
    return row


def validate_rows(rows: list, campaign: dict, output: Path, complete: bool) -> set:
    if (campaign.get("trust_scope") != tasks.TRUST_SCOPE or campaign.get("adversarial_attestation") is not False):
        raise ValueError("report trust scope mismatch")
    expected = {(m, t) for m in campaign["models"] for t in tasks.PILOT}
    seen = set()
    for row in rows:
        identity = (row["mode"], row["task"])
        if identity not in expected or identity in seen or row["campaign_hash"] != shared.hash_json(campaign):
            raise ValueError("episode identity/provenance mismatch")
        if (row["status"] != "graded" or type(row["passed"]) is not bool
                or row.get("trust_scope") != tasks.TRUST_SCOPE
                or row.get("adversarial_attestation") is not False):
            raise ValueError("infrastructure/incomplete episodes are not scorable or resumable")
        if row["passed"] and (row.get("submission_accepted") is not True
                              or not aider_agent.complete_green(row.get("verifier", {}))):
            raise ValueError("success requires authorized submission and complete final verification")
        raw = (output / (identity[0] + "__" + identity[1] + ".jsonl")).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["transcript_sha256"]:
            raise ValueError("transcript provenance mismatch")
        trace = [tasks.strict_json(line) for line in raw.decode("utf-8").splitlines()]
        end = [{k: v for k, v in row.items() if k != "transcript_sha256"}]
        if [r for r in trace if r.get("kind") == "end"] != [{"kind": "end", **end[0]}]:
            raise ValueError("transcript/verdict disagreement")
        requests = [r for r in trace if r.get("kind") == "request"]
        for request in requests:
            if (type(request.get("ceiling")) is not int or request["ceiling"] <= 0
                    or any(v is not None and (type(v) is not int or v < 0) for v in
                           (request.get("completion_tokens"), request.get("known_completion_tokens")))
                    or any(v is not None and v > request["ceiling"] for v in
                           (request.get("completion_tokens"), request.get("known_completion_tokens")))
                    or not math.isfinite(request["seconds"]) or request["seconds"] < 0):
                raise ValueError("invalid recorded request accounting")
        accounting = shared.BudgetChat.accounting(SimpleNamespace(requests=requests))
        if any(row.get(k) != v for k, v in accounting.items()):
            raise ValueError("request/episode accounting disagreement")
        if (not math.isfinite(row["seconds"]) or row["seconds"] < 0
                or ("limits" in campaign
                    and accounting["charged_request_ceilings"] > campaign["limits"]["total_tokens"])):
            raise ValueError("episode time/reservation limit mismatch")
        seen.add(identity)
    if complete and seen != expected:
        raise ValueError("report requires the complete predeclared grid")
    return seen


def report(rows: list, campaign: dict, output: Path) -> dict:
    validate_rows(rows, campaign, output, True)
    modes = list(campaign["models"])
    summary = {"scope": "two authorized DEV Hard Instruct tasks; not official pass@1 on 148", "paired": {},
               "trust_scope": tasks.TRUST_SCOPE, "adversarial_attestation": False,
               "modes": {m: summarize([r for r in rows if r["mode"] == m], [m], tasks.PILOT)["modes"][m]
                         for m in modes}}
    lines = ["# Pilote BigCodeBench-Hard Instruct", "", "Deux tâches DEV autorisées ; aucun score officiel "
             "sur 148, aucune représentativité. TEST non consulté. Boucle E17 existante, pas génération pass@1.",
             "", "Le temps mur inclut sélection, appels agent et vérification ; temps d'appels distinct. "
             "Énergie/coût hypothétiques, pas mesurés. Préparation et smoke préalables exclus du temps par tâche.",
             f"Hypothèses : {campaign['watts']} W agrégés ; {campaign['eur_kwh']} EUR/kWh.", "",
             "| stratégie | réussites | jetons observés | appels | plafonds réservés | s/tâche | Wh estimés/tâche |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        metric = summary["modes"][mode]
        seconds = sum(r["seconds"] for r in selected) / len(tasks.PILOT)
        counts = [r["completion_tokens"] for r in selected]
        wh = campaign["watts"] * seconds / 3600
        metric.update(seconds_per_task=seconds, completion_tokens=sum(counts) if None not in counts else None,
                      known_completion_tokens=sum(r["known_completion_tokens"] for r in selected),
                      unknown_usage_requests=sum(r["unknown_usage_requests"] for r in selected),
                      requests=sum(r["requests"] for r in selected),
                      request_seconds=sum(r["request_seconds"] for r in selected),
                      charged_request_ceilings=sum(r["charged_request_ceilings"] for r in selected),
                      estimated_wh_per_task=wh, estimated_eur_per_task=wh * campaign["eur_kwh"] / 1000)
        lines.append(f"| {mode} | {metric['passed']}/{len(tasks.PILOT)} | "
                     f"{metric['completion_tokens']} | {metric['requests']} | "
                     f"{metric['charged_request_ceilings']} | {seconds:.6f} | {wh:.9f} |")
    lines.extend(["", shared.RESERVATION_NOTE, "", "Notation publique non adversariale ; aucune attestation hostile."])
    shared.write_json(output / "summary.json", summary)
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shared.figures(summary, output)
    return summary


def execute(args) -> int:
    if args.sandbox_backend != "strict":
        raise ValueError("WSL Landlock strict required")
    manifest = tasks.provenance()
    if args.prepare:
        if args.source_cache is None:
            raise ValueError("--source-cache required; no downloads")
        tasks.prepare(args.tasks_dir, args.source_cache)
        print("Données épinglées vérifiées ; aucune inférence.")
        return 0
    tasks.check_data(args.tasks_dir)
    if args.check:
        if sys.platform.startswith("linux"):
            tasks.check_runtime()
        if args.provenance:
            modes = shared.model_modes(args)
            declarations(args.provenance, sorted({m for peers in modes.values() for m in peers}),
                         args.watts, args.eur_kwh)
        print("Deux tâches DEV et ressources vérifiées ; aucune inférence.")
        print("Notation publique non adversariale ; aucune attestation hostile.")
        return 0
    if args.report:
        if not args.output:
            raise ValueError("--output required")
        campaign = tasks.strict_json((args.output / "manifest.json").read_text(encoding="utf-8"))
        rows = tasks.strict_json((args.output / "verdicts.json").read_text(encoding="utf-8"))
        if (campaign["version"] != tasks.VERSION or campaign["tasks"] != tasks.PILOT
                or campaign["dataset_hash"] != shared.hash_json(manifest)):
            raise ValueError("report dataset/protocol mismatch")
        report(rows, campaign, args.output)
        return 0
    if not sys.platform.startswith("linux") or Path(sys.prefix).resolve() != args.runtime.resolve():
        raise ValueError("use the dedicated WSL runtime for smoke and inference")
    tasks.check_runtime()
    if args.smoke:
        result = tasks.smoke(args.tasks_dir, args.runtime)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            shared.write_json(args.output / "smoke.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if not args.output or not args.provenance:
        raise ValueError("--output and --provenance required before inference")
    models = shared.model_modes(args)
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in
           ("max_steps", "max_tokens", "total_tokens", "timeout", "cmd_timeout", "verify_timeout", "watts")):
        raise ValueError("positive finite limits required")
    if not math.isfinite(args.eur_kwh) or args.eur_kwh < 0:
        raise ValueError("nonnegative finite electricity price required")
    url, key = os.environ.get("MYRIAD_BENCH_URL", "http://127.0.0.1:8400/v1"), os.environ.get("MYRIAD_BENCH_TOKEN", "")
    validate_endpoint(url, key)
    declared = declarations(args.provenance, sorted({m for peers in models.values() for m in peers}),
                            args.watts, args.eur_kwh)
    code = ["bigcode_pilot.py", "bigcode_tasks.py", "code_pilot.py",
            "agent.py", "aider_agent.py", "local_linux.py", "bigcode_unittest.py",
            "aider_python.py", "code_provenance.py", "terminal_bench.py"]
    campaign = {"version": tasks.VERSION, "sandbox": VERSION, "dataset_hash": shared.hash_json(manifest),
                "tasks": tasks.PILOT, "models": models, "reference": args.reference, "execution_provenance": declared,
                "endpoint": url, "system": aider_agent.SYSTEM,
                "trust_scope": tasks.TRUST_SCOPE, "adversarial_attestation": False,
                "instructions": {t: tasks.instructions(args.tasks_dir, t) for t in tasks.PILOT},
                "limits": {k: getattr(args, k) for k in ("max_steps", "max_tokens", "total_tokens", "timeout",
                           "cmd_timeout", "verify_timeout", "seed")}, "watts": args.watts, "eur_kwh": args.eur_kwh,
                "platform": platform.platform(), "python": platform.python_version(),
                "environment_lock_sha256": tasks.digest((tasks.RESOURCE / "uv.lock").read_bytes())["sha256"],
                "runtime_versions": {p: importlib.metadata.version(p) for p in
                                     json.loads((tasks.RESOURCE / "runtime.json").read_text())["packages"]},
                "code_hashes": {p: tasks.digest((tasks.ROOT / "essaim" / p).read_bytes())["sha256"] for p in code}}
    path = args.output / "manifest.json"
    if path.exists():
        if not args.resume or tasks.strict_json(path.read_text(encoding="utf-8")) != campaign:
            raise ValueError("resume requires identical full provenance")
    elif any(p.name != ".campaign.lock" for p in args.output.iterdir()):
        raise ValueError("new output must be empty")
    else:
        shared.write_json(path, campaign)
    # Mandatory complete oracle and negative controls before the first request, including every resume.
    shared.write_json(args.output / "smoke.json", tasks.smoke(args.tasks_dir, args.runtime))
    verdicts = args.output / "verdicts.json"
    rows = tasks.strict_json(verdicts.read_text(encoding="utf-8")) if verdicts.exists() else []
    completed = validate_rows(rows, campaign, args.output, False)
    client = shared.DeadlineClient(url, key)
    for mode, peers in models.items():
        for task in tasks.PILOT:
            if (mode, task) in completed:
                continue
            if (args.output / f"{mode}__{task}.jsonl").exists():
                raise ValueError("orphan transcript requires a fresh campaign, never silent rerun")
            row = run_one(args, task, mode, peers, client, args.output, shared.hash_json(campaign))
            rows.append(row)
            shared.write_json(verdicts, rows)
            if row["status"] != "graded":
                return 2
    report(rows, campaign, args.output)
    return 0


def cli_main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.output and not (args.prepare or args.check or args.smoke):
        args.output.mkdir(parents=True, exist_ok=True)
        with shared.campaign_lock(args.output):
            return execute(args)
    return execute(args)
