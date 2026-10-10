"""Strict offline publication of one unchanged-source SymPy pilot; never execute candidate code."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from pathlib import Path

from essaim import code_pilot, repo_tasks

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = "repo_pilot_cpu_20261010_01"
PUBLIC = "repo_pilot_public_cpu_20261010_01"
SCHEMA = "repo-pilot-publication-v1"
DATASET_HASH = "7134d1e468382330ecb350a2ed8bd7a67b48f11ddfe7c1c5a0fded1afbc8685d"
LIMITS = dict(max_steps=20, max_tokens=1024, total_tokens=12288, timeout=900,
              cmd_timeout=30, verify_timeout=120, seed=0)
MODEL = "Qwen3.5-2B"
ARTIFACTS = {"manifest.json", "verdicts.json", "summary.json", "report.md"} | {
    f"single__{task}.jsonl" for task in repo_tasks.PILOT
} | {f"accuracy_{metric}.{ext}" for metric in ("seconds_per_task", "estimated_eur_per_task")
     for ext in ("png", "svg", "pdf")}
TEXT_FIELDS = {"content", "reasoning_content", "reply", "command", "output", "diagnostics"}
HOME = re.compile(r"/(?:home|Users)/[^/\s'\"<>:]+|/root(?=/|\b)|"
                  r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\r\n'\"<>]+", re.I)
FIGURE_EPOCH = "1791626767"
RULES = ["unchanged pinned sources only", "changes replaced by pinned_source_references",
         "home prefixes replaced with <HOME> only in text fields",
         "scientific fields unchanged; public transcript bindings recomputed",
         "commands are not an exact textual replay; originals remain private"]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def inventory(raw: dict[str, bytes]) -> dict:
    return {name: {"bytes": len(value), "sha256": digest(value)} for name, value in sorted(raw.items())}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def read_files(directory: Path, names) -> dict[str, bytes]:
    require(not any(p.is_symlink() for p in (directory, *directory.parents)), "linked input directory")
    result = {}
    for name in sorted(names):
        path = directory / name
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_size <= 32 << 20, "nonregular or oversized input")
        raw = path.read_bytes()
        after = path.lstat()
        require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns), "input changed while reading")
        result[name] = raw
    return result


def load_json(raw: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, "nonfinite JSON number"))


def frozen_inputs() -> dict:
    return load_json((ROOT / "repo_pilot/cpu_20261010_private_hashes.json").read_bytes())


def generators() -> dict:
    return {"publication": {"file": "repo_publication.py", "sha256": digest(Path(__file__).read_bytes())},
            "report": {"file": "code_pilot.py", "sha256": digest(Path(code_pilot.__file__).read_bytes())}}


def pinned_sources(task: str, dataset: dict) -> dict:
    cfg = dataset["tasks"][task]
    return {name: {"bytes": dataset["files"][task + "/base/" + name]["size"],
                   "sha256": dataset["files"][task + "/base/" + name]["sha256"]}
            for name in cfg["sources"]}


def source_reference(task: str, dataset: dict) -> dict:
    cfg = dataset["tasks"][task]
    return {"kind": "pinned_source_references", "repository": cfg["repo"], "base_commit": cfg["base_commit"],
            "archive": cfg["archive"], "initial_and_final_identical": True,
            "files": pinned_sources(task, dataset)}


def validate(campaign: dict, rows: list, records: dict, dataset: dict, public: bool = False) -> None:
    """Check scientific bindings independently of the immutable-file allowlist."""
    require(code_pilot.hash_json(dataset) == DATASET_HASH == campaign["dataset_hash"], "unpinned dataset")
    require(campaign["tasks"] == repo_tasks.PILOT and campaign["models"] == {"single": [MODEL]}
            and campaign["reference"] is None and campaign["version"] == "repo-pilot-v1", "wrong campaign grid")
    require(campaign["limits"] == LIMITS and campaign["watts"] == 100 and campaign["eur_kwh"] == 0.25,
            "changed campaign parameters")
    require(len(rows) == 3 and [row["task"] for row in rows] == repo_tasks.PILOT, "wrong verdict grid")
    for row in rows:
        task = row["task"]
        log = records[task]
        require(row["campaign_hash"] == code_pilot.hash_json(campaign) and row["mode"] == "single"
                and row["status"] == "graded" and row["passed"] is False, "wrong campaign binding or grade")
        expected_kinds = ["start"] + [kind for _ in range(12) for kind in ("request", "step")]
        expected_kinds += ["isolation", "pinned_source_references" if public else "changes", "end"]
        require([record["kind"] for record in log] == expected_kinds, "unexpected transcript records")
        require(log[0] == {"kind": "start", "task": task, "mode": "single",
                           "instruction": campaign["instructions"][task], "system": campaign["system"]},
                "wrong transcript identity or instructions")
        require(log[-1] == {"kind": "end", **{k: v for k, v in row.items() if k != "transcript_sha256"}},
                "end/verdict disagreement")
        source = log[-2]
        pinned = pinned_sources(task, dataset)
        if public:
            require(source == source_reference(task, dataset), "changed public source references")
        else:
            require(set(source) == {"kind", "initial", "sources"} and source["initial"] == source["sources"],
                    "changed final sources cannot use this publication")
            require(set(source["initial"]) == set(pinned), "undeclared or missing source")
            actual = inventory({name: value.encode("utf-8") for name, value in source["initial"].items()})
            require(actual == pinned, "initial or final source is not pinned")
        require(row["source_hashes"] == {name: info["sha256"] for name, info in pinned.items()},
                "wrong verdict source hashes")
        requests = log[1:25:2]
        for request in requests:
            parameters = request["parameters"]
            require(set(parameters) == {"model", "messages", "max_tokens", "seed", "temperature", "myriad"}
                    and parameters["model"] == request["model"] == request["served_model"] == MODEL
                    and parameters["max_tokens"] == request["ceiling"] == 1024
                    and parameters["seed"] == 0 and parameters["temperature"] == 0
                    and parameters["myriad"] == {"k": 1, "early_stop": False, "format_instruction": False},
                    "wrong request identity or parameters")
            count = request["completion_tokens"]
            require(type(count) is int and 0 <= count <= 1024 and count == request["known_completion_tokens"]
                    == request["usage"]["completion_tokens"] and request["myriad"] is None,
                    "wrong usage accounting")
            require(math.isfinite(request["seconds"]) and request["seconds"] >= 0, "invalid request duration")
        require(row["requests"] == row["steps"] == 12 and row["stopped"] == "generation_limit"
                and row["unknown_usage_requests"] == 0 and row["charged_request_ceilings"] == 12288
                and row["completion_tokens"] == row["known_completion_tokens"]
                == sum(r["completion_tokens"] for r in requests)
                and row["request_seconds"] == sum(r["seconds"] for r in requests), "wrong episode accounting")
        verifier = row["verifier"]
        require(verifier["status"] == "graded" and verifier["passed"] is False
                and verifier["tests_expected"] == dataset["tasks"][task]["cases"]
                and verifier["tests"] == len(verifier["calls"]) and verifier["fail_fast"] is True
                and verifier["tests_complete"] is False and not verifier["broken"]
                and verifier["calls"][-1]["candidate_failure"] == "wrong_string", "wrong verifier outcome")
        require(math.isfinite(row["seconds"]) and row["seconds"] > 0 and row["episode_seconds"] >= 0
                and row["request_seconds"] <= row["episode_seconds"] <= row["seconds"] - verifier["seconds"],
                "wrong episode duration")


def sanitize(value, counts: Counter, path: tuple = ()):
    if isinstance(value, dict):
        require(not any(HOME.search(key) for key in value), "home path in a JSON key")
        return {key: sanitize(item, counts, path + (key,)) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, counts, path + (str(index),)) for index, item in enumerate(value)]
    if isinstance(value, str):
        cleaned, count = HOME.subn("<HOME>", value)
        if count:
            require(path[-1] in TEXT_FIELDS or "replies" in path, "redaction would change a scientific field")
            counts["/".join(path)] += count
        return cleaned
    return value


def render(directory: Path, rows: list, campaign: dict) -> None:
    """Use fixed figure metadata only for this offline derivation, restoring the process environment."""
    import matplotlib
    previous = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = FIGURE_EPOCH
    try:
        with matplotlib.rc_context({"svg.hashsalt": SCHEMA}):
            code_pilot.report(rows, campaign, directory, repo_tasks)
    finally:
        if previous is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous


def derive(source: Path, output: Path) -> dict:
    raw = read_files(source, ARTIFACTS)
    require(inventory(raw) == frozen_inputs(), "private artifacts differ from the immutable campaign")
    campaign, rows = load_json(raw["manifest.json"]), load_json(raw["verdicts.json"])
    dataset = repo_tasks.provenance()
    records = {}
    for row in rows:
        name = "single__" + row["task"] + ".jsonl"
        require(digest(raw[name]) == row["transcript_sha256"], "private transcript hash disagreement")
        records[row["task"]] = [load_json(line) for line in raw[name].splitlines()]
    validate(campaign, rows, records, dataset)
    require(not output.exists(), "publication output must be new")
    require(output.resolve() != source.resolve(), "publication cannot overwrite private artifacts")
    counts = Counter()
    public_rows = sanitize(copy.deepcopy(rows), counts, ("verdicts",))
    public_logs = {}
    for row in public_rows:
        task = row["task"]
        replaced = records[task][:-2] + [source_reference(task, dataset), records[task][-1]]
        public_logs[task] = sanitize(replaced, counts, ("transcripts", task))
        encoded = b"".join((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                           for record in public_logs[task])
        row["transcript_sha256"] = digest(encoded)
    validate(campaign, public_rows, public_logs, dataset, public=True)
    output.mkdir(parents=True)
    (output / "manifest.json").write_bytes(raw["manifest.json"])
    for row in public_rows:
        log = public_logs[row["task"]]
        (output / ("single__" + row["task"] + ".jsonl")).write_bytes(b"".join(
            (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for record in log))
    code_pilot.write_json(output / "verdicts.json", public_rows)
    publication = {"schema": SCHEMA, "private_campaign": PRIVATE, "public_campaign": PUBLIC,
                   "original_artifacts": inventory(raw), "campaign_hash": code_pilot.hash_json(campaign),
                   "rules": RULES,
                   "redactions": {"occurrences": sum(counts.values()), "fields": dict(sorted(counts.items()))},
                   "source_snapshot_records_replaced": 3, "figure_epoch": FIGURE_EPOCH,
                   "generators": generators()}
    code_pilot.write_json(output / "publication.json", publication)
    render(output, public_rows, campaign)
    require((output / "summary.json").read_bytes() == raw["summary.json"], "summary changed")
    for name in ARTIFACTS:
        if name.endswith(".png"):
            require((output / name).read_bytes() == raw[name], "figure pixels changed")
    expected_report = raw["report.md"].decode().replace("plafonds facturés", "plafonds réservés")
    expected_report += "\n" + code_pilot.RESERVATION_NOTE + "\n\n" + code_pilot.PUBLICATION_NOTE + "\n"
    require((output / "report.md").read_text(encoding="utf-8") == expected_report, "unexpected report changes")
    publication["public_artifacts"] = inventory(read_files(output, ARTIFACTS))
    code_pilot.write_json(output / "publication.json", publication)
    require(inventory(read_files(source, ARTIFACTS)) == inventory(raw), "private artifacts changed")
    return publication


def reproduce(directory: Path) -> None:
    """Validate and regenerate a public copy without needing any private source snapshots."""
    raw = read_files(directory, ARTIFACTS | {"publication.json"})
    publication = load_json(raw.pop("publication.json"))
    require(set(publication) == {"schema", "private_campaign", "public_campaign", "original_artifacts",
                                 "campaign_hash", "rules", "redactions", "source_snapshot_records_replaced",
                                 "figure_epoch", "generators", "public_artifacts"}, "unexpected publication fields")
    require(publication["schema"] == SCHEMA and publication["private_campaign"] == PRIVATE
            and publication["public_campaign"] == PUBLIC and publication["original_artifacts"] == frozen_inputs(),
            "wrong publication identity or original bindings")
    require(publication["public_artifacts"] == inventory(raw), "public artifacts changed")
    require(digest(raw["manifest.json"]) == frozen_inputs()["manifest.json"]["sha256"], "execution manifest changed")
    campaign, rows = load_json(raw["manifest.json"]), load_json(raw["verdicts.json"])
    require(publication["campaign_hash"] == code_pilot.hash_json(campaign) and publication["rules"] == RULES
            and publication["generators"] == generators() and publication["figure_epoch"] == FIGURE_EPOCH
            and publication["source_snapshot_records_replaced"] == 3, "publication rules or generators changed")
    redactions = publication["redactions"]
    require(set(redactions) == {"occurrences", "fields"} and type(redactions["occurrences"]) is int
            and all(type(count) is int and count > 0 for count in redactions["fields"].values())
            and redactions["occurrences"] == sum(redactions["fields"].values()), "invalid redaction counts")
    records = {}
    for row in rows:
        name = "single__" + row["task"] + ".jsonl"
        require(digest(raw[name]) == row["transcript_sha256"], "public transcript hash disagreement")
        records[row["task"]] = [load_json(line) for line in raw[name].splitlines()]
    validate(campaign, rows, records, repo_tasks.provenance(), public=True)
    require(not any(HOME.search(value.decode("utf-8")) for name, value in raw.items()
                    if name.endswith((".json", ".jsonl", ".md", ".svg"))), "unmasked home path")
    render(directory, rows, campaign)
    require(inventory(read_files(directory, ARTIFACTS)) == inventory(raw), "public report did not reproduce")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Dérivation publique SymPy strictement hors inférence")
    parser.add_argument("--source", type=Path, default=ROOT / "results" / PRIVATE)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / PUBLIC)
    parser.add_argument("--report", action="store_true", help="Reproduire un dossier public copié, sans bruts privés")
    args = parser.parse_args(argv)
    if args.report:
        reproduce(args.output)
    else:
        derive(args.source, args.output)
    print("Dérivation et empreintes vérifiées ; aucune inférence.")
    return 0
