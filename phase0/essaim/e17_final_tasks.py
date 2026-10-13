"""VM-only pinned reserved loaders and public unittest adapters; never use a canonical solution."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
import urllib.request
from pathlib import Path

from essaim import aider_python, aider_unittest, bigcode_tasks
from essaim.ephemeral_vm import require_colab
from essaim.terminal_bench import task_path


def fetch_bytes(url, sha256, maximum, deadline):
    require_colab()
    data = bytearray()
    with urllib.request.urlopen(url, timeout=min(30, max(.001, deadline - time.monotonic()))) as response:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("pinned dataset download deadline")
            block = response.read(1 << 20)
            if not block:
                break
            data.extend(block)
            if len(data) > maximum:
                raise ValueError("dataset exceeds its pinned bound")
    if time.monotonic() >= deadline or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("pinned dataset hash/deadline mismatch")
    return bytes(data)


def stage_aider(archive, manifest, selected, directory, license_data):
    """Synthetic-testable archive parser; only declared stubs, assertions, configs and instructions are opened."""
    if (hashlib.sha256(archive).hexdigest() != manifest["archive_sha256"] or len(archive) > 32 << 20
            or len(set(selected)) != len(selected) or any(t not in manifest["tasks"] for t in selected)):
        raise ValueError("Aider archive or reserved identity mismatch")
    prefixes = {t: "python/exercises/practice/" + t + "/" for t in selected}
    directory.mkdir(parents=True, exist_ok=False)
    copied = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        members = {}
        for member in source:
            relative = "/".join(member.name.split("/")[1:])
            if relative == "README.md" or any(relative.startswith(p) for p in prefixes.values()):
                if relative in members:
                    raise ValueError("duplicate archive member")
                members[relative] = member

        def copy(relative):
            member = members[relative]
            info = manifest["files"][relative]
            if not member.isfile() or member.size != info["size"] or member.size > 1 << 20:
                raise ValueError("unsafe declared archive file")
            raw = source.extractfile(member).read((1 << 20) + 1)
            if len(raw) != info["size"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
                raise ValueError("Aider member provenance mismatch")
            path = task_path(directory, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as out:
                out.write(raw)
            copied[relative] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            return raw

        copy("README.md")
        for task, prefix in prefixes.items():
            cfg = bigcode_tasks.strict_json(copy(prefix + ".meta/config.json").decode("utf-8"))["files"]
            solutions, tests = cfg["solution"], cfg["test"]
            declared = [*solutions, *tests]
            if (not solutions or not tests or len(set(declared)) != len(declared)
                    or any(not n.endswith(".py") or n in aider_unittest.RESERVED for n in declared)):
                raise ValueError("unsupported generic Aider declaration")
            for name in declared:
                task_path(Path("."), name)
                copy(prefix + name)
            docs = sorted(n for n in members if n.startswith(prefix + ".docs/instructions") and n.endswith(".md"))
            if not docs:
                raise ValueError("reserved instructions missing; no substitution")
            for name in docs:
                copy(name)
            aider_unittest.inventory({n: (directory / prefix / n).read_text(encoding="utf-8") for n in tests})
    info = manifest["files"]["LICENSE.python"]
    if len(license_data) != info["size"] or hashlib.sha256(license_data).hexdigest() != info["sha256"]:
        raise ValueError("Aider licence provenance mismatch")
    with (directory / "LICENSE.python").open("xb") as out:
        out.write(license_data)
    copied["LICENSE.python"] = {"bytes": len(license_data), "sha256": hashlib.sha256(license_data).hexdigest()}
    return copied


RECORD_FIELDS = ("task_id", "instruct_prompt", "code_prompt", "canonical_solution", "test", "entry_point")


def record_digest(row, encoding):
    if encoding != "json-sort-ascii-default-v1":
        raise ValueError("confirmed historical six-field record encoding required")
    projected = {key: row[key] for key in RECORD_FIELDS}
    return hashlib.sha256(json.dumps(projected, sort_keys=True).encode()).hexdigest()


def stage_bigcode(rows, selected, pins, encoding, directory):
    """Use historical six-field hashes only for provenance; canonical_solution is never staged or passed to an adapter."""
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate reserved identity")
    found = {}
    for row in rows:
        identity = row.get("task_id")
        if identity in selected:
            if identity in found or record_digest(row, encoding) != pins[identity]["record_sha256"]:
                raise ValueError("reserved record hash/identity mismatch")
            # Original test text is copied verbatim. No oracle, solution example or hidden extra field is staged.
            kept = {key: row[key] for key in ("task_id", "instruct_prompt", "code_prompt", "test", "entry_point")}
            if any(type(value) is not str for value in kept.values()):
                raise ValueError("unsupported reserved record schema")
            expected = [n.rsplit(".", 1)[-1] for n in aider_unittest.inventory({"_official.py": kept["test"]})]
            if len(set(expected)) != len(expected):
                raise ValueError("ambiguous public unittest collection")
            kept["expected"] = expected
            found[identity] = kept
    if set(found) != set(selected):
        raise ValueError("reserved record missing; no replacement")
    directory.mkdir(parents=True, exist_ok=False)
    raw = json.dumps(found, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    with (directory / "records.json").open("xb") as out:
        out.write(raw)
    return {"records.json": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}


def prepare_reserved(plan, source, destination, deadline, *, fetch=fetch_bytes, parquet_rows=None):
    """The VM gate precedes all scientific reads. No oracle smoke or adaptive dependency installation."""
    require_colab()
    from e17_final_freeze import validate_plan
    validate_plan(plan)
    destination.mkdir(parents=True, exist_ok=False)
    raw = (source / "code_pilot/tasks.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan["selection"]["aider"]["manifest_sha256"]:
        raise ValueError("reserved Aider manifest differs from existing pins")
    aider = bigcode_tasks.strict_json(raw.decode())
    raw = (source / "bigcode_pilot/tasks.json").read_bytes()
    bigcode = bigcode_tasks.strict_json(raw.decode())
    selection = plan["selection"]["bigcode"]
    if (bigcode["origin"]["manifest_sha256"] != selection["manifest_sha256"]
            or bigcode["origin"]["commit"] != selection["private_branch_commit"]
            or bigcode["dataset"] != selection["dataset"]):
        raise ValueError("historical BigCode reservation provenance mismatch")
    archive = fetch(aider["archive_url"], aider["archive_sha256"], 32 << 20, deadline)
    licence = aider["files"]["LICENSE.python"]
    notice = fetch(aider["license_url"], licence["sha256"], 65536, deadline)
    aider_files = stage_aider(archive, aider, plan["selection"]["aider"]["test"], destination / "aider", notice)
    parquet = bigcode["sources"]["hard_parquet"]
    data = fetch(parquet["url"], parquet["sha256"], parquet["size"], deadline)
    if len(data) != parquet["size"]:
        raise ValueError("pinned parquet size mismatch")
    if parquet_rows is None:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        def parquet_rows(raw):
            table = pq.read_table(io.BytesIO(raw))
            chosen = pc.is_in(table["task_id"], value_set=pa.array(selection["test"]))
            return table.filter(chosen).to_pylist()
    bigcode_files = stage_bigcode(parquet_rows(data), selection["test"], selection["test_metadata"],
                                 plan["parent_decision"]["bigcode_record_encoding"], destination / "bigcode")
    return {"freeze_sha256": plan["freeze_sha256"], "selection_sha256": plan["selection_sha256"],
            "aider": aider_files, "bigcode": bigcode_files, "oracle_used": False}


class AiderAdapter:
    collect_sources = staticmethod(aider_python.collect_sources)

    def __init__(self, reserved, cache, pinned):
        self.reserved, self.cache, self.pinned = reserved, cache, pinned

    def check(self):
        for name, info in self.pinned.items():
            path = task_path(self.cache, name)
            if not path.is_file() or path.stat().st_size != info["bytes"] or path.stat().st_nlink != 1:
                raise ValueError("frozen Aider cache file changed")
            raw = path.read_bytes()
            if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
                raise ValueError("frozen Aider cache/assertion provenance changed")

    def copy_task(self, cache, task, work):
        self.check()
        return aider_python.copy_task(cache, self.reserved, work)

    def instructions(self, cache, task):
        self.check()
        from essaim.code_pilot import instructions
        return instructions(cache, self.reserved)

    def verify(self, cache, task, sources, runtime, timeout_s, *, environment):
        self.check()
        return aider_unittest.verify(cache, self.reserved, sources, runtime, timeout_s, environment=environment)


class BigCodeAdapter:
    collect_sources = staticmethod(aider_python.collect_sources)

    def __init__(self, cache, reserved, pinned):
        self.cache, self.pinned = cache, pinned
        self.check()
        self.row = bigcode_tasks.strict_json((cache / "records.json").read_text(encoding="utf-8"))[reserved]

    def check(self):
        path = task_path(self.cache, "records.json")
        if not path.is_file() or path.stat().st_size != self.pinned["bytes"] or path.stat().st_nlink != 1:
            raise ValueError("frozen BigCode cache file changed")
        raw = path.read_bytes()
        if len(raw) != self.pinned["bytes"] or hashlib.sha256(raw).hexdigest() != self.pinned["sha256"]:
            raise ValueError("frozen BigCode public assertion cache changed")

    def copy_task(self, cache, task, work):
        self.check()
        (work / "solution.py").write_text(self.row["code_prompt"] + "    raise NotImplementedError\n", encoding="utf-8")
        return ["solution.py"]

    def instructions(self, cache, task):
        self.check()
        return (self.row["instruct_prompt"] + "\n\nWrite the complete self-contained implementation to solution.py. "
                "Only solution.py is submitted. Official tests and reference are hidden. "
                "The controller evaluates a fresh source snapshot with the original official unittest assertions. "
                "Use MYRIAD_RUN_OFFICIAL_TESTS to test before submission.")

    def verify(self, cache, task, sources, runtime, timeout_s, *, environment):
        self.check()
        return bigcode_tasks.evaluate(sources, self.row["test"], self.row["entry_point"], self.row["expected"],
                                      runtime, timeout_s, environment=environment)
