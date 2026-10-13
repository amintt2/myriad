"""Real frozen orchestration on synthetic records/archives/CLI/model responses only, never scientific TEST."""
from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e17_final_freeze as final
import e17_final_vm as vm
from essaim import aider_agent, aider_python, aider_unittest, bigcode_tasks, code_pilot, ephemeral_vm
from essaim.e17_final_tasks import stage_aider, stage_bigcode, record_digest, prepare_reserved
from essaim.agent import SUBMIT
import test_e17_final_freeze as foundation
from test_code_pilot import answer

COLAB = Path(__file__).resolve().parents[1] / "colab"
sys.path.insert(0, str(COLAB))
spec = importlib.util.spec_from_file_location("final_owned_supervision", COLAB / "e17_final_supervise.py")
supervision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervision)

READY = {"isolation_ready": "ephemeral-vm-v1", "landlock_abi": None, "seccomp": "on",
         "namespaces": ["user", "mount", "pid", "net", "ipc", "uts"]}


def fixture_plan(root):
    foundation.Preparation().fixture(root)
    pin = {"python": "3.12.15", "binary": {"size": 1, "sha256": "c" * 64},
           "lock": {"size": 1, "sha256": "d" * 64}, "packages": {"fixture": "1"}, "licenses": {},
           "bundled_components": [],
           "python_licence": {"size": 1, "sha256": "e" * 64, "url": "https://synthetic.invalid/licence"}}
    (root / "phase0/bigcode_pilot/runtime.json").write_bytes(final.canonical(pin))
    for name in [*final.FINAL_METADATA, *final.REQUIRED_OPERATIONAL_FILES]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if name in final.FINAL_METADATA else b"# synthetic operational source\n")
    files = final.final_sources(root)
    decision = {"schema": 1, "actor": "parent", "approved": True, "code_commit": "a" * 40,
                "adapter_fix_commit": "b" * 40, "scientific_revision": "R00", "selected": "P01",
                "proposals": [{"id": "P01", "decision": "accepted", "sha256": final.digest(files[final.PROPOSAL])}],
                "sources": final.source_hashes(files), "audit_sha256": "e" * 64,
                "dev_smoke_sha256": "f" * 64, "synthetic_smoke_sha256": "a" * 64,
                "system_sha256": final.digest(aider_agent.SYSTEM.encode()),
                "bigcode_record_encoding": "json-sort-ascii-default-v1"}
    policy = final.budget(1.15, 18.85, 5.3, 10000, 3000, 1800)
    return final.frozen_plan(files, policy, decision), files


def grade(passed=True):
    return {"status": "graded", "passed": passed, "tests": 1, "tests_expected": 1, "tests_complete": True,
            "failures": int(not passed), "errors": 0, "broken": [], "skipped": 0, "expected_failures": 0,
            "unexpected_successes": 0, "calls": [{"test": "test_fixed", "complete": True}], "feedback": [],
            "sandbox_backend": "ephemeral-vm", "sandbox": "ephemeral-vm-v1", "isolation": READY,
            "python_audit_hook": True, "seconds": .001,
            "trust_scope": "public-benchmark-nonadversarial", "adversarial_attestation": False}


def fixture_manifest(plan, request):
    image = {"markers": {"hostname_file": True, "drive_binary": True}, "kernel": "synthetic-linux",
             "system": "Linux", "machine": "x86_64"}
    pin = plan["runtime_pin"]
    runtime = {"sources": request["sources"], "sandbox_backend": "ephemeral-vm", "colab_image": image,
               "python": pin["python"], "executable_sha256": pin["binary"]["sha256"],
               "executable_bytes": pin["binary"]["size"], "bundled_components": [],
               "lock_sha256": pin["lock"]["sha256"], "packages": pin["packages"], "recipe_verified": True,
               "license_hashes": {}, "python_license": {k: pin["python_licence"][k] for k in ("size", "sha256")},
               "cuda_devices": "CUDA0: synthetic", "library_path": "/synthetic:/usr/lib64-nvidia"}
    runtime.update(server_version="b11505 ff5888f99 synthetic", gpu_observed=["A100 synthetic, 40000"],
                   server_files={"bin/llama-server": "f" * 64})
    from e17_vm import execution_provenance
    provenance = execution_provenance(plan, {"cpu": "synthetic", "host_memory_gib": 32, "execution": "synthetic",
                                           "gpu": "A100", "cpu_threads": "1"})
    dataset = {"freeze_sha256": plan["freeze_sha256"], "selection_sha256": plan["selection_sha256"], "oracle_used": False}
    return {"request": request, "freeze_sha256": plan["freeze_sha256"], "system": aider_agent.SYSTEM,
            "runtime": runtime, "execution_provenance": provenance, "dataset": dataset, "isolation": READY,
            "colab_image": image, "limits": final.LIMITS, "models": final.MODELS, "cells": plan["cells"]}


class FixtureEnv:
    version = "ephemeral-vm-v1"
    def __init__(self, work, runtime):
        self.work, self.isolation = work, None
    def run(self, command, timeout):
        self.isolation = READY
        return 0, "synthetic command"
    def close(self):
        pass


class FixtureAdapter:
    collect_sources = staticmethod(aider_python.collect_sources)
    def copy_task(self, cache, task, work):
        (work / "solution.py").write_text("def task_func(): return 2\n", encoding="utf-8")
        return ["solution.py"]
    def instructions(self, cache, task):
        return "Synthetic instruction fixed before any scientific TEST."
    def verify(self, cache, task, sources, runtime, timeout_s, *, environment):
        return grade()


class FixtureClient:
    loading_seconds = request_seconds = 0
    def __init__(self):
        self.calls, self.closed = [], False
    def post(self, path, json, timeout):
        self.calls.append(json)
        # Existing official feedback chooses the next reply; no reference or actual model is executed.
        green = any("GREEN: current sources" in str(m.get("content")) for m in json["messages"])
        command = f"echo {SUBMIT}" if green else aider_agent.TEST
        value = answer(f"```bash\n{command}\n```", model=json["model"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: value, headers={"server": "synthetic"})
    def close(self):
        self.closed = True


class FreezeTests(unittest.TestCase):
    def test_historical_source_bundle_still_refuses_r01_without_exemption(self):
        import e17_campaign
        original = Path.read_bytes
        path = e17_campaign.ROOT / "essaim/aider_agent.py"
        def read(source):
            raw = original(source)
            return raw + b"\nSYSTEM += 'synthetic R01 change'\n" if source == path else raw
        with mock.patch.object(Path, "read_bytes", read), self.assertRaisesRegex(ValueError, "R00 scientific harness"):
            e17_campaign.source_bundle()

    def test_real_freeze_validate_mutation_and_no_replay_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            plan, files = fixture_plan(root)
            prep, sealed = Path(temporary) / "prep", Path(temporary) / "frozen"
            final.prepare(prep, plan["budget"], root)
            final.freeze(prep, sealed, plan["parent_decision"], root)
            self.assertEqual(final.require_final_freeze(sealed, root), (plan, files))
            with self.assertRaises(FileExistsError):
                final.freeze(prep, sealed, plan["parent_decision"], root)
            for key, change in (("models", {}), ("cells", plan["cells"][:-1]), ("mode_order", ["vote", "single"]),
                                ("limits", {"max_tokens": 1}), ("limits", {**final.LIMITS, "seed": False}),
                                ("temperature", False), ("server", {}), ("partition", "DEV")):
                altered = copy.deepcopy(plan)
                altered[key] = change
                altered["freeze_sha256"] = final.digest(final.canonical({k: v for k, v in altered.items()
                                                                        if k != "freeze_sha256"}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    final.validate_plan(altered)
            (root / "phase0/essaim/aider_agent.py").write_text("# changed after freeze\n")
            with self.assertRaises(ValueError):
                final.require_final_freeze(sealed, root)

    def test_parent_decisions_and_p2_fix_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, files = fixture_plan(Path(temporary))
            for key, changed in (("approved", False), ("actor", "proposer"), ("selected", "P03"),
                                 ("adapter_fix_commit", "7fbe584fb8cebc1fa0c7612f5097f73aa9914258"),
                                 ("bigcode_record_encoding", None), ("audit_sha256", "")):
                decision = copy.deepcopy(plan["parent_decision"])
                decision[key] = changed
                with self.subTest(key=key), self.assertRaises(ValueError):
                    final.frozen_plan(files, plan["budget"], decision)
            decision = copy.deepcopy(plan["parent_decision"])
            decision["proposals"].append({"id": "P04", "decision": "accepted", "sha256": "b" * 64})
            with self.assertRaises(ValueError):
                final.frozen_plan(files, plan["budget"], decision)

    def test_vm_gate_refuses_before_metadata_or_content_read(self):
        with mock.patch.object(ephemeral_vm, "colab_markers", return_value={"hostname_file": False, "drive_binary": False}), \
             mock.patch.object(Path, "read_bytes", side_effect=AssertionError("no TEST before VM/freeze")):
            with self.assertRaises(Exception) as error:
                prepare_reserved({}, Path("absent"), Path("absent"), time.monotonic() + 1)
            self.assertNotIsInstance(error.exception, AssertionError)
        marker = Path(tempfile.gettempdir()) / ("synthetic-consumed-" + str(time.time_ns()))
        try:
            final.consume(marker, {"started": True})
            with self.assertRaises(FileExistsError):
                final.consume(marker, {"different_output": True})
        finally:
            marker.unlink(missing_ok=True)


class LoaderTests(unittest.TestCase):
    def test_cached_original_assertions_cannot_change_between_controller_tests(self):
        from essaim.e17_final_tasks import AiderAdapter, BigCodeAdapter
        data, manifest, notice = self.archive()
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "aider"
            pins = stage_aider(data, manifest, ["synthetic-public"], cache, notice)
            adapter = AiderAdapter("synthetic-public", cache, pins)
            adapter.check()
            test = cache / "python/exercises/practice/synthetic-public/public_test.py"
            test.write_bytes(b"print('forged green')")
            with self.assertRaises(ValueError):
                adapter.check()
            records = Path(temporary) / "bigcode"
            records.mkdir()
            raw = final.canonical({"Synthetic/700": {"test": "original"}})
            (records / "records.json").write_bytes(raw)
            adapter = BigCodeAdapter(records, "Synthetic/700", {"bytes": len(raw), "sha256": final.digest(raw)})
            (records / "records.json").write_bytes(raw.replace(b"original", b"tampered"))
            with self.assertRaises(ValueError):
                adapter.check()

    def archive(self, malformed=None):
        prefix = "python/exercises/practice/synthetic-public/"
        files = {"README.md": b"Synthetic attribution", prefix + ".meta/config.json": final.canonical(
                 {"files": {"solution": ["solution.py"], "test": ["public_test.py"], "example": ["oracle.py"]}}),
                 prefix + "solution.py": b"def task_func(): raise NotImplementedError\n",
                 prefix + "public_test.py": b"import unittest\nclass Public(unittest.TestCase):\n def test_fixed(self): pass\n",
                 prefix + ".docs/instructions.md": b"Synthetic public exercise", prefix + "oracle.py": b"MUST NOT BE OPENED"}
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w:gz") as tar:
            for name, content in files.items():
                entry = tarfile.TarInfo("archive/" + name)
                entry.size = len(content)
                if name == malformed:
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = "../foreign"
                tar.addfile(entry, io.BytesIO(content))
        data = raw.getvalue()
        licence = b"Synthetic licence"
        pins = {n: {"size": len(v), "sha256": final.digest(v)} for n, v in files.items()}
        pins["LICENSE.python"] = {"size": len(licence), "sha256": final.digest(licence)}
        return data, {"archive_sha256": final.digest(data), "tasks": ["synthetic-public"], "files": pins}, licence

    def test_generic_archive_only_declared_stubs_assertions_and_instructions(self):
        data, manifest, notice = self.archive()
        original = tarfile.TarFile.extractfile
        def extract(source, member):
            self.assertNotIn("oracle.py", member.name)
            return original(source, member)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(tarfile.TarFile, "extractfile", extract):
            directory = Path(temporary) / "selected"
            copied = stage_aider(data, manifest, ["synthetic-public"], directory, notice)
            self.assertFalse(any("oracle.py" in n for n in copied))
            with self.assertRaises(FileExistsError):
                stage_aider(data, manifest, ["synthetic-public"], directory, notice)
        data, manifest, notice = self.archive("python/exercises/practice/synthetic-public/solution.py")
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            stage_aider(data, manifest, ["synthetic-public"], Path(temporary) / "selected", notice)

    def test_bigcode_six_field_provenance_without_oracle_staging_or_replacement(self):
        row = {"task_id": "Synthetic/700", "entry_point": "task_func", "code_prompt": "def task_func():\n",
               "instruct_prompt": "Synthetic public task \u00e9", "canonical_solution": "DO NOT USE",
               "extra_column": "ignored \u03bb",
               "test": "import unittest\nclass Public(unittest.TestCase):\n def test_fixed(self): pass\n"}
        fields = ("task_id", "instruct_prompt", "code_prompt", "canonical_solution", "test", "entry_point")
        expected = hashlib.sha256(json.dumps({k: row[k] for k in fields}, sort_keys=True).encode()).hexdigest()
        self.assertEqual(record_digest(row, "json-sort-ascii-default-v1"), expected)
        self.assertEqual(record_digest({**row, "extra_column": "different"}, "json-sort-ascii-default-v1"), expected)
        for key in fields:
            self.assertNotEqual(record_digest({**row, key: row[key] + "changed"},
                                               "json-sort-ascii-default-v1"), expected)
        pins = {row["task_id"]: {"record_sha256": record_digest(row, "json-sort-ascii-default-v1")}}
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "data"
            stage_bigcode([row], [row["task_id"]], pins, "json-sort-ascii-default-v1", stage)
            raw = (stage / "records.json").read_bytes()
            self.assertNotIn(b"extra_column", raw)
            self.assertNotIn(b"canonical_solution", raw)
            self.assertNotIn(b"DO NOT USE", raw)
            for records in ([], [row, row], *([{**row, key: row[key] + "changed"}] for key in fields)):
                with self.assertRaises(ValueError):
                    stage_bigcode(records, [row["task_id"]], pins, "json-sort-ascii-default-v1", Path(temporary) / "other")


class RunnerTests(unittest.TestCase):
    def run_fixture(self, root, output, fail=None):
        plan, files = fixture_plan(root)
        request = {"plan": plan, "sources": {final.vm_name(n): final.digest(v) for n, v in files.items()},
                   "campaign_id": "a" * 32, "nonce": "synthetic", "phase0": {"synthetic": True}}
        manifest = fixture_manifest(plan, request)
        output.mkdir()
        client = FixtureClient()
        args = SimpleNamespace(**final.LIMITS, sandbox_backend="ephemeral-vm", runtime=root / "runtime",
                               tasks_dir=root, reference=final.REFERENCE)
        def adapters(cell):
            if cell["index"] == fail:
                raise RuntimeError("synthetic unsupported runtime; no replacement")
            return FixtureAdapter(), root
        with mock.patch.object(code_pilot, "select_backend", return_value=FixtureEnv):
            vm.run_cells(plan, args, client, output, time.monotonic() + 10000, manifest, adapters)
        self.assertTrue(client.closed)
        return plan, request, manifest, client

    def test_real_run_one_sixteen_episodes_exact_order_and_full_retrieval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "source", Path(temporary) / "artifacts"
            plan, request, manifest, client = self.run_fixture(root, output)
            registry = final.read_json((output / "registry.json").read_bytes())
            self.assertEqual([e["state"] for e in registry["attempts"]], ["graded"] * 16)
            self.assertEqual([e["mode"] for e in registry["attempts"]], [c["mode"] for c in plan["cells"]])
            rows = final.read_json((output / "verdicts.json").read_bytes())
            self.assertTrue(all(r["passed"] for r in rows))
            for name, value in {"manifest.json": manifest, "runtime.json": manifest["runtime"],
                "dataset-manifest.json": manifest["dataset"], "provenance.json": manifest["execution_provenance"],
                "preparation-started.json": {"freeze_sha256": plan["freeze_sha256"], "request": request},
                "smoke.json": {"kind": "prefixed-synthetic-public-unittest", "freeze_sha256": plan["freeze_sha256"],
                               "oracle_test_used": False, "positive": grade(), "negative": grade(False),
                               "aider_positive": grade(), "aider_negative": grade(False)},
                "vm-status.json": {"request": request, "state": "complete", "error": None}}.items():
                vm.save(output / name, value)
            (output / "controller.log").write_bytes(b"synthetic closed controller")
            (output / "server-events.jsonl").write_bytes(b"{\"synthetic\":true}\n")
            inventory = {"schema": 1, "request": request, "complete": True,
                         "files": {n: {"bytes": (output / n).stat().st_size, "sha256": final.digest((output / n).read_bytes())}
                                   for n in final.artifacts(plan)}}
            self.assertTrue(final.validate_retrieved(output, inventory, request))
            vm.save(output / "retrieval.json", inventory)
            report = {"verdict": "test-complete", "closed": True, "absence_confirmed": True,
                      "retrieved_complete": True, "error": None, "recovery_error": None,
                      "plan": plan, "inventory": inventory, "usage": []}
            from analyze_e17_final import analyze
            summary = analyze(output, report, Path(temporary) / "analysis")
            self.assertEqual([summary["modes"][m]["episodes"] for m in plan["mode_order"]], [4] * 4)
            self.assertFalse(summary["partial_scores_allowed"])
            with self.assertRaises(ValueError):
                analyze(output, {**report, "closed": False}, Path(temporary) / "no-partial-analysis")
            self.assertFalse((Path(temporary) / "no-partial-analysis").exists())
            with self.assertRaises(FileExistsError):
                vm.run_cells(plan, SimpleNamespace(), FixtureClient(), output, time.monotonic() + 10000, manifest,
                             lambda c: (FixtureAdapter(), root))
            partial = copy.deepcopy(inventory)
            partial["complete"] = False
            with self.assertRaisesRegex(ValueError, "without score or replay"):
                final.validate_retrieved(output, partial, request)
            (output / "controller.log").write_bytes(b"false green stdout")
            with self.assertRaises(ValueError):
                final.validate_retrieved(output, inventory, request)

    def test_interruption_consumes_started_cell_and_never_substitutes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "source", Path(temporary) / "artifacts"
            with self.assertRaises(RuntimeError):
                self.run_fixture(root, output, fail=2)
            registry = final.read_json((output / "registry.json").read_bytes())
            self.assertEqual([e["state"] for e in registry["attempts"][:4]], ["graded", "graded", "error", "unstarted"])
            with self.assertRaises(FileExistsError):
                final.consume(output / "registry.json", {})

    def test_partial_and_false_stdout_never_green_and_backend_error_distinct(self):
        for change in ({"tests_complete": False}, {"tests": 0}, {"python_audit_hook": False},
                       {"isolation": {"forged": True}}, {"status": "error", "passed": None},
                       {"calls": []}, {"sandbox_backend": "strict"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                vm.validate_grade({**grade(), **change})
        vm.validate_grade(grade(False))
        with self.assertRaisesRegex(ValueError, "unsupported public test completion"):
            vm.validate_grade({**grade(False), "skipped": 1})


def synthetic_final(request, mode="success"):
    """A fake remote runs the real scientific loop over fixed synthetic adapters, never over reserved contents."""
    cached = Path(os.environ["TEST_ROOT"]) / "sealed-final-fixture" if "TEST_ROOT" in os.environ else None
    if cached is not None and cached.exists():
        saved = final.read_json((cached / "inventory.json").read_bytes())
        if saved["request"] != request:
            raise ValueError("foreign synthetic fixture request")
        return {n: (cached / n).read_bytes() for n in saved["files"]}, saved
    plan = request["plan"]
    manifest = fixture_manifest(plan, request)
    with tempfile.TemporaryDirectory() as temporary:
        root, output = Path(temporary), Path(temporary) / "artifacts"
        output.mkdir()
        args = SimpleNamespace(**final.LIMITS, sandbox_backend="ephemeral-vm", runtime=root,
                               tasks_dir=root, reference=final.REFERENCE)
        with mock.patch.object(code_pilot, "select_backend", return_value=FixtureEnv):
            vm.run_cells(plan, args, FixtureClient(), output, time.monotonic() + 10000, manifest,
                         lambda c: (FixtureAdapter(), root))
        for name, value in {"manifest.json": manifest, "runtime.json": manifest["runtime"],
            "dataset-manifest.json": manifest["dataset"], "provenance.json": manifest["execution_provenance"],
            "preparation-started.json": {"freeze_sha256": plan["freeze_sha256"], "request": request},
            "smoke.json": {"kind": "prefixed-synthetic-public-unittest", "freeze_sha256": plan["freeze_sha256"],
                           "oracle_test_used": False, "positive": grade(), "negative": grade(False),
                           "aider_positive": grade(), "aider_negative": grade(False)},
            "vm-status.json": {"request": request, "state": "complete", "error": None}}.items():
            vm.save(output / name, value)
        (output / "controller.log").write_bytes(b"closed synthetic controller")
        (output / "server-events.jsonl").write_bytes(b"{\"synthetic\":true}\n")
        files = {n: (output / n).read_bytes() for n in final.artifacts(plan)}
    if mode == "partial":
        files["vm-status.json"] = final.canonical({"request": request, "state": "incomplete", "error": "synthetic error"})
    inventory = {"schema": 1, "request": request, "complete": mode != "partial",
                 "files": {n: {"bytes": len(raw), "sha256": final.digest(raw)} for n, raw in files.items()}}
    if mode == "bad-hash":
        inventory["files"]["controller.log"]["sha256"] = "0" * 64
    if cached is not None:
        cached.mkdir(exist_ok=False)
        for name, raw in files.items():
            (cached / name).write_bytes(raw)
        vm.save(cached / "inventory.json", inventory)
    return files, inventory


class SupervisionTests(unittest.TestCase):
    def test_prelaunch_rechecks_sources_after_account_wait_before_allocation_or_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, files = fixture_plan(root / "source")
            args = supervision.controller_args(plan, root / "output", root / "receipt")
            campaign = supervision.FinalCampaign(args, plan, final.vm_bundle(files))
            campaign.source_directory = root / "source"
            with mock.patch.object(final, "git_revision", return_value=plan["parent_decision"]["code_commit"]):
                campaign.check_prelaunch()
                (root / "source/phase0/essaim/aider_agent.py").write_bytes(b"altered after account wait")
                with self.assertRaisesRegex(ValueError, "no allocation/start"):
                    campaign.check_prelaunch()
    def test_permanent_consumption_survives_interrupt_and_different_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, files = fixture_plan(root / "source")
            marker = root / "permanent-consumed"
            with mock.patch.object(final, "require_final_freeze", return_value=(plan, files)), \
                 mock.patch.object(final, "git_revision", return_value=plan["parent_decision"]["code_commit"]), \
                 mock.patch.object(supervision.sys, "platform", "linux"), \
                 mock.patch.object(supervision, "consumption_path", return_value=marker), \
                 mock.patch.object(supervision.signal, "signal"), \
                 mock.patch.object(supervision.lifecycle, "lifecycle_lock", side_effect=lambda p: contextlib.nullcontext()), \
                 mock.patch.object(supervision.FinalCampaign, "execute", side_effect=SystemExit(143)) as execute:
                arguments = ["--freeze", str(root / "freeze"), "--output", str(root / "report.json"),
                             "--execute", "--parent-validated-freeze", plan["freeze_sha256"],
                             "--receipt", str(root / "receipt")]
                with self.assertRaises(SystemExit):
                    supervision.main(arguments)
                self.assertEqual(final.read_json(marker.read_bytes())["state"], "started-no-replay")
                arguments[3] = str(root / "different-output.json")
                with self.assertRaises(FileExistsError):
                    supervision.main(arguments)
                self.assertEqual(execute.call_count, 1)
                marker.write_bytes(b"interrupted marker")
                with self.assertRaises(FileExistsError):
                    supervision.main(arguments)
                self.assertEqual(execute.call_count, 1)

    def test_global_consumption_path_ignores_receipt_override(self):
        expected = supervision.consumption_path()
        with mock.patch.dict(os.environ, {"DLLM_CAMPAIGN_RECEIPT": "synthetic-other-receipt"}):
            self.assertEqual(supervision.consumption_path(), expected)

    def test_parent_confirmation_missing_refuses_before_campaign_or_consumption(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, files = fixture_plan(Path(temporary) / "source")
            with mock.patch.object(final, "require_final_freeze", return_value=(plan, files)), \
                 mock.patch.object(final, "git_revision", return_value=plan["parent_decision"]["code_commit"]), \
                 mock.patch.object(supervision.sys, "platform", "linux"), \
                 mock.patch.object(supervision.FinalCampaign, "execute", side_effect=AssertionError("no allocation")), \
                 mock.patch.object(final, "consume", side_effect=AssertionError("no consumption before parent validation")):
                with self.assertRaisesRegex(ValueError, "parent validation"):
                    supervision.main(["--freeze", temporary, "--output", str(Path(temporary) / "report"), "--execute"])

    def test_final_worker_refuses_non_colab_before_bootstrap_and_keeps_closed_inventory(self):
        import e17_final_remote
        import e17_campaign_remote
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, files = fixture_plan(root / "source")
            request = {"plan": plan, "sources": {final.vm_name(n): final.digest(v) for n, v in files.items()},
                       "work_seconds": 1000, "campaign_id": "a" * 32}
            (root / "artifacts").mkdir()
            vm.save(root / "request.json", request)
            before = e17_campaign_remote.RUNTIME_PROJECT
            with mock.patch.object(ephemeral_vm, "colab_markers", return_value={"hostname_file": False, "drive_binary": False}), \
                 mock.patch.object(e17_campaign_remote, "bootstrap_uv", side_effect=AssertionError("no local bootstrap")):
                e17_final_remote.worker(root)
            status = final.read_json((root / "artifacts/vm-status.json").read_bytes())
            self.assertEqual(status["phase"], "final-preflight")
            self.assertEqual(status["state"], "incomplete")
            self.assertFalse(final.read_json((root / "inventory.json").read_bytes())["complete"])
            self.assertEqual(e17_campaign_remote.RUNTIME_PROJECT, before)

    def test_explicit_ephemeral_adapter_stages_existing_audit_policy_without_strict_fallback(self):
        import test_aider_generic as fixtures
        owner = self
        class Environment(ephemeral_vm.EphemeralVMEnv):
            def __init__(self, work, runtime):
                self.work, self.isolation = work, READY
            def run(self, command, timeout):
                owner.assertIn("sys.addaudithook(_hook)", (self.work / "_myriad_policy.py").read_text())
                expected = aider_unittest.inventory({"public_test.py": fixtures.OFFICIAL})
                return 0, fixtures.artifact(expected)
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            fixtures.fixture(cache, {"operations.py": fixtures.FUNCTION}, {"public_test.py": fixtures.OFFICIAL})
            with mock.patch.object(aider_python, "provenance", return_value={"tasks": ["synthetic-operations"]}), \
                 mock.patch.object(ephemeral_vm, "require_colab", return_value={"hostname_file": True, "drive_binary": True}):
                result = aider_unittest.verify(cache, "synthetic-operations", {"operations.py": fixtures.FUNCTION},
                                               cache, 120, environment=Environment)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["sandbox_backend"], "ephemeral-vm")
            self.assertTrue(result["python_audit_hook"])
            vm.validate_grade(result)
        with mock.patch.object(ephemeral_vm, "require_colab", side_effect=RuntimeError("not a Colab VM")), \
             mock.patch.object(Path, "read_bytes", side_effect=AssertionError("no reserved content before VM gate")):
            result = aider_unittest.verify(Path("absent"), "synthetic", {}, Path("runtime"), 120, environment=Environment)
            self.assertEqual((result["status"], result["passed"]), ("error", None))


@unittest.skipUnless(sys.platform == "linux", "isolated lifecycle CLI processes require Linux/WSL")
class FinalProcesses(unittest.TestCase):
    def test_fixed_generic_smoke_uses_real_strict_backend_and_native_classes(self):
        from essaim.local_linux import LocalLinuxEnv
        with tempfile.TemporaryDirectory() as temporary:
            positive, negative = vm.generic_smoke(Path(temporary), Path(sys.prefix), LocalLinuxEnv)
            self.assertTrue(aider_agent.complete_green(positive), positive)
            self.assertEqual((negative["status"], negative["passed"]), ("graded", False))
            self.assertEqual(positive["sandbox_backend"], "strict")
            self.assertGreaterEqual(positive["isolation"]["landlock_abi"], 3)

    def test_owned_final_campaign_recovery_and_cleanup_use_existing_authority(self):
        import test_e17_campaign as existing
        for mode in ("success", "partial", "bad-hash", "lost-start-ack", "foreign-preflight"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan, files = fixture_plan(root / "sources")
                vm.save(root / "fixture-plan.json", plan)
                bins = root / ".local/bin"
                bins.mkdir(parents=True)
                fake = existing.FAKE.replace("Hardware: L4", "Hardware: A100")
                self.assertEqual(fake.count("from test_e17_campaign import synthetic"), 2)
                fake = fake.replace("from test_e17_campaign import synthetic", "from test_e17_final_runner import synthetic_final as synthetic")
                cli = bins / "colab"
                cli.write_text(f"#!{sys.executable}\n" + fake, encoding="utf-8")
                cli.chmod(0o755)
                runner = root / "runner.py"
                runner.write_text('''import json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import test_e17_final_runner as fixture
root=Path(sys.argv[2]); plan=json.loads((root/'fixture-plan.json').read_text())
args=SimpleNamespace(output=root/'report.json',receipt=root/'receipt',seconds=90,budget_units=4,spent_units=1.15,
 min_balance_units=16.15,cleanup_seconds=3,max_rate_units_hour=5.3,recover_seconds=25,read_seconds=2,
 usage_seconds=.2,grace=.2,retry_seconds=.01,poll_seconds=.01)
files=fixture.final.vm_bundle(fixture.final.final_sources(root/'sources'))
fixture.supervision.FinalCampaign.source_directory=root/'sources'
fixture.final.git_revision=lambda root=None: plan['parent_decision']['code_commit']
sys.exit(fixture.supervision.FinalCampaign(args,plan,files).execute())
''', encoding="utf-8")
                env = {**os.environ, "HOME": str(root), "PATH": str(bins) + ":" + os.environ["PATH"],
                       "TEST_ROOT": str(root), "MODE": mode, "TEST_MODULES": str(Path(__file__).parent),
                       "DLLM_CLI_SECONDS": "2", "DLLM_CLI_GRACE": ".2"}
                for key in ("DLLM_LIFECYCLE_LOCK_FD", "DLLM_OPERATION_IDENTITY", "DLLM_CAMPAIGN_ID",
                            "DLLM_CAMPAIGN_RECEIPT"):
                    env.pop(key, None)
                result = subprocess.run([sys.executable, str(runner), str(Path(__file__).parent), str(root)],
                                        env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0 if mode == "success" else 1, result.stdout + result.stderr)
                report = final.read_json((root / "report.json").read_bytes())
                calls = (root / "calls").read_text().splitlines()
                if mode == "foreign-preflight":
                    self.assertNotIn("stop", calls)
                    self.assertNotIn("new", calls)
                else:
                    self.assertEqual(calls.count("new"), 1)
                    self.assertEqual(calls.count("stop"), 1)
                    self.assertTrue(report["closed"], result.stdout + result.stderr)
                    self.assertTrue(report["absence_confirmed"])
                    self.assertEqual((root / "rpc-actions").read_text().splitlines().count("start"), 1)
                    if mode in ("success", "lost-start-ack"):
                        self.assertTrue(report["retrieved_complete"])
                    if mode == "partial":
                        self.assertFalse(report["retrieved_complete"])
                        self.assertTrue((root / "report-artifacts/recovery.json").exists())
                self.assertEqual(report["verdict"], "test-complete" if mode == "success" else "infrastructure-incomplete")


if __name__ == "__main__":
    unittest.main()
