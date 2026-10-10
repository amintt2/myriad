"""Offline immutable-input validation, constrained redaction and deterministic public bindings."""
import copy
import json
import os
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from essaim import code_pilot, repo_publication as pub, repo_tasks


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "private"
        self.source.mkdir()
        self.home = "/" + "home/fixture/runtime"
        self.windows = "C:" + "\\Users\\fixture\\runtime"
        source = "# public example\nvalue = 1\n"
        self.dataset = {"tasks": {}, "files": {}}
        for task in repo_tasks.PILOT:
            self.dataset["tasks"][task] = {"sources": ["module.py"], "repo": "sympy/sympy",
                                           "base_commit": "a" * 40, "cases": 7,
                                           "archive": {"url": "https://example.org/base.tar.gz",
                                                       "size": 20, "sha256": "b" * 64}}
            self.dataset["files"][task + "/base/module.py"] = {"size": len(source.encode()),
                                                               "sha256": pub.digest(source.encode())}
        self.campaign = {"dataset_hash": code_pilot.hash_json(self.dataset), "version": "repo-pilot-v1",
                         "tasks": repo_tasks.PILOT, "models": {"single": [pub.MODEL]}, "reference": None,
                         "limits": pub.LIMITS, "watts": 100, "eur_kwh": 0.25, "system": "system",
                         "instructions": {task: "instruction" for task in repo_tasks.PILOT}}
        self.logs, self.rows = {}, []
        for task in repo_tasks.PILOT:
            log = [{"kind": "start", "task": task, "mode": "single", "instruction": "instruction", "system": "system"}]
            for index in range(12):
                request = {"kind": "request", "model": pub.MODEL, "served_model": pub.MODEL, "ceiling": 1024,
                           "parameters": {"model": pub.MODEL, "max_tokens": 1024, "seed": 0, "temperature": 0.0,
                                          "myriad": {"k": 1, "early_stop": False, "format_instruction": False},
                                          "messages": [{"role": "user", "content": self.home}]},
                           "reply": self.windows, "message": {"content": self.windows}, "completion_tokens": 1,
                           "known_completion_tokens": 1, "usage": {"completion_tokens": 1}, "myriad": None,
                           "seconds": 1.0, "finish_reason": "stop"}
                log.extend([request, {"kind": "step", "reply": self.windows, "command": "cat " + self.home,
                                      "output": self.home, "exit_code": 1,
                                      "note": {"replies": {pub.MODEL: self.windows}}}])
            row = {"task": task, "campaign_hash": code_pilot.hash_json(self.campaign), "mode": "single",
                   "status": "graded", "passed": False, "source_hashes": {"module.py": pub.digest(source.encode())},
                   "requests": 12, "steps": 12, "stopped": "generation_limit", "unknown_usage_requests": 0,
                   "charged_request_ceilings": 12288, "completion_tokens": 12, "known_completion_tokens": 12,
                   "request_seconds": 12.0, "seconds": 13.0, "episode_seconds": 12.0,
                   "verifier": {"status": "graded", "passed": False, "tests_expected": 7, "tests": 1,
                                "fail_fast": True, "tests_complete": False, "broken": [], "seconds": 1.0,
                                "calls": [{"candidate_failure": "wrong_string", "diagnostics": self.home,
                                           "returned": {"value": "scientific string"}}]}}
            log += [{"kind": "isolation", "receipt": {"landlock_abi": 7}},
                    {"kind": "changes", "initial": {"module.py": source}, "sources": {"module.py": source}},
                    {"kind": "end", **row}]
            self.rows.append(row)
            self.logs[task] = log
        self.refresh()

    def refresh(self):
        for row in self.rows:
            raw = b"".join((json.dumps(record) + "\n").encode() for record in self.logs[row["task"]])
            (self.source / ("single__" + row["task"] + ".jsonl")).write_bytes(raw)
            row["transcript_sha256"] = pub.digest(raw)
        (self.source / "manifest.json").write_bytes(encoded(self.campaign))
        (self.source / "verdicts.json").write_bytes(encoded(self.rows))
        for name in pub.ARTIFACTS - {"manifest.json", "verdicts.json", *["single__" + t + ".jsonl" for t in repo_tasks.PILOT]}:
            (self.source / name).write_bytes(b"fixture\n")
        (self.source / "summary.json").write_bytes(b"summary\n")
        (self.source / "report.md").write_bytes(b"plafonds factur\xc3\xa9s\n")
        self.pins = pub.inventory(pub.read_files(self.source, pub.ARTIFACTS))

    def fake_render(self, output, rows, campaign):
        for name in pub.ARTIFACTS:
            if name.startswith("accuracy_") or name == "summary.json":
                (output / name).write_bytes((self.source / name).read_bytes())
        (output / "report.md").write_text("plafonds réservés\n\n" + code_pilot.RESERVATION_NOTE + "\n\n"
                                          + code_pilot.PUBLICATION_NOTE + "\n", encoding="utf-8")

    def derive(self, output="public"):
        with patch.object(pub, "frozen_inputs", return_value=self.pins), \
                patch.object(pub, "DATASET_HASH", self.campaign["dataset_hash"]), \
                patch.object(repo_tasks, "provenance", return_value=self.dataset), \
                patch.object(pub, "render", side_effect=self.fake_render):
            return pub.derive(self.source, self.root / output)

    def validate(self):
        with patch.object(pub, "DATASET_HASH", self.campaign["dataset_hash"]):
            pub.validate(self.campaign, self.rows, self.logs, self.dataset)

    def test_two_derivations_identical_sources_private_and_public_bindings(self):
        first = self.derive("first")
        second = self.derive("second")
        self.assertEqual(first, second)
        self.assertEqual(self.pins, pub.inventory(pub.read_files(self.source, pub.ARTIFACTS)))
        names = pub.ARTIFACTS | {"publication.json"}
        self.assertEqual(pub.read_files(self.root / "first", names), pub.read_files(self.root / "second", names))
        for row in json.loads((self.root / "first/verdicts.json").read_text()):
            raw = (self.root / "first" / ("single__" + row["task"] + ".jsonl")).read_bytes()
            self.assertEqual(pub.digest(raw), row["transcript_sha256"])
            self.assertNotIn(self.home.encode(), raw)
            self.assertNotIn(self.windows.encode(), raw)
            log = [json.loads(line) for line in raw.splitlines()]
            self.assertEqual(log[-2], pub.source_reference(row["task"], self.dataset))
            self.assertEqual(log[-1], {"kind": "end", **{k: v for k, v in row.items() if k != "transcript_sha256"}})
            original = next(r for r in self.rows if r["task"] == row["task"])
            for key in ("seconds", "passed", "campaign_hash", "completion_tokens", "charged_request_ceilings"):
                self.assertEqual(row[key], original[key])
        self.assertGreater(first["redactions"]["occurrences"], 0)
        self.assertNotIn("fixture/runtime", json.dumps(first))
        self.assertEqual((self.root / "first/summary.json").read_bytes(), (self.source / "summary.json").read_bytes())

    def test_corrupted_input_hash_and_existing_output_refused(self):
        (self.source / "manifest.json").write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.derive()
        self.refresh()
        self.derive()
        with self.assertRaisesRegex(ValueError, "must be new"):
            self.derive()

    def test_public_copy_bindings_generator_and_unmasked_path_refused(self):
        self.derive()
        names = pub.ARTIFACTS | {"publication.json"}
        original = pub.read_files(self.root / "public", names)
        with patch.object(pub, "frozen_inputs", return_value=self.pins), \
                patch.object(pub, "DATASET_HASH", self.campaign["dataset_hash"]), \
                patch.object(repo_tasks, "provenance", return_value=self.dataset), \
                patch.object(pub, "render", side_effect=self.fake_render):
            pub.reproduce(self.root / "public")
            for kind in ("hash", "generator", "count", "private_path", "metric"):
                for name, raw in original.items():
                    (self.root / "public" / name).write_bytes(raw)
                metadata = json.loads(original["publication.json"])
                if kind == "hash":
                    metadata["public_artifacts"]["manifest.json"]["sha256"] = "0" * 64
                elif kind == "generator":
                    metadata["generators"]["report"]["sha256"] = "0" * 64
                elif kind == "count":
                    metadata["redactions"]["occurrences"] += 1
                else:
                    path = self.root / "public/verdicts.json"
                    rows = json.loads(path.read_bytes())
                    if kind == "private_path":
                        rows[0]["verifier"]["calls"][0]["diagnostics"] = self.home
                    else:
                        rows[0]["seconds"] += 1
                    path.write_bytes(encoded(rows))
                    metadata["public_artifacts"]["verdicts.json"] = pub.inventory({"verdicts.json": path.read_bytes()})[
                        "verdicts.json"]
                (self.root / "public/publication.json").write_bytes(encoded(metadata))
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    pub.reproduce(self.root / "public")
    def test_bindings_identity_parameters_accounting_and_grade_refused(self):
        task = repo_tasks.PILOT[0]
        mutations = [lambda: self.rows[0].update(campaign_hash="a" * 64),
                     lambda: self.logs[task][0].update(task="other"),
                     lambda: self.logs[task][-1].update(seconds=14),
                     lambda: self.logs[task][1]["parameters"].update(seed=1),
                     lambda: self.logs[task][1].update(served_model="other"),
                     lambda: self.logs[task][1]["usage"].update(completion_tokens=2),
                     lambda: self.rows[0].update(charged_request_ceilings=1),
                     lambda: self.rows[0].update(completion_tokens=99),
                     lambda: self.rows[0].update(passed=True),
                     lambda: self.campaign["limits"].update(max_tokens=2048)]
        saved = copy.deepcopy((self.campaign, self.logs, self.rows))
        for change in mutations:
            self.campaign, self.logs, self.rows = copy.deepcopy(saved)
            change()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate()

    def test_changed_final_initial_unpinned_missing_and_data_refused(self):
        task = repo_tasks.PILOT[0]
        saved = copy.deepcopy(self.logs)
        for kind in ("final", "initial", "both", "missing"):
            self.logs = copy.deepcopy(saved)
            changes = self.logs[task][-2]
            if kind in ("initial", "both"):
                changes["initial"]["module.py"] = "different\n"
            if kind in ("final", "both"):
                changes["sources"]["module.py"] = "different\n"
            if kind == "missing":
                changes["initial"].clear()
                changes["sources"].clear()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate()
        self.logs = saved
        self.dataset["files"][task + "/base/module.py"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "dataset"):
            self.validate()

    def test_only_text_is_redacted_scientific_string_refused(self):
        values = {"message": {"content": self.home}, "command": self.windows, "diagnostics": self.home}
        counts = Counter()
        clean = pub.sanitize(values, counts)
        self.assertEqual(sum(counts.values()), 3)
        self.assertEqual(clean["message"]["content"], "<HOME>/runtime")
        self.assertEqual(clean["command"], "<HOME>\\runtime")
        for value in ({"returned": {"value": self.home}}, {self.home: "diagnostic"}):
            with self.assertRaises(ValueError):
                pub.sanitize(value, Counter())

    def test_candidate_sources_are_only_hashed_never_executed(self):
        source = "raise AssertionError('candidate code must stay text')\n"
        for task in repo_tasks.PILOT:
            self.dataset["files"][task + "/base/module.py"] = {"size": len(source.encode()),
                                                               "sha256": pub.digest(source.encode())}
            self.logs[task][-2].update(initial={"module.py": source}, sources={"module.py": source})
        self.campaign["dataset_hash"] = code_pilot.hash_json(self.dataset)
        for row in self.rows:
            row.update(campaign_hash=code_pilot.hash_json(self.campaign),
                       source_hashes={"module.py": pub.digest(source.encode())})
            self.logs[row["task"]][-1] = {"kind": "end", **{k: v for k, v in row.items() if k != "transcript_sha256"}}
        self.refresh()
        self.derive()
        for task in repo_tasks.PILOT:
            self.assertNotIn(source, (self.root / "public" / ("single__" + task + ".jsonl")).read_text())

    def test_duplicate_keys_nonfinite_and_linked_inputs_refused(self):
        for raw in (b'{"x": 1, "x": 2}', b'{"x": NaN}'):
            with self.assertRaises(ValueError):
                pub.load_json(raw)
        link = self.root / "linked"
        try:
            link.symlink_to(self.source, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaises(ValueError):
            pub.read_files(link, pub.ARTIFACTS)


@unittest.skipUnless(os.environ.get("REPO_PUBLICATION_REAL_TESTS") == "1", "explicit offline real-artifact test")
class RealPublicationTests(unittest.TestCase):
    def test_real_two_derivations_and_public_report_copy(self):
        source = pub.ROOT / "results" / pub.PRIVATE
        before = pub.inventory(pub.read_files(source, pub.ARTIFACTS))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pub.derive(source, root / "one")
            pub.derive(source, root / "two")
            names = pub.ARTIFACTS | {"publication.json"}
            self.assertEqual(pub.read_files(root / "one", names), pub.read_files(root / "two", names))
            shutil.copytree(root / "one", root / "copy")
            pub.reproduce(root / "copy")
            self.assertEqual(pub.read_files(root / "one", names), pub.read_files(root / "copy", names))
        self.assertEqual(before, pub.inventory(pub.read_files(source, pub.ARTIFACTS)))
