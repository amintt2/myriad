"""Pinned selection, isolated printer grading and native historical-runner contracts."""
import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from essaim import code_pilot, repo_tasks
from essaim.local_linux import IsolationError

ROOT = Path(__file__).resolve().parents[1]
IDS = ["psf__requests-1963", "psf__requests-2148", "psf__requests-2317", "psf__requests-2674",
       "psf__requests-3362", "psf__requests-863"]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "repo_pilot/requests_diagnostic.json").read_text(encoding="utf-8"))

    def test_no_excluded_task_becomes_an_executable_pilot(self):
        data = self.manifest
        self.assertEqual(data["status"], "blocked")
        self.assertIs(data["model_inference"], False)
        self.assertEqual(list(data["tasks"]), IDS)
        selected = sorted(task for task, row in data["tasks"].items() if row["eligible"])
        self.assertEqual(data["selection"]["eligible"], selected)
        self.assertEqual(data["selection"]["pilot"], selected[:data["selection"]["limit"]])
        self.assertEqual(selected, [])
        self.assertEqual(data["selection"]["limit"], 3)
        for row in data["tasks"].values():
            self.assertIs(row["eligible"], False)
            self.assertIn("nonpermissive_vendored_dependency", row["exclusions"])
            self.assertEqual(row["vendored_dependency"]["licence"], "LGPL-2.1-or-later")
            self.assertEqual(row["controller_adapter"], "not_validated")
        self.assertEqual(data["tasks"][IDS[-1]]["repository_licence"], "ISC")

    def test_manifest_has_only_metadata_not_issues_solutions_or_tests(self):
        data = self.manifest
        self.assertEqual(set(data), {"version", "date", "status", "model_inference", "scope", "dataset",
                                     "swe_bench_revision", "sources", "selection", "tasks"})
        expected = {"repo", "base_commit", "archive", "evidence_files", "repository_licence",
                    "vendored_dependency", "added_tests_network", "controller_adapter", "eligible",
                    "exclusions", "record_sha256"}
        for task, row in data["tasks"].items():
            self.assertEqual(set(row), expected)
            evidence = {"LICENSE", "NOTICE", "requests/compat.py", "requests/packages/chardet/__init__.py"}
            if task == "psf__requests-863":
                evidence.add("requests/packages/chardet2/__init__.py")
            self.assertEqual(set(row["evidence_files"]), evidence)
            for entry in row["evidence_files"].values():
                self.assertEqual(set(entry), {"size", "sha256"})
        self.assertEqual({task for task, row in data["tasks"].items() if row["added_tests_network"]},
                         {"psf__requests-2317", "psf__requests-2674"})

    def test_evidence_pins_exact_revision_and_bounded_content_hashes(self):
        data = self.manifest
        revision = "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
        self.assertEqual(data["dataset"]["revision"], revision)
        self.assertEqual(data["dataset"]["split"], "test")
        self.assertEqual(data["dataset"]["rows"], 300)
        parquet = data["sources"]["test-parquet"]
        self.assertIn("/" + revision + "/", parquet["url"])
        self.assertEqual(parquet["size"], 1119540)
        self.assertEqual(parquet["sha256"], "7a21f37b8bc179c7db5beeb14e88ac538ba283455c776e6b2535bbfb6e3551b4")
        entries = list(data["sources"].values())
        for row in data["tasks"].values():
            self.assertRegex(row["base_commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(row["archive"]["url"], "https://codeload.github.com/psf/requests/tar.gz/"
                             + row["base_commit"])
            entries += [row["archive"], *row["evidence_files"].values()]
        for entry in entries:
            self.assertIs(type(entry["size"]), int)
            self.assertGreater(entry["size"], 0)
            self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]))


class PrinterSelectionTests(unittest.TestCase):
    def test_first_three_printers_and_intervening_exclusion(self):
        data = repo_tasks.provenance()
        prefix = [task for task in sorted(data["selection"]["candidates"]) if task <= repo_tasks.PILOT[-1]]
        self.assertEqual(prefix, [repo_tasks.PILOT[0], "sympy__sympy-11870", *repo_tasks.PILOT[1:]])
        self.assertEqual(data["selection"]["candidates"][prefix[1]]["state"], "excluded_nonprinting")
        self.assertTrue(all("sympy/printing/" in path for task in repo_tasks.PILOT
                            for path in data["selection"]["candidates"][task]["changed_source_paths"]))
        self.assertEqual(len(data["selection"]["candidates"]), 77)
        self.assertEqual([len(repo_tasks.cases(task)) for task in repo_tasks.PILOT], [7, 6, 5])

    def test_sources_and_agent_copy_cannot_include_final_artifacts(self):
        data = repo_tasks.provenance()
        for task, cfg in data["tasks"].items():
            self.assertFalse(cfg["pass_to_pass_covered"])
            self.assertEqual(cfg["licence"], "BSD-3-Clause")
            for name in cfg["sources"]:
                self.assertTrue(name.startswith("sympy/") and name.endswith(".py"))
                self.assertNotIn("tests", Path(name).parts)
                self.assertNotIn("benchmarks", Path(name).parts)
                self.assertIn(task + "/base/" + name, data["files"])
            self.assertNotIn(".git", cfg["base_files"])
            self.assertNotIn("reference.patch", cfg["base_files"])
            self.assertNotIn("final_cases.json", cfg["base_files"])

    def test_original_instructions_and_fixture_boundary(self):
        for task in repo_tasks.PILOT:
            instruction = repo_tasks.instructions(Path(), task)
            self.assertIn("https://github.com/sympy/sympy/pull/", instruction)
            for case in repo_tasks.cases(task):
                self.assertEqual(set(case), {"test", "code", "expected"})
                self.assertIs(type(case["expected"]), str)
                self.assertNotIn("assert ", case["code"])
                self.assertNotIn("expected", case["code"])
        self.assertNotIn("expected", repo_tasks.WORKER)

    def test_native_commands_use_relevant_files_and_restrict_latex(self):
        prefix = "python bin/test --no-subprocess --no-colors --seed 0 sympy/printing/tests/"
        commands = [prefix + "test_ccode.py", prefix + "test_latex.py -k test_latex_Piecewise",
                    prefix + "test_mathematica.py"]
        for task, command in zip(repo_tasks.PILOT, commands):
            self.assertEqual(repo_tasks.visible_command(task), command)
            instruction = repo_tasks.instructions(Path(), task)
            self.assertIn(command, instruction)
            self.assertNotIn("python -m pytest", instruction)
            self.assertIn("not the final score", instruction)

    def test_repo_backend_defaults_do_not_change_aider_defaults(self):
        aider = code_pilot.parser().parse_args([])
        repo = code_pilot.parser(repo_tasks).parse_args([])
        self.assertEqual(aider.tasks_dir, ROOT / "data/code-pilot")
        self.assertEqual(repo.tasks_dir, ROOT / "data/repo-pilot")
        for key in ("max_steps", "max_tokens", "total_tokens", "timeout", "cmd_timeout", "verify_timeout", "seed"):
            self.assertEqual(getattr(aider, key), getattr(repo, key))


def fixture(root: Path):
    sources = {"sympy/__init__.py": "def symbols(s): return s.split()\n",
               "sympy/printing/__init__.py": "def ccode(x): return 'expected'\ndef latex(x): return 'expected'\n",
               "sympy/printing/mathematica.py": "def mathematica_code(x): return 'expected'\n"}
    base = root / "synthetic/base"
    for name, text in sources.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (base / "visible_test.py").write_text("assert False", encoding="utf-8")
    cfg = {"sources": list(sources), "base_files": [*sources, "visible_test.py"],
           "instruction": "Repair a synthetic security probe.", "pr": "https://example.org/synthetic"}
    return sources, {"tasks": {"synthetic": cfg}}


class VerifierTests(unittest.TestCase):
    def test_native_diagnostic_preserves_tests_and_accepts_only_known_assertion_conflict(self):
        task = repo_tasks.PILOT[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources, manifest = fixture(root)
            (root / "synthetic").rename(root / task)
            cfg = manifest["tasks"].pop("synthetic")
            cfg["reference_file"] = "sympy/__init__.py"
            manifest["tasks"][task] = cfg
            reference = root / task / "reference/sympy/__init__.py"
            reference.parent.mkdir(parents=True)
            reference.write_text("corrected source\n", encoding="utf-8")

            class Env:
                isolation = {"synthetic": True}
                def __init__(self, work, runtime):
                    self.work = work
                def run(self, command, timeout):
                    self_test.assertEqual(command, repo_tasks.visible_command(task))
                    self_test.assertEqual((self.work / "visible_test.py").read_text(), "assert False")
                    source = (self.work / "sympy/__init__.py").read_text()
                    if source == sources["sympy/__init__.py"]:
                        return 0, "tests finished: 1 passed, in 0.1 seconds"
                    self_test.assertEqual(source, "corrected source\n")
                    return 1, "tests finished: 0 passed, 1 failed, in 0.1 seconds"
                def close(self):
                    pass

            self_test = self
            with patch.object(repo_tasks, "provenance", return_value=manifest), \
                    patch.object(repo_tasks, "LocalLinuxEnv", Env):
                base = repo_tasks.visible_tests(root, task, Path(), False)
                corrected = repo_tasks.visible_tests(root, task, Path(), True)
                self.assertEqual(base["exit_code"], 0)
                self.assertEqual(corrected["exit_code"], 1)
                self.assertIs(corrected["historical_assertion_conflict"], True)
                self.assertIs(corrected["final_score"], False)
                with patch.object(Env, "run", return_value=(1, "tests finished: 0 passed, 1 exceptions")):
                    with self.assertRaises(IsolationError):
                        repo_tasks.visible_tests(root, task, Path(), True)
            self.assertEqual((root / task / "base/visible_test.py").read_text(), "assert False")

    def test_repository_report_requires_its_own_complete_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = {"models": {"single": ["probe"]}, "watts": 100, "eur_kwh": 0.25}
            rows = []
            for task in repo_tasks.PILOT:
                transcript = root / ("single__" + task + ".jsonl")
                transcript.write_text("synthetic trace\n")
                rows.append({"task": task, "mode": "single", "status": "graded", "passed": False,
                             "campaign_hash": code_pilot.hash_json(campaign), "seconds": 1,
                             "completion_tokens": None, "charged_request_ceilings": 10,
                             "transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest()})
            with patch.object(code_pilot, "figures"):
                summary = code_pilot.report(rows, campaign, root, repo_tasks)
                self.assertEqual(summary["modes"]["single"]["passed"], 0)
                self.assertIn("SymPy sur dépôt réel", (root / "report.md").read_text(encoding="utf-8"))
                with self.assertRaises(ValueError):
                    code_pilot.report(rows[:-1], campaign, root, repo_tasks)
                with self.assertRaises(ValueError):
                    code_pilot.report(rows, campaign, root)

    def test_infrastructure_failure_is_not_a_candidate_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources, manifest = fixture(root)
            with patch.object(repo_tasks, "provenance", return_value=manifest), \
                    patch.object(repo_tasks, "cases", return_value=[{"test": "probe", "code": "value = ccode('input')",
                                                                  "expected": "expected"}]), \
                    patch.object(repo_tasks, "LocalLinuxEnv", side_effect=IsolationError("missing kernel protection")):
                grade = repo_tasks.verify(root, "synthetic", sources, Path())
            self.assertEqual(grade["status"], "error")
            self.assertIsNone(grade["passed"])

    def test_shared_episode_uses_repository_backend_and_discards_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources, manifest = fixture(root)
            args = SimpleNamespace(tasks_dir=root, runtime=Path(), max_tokens=10, total_tokens=30, seed=0,
                                   max_steps=2, timeout=10, cmd_timeout=1, verify_timeout=5, reference=None)

            class Env:
                isolation = {"synthetic": True}
                def __init__(self, work, runtime):
                    self.work = work
                def run(self, command, timeout):
                    (self.work / "visible_test.py").write_text("forged score")
                    return 0, "fake score"
                def close(self):
                    pass

            class Client:
                def post(self, path, json, timeout):
                    reply = "```bash\necho done\n```" if json["messages"][-1]["role"] == "user" else ""
                    return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {
                        "choices": [{"message": {"content": reply}}], "usage": {"completion_tokens": 1}})

            verdict = {"status": "graded", "passed": True, "seconds": 0}
            with patch.object(repo_tasks, "provenance", return_value=manifest), \
                    patch.object(repo_tasks, "visible_command", return_value="synthetic diagnostic"), \
                    patch.object(code_pilot, "LocalLinuxEnv", Env), \
                    patch.object(repo_tasks, "verify", return_value=verdict) as grade:
                row = code_pilot.run_one(args, "synthetic", "single", ["probe"], Client(), root, "hash", repo_tasks)
            self.assertIs(row["passed"], True)
            self.assertEqual(grade.call_args.args[2], sources)


@unittest.skipUnless(sys.platform.startswith("linux") and os.environ.get("REPO_PILOT_LINUX_TESTS") == "1",
                     "explicit real Linux printer transport tests")
class LinuxPrinterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="myriad-repo-probe-")
        self.root = Path(self.temporary.name)
        self.sources, self.manifest = fixture(self.root)
        default = str(Path.home() / ".local/share/myriad-repo-pilot/venv")
        self.runtime = Path(os.environ.get("REPO_PILOT_RUNTIME", default))
        self.pins = patch.object(repo_tasks, "provenance", return_value=self.manifest)
        cases = [{"test": "probe", "code": "value = ccode('input')", "expected": "expected"}]
        self.fixtures = patch.object(repo_tasks, "cases", return_value=cases)
        self.pins.start()
        self.fixtures.start()

    def tearDown(self):
        self.fixtures.stop()
        self.pins.stop()
        self.temporary.cleanup()

    def test_base_positive_and_printed_scores_do_not_control_grade(self):
        self.assertIs(repo_tasks.verify(self.root, "synthetic", self.sources, self.runtime)["passed"], True)
        for returned, expected in (("expected", True), ("wrong", False)):
            source = "print('{\"passed\":true}')\ndef ccode(x): return " + repr(returned)
            sources = {**self.sources, "sympy/printing/__init__.py": source + "\ndef latex(x): return ''\n"}
            grade = repo_tasks.verify(self.root, "synthetic", sources, self.runtime)
            self.assertEqual(grade["status"], "graded", grade)
            self.assertIs(grade["passed"], expected)
            self.assertIn("passed", grade["calls"][0]["diagnostics"])

    def test_exit_syntax_and_timeout_are_candidate_failures(self):
        for source in ("import os; os._exit(0)", "def broken(:", "while True: pass"):
            sources = {**self.sources, "sympy/__init__.py": source}
            grade = repo_tasks.verify(self.root, "synthetic", sources, self.runtime, timeout_s=0.8)
            self.assertEqual(grade["status"], "graded", grade)
            self.assertIs(grade["passed"], False)

    def test_forged_nonstring_or_invalid_utf8_result_is_rejected(self):
        for payload in ('{"value":true}', '{"value":"\\udfff"}', '{"value":"' + 'a' * 65537 + '"}'):
            source = "open('_return.json', 'w').write(" + repr(payload) + "); import os; os._exit(0)"
            sources = {**self.sources, "sympy/__init__.py": source}
            grade = repo_tasks.verify(self.root, "synthetic", sources, self.runtime)
            self.assertEqual(grade["status"], "graded", grade)
            self.assertIs(grade["passed"], False)
            self.assertEqual(grade["calls"][0]["candidate_failure"], "invalid_result_artifact")

    def test_symlink_result_and_hidden_assertion_access_fail(self):
        for source in ("import os; os.symlink('/etc/passwd', '_return.json')\n",
                       "open(" + repr(str(ROOT / "repo_pilot/final_cases.json")) + ").read()\n"):
            sources = {**self.sources, "sympy/__init__.py": source + "def symbols(s): return s.split()\n"}
            grade = repo_tasks.verify(self.root, "synthetic", sources, self.runtime)
            self.assertEqual(grade["status"], "graded", grade)
            self.assertIs(grade["passed"], False)

    def test_visible_test_modification_is_not_reapplied(self):
        work = self.root / "agent"
        work.mkdir()
        allowed = repo_tasks.copy_task(self.root, "synthetic", work)
        (work / "visible_test.py").write_text("raise SystemExit(0)")
        sources = repo_tasks.collect_sources(work, allowed)
        self.assertNotIn("visible_test.py", sources)
        self.assertIs(repo_tasks.verify(self.root, "synthetic", sources, self.runtime)["passed"], True)


if __name__ == "__main__":
    unittest.main()
