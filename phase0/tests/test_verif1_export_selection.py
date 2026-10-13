"""Public selection and scanner checks use synthetic text only; never read stored research results."""
import importlib.util
import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

EXPORTER = Path(__file__).resolve().parents[2] / "tools/export_public.py"
export = None
if EXPORTER.exists():
    spec = importlib.util.spec_from_file_location("verif_export_fixture", EXPORTER)
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)


@unittest.skipIf(export is None, "tools/export_public.py absent (public tree)")
class VerifExportTests(unittest.TestCase):
    def test_private_dev_results_excluded_but_operational_sources_and_guide_selected(self):
        for name in ("verif1_registry_dev.jsonl", "percees_report_dev.md", "verif1_summary_dev.json",
                     "verif1_campaign/future.jsonl", "percees_future/nested/report.md"):
            self.assertFalse(export.selected("phase0/results/" + name), name)
        for name in ("phase0/colab/verif1_supervisor.py", "phase0/verif1_analyze.py",
                     "docs/12_percees_verif1_colab.md", "phase0/results/code_fixture_dev.jsonl"):
            self.assertTrue(export.selected(name), name)

    def test_scanner_still_blocks_private_absolute_path_in_synthetic_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "phase0/results/code_fixture_dev.jsonl"
            path = root / name
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"diagnostic": "C:" + "/Users/" + "private-owner/example"}) + "\n")
            blocking, _, _ = export.scan(root, [name])
            self.assertTrue(blocking)

class VerifPublicCliTests(unittest.TestCase):
    def test_public_discovery_without_private_exporter_keeps_cli_test_active(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / "phase0/tests"
            tests.mkdir(parents=True)
            target = tests / "test_public_selection_fixture.py"
            target.write_bytes(Path(__file__).read_bytes())
            loader = unittest.TestLoader()
            suite = loader.discover(str(tests), pattern=target.name)
            self.assertEqual(suite.countTestCases(), 4)
            module = sys.modules["test_public_selection_fixture"]
            self.assertIsNone(module.export)
            selected = unittest.TestSuite([loader.loadTestsFromTestCase(module.VerifExportTests),
                module.VerifPublicCliTests("test_differential_cli_requires_explicit_dev_paths_before_artifact_access")])
            class PublicResult(unittest.TestResult):
                def addSkip(self, test, reason):
                    self.skipped.append((test, reason))
            result = PublicResult()
            selected.run(result)
            self.assertEqual(result.testsRun, 3)
            self.assertEqual(len(result.skipped), 2)
            self.assertFalse(result.errors)
            self.assertFalse(result.failures)
            sys.modules.pop("test_public_selection_fixture", None)

    def test_differential_cli_requires_explicit_dev_paths_before_artifact_access(self):
        import verif1_differential as differential
        for mode in ("prepare", "analyze"):
            with patch.object(sys, "argv", ["verif1_differential.py", mode]), \
                    patch.object(differential, "path", side_effect=AssertionError("no artifact access")) as path, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    differential.main()
                self.assertEqual(failure.exception.code, 2)
                path.assert_not_called()


if __name__ == "__main__":
    unittest.main()
