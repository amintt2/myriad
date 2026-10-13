"""Hermetic preparation tests; no scientific TEST contents, network, model or Colab operations."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e17_final_freeze as final
import e17_campaign as historical


class Preparation(unittest.TestCase):
    def fixture(self, root):
        value = {"schema": 1, "content_opened": False, "holdout_executed": False,
                 "status": "metadata-only-reservation-before-new-scores",
                 "aider": {"test": ["bottle-song", "bowling"], "dev": ["synthetic-dev"]},
                 "bigcode": {"test": ["BigCodeBench/208", "BigCodeBench/211"], "dev": ["BigCodeBench_120"],
                             "test_metadata": {t: {"state": "not_examined"} for b, t in final.TASKS if b == "bigcode"}}}
        for name in [final.SELECTION, final.PROPOSAL, final.ANALYSIS, *final.RESOURCE_FILES,
                     "phase0/essaim/synthetic.py", "phase0/colab/fake_cli.py"]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(final.canonical(value) if name == final.SELECTION else b"synthetic source\n")
        return value

    def policy(self):
        return final.budget(0, 18.85, 5.3, 10000, 1800, 1800)

    def test_exact_cells_models_and_historical_r00_guard(self):
        cells = final.cells()
        self.assertEqual(len(cells), 16)
        self.assertEqual([c["index"] for c in cells], list(range(16)))
        self.assertEqual([(c["mode"], c["benchmark"], c["task"]) for c in cells],
                         [(m, b, t) for m in ("single", "vote", "cascade", "reference") for b, t in final.TASKS])
        self.assertEqual(cells[8]["reference"], historical.REFERENCE)
        self.assertEqual(cells[12]["models"], [historical.REFERENCE])
        self.assertEqual(final.MODELS, historical.MODELS)
        self.assertEqual(final.LIMITS, historical.LIMITS)
        with self.assertRaises(ValueError):
            historical.plan(final.NAME)

    def test_prepare_validate_and_refuse_all_final_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            self.fixture(root)
            output = Path(temporary) / "prepared"
            value = final.prepare(output, self.policy(), root)
            self.assertEqual(final.validate(output, root), value)
            self.assertFalse(value["frozen"])
            self.assertFalse(value["ready_for_test"])
            self.assertIsNone(value["score"])
            with self.assertRaisesRegex(ValueError, "parent decisions"):
                final.require_final_freeze(output, root)
            with self.assertRaises(FileExistsError):
                final.prepare(output, self.policy(), root)

    def test_no_task_content_opened_or_included(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            poison = ["phase0/data/TEST.json", "phase0/bigcode_pilot/official_tests.json",
                      "phase0/code_pilot/tasks.json", "phase0/results/partial.json"]
            for name in poison:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic sentinel -- must never be read")
            original = Path.read_bytes
            def read(path):
                self.assertNotIn(path.relative_to(root).as_posix(), poison)
                return original(path)
            with mock.patch.object(Path, "read_bytes", read):
                files = final.code_bundle(root)
            self.assertFalse(set(poison) & set(files))
            self.assertEqual(final.archive(files), final.archive(files))

    def test_mutations_missing_foreign_and_added_source_refused(self):
        changes = ["cells", "models", "limits", "server", "budget", "selection_sha256", "ready_for_test"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            self.fixture(root)
            output = Path(temporary) / "prepared"
            value = final.prepare(output, self.policy(), root)
            path = output / "preparation.json"
            for key in changes:
                changed = copy.deepcopy(value)
                changed[key] = "foreign"
                path.write_bytes(final.canonical(changed))
                with self.subTest(key=key), self.assertRaises((ValueError, KeyError, TypeError)):
                    final.validate(output, root)
            path.write_bytes(final.canonical(value))
            added = root / "phase0/essaim/new_adapter.py"
            added.write_bytes(b"new source\n")
            with self.assertRaises(ValueError):
                final.validate(output, root)
            added.unlink()
            source = root / "phase0/essaim/synthetic.py"
            source.write_bytes(b"altered\n")
            with self.assertRaises(ValueError):
                final.validate(output, root)
            source.unlink()
            with self.assertRaises(ValueError):
                final.validate(output, root)

    def test_missing_archive_corruption_and_interrupted_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            self.fixture(root)
            output = Path(temporary) / "prepared"
            final.prepare(output, self.policy(), root)
            archive = output / "code.tar"
            archive.write_bytes(b"false success stdout")
            with self.assertRaises(ValueError):
                final.validate(output, root)
            archive.unlink()
            with self.assertRaises(OSError):
                final.validate(output, root)
            partial = Path(temporary) / "interrupted"
            partial.mkdir()
            with self.assertRaises(FileExistsError):
                final.prepare(partial, self.policy(), root)
            with self.assertRaises(OSError):
                final.validate(partial, root)

    def test_reservation_no_replacement_or_dev_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = self.fixture(Path(temporary))
            for benchmark in ("aider", "bigcode"):
                altered = copy.deepcopy(value)
                altered[benchmark]["test"][0] = "unsupported-replacement"
                with self.assertRaises(ValueError):
                    final.reservation(final.canonical(altered))
            value["bigcode"]["dev"] = ["BigCodeBench_208"]
            with self.assertRaises(ValueError):
                final.reservation(final.canonical(value))

    def test_budget_total_reserves_and_audited_admission(self):
        policy = self.policy()
        self.assertEqual(policy["charged_prior_units"], 1.15)
        self.assertEqual(policy["policy"]["minimum"], 16.15)
        self.assertEqual(policy["usage_wait_seconds"], 600)
        for values in ((2, 19, 5.3, 10000, 1800, 1800), (0, 18.85, 5.3, 11501, 1800, 1800),
                       (0, 18.85, 5.3, 10000, 1799, 1800), (0, 18.85, 5.3, 10000, 1800, 1799),
                       (0, 18.85, 100, 10000, 1800, 1800), (float("nan"), 18.85, 5.3, 10000, 1800, 1800)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                final.budget(*values)

    def test_duplicate_json_and_bundle_limit(self):
        with self.assertRaises(ValueError):
            final.read_json('{"ready":false,"ready":true}')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with mock.patch.object(final, "MAX_BUNDLE", 10):
                with self.assertRaises(ValueError):
                    final.code_bundle(root)


if __name__ == "__main__":
    unittest.main()
