"""E10 campaign completeness and paper macros, without a model or GPU."""
import argparse
import ast
import copy
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase0"))
import analyze_sot
from colab_jobs import SOT
from essaim import sot


class Completeness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.res = Path(self.tmp.name)
        for p in (ROOT / "phase0" / "results").glob("sot_*_colab_test.jsonl*"):
            shutil.copyfile(p, self.res / p.name)
        self.paths = patch.object(sot, "RESULTS", self.res)
        self.paths.start()
        self.outline = sot.outline_path(SOT["outline"], "_colab", "test")
        man = json.loads(self.outline.with_name(self.outline.name + ".meta.json").read_text(encoding="utf-8"))
        self.identity = patch.object(analyze_sot.data, "dataset_identity", return_value=man["data"])
        self.identity.start()
        self.items = [{"id": r["id"]} for r in self.rows(self.outline)]
        self.args = argparse.Namespace(tag="sot1", outline_model=SOT["outline"], judge=SOT["judge"],
                                       peers=SOT["peers"], baselines=SOT["peers"], suffix="_colab", split="test")

    def tearDown(self):
        self.identity.stop()
        self.paths.stop()
        self.tmp.cleanup()

    def rows(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def write(self, path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def test_complete_test_partition(self):
        self.assertEqual(len(analyze_sot.validate_split(self.args, self.items)), 28)

    def test_baseline_subset_and_independent_order(self):
        for baselines in ([SOT["outline"]], [SOT["peers"][1], SOT["outline"]], [SOT["peers"][1]]):
            with self.subTest(baselines=baselines):
                self.args.baselines = baselines
                hashes = analyze_sot.validate_split(self.args, self.items)
                self.assertEqual(len(hashes), 28)
                # The complete judge input remains validated and hashed, including unrequested solos.
                self.assertIn(sot.base_path(SOT["peers"][-1], "_colab", "test").name, hashes)
                out = self.res / "subset.json"
                argv = ["analyze_sot.py", "--tag", "sot1", "--outline-model", SOT["outline"],
                        "--judge", SOT["judge"], "--suffix", "_colab", "--peers", *SOT["peers"],
                        "--baselines", *baselines, "--splits", "test", "--best", baselines[0],
                        "--boot", "100", "--json", str(out)]
                with patch.object(sys, "argv", argv), patch.object(analyze_sot.data, "mt_bench", return_value=self.items), \
                        contextlib.redirect_stdout(io.StringIO()):
                    analyze_sot.main()
                summary = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual(list(summary["splits"]["test"]["quality"]), baselines)
                self.assertEqual(summary["sources_sha256"], hashes)

    def test_subset_still_requires_unrequested_judgments_and_answers(self):
        self.args.baselines = [SOT["outline"]]
        judge = sot.judge_path("sot1", SOT["judge"], "_colab", "test")
        rows = self.rows(judge)
        self.write(judge, [r for r in rows if r["vs"] != SOT["peers"][-1]])
        with self.assertRaisesRegex(SystemExit, "clés"):
            analyze_sot.validate_split(self.args, self.items)
        self.write(judge, rows)
        path = sot.base_path(SOT["peers"][-1], "_colab", "test")
        self.write(path, self.rows(path)[:-1])
        with self.assertRaisesRegex(SystemExit, "clés"):
            analyze_sot.validate_split(self.args, self.items)

    def test_missing_extra_and_duplicate_rows_refused(self):
        path = sot.base_path(SOT["outline"], "_colab", "test")
        rows = self.rows(path)
        for changed in (rows[:-1], rows + [rows[0]], rows + [{**rows[0], "id": "foreign"}]):
            with self.subTest(size=len(changed)):
                self.write(path, changed)
                with self.assertRaisesRegex(SystemExit, "clés"):
                    analyze_sot.validate_split(self.args, self.items)

    def test_incomplete_tail_refused(self):
        self.outline.write_bytes(self.outline.read_bytes() + b'{"id":')
        with self.assertRaisesRegex(SystemExit, "tronqué"):
            analyze_sot.validate_split(self.args, self.items)

    def test_wrong_peer_order_refused(self):
        self.args.peers = list(reversed(self.args.peers))
        with self.assertRaisesRegex(SystemExit, "peers"):
            analyze_sot.validate_split(self.args, self.items)

    def test_foreign_data_manifest_refused(self):
        path = self.outline.with_name(self.outline.name + ".meta.json")
        man = json.loads(path.read_text(encoding="utf-8"))
        man["data"] = {**man["data"], "sha256": "foreign"}
        path.write_text(json.dumps(man), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "data"):
            analyze_sot.validate_split(self.args, self.items)

    def test_missing_judge_probabilities_refused(self):
        path = sot.judge_path("sot1", SOT["judge"], "_colab", "test")
        rows = self.rows(path)
        rows[0]["probs"] = None
        self.write(path, rows)
        with self.assertRaisesRegex(SystemExit, "probabilités"):
            analyze_sot.validate_split(self.args, self.items)


class PaperNumbers(unittest.TestCase):
    def setUp(self):
        # Extract the macro functions without executing unrelated E1--E7 data loading.
        source = ast.parse((ROOT / "paper" / "make_numbers.py").read_text(encoding="utf-8"))
        funcs = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ("put", "put_e10")]
        self.env = {"macros": {}, "re": __import__("re"), "SOT": SOT, "hashlib": hashlib, "json": json,
                    "RES": ROOT / "phase0" / "results", "pick_best": analyze_sot.pick_best}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "make_numbers.py", "exec"), self.env)
        self.summary = json.loads((self.env["RES"] / "sot_summary_colab.json").read_text(encoding="utf-8"))

    def test_generated_numbers_match_frozen_campaign(self):
        self.env["put_e10"](self.summary)
        macros = self.env["macros"]
        self.assertEqual(macros["eTenBestName"], "Qwen3.5-4B")
        self.assertEqual([macros["eTenTest" + k] for k in ("Win", "Tie", "Loss")], ["0", "4", "20"])
        self.assertEqual(macros["eTenTestScore"], "0.115")
        self.assertEqual(macros["eTenTestScoreCI"], "[0.042, 0.198]")
        self.assertEqual(macros["eTenTestSpeedup"], "1.68")
        self.assertEqual(macros["eTenTestCut"], "30")
        self.assertEqual(macros["eTenOtherFallback"], "1")
        generated = (ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8")
        for name, value in macros.items():
            self.assertIn(f"\\newcommand{{\\{name}}}{{{value}}}", generated)

    def test_incomplete_report_refused(self):
        mutations = [lambda e: e.update(complete=False),
                     lambda e: e["splits"].pop("other"),
                     lambda e: e["splits"]["test"]["quality"].pop(SOT["peers"][-1]),
                     lambda e: e["splits"]["test"]["structure"].update(n=23),
                     lambda e: e["splits"]["test"]["quality"][SOT["outline"]].update(n=23),
                     lambda e: e["splits"]["test"]["quality"][SOT["outline"]].pop("score_ci"),
                     lambda e: e["splits"]["test"]["structure"].pop("single_cut"),
                     lambda e: e["splits"]["test"]["speed"].pop("mono-flux + défauts|100"),
                     lambda e: e["sources_sha256"].pop(next(iter(e["sources_sha256"])))]
        for mutate in mutations:
            e = copy.deepcopy(self.summary)
            mutate(e)
            with self.subTest(mutation=mutate), self.assertRaises(SystemExit):
                self.env["put_e10"](e)
            self.assertEqual(self.env["macros"], {})

    def test_speed_overrides_and_defaulted_mutations_refused(self):
        mutations = [lambda e: e["speeds"].update({SOT["outline"]: 400.0}),
                     lambda e: e["speeds"].update({SOT["peers"][2]: 560.0}),
                     lambda e: e["speeds"].pop(SOT["peers"][-1]),
                     lambda e: e["defaulted"].remove(SOT["outline"]),
                     lambda e: e["defaulted"].append(SOT["peers"][2])]
        for mutate in mutations:
            e = copy.deepcopy(self.summary)
            mutate(e)
            with self.subTest(mutation=mutate), self.assertRaisesRegex(SystemExit, "débits effectifs"):
                self.env["put_e10"](e)
            self.assertEqual(self.env["macros"], {})

    def test_changed_speed_table_refused(self):
        table = json.loads((self.env["RES"].parent / "speeds_consumer.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            self.env["RES"] = Path(d) / "results"
            # Even the same numeric rate is incompatible if it is newly labelled measured.
            table[SOT["outline"]] = 40.0
            (Path(d) / "speeds_consumer.json").write_text(json.dumps(table), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "table de débits"):
                self.env["put_e10"](self.summary)

    def test_test_selection_and_modified_source_refused(self):
        for mutate in (lambda e: e["protocol"].update(select_split="test"),
                       lambda e: e.update(best_single=SOT["peers"][-1]),
                       lambda e: e["sources_sha256"].update({next(iter(e["sources_sha256"])): "bad"})):
            e = copy.deepcopy(self.summary)
            mutate(e)
            with self.assertRaises(SystemExit):
                self.env["put_e10"](e)


if __name__ == "__main__":
    unittest.main()
