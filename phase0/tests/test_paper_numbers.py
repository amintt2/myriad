"""Historical E1/E7 macro extraction and E4 source guards, without inference."""
import ast
import copy
import json
import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class PaperNumbers(unittest.TestCase):
    def setUp(self):
        source = ast.parse((ROOT / "paper" / "make_numbers.py").read_text(encoding="utf-8"))
        names = {"put", "signed", "ci", "parse_gain", "put_e1_reference", "put_e7_quality", "need"}
        funcs = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.env = {"macros": {}, "re": re, "math": math,
                    "E4_BENCHES": dict.fromkeys(("gsm8k", "math500", "arc", "mmlupro"))}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "make_numbers.py", "exec"), self.env)
        res = ROOT / "phase0" / "results"
        self.mc = json.loads((res / "mc_summary_test.json").read_text(encoding="utf-8"))
        self.report = (res / "mc_report_test.md").read_text(encoding="utf-8")
        self.e7 = json.loads((ROOT / "app" / "bench" / "results" / "e7_v11_throughput.json")
                             .read_text(encoding="utf-8"))
        self.generated = (ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8")

    def test_e1_reference_intervals_match_historical_report(self):
        self.env["put_e1_reference"](self.report, self.mc)
        macros = self.env["macros"]
        self.assertEqual(macros["eOneArcRefGain"], "$-$3.2")
        self.assertEqual(macros["eOneArcRefGainCI"], "[$-$6.3, $-$0.5]")
        self.assertEqual(macros["eOneMmluRefGain"], "$-$7.2")
        self.assertEqual(macros["eOneMmluRefGainCI"], "[$-$10.5, $-$3.9]")
        for name, value in macros.items():
            self.assertIn(f"\\newcommand{{\\{name}}}{{{value}}}", self.generated)

    def test_e1_missing_ambiguous_or_inconsistent_report_refused(self):
        line = "- mélange calibré : 1 passage -3.2 [-6.3 ; -0.5],"
        for report in (self.report.replace("## ARC-Challenge", "## Removed"),
                       self.report.replace(line, ""), self.report.replace(line, line + "\n" + line),
                       self.report.replace(line, line.replace("-3.2", "-2.2")),
                       self.report.replace(line, line.replace("-6.3", "-1.3"))):
            with self.subTest(report=report[:40]), self.assertRaises(SystemExit):
                self.env["put_e1_reference"](report, self.mc)

    def test_e7_quality_matches_quoted_plateau(self):
        self.env["put_e7_quality"](self.e7, "73.7")
        macros = self.env["macros"]
        self.assertEqual(macros["eSevenStageSeconds"], "15")
        self.assertEqual(macros["eSevenFailedSixtyFour"], "32")
        self.assertEqual(macros["eSevenAccuracySixtyFour"], "85.9")
        self.assertEqual(macros["eSevenAskedSixtyFour"], "3.25")
        for name, value in macros.items():
            self.assertIn(f"\\newcommand{{\\{name}}}{{{value}}}", self.generated)

    def test_e1_invalid_or_unpaired_scores_refused(self):
        for values in ([], [1.1], [float("nan")], [0]):
            source = copy.deepcopy(self.mc)
            source["benches"]["arc"]["systems"]["fusion:mélange calibré"]["pass"] = values
            with self.subTest(values=values), self.assertRaises(SystemExit):
                self.env["put_e1_reference"](self.report, source)
            self.assertEqual(self.env["macros"], {})

    def test_e7_missing_ambiguous_or_wrong_protocol_refused(self):
        for mutate in (lambda d: d.update(protocol_version="essaim/1.3"),
                       lambda d: d.update(scenarios=[]),
                       lambda d: d["scenarios"].append(copy.deepcopy(d["scenarios"][0])),
                       lambda d: d["scenarios"][0]["scenario"].update(rtt_ms=0),
                       lambda d: d["scenarios"][0]["phases"].extend(copy.deepcopy(d["scenarios"][0]["phases"]))):
            source = copy.deepcopy(self.e7)
            mutate(source)
            with self.subTest(mutation=mutate), self.assertRaises(SystemExit):
                self.env["put_e7_quality"](source, "73.7")
            self.assertEqual(self.env["macros"], {})
        with self.assertRaises(SystemExit):
            self.env["put_e7_quality"](self.e7, "999.9")

    def test_e7_invalid_quality_refused(self):
        for key, value in (("failed", -1), ("failed", 0.5), ("accuracy_all", 1.1),
                           ("accuracy_all", float("nan")), ("peers_asked_mean", 5), ("duration", 0)):
            source = copy.deepcopy(self.e7)
            p = next(p for p in source["scenarios"][0]["phases"] if f"{p['goodput_rps']:.1f}" == "73.7")
            (p["phase"] if key == "duration" else p)[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(SystemExit):
                self.env["put_e7_quality"](source, "73.7")
            self.assertEqual(self.env["macros"], {})

    def test_e4_missing_preview_and_incomplete_sources_refused(self):
        complete = {"benches": dict(self.env["E4_BENCHES"])}
        for source in (None, {**complete, "preview": True}, {"benches": {"gsm8k": {}}}):
            self.env["load"] = lambda name: source
            with self.subTest(source=source), self.assertRaises(SystemExit):
                self.env["need"]("summary.json")
        self.env["load"] = lambda name: complete
        self.assertIs(self.env["need"]("summary.json"), complete)


if __name__ == "__main__":
    unittest.main()
