"""K-class vote weights (analyze_e3.py, analyze_e4.py): abstentions, and the dev/test separation of E3."""
import json
import tempfile
import unittest
from pathlib import Path

import analyze_e3
import analyze_e4


def row(gold, answers):
    return {"gold": gold, "per_peer": [{"answer": a} for a in answers]}


class TestAbstentions(unittest.TestCase):
    def test_e3_collision_ignores_abstentions(self):
        # Audit 2026-10-09 (phase0, 5): None counted in the denominator of c (7/79 instead of 7/32).
        rows = {"q1": row("1", ["2", "2", None, None]), "q2": row("1", ["3", None, "4", "1"])}
        same, both, c = analyze_e3.collision(rows)
        self.assertEqual((same, both), (1, 2))  # pairs (2, 2) and (3, 4); no pair with an abstention
        self.assertAlmostEqual(c, 0.5)
        p = analyze_e3.vote_accuracies(["a", "b", "c", "d"], rows)
        self.assertEqual(p, [0.0, 0.0, 0.0, 1.0])  # among answers given; peer c gave one wrong answer
        self.assertEqual(analyze_e3.accuracies(["a", "b", "c", "d"], rows), [0.0, 0.0, 0.0, 0.5])

    def test_e4_collision_and_accuracy_given_an_answer(self):
        mk = lambda a: {"answer": a, "gold": "1"}
        by_model = {"m1": {"q": mk("2"), "r": mk(None)}, "m2": {"q": mk("2"), "r": mk("3")},
                    "m3": {"q": mk(None), "r": mk("1")}}
        self.assertAlmostEqual(analyze_e4.collision(by_model, ["q", "r"]), 1.0)  # one pair (2, 2), equal
        self.assertAlmostEqual(analyze_e4.valid_accuracy(by_model["m3"], ["q", "r"]), 1.0)
        self.assertAlmostEqual(analyze_e4.valid_accuracy(by_model["m1"], ["q", "r"]), 0.0)


class TestE3Partitions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = analyze_e3.RESULTS
        analyze_e3.RESULTS = Path(self.tmp.name)
        self.man = {"peers": [{"model": "m", "weights": {"sha256": "x"}}], "data": {"sha256": "d"}, "prompt": "p",
                    "protocol": "gen-v2", "template_date": "2026-10-08", "solo_tokens": 320, "block": 16, "k": None,
                    "max_rounds": 40, "tau": -2.5}

    def tearDown(self):
        analyze_e3.RESULTS = self.old
        self.tmp.cleanup()

    def write(self, split, **over):
        p = Path(self.tmp.name) / f"gen_t_{split}.jsonl.meta.json"
        p.write_text(json.dumps({**self.man, "split": split, **over}), encoding="utf-8")

    def test_overlap_and_identity_refused(self):
        # Audit 2026-10-09 (phase0, 6): the same rows loaded as dev and test were accepted.
        self.write("dev")
        self.write("test")
        analyze_e3.check_fit_test("t", {"a": 1}, {"b": 1})  # disjoint, same peers and protocol: accepted
        with self.assertRaises(SystemExit):
            analyze_e3.check_fit_test("t", {"a": 1, "b": 1}, {"b": 1})
        self.write("test", peers=[{"model": "m", "weights": {"sha256": "y"}}])
        with self.assertRaises(SystemExit):
            analyze_e3.check_fit_test("t", {"a": 1}, {"b": 1})
        self.write("test", tau=-3.0)
        with self.assertRaises(SystemExit):
            analyze_e3.check_fit_test("t", {"a": 1}, {"b": 1})


if __name__ == "__main__":
    unittest.main()
