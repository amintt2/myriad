"""analyze_cascade_e4.py: call signals (support, certificate margin) and the dev-chosen threshold."""
import unittest

import analyze_cascade_e4 as cas
import analyze_e4


class TestSignals(unittest.TestCase):
    def test_decision_support_margin(self):
        dec, support, margin = cas.signals([("1", 2.0), ("1", 1.0), ("2", 2.5), (None, 4.0), ("3", 0.0)])
        self.assertEqual(dec, "1")  # 3.0 against 2.5; the abstention and the zero-weight peer cast no vote
        self.assertEqual(support, 2)
        self.assertAlmostEqual(margin, 0.5)
        self.assertEqual(dec, analyze_e4.decide([("1", 2.0, None), ("1", 1.0, None), ("2", 2.5, None),
                                                 (None, 4.0, None), ("3", 0.0, None)]))

    def test_support_is_the_most_voted_answer_not_the_decision(self):
        # One heavy peer beats two light ones: the decision has one vote, but an answer has two.
        dec, support, margin = cas.signals([("1", 5.0), ("2", 1.0), ("2", 1.0)])
        self.assertEqual((dec, support), ("1", 2))
        self.assertAlmostEqual(margin, 3.0)

    def test_no_answer_and_single_answer(self):
        self.assertEqual(cas.signals([(None, 1.0), ("x", 0.0)]), (None, 0, 0.0))
        dec, support, margin = cas.signals([("x", 1.5), (None, 2.0)])
        self.assertEqual((dec, support), ("x", 1))
        self.assertAlmostEqual(margin, 1.5)  # no runner-up: the margin is the decision's score

    def test_margin_is_the_certificate_with_a_reserve(self):
        # Calling when margin <= t is calling when the exact certificate fails with missing weight t.
        answers = [("1", 2.0), ("1", 1.0), ("2", 2.5)]
        _, _, margin = cas.signals(answers)
        for t in (0.2, 0.5, 0.8):
            with_reserve = [(a, w, None) for a, w in answers] + [("2", t, None)]
            certified = analyze_e4.certificate_waits(with_reserve, [0, 1, 2, 3]) < 4
            self.assertEqual(certified, margin > t)


class TestChoose(unittest.TestCase):
    def setUp(self):
        self.sig = {"support": [7, 3, 2, 1], "margin": [9.0, 2.0, 1.0, 0.5]}
        self.swarm = [1.0, 0.0, 0.0, 0.0]
        self.ref = [1.0, 1.0, 1.0, 0.0]

    def test_budget_is_respected_and_ties_prefer_fewer_calls(self):
        self.assertEqual(cas.choose("margin", self.sig, self.swarm, self.ref, 0.0), -1.0)  # never call
        # one call allowed: it would not change the accuracy, so the rule without calls is preferred
        self.assertEqual(cas.choose("margin", self.sig, self.swarm, self.ref, 0.25), -1.0)
        t = cas.choose("margin", self.sig, self.swarm, self.ref, 0.5)
        self.assertEqual(t, 1.0)  # calls questions 3 and 4: one repaired
        t = cas.choose("margin", self.sig, self.swarm, self.ref, 1.0)
        self.assertEqual(t, 2.0)  # 3 calls reach the best accuracy; calling all 4 adds nothing
        self.assertEqual(cas.calls_for("margin", t, self.sig["support"], self.sig["margin"]),
                         [False, True, True, True])

    def test_support_threshold(self):
        m = cas.choose("support", self.sig, self.swarm, self.ref, 0.5)
        self.assertEqual(cas.calls_for("support", m, self.sig["support"], self.sig["margin"]),
                         [False, False, True, True])


if __name__ == "__main__":
    unittest.main()
