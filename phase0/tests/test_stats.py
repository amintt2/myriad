"""Paired right/wrong comparisons (essaim/stats.py): the equivalence verdict must keep its 5 % risk."""
import math
import unittest

import numpy as np

from essaim import stats


class TestPairedBinary(unittest.TestCase):
    def test_no_discordant_pair_is_not_equivalence(self):
        # Audit 2026-10-09 (phase0, 4): 82 identical answers gave a bootstrap interval [0, 0] and the
        # verdict "equivalent (+-2)". Under a true difference of +2 points (P(diff = 1) = 0.02), this
        # sample still occurs with probability 0.98^82 = 19 %: the margin hypothesis cannot be rejected.
        g = stats.compare([1.0] * 82, [1.0] * 82, 2.0)
        self.assertEqual(g["verdict"], "indéterminé")
        self.assertAlmostEqual(g["p_high_margin"], 0.98 ** 82, places=6)
        self.assertAlmostEqual(g["p_low_margin"], 0.98 ** 82, places=6)
        self.assertLess(g["lo95"], -4.0)  # Tango's interval is not degenerate
        self.assertGreater(g["hi95"], 4.0)

    def test_equivalence_reachable_and_superiority(self):
        g = stats.compare([1.0] * 300 + [0.0] * 300, [1.0] * 300 + [0.0] * 300, 2.0)  # 0.98^600: tiny
        self.assertEqual(g["verdict"], "équivalent (±2)")
        a = [1.0] * 60 + [0.0] * 40
        b = [0.0] * 30 + [1.0] * 30 + [0.0] * 40
        self.assertEqual(stats.compare(a, b, 2.0)["verdict"], "supérieur")
        self.assertEqual(stats.compare(b, a, 2.0)["verdict"], "inférieur")

    def test_score_statistic(self):
        self.assertAlmostEqual(float(stats.tango_z(7, 3, 50, 0.0)), 4 / math.sqrt(10))  # McNemar at d0 = 0
        self.assertEqual(float(stats.tango_z(0, 0, 50, 0.0)), 0.0)
        lo, hi = stats.score_ci(12, 4, 100, 0.95)
        self.assertLess(lo, 0.08)
        self.assertGreater(hi, 0.08)

    def test_size_of_the_equivalence_test(self):
        """Exact rejection probability of the TOST rule, everywhere on and outside the margins: <= 5 %."""
        n, delta = 30, 0.15
        X, Y, logc = stats._triangle(n)
        reject = np.array([stats.p_upper(int(y), int(x), n, delta) < 0.05 and stats.p_upper(int(x), int(y), n, delta) < 0.05
                           for x, y in zip(X, Y)])
        self.assertTrue(reject.any())  # the test has some power at this size
        worst = 0.0
        for d in (delta, delta + 0.05, -delta, -delta - 0.1):
            for q in np.linspace(0.0, (1 - abs(d)) / 2, 25):
                p10, p01 = (q + d, q) if d > 0 else (q, q - d)
                rest = max(0.0, 1.0 - p10 - p01)
                lp = logc + stats._xlogp(X, p10) + stats._xlogp(Y, p01) + stats._xlogp(n - X - Y, rest)
                worst = max(worst, float(np.exp(lp)[reject].sum()))
        self.assertLessEqual(worst, 0.05 + 1e-9)


if __name__ == "__main__":
    unittest.main()
