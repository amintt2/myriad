"""Paired comparison of two systems graded right/wrong on the same questions (E4, E11).

A difference of accuracies d = p(A right) - p(B right) only depends on the discordant pairs: x questions
where A is right and B wrong, y where B is right and A wrong, out of n. The bootstrap percentile interval
used before collapses when discordant pairs are rare: with x = y = 0 every resample gives exactly 0, the
interval is [0, 0], and "equivalent within +-2 points" was declared although, with a true difference of
+2 points, such a sample still occurs with probability 0.98^n (19 % for n = 82).

Here every decision is an exact unconditional test (Barnard-style, maximised over the nuisance parameter):
  * statistic: Tango's score statistic for the paired difference (Tango 1998, Stat. Med. 17:891-908,
    "Equivalence test and confidence interval for the difference in proportions for the paired-sample
    design"), Z(d0) = (x - y - n d0) / sqrt(n (2 q + d0 - d0^2)), q the restricted maximum-likelihood
    estimate of P(B right, A wrong) under d = d0. It is defined with no discordant pair at all (it reduces
    to McNemar's statistic when d0 = 0);
  * p-value: the probability, under the trinomial law of (x, y), of a statistic at least as extreme as the
    observed one, MAXIMISED over the nuisance parameter q on the boundary d = d0 of the null hypothesis
    (evaluated on a dense grid, then refined). Its size is at most alpha whatever q, which an asymptotic
    score test only approximates when discordant pairs are rare (exactly the case that broke the bootstrap).
Why this choice: an exact conditional test on the discordant pairs (binomial, McNemar) is exact only for
d0 = 0, since under d = d0 != 0 the conditional law still depends on the number of discordant pairs; the
unconditional test handles the +-margin hypotheses of an equivalence test exactly. Two one-sided tests at
5 % each (TOST) give the equivalence decision.
The displayed intervals are Tango's score intervals (inverting Z), which are never degenerate (x = y = 0,
n = 82 gives about +-4.5 points at 95 %); the verdicts come from the exact p-values, which are reported.
"""
from __future__ import annotations

import math

import numpy as np

ALPHA = 0.05
_GRID = 101
_TRI: dict[int, tuple] = {}


def tango_z(x, y, n: int, d0: float):
    """Tango's score statistic for H0: p10 - p01 = d0 (x = n10, y = n01; arrays or scalars)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = 2.0 * n
    b = d0 * (2.0 * n - x + y) - (x + y)
    c = -y * d0 * (1.0 - d0)
    q = (-b + np.sqrt(np.maximum(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)
    var = n * (2.0 * q + d0 - d0 * d0)
    num = x - y - n * d0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 0, num / np.sqrt(np.where(var > 0, var, 1.0)), np.sign(num) * np.inf)
    return np.where((var <= 0) & (num == 0), 0.0, z)


def _triangle(n: int):
    """Every (x, y) with x + y <= n, and the log multinomial coefficients (cached per n)."""
    if n not in _TRI:
        lf = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])
        xs, ys = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = xs + ys <= n
        x, y = xs[keep], ys[keep]
        _TRI[n] = (x, y, lf[n] - lf[x] - lf[y] - lf[n - x - y])
    return _TRI[n]


def _xlogp(k, p: float):
    """k log p with 0 log 0 = 0 (and k log 0 = -inf for k > 0)."""
    if p > 0:
        return k * math.log(p)
    return np.where(k > 0, -np.inf, 0.0)


def p_upper(x: int, y: int, n: int, d0: float) -> float:
    """Exact unconditional p-value of H0: p10 - p01 >= d0 against p10 - p01 < d0 (0 <= d0 < 1)."""
    if n == 0:
        return 1.0
    X, Y, logc = _triangle(n)
    z_obs = float(tango_z(x, y, n, d0))
    extreme = tango_z(X, Y, n, d0) <= z_obs + 1e-9  # at least as extreme as the observation (ties included)
    Xe, Ye, Ce, Me = X[extreme], Y[extreme], logc[extreme], (n - X - Y)[extreme]
    qmax = (1.0 - d0) / 2.0

    def prob(q: float) -> float:
        p10, p01 = q + d0, q
        rest = max(0.0, 1.0 - p10 - p01)
        lp = Ce + _xlogp(Xe, p10) + _xlogp(Ye, p01) + _xlogp(Me, rest)
        return float(np.exp(lp).sum())

    grid = np.linspace(0.0, qmax, _GRID)
    vals = [prob(q) for q in grid]
    k = int(np.argmax(vals))
    best = vals[k]
    lo, hi = grid[max(0, k - 1)], grid[min(_GRID - 1, k + 1)]
    for _ in range(40):  # golden-section refinement around the best grid point (p(q) is smooth)
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        v1, v2 = prob(m1), prob(m2)
        best = max(best, v1, v2)
        if v1 >= v2:
            hi = m2
        else:
            lo = m1
    return min(1.0, best)


def score_ci(x: int, y: int, n: int, level: float) -> tuple[float, float]:
    """Tango's score interval for p10 - p01 (inverts Z = +-z_{(1+level)/2}; bisection, Z decreases in d0)."""
    zc = _norm_ppf(0.5 + level / 2.0)

    def root(target: float) -> float:
        lo, hi = -1.0 + 1e-12, 1.0 - 1e-12
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if float(tango_z(x, y, n, mid)) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    return root(zc), root(-zc)


def _norm_ppf(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compare(a: list[float], b: list[float], delta_points: float) -> dict:
    """Paired comparison of two right/wrong vectors (1.0 = right). Differences in points (A minus B)."""
    if len(a) != len(b):
        raise ValueError("vecteurs de longueurs différentes")
    n = len(a)
    x = sum(1 for u, v in zip(a, b) if u and not v)
    y = sum(1 for u, v in zip(a, b) if v and not u)
    delta = delta_points / 100.0
    lo95, hi95 = score_ci(x, y, n, 0.95)
    lo90, hi90 = score_ci(x, y, n, 0.90)
    g = {"n": n, "n10": x, "n01": y, "mean": 100.0 * (x - y) / n if n else 0.0,
         "lo95": 100 * lo95, "hi95": 100 * hi95, "lo90": 100 * lo90, "hi90": 100 * hi90,
         "ci": "Tango score",
         # exact unconditional one-sided p-values
         "p_superior": p_upper(y, x, n, 0.0),        # H0: d <= 0
         "p_inferior": p_upper(x, y, n, 0.0),        # H0: d >= 0
         "p_low_margin": p_upper(y, x, n, delta),    # H0: d <= -delta
         "p_high_margin": p_upper(x, y, n, delta),   # H0: d >= +delta
         "delta": delta_points, "test": "exact unconditional (Tango score ordering), TOST at 5 % per side"}
    g["verdict"] = verdict(g)
    return g


def verdict(g: dict) -> str:
    """superior / inferior: one-sided exact test at 2.5 % (a two-sided 5 % decision); equivalent: both
    one-sided margin tests at 5 % (TOST); non-inferior: the lower margin test alone at 5 %."""
    d = g["delta"]
    if g["p_superior"] < ALPHA / 2:
        return "supérieur"
    if g["p_low_margin"] < ALPHA and g["p_high_margin"] < ALPHA:
        return f"équivalent (±{d:g})"
    if g["p_inferior"] < ALPHA / 2:
        return "inférieur"
    if g["p_low_margin"] < ALPHA:
        return f"non inférieur (-{d:g})"
    return "indéterminé"
