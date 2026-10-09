"""Split-conformal thresholds for a missed-PII guarantee, and exact binomial intervals (pure Python).

Setting. Each calibration document i that contains personal data gets a *critical score* c_i: the
largest threshold at which the detector still catches it (gate: the document's maximum token score;
coverage: the minimum, over its gold spans, of the best score of a token overlapping the span). With a
threshold lam the document is missed iff c_i < lam, a loss non-decreasing in lam.

marginal_threshold (conformal risk control with a 0/1 loss, Angelopoulos et al. 2022, equivalently
the split-conformal quantile): with k = floor(alpha (n + 1)) and lam = c_(k), the k-th smallest of
the n calibration scores, P(c_new < lam) <= k / (n + 1) <= alpha when the n + 1 scores are
exchangeable (ties only make the strict event rarer). If k = 0 no threshold is safe: return 0.

pac_threshold (training-conditional, "with probability >= 1 - delta over the calibration draw, the
miss rate of the chosen threshold is <= alpha"): the miss rate of c_(k) is F(c_(k)-) <= U_(k) with
U_(k) ~ Beta(k, n - k + 1), so P(miss rate > alpha) <= P(Beta(k, n-k+1) > alpha) = P(Bin(n, alpha) < k);
take the largest k with that probability <= delta."""
from __future__ import annotations

import math


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k), summed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq
             for j in range(k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def marginal_k(n: int, alpha: float) -> int:
    return math.floor(alpha * (n + 1))


def pac_k(n: int, alpha: float, delta: float) -> int:
    k = 0
    for j in range(1, n + 1):
        if binom_cdf(j - 1, n, alpha) <= delta:  # P(Bin(n, alpha) < j) <= delta
            k = j
        else:
            break
    return k


def threshold_from_k(scores: list[float], k: int) -> float:
    """c_(k) (1-based) or 0.0 when k = 0: every document must then be treated as sensitive."""
    if k <= 0 or not scores:
        return 0.0
    return sorted(scores)[k - 1]


def marginal_threshold(scores: list[float], alpha: float) -> float:
    return threshold_from_k(scores, marginal_k(len(scores), alpha))


def pac_threshold(scores: list[float], alpha: float, delta: float = 0.05) -> float:
    return threshold_from_k(scores, pac_k(len(scores), alpha, delta))


def clopper_pearson(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a binomial proportion x / n."""
    if n == 0:
        return 0.0, 1.0
    a = (1 - conf) / 2

    def solve(f, lo=0.0, hi=1.0):
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    # lower: P(Bin(n, p) >= x) = a ; upper: P(Bin(n, p) <= x) = a
    lo = 0.0 if x == 0 else solve(lambda p: 1 - binom_cdf(x - 1, n, p) >= a)
    hi = 1.0 if x == n else solve(lambda p: binom_cdf(x, n, p) <= a)
    return lo, hi


def min_n(alpha: float, delta: float | None = None) -> int:
    """Smallest number of positive calibration documents for a non-trivial threshold."""
    n = 1
    while (pac_k(n, alpha, delta) if delta else marginal_k(n, alpha)) == 0:
        n += 1
    return n
