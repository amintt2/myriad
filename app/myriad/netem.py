"""WAN emulation for experiments: a one-way network delay sampled per message.

Off by default. When the tracker is given a WanDelay, every frame it receives from a connection and
every frame it sends to one is held back by an independent one-way delay, so a job travelling
requester -> tracker -> node -> tracker -> requester crosses four delayed hops (two round trips
between a client and the tracker). Delays are lognormal: median `median_ms` and log-scale `sigma`
(sigma = 0 gives a constant delay). Per-connection order is preserved, as on a TCP stream: a frame
is never delivered before the frame sent ahead of it on the same connection (head-of-line).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

MAX_DELAY_MS = 10_000.0


@dataclass
class WanDelay:
    median_ms: float
    sigma: float = 0.0
    seed: int | None = None
    max_ms: float = MAX_DELAY_MS  # a lognormal tail is cut here, so one sample cannot stall a stream
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.median_ms) and self.median_ms >= 0):
            raise ValueError("median_ms must be finite and >= 0")
        if not (math.isfinite(self.sigma) and self.sigma >= 0):
            raise ValueError("sigma must be finite and >= 0")
        if not (math.isfinite(self.max_ms) and self.max_ms >= 0):
            raise ValueError("max_ms must be finite and >= 0")
        self._rng = random.Random(self.seed)

    @property
    def enabled(self) -> bool:
        return self.median_ms > 0

    def sample(self) -> float:
        """One one-way delay, in seconds."""
        if self.median_ms <= 0:
            return 0.0
        ms = self.median_ms * (math.exp(self.sigma * self._rng.gauss(0.0, 1.0)) if self.sigma > 0 else 1.0)
        return min(ms, self.max_ms) / 1000.0

    def describe(self) -> dict:
        return {"median_ms": self.median_ms, "sigma": self.sigma, "max_ms": self.max_ms,
                "nominal_rtt_ms": 2 * self.median_ms}


def wan_from_rtt(rtt_ms: float, sigma: float = 0.0, seed: int | None = None) -> WanDelay | None:
    """A WanDelay whose nominal client <-> tracker round trip is rtt_ms (one-way median rtt_ms / 2);
    None (emulation off) for rtt_ms <= 0."""
    if rtt_ms is None or rtt_ms <= 0:
        return None
    return WanDelay(median_ms=rtt_ms / 2.0, sigma=sigma, seed=seed)
