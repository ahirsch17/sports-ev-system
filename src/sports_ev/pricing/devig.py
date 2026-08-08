from __future__ import annotations

import math
from typing import Sequence

from sports_ev.pricing.devig_method import DevigMethod
from sports_ev.pricing.odds import american_to_implied_prob


def implied_probs_from_american(odds_home: int, odds_away: int) -> tuple[float, float]:
    return american_to_implied_prob(odds_home), american_to_implied_prob(odds_away)


def overround(probs: Sequence[float]) -> float:
    return sum(probs)


def multiplicative_devig(probs: Sequence[float]) -> list[float]:
    """
    Proportional / multiplicative no-vig normalization.
    Each fair prob = raw_implied / sum(raw_implied).
    """
    total = sum(probs)
    if total <= 0:
        raise ValueError("Sum of implied probabilities must be positive")
    return [p / total for p in probs]


def shin_devig(probs: Sequence[float], *, max_iterations: int = 200) -> list[float]:
    """
    Shin (1991/1992) method for removing bookmaker margin.
    Solves for insider parameter z via bisection so fair probs sum to 1.
    """
    implied = [float(p) for p in probs]
    n = len(implied)
    if n < 2:
        raise ValueError("Shin devig requires at least two outcomes")

    total = sum(implied)
    if total <= 0:
        raise ValueError("Sum of implied probabilities must be positive")

    def fair_sum(z: float) -> float:
        if z >= 1.0:
            return total
        return sum(
            (math.sqrt(z * z + 4.0 * (1.0 - z) * (p * p) / total) - z) / (2.0 * (1.0 - z))
            for p in implied
        )

    lo, hi = 0.0, 0.999999
    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        if fair_sum(mid) > 1.0:
            lo = mid
        else:
            hi = mid

    z = (lo + hi) / 2.0
    if z <= 0.0:
        return multiplicative_devig(implied)

    return [
        (math.sqrt(z * z + 4.0 * (1.0 - z) * (p * p) / total) - z) / (2.0 * (1.0 - z))
        for p in implied
    ]


def devig_two_way(
    odds_home: int,
    odds_away: int,
    *,
    method: DevigMethod = DevigMethod.MULTIPLICATIVE,
) -> tuple[float, float]:
    """Return fair win probabilities for home and away in a two-way market."""
    raw_home, raw_away = implied_probs_from_american(odds_home, odds_away)
    if method == DevigMethod.MULTIPLICATIVE:
        fair_home, fair_away = multiplicative_devig([raw_home, raw_away])
    elif method == DevigMethod.SHIN:
        fair_home, fair_away = shin_devig([raw_home, raw_away])
    else:
        raise ValueError(f"Unknown devig method: {method}")
    return fair_home, fair_away
