"""Odds conversion and EV helpers."""

from __future__ import annotations

import math
from typing import Iterable


def american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability (includes vig if raw)."""
    if odds == 0:
        raise ValueError("odds cannot be 0")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def implied_to_american(prob: float) -> int:
    """Convert a fair probability to American odds."""
    p = min(max(prob, 1e-6), 1 - 1e-6)
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    if total <= 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def expected_value(model_prob: float, american_odds: float) -> float:
    """EV per $1 stake at given American odds."""
    if american_odds > 0:
        profit = american_odds / 100.0
    else:
        profit = 100.0 / abs(american_odds)
    return model_prob * profit - (1 - model_prob)


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def prob_over_line(mu: float, sigma: float, line: float) -> float:
    """P(X > line) for X ~ N(mu, sigma^2). Push ignored (continuous)."""
    return 1.0 - normal_cdf(line, mu, sigma)


def margin_to_home_win_prob(mu_margin: float, sigma: float = 13.5) -> float:
    """P(home wins) from predicted home margin (home - away)."""
    return 1.0 - normal_cdf(0.0, mu_margin, sigma)


def combine_independent(probs: Iterable[float]) -> float:
    p = 1.0
    for x in probs:
        p *= x
    return p
