from __future__ import annotations

import math


def american_to_decimal(american: int) -> float:
    if american == 0:
        raise ValueError("American odds cannot be zero")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> int:
    if decimal <= 1.0:
        raise ValueError(f"Invalid decimal odds: {decimal}")
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100))
    return int(round(-100.0 / (decimal - 1.0)))


def american_to_implied_prob(american: int) -> float:
    """Raw implied probability including vig."""
    if american == 0:
        raise ValueError("American odds cannot be zero")
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def decimal_to_implied_prob(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError(f"Invalid decimal odds: {decimal}")
    return 1.0 / decimal


def profit_if_win(stake: float, american: int) -> float:
    """Profit (not including returned stake) on a winning bet."""
    if stake <= 0:
        raise ValueError("Stake must be positive")
    if american > 0:
        return stake * american / 100.0
    return stake * 100.0 / abs(american)


def payout_multiplier(american: int) -> float:
    """Total return per $1 staked, including the original stake."""
    return american_to_decimal(american)
