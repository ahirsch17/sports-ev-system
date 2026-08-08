from __future__ import annotations

from sports_ev.pricing.odds import american_to_decimal, profit_if_win


def kelly_fraction(
    true_prob: float,
    american_odds: int,
    *,
    fraction: float = 0.25,
    max_stake_pct: float = 2.0,
) -> float:
    """
    Fractional Kelly stake as a percentage of bankroll.

    Full Kelly: f* = (bp - q) / b where b is net decimal odds.
    Returns capped fractional Kelly in percent (e.g. 1.5 means 1.5% of bankroll).
    """
    if not 0.0 < true_prob < 1.0:
        return 0.0
    if fraction <= 0 or max_stake_pct <= 0:
        return 0.0

    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0

    q = 1.0 - true_prob
    full_kelly = (b * true_prob - q) / b
    if full_kelly <= 0:
        return 0.0

    stake_pct = full_kelly * fraction * 100.0
    return min(stake_pct, max_stake_pct)


def expected_profit_pct(stake_pct: float, american_odds: int, won: bool, pushed: bool = False) -> float:
    """Bankroll return in percentage points for a settled paper bet."""
    if pushed or stake_pct <= 0:
        return 0.0
    if won:
        return stake_pct * profit_if_win(1.0, american_odds)
    return -stake_pct
