from __future__ import annotations

from sports_ev.backtest.models import BetOutcome


def home_win_label(*, home_score: int, away_score: int) -> int | None:
    """Binary label for home moneyline win. None for ties (rare)."""
    if home_score > away_score:
        return 1
    if home_score < away_score:
        return 0
    return None
