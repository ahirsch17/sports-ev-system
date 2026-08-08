from __future__ import annotations

from sports_ev.backtest.models import BetOutcome
from sports_ev.backtest.settlement import home_covered


def home_cover_label(*, home_score: int, away_score: int, home_line: float) -> int | None:
    """
    Binary label for home spread cover.
    Returns None for pushes (excluded from training).
    """
    result = home_covered(home_score=home_score, away_score=away_score, home_line=home_line)
    if result == BetOutcome.WIN:
        return 1
    if result == BetOutcome.LOSS:
        return 0
    return None
