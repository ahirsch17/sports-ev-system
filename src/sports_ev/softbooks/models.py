from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ParsedSpreadLine:
    home_team: str
    away_team: str
    odds_home: int
    odds_away: int
    line: float
    kickoff_time: datetime | None = None
    external_event_id: str | None = None


@dataclass(frozen=True)
class ParsedMoneylineLine:
    home_team: str
    away_team: str
    odds_home: int
    odds_away: int
    kickoff_time: datetime | None = None
    external_event_id: str | None = None
