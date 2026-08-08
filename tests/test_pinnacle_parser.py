from __future__ import annotations

import pytest

from sports_ev.softbooks.pinnacle_parser import parse_pinnacle_spreads


MATCHUP = {
    "id": 1630865293,
    "type": "matchup",
    "isLive": False,
    "startTime": "2026-09-14T00:20:00Z",
    "participants": [
        {"alignment": "away", "name": "Buffalo Bills"},
        {"alignment": "home", "name": "Kansas City Chiefs"},
    ],
}

SPREAD_MARKET = {
    "matchupId": 1630865293,
    "type": "spread",
    "period": 0,
    "isAlternate": False,
    "key": "s;0;s;2.0",
    "prices": [
        {"designation": "home", "points": 2.0, "price": 116},
        {"designation": "away", "points": -2.0, "price": -130},
    ],
}


MONEYLINE_MARKET = {
    "matchupId": 1630865293,
    "type": "moneyline",
    "period": 0,
    "isAlternate": False,
    "prices": [
        {"designation": "home", "price": -120},
        {"designation": "away", "price": 100},
    ],
}


class TestPinnacleParser:
    def test_parse_spread_line(self):
        lines = parse_pinnacle_spreads([MATCHUP], [SPREAD_MARKET])
        assert len(lines) == 1
        line = lines[0]
        assert line.home_team == "Kansas City Chiefs"
        assert line.away_team == "Buffalo Bills"
        assert line.line == pytest.approx(2.0)
        assert line.odds_home == 116
        assert line.odds_away == -130
        assert line.kickoff_time is not None

    def test_skips_live_matchups(self):
        live = {**MATCHUP, "isLive": True}
        lines = parse_pinnacle_spreads([live], [SPREAD_MARKET])
        assert lines == []

    def test_skips_alternate_spreads(self):
        alt = {**SPREAD_MARKET, "isAlternate": True}
        lines = parse_pinnacle_spreads([MATCHUP], [alt])
        assert lines == []

    def test_parse_moneyline(self):
        from sports_ev.softbooks.pinnacle_parser import parse_pinnacle_moneylines

        lines = parse_pinnacle_moneylines([MATCHUP], [MONEYLINE_MARKET])
        assert len(lines) == 1
        assert lines[0].odds_home == -120
        assert lines[0].odds_away == 100
