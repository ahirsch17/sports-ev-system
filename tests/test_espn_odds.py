from __future__ import annotations

import json
from pathlib import Path

from sports_ev.softbooks.espn_odds import parse_american_odds, parse_espn_draftkings_moneylines

FIXTURES = Path(__file__).parent / "fixtures"


class TestEspnOdds:
    def test_parse_american_odds(self):
        assert parse_american_odds("+187") == 187
        assert parse_american_odds("-231") == -231
        assert parse_american_odds("EVEN") == 100

    def test_parse_mlb_moneylines(self):
        raw = json.loads((FIXTURES / "espn_mlb_odds.json").read_text())
        lines = parse_espn_draftkings_moneylines(raw)
        assert len(lines) == 1
        line = lines[0]
        assert line.home_team == "Atlanta Braves"
        assert line.away_team == "San Diego Padres"
        assert line.odds_home == -231
        assert line.odds_away == 187
