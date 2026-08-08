from __future__ import annotations

from datetime import datetime, timezone

from sports_ev.game_clock import (
    LiveGameStatus,
    format_countdown,
    format_game_clock,
    parse_espn_scoreboard_events,
)


class TestFormatGameClock:
    def test_countdown_minutes(self):
        now = datetime(2026, 7, 23, 18, 0, tzinfo=timezone.utc)
        kickoff = datetime(2026, 7, 23, 19, 30, tzinfo=timezone.utc)
        assert format_countdown(kickoff=kickoff, now=now) == "Starts in 1h 30m"

    def test_live_from_espn(self):
        now = datetime(2026, 7, 23, 20, 0, tzinfo=timezone.utc)
        kickoff = datetime(2026, 7, 23, 19, 0, tzinfo=timezone.utc)
        live = LiveGameStatus(
            home_team="Padres",
            away_team="Dodgers",
            state="in",
            detail="Top 7th",
            home_score=3,
            away_score=2,
        )
        assert format_game_clock(kickoff=kickoff, now=now, live=live) == "2-3 · Top 7th"

    def test_final_from_espn(self):
        now = datetime(2026, 7, 23, 23, 0, tzinfo=timezone.utc)
        kickoff = datetime(2026, 7, 23, 19, 0, tzinfo=timezone.utc)
        live = LiveGameStatus(
            home_team="Padres",
            away_team="Dodgers",
            state="post",
            detail="Final",
            home_score=5,
            away_score=4,
        )
        assert format_game_clock(kickoff=kickoff, now=now, live=live) == "4-5 Final"


class TestParseEspnScoreboardEvents:
    def test_parses_in_progress_detail(self):
        payload = {
            "events": [
                {
                    "competitions": [
                        {
                            "status": {
                                "type": {
                                    "state": "in",
                                    "shortDetail": "Bot 5th",
                                }
                            },
                            "competitors": [
                                {"homeAway": "home", "team": {"displayName": "Yankees"}, "score": "4"},
                                {"homeAway": "away", "team": {"displayName": "Red Sox"}, "score": "3"},
                            ],
                        }
                    ]
                }
            ]
        }
        events = parse_espn_scoreboard_events(payload)
        assert len(events) == 1
        assert events[0].state == "in"
        assert events[0].detail == "Bot 5th"
        assert events[0].home_score == 4
        assert events[0].away_score == 3
