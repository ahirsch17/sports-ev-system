"""A kicked-off game stays on the bettable board until a final score is posted."""

import polars as pl

from nfl_edge.services.slate import upcoming_games


def test_started_game_without_a_score_stays_on_the_board():
    sched = pl.DataFrame(
        {
            "game_id": ["final", "just_kicked", "later"],
            "home_score": [27, None, None],
            "away_score": [19, None, None],
            "gameday": ["2026-10-29", "2026-11-01", "2026-11-01"],
            "gametime": ["20:15", "13:00", "16:05"],
        }
    )
    kept = upcoming_games(sched)["game_id"].to_list()
    assert kept == ["just_kicked", "later"]
