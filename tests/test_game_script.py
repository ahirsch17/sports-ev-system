"""Offense yards move with the defense they face, and missing catchers become runs."""

from __future__ import annotations

import polars as pl

from nfl_edge.features.injuries import team_injury_impact
from nfl_edge.features.ratings import attach_pass_rush_splits
from nfl_edge.features.script import script_note, scripted_pass_yards, scripted_rush_yards, te_volume_multiplier
from nfl_edge.models.props import reconcile_receiving


def test_stingy_pass_defense_cuts_the_offense_below_its_own_yards():
    own = 240.0
    normal, _, normal_index = scripted_pass_yards(own, 215.0)
    stingy, _, stingy_index = scripted_pass_yards(own, 145.0)
    leaky, _, _ = scripted_pass_yards(own, 280.0)
    assert normal_index == 1.0
    assert stingy_index < 0.90
    assert stingy < normal < leaky
    assert stingy < own * 0.85


def test_missing_catchers_against_a_stingy_pass_defense_lean_run():
    off_pass = 240.0
    off_rush = 120.0
    allowed_pass = 145.0
    open_pass, open_cut, index = scripted_pass_yards(off_pass, allowed_pass, catchers_out=0)
    scripted, cut, _ = scripted_pass_yards(off_pass, allowed_pass, catchers_out=1.7)
    before = scripted / (1.0 - cut)
    rush = scripted_rush_yards(off_rush, 120.0, pass_before_cut=before, cut=cut)
    base_rush = scripted_rush_yards(off_rush, 120.0, pass_before_cut=before, cut=0)
    assert index < 0.90
    assert cut > open_cut
    assert scripted < open_pass
    assert scripted < 170
    assert rush > base_rush + 15
    note = script_note("BUF", 2, index, te_mult=0.7)
    assert "leans run" in note
    assert "tight ends" in note


def test_tight_end_shrinks_when_the_defense_takes_that_position_away():
    # Rams shape: pass defense already stingy, TE yards allowed much stingier.
    mult = te_volume_multiplier(21.0, pass_index=0.78, league_te=48.0)
    assert mult < 0.85
    assert te_volume_multiplier(48.0, pass_index=0.78, league_te=48.0) == 1.0
    assert te_volume_multiplier(50.0, pass_index=1.22, league_te=52.0) == 1.0


def test_an_absent_receivers_targets_are_not_given_to_teammates():
    rows = [
        {"player_name": "Coleman", "position": "WR", "roll_recv_share": 0.22, "roll_recv_ypt": 8.0, "roll_usage_collapsed": False, "roll_season_ypg": 50},
        {"player_name": "Kincaid", "position": "TE", "roll_recv_share": 0.06, "roll_recv_ypt": 7.0, "roll_usage_collapsed": True, "roll_season_ypg": 67},
    ]
    team_pass = 180.0
    out = reconcile_receiving(rows, team_pass)
    attempts = team_pass / 7.1
    coleman = next(row for row in out if row["player_name"] == "Coleman")
    # 0.22 stays 0.22. It is not scaled up to fill the share of the receivers who are out.
    assert abs(coleman["mu_recv"] - attempts * 0.22 * 8.0) < 1.0


def test_two_starting_receivers_out_are_counted():
    injuries = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [5, 5],
            "team": ["BUF", "BUF"],
            "gsis_id": ["m1", "p1"],
            "full_name": ["DJ Moore", "Joshua Palmer"],
            "position": ["WR", "WR"],
            "report_status": ["Out", "Out"],
        }
    )
    depth = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [5, 5],
            "team": ["BUF", "BUF"],
            "gsis_id": ["m1", "p1"],
            "full_name": ["DJ Moore", "Joshua Palmer"],
            "position": ["WR", "WR"],
            "depth_team": [1, 2],
        }
    )
    impact = team_injury_impact(
        season=2026, week=5, team="BUF", injuries=injuries, depth_charts=depth, weights=None
    )
    assert impact["catchers_n"] == 2
    assert float(impact["catchers_out"]) == 1.7


def test_yards_allowed_come_from_the_opponent_not_the_offenses_own_total():
    games = pl.DataFrame(
        {
            "season": [2026] * 6,
            "week": [1, 1, 2, 2, 3, 3],
            "team": ["BUF", "LA", "BUF", "LA", "BUF", "LA"],
            "opp": ["LA", "BUF", "LA", "BUF", "LA", "BUF"],
        }
    )
    stats = pl.DataFrame(
        {
            "season": [2026] * 6,
            "week": [1, 1, 2, 2, 3, 3],
            "team": ["BUF", "LA", "BUF", "LA", "BUF", "LA"],
            "passing_yards": [250.0, 140.0, 230.0, 150.0, 200.0, 160.0],
            "rushing_yards": [100.0, 90.0, 110.0, 80.0, 120.0, 70.0],
        }
    )
    joined = attach_pass_rush_splits(games, stats)
    buf_week3 = joined.filter((pl.col("team") == "BUF") & (pl.col("week") == 3)).to_dicts()[0]
    assert abs(float(buf_week3["roll_pass_allowed"]) - 145.0) < 0.1
