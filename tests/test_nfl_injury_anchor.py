"""Quarterback identity, injury margin, and market-anchored forecasts."""

import polars as pl

from nfl_edge.features.injuries import recent_starting_qbs, team_injury_impact
from nfl_edge.features.matchups import build_matchup_frame
from nfl_edge.models.game import anchor_forecast


INJURY_WEIGHTS = {
    "injuries": {
        "qb1_out_points": -5.0,
        "qb2_out_points": -1.5,
        "wr1_out_points": -1.2,
        "rb1_out_points": -0.8,
        "ol_starter_out_points": -0.6,
        "cb1_out_vs_pass": 1.0,
        "inactive_depth_gt_2": 0.0,
    }
}


def _qb_row(name: str, gid: str, status: str, practice: str) -> dict:
    return {
        "season": 2026,
        "week": 6,
        "team": "WAS",
        "gsis_id": gid,
        "full_name": name,
        "position": "QB",
        "report_status": status,
        "practice_status": practice,
        "report_primary_injury": "Knee",
        "practice_primary_injury": "Knee",
        "report_secondary_injury": None,
    }


def test_a_kick_return_listing_does_not_change_the_back_penalty():
    """Velus Jones is a returner and the fifth back. That must not price him as a starter, and it must not change between calls."""
    injuries = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 8,
                "team": "SEA",
                "gsis_id": "VJ",
                "full_name": "Velus Jones Jr.",
                "position": "RB",
                "report_status": "Out",
                "practice_status": None,
                "report_primary_injury": "Hamstring",
                "practice_primary_injury": None,
                "report_secondary_injury": None,
            }
        ]
    )
    depth = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [8, 8],
            "team": ["SEA", "SEA"],
            "gsis_id": ["VJ", "VJ"],
            "full_name": ["Velus Jones Jr.", "Velus Jones Jr."],
            "position": ["KR", "RB"],
            "depth_team": [2, 5],
        }
    )
    flipped = depth.sort("position", descending=True)
    points = [
        team_injury_impact(
            season=2026, week=8, team="SEA", injuries=injuries, depth_charts=chart, weights=INJURY_WEIGHTS
        )["points"]
        for chart in (depth, flipped, depth)
    ]
    assert points == [0.0, 0.0, 0.0]


def test_only_the_current_starter_is_qb1():
    stats = pl.DataFrame(
        {
            "season": [2024, 2026, 2026],
            "week": [8, 5, 5],
            "recent_team": ["WAS", "WAS", "WAS"],
            "player_display_name": ["Marcus Mariota", "Jayden Daniels", "Marcus Mariota"],
            "player_id": ["MAR", "DAN", "MAR"],
            "position": ["QB", "QB", "QB"],
            "attempts": [30, 28, 4],
        }
    )
    starters = recent_starting_qbs(stats, season=2026, week=6)
    was = starters["WAS"]
    assert was["name"] == "Jayden Daniels"
    assert was["known_starter_ids"] == {"DAN"}
    assert "marcus mariota" not in was["known_starter_names"]


def test_backup_out_does_not_price_the_starter_out():
    injuries = pl.DataFrame([_qb_row("Marcus Mariota", "MAR", "Out", "Did Not Participate In Practice")])
    impact = team_injury_impact(
        season=2026,
        week=6,
        team="WAS",
        injuries=injuries,
        depth_charts=pl.DataFrame(),
        weights=INJURY_WEIGHTS,
        recent_qb={
            "gsis_id": "DAN",
            "name": "Jayden Daniels",
            "known_starter_ids": {"DAN"},
            "known_starter_names": {"jayden daniels": "Jayden Daniels", "DAN": "Jayden Daniels"},
        },
    )
    assert impact["qb_out"] == 0.0
    assert "Mariota" not in str(impact["qb_note"])
    assert impact["points"] == -1.5


def test_current_starter_out_is_full_qb1_and_not_stacked():
    injuries = pl.DataFrame(
        [
            _qb_row("Jayden Daniels", "DAN", "Out", "Did Not Participate In Practice"),
            _qb_row("Marcus Mariota", "MAR", "Out", "Did Not Participate In Practice"),
        ]
    )
    impact = team_injury_impact(
        season=2026,
        week=6,
        team="WAS",
        injuries=injuries,
        depth_charts=pl.DataFrame(),
        weights=INJURY_WEIGHTS,
        recent_qb={
            "gsis_id": "DAN",
            "name": "Jayden Daniels",
            "known_starter_ids": {"DAN"},
            "known_starter_names": {"jayden daniels": "Jayden Daniels"},
        },
    )
    assert impact["qb_out"] == 1.0
    assert "Jayden Daniels" in str(impact["qb_note"])
    # one QB1 (-5) plus one backup (-1.5), not two starters
    assert impact["points"] == -6.5


def test_injury_margin_is_not_diluted_by_elo():
    schedules = pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2026, 2026, 2026],
            "week": [1, 2, 3],
            "gameday": ["2026-09-07", "2026-09-14", "2026-09-21"],
            "home_team": ["GB", "GB", "GB"],
            "away_team": ["CHI", "CHI", "CHI"],
            "home_score": [24, 21, None],
            "away_score": [20, 17, None],
            "roof": ["outdoors", "outdoors", "outdoors"],
            "temp": [70, 70, 70],
            "wind": [5, 5, 5],
        }
    )
    weights = {
        "elo": {"blend": 0.62, "k": 20, "hfa_elo": 48},
        "hfa_points": {"pre_2016": 2.0, "2016_2019": 2.0, "2020": 0.2, "2021_plus": 2.0},
        "weather": {
            "wind_pass_yards_pct_per_10mph": 0.0,
            "wind_total_points_per_10mph": 0.0,
            "wind_high_threshold_mph": 20,
            "wind_high_total_extra": 0.0,
            "rain_pass_yards_pct": 0.0,
            "rain_total_points": 0.0,
            "dome_total_boost": 0.0,
        },
        "momentum": {
            "form_weight_early": 0.0,
            "form_weight_mid": 0.0,
            "form_weight_late": 0.0,
            "early_weeks": 4,
            "late_week_start": 15,
        },
        "injuries": INJURY_WEIGHTS["injuries"],
        "deep_matchup": {"weight": 0.0},
        "situational": {"weight": 0.0},
    }
    base = build_matchup_frame(
        schedules, weights=weights, use_injuries=False, use_deep=False, use_situational=False
    )
    injured = build_matchup_frame(
        schedules,
        injury_features=pl.DataFrame(
            {"game_id": ["g3"], "home_inj_points": [-6.0], "away_inj_points": [0.0]}
        ),
        weights=weights,
        use_deep=False,
        use_situational=False,
    )
    base_m = float(base.filter(pl.col("game_id") == "g3")["pred_margin"][0])
    inj_m = float(injured.filter(pl.col("game_id") == "g3")["pred_margin"][0])
    # Full 6-point hit, not the 0.38 * 6 left after the Elo blend.
    assert abs((base_m - inj_m) - 6.0) < 0.15


def test_published_forecast_tilts_off_the_line_and_keeps_raw_margin():
    anchored = anchor_forecast(
        pl.DataFrame(
            {
                "pred_margin": [10.0],
                "pred_total": [60.0],
                "pred_home_points": [35.0],
                "pred_away_points": [25.0],
                "spread_line": [0.0],
                "total_line": [44.0],
            }
        ),
        {"market_blend": {"model_weight": 0.1}},
    )
    row = anchored.row(0, named=True)
    assert row["model_margin"] == 10.0
    assert row["model_total"] == 60.0
    assert row["pred_margin"] == 1.0
    assert row["pred_total"] == 45.6
    assert row["pred_home_points"] == 23.3
    assert row["pred_away_points"] == 22.3
