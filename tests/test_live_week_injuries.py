"""An unpublished week takes the live Out/Doubtful list. A published week does not."""

import polars as pl

from nfl_edge.features.injuries import build_injury_features
from nfl_edge.ingest.espn_injuries import (
    bind_live_rows_to_depth,
    rows_from_payload,
    supplement_unpublished_week,
)
from nfl_edge.models.props import _ruled_out


def _payload() -> dict:
    def athlete(name: str, team: str, pos: str, eid: str) -> dict:
        return {
            "id": eid,
            "displayName": name,
            "position": {"abbreviation": pos},
            "team": {"abbreviation": team},
        }

    return {
        "injuries": [
            {
                "displayName": "Baltimore Ravens",
                "injuries": [
                    {"status": "Out", "athlete": athlete("Lamar Jackson", "BAL", "QB", "1")},
                    {"status": "Doubtful", "athlete": athlete("Tyler Huntley", "BAL", "QB", "2")},
                    {"status": "Questionable", "athlete": athlete("Joe Fagnano", "BAL", "QB", "3")},
                ],
            }
        ]
    }


def _schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2026_09_JAX_BAL"],
            "season": [2026],
            "week": [9],
            "home_team": ["BAL"],
            "away_team": ["JAX"],
        }
    )


def _passer(name: str, attempts: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2026],
            "week": [8],
            "team": ["BAL"],
            "player_display_name": [name],
            "position": ["QB"],
            "attempts": [attempts],
            "passing_yards": [200],
        }
    )


def test_questionable_and_a_missing_team_stay_off_the_live_rows():
    rows = rows_from_payload(_payload(), season=2026, week=9)
    names = {row["full_name"] for row in rows}
    assert names == {"Lamar Jackson", "Tyler Huntley"}
    assert all(row["team"] == "BAL" and row["gsis_id"].startswith("espn-") for row in rows)


def test_a_published_week_is_not_rewritten():
    filed = pl.DataFrame(
        {
            "season": [2026],
            "week": [9],
            "team": ["BAL"],
            "gsis_id": ["00-1"],
            "full_name": ["Someone Else"],
            "position": ["QB"],
            "report_status": ["Questionable"],
        }
    )
    kept = supplement_unpublished_week(filed, 2026, 9, payload=_payload())
    assert kept["full_name"].to_list() == ["Someone Else"]


def test_the_current_starter_being_out_docks_the_score():
    injuries = supplement_unpublished_week(pl.DataFrame(), 2026, 9, payload=_payload())
    feat = build_injury_features(_schedule(), injuries, pl.DataFrame(), player_stats=_passer("Lamar Jackson", 30))
    row = feat.row(0, named=True)
    assert row["home_qb_out"] == 1.0
    assert row["home_inj_points"] == -6.125
    assert row["home_inj_pass_pct"] == -0.25
    assert "Lamar Jackson OUT (backup starting)" in row["home_inj_note"]


def test_a_former_starter_does_not_take_the_starter_dock():
    injuries = supplement_unpublished_week(pl.DataFrame(), 2026, 9, payload=_payload())
    feat = build_injury_features(_schedule(), injuries, pl.DataFrame(), player_stats=_passer("Joe Fagnano", 21))
    row = feat.row(0, named=True)
    assert row["home_qb_out"] == 0.0
    assert row["home_inj_points"] == -2.625
    assert "backup starting" not in row["home_inj_note"]


def test_a_depth_chart_starter_keeps_his_rank():
    payload = {
        "injuries": [
            {
                "injuries": [
                    {
                        "status": "Out",
                        "athlete": {
                            "id": "9",
                            "displayName": "Zay Flowers",
                            "position": {"abbreviation": "WR"},
                            "team": {"abbreviation": "BAL"},
                        },
                    }
                ]
            }
        ]
    }
    depth = pl.DataFrame(
        {
            "season": [2026],
            "week": [9],
            "team": ["BAL"],
            "gsis_id": ["00-zay"],
            "full_name": ["Zay Flowers"],
            "position": ["WR"],
            "depth_team": [1],
        }
    )
    injuries = bind_live_rows_to_depth(
        supplement_unpublished_week(pl.DataFrame(), 2026, 9, payload=payload),
        depth,
    )
    assert injuries["gsis_id"][0] == "00-zay"
    feat = build_injury_features(_schedule(), injuries, depth)
    row = feat.row(0, named=True)
    assert row["home_inj_points"] == -1.2
    assert row["home_inj_pass_pct"] == -0.05


def test_the_same_rows_remove_those_players_from_the_props():
    injuries = supplement_unpublished_week(pl.DataFrame(), 2026, 9, payload=_payload())
    assert _ruled_out(injuries, 2026, 9) == {"lamar jackson", "tyler huntley"}


def test_the_game_total_leg_is_labeled_in_points():
    script = open("ui/static/app.js", encoding="utf-8").read()
    assert 'leg.market === "total"' in script
    assert "Game total" in script
    assert "points" in script
