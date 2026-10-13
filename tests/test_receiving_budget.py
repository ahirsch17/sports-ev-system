"""Receiving props share one pass budget and do not name two playmakers."""

from __future__ import annotations

import polars as pl

from nfl_edge.edge.advisor import _prop_plays
from nfl_edge.models.props import project_props_for_game, rolling_player_yards


def _game() -> dict:
    return {
        "season": 2026,
        "week": 6,
        "home_team": "LV",
        "away_team": "BUF",
        "game_id": "buf-lv",
        "pred_home_pass_yds": 220.0,
        "pred_away_pass_yds": 210.0,
        "pred_home_rush_yds": 90.0,
        "pred_away_rush_yds": 140.0,
    }


def _roll(**kwargs) -> dict:
    base = {
        "season": 2026,
        "week": 6,
        "team": "BUF",
        "roll_pass": None,
        "roll_rush": 4.0,
        "roll_recv": 40.0,
        "roll_pass_std": None,
        "roll_rush_std": None,
        "roll_recv_std": 18.0,
        "roll_recv_share": 0.1,
        "roll_recv_ypt": 8.0,
        "roll_usage_collapsed": False,
        "roll_season_ypg": 40.0,
    }
    base.update(kwargs)
    return base


def test_one_playmaker_and_a_collapsed_under():
    rolls = pl.DataFrame(
        [
            _roll(player_name="Keon Coleman", position="WR", roll_recv_share=0.28, roll_recv_ypt=9.0, roll_recv=70.0, roll_season_ypg=70.0),
            _roll(player_name="Khalil Shakir", position="WR", roll_recv_share=0.18, roll_recv_ypt=8.0, roll_recv=50.0, roll_season_ypg=48.0),
            _roll(
                player_name="Dalton Kincaid",
                position="TE",
                roll_recv_share=0.06,
                roll_recv_ypt=8.0,
                roll_recv=55.0,
                roll_season_ypg=67.0,
                roll_usage_collapsed=True,
            ),
        ]
    )
    props = project_props_for_game(game_row=_game(), player_rolls=rolls)
    recv = [p for p in props if p["market"] == "player_reception_yds"]
    featured = [p for p in recv if p["role"] == "featured"]
    assert [p["player"] for p in featured] == ["Keon Coleman"]
    kincaid = next(p for p in recv if p["player"] == "Dalton Kincaid")
    assert kincaid["lean"] == "under"
    assert kincaid["role"] == "secondary"
    assert kincaid["mu"] + 12 < 67
    lead = featured[0]["mu"]
    for other in recv:
        if other["player"] == "Keon Coleman":
            continue
        assert other["mu"] <= lead * 0.55 + 0.05
    assert sum(p["mu"] for p in recv) <= 210.1


def test_a_committee_names_nobody():
    rolls = pl.DataFrame(
        [
            _roll(player_name="A", position="WR", roll_recv_share=0.22, roll_recv_ypt=8.0),
            _roll(player_name="B", position="WR", roll_recv_share=0.20, roll_recv_ypt=8.0),
        ]
    )
    props = [p for p in project_props_for_game(game_row=_game(), player_rolls=rolls) if p["market"] == "player_reception_yds"]
    assert props
    assert all(p["role"] != "featured" for p in props)


def test_collapsed_flag_uses_the_last_two_games():
    rows = []
    for week, targets, yards in ((1, 8, 90), (2, 6, 70), (3, 2, 10), (4, 2, 7)):
        rows.append(
            {
                "season": 2026,
                "week": week,
                "recent_team": "BUF",
                "player_display_name": "Dalton Kincaid",
                "position": "TE",
                "targets": targets,
                "receiving_yards": yards,
                "attempts": 0,
                "passing_yards": 0,
                "rushing_yards": 0,
            }
        )
        rows.append(
            {
                "season": 2026,
                "week": week,
                "recent_team": "BUF",
                "player_display_name": "Josh Allen",
                "position": "QB",
                "targets": 0,
                "receiving_yards": 0,
                "attempts": 30,
                "passing_yards": 240,
                "rushing_yards": 20,
            }
        )
    rolled = rolling_player_yards(pl.DataFrame(rows))
    kincaid = rolled.filter(pl.col("player_name") == "Dalton Kincaid").sort("week")
    assert kincaid.filter(pl.col("week") == 3)["post_roll_usage_collapsed"][0] is False
    assert kincaid.filter(pl.col("week") == 4)["post_roll_usage_collapsed"][0] is True


def test_advisor_does_not_bet_the_secondary_receiver():
    plays = _prop_plays(
        [
            {
                "player": "Keon Coleman",
                "team": "BUF",
                "market": "player_reception_yds",
                "mu": 74.0,
                "sigma": 22.0,
                "roll": 70.0,
                "edge_vs_roll": 4.0,
                "role": "featured",
                "lean": "featured",
            },
            {
                "player": "Khalil Shakir",
                "team": "BUF",
                "market": "player_reception_yds",
                "mu": 30.0,
                "sigma": 22.0,
                "roll": 50.0,
                "edge_vs_roll": -20.0,
                "role": "secondary",
                "lean": None,
            },
            {
                "player": "Dalton Kincaid",
                "team": "BUF",
                "market": "player_reception_yds",
                "mu": 14.0,
                "sigma": 22.0,
                "roll": 55.0,
                "edge_vs_roll": -41.0,
                "role": "secondary",
                "lean": "under",
                "season_ypg": 67.0,
            },
        ],
        {},
    )
    names = [p["player"] for p in plays]
    assert "Khalil Shakir" not in names
    assert any(p["player"] == "Keon Coleman" and "only playmaker" in p["instruction"] for p in plays)
    assert any(p["player"] == "Dalton Kincaid" and p["verb"] == "UNDER" for p in plays)
