"""Passing and receiving yards from a published box score fill a hole in the stat feed."""

import json
from pathlib import Path

import polars as pl

from nfl_edge.ingest.late_boxscores import (
    apply_player_boxscores,
    apply_team_boxscores,
    parse_summary,
    week_should_fill,
)


def _payload() -> dict:
    return {
        "header": {
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"abbreviation": "SEA"}},
                        {"homeAway": "away", "team": {"abbreviation": "CHI"}},
                    ]
                }
            ]
        },
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "CHI"},
                    "statistics": [
                        {
                            "name": "passing",
                            "labels": ["C/ATT", "YDS", "TD", "INT", "SACKS"],
                            "athletes": [
                                {"athlete": {"displayName": "Caleb Williams"}, "stats": ["24/34", "247", "1", "0", "1-10"]}
                            ],
                        },
                        {
                            "name": "receiving",
                            "labels": ["REC", "YDS", "TD", "TGTS"],
                            "athletes": [
                                {"athlete": {"displayName": "Rome Odunze"}, "stats": ["4", "87", "0", "6"]},
                                {"athlete": {"displayName": "Luther Burden III"}, "stats": ["7", "56", "1", "9"]},
                            ],
                        },
                    ],
                },
                {
                    "team": {"abbreviation": "SEA"},
                    "statistics": [
                        {
                            "name": "passing",
                            "labels": ["C/ATT", "YDS", "TD", "INT", "SACKS"],
                            "athletes": [
                                {"athlete": {"displayName": "Sam Darnold"}, "stats": ["17/34", "188", "2", "1", "3-17"]}
                            ],
                        },
                        {
                            "name": "receiving",
                            "labels": ["REC", "YDS", "TD", "TGTS"],
                            "athletes": [
                                {"athlete": {"displayName": "Jaxon Smith-Njigba"}, "stats": ["6", "118", "1", "11"]},
                            ],
                        },
                    ],
                },
            ]
        },
    }


def test_the_box_score_keeps_gross_passing_and_every_catch():
    game = {"home_team": "SEA", "away_team": "CHI", "season": 2026, "week": 8, "game_id": "2026_08_CHI_SEA"}
    players, teams = parse_summary(_payload(), game)
    by_name = {row["player_display_name"]: row for row in players}
    assert by_name["Caleb Williams"]["passing_yards"] == 247
    assert by_name["Caleb Williams"]["attempts"] == 34
    assert by_name["Rome Odunze"]["receiving_yards"] == 87
    assert by_name["Luther Burden III"]["receiving_yards"] == 56
    assert by_name["Sam Darnold"]["passing_yards"] == 188
    assert by_name["Jaxon Smith-Njigba"]["receiving_yards"] == 118
    assert by_name["Jaxon Smith-Njigba"]["opponent_team"] == "CHI"
    team = {row["team"]: row for row in teams}
    # Gross quarterback yards, not the net-of-sacks team total.
    assert team["CHI"]["passing_yards"] == 247
    assert team["SEA"]["passing_yards"] == 188


def test_a_partial_frame_is_not_treated_as_the_live_feed():
    stats = pl.DataFrame(
        {
            "player_display_name": ["Caleb Williams"],
            "season": [2026],
            "week": [7],
            "game_id": ["2026_07_NE_CHI"],
            "passing_yards": [308],
        }
    )
    assert apply_player_boxscores(stats).height == 1


def test_the_published_monday_lines_match_the_box_score():
    """The saved ESPN summary is the source. The stat file does not have the game yet."""
    path = Path("/tmp/chi_sea.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    game = {"home_team": "SEA", "away_team": "CHI", "season": 2026, "week": 8, "game_id": "2026_08_CHI_SEA"}
    players, teams = parse_summary(payload, game)
    by_name = {row["player_display_name"]: row for row in players}
    assert by_name["Caleb Williams"]["passing_yards"] == 247
    assert by_name["Sam Darnold"]["passing_yards"] == 188
    assert by_name["Rome Odunze"]["receiving_yards"] == 87
    assert by_name["Colston Loveland"]["receiving_yards"] == 65
    assert by_name["Luther Burden III"]["receiving_yards"] == 56
    assert by_name["Jaxon Smith-Njigba"]["receiving_yards"] == 118
    assert by_name["D'Andre Swift"]["rushing_yards"] == 69
    team = {row["team"]: row for row in teams}
    assert team["CHI"]["rushing_yards"] == 126
    assert team["SEA"]["rushing_yards"] == 81


def test_only_a_real_current_season_file_fills_an_unpublished_week():
    assert week_should_fill(14, 16, season=2026, earlier_counts=[], current_season=2026)
    assert not week_should_fill(0, 16, season=2026, earlier_counts=[1], current_season=2026)
    assert week_should_fill(0, 16, season=2026, earlier_counts=[16], current_season=2026)
    assert not week_should_fill(0, 16, season=2025, earlier_counts=[16], current_season=2026)
    assert not week_should_fill(16, 16, season=2026, earlier_counts=[16], current_season=2026)


def test_an_older_season_file_is_left_alone():
    old = pl.DataFrame(
        {
            "player_display_name": ["Caleb Williams"],
            "season": [2025],
            "week": [8],
            "game_id": ["2025_08_CHI_SEA"],
            "passing_yards": [100],
        }
    )
    assert apply_player_boxscores(old).height == 1
    teams = pl.DataFrame(
        {
            "team": ["CHI"],
            "season": [2025],
            "week": [8],
            "game_id": ["2025_08_CHI_SEA"],
            "passing_yards": [100],
            "rushing_yards": [50],
        }
    )
    assert apply_team_boxscores(teams).height == 1
