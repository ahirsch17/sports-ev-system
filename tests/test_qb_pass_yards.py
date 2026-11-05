"""A soft defense and an injured receiver do not knock a quarterback under his own yards."""

import polars as pl

from nfl_edge.models.props import (
    blend_pass_yards,
    defense_shape_rooms,
    pass_yards_allowed_by_defense,
    project_props_for_game,
)


def test_a_doubtful_starter_gives_the_ball_to_the_next_quarterback():
    rolls = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [8, 8],
            "team": ["BAL", "BAL"],
            "player_name": ["Tyler Huntley", "Joe Fagnano"],
            "position": ["QB", "QB"],
            "roll_pass": [145.0, 91.0],
            "roll_pass_std": [40.0, 30.0],
            "roll_rush": [15.0, 0.0],
            "roll_recv": [0.0, 0.0],
        }
    )
    depth = pl.DataFrame(
        {
            "team": ["BAL", "BAL"],
            "full_name": ["Tyler Huntley", "Joe Fagnano"],
            "position": ["QB", "QB"],
            "depth_team": [1, 2],
        }
    )
    game = {
        "game_id": "2026_09_JAX_BAL",
        "season": 2026,
        "week": 9,
        "home_team": "BAL",
        "away_team": "JAX",
        "pred_home_pass_yds": 210.0,
        "pred_away_pass_yds": 240.0,
        "pred_home_rush_yds": 100.0,
        "pred_away_rush_yds": 110.0,
    }
    props = project_props_for_game(
        game_row=game,
        player_rolls=rolls,
        depth_charts=depth,
        out_players={"Tyler Huntley", "Lamar Jackson"},
        opp_pass_allowed={"JAX": 220.0},
    )
    passers = [p["player"] for p in props if p["market"] == "player_pass_yds" and p["team"] == "BAL"]
    assert passers == ["Joe Fagnano"]
    assert all(p["player"] != "Tyler Huntley" for p in props)


def test_one_game_starter_uses_the_yards_he_threw():
    rolls = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [8, 8],
            "team": ["BAL", "BAL"],
            "player_name": ["Tyler Huntley", "Joe Fagnano"],
            "position": ["QB", "QB"],
            "roll_pass": [145.0, None],
            "roll_pass_std": [40.0, None],
            "roll_rush": [15.0, 0.0],
            "roll_recv": [0.0, 0.0],
        }
    )
    depth = pl.DataFrame(
        {
            "team": ["BAL", "BAL"],
            "full_name": ["Tyler Huntley", "Joe Fagnano"],
            "position": ["QB", "QB"],
            "depth_team": [1, 2],
        }
    )
    stats = pl.DataFrame(
        {
            "season": [2026],
            "team": ["BAL"],
            "player_name": ["J.Fagnano"],
            "player_display_name": ["Joe Fagnano"],
            "attempts": [21],
            "passing_yards": [91],
        }
    )
    props = project_props_for_game(
        game_row={
            "game_id": "2026_09_JAX_BAL",
            "season": 2026,
            "week": 9,
            "home_team": "BAL",
            "away_team": "JAX",
            "pred_home_pass_yds": 210.0,
            "pred_away_pass_yds": 240.0,
            "pred_home_rush_yds": 100.0,
            "pred_away_rush_yds": 110.0,
        },
        player_rolls=rolls,
        depth_charts=depth,
        out_players={"Tyler Huntley"},
        player_stats=stats,
        opp_pass_allowed={"JAX": 220.0},
    )
    fagnano = next(p for p in props if p["player"] == "Joe Fagnano" and p["market"] == "player_pass_yds")
    assert fagnano["roll"] == 91.0


def test_questionable_is_not_an_out():
    from nfl_edge.ingest.espn_injuries import outs_from_payload

    names = outs_from_payload(
        {
            "injuries": [
                {
                    "displayName": "Baltimore Ravens",
                    "injuries": [
                        {"status": "Out", "athlete": {"displayName": "Lamar Jackson"}},
                        {"status": "Doubtful", "athlete": {"displayName": "Tyler Huntley"}},
                        {"status": "Questionable", "athlete": {"displayName": "Joe Fagnano"}},
                    ],
                }
            ]
        }
    )
    assert names == {"lamar jackson", "tyler huntley"}


def test_missing_receiver_does_not_pull_the_quarterback_under():
    rolls = pl.DataFrame(
        {
            "season": [2026],
            "week": [8],
            "team": ["TEN"],
            "player_name": ["Cam Ward"],
            "position": ["QB"],
            "roll_pass": [171.0],
            "roll_pass_std": [40.0],
            "roll_rush": [0.0],
            "roll_recv": [0.0],
        }
    )
    props = project_props_for_game(
        game_row={
            "game_id": "2026_08_TEN_CIN",
            "season": 2026,
            "week": 8,
            "home_team": "CIN",
            "away_team": "TEN",
            "pred_home_pass_yds": 280.0,
            "pred_away_pass_yds": 156.0,
            "pred_home_rush_yds": 100.0,
            "pred_away_rush_yds": 90.0,
        },
        player_rolls=rolls,
        opp_pass_allowed={"CIN": 254.0},
    )
    ward = next(p for p in props if p["player"] == "Cam Ward")
    # Old blend of his 171 with a 156 team total landed near 154 and made 165.5 an under.
    assert ward["mu"] > 190


def test_stingy_defense_pulls_a_big_quarterback_down():
    rolls = pl.DataFrame(
        {
            "season": [2026],
            "week": [8],
            "team": ["DET"],
            "player_name": ["Jared Goff"],
            "position": ["QB"],
            "roll_pass": [316.0],
            "roll_pass_std": [50.0],
            "roll_rush": [0.0],
            "roll_recv": [0.0],
        }
    )
    props = project_props_for_game(
        game_row={
            "game_id": "2026_08_MIN_DET",
            "season": 2026,
            "week": 8,
            "home_team": "DET",
            "away_team": "MIN",
            "pred_home_pass_yds": 278.0,
            "pred_away_pass_yds": 213.0,
            "pred_home_rush_yds": 120.0,
            "pred_away_rush_yds": 90.0,
        },
        player_rolls=rolls,
        opp_pass_allowed={"MIN": 172.0},
    )
    goff = next(p for p in props if p["player"] == "Jared Goff")
    # Weighting his 316-yard average and ignoring Minnesota left him at 277.
    assert goff["mu"] < 250


def test_quarterback_rush_yards_are_not_the_team_rush_total():
    rolls = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [8, 8],
            "team": ["BUF", "BUF"],
            "player_name": ["Josh Allen", "James Cook"],
            "position": ["QB", "RB"],
            "roll_pass": [250.0, 0.0],
            "roll_pass_std": [40.0, 0.0],
            "roll_rush": [43.0, 104.0],
            "roll_recv": [0.0, 0.0],
        }
    )
    props = project_props_for_game(
        game_row={
            "game_id": "2026_08_BAL_BUF",
            "season": 2026,
            "week": 8,
            "home_team": "BUF",
            "away_team": "BAL",
            "pred_home_pass_yds": 276.0,
            "pred_away_pass_yds": 216.0,
            "pred_home_rush_yds": 160.0,
            "pred_away_rush_yds": 100.0,
        },
        player_rolls=rolls,
        opp_pass_allowed={"BAL": 230.0},
    )
    allen = next(p for p in props if p["player"] == "Josh Allen" and p["market"] == "player_rush_yds")
    assert allen["mu"] == 43.0


def test_defense_allowed_ignores_the_current_week():
    stats = pl.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "week": [5, 6, 7, 8],
            "season_type": ["REG", "REG", "REG", "REG"],
            "opponent_team": ["CIN", "CIN", "CIN", "CIN"],
            "position": ["QB", "QB", "QB", "QB"],
            "attempts": [30, 30, 30, 40],
            "passing_yards": [200, 220, 240, 400],
        }
    )
    allowed = pass_yards_allowed_by_defense(stats, 2026, 8)
    assert allowed["CIN"] == 220


def test_receivers_are_sized_off_the_blended_quarterback_total():
    rolls = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [8, 8],
            "team": ["CHI", "CHI"],
            "player_name": ["Caleb Williams", "Luther Burden III"],
            "position": ["QB", "WR"],
            "roll_pass": [255.0, 0.0],
            "roll_pass_std": [40.0, 0.0],
            "roll_rush": [0.0, 0.0],
            "roll_recv": [0.0, 70.0],
            "roll_recv_std": [0.0, 25.0],
        }
    )
    props = project_props_for_game(
        game_row={
            "game_id": "2026_08_CHI_SEA",
            "season": 2026,
            "week": 8,
            "home_team": "SEA",
            "away_team": "CHI",
            "pred_home_pass_yds": 230.0,
            "pred_away_pass_yds": 230.5,
            "pred_home_rush_yds": 113.0,
            "pred_away_rush_yds": 120.0,
        },
        player_rolls=rolls,
        opp_pass_allowed={"SEA": 158.0},
    )
    qb = next(p for p in props if p["player"] == "Caleb Williams")
    wr = next(p for p in props if p["player"] == "Luther Burden III")
    # Pace said 230. Seattle has allowed 158. The one number is the blend.
    assert 200 < qb["mu"] < 210
    # He is the only catcher in the room, so his yards are the quarterback's yards.
    assert wr["mu"] == qb["mu"]
    assert wr["mu"] < 230

    cut = project_props_for_game(
        game_row={
            "game_id": "2026_08_CHI_SEA",
            "season": 2026,
            "week": 8,
            "home_team": "SEA",
            "away_team": "CHI",
            "pred_home_pass_yds": 230.0,
            "pred_away_pass_yds": 230.5,
            "pred_home_rush_yds": 113.0,
            "pred_away_rush_yds": 120.0,
        },
        player_rolls=rolls,
        opp_pass_allowed={"SEA": 158.0},
        defense_cuts={"SEA": {"pass": 20.0, "rush": 0.0}},
    )
    cut_qb = next(p for p in cut if p["player"] == "Caleb Williams")
    cut_wr = next(p for p in cut if p["player"] == "Luther Burden III")
    # 45% of the 20 yards the new defender takes off the defense.
    assert cut_qb["mu"] == round(qb["mu"] - 9.0, 1)
    assert cut_wr["mu"] == cut_qb["mu"]


def test_chicago_skill_lines_have_to_clear_both_the_defense_and_the_blend():
    low = defense_shape_rooms(158.0, 91.7, 34.4, 31.9)
    high_pass = blend_pass_yards(255.0, 158.0, 230.5)
    high = defense_shape_rooms(high_pass, 91.7, 34.4, 31.9)
    burden_low = low["WR"] * 320 / 523
    burden_high = high["WR"] * 320 / 523
    odunze_low = low["WR"] * 203 / 523
    odunze_high = high["WR"] * 203 / 523
    loveland_low = low["TE"] * 0.9
    loveland_high = high["TE"] * 0.9
    # 59.5, 44.5, and 39.5 all sit inside the range, so none of them is a side.
    assert burden_low < 59.5 < burden_high
    assert odunze_low < 44.5 < odunze_high
    assert loveland_low < 39.5 < loveland_high
    assert high_pass < 220


def test_catchers_split_the_quarterback_and_sum_to_him():
    rolls = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [8, 8, 8],
            "team": ["CHI", "CHI", "CHI"],
            "player_name": ["Caleb Williams", "Luther Burden III", "Colston Loveland"],
            "position": ["QB", "WR", "TE"],
            "roll_pass": [255.0, 0.0, 0.0],
            "roll_pass_std": [40.0, 0.0, 0.0],
            "roll_rush": [20.0, 0.0, 0.0],
            "roll_recv": [0.0, 80.0, 40.0],
            "roll_recv_std": [0.0, 25.0, 20.0],
        }
    )
    props = project_props_for_game(
        game_row={
            "game_id": "2026_08_CHI_SEA",
            "season": 2026,
            "week": 8,
            "home_team": "SEA",
            "away_team": "CHI",
            "pred_away_pass_yds": 230.0,
            "pred_home_pass_yds": 230.0,
            "pred_away_rush_yds": 110.0,
            "pred_home_rush_yds": 110.0,
        },
        player_rolls=rolls,
        opp_pass_allowed={"SEA": 158.0},
    )
    qb = next(p for p in props if p["market"] == "player_pass_yds")
    burden = next(p for p in props if p["player"] == "Luther Burden III")
    loveland = next(p for p in props if p["player"] == "Colston Loveland")
    assert abs((burden["mu"] + loveland["mu"]) - qb["mu"]) <= 0.1
    assert burden["mu"] == round(loveland["mu"] * 2, 1)


def test_an_out_back_does_not_keep_a_share_of_the_rushing_total():
    stats = pl.DataFrame(
        {
            "season": [2026] * 8,
            "week": [1, 1, 2, 2, 3, 3, 4, 4],
            "season_type": ["REG"] * 8,
            "team": ["CHI"] * 8,
            "player_display_name": ["D'Andre Swift", "Kyle Monangai"] * 4,
            "position": ["RB", "RB"] * 4,
            "receiving_yards": [10, 5] * 4,
            "rushing_yards": [50, 50] * 4,
            "passing_yards": [0] * 8,
            "carries": [12, 12] * 4,
        }
    )
    rolls = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [5, 5, 5],
            "team": ["CHI", "CHI", "CHI"],
            "player_name": ["Caleb Williams", "D'Andre Swift", "Kyle Monangai"],
            "position": ["QB", "RB", "RB"],
            "roll_pass": [240.0, 0.0, 0.0],
            "roll_rush": [10.0, 50.0, 50.0],
            "roll_recv": [0.0, 10.0, 5.0],
        }
    )
    props = project_props_for_game(
        game_row={
            "game_id": "2026_05_CHI_SEA",
            "season": 2026,
            "week": 5,
            "home_team": "SEA",
            "away_team": "CHI",
            "pred_away_pass_yds": 220.0,
            "pred_home_pass_yds": 220.0,
            "pred_away_rush_yds": 100.0,
            "pred_home_rush_yds": 100.0,
        },
        player_rolls=rolls,
        out_players={"kyle monangai"},
        player_stats=stats,
    )
    swift = next(p for p in props if p["player"] == "D'Andre Swift" and p["market"] == "player_rush_yds")
    assert all(p["player"] != "Kyle Monangai" for p in props)
    # Monangai had half the recent rushing. Handing all 100 to Swift runs high.
    # 70% of that share plus 30% of Swift's own 50 is 85. The other 15 stays
    # in the team total with the rest of the backfield.
    assert swift["mu"] == 85.0
