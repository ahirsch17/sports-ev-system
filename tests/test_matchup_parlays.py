"""Every game uses the same parlay rules, including a defender who just arrived."""

import polars as pl

from nfl_edge.models.matchup_parlays import (
    build_matchup_parlays,
    starter_yard_cuts,
    write_script,
    _checked_and_left_off,
    _corners_already_counted,
    _mix_tickets,
    _receptions_when_yards_move,
    _typical_yards,
)


def _stats():
    rows = []
    # Chicago's new tackle played for Tennessee, not Chicago.
    for week, sacks, tfl in ((1, 1, 1), (2, 1, 1), (3, 1, 2), (4, 1, 1)):
        rows.append(
            {
                "player_display_name": "Jeffery Simmons",
                "position": "DT",
                "season": 2026,
                "week": week,
                "season_type": "REG",
                "team": "TEN",
                "opponent_team": "NYJ",
                "attempts": 0,
                "passing_yards": 0,
                "receiving_yards": 0,
                "rushing_yards": 0,
                "carries": 0,
                "def_sacks": float(sacks),
                "def_tackles_for_loss": tfl,
            }
        )
    # Seattle offense and Chicago defense, four weeks.
    for week, jsn, wilson_rush, wilson_car, pass_yds, wr, te, rb_rec, rush in (
        (1, 120, 70, 18, 240, 150, 40, 30, 80),
        (2, 110, 60, 16, 230, 140, 50, 30, 70),
        (3, 130, 80, 20, 250, 160, 40, 20, 75),
        (4, 100, 65, 15, 220, 140, 40, 30, 72),
    ):
        rows.append(_qb("Sam Darnold", "SEA", "CHI", week, pass_yds))
        rows.append(_skill("Jaxon Smith-Njigba", "WR", "SEA", "CHI", week, jsn, 0, 0))
        rows.append(_skill("Emanuel Wilson", "RB", "SEA", "CHI", week, 10, wilson_rush, wilson_car))
        rows.append(_skill("AJ Barner", "TE", "SEA", "CHI", week, 12, 0, 0))
    return pl.DataFrame(rows)


def _qb(name, team, opp, week, yards):
    return {
        "player_display_name": name,
        "position": "QB",
        "season": 2026,
        "week": week,
        "season_type": "REG",
        "team": team,
        "opponent_team": opp,
        "attempts": 30,
        "passing_yards": yards,
        "receiving_yards": 0,
        "rushing_yards": 5,
        "carries": 2,
        "def_sacks": 0.0,
        "def_tackles_for_loss": 0,
    }


def _skill(name, pos, team, opp, week, rec, rush, carries):
    return {
        "player_display_name": name,
        "position": pos,
        "season": 2026,
        "week": week,
        "season_type": "REG",
        "team": team,
        "opponent_team": opp,
        "attempts": 0,
        "passing_yards": 0,
        "receiving_yards": rec,
        "rushing_yards": rush,
        "carries": carries,
        "def_sacks": 0.0,
        "def_tackles_for_loss": 0,
    }


def _depth():
    return pl.DataFrame(
        {
            "dt": ["2026-11-01T00:00:00Z"],
            "team": ["CHI"],
            "player_name": ["Jeffery Simmons"],
            "pos_abb": ["RDT"],
            "pos_rank": [1],
        }
    )


def test_the_third_ticket_pairs_a_yards_leg_with_that_players_count():
    def leg(player, market, price=-110):
        return {
            "player": player,
            "market": market,
            "side": "over",
            "line": 10,
            "price": price,
            "label": f"{player} {market}",
            "detail": "He averages 10 a game over his last 4.",
        }

    tickets = _mix_tickets(
        [
            {
                "legs": [
                    leg("Caleb Williams", "pass"),
                    leg("Jaxon Smith-Njigba", "rec"),
                    leg("Jaxon Smith-Njigba", "receptions", -150),
                    leg("Luther Burden III", "receptions", -150),
                    leg("D'Andre Swift", "carries", -150),
                ]
            }
        ]
    )
    assert [ticket["team"] for ticket in tickets] == ["Yards", "Counts", "Combo"]
    combo = {(leg["player"], leg["market"]) for leg in tickets[2]["legs"]}
    assert combo == {
        ("Jaxon Smith-Njigba", "rec"),
        ("Jaxon Smith-Njigba", "receptions"),
    }


def test_one_spike_game_does_not_set_the_average():
    number, spike = _typical_yards([32, 61, 63, 47, 95, 130])
    assert spike == 130
    assert number == (32 + 61 + 63 + 47 + 95) / 5
    number, spike = _typical_yards([155, 128, 76, 102, 134, 150])
    assert spike is None
    assert round(number) == 124
    number, spike = _typical_yards([81, 61, 52, 100])
    assert spike == 100
    assert number == (81 + 61 + 52) / 3


def test_a_new_tackle_trims_the_room_and_does_not_rewrite_it():
    cuts = starter_yard_cuts(_stats(), _depth(), "CHI", 2026, 8)
    assert cuts["names"] == ["Jeffery Simmons (TEN, RDT)"]
    assert 0 < cuts["pass"] <= 12
    assert 0 < cuts["rush"] <= 14
    assert cuts["wr"] == 0


def test_every_game_gets_the_same_ticket_shape_without_a_moneyline():
    game = {
        "game_id": "2026_08_CHI_SEA",
        "season": 2026,
        "week": 8,
        "gameday": "2026-11-02",
        "home_team": "CHI",
        "away_team": "SEA",
        "home_score": None,
        "pred_home_pass_yds": 230,
        "pred_away_pass_yds": 240,
    }
    other = dict(game, game_id="2026_08_OTHER", home_team="CHI", away_team="SEA")
    cards = build_matchup_parlays([game, other], _stats(), _depth(), None)
    assert len(cards) == 2
    assert set(cards[0]) == set(cards[1])
    for card in cards:
        assert "moneyline" not in card["note"].lower()
        for ticket in card["tickets"]:
            assert ticket["legs"]
            markets = []
            for leg in ticket["legs"]:
                assert "moneyline" not in leg["label"].lower()
                assert "a game" in leg["detail"]
                markets.append((leg["player"], leg["market"]))
            assert len(markets) == len(set(markets))
    sea = cards[0]["tickets"]
    names = [leg["player"] for ticket in sea for leg in ticket["legs"]]
    assert "AJ Barner" not in names
    for ticket in sea:
        for leg in ticket["legs"]:
            if leg["market"] not in {"carries", "receptions", "attempts"}:
                continue
            avg = next(bar["value"] for bar in leg["bars"] if bar["name"] == "Avg")
            assert leg["line"] >= avg - 4.5
    assert "touchdown" not in cards[0]["script"].lower()
    for ticket in cards[0]["tickets"]:
        for leg in ticket["legs"]:
            assert "td" not in leg["market"]


def test_an_absent_teammate_does_not_donate_his_old_yards():
    rows = _stats()
    extra = [_skill("Rashid Shaheed", "WR", "SEA", "CHI", week, 40, 0, 0) for week in (1, 2, 3, 4)]
    for week in (5, 6, 7):
        extra.append(_qb("Other Qb", "NE", "CHI", week, 180))
        extra.append(_skill("Other Wr", "WR", "NE", "CHI", week, 40, 0, 0))
    stats = pl.concat([rows, pl.DataFrame(extra)], how="diagonal")
    game = {
        "game_id": "2026_08_SEA_CHI",
        "season": 2026,
        "week": 8,
        "home_team": "CHI",
        "away_team": "SEA",
        "pred_home_pass_yds": 220,
        "pred_away_pass_yds": 230,
    }
    injuries = pl.DataFrame(
        {
            "season": [2026],
            "week": [8],
            "team": ["SEA"],
            "full_name": ["Rashid Shaheed"],
            "position": ["WR"],
            "report_status": ["Out"],
        }
    )
    kept = build_matchup_parlays([game], stats, None, None)
    gone = build_matchup_parlays([game], stats, None, injuries)

    def jsn_line(cards):
        for ticket in cards[0]["tickets"]:
            for leg in ticket["legs"]:
                if leg["player"] == "Jaxon Smith-Njigba" and leg["market"] == "rec":
                    return leg["line"]
        return None

    assert jsn_line(kept) == jsn_line(gone)
    assert "Rashid Shaheed" not in {leg["player"] for t in gone[0]["tickets"] for leg in t["legs"]}


def test_a_receiver_who_fell_off_and_a_corner_already_in_the_room_are_named_every_game():
    stats = pl.DataFrame(
        [
            {
                "player_display_name": "Kalif Raymond",
                "position": "WR",
                "team": "CHI",
                "season": 2026,
                "week": week,
                "season_type": "REG",
                "targets": targets,
                "receiving_yards": yards,
            }
            for week, targets, yards in ((1, 9, 84), (2, 5, 40), (3, 7, 90), (4, 2, 6), (5, 4, 6), (6, 2, 18), (7, 2, 0))
        ]
        + [
            {
                "player_display_name": "Jaylon Johnson",
                "position": "CB",
                "team": "CHI",
                "season": 2026,
                "week": week,
                "season_type": "REG",
                "targets": 0,
                "receiving_yards": 0,
            }
            for week in (1, 2, 3, 5, 6, 7)
        ]
    )
    depth = pl.DataFrame(
        {
            "dt": ["2026-11-01"] * 2,
            "team": ["CHI", "CHI"],
            "player_name": ["Jaylon Johnson", "Kalif Raymond"],
            "pos_abb": ["LCB", "WR"],
            "pos_rank": [1, 3],
        }
    )
    checked = _checked_and_left_off(stats, "CHI", 2026, 8, set())
    assert "Kalif Raymond (6, 6, 18, 0)" in checked
    corner = _corners_already_counted(stats, depth, "CHI", "SEA", 2026, 8)
    assert "Jaylon Johnson (6 games)" in corner
    assert "Seattle's receiver room" in corner
    assert "not a second cut" in corner


def test_a_rushing_over_that_misses_half_the_window_drops_and_carries_are_not_a_posted_market():
    rows = []
    for week, rush, carries in ((4, 58, 15), (5, 43, 9), (6, 44, 16), (7, 78, 22)):
        rows.append(_skill("D'Andre Swift", "RB", "CHI", "SEA", week, 10, rush, carries))
    for week, yards in ((1, 280), (2, 270), (6, 290), (7, 300)):
        rows.append(_qb("Caleb Williams", "CHI", "SEA", week, yards))
    for week, rush in ((1, 90), (2, 90), (3, 90)):
        rows.append(_qb("Other Qb", "NE", "SEA", week, 180))
        rows.append(_skill("Other Back", "RB", "NE", "SEA", week, 10, rush, 14))
    game = {
        "game_id": "2026_08_CHI_SEA",
        "season": 2026,
        "week": 8,
        "home_team": "SEA",
        "away_team": "CHI",
        "pred_away_pass_yds": 210,
        "pred_home_pass_yds": 210,
    }
    cards = build_matchup_parlays([game], pl.DataFrame(rows), None, None)
    legs = [leg for ticket in cards[0]["tickets"] for leg in ticket["legs"]]
    markets = {(leg["player"], leg["market"]) for leg in legs}
    assert ("D'Andre Swift", "rush") not in markets
    assert ("D'Andre Swift", "carries") not in markets


def test_catches_are_the_leg_when_the_yards_move_around():
    rows = []
    for week, catches, yards in ((2, 3, 32), (3, 7, 61), (4, 4, 63), (5, 4, 47), (6, 8, 95), (7, 4, 130)):
        rows.append(
            {
                "player_display_name": "Luther Burden III",
                "position": "WR",
                "week": week,
                "receptions": catches,
                "targets": catches + 2,
                "receiving_yards": yards,
                "rushing_tds": 0,
                "receiving_tds": 0,
            }
        )
    for week, catches in ((4, 4), (5, 4), (6, 5), (7, 4)):
        rows.append(
            {
                "player_display_name": "Steady Two",
                "position": "WR",
                "week": week,
                "receptions": catches,
                "targets": 6,
                "receiving_yards": 40,
                "rushing_tds": 0,
                "receiving_tds": 0,
            }
        )
    legs = _receptions_when_yards_move([], pl.DataFrame(rows))
    assert len(legs) == 1
    assert legs[0]["player"] == "Luther Burden III"
    assert legs[0]["line"] == 3.5
    steady = []
    for week, catches in ((2, 6), (3, 6), (4, 7), (5, 6), (6, 8), (7, 6)):
        steady.append(
            {
                "player_display_name": "Posted Catch",
                "position": "WR",
                "week": week,
                "receptions": catches,
                "targets": 8,
                "receiving_yards": 40 + week,
                "rushing_tds": 0,
                "receiving_tds": 0,
            }
        )
    posted = _receptions_when_yards_move([], pl.DataFrame(steady))
    assert len(posted) == 1
    assert posted[0]["player"] == "Posted Catch"
    assert posted[0]["line"] == 5.5
    assert posted[0]["label"] == "Posted Catch over 5.5 receptions."
    # A player whose yards already cleared does not get a second catch price.
    assert _receptions_when_yards_move([{"player": "Luther Burden III", "market": "rec"}], pl.DataFrame(rows)) == []


def test_the_script_uses_the_score_and_does_not_add_a_touchdown():
    text = write_script(
        {
            "home_team": "SEA",
            "away_team": "CHI",
            "pred_margin": 2.79,
            "model_margin": 0.88,
            "pred_total": 46.03,
            "model_total": 41.78,
            "pred_home_points": 24.41,
            "pred_away_points": 21.62,
            "roof": "outdoors",
            "wind": None,
            "temp": None,
            "rest_note": "rest 8/11",
            "deep_net_margin_adj": -0.44,
            "away_inj_note": "CHI: healthy-ish",
            "home_inj_note": "SEA: skill/OL hit (-2.9 pts)",
        },
        [{"team": "CHI", "legs": [{"player": "Caleb Williams"}, {"player": "D'Andre Swift"}]}],
        ["Chicago added Jeffery Simmons (TEN, RDT) after these totals were built."],
    )
    assert "Our number is Seattle by 1, 21–20" in text
    assert "Our total is 42" in text
    assert "The book is Seattle by 3 with a total of 46" in text
    assert "The side stays off" in text
    assert "no wind or temperature" in text
    assert "Seattle is on 8 days of rest and Chicago is on 11" in text
    assert "Chicago is healthy enough" in text
    assert "docks them 2.9 points" in text
    assert "Caleb Williams" in text
    assert "touchdown" not in text.lower()
    named = write_script(
        {
            "home_team": "SEA",
            "away_team": "CHI",
            "season": 2026,
            "week": 8,
        },
        [],
        [],
        pl.DataFrame(
            {
                "season": [2026, 2026],
                "week": [8, 8],
                "team": ["CHI", "SEA"],
                "full_name": ["Kyle Monangai", "Charles Cross"],
                "position": ["RB", "T"],
                "report_status": ["Out", "Out"],
            }
        ),
    )
    assert "Kyle Monangai (RB)" in named
    assert "Charles Cross (T)" in named


def test_a_fill_in_is_not_the_rush_leg_when_the_starter_returns():
    rows = []
    for week, rush, carries in ((4, 81, 21), (5, 61, 20), (6, 52, 14), (7, 100, 24)):
        rows.append(_skill("Emanuel Wilson", "RB", "SEA", "CHI", week, 5, rush, carries))
        rows.append(_qb("Sam Darnold", "SEA", "CHI", week, 240))
        rows.append(_skill("Jaxon Smith-Njigba", "WR", "SEA", "CHI", week, 110, 0, 0))
        rows.append(_qb("Other Qb", "NE", "SEA", week, 160))
        rows.append(_skill("Other Back", "RB", "NE", "SEA", week, 0, 70, 14))
    depth = pl.DataFrame(
        {
            "dt": ["2026-11-02T00:00:00Z", "2026-11-02T00:00:00Z"],
            "team": ["SEA", "SEA"],
            "player_name": ["Zach Charbonnet", "Emanuel Wilson"],
            "pos_abb": ["RB", "RB"],
            "pos_rank": [1, 2],
        }
    )
    injuries = pl.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "week": [4, 5, 6, 7],
            "team": ["SEA", "SEA", "SEA", "SEA"],
            "full_name": ["Zach Charbonnet"] * 4,
            "position": ["RB"] * 4,
            "report_status": ["Out"] * 4,
        }
    )
    game = {
        "game_id": "2026_08_CHI_SEA",
        "season": 2026,
        "week": 8,
        "home_team": "CHI",
        "away_team": "SEA",
        "pred_away_pass_yds": 230,
        "pred_home_pass_yds": 210,
    }
    cards = build_matchup_parlays([game], pl.DataFrame(rows), depth, injuries)
    markets = {(leg["player"], leg["market"]) for ticket in cards[0]["tickets"] for leg in ticket["legs"]}
    assert ("Emanuel Wilson", "rush") not in markets
    assert ("Emanuel Wilson", "carries") not in markets
    assert "Zach Charbonnet is active" in cards[0]["script"]
    assert "Emanuel Wilson" in cards[0]["script"]


def test_the_pass_under_clears_the_projection_plus_one_long_gain():
    rows = []
    for week, yards in ((1, 240), (2, 250), (3, 260), (4, 250)):
        rows.append(_qb("Caleb Williams", "CHI", "GB", week, yards))
    for week, yards in ((1, 170), (2, 180), (3, 190), (4, 180)):
        rows.append(_qb("Other Qb", "NE", "SEA", week, yards))
    game = {
        "game_id": "2026_05_CHI_SEA",
        "season": 2026,
        "week": 5,
        "home_team": "SEA",
        "away_team": "CHI",
        "pred_away_pass_yds": 210,
        "pred_home_pass_yds": 210,
    }
    cards = build_matchup_parlays([game], pl.DataFrame(rows), None, None)
    unders = [
        leg
        for ticket in cards[0]["tickets"]
        for leg in ticket["legs"]
        if leg["market"] == "pass" and leg["side"] == "under"
    ]
    assert len(unders) == 1
    assert unders[0]["line"] == 244.5
    assert unders[0]["label"] == "Caleb Williams under 244.5 passing yards."
    assert "Take -110 or longer." in unders[0]["detail"]


def test_the_fanduel_ticket_is_the_side_of_our_number():
    from nfl_edge.models.matchup_parlays import _fanduel_ticket

    ticket = _fanduel_ticket(
        [
            {"player": "Caleb Williams", "market": "player_pass_yds", "mu": 222, "roll": 255},
            {"player": "Jaxon Smith-Njigba", "market": "player_reception_yds", "mu": 91, "roll": 118},
            {"player": "Luther Burden III", "market": "player_reception_yds", "mu": 68, "roll": 79},
            {"player": "D'Andre Swift", "market": "player_rush_yds", "mu": 63, "roll": 61},
        ]
    )
    assert [leg["label"] for leg in ticket["legs"]] == ["Caleb Williams passing yards under."]
    assert ticket["odds"] == -110


def test_anytime_scorers_join_the_passing_leg_and_a_fill_in_stays_off():
    from nfl_edge.models.matchup_parlays import _aligned_tickets, _anytime_legs

    def row(name, pos, team, week, catches, carries, rec_td, rush_td):
        return {
            "player_display_name": name,
            "position": pos,
            "season": 2026,
            "week": week,
            "season_type": "REG",
            "team": team,
            "receptions": catches,
            "carries": carries,
            "receiving_tds": rec_td,
            "rushing_tds": rush_td,
        }

    rows = []
    for week, jsn, swift, wilson in (
        (1, 1, 1, 1),
        (2, 0, 1, 0),
        (3, 1, 0, 1),
        (4, 1, 1, 0),
        (5, 0, 1, 1),
        (6, 1, 0, 0),
        (7, 1, 0, 0),
    ):
        rows.append(row("Jaxon Smith-Njigba", "WR", "SEA", week, 8, 0, jsn, 0))
        rows.append(row("D'Andre Swift", "RB", "CHI", week, 3, 14, 0, swift))
        rows.append(row("Emanuel Wilson", "RB", "SEA", week, 2, 12, 0, wilson))
        rows.append(row("Quiet Receiver", "WR", "CHI", week, 4, 0, 0, 0))
    for week, team in ((1, "SEA"), (2, "SEA"), (3, "SEA"), (4, "SEA"), (5, "IND"), (6, "IND"), (7, "IND")):
        rows.append(row("Moved On", "WR", team, week, 6, 0, 1, 0))
    stats = pl.DataFrame(rows)
    depth = pl.DataFrame(
        {
            "dt": ["2026-11-01"],
            "team": ["SEA"],
            "player_name": ["Zach Charbonnet"],
            "pos_abb": ["RB"],
            "pos_rank": [1],
        }
    )
    injuries = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [5, 6],
            "team": ["SEA", "SEA"],
            "full_name": ["Zach Charbonnet", "Zach Charbonnet"],
            "report_status": ["Out", "Out"],
            "position": ["RB", "RB"],
        }
    )
    scorers = _anytime_legs(stats, depth, injuries, ["CHI", "SEA"], 2026, 8)
    assert [leg["player"] for leg in scorers] == ["Jaxon Smith-Njigba", "D'Andre Swift"]
    assert "Moved On" not in [leg["player"] for leg in scorers]
    tickets = _aligned_tickets(
        {"model_total": 42, "total_line": 46.5, "spread_line": 3, "home_team": "SEA", "away_team": "CHI"},
        [
            {"player": "Caleb Williams", "team": "CHI", "market": "player_pass_yds", "mu": 222, "roll": 255},
            {"player": "Sam Darnold", "team": "SEA", "market": "player_pass_yds", "mu": 239, "roll": 255},
            {"player": "Jaxon Smith-Njigba", "team": "SEA", "market": "player_reception_yds", "mu": 91, "roll": 118},
            {"player": "Luther Burden III", "team": "CHI", "market": "player_reception_yds", "mu": 68, "roll": 79},
        ],
        scorers,
    )
    assert [ticket["team"] for ticket in tickets] == ["Props", "Total", "Scores"]
    assert tickets[0]["legs"][0]["bars"][0]["value"] == 222.0
    assert [leg["player"] for leg in tickets[0]["legs"]] == [
        "Caleb Williams",
        "Jaxon Smith-Njigba",
    ]
    assert tickets[0]["odds"] == 264
    assert tickets[0]["price_label"] == "two bets"
    assert "+224" in tickets[0]["why"]
    scores = next(ticket for ticket in tickets if ticket["team"] == "Scores")
    assert [leg["player"] for leg in scores["legs"]] == ["Jaxon Smith-Njigba", "D'Andre Swift"]
    assert "half" not in tickets[0]["why"].lower()
    same_backfield = _aligned_tickets(
        {"home_team": "SEA", "away_team": "CHI"},
        [
            {"player": "Bucky Irving", "team": "TB", "market": "player_rush_yds", "mu": 79, "roll": 110},
            {"player": "Kenny Gainwell", "team": "TB", "market": "player_rush_yds", "mu": 35, "roll": 54},
        ],
        [],
    )
    assert same_backfield == []
    filled = _aligned_tickets(
        {"model_total": 42, "total_line": 46.5, "home_team": "SEA", "away_team": "CHI"},
        [
            {"player": "Caleb Williams", "team": "CHI", "market": "player_pass_yds", "mu": 222, "roll": 255},
            {"player": "Emanuel Wilson", "team": "SEA", "market": "player_rush_yds", "mu": 36, "roll": 62},
            {"player": "Zach Charbonnet", "team": "SEA", "market": "player_rush_yds", "mu": 36, "roll": 53},
        ],
        [],
        {"SEA": "Zach Charbonnet"},
    )
    assert [leg["player"] for leg in filled[0]["legs"]] == ["Caleb Williams", "Zach Charbonnet"]
    same_pie = _aligned_tickets(
        {"model_total": 42, "total_line": 46.5, "home_team": "SEA", "away_team": "CHI"},
        [
            {"player": "Caleb Williams", "team": "CHI", "market": "player_pass_yds", "mu": 222, "roll": 255},
            {"player": "Luther Burden III", "team": "CHI", "market": "player_reception_yds", "mu": 50, "roll": 80},
        ],
        [],
    )
    assert [ticket["team"] for ticket in same_pie] == ["Total"]


def test_a_back_under_does_not_join_a_game_over():
    from nfl_edge.models.matchup_parlays import _aligned_tickets

    tickets = _aligned_tickets(
        {"model_total": 47, "total_line": 41.5, "home_team": "SEA", "away_team": "ARI"},
        [
            {"player": "Zach Charbonnet", "team": "SEA", "market": "player_rush_yds", "mu": 29, "roll": 52},
            {"player": "Emanuel Wilson", "team": "SEA", "market": "player_rush_yds", "mu": 29, "roll": 62},
        ],
        [],
    )
    assert [ticket["team"] for ticket in tickets] == ["Total"]
    assert tickets[0]["legs"][0]["side"] == "over"


def test_a_collapsed_line_does_not_make_the_pair():
    from nfl_edge.models.matchup_parlays import _aligned_tickets

    tickets = _aligned_tickets(
        {"model_total": 41, "total_line": 40, "home_team": "PHI", "away_team": "NYG"},
        [
            {"player": "DeVonta Smith", "team": "PHI", "market": "player_reception_yds", "mu": 26, "roll": 76},
            {"player": "Malik Nabers", "team": "NYG", "market": "player_reception_yds", "mu": 70, "roll": 107},
        ],
        [],
    )
    assert tickets == []


def test_the_same_offense_pairs_the_pass_under_with_the_back():
    from nfl_edge.models.matchup_parlays import _aligned_tickets

    tickets = _aligned_tickets(
        {"model_total": 49, "total_line": 52.5, "home_team": "ATL", "away_team": "CIN"},
        [
            {"player": "Joe Burrow", "team": "CIN", "market": "player_pass_yds", "mu": 279, "roll": 331},
            {"player": "Chase Brown", "team": "CIN", "market": "player_rush_yds", "mu": 61, "roll": 41},
            {"player": "Bijan Robinson", "team": "ATL", "market": "player_rush_yds", "mu": 101, "roll": 86},
        ],
        [],
    )
    assert [leg["player"] for leg in tickets[0]["legs"]] == ["Joe Burrow", "Chase Brown"]
    assert "carries" in tickets[0]["blurb"]
