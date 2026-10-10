"""Bet log: price parsing, grading, profit and summary math."""

import pytest

from nfl_edge import tracker


@pytest.fixture(autouse=True)
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "LOG_PATH", tmp_path / "bet_log.json")


def test_parse_price_formats():
    assert tracker.parse_price("-150") == pytest.approx(0.6)
    assert tracker.parse_price("+300") == pytest.approx(0.25)
    assert tracker.parse_price("62c") == pytest.approx(0.62)
    assert tracker.parse_price("62¢") == pytest.approx(0.62)
    assert tracker.parse_price("0.62") == pytest.approx(0.62)
    for bad in ["", "abc", "-50", "150c"]:
        with pytest.raises(ValueError):
            tracker.parse_price(bad)


def _bet(**kw):
    base = {
        "sport": "NFL", "season": 2026, "week": 3, "game_id": "g1", "matchup": "ARI @ SF",
        "market": "h2h", "pick": "SF to win", "team": "SF", "is_home": True,
        "price": "-200", "stake": 1, "model_prob": 0.7,
    }
    base.update(kw)
    return base


def test_grade_moneyline_spread_total():
    ml = tracker.add_bet(_bet())
    assert tracker._grade_game_market(ml, home=24, away=20)[0] == "won"
    assert tracker._grade_game_market(ml, home=17, away=20)[0] == "lost"

    away_spread = tracker.add_bet(_bet(market="spreads", team="ARI", is_home=False, line=8.5))
    assert tracker._grade_game_market(away_spread, home=27, away=20)[0] == "won"
    assert tracker._grade_game_market(away_spread, home=30, away=20)[0] == "lost"
    home_spread = tracker.add_bet(_bet(market="spreads", line=-7))
    assert tracker._grade_game_market(home_spread, home=27, away=20)[0] == "push"

    under = tracker.add_bet(_bet(market="totals", side="Under", line=47.5, is_home=None))
    assert tracker._grade_game_market(under, home=24, away=20)[0] == "won"
    assert tracker._grade_game_market(under, home=28, away=20)[0] == "lost"


def test_bad_payloads_rejected():
    with pytest.raises(ValueError):
        tracker.add_bet(_bet(market="spreads", line=None))
    with pytest.raises(ValueError):
        tracker.add_bet(_bet(market="totals", line=45, side=None, is_home=None))
    with pytest.raises(ValueError):
        tracker.add_bet(_bet(stake=0))
    with pytest.raises(ValueError):
        tracker.add_bet(_bet(price="nope"))


def test_profit_and_summary():
    fav = tracker.add_bet(_bet(price="-200", stake=1))
    dog = tracker.add_bet(_bet(price="40c", stake=2, model_prob=0.45))
    push = tracker.add_bet(_bet(price="-110", stake=1))
    tracker.set_result(fav["id"], "won")
    tracker.set_result(dog["id"], "lost")
    tracker.set_result(push["id"], "push")

    bets = {b["id"]: b for b in tracker.list_bets()}
    assert bets[fav["id"]]["profit"] == pytest.approx(0.5)
    assert bets[dog["id"]]["profit"] == pytest.approx(-2.0)
    assert bets[push["id"]]["profit"] == 0.0

    o = tracker.summary()["overall"]
    assert (o["won"], o["lost"], o["push"]) == (1, 1, 1)
    assert o["profit"] == pytest.approx(-1.5)
    assert o["staked"] == pytest.approx(4.0)
    assert o["win_rate"] == pytest.approx(0.5)
    assert o["model_expected_wins"] == pytest.approx(1.15)
    assert o["model_actual_wins"] == 1

    tracker.set_result(fav["id"], "pending")
    assert tracker.summary()["overall"]["pending"] == 1


def test_grade_pending_uses_final_scores(monkeypatch):
    b = tracker.add_bet(_bet(market="totals", side="Under", line=47.5, is_home=None))
    open_game = tracker.add_bet(_bet(game_id="g2"))
    monkeypatch.setattr(tracker, "_nfl_scores", lambda season: {"g1": (24.0, 20.0)})
    out = tracker.grade_pending()
    assert out["graded"] == 1
    bets = {x["id"]: x for x in tracker.list_bets()}
    assert bets[b["id"]]["status"] == "won"
    assert bets[b["id"]]["graded_by"] == "auto"
    assert bets[b["id"]]["profit"] == pytest.approx(0.5)
    assert bets[open_game["id"]]["status"] == "pending"
