"""College totals follow points scored and allowed, and the board can open a game."""

from cfb_edge.predict import _kickoff
from cfb_edge.scoring import score_game
from cfb_edge.weights import DEFAULT_WEIGHTS


def _score(**kwargs):
    base = dict(
        elo_pred_margin=0.0,
        form_home_diff=0.0,
        form_away_diff=0.0,
        home_rest=7.0,
        away_rest=7.0,
        conference_game=True,
        neutral_site=False,
        rivalry_note=False,
        similar_home=0.0,
        similar_away=0.0,
        weights=DEFAULT_WEIGHTS,
    )
    base.update(kwargs)
    return score_game(**base)


def test_a_high_scoring_pair_is_not_the_same_total_as_a_low_one():
    high = _score(home_for=40, home_against=40, away_for=38, away_against=36)
    low = _score(home_for=18, home_against=17, away_for=16, away_against=15)
    assert high["pred_total"] - low["pred_total"] > 12


def test_equal_point_differential_still_keeps_the_scoring_level():
    shootout = _score(home_for=42, home_against=42, away_for=40, away_against=40)
    rock = _score(home_for=17, home_against=17, away_for=15, away_against=15)
    assert shootout["pred_total"] > rock["pred_total"] + 10


def test_no_history_sits_on_the_college_average():
    blank = _score()
    assert 50 < blank["pred_total"] < 56


def test_neutral_site_drops_the_home_nudge():
    home = _score(neutral_site=False)
    neutral = _score(neutral_site=True)
    assert home["pred_margin"] > neutral["pred_margin"]


def test_the_two_scores_add_up_to_the_total():
    scored = _score(elo_pred_margin=6.0, home_for=31, home_against=22, away_for=24, away_against=27)
    assert abs(
        scored["pred_home_points"] + scored["pred_away_points"] - scored["pred_total"]
    ) < 0.01


def test_kickoff_is_shown_in_eastern_time():
    day, clock = _kickoff("2026-11-04T00:00:00.000Z")
    assert day == "2026-11-03"
    assert clock == "19:00"


def test_the_page_opens_a_college_game_with_school_colors():
    page = open("ui/static/app.js", encoding="utf-8").read()
    colors = open("ui/static/cfb-colors.js", encoding="utf-8").read()
    assert "function renderCfb" in page
    assert "request !== state.request" in page
    assert "CFB_COLORS" in page
    assert "No player props" in page
    for school in ("Alabama", "Ohio State", "LSU", "Hawai'i", "San José State"):
        assert f'"{school}"' in colors
