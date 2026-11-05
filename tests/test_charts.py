"""The game figures are real SVG, from the numbers we pass in."""

import polars as pl

from ui.charts import payout_chart, recent_points, score_chart, trend_chart


def test_score_chart_is_an_svg_with_both_teams():
    svg = score_chart(away="ARI", home="SEA", our_away=19, our_home=28, book_away=17, book_home=28)
    assert svg.startswith("<svg")
    assert "ARI" in svg and "SEA" in svg


def test_trend_and_payout_charts_draw():
    trend = trend_chart("ARI", "SEA", [10, 17, 24], [20, 14, 31])
    payout = payout_chart(label="Two bets", payout=264, floor=224)
    assert trend.startswith("<svg") and "ARI" in trend
    assert payout.startswith("<svg") and "+264" in payout and "+224" in payout


def test_trend_chart_uses_the_team_colors():
    trend = trend_chart("ARI", "SEA", [10, 17], [20, 14], away_color="#97233F", home_color="#002244")
    assert "#97233F" in trend and "#002244" in trend


def test_team_colors_are_saved_for_every_club():
    colors = open("ui/static/team-colors.js", encoding="utf-8").read()
    page = open("ui/static/app.js", encoding="utf-8").read()
    for abbr in ("BAL", "ATL", "CIN", "SEA", "KC"):
        assert f"{abbr}:" in colors
    assert "function paint" in page
    assert "Neutral site" in page
    assert "teamStyle" in page


def test_recent_points_stop_before_this_week():
    schedules = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [7, 8, 9],
            "home_team": ["SEA", "CHI", "SEA"],
            "away_team": ["KC", "SEA", "ARI"],
            "home_score": [20, 34, 99],
            "away_score": [3, 17, 99],
        }
    )
    assert recent_points(schedules, "SEA", 2026, 9) == [20, 17]
