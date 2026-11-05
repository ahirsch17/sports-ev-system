"""Home-field detail is walk-forward: a later score cannot move an earlier game."""

import polars as pl

from nfl_edge.features.detail import build_detail_features


def _schedule(week8_margin: int) -> pl.DataFrame:
    # Three home and three road results before week 7, then a week 8 score we edit.
    rows = []
    home_margins = [14, 10, 7]
    road_margins = [-3, -6, -1]
    for i, margin in enumerate(home_margins):
        rows.append(("h" + str(i), 2024, i * 2 + 1, "H", "A" + str(i), 20 + margin, 20))
    for i, margin in enumerate(road_margins):
        rows.append(("a" + str(i), 2024, i * 2 + 2, "R" + str(i), "H", 17 - margin, 17))
    rows.append(("w7", 2024, 7, "H", "G", None, None))
    rows.append(("w8", 2024, 8, "H", "I", 20 + week8_margin, 20))
    return pl.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "season": [r[1] for r in rows],
            "week": [r[2] for r in rows],
            "home_team": [r[3] for r in rows],
            "away_team": [r[4] for r in rows],
            "home_score": [r[5] for r in rows],
            "away_score": [r[6] for r in rows],
            "stadium_id": ["STAD"] * len(rows),
            "location": ["Home"] * len(rows),
        }
    )


def test_later_score_does_not_change_an_earlier_field_edge():
    close = build_detail_features(_schedule(3))
    blowout = build_detail_features(_schedule(28))
    a = float(close.filter(pl.col("game_id") == "w7")["field_margin_adj"][0])
    b = float(blowout.filter(pl.col("game_id") == "w7")["field_margin_adj"][0])
    assert a == b
    assert a > 0.5
