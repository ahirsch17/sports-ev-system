"""Compare model fair prices to DraftKings lines and build correlated parlays."""

from __future__ import annotations

import re
from typing import Any

import polars as pl

from nfl_edge.features.weather import load_weights
from nfl_edge.models.game import cover_prob, total_side_probs
from nfl_edge.models.props import prop_over_prob
from nfl_edge.odds_math import american_to_implied, expected_value, implied_to_american


# Map nflverse abbreviations / Odds API names loosely
TEAM_ALIASES = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Oakland Raiders": "LV",
    "San Diego Chargers": "LAC",
    "St. Louis Rams": "LA",
}


def normalize_team(name: str) -> str:
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    # already an abbreviation?
    if len(name) <= 3:
        return name.upper().replace("JAC", "JAX").replace("WSH", "WAS")
    # fuzzy: last word
    for full, abbr in TEAM_ALIASES.items():
        if name.lower() in full.lower() or full.lower() in name.lower():
            return abbr
    return name


def _pred_lookup(predictions: pl.DataFrame) -> dict[tuple[str, str], dict]:
    out = {}
    for r in predictions.iter_rows(named=True):
        key = (normalize_team(str(r["home_team"])), normalize_team(str(r["away_team"])))
        out[key] = r
        # also by game_id
        out[("id", str(r["game_id"]))] = r  # type: ignore
    return out


def find_prediction(predictions: pl.DataFrame, home: str, away: str) -> dict | None:
    h, a = normalize_team(home), normalize_team(away)
    for r in predictions.iter_rows(named=True):
        if normalize_team(str(r["home_team"])) == h and normalize_team(str(r["away_team"])) == a:
            return r
    return None


def compute_main_edges(
    predictions: pl.DataFrame,
    odds_rows: list[dict[str, Any]],
    weights: dict | None = None,
) -> pl.DataFrame:
    weights = weights or load_weights()
    thr = weights.get("edge_thresholds", {})
    margin_sigma = float(weights.get("margin_sigma", 13.5))
    total_sigma = float(weights.get("total_sigma", 10.0))
    conf_min = float((weights.get("confidence") or {}).get("min_abs_margin", 5.0))

    edges: list[dict] = []
    # group odds by event
    by_event: dict[str, list[dict]] = {}
    for row in odds_rows:
        by_event.setdefault(str(row["event_id"]), []).append(row)

    for event_id, rows in by_event.items():
        home = rows[0]["home_team"]
        away = rows[0]["away_team"]
        pred = find_prediction(predictions, home, away)
        if not pred:
            continue
        high_conf = abs(float(pred.get("pred_margin") or 0)) >= conf_min

        for row in rows:
            market = row["market"]
            price = row.get("price")
            if price is None:
                continue
            point = row.get("point")
            selection = row.get("selection")
            model_prob = None
            min_ev = 0.05
            label = ""

            if market == "h2h":
                min_ev = float(thr.get("h2h_min_ev", 0.05))
                p_home = float(pred["home_win_prob"])
                sel_abbr = normalize_team(str(selection))
                if sel_abbr == normalize_team(home):
                    model_prob = p_home
                    label = f"{home} ML"
                elif sel_abbr == normalize_team(away):
                    model_prob = 1 - p_home
                    label = f"{away} ML"
                else:
                    continue

            elif market == "spreads" and point is not None:
                min_ev = float(thr.get("spreads_min_ev", 0.05))
                sel_abbr = normalize_team(str(selection))
                if sel_abbr == normalize_team(home):
                    model_prob = cover_prob(float(pred["pred_margin"]), float(point), margin_sigma)
                    label = f"{home} {point:+g}"
                elif sel_abbr == normalize_team(away):
                    # away spread: equivalent home spread is -point for away line
                    # If away is +3.5, home spread is -3.5; P(away covers)=1-P(home covers at -3.5)
                    home_spread = -float(point)
                    model_prob = 1 - cover_prob(float(pred["pred_margin"]), home_spread, margin_sigma)
                    label = f"{away} {point:+g}"
                else:
                    continue

            elif market == "totals" and point is not None:
                min_ev = float(thr.get("totals_min_ev", 0.04))
                p_over, p_under = total_side_probs(float(pred["pred_total"]), float(point), total_sigma)
                if str(selection).lower().startswith("over"):
                    model_prob = p_over
                    label = f"Over {point}"
                else:
                    model_prob = p_under
                    label = f"Under {point}"
            else:
                continue

            ev = expected_value(model_prob, float(price))
            fair = implied_to_american(model_prob)
            # Prefer high-confidence games for live tips
            adj_min = min_ev * (0.8 if high_conf else 1.25)
            edges.append(
                {
                    "event_id": event_id,
                    "game_id": pred["game_id"],
                    "home_team": home,
                    "away_team": away,
                    "market": market,
                    "label": label,
                    "selection": selection,
                    "point": point,
                    "dk_price": price,
                    "model_prob": round(model_prob, 4),
                    "fair_american": fair,
                    "dk_implied": round(american_to_implied(float(price)), 4),
                    "ev": round(ev, 4),
                    "passes_threshold": ev >= adj_min,
                    "high_confidence": high_conf,
                    "priority": {"totals": 2, "spreads": 3, "h2h": 3}.get(market, 9),
                    "correlation_group": f"game:{pred['game_id']}",
                    "weather_tag": _weather_tag(pred),
                }
            )

    if not edges:
        return pl.DataFrame()
    return pl.DataFrame(edges).sort(["passes_threshold", "ev"], descending=True)


def compute_prop_edges(
    prop_projections: pl.DataFrame,
    prop_odds: list[dict[str, Any]],
    weights: dict | None = None,
) -> pl.DataFrame:
    weights = weights or load_weights()
    min_ev = float(weights.get("edge_thresholds", {}).get("props_min_ev", 0.03))
    edges: list[dict] = []

    if prop_projections.is_empty() or not prop_odds:
        return pl.DataFrame()

    # index projections by normalized player last name + market
    proj_rows = list(prop_projections.iter_rows(named=True))

    def match_prop(player: str | None, market: str) -> dict | None:
        if not player:
            return None
        plen = _last_name(player)
        for p in proj_rows:
            if p["market"] != market:
                continue
            if plen and _last_name(str(p["player"])) == plen:
                return p
            if player.lower() in str(p["player"]).lower() or str(p["player"]).lower() in player.lower():
                return p
        return None

    for row in prop_odds:
        market = row.get("market")
        if market not in {"player_pass_yds", "player_rush_yds", "player_reception_yds"}:
            continue
        proj = match_prop(row.get("player"), market)
        if not proj or row.get("point") is None or row.get("price") is None:
            continue
        p_over = prop_over_prob(float(proj["mu"]), float(proj["sigma"]), float(row["point"]))
        selection = str(row.get("selection") or "Over")
        model_prob = p_over if selection.lower().startswith("o") else 1 - p_over
        ev = expected_value(model_prob, float(row["price"]))
        edges.append(
            {
                "event_id": row.get("event_id"),
                "game_id": proj["game_id"],
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "market": market,
                "label": f"{row.get('player')} {selection} {row.get('point')} {market.replace('player_', '')}",
                "selection": selection,
                "player": row.get("player"),
                "point": row.get("point"),
                "dk_price": row.get("price"),
                "model_mu": proj["mu"],
                "model_prob": round(model_prob, 4),
                "fair_american": implied_to_american(model_prob),
                "dk_implied": round(american_to_implied(float(row["price"])), 4),
                "ev": round(ev, 4),
                "passes_threshold": ev >= min_ev,
                "priority": 1,
                "correlation_group": f"game:{proj['game_id']}",
                "weather_tag": "",
            }
        )

    if not edges:
        return pl.DataFrame()
    return pl.DataFrame(edges).sort(["passes_threshold", "ev"], descending=True)


def _last_name(name: str) -> str:
    parts = re.split(r"\s+", name.strip())
    return parts[-1].lower() if parts else ""


def _weather_tag(pred: dict) -> str:
    wind = pred.get("wind")
    roof = str(pred.get("roof") or "")
    tags = []
    if roof.lower() in {"outdoors", "open", "outdoor"} and wind is not None and float(wind) >= 15:
        tags.append(f"wind{int(float(wind))}")
    return ",".join(tags)


# Correlation hints for SGP construction
POSITIVE_CORR = {
    ("totals:Under", "player_pass_yds:Under"),
    ("totals:Under", "player_rush_yds:Over"),
    ("player_pass_yds:Under", "player_rush_yds:Over"),
    ("totals:Over", "player_pass_yds:Over"),
    # main-market building blocks (soft positive / narrative SGPs)
    ("totals:Under", "spreads"),
    ("totals:Over", "spreads"),
    ("totals:Under", "h2h"),
    ("totals:Over", "h2h"),
}


def _corr_key(edge: dict) -> str:
    market = edge.get("market", "")
    label = str(edge.get("label", ""))
    if market == "totals":
        side = "Over" if "Over" in label else "Under"
        return f"totals:{side}"
    if market.startswith("player_"):
        side = "Over" if str(edge.get("selection", "")).lower().startswith("o") else "Under"
        return f"{market}:{side}"
    if market == "spreads":
        return "spreads"
    if market == "h2h":
        return "h2h"
    return market


def edges_are_correlated(a: dict, b: dict) -> bool:
    if a.get("game_id") != b.get("game_id"):
        return False
    # never pair opposite sides of the same market family
    ka, kb = _corr_key(a), _corr_key(b)
    if ka == kb:
        return False
    if {ka, kb} == {"totals:Over", "totals:Under"}:
        return False
    if (ka, kb) in POSITIVE_CORR or (kb, ka) in POSITIVE_CORR:
        return True
    # same-game total + moneyline/spread is allowed when both pass EV
    markets = {a.get("market"), b.get("market")}
    if "totals" in markets and markets & {"spreads", "h2h"}:
        return True
    return False


def build_parlays(edges: pl.DataFrame, *, max_legs: int = 3, max_parlays: int = 10) -> list[dict]:
    """Greedy 2–3 leg SGPs from +EV correlated edges only."""
    if edges.is_empty():
        return []
    good = edges.filter(pl.col("passes_threshold") == True)  # noqa: E712
    if good.is_empty():
        good = edges.head(20)
    rows = [r for r in good.iter_rows(named=True)]
    # prefer props then totals
    rows.sort(key=lambda r: (r.get("priority", 9), -float(r.get("ev") or 0)))

    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(str(r.get("game_id")), []).append(r)

    parlays: list[dict] = []
    for game_id, legs_pool in by_game.items():
        # find correlated pairs / triples
        n = len(legs_pool)
        for i in range(n):
            for j in range(i + 1, n):
                if not edges_are_correlated(legs_pool[i], legs_pool[j]):
                    continue
                pair = [legs_pool[i], legs_pool[j]]
                fair_p = 1.0
                for leg in pair:
                    fair_p *= float(leg["model_prob"])
                parlays.append(
                    {
                        "game_id": game_id,
                        "legs": [leg["label"] for leg in pair],
                        "n_legs": 2,
                        "fair_prob": round(fair_p, 4),
                        "fair_american": implied_to_american(fair_p),
                        "combined_model_ev_proxy": round(sum(float(l["ev"]) for l in pair), 4),
                        "note": "SGP juice not modeled — require extra edge vs DK SGP price",
                    }
                )
                # try add third correlated leg
                for k in range(j + 1, n):
                    if edges_are_correlated(legs_pool[i], legs_pool[k]) or edges_are_correlated(legs_pool[j], legs_pool[k]):
                        triple = pair + [legs_pool[k]]
                        if len({leg["label"] for leg in triple}) < 3:
                            continue
                        fair3 = fair_p * float(legs_pool[k]["model_prob"])
                        parlays.append(
                            {
                                "game_id": game_id,
                                "legs": [leg["label"] for leg in triple],
                                "n_legs": 3,
                                "fair_prob": round(fair3, 4),
                                "fair_american": implied_to_american(fair3),
                                "combined_model_ev_proxy": round(sum(float(l["ev"]) for l in triple), 4),
                                "note": "SGP juice not modeled — require extra edge vs DK SGP price",
                            }
                        )
                        if len(parlays) >= max_parlays:
                            return parlays[:max_parlays]
        if len(parlays) >= max_parlays:
            break

    parlays.sort(key=lambda p: -p["combined_model_ev_proxy"])
    return parlays[:max_parlays]
