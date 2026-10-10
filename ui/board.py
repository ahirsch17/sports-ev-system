"""Per-game betting board cards for the UI."""

from __future__ import annotations

from typing import Any

from nfl_edge.odds_math import implied_to_american


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _norm_matchup(away: str, home: str) -> str:
    return f"{away} @ {home}"


def _market_label(market: str) -> str:
    return {
        "player_pass_yds": "pass yds",
        "player_rush_yds": "rush yds",
        "player_reception_yds": "recv yds",
    }.get(market, market.replace("player_", "").replace("_", " "))


MIN_CARD_EV = 0.03
# Walk-forward audits: ATS 47-50% and totals ~52% at the gates, both below -110 break-even.
# Only the winner market is staked until a priced backtest proves otherwise.
STAKED_MARKETS = {"h2h"}
UNPROVEN_NOTE = "info only (not profitable in backtest)"


def _truthy(x: Any) -> bool:
    return x in (True, 1, "true", "True")


def _best_side(edges: list[dict], market: str) -> dict | None:
    """Side of a market the model rates most likely at the book's own line."""
    rows = [e for e in edges if e.get("market") == market and e.get("model_prob") is not None]
    return max(rows, key=lambda e: _f(e.get("model_prob")), default=None)


def _has_edge(edge: dict) -> bool:
    return _f(edge.get("ev")) >= MIN_CARD_EV and _truthy(edge.get("passes_threshold"))


def _edge_stake(edge: dict, full: str = "0.5u") -> str:
    if edge.get("market") not in STAKED_MARKETS:
        return UNPROVEN_NOTE
    if edge.get("dk_price") is None:
        return "no book price"
    return full if _has_edge(edge) else "skip (no edge)"


def _price_instruction(adv: dict, action: str, favorite: str, fav_prob: float, ml_edge: dict | None) -> tuple[str, str]:
    """Winner advice that respects the book price. Returns (action, instruction)."""
    base = adv.get("instruction") or adv.get("detail") or ""
    if action not in {"BET", "LEAN"} or str(base).upper().startswith("SKIP"):
        return action, base
    fair = implied_to_american(fav_prob)
    fair_txt = f"{fair:+d}"
    if ml_edge is None or ml_edge.get("dk_price") is None:
        return action, f"{favorite} to win ({fav_prob:.0%}). Only bet the moneyline at {fair_txt} or better."
    price = int(_f(ml_edge.get("dk_price")))
    if _has_edge(ml_edge):
        return action, f"TAKE: {favorite} moneyline at {price:+d} (model {fav_prob:.0%}, fair {fair_txt})."
    if _f(ml_edge.get("ev")) > 0:
        return (
            "WAIT",
            f"{favorite} should win ({fav_prob:.0%}). {price:+d} is barely better than fair ({fair_txt}), "
            "too thin an edge to bet.",
        )
    return (
        "WAIT",
        f"{favorite} should win ({fav_prob:.0%}), but {price:+d} costs more than it's worth. "
        f"No bet unless the price gets to {fair_txt} or better.",
    )


def _balanced_props(props: list[dict], *, limit: int = 10) -> list[dict]:
    """Prefer each side's pass + top recv + rush so one team doesn't wipe the list."""
    if not props:
        return []
    by_team: dict[str, list[dict]] = {}
    for p in props:
        by_team.setdefault(str(p.get("team") or "?"), []).append(p)
    picked: list[dict] = []
    seen: set[tuple] = set()

    def take(rows: list[dict], market: str, n: int) -> None:
        mil = [p for p in rows if str(p.get("market")) == market]
        mil.sort(key=lambda p: _f(p.get("mu")), reverse=True)
        for p in mil[:n]:
            key = (p.get("player"), p.get("market"))
            if key in seen:
                continue
            seen.add(key)
            picked.append(p)

    for rows in by_team.values():
        take(rows, "player_pass_yds", 1)
    for rows in by_team.values():
        take(rows, "player_reception_yds", 2)
    for rows in by_team.values():
        take(rows, "player_rush_yds", 1)
    for p in sorted(props, key=lambda x: _f(x.get("mu")), reverse=True):
        if len(picked) >= limit:
            break
        key = (p.get("player"), p.get("market"))
        if key in seen:
            continue
        seen.add(key)
        picked.append(p)
    return picked[:limit]


def build_game_board(
    *,
    sport: str,
    games: list[dict],
    advice: dict,
    market_edges: list[dict] | None = None,
    parlays: list[dict] | None = None,
) -> list[dict]:
    """
    One card per upcoming game:
      winner / spread / total with fair odds
      player props
      1-2 recommended parlays that touch this game
    """
    advice_by_id = {str(g.get("game_id")): g for g in (advice.get("games") or [])}
    # also index by matchup
    for g in advice.get("games") or []:
        advice_by_id[str(g.get("matchup") or "")] = g

    prop_by_game: dict[str, list] = {}
    for p in advice.get("prop_plays") or []:
        prop_by_game.setdefault(str(p.get("game_id")), []).append(p)

    edges_by_game: dict[str, list] = {}
    for e in market_edges or []:
        edges_by_game.setdefault(str(e.get("game_id") or e.get("event_id") or ""), []).append(e)

    parlays = parlays or advice.get("best_parlays") or []

    cards: list[dict] = []
    for g in games:
        if g.get("actual_home_score") is not None or g.get("home_score") is not None:
            continue
        gid = str(g.get("game_id") or g.get("game_pk") or f"{g.get('away_team')}_{g.get('home_team')}")
        away = str(g.get("away_team") or "")
        home = str(g.get("home_team") or "")
        matchup = _norm_matchup(away, home)
        adv = advice_by_id.get(gid) or advice_by_id.get(matchup) or {}

        margin = _f(g.get("pred_margin"), _f(adv.get("pred_margin")))
        total = _f(g.get("pred_total"), _f(adv.get("pred_total"), 44.0))
        home_win = _f(g.get("home_win_prob"), 0.5)
        away_win = 1.0 - home_win
        favorite = home if margin >= 0 else away
        dog = away if margin >= 0 else home
        fav_prob = home_win if margin >= 0 else away_win
        action = adv.get("action") or ("BET" if g.get("high_confidence") else "PASS")

        # Book lines: for each market, the side the model likes at the book's line/price
        medges = edges_by_game.get(gid) or []
        ml_edge = _best_side(medges, "h2h")
        if ml_edge is not None and _f(ml_edge.get("model_prob")) < 0.5:
            ml_edge = None
        spread_edge = _best_side(medges, "spreads")
        total_edge = _best_side(medges, "totals")

        home_spread = round(-margin, 1)
        fav_spread = round(-abs(margin), 1)
        winner_fair = implied_to_american(fav_prob)

        if action == "BET":
            winner_stake = "1.0u" if fav_prob < 0.72 else "0.5u"
        elif action == "LEAN":
            winner_stake = "0.25u"
        else:
            winner_stake = "skip"
        if ml_edge is not None and ml_edge.get("dk_price") is not None and not _has_edge(ml_edge):
            winner_stake = "skip (price too short)"

        if spread_edge is not None:
            spread_row = {
                "pick": spread_edge.get("selection") or favorite,
                "is_home": (
                    spread_edge.get("selection") == spread_edge.get("home_team")
                    if spread_edge.get("selection")
                    else favorite == home
                ),
                "line": _f(spread_edge.get("point")),
                "label": spread_edge.get("label"),
                "prob": round(_f(spread_edge.get("model_prob")), 4),
                "fair_american": spread_edge.get("fair_american")
                or implied_to_american(_f(spread_edge.get("model_prob"))),
                "dk_price": spread_edge.get("dk_price"),
                "model_line": fav_spread,
                "stake": _edge_stake(spread_edge),
                "from_book": True,
            }
        else:
            spread_row = {
                "pick": favorite,
                "is_home": favorite == home,
                "line": fav_spread,
                "label": f"{favorite} {fav_spread:+g}",
                "prob": None,
                "fair_american": None,
                "dk_price": None,
                "model_line": fav_spread,
                "stake": "no book line",
                "from_book": False,
            }

        if total_edge is not None:
            total_row = {
                "side": total_edge.get("selection") or ("Over" if total >= _f(total_edge.get("point")) else "Under"),
                "model_total": round(total, 1),
                "line": round(_f(total_edge.get("point")), 1),
                "prob": round(_f(total_edge.get("model_prob")), 4),
                "fair_american": total_edge.get("fair_american")
                or implied_to_american(_f(total_edge.get("model_prob"))),
                "dk_price": total_edge.get("dk_price"),
                "stake": _edge_stake(total_edge),
                "from_book": True,
            }
        else:
            total_row = {
                "side": None,
                "model_total": round(total, 1),
                "line": None,
                "prob": None,
                "fair_american": None,
                "dk_price": None,
                "stake": "no book line",
                "from_book": False,
            }

        action, instruction = _price_instruction(adv, action, favorite, fav_prob, ml_edge)
        if action == "WAIT":
            winner_stake = "skip (price too short)"

        priced = [
            e
            for e in (ml_edge, spread_edge, total_edge)
            if e is not None and e.get("dk_price") is not None and e.get("market") in STAKED_MARKETS
        ]
        best_edge = max((e for e in priced if _has_edge(e)), key=lambda e: _f(e.get("ev")), default=None)
        if action not in {"BET", "LEAN"} or not instruction.startswith("TAKE"):
            best_edge = None
        best_play = (
            {
                "label": best_edge.get("label"),
                "market": best_edge.get("market"),
                "prob": round(_f(best_edge.get("model_prob")), 4),
                "dk_price": best_edge.get("dk_price"),
                "ev": round(_f(best_edge.get("ev")), 3),
            }
            if best_edge is not None
            else None
        )

        # Props: merge advised plays + raw slate props, balanced across both teams
        advised = prop_by_game.get(gid) or []
        raw = g.get("player_props") or []
        # Normalize advised rows into the same shape
        merged: list[dict] = []
        seen_prop: set[tuple] = set()
        for src in (advised, raw):
            for p in src:
                player = p.get("player")
                market = p.get("market")
                key = (player, market)
                if not player or key in seen_prop:
                    continue
                seen_prop.add(key)
                mu = _f(p.get("mu"))
                sigma = max(_f(p.get("sigma"), 40.0), 20.0)
                buffer = max(10.0, sigma * 0.28)
                merged.append(
                    {
                        "player": player,
                        "team": p.get("team"),
                        "market": market,
                        "market_label": p.get("market_label") or _market_label(str(market or "")),
                        "mu": mu,
                        "stake": p.get("stake") or "0.25u",
                        "verb": p.get("verb") or "COMPARE",
                        "instruction": p.get("instruction")
                        or (
                            f"COMPARE DK {player} {_market_label(str(market or ''))}. "
                            f"Model {mu:.0f}. Over if line <= {mu - buffer:.0f}, Under if >= {mu + buffer:.0f}."
                        ),
                        "over_if_line_at_or_below": p.get("over_if_line_at_or_below")
                        if p.get("over_if_line_at_or_below") is not None
                        else round(mu - buffer, 1),
                        "under_if_line_at_or_above": p.get("under_if_line_at_or_above")
                        if p.get("under_if_line_at_or_above") is not None
                        else round(mu + buffer, 1),
                    }
                )
        props = _balanced_props(merged, limit=10)

        # Same-game parlays only for this matchup (Robinhood / no cross-game)
        related = []
        for p in parlays:
            if str(p.get("kind") or "").lower() == "cross":
                continue
            legs = p.get("legs") or []
            if not legs:
                continue
            # Prefer game_id match; fall back to team tokens (abbrev or full name)
            gids = {str(l.get("game_id") or "") for l in legs if l.get("game_id")}
            if gids and gid not in gids:
                continue
            if not gids:
                games_on_ticket = {str(l.get("game") or "") for l in legs if l.get("game")}
                if len(games_on_ticket) > 1:
                    continue
                same_game = all(
                    (
                        away in str(l.get("game") or "")
                        and home in str(l.get("game") or "")
                    )
                    or (
                        str(l.get("away_team") or "") in (away, str(l.get("away_team") or ""))
                        and str(l.get("home_team") or "") in (home, str(l.get("home_team") or ""))
                        and (
                            str(l.get("away_team") or "") == away
                            or str(l.get("home_team") or "") == home
                            or away in str(l.get("game") or "")
                            or home in str(l.get("game") or "")
                        )
                    )
                    for l in legs
                )
                # Token match: edges often use full team names while board uses abbrev
                if not same_game:
                    blob = " ".join(
                        f"{l.get('game') or ''} {l.get('away_team') or ''} {l.get('home_team') or ''} {l.get('label') or ''}"
                        for l in legs
                    )
                    same_game = away in blob and home in blob
                if not same_game:
                    continue
            related.append(
                {
                    "kind": "sgp",
                    "n_legs": p.get("n_legs"),
                    "fair_american": p.get("fair_american"),
                    "grade": p.get("grade"),
                    "confidence_pct": p.get("confidence_pct") or p.get("confidence"),
                    "advice": p.get("advice"),
                    "payout_hint": p.get("payout_hint"),
                    "risk": p.get("risk"),
                    "legs": [
                        {
                            "label": l.get("label"),
                            "game": l.get("game") or matchup,
                            "game_id": l.get("game_id") or gid,
                            "model_prob": l.get("model_prob"),
                        }
                        for l in legs
                    ],
                }
            )
            if len(related) >= 2:
                break

        cards.append(
            {
                "game_id": gid,
                "sport": sport.upper(),
                "matchup": matchup,
                "away_team": away,
                "home_team": home,
                "gameday": g.get("gameday"),
                "action": action,
                "instruction": instruction,
                "detail": adv.get("detail") or "",
                "confidence_score": adv.get("confidence_score"),
                "reasons": (adv.get("reasons") or [])[:2],
                "warnings": (adv.get("warnings") or [])[:2],
                "inj": {
                    "home": g.get("home_inj_note"),
                    "away": g.get("away_inj_note"),
                },
                "winner": {
                    "pick": favorite,
                    "is_home": favorite == home,
                    "dog": dog,
                    "prob": round(fav_prob, 4),
                    "fair_american": winner_fair,
                    "dk_price": ml_edge.get("dk_price") if ml_edge is not None else None,
                    "stake": winner_stake,
                },
                "spread": {**spread_row, "home_line": home_spread},
                "total": total_row,
                "best_play": best_play,
                "props": props,
                "parlays": related,
                "rank_score": _f(adv.get("rank_score")),
            }
        )

    order = {"BET": 0, "LEAN": 1, "WAIT": 2, "PASS": 3}
    cards.sort(key=lambda c: (order.get(c["action"], 9), -_f(c.get("rank_score")), -_f(c.get("confidence_score"))))
    return cards
