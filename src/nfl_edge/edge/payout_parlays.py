"""
Parlay builder aimed at solid mid-tier payouts, not chalk, not lottery tickets.

Target fair American odds roughly +250 to +900 (about 2.5x-10x).
Typically 2-4 legs with model probs in the 45-70% band.

Ticket confidence is a weighted blend of joint hit probability, average leg
gate strength, and the weakest leg.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

from nfl_edge.odds_math import combine_independent, implied_to_american


# Sweet-spot payout window (American) — used as default for SGP
DEFAULT_MIN_ODDS = 110  # 2-leg SGPs often land +110–+250 fair
DEFAULT_MAX_ODDS = 900
# Prefer legs that aren't coin-flips or locks (SGP allows slightly chalkier anchors)
MIN_LEG_PROB = 0.48
MAX_LEG_PROB = 0.80
MAX_LEG_PROB_STRICT = 0.72  # hard chalk; ok as SGP anchor with a second market

# Spreads (ATS 47-50%) and totals (~52%) lose to -110 juice in walk-forward audits,
# so they are not allowed as parlay legs until a priced backtest says otherwise.
PARLAY_MARKETS = {"h2h"}

# Same-game parlays are positively correlated; discount independent product.
SGP_CORRELATION_DISCOUNT = 0.88

# Drop tickets whose weakest leg is below this gate strength.
MIN_WEAKEST_LEG = 0.35

# Confidence blend weights
W_JOINT = 0.45
W_AVG_GATE = 0.35
W_WEAKEST = 0.20

# Market preference within a matchup (higher = preferred)
_MARKET_PREF = {"h2h": 3.0, "spreads": 2.0, "totals": 1.2, "player_pass_yds": 1.0, "player_reception_yds": 0.8, "player_rush_yds": 0.6}


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _leg_ok(prob: float, *, min_p: float = MIN_LEG_PROB, max_p: float = MAX_LEG_PROB) -> bool:
    return min_p <= prob <= max_p


def _odds_in_band(american: int, lo: int, hi: int) -> bool:
    return lo <= american <= hi


def _leg_confidence(leg: dict, sport: str) -> float:
    """
    Per-leg confidence from sport-specific gates.
    NFL: |pred_margin| / 10
    MLB: blend of SP ERA gap and run margin
    Falls back to model_prob scaled when margin fields are missing.
    """
    sport_u = (sport or "").upper()
    explicit = leg.get("leg_confidence")
    if explicit is not None:
        return min(1.0, max(0.0, _f(explicit)))

    margin = abs(_f(leg.get("pred_margin")))
    if sport_u == "MLB":
        era_gap = abs(_f(leg.get("sp_era_gap")))
        # Prefer SP gap + margin when present
        if era_gap > 0 or margin > 0:
            return min(1.0, (era_gap / 2.0) * 0.55 + (margin / 1.5) * 0.45)
        # soft fallback from win prob distance from 0.5
        p = _f(leg.get("model_prob"), 0.5)
        return min(1.0, max(0.0, abs(p - 0.5) * 2.0))

    if sport_u == "CFB":
        if margin > 0:
            return min(1.0, margin / 14.0)
        p = _f(leg.get("model_prob"), 0.5)
        return min(1.0, max(0.0, (p - 0.48) / 0.30))

    # NFL default
    if margin > 0:
        from_margin = min(1.0, margin / 10.0)
    else:
        from_margin = 0.0
    p = _f(leg.get("model_prob"), 0.5)
    # Edges often lack pred_margin; map mid-band probs into usable gate scores
    from_prob = min(1.0, max(0.0, (p - 0.48) / 0.28))
    return max(from_margin, from_prob)


def _grade(confidence: float) -> str:
    if confidence >= 0.68:
        return "A"
    if confidence >= 0.58:
        return "B"
    if confidence >= 0.48:
        return "C"
    return "D"


def _ticket_confidence(
    legs: list[dict],
    *,
    sport: str,
    kind: str,
    joint_hit: float,
) -> dict[str, Any]:
    """
    Thorough ticket confidence:
      combined = 0.45*scaled_joint_hit + 0.35*avg_leg_gate + 0.20*weakest_leg
    SGP applies 0.88 correlation discount on the independent product first.
    """
    gates = [_leg_confidence(l, sport) for l in legs]
    avg_gate = sum(gates) / max(len(gates), 1)
    weakest = min(gates) if gates else 0.0

    raw_joint = joint_hit
    if kind == "sgp":
        raw_joint = joint_hit * SGP_CORRELATION_DISCOUNT

    # Scale joint hit into a 0-1 "feel" band. Mid-tier parlays often sit ~0.12-0.28.
    # Map so ~0.22 independent hits near 0.65 after blend with strong legs.
    scaled_joint = min(1.0, max(0.0, raw_joint / 0.35))

    combined = W_JOINT * scaled_joint + W_AVG_GATE * avg_gate + W_WEAKEST * weakest
    combined = min(1.0, max(0.0, combined))
    grade = _grade(combined)

    method = (
        f"0.45*scaled_joint({scaled_joint:.3f}) + 0.35*avg_gate({avg_gate:.3f}) "
        f"+ 0.20*weakest({weakest:.3f})"
        + (f"; SGP x{SGP_CORRELATION_DISCOUNT}" if kind == "sgp" else "")
    )

    if grade == "A":
        advice = "Strong ticket. Confidence-gated legs and a mid payout. Size a half unit."
    elif grade == "B":
        advice = "Solid ticket. Prefer as a secondary play behind singles."
    elif grade == "C":
        advice = "Playable but soft. Smaller unit; skip if any leg price is chalky."
    else:
        advice = "Weak confidence. Prefer singles or pass this combination."

    return {
        "confidence": round(combined, 4),
        "confidence_pct": round(combined * 100, 1),
        "grade": grade,
        "model_hit_prob": round(raw_joint, 4),
        "avg_leg_confidence": round(avg_gate, 4),
        "weakest_leg_confidence": round(weakest, 4),
        "method": method,
        "advice": advice,
        "leg_gates": [round(g, 4) for g in gates],
    }


def _parlay_payload(
    legs: list[dict],
    *,
    sport: str,
    kind: str,
) -> dict[str, Any]:
    probs = [_f(l["model_prob"]) for l in legs]
    fair_p_indep = combine_independent(probs)
    conf = _ticket_confidence(legs, sport=sport, kind=kind, joint_hit=fair_p_indep)
    # Fair odds still use independent product (book SGPs price correlation differently)
    fair_american = implied_to_american(fair_p_indep)
    avg_ev = sum(_f(l.get("ev")) for l in legs) / max(len(legs), 1)
    gates = conf.pop("leg_gates")
    leg_rows = []
    for l, gate in zip(legs, gates):
        leg_rows.append(
            {
                "label": l.get("label"),
                "game": f"{l.get('away_team')} @ {l.get('home_team')}",
                "game_id": str(l.get("game_id") or ""),
                "market": l.get("market"),
                "model_prob": round(_f(l["model_prob"]), 3),
                "ev": round(_f(l.get("ev")), 3),
                "leg_confidence": gate,
                "pred_margin": l.get("pred_margin"),
                "sp_era_gap": l.get("sp_era_gap"),
                "home_team": l.get("home_team"),
                "away_team": l.get("away_team"),
            }
        )
    return {
        "sport": sport,
        "kind": kind,  # sgp | cross
        "n_legs": len(legs),
        "legs": leg_rows,
        "fair_prob": round(fair_p_indep, 4),
        "fair_american": fair_american,
        "payout_hint": _payout_copy(fair_american),
        "avg_leg_ev": round(avg_ev, 3),
        "score": round(
            _score(fair_american, avg_ev, len(legs), conf["confidence"], conf["grade"]),
            3,
        ),
        "note": "Same-game only (Robinhood). Fair odds from model — compare to book SGP price.",
        **conf,
    }


def _payout_copy(american: int) -> str:
    if american >= 0:
        return f"+{american} · $100 -> ${100 + american}"
    return f"{american} · risk ${abs(american)} to win $100"


def _score(
    american: int,
    avg_ev: float,
    n_legs: int,
    confidence: float,
    grade: str,
) -> float:
    """Prefer mid payout + positive EV + strong confidence + A/B/C grades."""
    center = 450
    dist = abs(american - center) / 450
    leg_bonus = 0.15 if n_legs == 3 else (0.08 if n_legs == 4 else 0.0)
    grade_bonus = {"A": 0.35, "B": 0.22, "C": 0.10, "D": -0.25}.get(grade, 0.0)
    return (
        (1.0 - min(dist, 1.0))
        + avg_ev * 2
        + leg_bonus
        + confidence * 1.2
        + grade_bonus
    )


def _leg_rank_key(e: dict) -> float:
    """Prefer gated ML favorites over totals/spreads when ranking pool legs."""
    market = str(e.get("market") or "")
    pref = _MARKET_PREF.get(market, 0.5)
    conf_boost = 1.0 if e.get("high_confidence") else 0.0
    # Mild chalk penalty so heavy favorites don't dominate cross tickets
    p = _f(e.get("model_prob"), 0.5)
    chalk_pen = 0.35 if p >= 0.72 else 0.0
    return pref + conf_boost + _f(e.get("ev")) * 2.0 - chalk_pen


def build_payout_parlays(
    edges: list[dict],
    *,
    sport: str = "NFL",
    min_odds: int = DEFAULT_MIN_ODDS,
    max_odds: int = DEFAULT_MAX_ODDS,
    max_parlays: int = 18,
) -> list[dict]:
    """
    Build same-game parlays only (Robinhood / single-matchup books).
    Never mixes legs across games. Prefers grades A/B/C; skips weak-leg tickets.
    CFB caps at 3 legs to avoid blowout-chalk ladders.
    """
    sport_u = (sport or "NFL").upper()
    max_legs = 3 if sport_u == "CFB" else 3

    pool = [
        e
        for e in edges
        if e.get("model_prob") is not None
        and _leg_ok(_f(e["model_prob"]))
        and (
            e.get("passes_threshold") in (True, "true", "True", 1)
            or _f(e.get("ev")) > 0.02
            or e.get("high_confidence")
        )
        and str(e.get("game_action") or "").upper() != "PASS"
        and str(e.get("market") or "") in PARLAY_MARKETS
    ]
    # One leg per market per game (avoid "NO ML" + "New Orleans Saints ML" doubles)
    best_by: dict[tuple[str, str], dict] = {}
    for e in pool:
        gid = str(e.get("game_id") or f"{e.get('away_team')}_{e.get('home_team')}")
        mkt = str(e.get("market") or "other")
        key = (gid, mkt)
        prev = best_by.get(key)
        if prev is None or _leg_rank_key(e) > _leg_rank_key(prev):
            best_by[key] = e
    pool = list(best_by.values())

    out: list[dict] = []

    # --- Same-game 2-3 legs only (no cross-game; Robinhood constraint) ---
    by_game: dict[str, list[dict]] = {}
    for e in pool:
        gid = str(e.get("game_id") or f"{e.get('away_team')}_{e.get('home_team')}")
        by_game.setdefault(gid, []).append(e)

    for gid, legs_pool in by_game.items():
        legs_pool = sorted(legs_pool, key=_leg_rank_key, reverse=True)[:5]
        if len(legs_pool) < 2:
            continue
        for n in (2, 3):
            if n > max_legs or len(legs_pool) < n:
                continue
            for combo in combinations(legs_pool, n):
                markets = [c.get("market") for c in combo]
                labels = [c.get("label") for c in combo]
                joined = " ".join(str(x) for x in labels)
                if "Over" in joined and "Under" in joined and markets.count("totals") >= 2:
                    continue
                if len(set(labels)) < n:
                    continue
                if all(_f(c.get("model_prob")) > MAX_LEG_PROB_STRICT for c in combo):
                    continue
                payload = _parlay_payload(list(combo), sport=sport, kind="sgp")
                if payload.get("weakest_leg_confidence", 0) < MIN_WEAKEST_LEG:
                    continue
                if _odds_in_band(payload["fair_american"], min_odds, max_odds):
                    out.append(payload)

    # Prefer A/B/C; keep D only to fill if needed (still must clear weakest-leg gate)
    out.sort(key=lambda p: (-p["score"], -p.get("confidence", 0)))
    preferred = [p for p in out if p.get("grade") in {"A", "B", "C"}]
    fallback = [p for p in out if p.get("grade") == "D"]
    ranked = preferred + fallback

    # Cap to 2 SGPs per game so the board stays readable
    uniq_out = []
    seen_legs = set()
    per_game: dict[str, int] = {}
    for p in ranked:
        key = tuple(sorted(str(l["label"]) for l in p["legs"]))
        if key in seen_legs:
            continue
        games = {str(l.get("game") or "") for l in p["legs"]}
        gkey = next(iter(games)) if games else "?"
        if per_game.get(gkey, 0) >= 2:
            continue
        seen_legs.add(key)
        per_game[gkey] = per_game.get(gkey, 0) + 1
        uniq_out.append(p)
        if len(uniq_out) >= max_parlays:
            break
    return uniq_out


def legs_from_confidence_games(games: Iterable[dict], sport: str = "NFL") -> list[dict]:
    """
    Seed ML (+ soft spread when available) from high-confidence favorites so each
    matchup can fuel same-game parlays when prop/total edges also exist.
    """
    sport_u = (sport or "NFL").upper()
    edges: list[dict] = []
    for g in games:
        # Skip finished games and CFB cupcake mismatches for betting fuel
        if g.get("actual_home_score") is not None:
            continue
        if sport_u == "CFB" and g.get("is_cupcake"):
            continue
        high = bool(g.get("high_confidence"))
        margin = _f(g.get("pred_margin"))
        abs_m = abs(margin)
        if sport_u == "NFL":
            if not high and abs_m < 5.0:
                continue
        elif sport_u == "CFB":
            if not high and abs_m < 10.0:
                continue
        else:
            era_gap = abs(_f(g.get("sp_era_gap")))
            if not high and not (era_gap >= 1.0 and abs_m >= 0.5):
                continue

        home = g.get("home_team")
        away = g.get("away_team")
        hw = _f(g.get("home_win_prob"), 0.5)
        favorite = home if margin > 0 else away
        fav_prob = hw if margin > 0 else (1.0 - hw)
        if not favorite:
            continue
        # Tight band: avoid chalk traps (>0.72) and coin flips
        if not _leg_ok(fav_prob, min_p=0.52, max_p=0.72):
            continue

        gid = str(g.get("game_id") or g.get("game_pk") or f"{away}_{home}")
        if sport_u == "NFL":
            leg_conf = min(1.0, abs_m / 10.0)
        elif sport_u == "CFB":
            leg_conf = min(1.0, abs_m / 14.0)
        else:
            leg_conf = min(1.0, (_f(g.get("sp_era_gap")) / 2.0) * 0.55 + (abs_m / 1.5) * 0.45)
        edges.append(
            {
                "label": f"{favorite} ML",
                "market": "h2h",
                "model_prob": fav_prob,
                "ev": fav_prob - 0.52,
                "game_id": gid,
                "home_team": home,
                "away_team": away,
                "passes_threshold": True,
                "high_confidence": True,
                "pred_margin": margin,
                "sp_era_gap": g.get("sp_era_gap"),
                "leg_confidence": round(leg_conf, 4),
            }
        )
    return edges


def legs_from_mlb_games(games: Iterable[dict], *, high_conf_only: bool = True) -> list[dict]:
    """Flatten MLB suggested bets into edge-like dicts for the parlay builder."""
    edges = []
    for g in games:
        if high_conf_only and not g.get("high_confidence"):
            continue
        for b in g.get("suggested_bets") or []:
            edges.append(
                {
                    **b,
                    "game_id": str(g.get("game_pk") or f"{g.get('away_team')}_{g.get('home_team')}"),
                    "home_team": g.get("home_team"),
                    "away_team": g.get("away_team"),
                    "high_confidence": bool(g.get("high_confidence")),
                    "pred_margin": g.get("pred_margin"),
                    "sp_era_gap": g.get("sp_era_gap"),
                }
            )
        # always seed core model legs even if EV filter missed
        home = g.get("home_team")
        away = g.get("away_team")
        hw = _f(g.get("home_win_prob"), 0.5)
        total = _f(g.get("pred_total"), 8.5)
        line = round(total * 2) / 2
        from nfl_edge.odds_math import prob_over_line

        p_over = prob_over_line(total, 3.2, line)
        seeds = [
            {"label": f"{home} ML", "market": "h2h", "model_prob": hw, "ev": hw - 0.52},
            {"label": f"{away} ML", "market": "h2h", "model_prob": 1 - hw, "ev": (1 - hw) - 0.52},
            {"label": f"Over {line}", "market": "totals", "model_prob": p_over, "ev": p_over - 0.52},
            {"label": f"Under {line}", "market": "totals", "model_prob": 1 - p_over, "ev": (1 - p_over) - 0.52},
        ]
        for s in seeds:
            if _leg_ok(_f(s["model_prob"])):
                edges.append(
                    {
                        **s,
                        "game_id": str(g.get("game_pk")),
                        "home_team": home,
                        "away_team": away,
                        "passes_threshold": True,
                        "high_confidence": True,
                        "pred_margin": g.get("pred_margin"),
                        "sp_era_gap": g.get("sp_era_gap"),
                    }
                )
    # Also add confidence-seeded favorites
    edges.extend(legs_from_confidence_games(games, sport="MLB"))
    return edges
