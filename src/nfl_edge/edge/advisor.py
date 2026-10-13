"""
Best-bet advisor: rank every game/edge/parlay scenario and emit clear actions.

Policy (from multi-season walk-forward audit 2022-2025):
  NFL: BET when |pred_margin| >= 7 (~74% winner); LEAN at >= 5 (~72%).
       Skip favorite if QB1 out/DNP. Chalk ML cap ~0.72 (prefer spread).
       Dog-QB-out slice is strong but small-n; boost lean only.
  MLB: prefer SP ERA gap >= 1.0 AND |pred_margin| >= 0.5  (~65%)
  CFB: prefer |pred_margin| >= 10 (~80%+ on mismatches; talent gaps larger)
  Always prefer singles over same-game parlays when confidence is mixed.
  Never recommend cross-game parlays (Robinhood / single-matchup books).
  Never recommend opposite sides of the same market.
  Pass when data is thin (missing SP, tiny margin, coin-flip probs).
"""

from __future__ import annotations

from typing import Any


NFL_CONF_MARGIN = 7.0  # primary BET gate (audit ~74%)
NFL_LEAN_MARGIN = 5.0  # secondary LEAN gate (audit ~72%)
MLB_CONF_MARGIN = 0.5
MLB_SP_GAP = 1.0
CFB_CONF_MARGIN = 10.0

# Probability bands for actionable singles
MIN_SINGLE_PROB = 0.52
MAX_SINGLE_PROB = 0.78  # above this, price is usually chalk - still ok if +EV
MIN_EV_HIGH = 0.03
MIN_EV_LOW = 0.06  # stricter when not high-confidence

# Parlay keep threshold
MIN_PARLAY_CONF = 0.48


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _strip_em(text: str) -> str:
    """Replace em dashes and similar with commas / hyphens."""
    if not text:
        return text
    return (
        text.replace("\u2014", ",")
        .replace("\u2013", "-")
        .replace("—", ",")
        .replace("–", "-")
    )


def _game_key(g: dict) -> str:
    return str(g.get("game_id") or g.get("game_pk") or f"{g.get('away_team')}_{g.get('home_team')}")


def _nfl_game_advice(g: dict, conf_min: float = NFL_CONF_MARGIN, lean_min: float = NFL_LEAN_MARGIN) -> dict[str, Any]:
    margin = _f(g.get("pred_margin"))
    total = _f(g.get("pred_total"), 44.0)
    home_win = _f(g.get("home_win_prob"), 0.5)
    abs_m = abs(margin)
    high = bool(g.get("high_confidence")) or abs_m >= conf_min
    favorite = g.get("home_team") if margin > 0 else g.get("away_team")
    dog = g.get("away_team") if margin > 0 else g.get("home_team")
    fav_prob = home_win if margin > 0 else (1.0 - home_win)
    fav_qb_out = _f(g.get("home_qb_out") if margin > 0 else g.get("away_qb_out"))
    dog_qb_out = _f(g.get("away_qb_out") if margin > 0 else g.get("home_qb_out"))
    fav_inj = str(g.get("home_inj_note") if margin > 0 else g.get("away_inj_note") or "")
    nm_note = str(g.get("near_miss_note") or "")

    reasons: list[str] = []
    warnings: list[str] = []

    if abs_m >= conf_min:
        reasons.append(f"Primary BET gate (|margin| {abs_m:.1f} >= {conf_min:g}). Audit ~74%.")
    elif abs_m >= lean_min:
        reasons.append(f"LEAN gate (|margin| {abs_m:.1f} >= {lean_min:g}). Audit ~72%.")
    elif abs_m >= 3:
        reasons.append(f"Modest lean ({margin:+.1f}); usable only as a soft single, not a parlay anchor.")
        warnings.append("Below primary confidence gate. Size down or skip.")
    else:
        warnings.append(f"Coin-flip margin ({margin:+.1f}). PASS moneyline.")

    if fav_prob >= 0.72:
        warnings.append("Heavy chalk. ML price may be poor even if correct side.")
    if g.get("deep_net_margin_adj") is not None and abs(_f(g.get("deep_net_margin_adj"))) >= 2.5:
        reasons.append(f"Deep matchup adj {_f(g.get('deep_net_margin_adj')):+.1f}")
    for note in (g.get("home_inj_note"), g.get("away_inj_note")):
        if note and str(note) not in {"n/a", "None"} and "healthy-ish" not in str(note):
            reasons.append(str(note)[:100])
    if nm_note:
        reasons.append(nm_note[:100])

    # Favorite missing QB1: do not treat as a confidence BET
    if fav_qb_out >= 0.5:
        warnings.append(f"Favorite QB1 out/DNP. {fav_inj}. Demote or skip ML.")
        high = False
    if dog_qb_out >= 0.5:
        reasons.append("Dog missing QB1 (audit ~83% small-n). Helps favorite side.")

    if high and abs_m >= conf_min and fav_prob < 0.78 and fav_qb_out < 0.5:
        action = "BET"
        primary = f"{favorite} ML / side"
        detail = f"Take {favorite} (model {fav_prob:.0%}, margin {margin:+.1f}). Prefer spread if ML is crushed."
        if fav_prob >= 0.72:
            instruction = f"TAKE: {favorite} spread · stake 0.5u · SKIP ML (chalk)"
        else:
            instruction = f"TAKE: {favorite} moneyline · stake 1.0u · skip if juice is worse than fair"
    elif fav_qb_out >= 0.5 and abs_m >= lean_min:
        action = "LEAN"
        primary = f"{dog} lean / fade backup QB"
        detail = (
            f"Model still likes {favorite} ({fav_prob:.0%}) but QB1 is out. "
            f"Do not hammer {favorite} ML. Lean {dog} or skip."
        )
        instruction = (
            f"SKIP {favorite} ML (QB1 out). OPTIONAL: {dog} ML/+spread at 0.25u if price is good. Else PASS."
        )
    elif abs_m >= lean_min and fav_qb_out < 0.5:
        action = "LEAN"
        primary = f"{favorite} lean / spread"
        stake = "0.5u" if dog_qb_out >= 0.5 else "0.25u"
        detail = (
            f"Lean {favorite} (margin {margin:+.1f}). "
            + ("Dog QB out boosts this lean. " if dog_qb_out >= 0.5 else "")
            + "Only at a good price; not a full BET gate."
        )
        instruction = (
            f"OPTIONAL: {favorite} {'spread' if fav_prob >= 0.72 else 'ML'} at {stake} "
            f"if price beats fair. Else SKIP."
        )
    elif abs_m >= 3:
        action = "LEAN"
        primary = f"{favorite} lean"
        detail = f"Soft lean {favorite}; only bet if you have a clear +EV price vs model."
        instruction = f"OPTIONAL only: {favorite} at 0.25u if price beats model. Else SKIP"
    else:
        action = "PASS"
        primary = "No side"
        detail = "Skip winner markets; totals only if weather/injury create a clean over/under edge."
        instruction = "SKIP this game. No side clears the gate."

    return {
        "game_id": _game_key(g),
        "sport": "NFL",
        "matchup": f"{g.get('away_team')} @ {g.get('home_team')}",
        "action": action,
        "primary_bet": primary,
        "instruction": instruction,
        "detail": detail,
        "favorite": favorite,
        "dog": dog,
        "pred_margin": round(margin, 2),
        "pred_total": round(total, 2),
        "fav_win_prob": round(fav_prob, 4),
        "high_confidence": high and fav_qb_out < 0.5,
        "confidence_score": round(min(1.0, abs_m / 10.0) * (0.55 if fav_qb_out >= 0.5 else 1.0), 3),
        "reasons": reasons,
        "warnings": warnings,
        "rank_score": _rank_score(
            action, high and fav_qb_out < 0.5, abs_m / 10.0, fav_prob, -1.5 if fav_qb_out >= 0.5 else (0.4 if dog_qb_out >= 0.5 else 0.0)
        ),
    }


def _cfb_game_advice(g: dict, conf_min: float = CFB_CONF_MARGIN) -> dict[str, Any]:
    margin = _f(g.get("pred_margin"))
    total = _f(g.get("pred_total"), 52.0)
    home_win = _f(g.get("home_win_prob"), 0.5)
    abs_m = abs(margin)
    high = bool(g.get("high_confidence")) or abs_m >= conf_min
    favorite = g.get("home_team") if margin > 0 else g.get("away_team")
    dog = g.get("away_team") if margin > 0 else g.get("home_team")
    fav_prob = home_win if margin > 0 else (1.0 - home_win)

    reasons: list[str] = []
    warnings: list[str] = []

    if abs_m >= 14:
        reasons.append(f"Large CFB talent gap ({margin:+.1f} pts). Audit >=14pt slice is strongest.")
    elif abs_m >= conf_min:
        reasons.append(f"Confidence gate hit (|margin| {abs_m:.1f} >= {conf_min:g}).")
    elif abs_m >= 7:
        reasons.append(f"Modest lean ({margin:+.1f}); soft single only, not a parlay anchor.")
        warnings.append("Below primary CFB confidence gate. Size down or skip.")
    else:
        warnings.append(f"Coin-flip / small margin ({margin:+.1f}). PASS moneyline.")

    if g.get("is_cupcake"):
        warnings.append("FCS/cupcake opponent. Do not bet; Elo updates are shrunk.")
        high = False
    if fav_prob >= 0.72:
        warnings.append("Heavy chalk. ML price may be poor even if correct side. Prefer spread or skip.")
    if g.get("neutral_site"):
        reasons.append("Neutral site: HFA removed.")
    if g.get("conference_game"):
        reasons.append("Conference game.")
    warnings.append("CFB injury feeds are sparse; model does not use injuries.")

    if g.get("is_cupcake"):
        action = "PASS"
        primary = "No side"
        detail = "Cupcake / FCS mismatch. Skip betting; prediction only."
        instruction = "SKIP. Cupcake / FCS mismatch. Do not bet."
    elif high and abs_m >= conf_min and fav_prob < 0.78:
        action = "BET"
        primary = f"{favorite} ML / side"
        detail = (
            f"Take {favorite} (model {fav_prob:.0%}, margin {margin:+.1f}). "
            f"Prefer spread if ML is crushed chalk."
        )
        if fav_prob >= 0.72:
            instruction = f"TAKE: {favorite} spread · stake 0.5u · SKIP ML (chalk)"
        else:
            instruction = f"TAKE: {favorite} moneyline · stake 1.0u · or spread if ML is chalk"
    elif high and abs_m >= conf_min:
        action = "LEAN"
        primary = f"{favorite} lean / spread"
        detail = (
            f"Model likes {favorite} ({fav_prob:.0%}) but ML is chalk. "
            f"Only bet a spread or skip."
        )
        instruction = f"SKIP ML. OPTIONAL: {favorite} spread at 0.5u max"
    elif abs_m >= 7:
        action = "LEAN"
        primary = f"{favorite} lean"
        detail = f"Soft lean {favorite}; only bet if you have a clear +EV price vs model."
        instruction = f"OPTIONAL only: {favorite} at 0.25u if +EV. Else SKIP"
    else:
        action = "PASS"
        primary = "No side"
        detail = "Skip winner markets on this CFB game."
        instruction = "SKIP this game. No side clears the CFB gate."

    return {
        "game_id": _game_key(g),
        "sport": "CFB",
        "matchup": f"{g.get('away_team')} @ {g.get('home_team')}",
        "action": action,
        "primary_bet": primary,
        "instruction": instruction,
        "detail": detail,
        "favorite": favorite,
        "dog": dog,
        "pred_margin": round(margin, 2),
        "pred_total": round(total, 2),
        "fav_win_prob": round(fav_prob, 4),
        "high_confidence": high,
        "confidence_score": round(min(1.0, abs_m / 14.0), 3),
        "reasons": [_strip_em(x) for x in reasons],
        "warnings": [_strip_em(x) for x in warnings],
        "rank_score": _rank_score(action, high, abs_m / 14.0, fav_prob, 0.0),
    }


def _mlb_game_advice(
    g: dict,
    *,
    conf_margin: float = MLB_CONF_MARGIN,
    sp_gap_min: float = MLB_SP_GAP,
) -> dict[str, Any]:
    margin = _f(g.get("pred_margin"))
    total = _f(g.get("pred_total"), 8.5)
    home_win = _f(g.get("home_win_prob"), 0.5)
    abs_m = abs(margin)
    home_sp = g.get("home_sp_name")
    away_sp = g.get("away_sp_name")
    home_era = g.get("home_sp_era")
    away_era = g.get("away_sp_era")
    era_gap = _f(g.get("sp_era_gap"))
    if era_gap <= 0 and home_era is not None and away_era is not None:
        era_gap = abs(_f(home_era) - _f(away_era))

    sp_known = bool(home_sp and away_sp and home_era is not None and away_era is not None)
    high = bool(g.get("high_confidence")) or (sp_known and era_gap >= sp_gap_min and abs_m >= conf_margin)

    favorite = g.get("home_team") if margin > 0 else g.get("away_team")
    dog = g.get("away_team") if margin > 0 else g.get("home_team")
    fav_prob = home_win if margin > 0 else (1.0 - home_win)
    fav_sp = home_sp if margin > 0 else away_sp
    dog_sp = away_sp if margin > 0 else home_sp

    reasons: list[str] = []
    warnings: list[str] = []

    if not sp_known:
        warnings.append("Starting pitchers incomplete. WAIT or PASS until both SPs confirmed.")
    elif era_gap >= 1.5:
        reasons.append(f"Large SP ERA gap ({era_gap:.2f}). Strongest MLB signal in audit (~63%).")
    elif era_gap >= sp_gap_min:
        reasons.append(f"SP ERA gap {era_gap:.2f} >= {sp_gap_min:g}. Audit SP-gap slice ~61%.")
    else:
        warnings.append(f"SP quality close (ERA gap {era_gap:.2f}). Thin edge.")

    if abs_m >= conf_margin:
        reasons.append(f"Run margin {margin:+.2f} clears gate ({conf_margin:g})")
    else:
        warnings.append(f"Narrow projected margin ({margin:+.2f})")

    if high and sp_known:
        action = "BET"
        primary = f"{favorite} ML"
        detail = (
            f"Bet {favorite} behind {fav_sp or 'SP'} "
            f"(vs {dog_sp or 'opp SP'}; margin {margin:+.2f}, home win {home_win:.0%})."
        )
        instruction = f"TAKE: {favorite} ML · stake 1.0u · behind {fav_sp or 'SP'}"
    elif sp_known and era_gap >= sp_gap_min:
        action = "LEAN"
        primary = f"{favorite} lean"
        detail = f"SP edge favors {favorite}, but margin is soft. Only at a good ML price."
        instruction = f"OPTIONAL: {favorite} ML at 0.25u only if price beats fair. Else SKIP"
    elif not sp_known:
        action = "WAIT"
        primary = "Wait for SPs"
        detail = "Do not bet this game until both starters are posted and confirmed."
        instruction = "WAIT. Do not bet until both starting pitchers are confirmed."
    else:
        action = "PASS"
        primary = "No side"
        detail = "Skip ML; consider totals only if park/weather strongly push over/under."
        instruction = "SKIP ML. No SP + margin gate clear."

    return {
        "game_id": _game_key(g),
        "sport": "MLB",
        "matchup": f"{g.get('away_team')} @ {g.get('home_team')}",
        "action": action,
        "primary_bet": primary,
        "instruction": instruction,
        "detail": detail,
        "favorite": favorite,
        "dog": dog,
        "pred_margin": round(margin, 2),
        "pred_total": round(total, 2),
        "fav_win_prob": round(fav_prob, 4),
        "sp_era_gap": round(era_gap, 2),
        "home_sp": home_sp,
        "away_sp": away_sp,
        "high_confidence": high,
        "confidence_score": round(
            min(1.0, (era_gap / 2.0) * 0.55 + (abs_m / 1.5) * 0.45) if sp_known else 0.1,
            3,
        ),
        "reasons": reasons,
        "warnings": warnings,
        "rank_score": _rank_score(action, high, abs_m / 1.5 if sp_known else 0.0, fav_prob, era_gap / 2.0),
    }


def _rank_score(action: str, high: bool, strength: float, fav_prob: float, bonus: float) -> float:
    base = {"BET": 3.0, "LEAN": 1.2, "WAIT": 0.3, "PASS": 0.0}.get(action, 0.0)
    chalk_pen = 0.25 if fav_prob > 0.75 else 0.0
    return round(base + (0.8 if high else 0.0) + strength + bonus - chalk_pen, 4)


_MARKET_LABEL = {
    "player_pass_yds": "pass yds",
    "player_rush_yds": "rush yds",
    "player_reception_yds": "recv yds",
}


def _prop_row(p, player, label, mu, sigma, roll, edge, over_line, under_line, quality, action, verb, instruction, conf_note, gate, market) -> dict:
    return {
        "player": player,
        "team": p.get("team"),
        "opponent": p.get("opponent"),
        "game_id": p.get("game_id"),
        "market": market,
        "market_label": label,
        "mu": mu,
        "sigma": round(sigma, 1),
        "roll": roll,
        "edge_vs_roll": edge,
        "over_if_line_at_or_below": over_line,
        "under_if_line_at_or_above": under_line,
        "quality": quality,
        "action": action,
        "verb": verb,
        "instruction": instruction,
        "why": conf_note,
        "stake": "0.5u" if action == "BET" else ("0.25u" if action == "LEAN" else "0u"),
        "role": p.get("role"),
        "lean": p.get("lean"),
        "rank_score": round(
            (3.0 if market == "player_pass_yds" else (2.2 if market == "player_reception_yds" else 1.0))
            + min(roll, 300) / 100.0
            + abs(edge) / 12.0
            + (1.0 if abs(edge) >= gate else 0.0)
            + (2.0 if p.get("lean") in {"featured", "under"} else 0.0)
            - _f(p.get("priority"), 2) * 0.05,
            4,
        ),
    }


def _prop_plays(props: list[dict], weights: dict) -> list[dict]:
    """Rank player props into clear TAKE / COMPARE / SKIP instructions (full-game only)."""
    pw = weights.get("props") or {}
    pass_min = _f(pw.get("pass_min_edge_vs_roll"), 20.0)
    rush_min = _f(pw.get("rush_min_edge_vs_roll"), 12.0)
    recv_min = _f(pw.get("recv_min_edge_vs_roll"), 15.0)

    plays: list[dict] = []
    for p in props or []:
        market = str(p.get("market") or "")
        if market not in _MARKET_LABEL:
            continue
        mu = _f(p.get("mu"))
        sigma = _f(p.get("sigma"), 40.0)
        roll = _f(p.get("roll"))
        edge = _f(p.get("edge_vs_roll"))
        abs_edge = abs(edge)
        player = str(p.get("player") or "")
        label = _MARKET_LABEL[market]
        # Buffer vs DK line: half a sigma, floored
        buffer = max(12.0 if market == "player_pass_yds" else 8.0, sigma * 0.28)
        over_line = round(mu - buffer, 1)
        under_line = round(mu + buffer, 1)

        if market == "player_pass_yds":
            gate = pass_min
            # Prefer real volume QBs; ignore backup noise with low roll
            if roll < 140:
                continue
            quality = "BEST" if abs_edge >= pass_min else ("GOOD" if abs_edge >= pass_min * 0.6 else "WATCH")
            conf_note = "Pass props preferred (audit ~69% when |mu-roll|>=20)."
        elif market == "player_reception_yds":
            gate = recv_min
            role = p.get("role")
            if role == "secondary" and p.get("lean") != "under":
                continue
            if role == "featured":
                quality = "GOOD"
                conf_note = "Only playmaker on this team. Teammates are not a second receiving bet."
                verb = "COMPARE"
                action = "LEAN"
                instruction = (
                    f"{player} is the only playmaker. Model {mu:.0f} recv yds. "
                    f"TAKE Over if the line is <= {over_line:.0f}. "
                    f"Do not bet a teammate as a second playmaker."
                )
                plays.append(_prop_row(p, player, label, mu, sigma, roll, edge, over_line, under_line, quality, action, verb, instruction, conf_note, gate, market))
                continue
            if p.get("lean") == "under":
                season = _f(p.get("season_ypg"))
                quality = "GOOD"
                conf_note = "Targets fell off. The season yardage rate is not the projection."
                verb = "UNDER"
                action = "LEAN"
                instruction = (
                    f"UNDER {player} recv yds. Model {mu:.0f} after the target drop"
                    f"{f' (season rate {season:.0f})' if season else ''}. Not a playmaker."
                )
                plays.append(_prop_row(p, player, label, mu, sigma, roll, edge, over_line, under_line, quality, action, verb, instruction, conf_note, gate, market))
                continue
            if roll < 35:
                continue
            quality = "GOOD" if abs_edge >= recv_min else "WATCH"
            conf_note = "Recv yds noisier than pass. Size down."
        else:
            gate = rush_min
            if roll < 40:
                continue
            quality = "LEAN" if abs_edge >= rush_min else "WATCH"
            conf_note = "Rush props weaker in audit. Prefer pass/recv."

        # Always surface starter-level pass/recv for DK line compare; skip weak rush noise
        if abs_edge < gate * 0.35 and market == "player_rush_yds":
            continue

        if abs_edge >= gate or market in {"player_pass_yds", "player_reception_yds"}:
            verb = "COMPARE"
            action = "BET" if market == "player_pass_yds" and abs_edge >= gate else (
                "LEAN" if market != "player_rush_yds" or abs_edge >= gate else "PASS"
            )
            if abs_edge < gate * 0.5 and market == "player_reception_yds":
                action = "LEAN"
            instruction = (
                f"COMPARE DK {player} {label}. Model {mu:.0f} (fair line). "
                f"TAKE Over if line <= {over_line:.0f}. TAKE Under if line >= {under_line:.0f}. "
                f"Else SKIP. Stake {'0.5u' if action == 'BET' else '0.25u'}."
            )
        else:
            verb = "WATCH"
            action = "PASS"
            instruction = (
                f"WATCH {player} {label} (model {mu:.0f}, near roll {roll:.0f}). "
                f"Only bet if DK is clearly off those numbers. Else SKIP."
            )

        plays.append(
            _prop_row(p, player, label, mu, sigma, roll, edge, over_line, under_line, quality, action, verb, instruction, conf_note, gate, market)
        )

    plays.sort(key=lambda x: -_f(x.get("rank_score")))
    # Diversify: top pass, recv, rush separately so rush/recv are not drowned out
    by_mkt: dict[str, list[dict]] = {
        "player_pass_yds": [],
        "player_reception_yds": [],
        "player_rush_yds": [],
    }
    seen: set[str] = set()
    playmaker_teams: set[str] = set()
    for p in plays:
        key = f"{p['player']}:{p['market']}"
        if key in seen:
            continue
        if p["market"] == "player_reception_yds" and p.get("verb") != "UNDER":
            team = str(p.get("team") or "")
            if team in playmaker_teams:
                continue
            playmaker_teams.add(team)
        seen.add(key)
        m = p["market"]
        if m in by_mkt and len(by_mkt[m]) < 6:
            by_mkt[m].append(p)

    uniq = by_mkt["player_pass_yds"][:5] + by_mkt["player_reception_yds"][:5] + by_mkt["player_rush_yds"][:3]
    uniq.sort(key=lambda x: -_f(x.get("rank_score")))
    return uniq[:14]


def _totals_plays(edges: list[dict], games: list[dict]) -> list[dict]:
    """Full-game totals only. Skip wild model/book gaps (calibration). No halves."""
    by_id = {str(g.get("game_id")): g for g in games or []}
    plays: list[dict] = []
    for e in edges or []:
        if e.get("market") != "totals":
            continue
        if not e.get("passes_threshold"):
            continue
        line = _f(e.get("point"))
        gid = str(e.get("game_id") or "")
        g = by_id.get(gid) or {}
        pred = _f(g.get("pred_total"), line)
        gap = abs(pred - line)
        p = _f(e.get("model_prob"))
        # Require modest disagreement only — huge gaps are usually model error
        if gap < 2.5 or gap > 6.5:
            continue
        if p < 0.55 or p > 0.68:
            continue
        side = str(e.get("selection") or "")
        away = e.get("away_team") or g.get("away_team")
        home = e.get("home_team") or g.get("home_team")
        matchup = f"{away} @ {home}"
        instruction = (
            f"TAKE {side} {line:g} (full game) · stake 0.5u · "
            f"model total {pred:.1f} vs line {line:g}. SKIP 1H/2H."
        )
        plays.append(
            {
                "matchup": matchup,
                "game_id": gid,
                "side": side,
                "line": line,
                "pred_total": round(pred, 1),
                "gap": round(gap, 1),
                "model_prob": p,
                "label": e.get("label"),
                "verb": "TAKE",
                "action": "BET",
                "instruction": instruction,
                "stake": "0.5u",
                "quality": "GOOD",
                "rank_score": round(p + (0.15 if e.get("high_confidence") else 0.0), 4),
            }
        )
    plays.sort(key=lambda x: -_f(x.get("rank_score")))
    # One total per game
    seen: set[str] = set()
    uniq: list[dict] = []
    for p in plays:
        if p["game_id"] in seen:
            continue
        seen.add(p["game_id"])
        uniq.append(p)
        if len(uniq) >= 6:
            break
    return uniq


def _edge_advice(edge: dict, sport: str, game_advice_by_id: dict[str, dict]) -> dict[str, Any] | None:
    """Filter + annotate a single edge into actionable advice (or drop it)."""
    from nfl_edge.edge.calculator import normalize_team

    prob = _f(edge.get("model_prob"), 0.5)
    ev = _f(edge.get("ev"))
    market = str(edge.get("market") or "")
    gid = str(edge.get("game_id") or "")
    home = normalize_team(str(edge.get("home_team") or ""))
    away = normalize_team(str(edge.get("away_team") or ""))
    ga = (
        game_advice_by_id.get(gid)
        or game_advice_by_id.get(f"{away}_{home}")
        or game_advice_by_id.get(home)
        or {}
    )
    high = bool(edge.get("high_confidence") or ga.get("high_confidence"))
    game_action = ga.get("action") or "PASS"

    if not (MIN_SINGLE_PROB <= prob <= 0.92):
        return None
    if market == "h2h" and game_action in {"PASS", "WAIT"}:
        return None
    if market in {"totals", "spreads"} and game_action == "PASS":
        return None
    if market in {"totals", "spreads"} and game_action == "WAIT":
        return None
    # Totals on soft games: require BET action + high conf
    if market == "totals" and game_action != "BET":
        return None

    min_ev = MIN_EV_HIGH if high else MIN_EV_LOW
    if market == "h2h" and high and game_action == "BET" and prob >= 0.55:
        min_ev = min(min_ev, 0.01)
    if ev < min_ev and not (high and market == "h2h" and game_action == "BET"):
        return None

    label = str(edge.get("label") or "")
    fav = str(ga.get("favorite") or "")
    if market == "h2h" and fav:
        fav_ok = fav in label or normalize_team(label.replace(" ML", "").strip()) == fav
        if not fav_ok and game_action in {"BET", "LEAN"}:
            return None

    chalk = market == "h2h" and prob >= 0.72
    if market == "h2h" and high and game_action == "BET" and not chalk:
        quality = "BEST"
    elif market == "h2h" and high and game_action in {"BET", "LEAN"} and chalk:
        quality = "GOOD"  # chalk ML: still show, but not BEST
    elif high and game_action in {"BET", "LEAN"} and market in {"h2h", "spreads"}:
        quality = "GOOD"
    elif high and game_action == "BET" and market == "totals" and ev >= 0.04:
        quality = "GOOD"
    elif ev >= 0.06 and game_action == "LEAN":
        quality = "SPECULATIVE"
    else:
        quality = "SPECULATIVE"

    if quality == "SPECULATIVE":
        if sport == "MLB" and ev < 0.08:
            return None
        if sport == "CFB" and (ev < 0.07 or game_action != "LEAN"):
            return None
        if sport == "NFL" and (ev < 0.07 or game_action != "LEAN"):
            return None

    return {
        **edge,
        "quality": quality,
        "action": "BET" if quality in {"BEST", "GOOD"} else "LEAN",
        "advice": _edge_copy(edge, quality, ga, chalk=chalk),
        "high_confidence": high,
        "game_action": game_action,
        "chalk": chalk,
        "unit": {"BEST": "1.0 unit", "GOOD": "0.5 unit", "SPECULATIVE": "0.25 unit"}.get(quality, "0"),
        "rank_score": round(
            {"BEST": 8, "GOOD": 4, "SPECULATIVE": 1}[quality]
            + (2.5 if market == "h2h" else (1.2 if market == "spreads" else 0.4))
            + ev * 6
            + (0.8 if high else 0)
            + _f(ga.get("confidence_score")) * 2.0
            - (1.5 if chalk else 0),
            4,
        ),
    }


def _edge_copy(edge: dict, quality: str, ga: dict, *, chalk: bool = False) -> str:
    label = edge.get("label")
    matchup = f"{edge.get('away_team')} @ {edge.get('home_team')}"
    chalk_note = " CHALK warning: price may be poor; prefer spread." if chalk else ""
    if quality == "BEST":
        return f"Primary play: {label} ({matchup}). Audit-backed confidence gate.{chalk_note}"
    if quality == "GOOD":
        return f"Solid add: {label}. Use as single or one parlay leg max.{chalk_note}"
    return f"Speculative only: {label}. Skip unless price is clearly wrong."


def _parlay_advice(parlays: list[dict], game_advice: list[dict]) -> list[dict]:
    """
    Same-game parlays only (Robinhood). Keep tickets that clear confidence >= 0.48,
    grade A/B/C, and weakest leg >= 0.45. Empty is better than forcing weak chalk.
    """
    bet_matchups = {g["matchup"] for g in game_advice if g.get("action") == "BET"}
    pass_matchups = {g["matchup"] for g in game_advice if g.get("action") == "PASS"}
    scored: list[dict] = []
    for p in parlays:
        legs = p.get("legs") or []
        if not legs:
            continue
        # Robinhood: never recommend cross-game tickets
        if str(p.get("kind") or "").lower() == "cross":
            continue
        games = [str(l.get("game") or "") for l in legs]
        if len({g for g in games if g}) > 1:
            continue
        conf = _f(p.get("confidence"))
        grade = str(p.get("grade") or "").upper()
        weakest = _f(p.get("weakest_leg_confidence"), 1.0)
        if weakest < 0.35:
            continue
        keep = conf >= MIN_PARLAY_CONF or grade in {"A", "B", "C"}
        if not keep:
            continue

        # Drop tickets that include a PASS-game leg (esp. totals on soft games)
        if any(g in pass_matchups for g in games):
            has_total = any(str(l.get("market") or "") == "totals" for l in legs)
            if has_total or not any(g in bet_matchups for g in games):
                continue

        n_bet = sum(1 for g in games if g in bet_matchups)
        frac = n_bet / max(len(legs), 1)
        risk = "LOW" if frac >= 0.75 and len(legs) <= 3 else ("MED" if frac >= 0.5 else "HIGH")

        fair = p.get("fair_american")
        fair_txt = f"{int(fair):+d}" if fair is not None else "n/a"
        conf_pct = p.get("confidence_pct")
        if conf_pct is None:
            conf_pct = round(conf * 100, 1)
        base_advice = p.get("advice") or ""
        advice = (
            f"{'Best SGP' if risk == 'LOW' else 'Secondary SGP'}: "
            f"{p.get('n_legs')}-leg same-game at fair {fair_txt} "
            f"(grade {grade or '?'}, conf {conf_pct:g}%). "
            f"{base_advice}"
        )
        advice = _strip_em(advice)

        scored.append(
            {
                **p,
                "confidence_pct": conf_pct,
                "risk": risk,
                "bet_leg_frac": round(frac, 2),
                "advice": advice,
                "rank_score": round(
                    _f(p.get("score"))
                    + _f(p.get("confidence")) * 2.0
                    + ({"A": 1.2, "B": 0.8, "C": 0.4, "D": 0.0}.get(grade, 0.0))
                    + (1.0 if risk == "LOW" else (0.3 if risk == "MED" else 0.0))
                    + frac * 0.5,
                    4,
                ),
            }
        )

    scored.sort(key=lambda x: -_f(x.get("rank_score")))
    return scored[:12]


def build_advice(
    *,
    sport: str,
    games: list[dict],
    edges: list[dict],
    parlays: list[dict],
    weights: dict | None = None,
    props: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Full slate advice package for UI / API.

    Scenarios covered:
      BET: confidence gate cleared; primary recommendation
      LEAN: directional signal, bet only at good price
      WAIT: missing critical data (MLB SPs)
      PASS: no edge; skip
    """
    weights = weights or {}
    conf = weights.get("confidence") or {}
    props = props or []

    sport_u = sport.upper()
    if sport_u == "NFL":
        conf_min = _f(conf.get("min_abs_margin"), NFL_CONF_MARGIN)
        lean_min = _f(conf.get("bet_margin"), NFL_LEAN_MARGIN)
        game_rows = [_nfl_game_advice(g, conf_min=conf_min, lean_min=lean_min) for g in games]
    elif sport_u == "CFB":
        conf_min = _f(conf.get("min_abs_margin"), CFB_CONF_MARGIN)
        game_rows = [_cfb_game_advice(g, conf_min=conf_min) for g in games]
    else:
        conf_min = _f(conf.get("min_abs_margin"), MLB_CONF_MARGIN)
        sp_gap = _f(conf.get("min_sp_era_gap"), MLB_SP_GAP)
        game_rows = [
            _mlb_game_advice(g, conf_margin=conf_min, sp_gap_min=sp_gap) for g in games
        ]

    # Dedupe doubleheaders / duplicate schedule rows. Keep highest rank_score.
    deduped: dict[str, dict] = {}
    for g in game_rows:
        mkey = g["matchup"]
        prev = deduped.get(mkey)
        if prev is None or _f(g.get("rank_score")) > _f(prev.get("rank_score")):
            deduped[mkey] = g
    game_rows = sorted(deduped.values(), key=lambda g: -_f(g.get("rank_score")))
    by_id = {g["game_id"]: g for g in game_rows}
    for g in game_rows:
        parts = g["matchup"].split(" @ ")
        if len(parts) == 2:
            by_id[f"{parts[0]}_{parts[1]}"] = g
            by_id[parts[1]] = g  # home abbr fallback

    edge_rows = []
    for e in edges:
        advised = _edge_advice(e, sport.upper(), by_id)
        if advised:
            edge_rows.append(advised)
    edge_rows.sort(key=lambda e: -_f(e.get("rank_score")))

    parlay_rows = _parlay_advice(parlays, game_rows)

    bets = [g for g in game_rows if g["action"] == "BET"]
    leans = [g for g in game_rows if g["action"] == "LEAN"]
    waits = [g for g in game_rows if g["action"] == "WAIT"]
    passes = [g for g in game_rows if g["action"] == "PASS"]

    best_singles = [e for e in edge_rows if e.get("quality") in {"BEST", "GOOD"}][:10]

    prop_plays = _prop_plays(props, weights) if sport_u == "NFL" else []
    totals_plays = _totals_plays(edges, games) if sport_u in {"NFL", "CFB"} else []

    # Always surface confidence-gated ML favorites even if book edges missed them
    have_labels = {str(e.get("label")) for e in best_singles}
    for g in bets:
        label = f"{g['favorite']} ML"
        if label in have_labels:
            continue
        parts = g["matchup"].split(" @ ")
        chalk = _f(g.get("fav_win_prob")) >= 0.72
        synth = {
            "label": label,
            "market": "h2h",
            "home_team": parts[-1] if parts else g.get("favorite"),
            "away_team": parts[0] if len(parts) == 2 else g.get("dog"),
            "model_prob": g["fav_win_prob"],
            "ev": None,
            "fair_american": None,
            "quality": "GOOD" if chalk else "BEST",
            "action": "BET",
            "advice": (
                f"{g['detail']} CHALK: prefer spread or skip ML."
                if chalk
                else g["detail"]
            ),
            "high_confidence": True,
            "chalk": chalk,
            "rank_score": _f(g["rank_score"]) + (7.0 if chalk else 9.0),
            "reasons": g.get("reasons") or [],
            "warnings": g.get("warnings") or [],
            "confidence_score": g.get("confidence_score"),
            "unit": "0.5 unit" if chalk else "1.0 unit",
        }
        best_singles.append(synth)
        have_labels.add(label)
    best_singles.sort(key=lambda e: -_f(e.get("rank_score")))
    from nfl_edge.edge.calculator import normalize_team

    uniq_singles: list[dict] = []
    seen_side: set[str] = set()
    for e in best_singles:
        label = str(e.get("label") or "")
        side = label.replace(" ML", "").strip()
        if e.get("market") == "h2h" or label.endswith(" ML"):
            key = f"h2h:{normalize_team(side)}"
        else:
            key = f"{e.get('market')}:{label}:{e.get('home_team')}:{e.get('away_team')}"
        if key in seen_side:
            continue
        seen_side.add(key)
        uniq_singles.append(e)
        if len(uniq_singles) >= 8:
            break
    best_singles = uniq_singles

    headline = _strip_em(_headline(sport, bets, leans, waits, best_singles, parlay_rows))

    # Plain-language playbook for the UI
    playbook_steps: list[dict] = []
    take_lines: list[str] = []
    for i, g in enumerate(bets[:6], start=1):
        chalk = _f(g.get("fav_win_prob")) >= 0.72
        stake = "0.5u" if chalk else "1.0u"
        market = "spread (ML is chalk)" if chalk else "moneyline"
        pick = str(g.get("favorite") or g.get("primary_bet") or "").replace(" ML / side", "").replace(" ML", "")
        line = f"{i}. TAKE {pick} {market} · stake {stake} · {g.get('matchup')}"
        if chalk:
            line += " · CHALK"
        take_lines.append(line)
        playbook_steps.append(
            {
                "n": i,
                "verb": "TAKE",
                "pick": f"{pick} {market}",
                "matchup": g.get("matchup"),
                "stake": stake,
                "why": g.get("instruction") or g.get("detail"),
                "chalk": chalk,
                "action": "BET",
            }
        )
    for g in leans[:3]:
        pick = str(g.get("favorite") or "").strip()
        playbook_steps.append(
            {
                "n": len(playbook_steps) + 1,
                "verb": "OPTIONAL",
                "pick": f"{pick} lean (only if price is good)",
                "matchup": g.get("matchup"),
                "stake": "0.25u or skip",
                "why": g.get("instruction") or g.get("detail"),
                "chalk": False,
                "action": "LEAN",
            }
        )
    for g in waits[:4]:
        playbook_steps.append(
            {
                "n": len(playbook_steps) + 1,
                "verb": "WAIT",
                "pick": "No bet yet",
                "matchup": g.get("matchup"),
                "stake": "0u",
                "why": g.get("instruction") or g.get("detail"),
                "chalk": False,
                "action": "WAIT",
            }
        )

    if parlay_rows:
        top_p = parlay_rows[0]
        legs = ", ".join(str(l.get("label")) for l in (top_p.get("legs") or [])[:4])
        playbook_steps.append(
            {
                "n": len(playbook_steps) + 1,
                "verb": "OPTIONAL SGP",
                "pick": legs or f"{top_p.get('n_legs')}-leg same-game",
                "matchup": f"fair {top_p.get('fair_american'):+d} · grade {top_p.get('grade')} · conf {top_p.get('confidence_pct')}%",
                "stake": "0.25u to 0.5u max",
                "why": top_p.get("advice") or "Same-game only (Robinhood). Place after singles. Compare fair to book SGP.",
                "chalk": False,
                "action": "PARLAY",
            }
        )

    # Player props + full-game totals (NFL). Halves are never advised.
    for p in prop_plays[:5]:
        if p.get("verb") == "WATCH":
            continue
        playbook_steps.append(
            {
                "n": len(playbook_steps) + 1,
                "verb": p.get("verb") or "COMPARE",
                "pick": f"{p.get('player')} {p.get('market_label')}",
                "matchup": f"{p.get('team')} vs {p.get('opponent')} · model {p.get('mu')}",
                "stake": p.get("stake") or "0.5u",
                "why": p.get("instruction"),
                "chalk": False,
                "action": p.get("action") or "LEAN",
            }
        )
    for t in totals_plays[:3]:
        playbook_steps.append(
            {
                "n": len(playbook_steps) + 1,
                "verb": "TAKE",
                "pick": f"{t.get('side')} {t.get('line')} (full game)",
                "matchup": t.get("matchup"),
                "stake": t.get("stake") or "0.5u",
                "why": t.get("instruction"),
                "chalk": False,
                "action": "BET",
            }
        )
    if sport_u == "NFL":
        playbook_steps.append(
            {
                "n": len(playbook_steps) + 1,
                "verb": "SKIP",
                "pick": "1H / 2H / half totals / half props",
                "matchup": "No half model",
                "stake": "0u",
                "why": "SKIP all first-half and second-half markets. Advice is full-game only.",
                "chalk": False,
                "action": "PASS",
            }
        )

    if not bets and not leans and not prop_plays and not totals_plays:
        how_to = (
            f"Sit out {sport.upper()} today. Nothing clears the confidence gate."
            if not waits
            else f"Do not bet yet. Waiting on data for {len(waits)} game(s)."
        )
    else:
        how_to = (
            "1) Place TAKE singles first (ML/spread). "
            "2) Full-game totals only if listed below. "
            "3) Player props: pass yds first, then one receiver playmaker per team, rush last. "
            "4) SKIP all 1H/2H / half markets. "
            "5) Same-game parlay only (Robinhood). Never cross games. Small stake after singles."
        )

    return {
        "sport": sport.upper(),
        "headline": headline,
        "how_to": how_to,
        "take_now": take_lines,
        "playbook": playbook_steps,
        "policy": {
            "NFL": "BET |margin|>=7 (~74%). LEAN at >=5. Skip fav QB-out. Prefer ML winners over ATS. SKIP halves.",
            "MLB": "Rule: both SPs known, ERA gap >= 1, and |margin| >= 0.5.",
            "CFB": "Rule: only bet when |pred_margin| >= 10. Prefer spread if ML is chalk.",
        }.get(sport.upper()),
        "counts": {
            "bet": len(bets),
            "lean": len(leans),
            "wait": len(waits),
            "pass": len(passes),
            "best_singles": len(best_singles),
            "recommended_parlays": len(parlay_rows),
            "prop_plays": len([p for p in prop_plays if p.get("verb") != "WATCH"]),
            "totals_plays": len(totals_plays),
        },
        "best_singles": best_singles,
        "best_parlays": parlay_rows[:6],
        "prop_plays": prop_plays,
        "totals_plays": totals_plays,
        "games": game_rows,
        "unit_guide": {
            "BEST_single": "1.0 unit",
            "GOOD_single": "0.5 unit",
            "LEAN": "0.25 unit or skip",
            "prop_pass": "0.5 unit after comparing DK line",
            "prop_recv_rush": "0.25 unit",
            "total_full_game": "0.5 unit",
            "half_markets": "0 units (SKIP)",
            "parlay_LOW_risk": "0.5 unit",
            "parlay_MED_risk": "0.25 unit",
            "PASS_WAIT": "0 units",
        },
    }


def _headline(
    sport: str,
    bets: list,
    leans: list,
    waits: list,
    singles: list,
    parlays: list,
) -> str:
    if not bets and not leans:
        if waits:
            return f"{sport}: waiting on starters / data. No bets yet."
        return f"{sport}: no confidence-gated edges on this slate. Sit out."
    top = bets[0] if bets else leans[0]
    n_bet = len(bets)
    n_parlay = len(parlays)
    ml_singles = [s for s in singles if s.get("market") == "h2h" or str(s.get("label", "")).endswith(" ML")]
    single_txt = (ml_singles[0] if ml_singles else singles[0] if singles else None)
    single_txt = single_txt["label"] if single_txt else top.get("primary_bet")
    if n_bet:
        return (
            f"{sport}: {n_bet} TAKE play{'s' if n_bet != 1 else ''}. "
            f"Start with {single_txt}. "
            f"{'Same-game parlay only after singles.' if n_parlay else 'Singles only today. Skip parlays.'}"
        )
    return f"{sport}: soft leans only ({len(leans)}). Small size or skip."
