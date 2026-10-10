"""Bet log: record what was actually bet, grade it from final scores, summarize results."""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import polars as pl

from nfl_edge.config import CACHE_DIR, OUTPUT_DIR, ensure_dirs

LOG_PATH = OUTPUT_DIR / "bet_log.json"
SCORE_TTL_SECONDS = 15 * 60
RESULTS = {"pending", "won", "lost", "push"}
PROP_STAT_COLS = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
}


def _f(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def parse_price(text: Any) -> float:
    """
    Implied win probability from what was paid.
    Accepts American odds (-150, +120) or Robinhood-style contract cents (62c, 62¢, 0.62).
    """
    raw = str(text or "").strip().lower().replace(" ", "")
    if not raw:
        raise ValueError("Enter the price you got, e.g. -150, +120, or 62c")
    cents = re.fullmatch(r"(\d+(?:\.\d+)?)(c|¢)", raw)
    if cents:
        p = float(cents.group(1)) / 100.0
    elif re.fullmatch(r"0?\.\d+", raw):
        p = float(raw)
    elif re.fullmatch(r"[+-]\d+(?:\.\d+)?", raw) or re.fullmatch(r"\d{3,}", raw):
        a = float(raw)
        if abs(a) < 100:
            raise ValueError("American odds must be -100 or lower, or +100 or higher")
        p = 100.0 / (a + 100.0) if a > 0 else -a / (-a + 100.0)
    else:
        raise ValueError(f"Could not read price '{text}'. Use -150, +120, or 62c")
    if not 0.0 < p < 1.0:
        raise ValueError("Price must imply a probability between 0 and 1")
    return p


def _read() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return json.loads(LOG_PATH.read_text() or "[]")


def _write(bets: list[dict]) -> None:
    ensure_dirs()
    tmp = LOG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(bets, indent=2) + "\n")
    tmp.replace(LOG_PATH)


def _profit(bet: dict) -> float | None:
    status = bet.get("status")
    stake = _f(bet.get("stake"), 0.0) or 0.0
    if status == "won":
        p = _f(bet.get("implied_prob"))
        return round(stake * (1.0 / p - 1.0), 3) if p else None
    if status == "lost":
        return round(-stake, 3)
    if status == "push":
        return 0.0
    return None


def list_bets() -> list[dict]:
    bets = _read()
    bets.sort(key=lambda b: b.get("created_at") or "", reverse=True)
    return bets


def add_bet(payload: dict) -> dict:
    market = str(payload.get("market") or "")
    if market not in {"h2h", "spreads", "totals", "sgp"} and market not in PROP_STAT_COLS:
        raise ValueError(f"Unknown market '{market}'")
    stake = _f(payload.get("stake"))
    if stake is None or stake <= 0:
        raise ValueError("Stake must be a positive number of units")
    implied = parse_price(payload.get("price"))
    if market in {"spreads", "totals"} or market in PROP_STAT_COLS:
        if _f(payload.get("line")) is None:
            raise ValueError("This bet needs the line you got")
    if (market == "totals" or market in PROP_STAT_COLS) and payload.get("side") not in {"Over", "Under"}:
        raise ValueError("Pick Over or Under")
    if market in {"h2h", "spreads"} and payload.get("is_home") is None:
        raise ValueError("Missing which team was picked")

    bet = {
        "id": uuid.uuid4().hex[:10],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sport": str(payload.get("sport") or "NFL").upper(),
        "season": payload.get("season"),
        "week": payload.get("week"),
        "gameday": payload.get("gameday"),
        "game_id": str(payload.get("game_id") or ""),
        "matchup": payload.get("matchup"),
        "market": market,
        "pick": payload.get("pick"),
        "team": payload.get("team"),
        "is_home": payload.get("is_home"),
        "line": _f(payload.get("line")),
        "side": payload.get("side"),
        "player": payload.get("player"),
        "legs": payload.get("legs") or [],
        "price": str(payload.get("price")).strip(),
        "implied_prob": round(implied, 4),
        "model_prob": _f(payload.get("model_prob")),
        "card_action": payload.get("card_action"),
        "stake": stake,
        "status": "pending",
        "graded_by": None,
        "result_note": None,
        "profit": None,
    }
    bets = _read()
    bets.append(bet)
    _write(bets)
    return bet


def delete_bet(bet_id: str) -> bool:
    bets = _read()
    kept = [b for b in bets if b.get("id") != bet_id]
    if len(kept) == len(bets):
        return False
    _write(kept)
    return True


def set_result(bet_id: str, result: str) -> dict:
    if result not in RESULTS:
        raise ValueError(f"Result must be one of {sorted(RESULTS)}")
    bets = _read()
    for b in bets:
        if b.get("id") == bet_id:
            b["status"] = result
            b["graded_by"] = None if result == "pending" else "manual"
            b["result_note"] = None
            b["profit"] = _profit(b)
            _write(bets)
            return b
    raise KeyError(bet_id)


# ---------------------------------------------------------------- grading


def _ttl_frame(name: str, loader: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    ensure_dirs()
    path = CACHE_DIR / f"tracker_{name}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < SCORE_TTL_SECONDS:
        return pl.read_parquet(path)
    df = loader()
    if not df.is_empty():
        df.write_parquet(path)
    return df


def _nfl_scores(season: int) -> dict[str, tuple[float, float]]:
    import nflreadpy as nfl

    df = _ttl_frame(f"nfl_sched_{season}", lambda: nfl.load_schedules(seasons=[season]))
    done = df.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    return {r["game_id"]: (float(r["home_score"]), float(r["away_score"])) for r in done.iter_rows(named=True)}


def _cfb_scores(season: int) -> dict[str, tuple[float, float]]:
    from cfb_edge.ingest import load_schedules

    df = _ttl_frame(
        f"cfb_sched_{season}",
        lambda: load_schedules([season], regular_only=False, fbs_only=False, use_cache=False),
    )
    done = df.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    return {str(r["game_id"]): (float(r["home_score"]), float(r["away_score"])) for r in done.iter_rows(named=True)}


def _mlb_scores(day: str) -> dict[str, tuple[float, float]]:
    from mlb_edge.ingest import fetch_schedule_dates

    d = date.fromisoformat(str(day)[:10])
    out: dict[str, tuple[float, float]] = {}
    for g in fetch_schedule_dates(d, d, use_cache=False):
        if str(g.get("status") or "") not in {"Final", "Game Over", "Completed Early"}:
            continue
        if g.get("home_score") is None or g.get("away_score") is None:
            continue
        out[str(g.get("game_pk"))] = (float(g["home_score"]), float(g["away_score"]))
    return out


def _nfl_player_stats(season: int) -> pl.DataFrame:
    import nflreadpy as nfl

    return _ttl_frame(
        f"nfl_player_stats_{season}",
        lambda: nfl.load_player_stats(seasons=[season], summary_level="week"),
    )


def _grade_game_market(bet: dict, home: float, away: float) -> tuple[str, str]:
    market = bet["market"]
    score_txt = f"final {away:g}-{home:g}"
    if market == "totals":
        total = home + away
        line = float(bet["line"])
        if total == line:
            return "push", f"{score_txt}, total {total:g}"
        over = total > line
        won = over if bet["side"] == "Over" else not over
        return ("won" if won else "lost"), f"{score_txt}, total {total:g}"
    mine, theirs = (home, away) if bet["is_home"] else (away, home)
    if market == "h2h":
        margin = mine - theirs
    else:
        margin = mine + float(bet["line"]) - theirs
    if margin == 0:
        return "push", score_txt
    return ("won" if margin > 0 else "lost"), score_txt


def _grade_prop(bet: dict, stats: pl.DataFrame) -> tuple[str, str] | None:
    col = PROP_STAT_COLS[bet["market"]]
    week = bet.get("week")
    if stats.is_empty() or week is None or col not in stats.columns:
        return None
    name_col = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    rows = stats.filter(
        (pl.col("week") == int(week)) & (pl.col(name_col).str.to_lowercase() == str(bet.get("player") or "").lower())
    )
    if rows.is_empty():
        return None
    value = float(rows[col][0] or 0)
    line = float(bet["line"])
    if value == line:
        return "push", f"{value:g} yds"
    over = value > line
    won = over if bet["side"] == "Over" else not over
    return ("won" if won else "lost"), f"{value:g} yds"


def grade_pending() -> dict:
    """Auto-grade pending bets whose games are final. SGPs stay manual."""
    bets = _read()
    score_cache: dict[tuple, dict] = {}
    stats_cache: dict[int, pl.DataFrame] = {}
    graded, errors = 0, []

    def scores_for(bet: dict) -> dict[str, tuple[float, float]]:
        sport = bet.get("sport")
        key = (sport, bet.get("gameday") if sport == "MLB" else bet.get("season"))
        if key not in score_cache:
            if sport == "NFL":
                score_cache[key] = _nfl_scores(int(bet["season"]))
            elif sport == "CFB":
                score_cache[key] = _cfb_scores(int(bet["season"]))
            elif sport == "MLB":
                score_cache[key] = _mlb_scores(str(bet["gameday"]))
            else:
                score_cache[key] = {}
        return score_cache[key]

    for bet in bets:
        if bet.get("status") != "pending" or bet.get("market") == "sgp":
            continue
        try:
            if bet["market"] in PROP_STAT_COLS:
                if bet.get("sport") != "NFL":
                    continue
                season = int(bet["season"])
                if season not in stats_cache:
                    stats_cache[season] = _nfl_player_stats(season)
                outcome = _grade_prop(bet, stats_cache[season])
            else:
                final = scores_for(bet).get(str(bet.get("game_id")))
                outcome = _grade_game_market(bet, *final) if final else None
        except Exception as exc:  # network or schema issues should not block other bets
            errors.append(f"{bet.get('pick')}: {exc}")
            continue
        if outcome is None:
            continue
        bet["status"], bet["result_note"] = outcome
        bet["graded_by"] = "auto"
        bet["profit"] = _profit(bet)
        graded += 1

    if graded:
        _write(bets)
    return {"graded": graded, "errors": errors}


# ---------------------------------------------------------------- summary


def _bucket(rows: list[dict]) -> dict:
    settled = [b for b in rows if b.get("status") in {"won", "lost", "push"}]
    won = sum(1 for b in settled if b["status"] == "won")
    lost = sum(1 for b in settled if b["status"] == "lost")
    push = sum(1 for b in settled if b["status"] == "push")
    staked = sum(_f(b.get("stake"), 0.0) or 0.0 for b in settled)
    profit = sum(_f(b.get("profit"), 0.0) or 0.0 for b in settled)
    decided = [b for b in settled if b["status"] != "push"]
    with_model = [b for b in decided if b.get("model_prob") is not None]
    return {
        "bets": len(rows),
        "pending": sum(1 for b in rows if b.get("status") == "pending"),
        "won": won,
        "lost": lost,
        "push": push,
        "staked": round(staked, 2),
        "profit": round(profit, 2),
        "roi": round(profit / staked, 4) if staked else None,
        "win_rate": round(won / len(decided), 4) if decided else None,
        "avg_price_prob": round(sum(b["implied_prob"] for b in decided) / len(decided), 4) if decided else None,
        "model_expected_wins": round(sum(b["model_prob"] for b in with_model), 2) if with_model else None,
        "model_actual_wins": sum(1 for b in with_model if b["status"] == "won") if with_model else None,
    }


def summary(bets: list[dict] | None = None) -> dict:
    bets = bets if bets is not None else _read()
    by_market: dict[str, list[dict]] = {}
    for b in bets:
        key = "props" if b.get("market") in PROP_STAT_COLS else b.get("market")
        by_market.setdefault(str(key), []).append(b)
    return {
        "overall": _bucket(bets),
        "by_market": {k: _bucket(v) for k, v in sorted(by_market.items())},
    }


def log_path() -> Path:
    return LOG_PATH
