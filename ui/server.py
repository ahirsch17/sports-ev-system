"""Clean localhost Sports Edge board. FastAPI + static HTML."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Sports Edge")
STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _f_ev(e: dict) -> float:
    try:
        return float(e.get("ev") or 0)
    except (TypeError, ValueError):
        return 0.0


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/nfl")
def api_nfl(week: int | None = None, season: int | None = None):
    from nfl_edge.edge.advisor import build_advice
    from nfl_edge.edge.payout_parlays import build_payout_parlays, legs_from_confidence_games
    from nfl_edge.services.slate import build_nfl_slate
    from ui.board import build_game_board

    slate = build_nfl_slate(season=season, week=week)
    games = slate.get("games") or []
    edges = slate.get("edges") or []
    props = slate.get("props") or []
    # Full +EV / threshold pool so each matchup can form an SGP (not just global hi chalk)
    base = [e for e in edges if e.get("passes_threshold") or _f_ev(e) > 0.02]
    seeded = legs_from_confidence_games(games, sport="NFL")
    pool = base + seeded
    parlays = build_payout_parlays(pool, sport="NFL", min_odds=110, max_odds=900, max_parlays=24)
    advice = build_advice(
        sport="NFL",
        games=games,
        edges=edges,
        parlays=parlays,
        weights=slate.get("weights") or {},
        props=props,
    )
    board = build_game_board(
        sport="NFL",
        games=games,
        advice=advice,
        market_edges=edges,
        parlays=advice.get("best_parlays") or parlays,
    )
    return {
        "sport": "NFL",
        "season": slate.get("season"),
        "week": slate.get("week"),
        "games": games,
        "board": board,
        "edges": advice["best_singles"],
        "parlays": advice["best_parlays"],
        "advice": advice,
    }


@app.get("/api/mlb")
def api_mlb(days: int = 2):
    from mlb_edge.predict import predict_mlb_slate
    from nfl_edge.edge.advisor import build_advice
    from nfl_edge.edge.payout_parlays import (
        build_payout_parlays,
        legs_from_confidence_games,
        legs_from_mlb_games,
    )
    from ui.board import build_game_board

    slate = predict_mlb_slate(days_ahead=days, days_back=45)
    games = slate.get("games") or []
    edges = slate.get("edges") or []
    hi_legs = legs_from_mlb_games(games, high_conf_only=True)
    seeded = legs_from_confidence_games(games, sport="MLB")
    pool = edges + hi_legs + seeded
    if len(hi_legs) + len(seeded) < 3:
        pool = edges + legs_from_mlb_games(games, high_conf_only=False) + seeded
    parlays = build_payout_parlays(pool, sport="MLB", min_odds=110, max_odds=900, max_parlays=24)
    advice = build_advice(
        sport="MLB",
        games=games,
        edges=edges + hi_legs,
        parlays=parlays,
        weights=slate.get("weights") or {},
        props=slate.get("props") or [],
    )
    board = build_game_board(
        sport="MLB",
        games=games,
        advice=advice,
        market_edges=edges + hi_legs,
        parlays=advice.get("best_parlays") or parlays,
    )
    return {
        "sport": "MLB",
        "season": slate.get("season"),
        "games": games,
        "board": board,
        "edges": advice["best_singles"],
        "parlays": advice["best_parlays"],
        "factors": slate.get("factors") or [],
        "advice": advice,
    }


@app.get("/api/cfb")
def api_cfb(week: int | None = None, season: int | None = None):
    from cfb_edge.predict import predict_cfb_slate
    from nfl_edge.edge.advisor import build_advice
    from nfl_edge.edge.payout_parlays import build_payout_parlays, legs_from_confidence_games
    from ui.board import build_game_board

    slate = predict_cfb_slate(season=season, week=week)
    games = slate.get("games") or []
    edges = slate.get("edges") or []
    base = [e for e in edges if e.get("passes_threshold") or _f_ev(e) > 0.02]
    seeded = legs_from_confidence_games(games, sport="CFB")
    pool = base + seeded
    parlays = build_payout_parlays(pool, sport="CFB", min_odds=110, max_odds=900, max_parlays=24)
    advice = build_advice(
        sport="CFB",
        games=games,
        edges=edges,
        parlays=parlays,
        weights=slate.get("weights") or {},
    )
    board = build_game_board(
        sport="CFB",
        games=games,
        advice=advice,
        market_edges=edges,
        parlays=advice.get("best_parlays") or parlays,
    )
    return {
        "sport": "CFB",
        "season": slate.get("season"),
        "week": slate.get("week"),
        "games": games,
        "board": board,
        "edges": advice["best_singles"],
        "parlays": advice["best_parlays"],
        "factors": slate.get("factors") or [],
        "advice": advice,
        "data_source": slate.get("data_source"),
    }


def _bets_payload(**extra) -> dict:
    from nfl_edge import tracker

    bets = tracker.list_bets()
    return {"bets": bets, "summary": tracker.summary(bets), **extra}


@app.get("/api/bets")
def api_bets():
    return _bets_payload()


@app.post("/api/bets")
def api_add_bet(payload: dict = Body(...)):
    from nfl_edge import tracker

    try:
        bet = tracker.add_bet(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _bets_payload(bet=bet)


@app.post("/api/bets/grade")
def api_grade_bets():
    from nfl_edge import tracker

    return _bets_payload(**tracker.grade_pending())


@app.post("/api/bets/{bet_id}/result")
def api_set_result(bet_id: str, payload: dict = Body(...)):
    from nfl_edge import tracker

    try:
        tracker.set_result(bet_id, str(payload.get("result") or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Bet not found") from exc
    return _bets_payload()


@app.delete("/api/bets/{bet_id}")
def api_delete_bet(bet_id: str):
    from nfl_edge import tracker

    if not tracker.delete_bet(bet_id):
        raise HTTPException(status_code=404, detail="Bet not found")
    return _bets_payload()


def main(port: int = 8510):
    import uvicorn

    uvicorn.run("ui.server:app", host="127.0.0.1", port=port, reload=False, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
