"""CLI: research | predict | edges | parlays | ingest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from nfl_edge.config import OUTPUT_DIR, current_nfl_season, ensure_dirs

app = typer.Typer(add_completion=False, no_args_is_help=True, help="NFL DraftKings edge finder")
console = Console()


def _season_week(season: Optional[int], week: Optional[int]) -> tuple[int, int]:
    season = season or current_nfl_season()
    if week is None:
        # infer next unfinished week from schedule
        from nfl_edge.ingest.nflverse import load_schedules

        sched = load_schedules([season], regular_only=True)
        unfinished = sched.filter(pl.col("home_score").is_null())
        if unfinished.is_empty():
            week = int(sched["week"].max())
        else:
            week = int(unfinished["week"].min())
    return season, week


@app.command("ingest")
def ingest_cmd(
    start: int = typer.Option(2006, help="Start season"),
    end: Optional[int] = typer.Option(None, help="End season (default: current)"),
    refresh: bool = typer.Option(False, help="Wipe cache and re-download"),
):
    """Download/cache nflverse regular-season history."""
    ensure_dirs()
    end = end or current_nfl_season()
    seasons = list(range(start, end + 1))
    from nfl_edge.ingest.nflverse import (
        load_depth_charts,
        load_injuries,
        load_player_stats,
        load_schedules,
        load_team_stats,
        refresh_all,
    )

    if refresh:
        counts = refresh_all(seasons)
    else:
        counts = {
            "schedules": load_schedules(seasons).height,
            "team_stats": load_team_stats(seasons).height,
            "player_stats": load_player_stats(seasons).height,
            "injuries": load_injuries(seasons).height,
            "depth_charts": load_depth_charts(seasons).height,
        }
    console.print("[green]Ingest complete[/green]", counts)


@app.command("research")
def research_cmd(
    start: int = typer.Option(2006, help="Start season"),
    end: Optional[int] = typer.Option(None, help="End season"),
):
    """Recompute factor_weights.json from historical data."""
    ensure_dirs()
    from nfl_edge.ingest.nflverse import load_schedules
    from nfl_edge.research_runner import run_research

    end = end or current_nfl_season()
    console.print(f"Loading schedules {start}-{end}…")
    load_schedules(list(range(start, end + 1)))
    result = run_research(start_season=start, end_season=end)
    console.print(f"[green]Wrote[/green] {result['path']}")
    console.print_json(json.dumps({
        "hfa_points": result["weights"]["hfa_points"],
        "weather": {k: v for k, v in result["weights"]["weather"].items()},
        "momentum": result["weights"]["momentum"],
    }))


@app.command("predict")
def predict_cmd(
    week: Optional[int] = typer.Option(None, help="NFL week"),
    season: Optional[int] = typer.Option(None, help="Season year"),
    backtest: bool = typer.Option(False, help="Also score completed games this season"),
):
    """Model spreads/totals/team yard projections for a week."""
    ensure_dirs()
    season, week = _season_week(season, week)
    from nfl_edge.features.injuries import build_injury_features
    from nfl_edge.features.weather import load_weights
    from nfl_edge.ingest.nflverse import (
        load_depth_charts,
        load_injuries,
        load_player_stats,
        load_schedules,
        load_team_stats,
    )
    from nfl_edge.models.game import backtest_margins, predict_slate
    from nfl_edge.models.props import project_slate_props

    hist_seasons = list(range(max(2006, season - 3), season + 1))
    schedules_hist = load_schedules(hist_seasons)
    week_sched = schedules_hist.filter((pl.col("season") == season) & (pl.col("week") == week))
    if week_sched.is_empty():
        console.print(f"[red]No games for {season} week {week}[/red]")
        raise typer.Exit(1)

    weights = load_weights()
    team_stats = load_team_stats(hist_seasons)
    injuries = load_injuries([season])
    depth = load_depth_charts([season])
    player_stats = load_player_stats(hist_seasons)
    inj_feat = build_injury_features(week_sched, injuries, depth, weights)

    from nfl_edge.features.deep_matchups import build_deep_matchup_features, build_team_style_table
    from nfl_edge.features.ratings import build_team_game_log
    from nfl_edge.features.situational import build_situational_features

    console.print("[dim]Building similar-opponent + player mismatch features…[/dim]")
    team_games = build_team_game_log(schedules_hist)
    deep = build_deep_matchup_features(
        week_sched, team_stats, player_stats, depth, team_games, weights=weights
    )
    styles = build_team_style_table(team_stats) if not team_stats.is_empty() else None
    sit = build_situational_features(week_sched, styles=styles, weights=weights)

    # Ratings need full history; deep features attach to this week's game_ids only
    preds_all = predict_slate(
        schedules_hist,
        team_stats,
        inj_feat,
        weights,
        deep_features=deep,
        situational_features=sit,
    )
    preds_week = preds_all.filter((pl.col("season") == season) & (pl.col("week") == week))

    props = project_slate_props(preds_week, player_stats, depth_charts=depth)

    out = OUTPUT_DIR / f"week_{season}_{week}_predictions.csv"
    preds_week.write_csv(out)
    if not props.is_empty():
        props.write_csv(OUTPUT_DIR / f"week_{season}_{week}_props.csv")
    if not deep.is_empty():
        deep.write_csv(OUTPUT_DIR / f"week_{season}_{week}_deep_matchups.csv")

    table = Table(title=f"Predictions {season} Week {week}")
    for col in [
        "away_team",
        "home_team",
        "pred_margin",
        "pred_total",
        "home_win_prob",
        "deep",
        "cont",
    ]:
        table.add_column(col)
    for r in preds_week.iter_rows(named=True):
        cont_h = r.get("home_sim_continuity")
        cont_a = r.get("away_sim_continuity")
        cont = ""
        if cont_h is not None and cont_a is not None:
            cont = f"{float(cont_h):.0%}/{float(cont_a):.0%}"
        deep_adj = r.get("deep_net_margin_adj")
        table.add_row(
            str(r["away_team"]),
            str(r["home_team"]),
            f"{r['pred_margin']:+.1f}",
            f"{r['pred_total']:.1f}",
            f"{r['home_win_prob']:.1%}",
            f"{float(deep_adj):+.1f}" if deep_adj is not None else "",
            cont,
        )
    console.print(table)
    console.print(f"Wrote {out}")
    if not deep.is_empty():
        console.print(
            f"[dim]Deep matchup notes in output/week_{season}_{week}_deep_matchups.csv[/dim]"
        )

    if backtest:
        stats = backtest_margins(preds_all.filter(pl.col("season") == season))
        console.print("[cyan]Backtest (completed games this season):[/cyan]", stats)


@app.command("edges")
def edges_cmd(
    week: Optional[int] = typer.Option(None),
    season: Optional[int] = typer.Option(None),
    fetch: bool = typer.Option(True, help="Fetch live DraftKings odds (needs ODDS_API_KEY)"),
    props: bool = typer.Option(False, help="Also fetch player props (uses more quota)"),
):
    """Flag +EV bets vs DraftKings (or nflverse closing lines if no API key)."""
    ensure_dirs()
    season, week = _season_week(season, week)

    pred_path = OUTPUT_DIR / f"week_{season}_{week}_predictions.csv"
    if not pred_path.exists():
        console.print("No predictions found — running predict first…")
        predict_cmd(week=week, season=season, backtest=False)

    predictions = pl.read_csv(pred_path)
    prop_path = OUTPUT_DIR / f"week_{season}_{week}_props.csv"
    prop_proj = pl.read_csv(prop_path) if prop_path.exists() else pl.DataFrame()

    from nfl_edge.edge.calculator import compute_main_edges, compute_prop_edges
    from nfl_edge.ingest.odds import (
        OddsAPIError,
        fetch_event_odds,
        fetch_player_props,
        flatten_main_markets,
        flatten_prop_markets,
        load_cached_events,
    )
    from nfl_edge.ingest.nflverse import load_schedules
    from nfl_edge.ingest.schedule_odds import schedule_lines_as_odds

    odds_rows: list = []
    events = load_cached_events()
    used_source = "cache"

    if fetch:
        try:
            events = fetch_event_odds()
            odds_rows = flatten_main_markets(events)
            used_source = "draftkings"
            console.print(f"Fetched {len(events)} DK events")
        except OddsAPIError as e:
            console.print(f"[yellow]Odds API unavailable:[/yellow] {e}")
            events = events or []

    if not odds_rows and events:
        odds_rows = flatten_main_markets(events)
        used_source = "cache"

    if not odds_rows:
        week_sched = load_schedules([season]).filter(pl.col("week") == week)
        odds_rows = schedule_lines_as_odds(week_sched)
        used_source = "nflverse_lines"
        console.print(
            f"[cyan]Using nflverse schedule lines as market proxy[/cyan] ({len(odds_rows)} rows). "
            "Set ODDS_API_KEY for live DraftKings."
        )

    main = compute_main_edges(predictions, odds_rows)

    prop_edges = pl.DataFrame()
    if props and used_source == "draftkings" and events:
        all_prop_odds = []
        try:
            for ev in events:
                payload = fetch_player_props(ev["id"])
                all_prop_odds.extend(flatten_prop_markets(payload))
            prop_edges = compute_prop_edges(prop_proj, all_prop_odds)
        except OddsAPIError as e:
            console.print(f"[yellow]Prop fetch failed:[/yellow] {e}")
    elif props and used_source != "draftkings":
        console.print("[dim]Player prop odds require ODDS_API_KEY; skipping prop edges.[/dim]")

    frames = [f for f in [main, prop_edges] if f is not None and not f.is_empty()]
    if not frames:
        console.print("[yellow]No edges computed (team name mismatch or empty odds).[/yellow]")
        raise typer.Exit(0)

    combined = pl.concat(frames, how="diagonal_relaxed").sort(["passes_threshold", "ev"], descending=True)
    out = OUTPUT_DIR / f"week_{season}_{week}_edges.csv"
    combined.write_csv(out)

    table = Table(title=f"+EV candidates {season} W{week} ({used_source})")
    for col in ["away_team", "home_team", "label", "dk_price", "fair_american", "model_prob", "ev", "passes_threshold"]:
        table.add_column(col)
    show = combined.filter(pl.col("passes_threshold") == True)  # noqa: E712
    if show.is_empty():
        show = combined.head(15)
        console.print("[dim]No bets cleared EV threshold; showing top EV anyway.[/dim]")
    for r in show.head(25).iter_rows(named=True):
        table.add_row(
            str(r.get("away_team")),
            str(r.get("home_team")),
            str(r.get("label")),
            str(r.get("dk_price")),
            str(r.get("fair_american")),
            f"{float(r.get('model_prob') or 0):.1%}",
            f"{float(r.get('ev') or 0):+.1%}",
            str(r.get("passes_threshold")),
        )
    console.print(table)
    console.print(f"Wrote {out}")


@app.command("parlays")
def parlays_cmd(
    week: Optional[int] = typer.Option(None),
    season: Optional[int] = typer.Option(None),
):
    """Suggest correlated 2–3 leg SGPs from edged bets."""
    ensure_dirs()
    season, week = _season_week(season, week)
    edge_path = OUTPUT_DIR / f"week_{season}_{week}_edges.csv"
    if not edge_path.exists():
        console.print("No edges file — run `nfl-edge edges` first.")
        raise typer.Exit(1)

    from nfl_edge.edge.calculator import build_parlays

    edges = pl.read_csv(edge_path)
    parlays = build_parlays(edges)
    out = OUTPUT_DIR / f"week_{season}_{week}_parlays.json"
    out.write_text(json.dumps(parlays, indent=2) + "\n")

    if not parlays:
        console.print("[yellow]No correlated parlay candidates.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"SGP ideas {season} W{week}")
    table.add_column("legs")
    table.add_column("fair")
    table.add_column("ev_proxy")
    table.add_column("note")
    for p in parlays:
        table.add_row(
            " + ".join(p["legs"]),
            str(p["fair_american"]),
            f"{p['combined_model_ev_proxy']:+.2f}",
            p["note"],
        )
    console.print(table)
    console.print(f"Wrote {out}")


@app.command("backtest")
def backtest_cmd(
    season: Optional[int] = typer.Option(None),
    start: int = typer.Option(2018, help="History start for ratings"),
):
    """Winner / MAE backtest for a season."""
    season = season or current_nfl_season()
    from nfl_edge.ingest.nflverse import load_schedules, load_team_stats
    from nfl_edge.models.game import backtest_margins, predict_slate
    from nfl_edge.features.weather import load_weights

    seasons = list(range(start, season + 1))
    schedules = load_schedules(seasons)
    team_stats = load_team_stats(seasons)
    preds = predict_slate(schedules, team_stats, None, load_weights())
    stats = backtest_margins(preds.filter(pl.col("season") == season))
    console.print(stats)


@app.command("mlb")
def mlb_cmd(
    days: int = typer.Option(2, help="Days ahead to project"),
):
    """Predict upcoming MLB games with weighted factor breakdowns."""
    from mlb_edge.predict import predict_mlb_slate

    slate = predict_mlb_slate(days_ahead=days)
    games = slate.get("games") or []
    console.print(f"[green]MLB slate[/green]: {len(games)} games")
    table = Table(title="MLB predictions")
    for col in ["gameday", "away_team", "home_team", "away_sp_name", "home_sp_name", "pred_margin", "pred_total", "home_win_prob"]:
        table.add_column(col)
    for g in games:
        table.add_row(
            str(g.get("gameday")),
            str(g.get("away_team")),
            str(g.get("home_team")),
            str(g.get("away_sp_name") or "?"),
            str(g.get("home_sp_name") or "?"),
            f"{float(g.get('pred_margin') or 0):+.1f}",
            f"{float(g.get('pred_total') or 0):.1f}",
            f"{float(g.get('home_win_prob') or 0):.0%}",
        )
    console.print(table)
    console.print("Factor weights:", {f["key"]: f["weight"] for f in slate.get("factors") or []})


@app.command("cfb")
def cfb_cmd(
    week: Optional[int] = typer.Option(None, help="CFB week"),
    season: Optional[int] = typer.Option(None, help="Season year"),
):
    """Predict upcoming FBS college football slate."""
    from cfb_edge.predict import predict_cfb_slate

    slate = predict_cfb_slate(season=season, week=week)
    games = slate.get("games") or []
    console.print(
        f"[green]CFB slate[/green]: {slate.get('season')} W{slate.get('week')} · "
        f"{len(games)} games · {slate.get('data_source')}"
    )
    table = Table(title="CFB predictions")
    for col in ["gameday", "away_team", "home_team", "pred_margin", "pred_total", "home_win_prob", "high_confidence"]:
        table.add_column(col)
    for g in games[:40]:
        table.add_row(
            str(g.get("gameday")),
            str(g.get("away_team")),
            str(g.get("home_team")),
            f"{float(g.get('pred_margin') or 0):+.1f}",
            f"{float(g.get('pred_total') or 0):.1f}",
            f"{float(g.get('home_win_prob') or 0):.0%}",
            str(bool(g.get("high_confidence"))),
        )
    console.print(table)
    conf = (slate.get("weights") or {}).get("confidence") or {}
    console.print("Confidence gate |margin| >=", conf.get("min_abs_margin"))


@app.command("ui")
def ui_cmd(
    port: int = typer.Option(8510, help="Local port (8501/8502 belong to the paper dashboard)"),
):
    """Launch the clean Sports Edge webpage (parlay board)."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    console.print(f"[green]Sports Edge[/green] → http://127.0.0.1:{port}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ui.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(root),
        check=False,
    )


def main():
    app()


if __name__ == "__main__":
    main()
