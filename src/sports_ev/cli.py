import argparse
import sys

from sports_ev.backtest import BacktestConfig, MarketDivergenceStrategy, WalkForwardBacktester
from sports_ev.config import get_settings
from sports_ev.models import NflSpreadModel, WalkForwardModelStrategy, build_labeled_rows, save_prediction, train_and_save
from sports_ev.models.strategy import ModelSpreadStrategy
from sports_ev.db import init_db, migrate_db, session_scope
from sports_ev.features.registry import build_features_for_game
from sports_ev.fixtures.sync import PinnacleFixtureSync
from sports_ev.paper import OpportunityFlagger, PaperBetSettlementService
from sports_ev.scores import ScoreIngestService
from sports_ev.softbooks.ingest import SoftBookIngestService
from sports_ev.sports.registry import list_sports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sports_ev")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init-db", help="Create database tables")
    init_cmd.set_defaults(func=_init_db)

    migrate_cmd = sub.add_parser(
        "migrate-db",
        help="Apply additive schema updates to an existing database",
    )
    migrate_cmd.set_defaults(func=_migrate_db)

    sync_cmd = sub.add_parser("sync-fixtures", help="Upsert fixtures from Pinnacle guest API")
    sync_cmd.add_argument(
        "--sport",
        default="nfl",
        choices=list_sports(),
        help="Sport to sync (default: nfl)",
    )
    sync_cmd.add_argument(
        "--all",
        action="store_true",
        help="Sync all supported sports",
    )
    sync_cmd.set_defaults(func=_sync_fixtures)

    poll_cmd = sub.add_parser("poll-pinnacle", help="Scrape current Pinnacle lines")
    poll_cmd.add_argument("--sport", default="nfl", choices=list_sports())
    poll_cmd.set_defaults(func=_poll_pinnacle)

    soft_cmd = sub.add_parser("poll-softbooks", help="Poll DraftKings/FanDuel lines")
    soft_cmd.add_argument("--sport", default="nfl", choices=list_sports())
    soft_cmd.add_argument(
        "--books",
        default="draftkings,fanduel",
        help="Comma-separated book list (default: draftkings,fanduel)",
    )
    soft_cmd.set_defaults(func=_poll_softbooks)

    feat_cmd = sub.add_parser("build-features", help="Build feature vector for a game")
    feat_cmd.add_argument("game_id", help="Game ID to build features for")
    feat_cmd.set_defaults(func=_build_features)

    bt_cmd = sub.add_parser("run-backtest", help="Run walk-forward backtest on settled games")
    bt_cmd.add_argument("--sport", default="nfl", choices=["nfl", "mlb"])
    bt_cmd.add_argument("--season", type=int, default=None, help="Filter to season year")
    bt_cmd.add_argument("--book", default="draftkings", help="Soft book for market divergence strategy")
    bt_cmd.add_argument("--stake", type=float, default=1.0, help="Flat stake per bet")
    bt_cmd.add_argument(
        "--strategy",
        choices=["market", "model", "walk-forward-model"],
        default="market",
        help="Backtest strategy to use",
    )
    bt_cmd.add_argument(
        "--model-path",
        default=None,
        help="Path to trained model artifact (required for --strategy model)",
    )
    bt_cmd.set_defaults(func=_run_backtest)

    train_cmd = sub.add_parser("train-model", help="Train sport model with time-ordered validation")
    train_cmd.add_argument("--sport", default="nfl", choices=["nfl", "mlb"])
    train_cmd.add_argument("--season", type=int, default=None, help="Filter training games to season")
    train_cmd.add_argument(
        "--output",
        default=None,
        help="Artifact output path (default: settings.model_artifact_path)",
    )
    train_cmd.set_defaults(func=_train_model)

    predict_cmd = sub.add_parser("predict-model", help="Score a game with the trained NFL spread model")
    predict_cmd.add_argument("game_id", help="Game ID to score")
    predict_cmd.add_argument("--model-path", default=None, help="Path to model artifact")
    predict_cmd.add_argument("--save", action="store_true", help="Persist prediction to model_predictions table")
    predict_cmd.set_defaults(func=_predict_model)

    flag_cmd = sub.add_parser("flag-opportunities", help="Flag +EV paper bets on upcoming games")
    flag_cmd.add_argument("--sport", default="nfl", choices=list_sports())
    flag_cmd.add_argument("--days", type=int, default=7, help="Days ahead to scan")
    flag_cmd.add_argument("--book", default="draftkings", help="Soft book to evaluate")
    flag_cmd.add_argument(
        "--strategy",
        choices=["market", "model"],
        default="market",
        help="Opportunity detection strategy",
    )
    flag_cmd.add_argument("--model-path", default=None, help="Model artifact for --strategy model")
    flag_cmd.set_defaults(func=_flag_opportunities)

    settle_cmd = sub.add_parser("settle-paper-bets", help="Backfill outcomes and CLV for settled games")
    settle_cmd.set_defaults(func=_settle_paper_bets)

    scores_cmd = sub.add_parser("sync-scores", help="Pull final scores from ESPN into games table")
    scores_cmd.add_argument(
        "--sport",
        default=None,
        choices=list_sports(),
        help="Limit to one sport (default: all supported sports)",
    )
    scores_cmd.add_argument("--lookback-days", type=int, default=3)
    scores_cmd.set_defaults(func=_sync_scores)

    backfill_cmd = sub.add_parser(
        "backfill-mlb-stats",
        help="Backfill MLB team_game_stats from MLB Stats API (free)",
    )
    backfill_cmd.add_argument("--lookback-days", type=int, default=120)
    backfill_cmd.add_argument(
        "--with-boxscores",
        action="store_true",
        help="Fetch full boxscores for pitching stats (slow; default uses final scores only)",
    )
    backfill_cmd.set_defaults(func=_backfill_mlb_stats)

    nfl_backfill_cmd = sub.add_parser(
        "backfill-nfl-history",
        help="Load completed NFL games, spreads, and EPA stats from nflverse",
    )
    nfl_backfill_cmd.add_argument(
        "--seasons",
        default="2023,2024",
        help="Comma-separated seasons to load (default: 2023,2024)",
    )
    nfl_backfill_cmd.set_defaults(func=_backfill_nfl_history)

    prob_cmd = sub.add_parser(
        "sync-mlb-probables",
        help="Sync probable starter stats for upcoming/historical MLB games",
    )
    prob_cmd.add_argument("--days", type=int, default=7, help="Days ahead for upcoming sync")
    prob_cmd.add_argument(
        "--historical",
        action="store_true",
        help="Backfill probables from historical schedule (slow)",
    )
    prob_cmd.add_argument("--lookback-days", type=int, default=120)
    prob_cmd.set_defaults(func=_sync_mlb_probables)

    dash_cmd = sub.add_parser("run-dashboard", help="Launch Streamlit paper tracking dashboard")
    dash_cmd.set_defaults(func=_run_dashboard)

    scorecard_cmd = sub.add_parser(
        "model-scorecard",
        help="Walk-forward model backtest with edge-bucket scorecard",
    )
    scorecard_cmd.add_argument("--sport", default="mlb", choices=["nfl", "mlb"])
    scorecard_cmd.add_argument("--book", default="draftkings", help="Soft book for lines")
    scorecard_cmd.add_argument("--season", type=int, default=None, help="Filter to season year")
    scorecard_cmd.add_argument("--stake", type=float, default=1.0, help="Flat stake per bet")
    scorecard_cmd.add_argument(
        "--walk-forward",
        action="store_true",
        help="Retrain model on past games only (slower, more honest)",
    )
    scorecard_cmd.add_argument("--model-path", default=None, help="Model artifact path override")
    scorecard_cmd.set_defaults(func=_model_scorecard)

    refresh_cmd = sub.add_parser(
        "refresh-cycle",
        help="Poll odds, refresh model paper picks, settle games (for Task Scheduler / cron)",
    )
    refresh_cmd.add_argument("--sport", default="mlb", choices=list_sports())
    refresh_cmd.add_argument(
        "--all-sports",
        action="store_true",
        help="Run for every supported sport",
    )
    refresh_cmd.add_argument("--days", type=int, default=3, help="Days ahead for model picks")
    refresh_cmd.add_argument(
        "--all-picks",
        action="store_true",
        help="Log every model side including leans below +EV (default: +EV only)",
    )
    refresh_cmd.add_argument("--book", default="draftkings", help="Soft book for picks")
    refresh_cmd.add_argument(
        "--skip-settle",
        action="store_true",
        help="Skip score sync and bet settlement",
    )
    refresh_cmd.add_argument(
        "--sync-probables",
        action="store_true",
        help="Sync MLB probable starter stats before picks",
    )
    refresh_cmd.add_argument(
        "--capture-closing",
        action="store_true",
        help="Capture Pinnacle closing lines for recently started games",
    )
    refresh_cmd.add_argument(
        "--score-lookback-days",
        type=int,
        default=14,
        help="ESPN score sync lookback for settling stale bets (default: 14)",
    )
    refresh_cmd.set_defaults(func=_refresh_cycle)

    report_cmd = sub.add_parser(
        "paper-report",
        help="Print paper betting performance summary (for logs / email)",
    )
    report_cmd.add_argument("--days", type=int, default=7, help="Recent activity window")
    report_cmd.set_defaults(func=_paper_report)

    args = parser.parse_args(argv)
    return args.func(args)


def _init_db(_args: argparse.Namespace) -> int:
    settings = get_settings()
    print(f"Initializing database at {settings.database_url.split('@')[-1]} ...")
    init_db(settings=settings)
    print("Done.")
    return 0


def _migrate_db(_args: argparse.Namespace) -> int:
    settings = get_settings()
    print(f"Migrating database at {settings.database_url.split('@')[-1]} ...")
    applied = migrate_db(settings=settings)
    if applied:
        for item in applied:
            print(f"  added {item}")
    else:
        print("  schema already up to date")
    print("Done.")
    return 0


def _sync_fixtures(args: argparse.Namespace) -> int:
    settings = get_settings()
    with session_scope(settings=settings) as session:
        syncer = PinnacleFixtureSync(session, settings)
        try:
            if args.all:
                results = syncer.sync_all()
            else:
                results = [syncer.sync_fixtures(args.sport)]
        finally:
            syncer.close()

    for result in results:
        print(
            f"sport={result.sport} matchups_seen={result.matchups_seen} "
            f"games_upserted={result.games_upserted}"
        )
        for error in result.errors:
            print(f"  error: {error}", file=sys.stderr)
    return 0 if all(not r.errors for r in results) else 1


def _poll_pinnacle(args: argparse.Namespace) -> int:
    settings = get_settings()
    with session_scope(settings=settings) as session:
        service = SoftBookIngestService(session, settings)
        result = service.poll_books(["pinnacle"], sport=args.sport)

    print(
        f"lines_scraped={result.lines_scraped} snapshots_accepted={result.snapshots_accepted} "
        f"snapshots_rejected={result.snapshots_rejected} unmatched_games={result.unmatched_games}"
    )
    for error in result.errors:
        print(f"  error: {error}", file=sys.stderr)
    return 0


def _poll_softbooks(args: argparse.Namespace) -> int:
    settings = get_settings()
    books = [b.strip() for b in args.books.split(",") if b.strip()]
    with session_scope(settings=settings) as session:
        service = SoftBookIngestService(session, settings)
        result = service.poll_books(books, sport=args.sport)

    print(
        f"books_polled={result.books_polled} lines_scraped={result.lines_scraped} "
        f"snapshots_accepted={result.snapshots_accepted} snapshots_rejected={result.snapshots_rejected} "
        f"unmatched_games={result.unmatched_games}"
    )
    for error in result.errors:
        print(f"  error: {error}", file=sys.stderr)
    return 0


def _build_features(args: argparse.Namespace) -> int:
    settings = get_settings()
    with session_scope(settings=settings) as session:
        vector = build_features_for_game(session, args.game_id)

    print(f"feature_version={vector.feature_version} game_id={vector.game_id}")
    for key, value in sorted(vector.features.items()):
        if value is not None:
            print(f"  {key}: {value}")
    return 0


def _run_backtest(args: argparse.Namespace) -> int:
    settings = get_settings()
    config = BacktestConfig(season=args.season, sport=args.sport, flat_stake=args.stake)
    market_type = config.resolved_market_type()

    if args.strategy == "market":
        strategy = MarketDivergenceStrategy(soft_book=args.book, market_type=market_type)
    elif args.strategy == "model":
        if args.sport == "mlb":
            from sports_ev.models.mlb_artifact import MlbMoneylineModel
            from sports_ev.models.mlb_strategy import ModelMoneylineStrategy

            model_path = args.model_path or settings.mlb_model_artifact_path
            model = MlbMoneylineModel.load(model_path)
            strategy = ModelMoneylineStrategy(model=model, soft_book=args.book)
        else:
            model_path = args.model_path or settings.model_artifact_path
            model = NflSpreadModel.load(model_path)
            strategy = ModelSpreadStrategy(model=model, soft_book=args.book)
    elif args.sport == "mlb":
        from sports_ev.models.mlb_strategy import WalkForwardMlbModelStrategy

        strategy = WalkForwardMlbModelStrategy(
            soft_book=args.book,
            min_train_games=settings.mlb_model_min_train_games,
        )
    else:
        strategy = WalkForwardModelStrategy(
            soft_book=args.book,
            min_train_games=settings.model_min_train_games,
        )

    with session_scope(settings=settings) as session:
        report = WalkForwardBacktester(session, settings).run(strategy=strategy, config=config)

    print(f"backtest_version={report.backtest_version} strategy={report.strategy_name}")
    print(
        f"bets={report.total_bets} wins={report.wins} losses={report.losses} pushes={report.pushes} "
        f"win_rate={report.win_rate:.2f}% roi={report.roi:.2f}% "
        f"avg_clv={report.avg_clv if report.avg_clv is not None else 'n/a'} "
        f"max_drawdown={report.max_drawdown:.2f}"
    )
    print("edge_buckets:")
    for bucket in report.edge_buckets:
        clv = bucket.avg_clv if bucket.avg_clv is not None else "n/a"
        print(
            f"  {bucket.bucket}: bets={bucket.bets} win_rate={bucket.win_rate:.2f}% "
            f"roi={bucket.roi:.2f}% avg_clv={clv}"
        )
    print(f"reproducibility_hash={report.reproducibility_hash()}")
    return 0


def _model_scorecard(args: argparse.Namespace) -> int:
    from sports_ev.backtest.scorecard import format_scorecard

    settings = get_settings()
    config = BacktestConfig(season=args.season, sport=args.sport, flat_stake=args.stake)

    if args.walk_forward:
        if args.sport == "mlb":
            from sports_ev.models.mlb_strategy import WalkForwardMlbModelStrategy

            strategy = WalkForwardMlbModelStrategy(
                soft_book=args.book,
                min_train_games=settings.mlb_model_min_train_games,
            )
        else:
            strategy = WalkForwardModelStrategy(
                soft_book=args.book,
                min_train_games=settings.model_min_train_games,
            )
    elif args.sport == "mlb":
        from sports_ev.models.mlb_artifact import MlbMoneylineModel
        from sports_ev.models.mlb_strategy import ModelMoneylineStrategy

        model = MlbMoneylineModel.load(args.model_path or settings.mlb_model_artifact_path)
        strategy = ModelMoneylineStrategy(model=model, soft_book=args.book)
    else:
        model = NflSpreadModel.load(args.model_path or settings.model_artifact_path)
        strategy = ModelSpreadStrategy(model=model, soft_book=args.book)

    with session_scope(settings=settings) as session:
        report = WalkForwardBacktester(session, settings).run(strategy=strategy, config=config)

    print(format_scorecard(report, settings=settings))
    return 0


def _train_model(args: argparse.Namespace) -> int:
    settings = get_settings()

    if args.sport == "mlb":
        from sports_ev.models.mlb_dataset import build_mlb_labeled_rows
        from sports_ev.models.mlb_train import train_mlb_and_save

        output = args.output or settings.mlb_model_artifact_path
        with session_scope(settings=settings) as session:
            rows = build_mlb_labeled_rows(session, season=args.season)
            if not rows:
                print("No labeled MLB training rows found. Run backfill-mlb-stats first.", file=sys.stderr)
                return 1
            model, metrics = train_mlb_and_save(rows, output)
    else:
        output = args.output or settings.model_artifact_path
        with session_scope(settings=settings) as session:
            rows = build_labeled_rows(session, season=args.season)
            if not rows:
                print("No labeled training rows found.", file=sys.stderr)
                return 1
            model, metrics = train_and_save(rows, output)

    print(f"model_version={model.model_version} artifact={output}")
    print(
        f"train_rows={metrics.train_rows} validation_rows={metrics.validation_rows} "
        f"raw_val_brier={metrics.raw_val_brier} calibrated_val_brier={metrics.calibrated_val_brier}"
    )
    return 0


def _backfill_mlb_stats(args: argparse.Namespace) -> int:
    settings = get_settings()
    from sports_ev.stats.mlb_backfill import MlbStatsBackfillService

    with session_scope(settings=settings) as session:
        service = MlbStatsBackfillService(session)
        try:
            result = service.backfill_from_schedule(
                lookback_days=args.lookback_days,
                fetch_boxscores=args.with_boxscores,
            )
        finally:
            service.close()

    print(
        f"games_checked={result.games_checked} games_matched={result.games_matched} "
        f"stats_inserted={result.stats_inserted}"
    )
    for error in result.errors:
        print(f"  error: {error}", file=sys.stderr)
    return 0 if not result.errors else 1


def _backfill_nfl_history(args: argparse.Namespace) -> int:
    settings = get_settings()
    from sports_ev.stats.nfl_backfill import NflHistoryBackfillService

    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    with session_scope(settings=settings) as session:
        result = NflHistoryBackfillService(session).backfill_seasons(seasons)

    print(
        f"games_seen={result.games_seen} games_upserted={result.games_upserted} "
        f"odds_inserted={result.odds_inserted} stats_inserted={result.stats_inserted}"
    )
    for error in result.errors[:10]:
        print(f"  error: {error}", file=sys.stderr)
    return 0 if not result.errors else 1


def _sync_mlb_probables(args: argparse.Namespace) -> int:
    settings = get_settings()
    from sports_ev.stats.mlb_probables import MlbProbableSyncService

    with session_scope(settings=settings) as session:
        service = MlbProbableSyncService(session)
        try:
            if args.historical:
                result = service.backfill_from_schedule(lookback_days=args.lookback_days)
            else:
                result = service.sync_upcoming(days_ahead=args.days)
        finally:
            service.close()

    print(
        f"games_checked={result.games_checked} games_updated={result.games_updated} "
        f"stats_inserted={result.stats_inserted}"
    )
    for error in result.errors:
        print(f"  error: {error}", file=sys.stderr)
    return 0 if not result.errors else 1


def _predict_model(args: argparse.Namespace) -> int:
    settings = get_settings()
    with session_scope(settings=settings) as session:
        from sports_ev.db.models import Game
        from sports_ev.features.registry import build_features_for_game

        game = session.query(Game).filter_by(game_id=args.game_id).one()
        vector = build_features_for_game(session, args.game_id, validate_leakage=True)

        if game.sport == "mlb":
            from sports_ev.models.mlb_artifact import MlbMoneylineModel

            model_path = args.model_path or settings.mlb_model_artifact_path
            model = MlbMoneylineModel.load(model_path)
            result = model.predict_one(vector.features, compute_shap=True)
            if args.save:
                from sports_ev.models.storage import save_mlb_prediction

                save_mlb_prediction(
                    session,
                    game_id=args.game_id,
                    market_type="moneyline",
                    model_version=model.model_version,
                    result=result,
                    predicted_at=game.kickoff_time,
                )
            print(f"model_version={model.model_version} game_id={args.game_id}")
            print(f"home_win_prob={result.home_win_prob:.4f} away_win_prob={result.away_win_prob:.4f}")
        else:
            model_path = args.model_path or settings.model_artifact_path
            model = NflSpreadModel.load(model_path)
            result = model.predict_one(vector.features, compute_shap=True)
            if args.save:
                save_prediction(
                    session,
                    game_id=args.game_id,
                    market_type="spread",
                    model_version=model.model_version,
                    result=result,
                    predicted_at=game.kickoff_time,
                )
            print(f"model_version={model.model_version} game_id={args.game_id}")
            print(f"home_cover_prob={result.home_cover_prob:.4f} away_cover_prob={result.away_cover_prob:.4f}")

        if result.shap_values:
            print("top_shap:")
            for name, value in sorted(result.shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]:
                print(f"  {name}: {value:.4f}")
    return 0


def _flag_opportunities(args: argparse.Namespace) -> int:
    settings = get_settings()
    with session_scope(settings=settings) as session:
        flagger = OpportunityFlagger(session, settings)
        result = flagger.flag_upcoming(
            sport=args.sport,
            days_ahead=args.days,
            strategy=args.strategy,
            soft_book=args.book,
            model_path=args.model_path,
        )

    print(
        f"sport={result.sport} market={result.market_type} "
        f"games_scanned={result.games_scanned} opportunities={result.opportunities_found} "
        f"bets_flagged={result.bets_flagged} skipped_existing={result.skipped_existing} "
        f"suppressed_degraded={result.suppressed_degraded}"
    )
    for error in result.errors:
        print(f"  error: {error}", file=sys.stderr)
    return 0


def _settle_paper_bets(_args: argparse.Namespace) -> int:
    settings = get_settings()
    with session_scope(settings=settings) as session:
        service = PaperBetSettlementService(session, settings)
        result = service.settle_pending()

    print(
        f"pending_checked={result.pending_checked} settled={result.settled} "
        f"still_pending={result.still_pending}"
    )
    for error in result.errors:
        print(f"  error: {error}", file=sys.stderr)
    return 0


def _sync_scores(args: argparse.Namespace) -> int:
    settings = get_settings()
    with session_scope(settings=settings) as session:
        service = ScoreIngestService(session, settings)
        try:
            if args.sport:
                payload = {args.sport: service.ingest_sport(args.sport, lookback_days=args.lookback_days)}
            else:
                payload = service.ingest_all(lookback_days=args.lookback_days)
        finally:
            service.close()

    for sport, result in payload.items():
        print(
            f"sport={sport} scores_seen={result.scores_seen} "
            f"games_updated={result.games_updated} unmatched={result.unmatched}"
        )
        for error in result.errors:
            print(f"  error: {error}", file=sys.stderr)
    return 0


def _print_action_result(result) -> int:
    print(f"{result.title}: {result.message}")
    for error in result.details.get("errors") or []:
        print(f"  error: {error}", file=sys.stderr)
    return 0 if result.success else 1


def _refresh_cycle(args: argparse.Namespace) -> int:
    from sports_ev.dashboard.actions import (
        capture_closing_lines,
        log_model_picks,
        poll_pinnacle,
        poll_softbooks,
        settle_paper_bets,
        sync_fixtures,
        sync_mlb_probables,
        sync_scores,
    )

    settings = get_settings()
    sports = list_sports() if args.all_sports else [args.sport]
    exit_code = 0

    if getattr(args, "sync_probables", False):
        print("=== refresh-cycle MLB probables ===")
        result = sync_mlb_probables(settings=settings)
        if _print_action_result(result):
            exit_code = 1

    for sport in sports:
        print(f"=== refresh-cycle sport={sport} ===")
        steps = [
            sync_fixtures(sport=sport, settings=settings),
            poll_pinnacle(sport=sport, settings=settings),
            poll_softbooks(sport=sport, settings=settings),
            log_model_picks(
                sport=sport,
                soft_book=args.book,
                days=args.days,
                only_plus_ev=not args.all_picks,
                settings=settings,
            ),
        ]
        for result in steps:
            if _print_action_result(result):
                exit_code = 1

    if not args.skip_settle:
        print("=== refresh-cycle settle ===")
        lookback = getattr(args, "score_lookback_days", 14)
        settle_steps = [sync_scores(settings=settings, lookback_days=lookback)]
        if getattr(args, "capture_closing", False):
            settle_steps.insert(0, capture_closing_lines(settings=settings))
        settle_steps.append(settle_paper_bets(settings=settings))
        for result in settle_steps:
            if _print_action_result(result):
                exit_code = 1

    return exit_code


def _paper_report(args: argparse.Namespace) -> int:
    from sports_ev.paper.report import format_paper_report

    settings = get_settings()
    with session_scope(settings=settings) as session:
        print(format_paper_report(session, days=args.days, settings=settings))
    return 0


def _run_dashboard(args: argparse.Namespace) -> int:
    from sports_ev.dashboard.runner import run_dashboard

    port = getattr(args, "port", 8501)
    idle_seconds = getattr(args, "idle_seconds", 45)
    return run_dashboard(port=port, idle_shutdown_seconds=idle_seconds)


if __name__ == "__main__":
    sys.exit(main())
