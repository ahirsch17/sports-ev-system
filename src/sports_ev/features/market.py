from __future__ import annotations

from sports_ev.config import Settings, get_settings
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.features.time_utils import ensure_utc
from sports_ev.pricing import DevigMethod, devig_two_way


def build_market_features(
    loader: FeatureDataLoader,
    ctx: FeatureContext,
    *,
    devig_method: DevigMethod | None = None,
    soft_books: list[str] | None = None,
    settings: Settings | None = None,
    market_type: str = "spread",
) -> dict[str, float | None]:
    settings = settings or get_settings()
    devig_method = devig_method or DevigMethod(settings.default_devig_method)
    soft_books = soft_books or ["draftkings", "fanduel"]

    features: dict[str, float | None] = {}

    pinnacle = loader.latest_odds_snapshot(
        ctx, book=settings.pinnacle_bookmaker, market_type=market_type
    )
    if pinnacle and pinnacle.odds_home is not None and pinnacle.odds_away is not None:
        fair_home, fair_away = devig_two_way(
            pinnacle.odds_home,
            pinnacle.odds_away,
            method=devig_method,
        )
        captured = ensure_utc(pinnacle.captured_at)
        ctx.audit.add(
            "pinnacle_fair_prob_home",
            fair_home,
            captured,
            f"odds_snapshots:pinnacle:{pinnacle.id}",
        )
        ctx.audit.add(
            "pinnacle_fair_prob_away",
            fair_away,
            captured,
            f"odds_snapshots:pinnacle:{pinnacle.id}",
        )
        features["pinnacle_fair_prob_home"] = fair_home
        features["pinnacle_fair_prob_away"] = fair_away
        if market_type == "spread":
            features["pinnacle_line"] = pinnacle.line
            if pinnacle.line is not None:
                ctx.audit.add(
                    "pinnacle_line",
                    pinnacle.line,
                    captured,
                    f"odds_snapshots:pinnacle:{pinnacle.id}",
                )
        else:
            from sports_ev.pricing import american_to_implied_prob

            features["pinnacle_implied_home"] = american_to_implied_prob(pinnacle.odds_home)
            features["pinnacle_implied_away"] = american_to_implied_prob(pinnacle.odds_away)
    else:
        fair_home = None

    for book in soft_books:
        soft = loader.latest_odds_snapshot(ctx, book=book, market_type=market_type)
        if soft is None or soft.odds_home is None:
            continue

        captured = ensure_utc(soft.captured_at)
        ctx.audit.add(
            f"{book}_odds_home",
            float(soft.odds_home),
            captured,
            f"odds_snapshots:{book}:{soft.id}",
        )

        if fair_home is not None:
            from sports_ev.pricing import american_to_implied_prob

            soft_implied = american_to_implied_prob(soft.odds_home)
            divergence = fair_home - soft_implied
            ctx.audit.add(
                f"{book}_pinnacle_divergence",
                divergence,
                captured,
                f"odds_snapshots:{book}:{soft.id}",
            )
            features[f"{book}_pinnacle_divergence"] = divergence

        if market_type == "spread":
            features[f"{book}_line"] = soft.line
            if soft.line is not None:
                ctx.audit.add(
                    f"{book}_line",
                    soft.line,
                    captured,
                    f"odds_snapshots:{book}:{soft.id}",
                )
            movement = _line_movement(loader, ctx, book=book, market_type=market_type)
            features[f"{book}_line_move"] = movement
            if movement is not None:
                snaps = loader.odds_snapshots_before_kickoff(ctx, book=book, market_type=market_type)
                if len(snaps) >= 2:
                    last_captured = ensure_utc(snaps[-1].captured_at)
                    ctx.audit.add(
                        f"{book}_line_move",
                        movement,
                        last_captured,
                        f"odds_snapshots:{book}:{snaps[-1].id}",
                    )
        else:
            from sports_ev.pricing import american_to_implied_prob

            features[f"{book}_implied_home"] = american_to_implied_prob(soft.odds_home)
            if soft.odds_away is not None:
                features[f"{book}_implied_away"] = american_to_implied_prob(soft.odds_away)
            movement = _implied_movement(loader, ctx, book=book, market_type=market_type)
            features[f"{book}_implied_move"] = movement

    return features


def _implied_movement(
    loader: FeatureDataLoader,
    ctx: FeatureContext,
    *,
    book: str,
    market_type: str = "moneyline",
) -> float | None:
    from sports_ev.pricing import american_to_implied_prob

    snaps = loader.odds_snapshots_before_kickoff(ctx, book=book, market_type=market_type)
    if len(snaps) < 2:
        return None
    first, last = snaps[0], snaps[-1]
    if first.odds_home is None or last.odds_home is None:
        return None
    return american_to_implied_prob(last.odds_home) - american_to_implied_prob(first.odds_home)


def _line_movement(
    loader: FeatureDataLoader,
    ctx: FeatureContext,
    *,
    book: str,
    market_type: str = "spread",
) -> float | None:
    snaps = loader.odds_snapshots_before_kickoff(ctx, book=book, market_type=market_type)
    if len(snaps) < 2:
        return None
    first, last = snaps[0], snaps[-1]
    if first.line is None or last.line is None:
        return None
    return last.line - first.line
