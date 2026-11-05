"""Walk-forward matchup detail: field, momentum, QB vs similar defenses, coverage.

Every number for a game is computed from games that already happened. Home-field
is this team's home-versus-road split, shrunk toward how that stadium has
played, then toward the league. Momentum is a hot or cold streak beyond the
team's own recent level. Quarterback adjustments use only starts against
defenses that were allowing a similar passing EPA. Coverage compares the
offense's passing EPA per play with what this defense has been allowing, and
shifts the run/pass mix when the receivers are outclassed.
"""

from __future__ import annotations

from collections import deque

import polars as pl

LEAGUE_HFA = 2.0
LEAGUE_PASS_EPA = 0.03  # per attempt, roughly league average


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _shrink(value: float, n: float, prior: float, k: float) -> float:
    n = max(float(n), 0.0)
    return (n * value + k * prior) / (n + k)


def _rolling(items: deque[float], n: int) -> float | None:
    if not items:
        return None
    tail = list(items)[-n:]
    return _mean(tail)


def build_detail_features(
    schedules: pl.DataFrame,
    team_stats: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per scheduled game. Adjustments are home-margin points, pre-weight."""
    empty = pl.DataFrame({"game_id": []})
    if schedules.is_empty():
        return empty

    epa: dict[tuple[str, str], tuple[float, float]] = {}
    if team_stats is not None and not team_stats.is_empty() and {"game_id", "team"}.issubset(team_stats.columns):
        stats = team_stats
        if "season_type" in stats.columns:
            stats = stats.filter(pl.col("season_type") == "REG")
        for row in stats.select(
            [c for c in ["game_id", "team", "passing_epa", "attempts", "rushing_epa", "carries"] if c in stats.columns]
        ).iter_rows(named=True):
            att = float(row.get("attempts") or 0.0)
            carries = float(row.get("carries") or 0.0)
            pass_pp = float(row.get("passing_epa") or 0.0) / att if att >= 8 else None
            rush_pp = float(row.get("rushing_epa") or 0.0) / carries if carries >= 5 else None
            if pass_pp is None and rush_pp is None:
                continue
            epa[(str(row["game_id"]), str(row["team"]))] = (
                pass_pp if pass_pp is not None else LEAGUE_PASS_EPA,
                rush_pp if rush_pp is not None else 0.0,
            )

    games = schedules
    sort_cols = [c for c in ["season", "week", "gameday", "gametime", "game_id"] if c in games.columns]
    games = games.sort(sort_cols)

    home_sum: dict[str, float] = {}
    home_n: dict[str, int] = {}
    away_sum: dict[str, float] = {}
    away_n: dict[str, int] = {}
    margins: dict[str, deque[float]] = {}
    pass_for: dict[str, deque[float]] = {}
    pass_allowed: dict[str, deque[float]] = {}
    # Pass-defense quality (EPA allowed per attempt) each offense has actually seen.
    faced_def: dict[str, deque[float]] = {}
    stadium_sum: dict[str, float] = {}
    stadium_n: dict[str, int] = {}
    # qb id -> list of (opponent pass-epa allowed per play entering that game, own pass epa per play)
    qb_hist: dict[str, list[tuple[float, float]]] = {}

    rows: list[dict] = []
    for g in games.iter_rows(named=True):
        gid = str(g["game_id"])
        season = int(g["season"])
        week = int(g["week"])
        home = str(g["home_team"])
        away = str(g["away_team"])
        neutral = str(g.get("location") or "Home").lower() == "neutral"
        stadium = str(g.get("stadium_id") or "")

        def team_hfa(team: str) -> tuple[float, float]:
            nh, na = home_n.get(team, 0), away_n.get(team, 0)
            if nh < 3 or na < 3:
                return LEAGUE_HFA, float(min(nh, na))
            gap = (home_sum[team] / nh) - (away_sum[team] / na)
            return gap, float(min(nh, na))

        def stadium_hfa() -> tuple[float, float]:
            n = stadium_n.get(stadium, 0) if stadium else 0
            if not stadium or n < 4:
                return LEAGUE_HFA, float(n)
            return stadium_sum[stadium] / n, float(n)

        raw_team, n_team = team_hfa(home)
        raw_stad, n_stad = stadium_hfa()
        stad = _shrink(raw_stad, n_stad, LEAGUE_HFA, 12.0)
        field = 0.0 if neutral else _shrink(raw_team, n_team, stad, 8.0)
        # Baseline already books the league home edge. This is only the extra.
        field_adj = max(-3.0, min(3.0, field - LEAGUE_HFA))

        def mom(team: str) -> float:
            hist = margins.get(team)
            if not hist or len(hist) < 4:
                return 0.0
            recent = _rolling(hist, 3) or 0.0
            longer = _rolling(hist, 8) or recent
            return max(-6.0, min(6.0, recent - longer))

        momentum_adj = max(-3.0, min(3.0, mom(home) - mom(away)))

        def def_pp(team: str) -> float:
            return _rolling(pass_allowed.get(team, deque()), 6) or LEAGUE_PASS_EPA

        def off_pp(team: str) -> float:
            return _rolling(pass_for.get(team, deque()), 6) or LEAGUE_PASS_EPA

        def qb_points(qb_id: str | None, opp_def: float) -> tuple[float, int]:
            if not qb_id:
                return 0.0, 0
            hist = qb_hist.get(str(qb_id)) or []
            if len(hist) < 4:
                return 0.0, 0
            overall = _mean([own for _, own in hist])
            band = 0.08
            similar = [own for faced, own in hist if abs(faced - opp_def) <= band]
            if len(similar) < 3:
                similar = [own for faced, own in hist if abs(faced - opp_def) <= 0.12]
            if len(similar) < 3:
                return 0.0, len(similar)
            resid = _mean(similar) - overall
            shrink = min(1.0, len(similar) / 8.0)
            # per-play EPA residual across a normal dropback count, clamped
            return max(-2.5, min(2.5, resid * 28.0 * shrink)), len(similar)

        home_qb_pts, home_qb_n = qb_points(g.get("home_qb_id"), def_pp(away))
        away_qb_pts, away_qb_n = qb_points(g.get("away_qb_id"), def_pp(home))
        qb_adj = max(-3.5, min(3.5, home_qb_pts - away_qb_pts))

        def coverage(off_team: str, def_team: str) -> tuple[float, float, float]:
            # Compared with the secondaries this offense has already faced.
            # A stingier one (lower EPA allowed) shrinks the pass game and lifts the run.
            this_def = def_pp(def_team)
            seen = faced_def.get(off_team)
            usual = _rolling(seen, 6) if seen and len(seen) >= 3 else this_def
            gap = this_def - usual
            points = max(-1.5, min(1.5, gap * 14.0))
            pass_pct = max(-0.08, min(0.08, gap * 1.6))
            rush_pct = max(-0.08, min(0.08, -gap * 1.2))
            return points, pass_pct, rush_pct

        home_cov, home_pass, home_rush = coverage(home, away)
        away_cov, away_pass, away_rush = coverage(away, home)
        coverage_adj = max(-3.5, min(3.5, home_cov - away_cov))
        coverage_total = max(-3.0, min(3.0, (home_cov + away_cov) * 0.35))

        note_bits = []
        if abs(field_adj) >= 0.4:
            note_bits.append(f"field {field_adj:+.1f}")
        if abs(momentum_adj) >= 0.8:
            note_bits.append(f"momentum {momentum_adj:+.1f}")
        if abs(qb_adj) >= 0.4:
            note_bits.append(f"qb-vs-similar {qb_adj:+.1f} (n {home_qb_n}/{away_qb_n})")
        if home_pass <= -0.025:
            note_bits.append(f"{home} pass shrunk vs coverage")
        elif home_pass >= 0.025:
            note_bits.append(f"{home} pass boosted vs coverage")
        if away_pass <= -0.025:
            note_bits.append(f"{away} pass shrunk vs coverage")
        elif away_pass >= 0.025:
            note_bits.append(f"{away} pass boosted vs coverage")

        rows.append(
            {
                "game_id": gid,
                "season": season,
                "week": week,
                "field_margin_adj": round(field_adj, 3),
                "momentum_adj": round(momentum_adj, 3),
                "qb_sim_margin_adj": round(qb_adj, 3),
                "coverage_margin_adj": round(coverage_adj, 3),
                "coverage_total_adj": round(coverage_total, 3),
                "home_cov_pass_pct": round(home_pass, 4),
                "away_cov_pass_pct": round(away_pass, 4),
                "home_cov_rush_pct": round(home_rush, 4),
                "away_cov_rush_pct": round(away_rush, 4),
                "detail_note": "; ".join(note_bits),
            }
        )

        # Update state only with this game's result, after the features are stored.
        hs, aws = g.get("home_score"), g.get("away_score")
        if hs is not None and aws is not None:
            margin = float(hs) - float(aws)
            home_sum[home] = home_sum.get(home, 0.0) + margin
            home_n[home] = home_n.get(home, 0) + 1
            away_sum[away] = away_sum.get(away, 0.0) + (-margin)
            away_n[away] = away_n.get(away, 0) + 1
            margins.setdefault(home, deque(maxlen=8)).append(margin)
            margins.setdefault(away, deque(maxlen=8)).append(-margin)
            if stadium and not neutral:
                nh, na = home_n.get(home, 0), away_n.get(home, 0)
                road = (away_sum.get(home, 0.0) / na) if na >= 3 else 0.0
                # home just added to home_n; road sample is independent
                bump = margin - road
                stadium_sum[stadium] = stadium_sum.get(stadium, 0.0) + bump
                stadium_n[stadium] = stadium_n.get(stadium, 0) + 1

        home_epa = epa.get((gid, home))
        away_epa = epa.get((gid, away))
        if home_epa and away_epa and hs is not None:
            # Defense quality each offense walked into, before this game is added.
            faced_def.setdefault(home, deque(maxlen=6)).append(def_pp(away))
            faced_def.setdefault(away, deque(maxlen=6)).append(def_pp(home))
            pass_for.setdefault(home, deque(maxlen=6)).append(home_epa[0])
            pass_for.setdefault(away, deque(maxlen=6)).append(away_epa[0])
            pass_allowed.setdefault(home, deque(maxlen=6)).append(away_epa[0])
            pass_allowed.setdefault(away, deque(maxlen=6)).append(home_epa[0])
            home_def_prior = def_pp(away)
            away_def_prior = def_pp(home)
            # def_pp now includes this game if we appended first. Recompute prior
            # from the deque without the game we just pushed.
            def prior_mean(team: str) -> float:
                hist = list(pass_allowed.get(team, []))
                if len(hist) <= 1:
                    return LEAGUE_PASS_EPA
                return _mean(hist[:-1][-6:])

            if g.get("home_qb_id"):
                qb_hist.setdefault(str(g["home_qb_id"]), []).append((prior_mean(away), home_epa[0]))
            if g.get("away_qb_id"):
                qb_hist.setdefault(str(g["away_qb_id"]), []).append((prior_mean(home), away_epa[0]))

    if not rows:
        return empty
    return pl.DataFrame(rows)
