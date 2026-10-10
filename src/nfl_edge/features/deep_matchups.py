"""
Deep matchup layer:
- Team style profiles (pass/rush EPA, sack pressure, etc.)
- Similar-opponent historical results weighted by key-playmaker continuity
- Player-level mismatches (WR vs CB, pass rush vs OL proxies)
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_edge.features.injuries import normalize_depth_charts


# Positions we treat as "key playmakers" for continuity checks
KEY_OFFENSE = {"QB", "WR", "RB", "TE", "OT", "G", "C"}
KEY_DEFENSE = {"CB", "S", "FS", "SS", "SAF", "DE", "DT", "LB", "OLB", "ILB", "MLB", "EDGE"}


@dataclass
class StyleVector:
    pass_epa: float = 0.0
    rush_epa: float = 0.0
    pass_rate: float = 0.55  # attempts / (attempts + carries)
    def_sacks: float = 2.0
    points: float = 22.0
    points_allowed_proxy: float = 22.0  # from team game log when available
    plays: float = 62.0  # attempts + carries rolling

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.pass_epa,
            self.rush_epa,
            self.pass_rate,
            self.def_sacks / 5.0,
            (self.points - 22.0) / 10.0,
        )


def _safe_mean(series: pl.Series, default: float = 0.0) -> float:
    if series is None or series.len() == 0:
        return default
    vals = series.drop_nulls()
    if vals.len() == 0:
        return default
    return float(vals.mean())


def build_team_style_table(team_stats: pl.DataFrame, *, window: int = 6) -> pl.DataFrame:
    """
    Rolling style features per team-game, *including* that game.
    One row per season/week/team; read it through `style_before`, which only
    looks at weeks strictly before the game being modeled.
    """
    if team_stats.is_empty():
        return pl.DataFrame()

    df = team_stats
    if "season_type" in df.columns:
        df = df.filter(pl.col("season_type") == "REG")
    need = {"season", "week", "team"}
    if not need.issubset(set(df.columns)):
        return pl.DataFrame()

    for col, default in [
        ("passing_epa", 0.0),
        ("rushing_epa", 0.0),
        ("attempts", 35.0),
        ("carries", 27.0),
        ("def_sacks", 2.0),
        ("passing_yards", 220.0),
        ("rushing_yards", 110.0),
    ]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))

    df = df.with_columns(
        pass_rate=pl.col("attempts") / (pl.col("attempts") + pl.col("carries") + 1e-6)
    ).sort(["team", "season", "week"])

    metric_cols = ["passing_epa", "rushing_epa", "pass_rate", "def_sacks", "passing_yards", "rushing_yards", "attempts", "carries"]
    for col in metric_cols:
        df = df.with_columns(
            pl.col(col)
            .rolling_mean(window_size=window, min_samples=2)
            .over("team")
            .alias(f"style_{col}")
        )

    keep = ["season", "week", "team"] + [f"style_{c}" for c in metric_cols]
    return df.select(keep)


def style_before(
    styles: pl.DataFrame, team: str, season: int, week: int
) -> StyleVector:
    if styles.is_empty():
        return StyleVector()
    prior = styles.filter(
        (pl.col("team") == team)
        & (
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") < week))
        )
    )
    if prior.is_empty():
        return StyleVector()
    row = prior.sort(["season", "week"]).tail(1).to_dicts()[0]
    return StyleVector(
        pass_epa=float(row.get("style_passing_epa") or 0.0),
        rush_epa=float(row.get("style_rushing_epa") or 0.0),
        pass_rate=float(row.get("style_pass_rate") or 0.55),
        def_sacks=float(row.get("style_def_sacks") or 2.0),
        points=22.0 + float(row.get("style_passing_epa") or 0.0) * 3
        + float(row.get("style_rushing_epa") or 0.0) * 2,
        plays=float(row.get("style_attempts") or 35.0) + float(row.get("style_carries") or 27.0),
    )


def style_distance(a: StyleVector, b: StyleVector) -> float:
    """Euclidean distance in style space (lower = more similar)."""
    ta, tb = a.as_tuple(), b.as_tuple()
    return sum((x - y) ** 2 for x, y in zip(ta, tb)) ** 0.5


def key_playmakers_from_depth(
    depth_charts: pl.DataFrame,
    team: str,
    *,
    max_depth: int = 1,
) -> dict[str, str]:
    """
    Map role -> gsis_id for current depth-chart keys.
    Roles: QB1, WR1, WR2, RB1, TE1, CB1, EDGE1
    """
    if depth_charts.is_empty():
        return {}
    dc = normalize_depth_charts(depth_charts) if "depth_team" not in depth_charts.columns or "position" not in depth_charts.columns else depth_charts
    # If already normalized earlier, filter team
    if "team" not in dc.columns:
        return {}
    team_dc = dc.filter(pl.col("team") == team)
    if team_dc.is_empty():
        return {}

    roles: dict[str, str] = {}
    # sort by depth
    depth_col = "depth_team" if "depth_team" in team_dc.columns else None
    rows = team_dc.sort(depth_col if depth_col else "gsis_id").to_dicts()

    wr_i = 0
    for row in rows:
        gid = row.get("gsis_id")
        if not gid:
            continue
        pos = str(row.get("position") or "").upper()
        depth = int(row.get("depth_team") or 99)
        if depth > max_depth + 1:
            continue
        if pos.startswith("QB") and "QB1" not in roles and depth <= 1:
            roles["QB1"] = str(gid)
        elif pos.startswith("WR"):
            if depth <= 2:
                wr_i += 1
                key = f"WR{wr_i}"
                if key not in roles and wr_i <= 2:
                    roles[key] = str(gid)
        elif pos in {"RB", "HB"} and "RB1" not in roles and depth <= 1:
            roles["RB1"] = str(gid)
        elif pos.startswith("TE") and "TE1" not in roles and depth <= 1:
            roles["TE1"] = str(gid)
        elif pos in {"CB", "LCB", "RCB", "NB"} and "CB1" not in roles and depth <= 1:
            roles["CB1"] = str(gid)
        elif pos in {"DE", "EDGE", "OLB", "LDE", "RDE"} and "EDGE1" not in roles and depth <= 1:
            roles["EDGE1"] = str(gid)
        elif pos in {"S", "FS", "SS", "SAF"} and "S1" not in roles and depth <= 1:
            roles["S1"] = str(gid)
    return roles


def roster_continuity(
    current_keys: dict[str, str],
    historical_keys: dict[str, str],
    *,
    role_weights: dict[str, float] | None = None,
) -> float:
    """
    0–1 score: how many of today's key roles match the same gsis_ids as in the historical game.
    QB1 weighted heaviest.
    """
    weights = role_weights or {
        "QB1": 0.40,
        "WR1": 0.15,
        "WR2": 0.08,
        "RB1": 0.12,
        "TE1": 0.05,
        "CB1": 0.10,
        "EDGE1": 0.05,
        "S1": 0.05,
    }
    total_w = 0.0
    matched_w = 0.0
    for role, w in weights.items():
        if role not in current_keys:
            continue
        total_w += w
        if historical_keys.get(role) and historical_keys[role] == current_keys[role]:
            matched_w += w
    if total_w <= 0:
        return 0.35  # unknown → mild shrinkage
    return matched_w / total_w


def _keys_from_player_stats_week(
    player_stats: pl.DataFrame,
    team: str,
    season: int,
    week: int,
) -> dict[str, str]:
    """Infer key playmakers from that week's usage when depth chart snapshot missing."""
    if player_stats.is_empty():
        return {}
    df = player_stats.filter(
        (pl.col("team") == team) & (pl.col("season") == season) & (pl.col("week") == week)
    )
    if df.is_empty():
        return {}
    # id column
    id_col = "player_id" if "player_id" in df.columns else ("gsis_id" if "gsis_id" in df.columns else None)
    if id_col is None:
        return {}
    roles: dict[str, str] = {}
    if "passing_yards" in df.columns:
        qb = df.sort("passing_yards", descending=True).head(1)
        if qb.height and float(qb["passing_yards"][0] or 0) > 0:
            roles["QB1"] = str(qb[id_col][0])
    if "receiving_yards" in df.columns:
        wrs = df.filter(pl.col("position").cast(pl.Utf8).str.to_uppercase().str.starts_with("WR")).sort(
            "receiving_yards", descending=True
        )
        for i, row in enumerate(wrs.head(2).iter_rows(named=True), start=1):
            roles[f"WR{i}"] = str(row[id_col])
    if "rushing_yards" in df.columns:
        rbs = df.filter(pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(["RB", "HB"])).sort(
            "rushing_yards", descending=True
        )
        if rbs.height:
            roles["RB1"] = str(rbs[id_col][0])
    return roles


def similar_opponent_signal(
    *,
    team: str,
    opponent: str,
    season: int,
    week: int,
    team_games: pl.DataFrame,
    styles: pl.DataFrame,
    player_stats: pl.DataFrame,
    depth_charts: pl.DataFrame,
    current_keys: dict[str, str] | None = None,
    top_k: int = 8,
    max_distance: float = 1.35,
    margin_scale: float = 0.16,
) -> dict[str, float | str]:
    """
    Look at team's past games vs opponents with a similar style to *this* opponent.
    Weight each historical result by roster continuity of key playmakers.

    Exact rematches and high-continuity comps get stronger weight (film proxy).
    Returns margin/points adjustments and a short note.
    """
    empty = {
        "sim_margin_adj": 0.0,
        "sim_points_adj": 0.0,
        "sim_n": 0.0,
        "sim_avg_continuity": 0.0,
        "sim_note": "",
    }
    if team_games.is_empty():
        return empty

    opp_style = style_before(styles, opponent, season, week)
    current_keys = current_keys or key_playmakers_from_depth(depth_charts, team)

    # Prefer recent seasons for film relevance (still allow older if needed)
    history = team_games.filter(
        (pl.col("team") == team)
        & (
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") < week))
        )
        & (pl.col("season") >= season - 4)
    )
    if history.is_empty():
        return empty

    scored: list[tuple[float, float, float, float, str, bool]] = []
    # (distance, continuity, margin, points_for, opp, exact_rematch)
    for row in history.iter_rows(named=True):
        past_opp = str(row["opp"])
        # style of that opponent *entering that game*
        past_opp_style = style_before(styles, past_opp, int(row["season"]), int(row["week"]))
        dist = style_distance(opp_style, past_opp_style)
        exact = past_opp == opponent
        if exact:
            dist = max(0.0, dist - 0.25)
        if dist > max_distance and not exact:
            continue

        hist_keys = _keys_from_player_stats_week(
            player_stats, team, int(row["season"]), int(row["week"])
        )
        cont = roster_continuity(current_keys, hist_keys)
        # Recency: more recent games weigh slightly more via distance penalty
        age = max(0, season - int(row["season"]))
        dist = dist + 0.08 * age

        scored.append(
            (
                dist,
                cont,
                float(row["margin"]),
                float(row["points_for"]),
                past_opp,
                exact,
            )
        )

    if not scored:
        return empty

    scored.sort(key=lambda x: (0 if x[5] else 1, x[0], -x[1]))
    top = scored[:top_k]

    # weight = continuity / (1 + distance), with rematch boost
    weights = []
    margins = []
    points = []
    notes = []
    for dist, cont, margin, pf, past_opp, exact in top:
        w = (0.20 + 0.80 * cont) / (1.0 + dist)
        if exact:
            w *= 1.35
        weights.append(w)
        margins.append(margin)
        points.append(pf)
        tag = f"{past_opp}@{cont:.0%}"
        if exact:
            tag += "*"
        notes.append(tag)

    wsum = sum(weights) or 1.0
    avg_margin = sum(m * w for m, w in zip(margins, weights)) / wsum
    avg_points = sum(p * w for p, w in zip(points, weights)) / wsum
    avg_cont = sum(c for _, c, _, _, _, _ in top) / len(top)
    n_exact = sum(1 for row in top if row[5])

    # shrink toward 0 when few comps or low continuity; rematches relax shrinkage
    shrink = min(1.0, (len(top) + n_exact) / 5.0) * (0.35 + 0.65 * avg_cont)
    sim_margin_adj = max(-4.5, min(4.5, avg_margin * margin_scale * shrink))
    sim_points_adj = max(-2.5, min(2.5, (avg_points - 22.0) * 0.09 * shrink))

    return {
        "sim_margin_adj": round(sim_margin_adj, 3),
        "sim_points_adj": round(sim_points_adj, 3),
        "sim_n": float(len(top)),
        "sim_avg_continuity": round(avg_cont, 3),
        "sim_note": ",".join(notes[:4]),
    }


def player_mismatch_adjustments(
    *,
    offense_team: str,
    defense_team: str,
    season: int,
    week: int,
    player_stats: pl.DataFrame,
    depth_charts: pl.DataFrame,
    styles: pl.DataFrame,
) -> dict[str, float | str]:
    """
    Holistic + player proxies (clamped — game EPA is large; we normalize).
    """
    off = style_before(styles, offense_team, season, week)
    deff = style_before(styles, defense_team, season, week)

    # Normalize game-level EPA (~std 5–6) into roughly [-1.5, 1.5]
    pass_epa_n = max(-1.5, min(1.5, off.pass_epa / 8.0))
    rush_epa_n = max(-1.5, min(1.5, off.rush_epa / 5.0))
    sack_n = max(-1.5, min(1.5, (deff.def_sacks - 2.2) / 1.5))

    # Strong pass offense vs weak pass rush → positive; vs elite pass rush → negative
    pass_vs_pressure = pass_epa_n - 0.35 * sack_n
    # Run edge: good rush EPA, slightly helped when defense is pass-rush oriented
    rush_edge = rush_epa_n + 0.15 * sack_n

    points_adj = pass_vs_pressure * 1.8 + rush_edge * 1.2
    pass_yards_pct = pass_vs_pressure * 0.03
    rush_yards_pct = rush_edge * 0.035

    notes = []
    if pass_vs_pressure > 0.35:
        notes.append("pass_mismatch+")
    elif pass_vs_pressure < -0.35:
        notes.append("pass_mismatch-")
    if rush_edge > 0.35:
        notes.append("rush_mismatch+")

    off_keys = key_playmakers_from_depth(depth_charts, offense_team)
    def_keys = key_playmakers_from_depth(depth_charts, defense_team)
    if off_keys.get("WR1") and not def_keys.get("CB1"):
        points_adj += 0.5
        pass_yards_pct += 0.025
        notes.append("WR1_vs_no_CB1")
    elif off_keys.get("WR1") and def_keys.get("CB1"):
        notes.append("WR1_vs_CB1")

    wr_boost = _wr1_recent_yards(player_stats, offense_team, season, week)
    if wr_boost >= 80 and pass_vs_pressure > 0.1:
        points_adj += 0.35
        notes.append("hot_WR1")

    # Hard clamp — deep layer is a nudge, not a rewrite of the base model
    points_adj = max(-3.5, min(3.5, points_adj))
    pass_yards_pct = max(-0.12, min(0.12, pass_yards_pct))
    rush_yards_pct = max(-0.12, min(0.12, rush_yards_pct))

    return {
        "mm_points": round(points_adj, 3),
        "mm_pass_pct": round(pass_yards_pct, 4),
        "mm_rush_pct": round(rush_yards_pct, 4),
        "mm_note": "|".join(notes),
    }


def _wr1_recent_yards(
    player_stats: pl.DataFrame, team: str, season: int, week: int
) -> float:
    if player_stats.is_empty() or "receiving_yards" not in player_stats.columns:
        return 0.0
    df = player_stats.filter(
        (pl.col("team") == team)
        & (
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") < week))
        )
        & pl.col("position").cast(pl.Utf8).str.to_uppercase().str.starts_with("WR")
    )
    if df.is_empty():
        return 0.0
    # last 3 games for top WR
    top = (
        df.sort(["season", "week"])
        .group_by("player_id" if "player_id" in df.columns else "player_name")
        .agg(pl.col("receiving_yards").tail(3).mean().alias("ry"))
        .sort("ry", descending=True)
    )
    if top.is_empty():
        return 0.0
    return float(top["ry"][0] or 0.0)


def build_deep_matchup_features(
    schedules: pl.DataFrame,
    team_stats: pl.DataFrame,
    player_stats: pl.DataFrame,
    depth_charts: pl.DataFrame,
    team_games: pl.DataFrame,
    weights: dict | None = None,
) -> pl.DataFrame:
    """Per-game deep matchup adjustments for home and away offenses."""
    styles = build_team_style_table(team_stats) if not team_stats.is_empty() else pl.DataFrame()
    dc = normalize_depth_charts(depth_charts) if not depth_charts.is_empty() else depth_charts
    cfg = (weights or {}).get("deep_matchup") or {}
    top_k = int(cfg.get("top_k_similar", 8))
    max_dist = float(cfg.get("max_style_distance", 1.35))
    margin_scale = float(cfg.get("sim_margin_scale", 0.16))

    rows = []
    for r in schedules.iter_rows(named=True):
        season = int(r["season"])
        week = int(r["week"])
        home, away = str(r["home_team"]), str(r["away_team"])

        home_keys = key_playmakers_from_depth(dc, home)
        away_keys = key_playmakers_from_depth(dc, away)

        # How has HOME adapted to styles like AWAY?
        home_sim = similar_opponent_signal(
            team=home,
            opponent=away,
            season=season,
            week=week,
            team_games=team_games,
            styles=styles,
            player_stats=player_stats,
            depth_charts=dc,
            current_keys=home_keys,
            top_k=top_k,
            max_distance=max_dist,
            margin_scale=margin_scale,
        )
        away_sim = similar_opponent_signal(
            team=away,
            opponent=home,
            season=season,
            week=week,
            team_games=team_games,
            styles=styles,
            player_stats=player_stats,
            depth_charts=dc,
            current_keys=away_keys,
            top_k=top_k,
            max_distance=max_dist,
            margin_scale=margin_scale,
        )

        home_mm = player_mismatch_adjustments(
            offense_team=home,
            defense_team=away,
            season=season,
            week=week,
            player_stats=player_stats,
            depth_charts=dc,
            styles=styles,
        )
        away_mm = player_mismatch_adjustments(
            offense_team=away,
            defense_team=home,
            season=season,
            week=week,
            player_stats=player_stats,
            depth_charts=dc,
            styles=styles,
        )

        # Net margin adj: home sim margin - away sim margin, plus offensive mismatches
        net_margin = (
            float(home_sim["sim_margin_adj"])
            - float(away_sim["sim_margin_adj"])
            + float(home_mm["mm_points"])
            - float(away_mm["mm_points"])
        )

        rows.append(
            {
                "game_id": r["game_id"],
                "home_sim_margin_adj": home_sim["sim_margin_adj"],
                "away_sim_margin_adj": away_sim["sim_margin_adj"],
                "home_sim_continuity": home_sim["sim_avg_continuity"],
                "away_sim_continuity": away_sim["sim_avg_continuity"],
                "home_sim_n": home_sim["sim_n"],
                "away_sim_n": away_sim["sim_n"],
                "home_sim_note": home_sim["sim_note"],
                "away_sim_note": away_sim["sim_note"],
                "home_mm_points": home_mm["mm_points"],
                "away_mm_points": away_mm["mm_points"],
                "home_mm_pass_pct": home_mm["mm_pass_pct"],
                "away_mm_pass_pct": away_mm["mm_pass_pct"],
                "home_mm_rush_pct": home_mm["mm_rush_pct"],
                "away_mm_rush_pct": away_mm["mm_rush_pct"],
                "home_mm_note": home_mm["mm_note"],
                "away_mm_note": away_mm["mm_note"],
                "deep_net_margin_adj": round(net_margin, 3),
            }
        )

    if not rows:
        return pl.DataFrame({"game_id": []})
    return pl.DataFrame(rows)
