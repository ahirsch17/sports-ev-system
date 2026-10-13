"""Player prop projections (pass / rush / receiving yards)."""

from __future__ import annotations

import polars as pl

from nfl_edge.features.weather import load_weights
from nfl_edge.odds_math import prob_over_line


def _detect_cols(df: pl.DataFrame) -> dict[str, str | None]:
    cols = set(df.columns)
    team = "recent_team" if "recent_team" in cols else ("team" if "team" in cols else None)
    player = "player_display_name" if "player_display_name" in cols else (
        "player_name" if "player_name" in cols else ("player_id" if "player_id" in cols else None)
    )
    return {
        "team": team,
        "player": player,
        "pass_yards": "passing_yards" if "passing_yards" in cols else None,
        "rush_yards": "rushing_yards" if "rushing_yards" in cols else None,
        "recv_yards": "receiving_yards" if "receiving_yards" in cols else None,
        "pass_attempts": "attempts" if "attempts" in cols else ("passing_attempts" if "passing_attempts" in cols else None),
        "carries": "carries" if "carries" in cols else ("rushing_attempts" if "rushing_attempts" in cols else None),
        "position": "position" if "position" in cols else None,
    }


def rolling_player_yards(player_stats: pl.DataFrame, *, window: int = 5) -> pl.DataFrame:
    """Prior rolling means for pass/rush/recv yards per player."""
    if player_stats.is_empty():
        return player_stats

    c = _detect_cols(player_stats)
    if not c["team"] or "season" not in player_stats.columns or "week" not in player_stats.columns:
        return pl.DataFrame()

    df = player_stats
    # Prefer stable display name; avoid wiping existing player_name with nulls
    if "player_display_name" in df.columns and "player_name" in df.columns:
        df = df.with_columns(
            pl.coalesce([pl.col("player_display_name"), pl.col("player_name")]).alias("player_name")
        )
    elif "player_display_name" in df.columns:
        df = df.with_columns(pl.col("player_display_name").alias("player_name"))
    elif "player_name" not in df.columns and "player_id" in df.columns:
        df = df.with_columns(pl.col("player_id").cast(pl.Utf8).alias("player_name"))

    df = df.filter(pl.col("player_name").is_not_null())

    if c["team"] and c["team"] != "team" and c["team"] in df.columns:
        df = df.rename({c["team"]: "team"})

    if c["pass_yards"] and c["pass_yards"] != "passing_yards" and c["pass_yards"] in df.columns:
        df = df.rename({c["pass_yards"]: "passing_yards"})
    if c["rush_yards"] and c["rush_yards"] != "rushing_yards" and c["rush_yards"] in df.columns:
        df = df.rename({c["rush_yards"]: "rushing_yards"})
    if c["recv_yards"] and c["recv_yards"] != "receiving_yards" and c["recv_yards"] in df.columns:
        df = df.rename({c["recv_yards"]: "receiving_yards"})

    if "player_name" not in df.columns:
        return pl.DataFrame()

    for col in ["passing_yards", "rushing_yards", "receiving_yards"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))

    df = df.sort(["player_name", "season", "week"])

    def _rolls(lag: int, prefix: str) -> list[pl.Expr]:
        exprs = []
        for src, name in (("passing_yards", "pass"), ("rushing_yards", "rush"), ("receiving_yards", "recv")):
            col = pl.col(src).shift(lag) if lag else pl.col(src)
            exprs.append(col.rolling_mean(window_size=window, min_samples=2).over("player_name").alias(f"{prefix}roll_{name}"))
            exprs.append(col.rolling_std(window_size=window, min_samples=3).over("player_name").alias(f"{prefix}roll_{name}_std"))
        return exprs

    # roll_* = before that row's game; post_roll_* = including it (used when projecting the next game)
    df = df.with_columns(*_rolls(1, ""), *_rolls(0, "post_"))
    df = _add_receiving_usage(df)
    keep = ["season", "week", "team", "player_name"] + [
        f"{prefix}roll_{name}{suffix}"
        for prefix in ("", "post_")
        for name in ("pass", "rush", "recv")
        for suffix in ("", "_std")
    ] + [
        f"{prefix}roll_{name}"
        for prefix in ("", "post_")
        for name in ("recv_share", "recv_ypt", "usage_collapsed", "season_ypg")
    ]
    if "position" in df.columns:
        keep.append("position")
    if "player_id" in df.columns:
        keep.append("player_id")
    return df.select([k for k in keep if k in df.columns])


def _weighted_recent(col: str, *, before_row: bool) -> pl.Expr:
    """Newest-weighted mean of the last three games. Null games are left out."""
    start = 1 if before_row else 0
    weights = (2.25, 1.5, 1.0)
    num = pl.lit(0.0)
    den = pl.lit(0.0)
    for offset, weight in enumerate(weights):
        value = pl.col(col).shift(start + offset).over("player_name")
        num = num + pl.when(value.is_not_null()).then(value * weight).otherwise(0.0)
        den = den + pl.when(value.is_not_null()).then(weight).otherwise(0.0)
    return pl.when(den > 0).then(num / den).otherwise(None)


def _add_receiving_usage(df: pl.DataFrame) -> pl.DataFrame:
    """Recent target share, not the season yardage mean.

    A two-game spike stays in a five-game yardage average after the targets
    disappear. The last two games both at 3 targets or fewer is a collapse.
    """
    if "targets" not in df.columns:
        return df
    df = df.with_columns(pl.col("targets").fill_null(0.0).cast(pl.Float64))
    attempts = pl.col("attempts").fill_null(0.0).cast(pl.Float64) if "attempts" in df.columns else pl.lit(0.0)
    df = df.with_columns(attempts.alias("_att"))
    team_week = df.group_by(["season", "week", "team"]).agg(
        pl.col("_att").sum().alias("_team_att"),
        pl.col("targets").sum().alias("_team_tgt"),
    )
    df = df.join(team_week, on=["season", "week", "team"], how="left")
    df = df.with_columns(
        pl.when(pl.col("_team_att") > 0).then(pl.col("_team_att")).otherwise(pl.col("_team_tgt")).alias("_team_attempts"),
        pl.when(pl.col("targets") > 0).then(pl.col("receiving_yards") / pl.col("targets")).otherwise(None).alias("_ypt"),
    )
    df = df.with_columns(
        pl.when(pl.col("_team_attempts") > 0)
        .then(pl.col("targets") / pl.col("_team_attempts"))
        .otherwise(0.0)
        .alias("_tgt_share")
    )
    current = pl.col("targets")
    previous = pl.col("targets").shift(1).over("player_name")
    before_that = pl.col("targets").shift(2).over("player_name")
    season_incl = pl.col("receiving_yards").cum_sum().over(["player_name", "season"]) / pl.col(
        "receiving_yards"
    ).cum_count().over(["player_name", "season"])
    df = df.with_columns(
        _weighted_recent("_tgt_share", before_row=True).alias("roll_recv_share"),
        _weighted_recent("_tgt_share", before_row=False).alias("post_roll_recv_share"),
        _weighted_recent("_ypt", before_row=True).alias("roll_recv_ypt"),
        _weighted_recent("_ypt", before_row=False).alias("post_roll_recv_ypt"),
        (previous.is_not_null() & before_that.is_not_null() & (previous <= 3) & (before_that <= 3)).alias(
            "roll_usage_collapsed"
        ),
        (current.is_not_null() & previous.is_not_null() & (current <= 3) & (previous <= 3)).alias(
            "post_roll_usage_collapsed"
        ),
        season_incl.shift(1).over(["player_name", "season"]).alias("roll_season_ypg"),
        season_incl.alias("post_roll_season_ypg"),
    )
    return df.drop([c for c in ("_att", "_team_att", "_team_tgt", "_team_attempts", "_ypt", "_tgt_share") if c in df.columns])


def _featured_index(shares: list[float]) -> int | None:
    """A featured receiver needs a clear lead. A committee has no playmaker."""
    if not shares or max(shares) <= 0:
        return None
    order = sorted(range(len(shares)), key=lambda index: shares[index], reverse=True)
    top = shares[order[0]]
    second = shares[order[1]] if len(order) > 1 else 0.0
    if second <= 0 or top >= second * 1.35:
        return order[0]
    return None


def reconcile_receiving(rows: list[dict], team_pass_yards: float) -> list[dict]:
    """One pass budget per team, and at most one playmaker.

    Target shares are scaled so they sum to at most 1. Yards are capped by the
    team pass total. Only a clear usage lead is featured; everyone else is held
    to 55% of that lead. A collapsed target share cannot be the playmaker.
    """
    if not rows:
        return []
    attempts = max(float(team_pass_yards), 0.0) / 7.1
    shares = [max(float(row.get("roll_recv_share") or 0.0), 0.0) for row in rows]
    total = sum(shares)
    if total > 1:
        shares = [share / total for share in shares]
    raw = []
    for row, share in zip(rows, shares):
        ypt = float(row.get("roll_recv_ypt") or 0.0)
        raw.append(max(attempts, 0.0) * share * ypt)
    ceiling = max(float(team_pass_yards), 0.0)
    if ceiling > 0 and sum(raw) > ceiling:
        factor = ceiling / sum(raw)
        raw = [yards * factor for yards in raw]
    featured = _featured_index(shares)
    if featured is not None and not row_collapsed(rows[featured]):
        lead = raw[featured]
        raw = [yards if index == featured else min(yards, lead * 0.55) for index, yards in enumerate(raw)]
    out = []
    for index, row in enumerate(rows):
        collapsed = row_collapsed(row)
        is_featured = index == featured and not collapsed
        season = float(row.get("roll_season_ypg") or 0.0)
        yards = raw[index]
        lean = None
        if is_featured:
            lean = "featured"
        elif collapsed and yards + 12 < season:
            lean = "under"
        updated = dict(row)
        updated.update(
            {
                "mu_recv": round(yards, 1),
                "recv_share": round(shares[index], 3),
                "role": "featured" if is_featured else "secondary",
                "lean": lean,
                "season_ypg": round(season, 1),
            }
        )
        out.append(updated)
    return out


def _te_mult(game_row: dict, team: str, home: str, position: object) -> float:
    """Tight ends shrink further when this defense takes that position away."""
    if not str(position or "").upper().startswith("TE"):
        return 1.0
    key = "home_te_mult" if team == home else "away_te_mult"
    try:
        value = float(game_row.get(key) or 1.0)
    except (TypeError, ValueError):
        return 1.0
    if value != value or value <= 0:
        return 1.0
    return value


def row_collapsed(row: dict) -> bool:
    return bool(row.get("roll_usage_collapsed"))


def listed_receiving(rows: list[dict]) -> list[dict]:
    """The receiving bets for one team: one playmaker, plus any under from a target drop."""
    recv = [row for row in rows if str(row.get("market")) == "player_reception_yds"]
    if not recv:
        return []
    if any(row.get("role") for row in recv):
        featured = [row for row in recv if row.get("role") == "featured"][:1]
        chosen = {row.get("player") for row in featured}
        unders = [row for row in recv if row.get("lean") == "under" and row.get("player") not in chosen]
        return featured + unders
    recv.sort(key=lambda row: float(row.get("mu") or 0), reverse=True)
    return recv[:1]


def is_extra_playmaker(row: dict, picked: list[dict]) -> bool:
    """A receiving row that is not this team's one playmaker."""
    if str(row.get("market")) != "player_reception_yds" or row.get("lean") == "under":
        return False
    if row.get("role") == "secondary":
        return True
    return any(
        item.get("team") == row.get("team")
        and str(item.get("market")) == "player_reception_yds"
        and item.get("lean") != "under"
        for item in picked
    )


def _depth_starters(
    depth_charts: pl.DataFrame | None,
    team: str,
    *,
    positions: set[str] | None = None,
    max_depth: int = 2,
) -> dict[str, int]:
    """Map normalized player name -> depth rank for current roster."""
    if depth_charts is None or depth_charts.is_empty():
        return {}
    df = depth_charts.filter(pl.col("team") == team)
    if df.is_empty():
        return {}
    out: dict[str, int] = {}
    for r in df.iter_rows(named=True):
        pos = str(r.get("position") or "").upper()
        if positions:
            ok = False
            for p in positions:
                if pos == p or pos.startswith(p):
                    ok = True
                    break
            if not ok:
                continue
        rank = int(r.get("depth_team") or 99)
        if rank > max_depth:
            continue
        name = str(r.get("full_name") or r.get("player_name") or "").strip()
        if not name:
            continue
        key = name.lower()
        prev = out.get(key, 99)
        if rank < prev:
            out[key] = rank
    return out


def _name_in_depth(player: str, depth: dict[str, int]) -> int | None:
    if not depth:
        return None
    key = str(player or "").strip().lower()
    if key in depth:
        return depth[key]
    # soft match: last name
    last = key.split()[-1] if key else ""
    hits = [(n, d) for n, d in depth.items() if n.split()[-1] == last]
    if len(hits) == 1:
        return hits[0][1]
    return None


def project_props_for_game(
    *,
    game_row: dict,
    player_rolls: pl.DataFrame,
    depth_charts: pl.DataFrame | None = None,
    out_players: set[str] | None = None,
    top_n_pass: int = 1,
    top_n_rush: int = 3,
    top_n_recv: int = 3,
) -> list[dict]:
    """
    Build prop projections for a single game using rolling player means,
    scaled toward team pass/rush yard expectations.

    Only current depth-chart players are eligible (avoids stale ex-QBs like
    Winston still showing under CLE from old seasons).
    Skips out_players (Out / QB DNP) so ruled-out QBs are not projected.
    """
    if player_rolls.is_empty():
        return []

    season = int(game_row["season"])
    week = int(game_row["week"])
    home = game_row["home_team"]
    away = game_row["away_team"]
    out_l = {str(n).lower() for n in (out_players or set()) if n}
    # Prefer same-season / very recent rows for this team only
    week_df = player_rolls.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & pl.col("team").is_in([home, away])
        & pl.col("player_name").is_not_null()
    )
    if week_df.is_empty() or week_df.filter(
        pl.col("roll_pass").is_not_null() | pl.col("roll_rush").is_not_null() | pl.col("roll_recv").is_not_null()
    ).is_empty():
        # fall back to latest prior week for each team/player — same season first
        week_df = player_rolls.filter(
            (
                (pl.col("season") == season)
                | ((pl.col("season") == season - 1) & (pl.col("week") >= 10))
            )
            & (
                (pl.col("season") < season)
                | ((pl.col("season") == season) & (pl.col("week") < week))
            )
            & pl.col("team").is_in([home, away])
            & pl.col("player_name").is_not_null()
        )
        if week_df.is_empty():
            return []
        week_df = week_df.sort(["season", "week"]).group_by(["team", "player_name"]).tail(1)
        week_df = week_df.with_columns(
            [pl.col(c).alias(c.removeprefix("post_")) for c in week_df.columns if c.startswith("post_roll_")]
        )

    props: list[dict] = []
    for team, team_pass_pred, team_rush_pred in [
        (home, float(game_row.get("pred_home_pass_yds") or 230), float(game_row.get("pred_home_rush_yds") or 115)),
        (away, float(game_row.get("pred_away_pass_yds") or 230), float(game_row.get("pred_away_rush_yds") or 115)),
    ]:
        tdf = week_df.filter(pl.col("team") == team)
        if out_l:
            tdf = tdf.filter(~pl.col("player_name").str.to_lowercase().is_in(list(out_l)))
        if tdf.is_empty():
            continue

        qb_depth = _depth_starters(depth_charts, team, positions={"QB"}, max_depth=2)
        skill_depth = _depth_starters(
            depth_charts, team, positions={"RB", "HB", "FB", "WR", "TE"}, max_depth=3
        )

        # Pass props: current depth-chart QB1 only (never stale ex-starters)
        passers = tdf.filter(
            pl.col("roll_pass").is_not_null()
            & (
                pl.col("position").is_null()
                | pl.col("position").cast(pl.Utf8).str.to_uppercase().str.starts_with("QB")
            )
        )
        if qb_depth:
            keep_names = []
            for r in passers.iter_rows(named=True):
                d = _name_in_depth(str(r["player_name"]), qb_depth)
                if d is not None and d <= 1:
                    keep_names.append(r["player_name"])
            if keep_names:
                passers = passers.filter(pl.col("player_name").is_in(keep_names))
            else:
                # Depth QB1 may have no recent stats (return from injury) — synthesize later
                passers = passers.head(0)
        else:
            # No depth chart: require meaningful recent volume AND same-season row
            passers = passers.filter(
                (pl.col("roll_pass") >= 80) & (pl.col("season") >= season - 0)
            ).sort("roll_pass", descending=True).head(1)

        if passers.height > 1:
            passers = passers.sort("roll_pass", descending=True).head(1)

        # If depth says Watson but no roll row, create a projection from team pass yards
        if passers.height == 0 and qb_depth:
            qb1_display = None
            if depth_charts is not None and not depth_charts.is_empty():
                qb_rows = depth_charts.filter(
                    (pl.col("team") == team)
                    & (pl.col("depth_team") <= 1)
                    & pl.col("position").cast(pl.Utf8).str.to_uppercase().str.starts_with("QB")
                )
                if not qb_rows.is_empty():
                    qb1_display = str(qb_rows["full_name"][0])
            if not qb1_display:
                qb1_display = next((n.title() for n, d in qb_depth.items() if d <= 1), None)
            if qb1_display:
                hist = player_rolls.filter(
                    (pl.col("team") == team)
                    & (pl.col("season") >= season - 1)
                    & pl.col("player_name").str.to_lowercase().str.contains(
                        qb1_display.split()[-1].lower()
                    )
                ).sort(["season", "week"], descending=True).head(1)
                if not hist.is_empty():
                    passers = hist
                else:
                    props.append(
                        {
                            "game_id": game_row["game_id"],
                            "team": team,
                            "opponent": away if team == home else home,
                            "player": qb1_display,
                            "market": "player_pass_yds",
                            "mu": round(team_pass_pred * 0.92, 1),
                            "sigma": 55.0,
                            "roll": round(team_pass_pred * 0.85, 1),
                            "edge_vs_roll": round(team_pass_pred * 0.07, 1),
                            "priority": 1,
                            "from_depth": True,
                        }
                    )

        rushers = tdf.filter(
            pl.col("roll_rush").is_not_null()
            & (pl.col("roll_rush") >= 15)
            & (
                pl.col("position").is_null()
                | pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(["RB", "HB", "FB", "QB"])
            )
        )
        if skill_depth:
            keep = [
                r["player_name"]
                for r in rushers.iter_rows(named=True)
                if _name_in_depth(str(r["player_name"]), skill_depth) is not None
                or _name_in_depth(str(r["player_name"]), qb_depth) is not None
            ]
            if keep:
                rushers = rushers.filter(pl.col("player_name").is_in(keep))
        rushers = rushers.sort("roll_rush", descending=True).head(top_n_rush)

        share_ready = "roll_recv_share" in tdf.columns
        if share_ready:
            receivers = tdf.filter(
                (
                    pl.col("position").is_null()
                    | pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(["WR", "TE", "RB", "HB", "FB"])
                )
                & (
                    (pl.col("roll_recv_share").fill_null(0) >= 0.04)
                    | pl.col("roll_usage_collapsed").fill_null(False)
                )
            )
        else:
            receivers = tdf.filter(
                pl.col("roll_recv").is_not_null()
                & (pl.col("roll_recv") >= 25)
                & (
                    pl.col("position").is_null()
                    | pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(["WR", "TE", "RB", "HB", "FB"])
                )
            )
        if skill_depth:
            keep = [
                r["player_name"]
                for r in receivers.iter_rows(named=True)
                if _name_in_depth(str(r["player_name"]), skill_depth) is not None
            ]
            if keep:
                receivers = receivers.filter(pl.col("player_name").is_in(keep))
        if not share_ready:
            receivers = receivers.sort("roll_recv", descending=True).head(top_n_recv)

        pass_sum = float(passers["roll_pass"].sum()) or 1.0 if passers.height else 1.0
        for r in passers.iter_rows(named=True):
            share = float(r["roll_pass"]) / pass_sum
            mu = team_pass_pred * share
            if passers.height == 1:
                mu = team_pass_pred * 0.92
            roll = float(r["roll_pass"] or 0)
            if roll and roll == roll:  # not nan
                mu = 0.35 * roll + 0.65 * mu
            else:
                roll = mu
            sigma = float(r.get("roll_pass_std") or 55.0)
            sigma = max(sigma, 35.0)
            props.append(
                {
                    "game_id": game_row["game_id"],
                    "team": team,
                    "opponent": away if team == home else home,
                    "player": r["player_name"],
                    "market": "player_pass_yds",
                    "mu": round(mu, 1),
                    "sigma": round(sigma, 1),
                    "roll": round(roll, 1),
                    "edge_vs_roll": round(mu - roll, 1),
                    "priority": 1,
                }
            )

        rush_sum = float(rushers["roll_rush"].sum()) or 1.0
        for r in rushers.iter_rows(named=True):
            share = float(r["roll_rush"]) / rush_sum
            team_share_mu = team_rush_pred * min(share, 0.55)
            roll = float(r["roll_rush"])
            mu = 0.55 * roll + 0.45 * team_share_mu
            sigma = float(r.get("roll_rush_std") or 25.0)
            sigma = max(sigma, 18.0)
            props.append(
                {
                    "game_id": game_row["game_id"],
                    "team": team,
                    "opponent": away if team == home else home,
                    "player": r["player_name"],
                    "market": "player_rush_yds",
                    "mu": round(mu, 1),
                    "sigma": round(sigma, 1),
                    "roll": round(roll, 1),
                    "edge_vs_roll": round(mu - roll, 1),
                    "priority": 3,
                }
            )

        if share_ready:
            ranked = reconcile_receiving(list(receivers.iter_rows(named=True)), team_pass_pred)
            ranked.sort(
                key=lambda row: (
                    0 if row.get("role") == "featured" else 1 if row.get("lean") == "under" else 2,
                    -(row.get("mu_recv") or 0),
                )
            )
            kept = [row for row in ranked if row.get("role") == "featured" or row.get("lean") == "under"]
            rest = [row for row in ranked if row not in kept and (row.get("mu_recv") or 0) >= 8]
            ranked = (kept + rest)[: max(top_n_recv, len(kept))]
            for r in ranked:
                mu = float(r["mu_recv"]) * _te_mult(game_row, team, home, r.get("position"))
                roll = float(r.get("roll_recv") or mu)
                sigma = float(r.get("roll_recv_std") or 28.0)
                props.append(
                    {
                        "game_id": game_row["game_id"],
                        "team": team,
                        "opponent": away if team == home else home,
                        "player": r["player_name"],
                        "market": "player_reception_yds",
                        "mu": round(mu, 1),
                        "sigma": round(max(sigma, 20.0), 1),
                        "roll": round(roll, 1),
                        "edge_vs_roll": round(mu - roll, 1),
                        "priority": 2 if r.get("lean") else 4,
                        "role": r.get("role"),
                        "lean": r.get("lean"),
                        "season_ypg": r.get("season_ypg"),
                    }
                )
        else:
            recv_sum = float(receivers["roll_recv"].sum()) or 1.0
            team_recv_pool = team_pass_pred * 0.60
            for r in receivers.iter_rows(named=True):
                share = float(r["roll_recv"]) / recv_sum
                team_share_mu = team_recv_pool * min(share, 0.45)
                roll = float(r["roll_recv"])
                mu = (0.50 * roll + 0.50 * team_share_mu) * _te_mult(game_row, team, home, r.get("position"))
                sigma = float(r.get("roll_recv_std") or 28.0)
                sigma = max(sigma, 20.0)
                props.append(
                    {
                        "game_id": game_row["game_id"],
                        "team": team,
                        "opponent": away if team == home else home,
                        "player": r["player_name"],
                        "market": "player_reception_yds",
                        "mu": round(mu, 1),
                        "sigma": round(sigma, 1),
                        "roll": round(roll, 1),
                        "edge_vs_roll": round(mu - roll, 1),
                        "priority": 2,
                    }
                )

    return props


def _ruled_out(injuries: pl.DataFrame | None, season: int, week: int) -> set[str]:
    """Lower-cased names listed Out/Doubtful on that week's official injury report."""
    need = {"season", "week", "full_name", "report_status"}
    if injuries is None or injuries.is_empty() or not need.issubset(injuries.columns):
        return set()
    rows = injuries.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & pl.col("report_status").is_in(["Out", "Doubtful"])
    )
    return {str(n).lower() for n in rows["full_name"].to_list() if n}


def project_slate_props(
    predictions: pl.DataFrame,
    player_stats: pl.DataFrame,
    depth_charts: pl.DataFrame | None = None,
    injuries: pl.DataFrame | None = None,
) -> pl.DataFrame:
    from nfl_edge.features.injuries import normalize_depth_charts

    rolls = rolling_player_yards(player_stats)
    dc = normalize_depth_charts(depth_charts) if depth_charts is not None and not depth_charts.is_empty() else None
    all_props: list[dict] = []
    for r in predictions.iter_rows(named=True):
        out = _ruled_out(injuries, int(r["season"]), int(r["week"]))
        all_props.extend(
            project_props_for_game(game_row=r, player_rolls=rolls, depth_charts=dc, out_players=out)
        )
    if not all_props:
        return pl.DataFrame()
    return pl.DataFrame(all_props)


def prop_over_prob(mu: float, sigma: float, line: float) -> float:
    return prob_over_line(mu, sigma, line)


def annotate_prop_vs_line(props: pl.DataFrame, line: float, side: str = "Over") -> dict:
    """Helper for a single comparison."""
    weights = load_weights()
    _ = weights
    mu = float(props["mu"][0]) if props.height else 0.0
    sigma = float(props["sigma"][0]) if props.height else 40.0
    p_over = prop_over_prob(mu, sigma, line)
    return {"p_over": p_over, "p_under": 1 - p_over, "side_prob": p_over if side.lower().startswith("o") else 1 - p_over}
