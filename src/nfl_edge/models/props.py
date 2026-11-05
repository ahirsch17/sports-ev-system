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


def _allocate(weights: dict[str, float], pool: float) -> dict[str, float]:
    """Split one team total across the players who actually get those yards."""
    clean = {name: float(yards) for name, yards in weights.items() if yards and float(yards) > 0}
    total = sum(clean.values())
    if total <= 0 or pool <= 0:
        return {}
    return {name: pool * yards / total for name, yards in clean.items()}


def _split_sentence(pool: float, yards: dict[str, float], label: str, rest_label: str) -> str:
    """Name the pieces that are big enough to see, and keep the integers adding up."""
    named = sorted(((name, yards) for name, yards in yards.items() if yards >= 20), key=lambda item: -item[1])[:6]
    if not named:
        return ""
    def _whole(value: float) -> int:
        return int(float(f"{round(float(value), 1):.0f}"))

    shown = [(name, _whole(yards)) for name, yards in named]
    rest = _whole(pool) - sum(yards for _, yards in shown)
    parts = [f"{name} {yds}" for name, yds in shown]
    if rest >= 8:
        parts.append(f"{rest} to {rest_label}")
    elif parts and rest:
        name, yds = shown[-1]
        parts[-1] = f"{name} {yds + rest}"
    body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"The {round(pool)} {label} are {body}."


def _recent_team_yards(
    stats: pl.DataFrame | None,
    team: str,
    season: int,
    week: int,
    n: int = 6,
) -> dict[str, dict]:
    """Yards per player over the team's last n games. This is the share, not a per-player average."""
    if stats is None or stats.is_empty():
        return {}
    if not {"season", "week", "team", "player_display_name"}.issubset(stats.columns):
        return {}
    df = stats.filter(
        (pl.col("team") == team)
        & ((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week)))
    )
    if "season_type" in df.columns:
        df = df.filter(pl.col("season_type") == "REG")
    if df.is_empty():
        return {}
    weeks = df.select(["season", "week"]).unique().sort(["season", "week"]).tail(n)
    df = df.join(weeks, on=["season", "week"])
    room: dict[str, dict] = {}
    for row in df.iter_rows(named=True):
        name = str(row.get("player_display_name") or "")
        if not name:
            continue
        slot = room.setdefault(name, {"recv": 0.0, "rush": 0.0, "pos": str(row.get("position") or "")})
        slot["recv"] += float(row.get("receiving_yards") or 0)
        slot["rush"] += float(row.get("rushing_yards") or 0)
        if row.get("position"):
            slot["pos"] = str(row["position"])
    return room


def _returning_rb1(
    depth: pl.DataFrame | None,
    injuries: pl.DataFrame | None,
    stats: pl.DataFrame | None,
    team: str,
    season: int,
    week: int,
    out_players: set[str],
) -> str | None:
    """Depth-chart RB1 who was out recently, is active now, and has barely played.

    The carries in that window belong to the fill-in. He has kept about half
    of them on the returns since 2023, so half of that work goes back to the starter.
    """
    if (
        depth is None
        or depth.is_empty()
        or injuries is None
        or injuries.is_empty()
        or "full_name" not in depth.columns
        or "full_name" not in injuries.columns
        or "report_status" not in injuries.columns
    ):
        return None
    rbs = depth.filter(
        (pl.col("team") == team)
        & pl.col("position").cast(pl.Utf8).str.to_uppercase().str.starts_with("RB")
        & (pl.col("depth_team") == 1)
    )
    if rbs.is_empty():
        return None
    name = str(rbs["full_name"][0] or "")
    if not name or name.lower() in out_players:
        return None
    recent_out = injuries.filter(
        (pl.col("season") == season)
        & (pl.col("team") == team)
        & (pl.col("week") < week)
        & (pl.col("week") >= week - 4)
        & (pl.col("full_name").str.to_lowercase() == name.lower())
        & pl.col("report_status").is_in(["Out", "Doubtful"])
    )
    if recent_out.height < 2:
        return None
    if stats is None or stats.is_empty() or "carries" not in stats.columns or "player_display_name" not in stats.columns:
        return None
    played = stats.filter(
        (pl.col("season") == season)
        & (pl.col("week") < week)
        & (pl.col("team") == team)
        & (pl.col("player_display_name").str.to_lowercase() == name.lower())
        & (pl.col("carries") > 0)
    )
    if "season_type" in played.columns:
        played = played.filter(pl.col("season_type") == "REG")
    if played.height >= 2:
        return None
    return name


def _roster_names(depth_charts: pl.DataFrame | None, team: str) -> dict[str, int]:
    names = _depth_starters(depth_charts, team, positions={"QB"}, max_depth=2)
    names.update(
        _depth_starters(depth_charts, team, positions={"RB", "HB", "FB", "WR", "TE"}, max_depth=6)
    )
    return names


def _on_roster(yards: dict[str, float], roster: dict[str, int]) -> dict[str, float]:
    if not roster or not yards:
        return yards
    kept = {name: value for name, value in yards.items() if _name_in_depth(name, roster) is not None}
    if sum(kept.values()) < 0.5 * sum(yards.values()):
        return yards
    return kept


def _one_game_pass_roll(
    player_stats: pl.DataFrame | None,
    team: str,
    display_name: str,
    season: int,
) -> pl.DataFrame | None:
    """A starter with one game has no rolling mean. Use the yards from that game."""
    if player_stats is None or player_stats.is_empty() or "passing_yards" not in player_stats.columns:
        return None
    last = display_name.split()[-1].lower()
    if not last:
        return None
    df = player_stats.filter(pl.col("team") == team)
    if "season" in df.columns:
        df = df.filter(pl.col("season") >= season - 1)
    hit = pl.lit(False)
    for column in ("player_display_name", "player_name"):
        if column in df.columns:
            hit = hit | pl.col(column).fill_null("").str.to_lowercase().str.contains(last, literal=True)
    df = df.filter(hit & pl.col("passing_yards").is_not_null())
    if "attempts" in df.columns:
        df = df.filter(pl.col("attempts").fill_null(0) >= 8)
    if df.is_empty():
        return None
    yards = [float(value) for value in df["passing_yards"].to_list()]
    return pl.DataFrame(
        {
            "player_name": [display_name],
            "position": ["QB"],
            "roll_pass": [sum(yards) / len(yards)],
            "roll_pass_std": [None],
        }
    )


def _is_out(name: str, out_l: set[str]) -> bool:
    return str(name or "").strip().lower() in out_l


def _next_qb_rank(qb_depth: dict[str, int], out_l: set[str]) -> int | None:
    """Depth rank of the first quarterback who is not out or doubtful."""
    ranks = [rank for name, rank in qb_depth.items() if not _is_out(name, out_l)]
    return min(ranks) if ranks else None


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


def blend_pass_yards(roll: float, defense_allowed: float | None, team_pace: float) -> float:
    """One passing total for the quarterback and the receivers.

    The pace number and the defense's recent total are not two bets.
    40% recent quarterback, 45% what this defense has allowed, 15% pace.
    """
    pace = float(team_pace) * 0.92
    if defense_allowed is not None and roll == roll:
        return 0.40 * float(roll) + 0.45 * float(defense_allowed) + 0.15 * pace
    if roll == roll:
        return 0.65 * float(roll) + 0.35 * pace
    return pace


def defense_shape_rooms(
    pass_yards: float,
    wr: float,
    te: float,
    rb_rec: float,
) -> dict[str, float]:
    """Split one passing total the way this defense has actually given up yards."""
    total = float(wr) + float(te) + float(rb_rec)
    if total <= 0:
        return {"WR": float(pass_yards), "TE": 0.0, "RB": 0.0}
    py = float(pass_yards)
    return {"WR": py * wr / total, "TE": py * te / total, "RB": py * rb_rec / total}


def pass_yards_allowed_by_defense(
    player_stats: pl.DataFrame | None,
    season: int,
    week: int,
) -> dict[str, float]:
    """Mean passing yards allowed by each defense before this week.

    A missing receiver does not belong in this number. His targets move to
    someone else. The quarterback's yards stay with the quarterback.
    """
    need = {"season", "week", "opponent_team", "passing_yards", "attempts", "position"}
    if player_stats is None or player_stats.is_empty() or not need.issubset(set(player_stats.columns)):
        return {}
    qbs = player_stats.filter(
        (pl.col("position") == "QB")
        & (pl.col("attempts") >= 12)
        & (pl.col("season") >= season - 1)
        & ((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week)))
    )
    if "season_type" in qbs.columns:
        qbs = qbs.filter(pl.col("season_type") == "REG")
    if qbs.is_empty():
        return {}
    grouped = qbs.group_by("opponent_team").agg(
        pl.col("passing_yards").mean().alias("yds"),
        pl.len().alias("n"),
    )
    return {
        str(r["opponent_team"]): float(r["yds"])
        for r in grouped.iter_rows(named=True)
        if r["n"] >= 3 and r["yds"] is not None
    }


def project_props_for_game(
    *,
    game_row: dict,
    player_rolls: pl.DataFrame,
    depth_charts: pl.DataFrame | None = None,
    out_players: set[str] | None = None,
    opp_pass_allowed: dict[str, float] | None = None,
    top_n_pass: int = 1,
    top_n_rush: int = 3,
    top_n_recv: int = 3,
    player_stats: pl.DataFrame | None = None,
    injuries: pl.DataFrame | None = None,
    defense_cuts: dict[str, dict] | None = None,
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
        # A new starter on the defense changes the yards allowed. The old
        # average was built before he arrived. Receivers are a share of the
        # quarterback, so the same cut moves both.
        opp = away if team == home else home
        cuts = (defense_cuts or {}).get(opp) or {}
        if cuts.get("rush"):
            team_rush_pred = max(40.0, team_rush_pred - float(cuts["rush"]))
        tdf = week_df.filter(pl.col("team") == team)
        if out_l:
            tdf = tdf.filter(~pl.col("player_name").str.to_lowercase().is_in(list(out_l)))
        if tdf.is_empty():
            continue

        qb_depth = _depth_starters(depth_charts, team, positions={"QB"}, max_depth=4)
        skill_depth = _depth_starters(
            depth_charts, team, positions={"RB", "HB", "FB", "WR", "TE"}, max_depth=3
        )
        starter_rank = _next_qb_rank(qb_depth, out_l)

        # Pass props: the first quarterback on the depth chart who is not out or doubtful.
        passers = tdf.filter(
            pl.col("roll_pass").is_not_null()
            & (
                pl.col("position").is_null()
                | pl.col("position").cast(pl.Utf8).str.to_uppercase().str.starts_with("QB")
            )
        )
        if qb_depth and starter_rank is not None:
            keep_names = []
            for r in passers.iter_rows(named=True):
                d = _name_in_depth(str(r["player_name"]), qb_depth)
                if d is not None and d == starter_rank:
                    keep_names.append(r["player_name"])
            if keep_names:
                passers = passers.filter(pl.col("player_name").is_in(keep_names))
            else:
                # The starter may have no recent stats — synthesize later
                passers = passers.head(0)
        else:
            # No depth chart: require meaningful recent volume AND same-season row
            passers = passers.filter(
                (pl.col("roll_pass") >= 80) & (pl.col("season") >= season - 0)
            ).sort("roll_pass", descending=True).head(1)

        if passers.height > 1:
            passers = passers.sort("roll_pass", descending=True).head(1)

        # If depth says Watson but no roll row, create a projection from team pass yards
        if passers.height == 0 and qb_depth and starter_rank is not None:
            qb1_display = None
            if depth_charts is not None and not depth_charts.is_empty() and "full_name" in depth_charts.columns:
                qb_rows = depth_charts.filter(
                    (pl.col("team") == team)
                    & (pl.col("depth_team") == starter_rank)
                    & pl.col("position").cast(pl.Utf8).str.to_uppercase().str.starts_with("QB")
                )
                qb_rows = qb_rows.filter(
                    ~pl.col("full_name").str.to_lowercase().is_in(list(out_l or {""}))
                )
                if not qb_rows.is_empty():
                    qb1_display = str(qb_rows["full_name"][0])
            if not qb1_display:
                qb1_display = next(
                    (n.title() for n, d in qb_depth.items() if d == starter_rank and not _is_out(n, out_l)),
                    None,
                )
            if qb1_display:
                last = qb1_display.split()[-1].lower()
                hist = player_rolls.filter(
                    (pl.col("team") == team)
                    & (pl.col("season") >= season - 1)
                    & pl.col("player_name").str.to_lowercase().str.contains(last, literal=True)
                    & pl.col("roll_pass").is_not_null()
                ).sort(["season", "week"], descending=True).head(1)
                one_game = None if not hist.is_empty() else _one_game_pass_roll(player_stats, team, qb1_display, season)
                if not hist.is_empty():
                    passers = hist
                elif one_game is not None:
                    passers = one_game
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
        rush_room = rushers
        rushers = rush_room.sort("roll_rush", descending=True).head(top_n_rush)

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
        recv_room = receivers
        if not share_ready:
            receivers = receivers.sort("roll_recv", descending=True).head(top_n_recv)

        if passers.height and "roll_pass" in passers.columns:
            passers = passers.filter(pl.col("roll_pass").is_not_null())
        pass_sum = float(passers["roll_pass"].sum()) or 1.0 if passers.height else 1.0
        blended_pass: float | None = None
        for r in passers.iter_rows(named=True):
            share = float(r["roll_pass"]) / pass_sum
            mu = team_pass_pred * share
            if passers.height == 1:
                mu = team_pass_pred * 0.92
            roll = float(r["roll_pass"] or 0)
            allowed = (opp_pass_allowed or {}).get(opp)
            if allowed is not None and cuts.get("pass"):
                allowed = max(80.0, float(allowed) - float(cuts["pass"]))
            # What this defense has actually allowed has to move the number.
            # A 300-yard quarterback against a defense allowing 170 is not a
            # 277-yard game, and a 170-yard quarterback against a defense
            # allowing 250 does not drop into the 150s because a receiver is
            # out. Those targets get thrown to someone else.
            if roll and roll == roll:
                pace = mu / 0.92 if passers.height == 1 else mu
                mu = blend_pass_yards(roll, float(allowed) if allowed is not None else None, pace)
            else:
                roll = mu
            blended_pass = mu
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

        recent = _recent_team_yards(player_stats, team, season, week)
        roster = _roster_names(depth_charts, team)
        # Receiving yards are the quarterback's yards. Every catcher is a share
        # of that one number, including the tight end and the back. A cap on
        # the top receiver was leaving a chunk of the passing game with nobody.
        pass_pool = blended_pass if blended_pass is not None else team_pass_pred
        if recent:
            recv_weights = {
                name: slot["recv"]
                for name, slot in recent.items()
                if slot["recv"] > 0 and name.lower() not in out_l
            }
            recv_weights = _on_roster(recv_weights, roster)
        else:
            recv_weights = {
                str(row["player_name"]): float(row["roll_recv"])
                for row in recv_room.iter_rows(named=True)
                if row.get("roll_recv")
            }
        recv_yards = _allocate(recv_weights, pass_pool)

        # Rush yards are the team rush total. An out back's carries stay in
        # that total and move to the backs who are playing. A returning starter
        # splits the fill-in's recent work in half instead of both being the lead.
        if recent:
            positions = {name: slot["pos"] for name, slot in recent.items()}
            full_rush = _on_roster(
                {name: slot["rush"] for name, slot in recent.items() if slot["rush"] > 0},
                roster,
            )
            prior_total = sum(full_rush.values())
            out_rb = sum(
                yards
                for name, yards in full_rush.items()
                if name.lower() in out_l and positions.get(name, "").upper().startswith("RB")
            )
            # Giving the whole pile to the lead back runs about 10 yards high
            # (113 games, 2023-2026, an out back had at least 20% of the carries).
            # 70% of that share plus 30% of his own average was the unbiased mix.
            # The yards that come off him stay in the team total, with the rest
            # of the backfield. This was not fit to any one game.
            absorbed = prior_total > 0 and out_rb / prior_total >= 0.20
            rush_weights = {
                name: yards for name, yards in full_rush.items() if name.lower() not in out_l
            }
            starter = _returning_rb1(depth_charts, injuries, player_stats, team, season, week, out_l)
            if starter and rush_weights:
                others = {
                    name: yards
                    for name, yards in rush_weights.items()
                    if name.lower() != starter.lower() and positions.get(name, "").upper().startswith("RB")
                }
                if others:
                    fill_in = max(others, key=others.get)
                    half = rush_weights[fill_in] / 2.0
                    rush_weights[fill_in] = half
                    rush_weights[starter] = half
                    absorbed = False
            rush_yards = _allocate(rush_weights, team_rush_pred)
            if absorbed and rush_yards:
                backs = [
                    name
                    for name in rush_yards
                    if positions.get(name, "").upper().startswith("RB")
                ]
                rolls = {
                    str(row["player_name"]): float(row["roll_rush"])
                    for row in rush_room.iter_rows(named=True)
                    if row.get("roll_rush")
                }
                if backs:
                    lead = max(backs, key=lambda name: rush_yards[name])
                    own = rolls.get(lead)
                    if own:
                        rush_yards[lead] = min(
                            team_rush_pred, 0.70 * rush_yards[lead] + 0.30 * own
                        )
        else:
            qb_rolls = {
                str(row["player_name"]): float(row["roll_rush"])
                for row in rush_room.iter_rows(named=True)
                if str(row.get("position") or "").upper().startswith("QB") and row.get("roll_rush")
            }
            back_rolls = {
                str(row["player_name"]): float(row["roll_rush"])
                for row in rush_room.iter_rows(named=True)
                if not str(row.get("position") or "").upper().startswith("QB") and row.get("roll_rush")
            }
            # No game log to share from. The quarterback keeps his own average
            # and the backs split whatever rushing is left.
            rush_yards = dict(qb_rolls)
            rush_yards.update(_allocate(back_rolls, max(0.0, team_rush_pred - sum(qb_rolls.values()))))

        rush_sum = float(rushers["roll_rush"].sum()) or 1.0
        for r in rushers.iter_rows(named=True):
            share = float(r["roll_rush"]) / rush_sum
            team_share_mu = team_rush_pred * min(share, 0.55)
            roll = float(r["roll_rush"])
            is_qb = str(r.get("position") or "").upper().startswith("QB")
            # A quarterback's short rushing scores are not a yardage bet, but
            # his rushing yards are still part of the team total when we know
            # the backfield's shares.
            if r["player_name"] in rush_yards:
                mu = rush_yards[r["player_name"]]
            else:
                mu = roll if is_qb else 0.55 * roll + 0.45 * team_share_mu
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

        passing = _split_sentence(pass_pool, recv_yards, "passing yards", "the rest of the room")
        rushing = _split_sentence(team_rush_pred, rush_yards, "rushing yards", "the rest of the backfield")
        allocation = " ".join(part for part in (passing, rushing) if part)
        if allocation:
            for prop in reversed(props):
                if prop.get("team") == team and prop.get("market") == "player_pass_yds":
                    prop["allocation"] = allocation
                    break

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
    live_outs: set[str] | None = None,
) -> pl.DataFrame:
    from nfl_edge.features.injuries import normalize_depth_charts

    rolls = rolling_player_yards(player_stats)
    dc = normalize_depth_charts(depth_charts) if depth_charts is not None and not depth_charts.is_empty() else None
    allowed_cache: dict[tuple[int, int], dict[str, float]] = {}
    all_props: list[dict] = []
    for r in predictions.iter_rows(named=True):
        key = (int(r["season"]), int(r["week"]))
        if key not in allowed_cache:
            allowed_cache[key] = pass_yards_allowed_by_defense(player_stats, key[0], key[1])
        out = _ruled_out(injuries, key[0], key[1]) | {str(n).lower() for n in (live_outs or set()) if n}
        from nfl_edge.models.matchup_parlays import starter_yard_cuts

        cuts = {
            team: starter_yard_cuts(player_stats, depth_charts, team, key[0], key[1])
            for team in (r.get("home_team"), r.get("away_team"))
            if team
        }
        all_props.extend(
            project_props_for_game(
                game_row=r,
                player_rolls=rolls,
                depth_charts=dc,
                out_players=out,
                opp_pass_allowed=allowed_cache[key],
                player_stats=player_stats,
                injuries=injuries,
                defense_cuts=cuts,
            )
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
