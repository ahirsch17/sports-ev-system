"""Injury and depth-chart adjustments."""

from __future__ import annotations

import polars as pl

from nfl_edge.features.weather import load_weights

# Severity scores for report statuses
# Return and kicking spots are not the depth that sets an offensive penalty.
_SPECIAL = {"P", "K", "H", "LS", "PR", "KR", "KO", "PT", "PK"}

STATUS_SEVERITY = {
    "Out": 1.0,
    "Doubtful": 0.75,
    "Questionable": 0.35,
    "IR": 1.0,
    "PUP": 1.0,
    "Suspended": 1.0,
    "COVID-19": 0.9,
}


def _norm_status(s: str | None) -> float:
    if not s:
        return 0.0
    s = str(s).strip()
    for key, val in STATUS_SEVERITY.items():
        if key.lower() in s.lower():
            return val
    return 0.0


def _practice_severity(practice_status: str | None, *, is_qb: bool = False) -> float:
    """
    Official report_status often lags. Practice DNP for a QB is actionable signal
    (e.g. Jaxson Dart knee DNP before IR/Out posted).
    """
    if not practice_status:
        return 0.0
    s = str(practice_status).lower()
    if "did not participate" in s or "dnp" in s:
        return 0.9 if is_qb else 0.55
    if "limited" in s:
        return 0.4 if is_qb else 0.25
    return 0.0


def _row_severity(row: dict, *, is_qb: bool = False) -> float:
    report = _norm_status(row.get("report_status"))
    practice = _practice_severity(row.get("practice_status"), is_qb=is_qb)
    injury_txt = " ".join(
        str(row.get(k) or "")
        for k in ("report_primary_injury", "practice_primary_injury", "report_secondary_injury")
    ).lower()
    if any(x in injury_txt for x in ("acl", "achilles", "fracture")) and is_qb:
        practice = max(practice, 0.95)
    return max(report, practice)


def latest_injuries_by_week(injuries: pl.DataFrame) -> pl.DataFrame:
    """Keep the latest report per player-season-week."""
    if injuries.is_empty():
        return injuries
    keep = [
        c
        for c in [
            "season",
            "week",
            "team",
            "gsis_id",
            "full_name",
            "position",
            "report_status",
            "practice_status",
            "report_primary_injury",
            "practice_primary_injury",
            "report_secondary_injury",
        ]
        if c in injuries.columns
    ]
    if not keep:
        return injuries.head(0)
    df = injuries.select(keep)
    keys = [c for c in ["season", "week", "gsis_id"] if c in df.columns]
    return df.unique(subset=keys, keep="last")


def normalize_depth_charts(depth_charts: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize depth charts across nflverse schema versions.

    Current schema uses: dt, team, gsis_id, player_name, pos_abb, pos_rank, pos_slot
    Older schema used: season, week, team, position, depth_team
    """
    if depth_charts.is_empty():
        return depth_charts

    df = depth_charts
    cols = set(df.columns)

    # Modern snapshot format (no season/week) — take latest dt per team-player-pos
    if "dt" in cols and "season" not in cols:
        # newest snapshot per team, so one partial scrape can't drop other teams
        df = df.sort("dt").filter(pl.col("dt") == pl.col("dt").max().over("team"))
        rename = {}
        if "pos_abb" in cols:
            rename["pos_abb"] = "position"
        if "player_name" in cols and "full_name" not in cols:
            rename["player_name"] = "full_name"
        if "pos_rank" in cols:
            rename["pos_rank"] = "depth_team"
        if rename:
            df = df.rename(rename)
        df = df.with_columns(pl.lit(None, dtype=pl.Int32).alias("season"), pl.lit(None, dtype=pl.Int32).alias("week"))
        if "depth_team" in df.columns:
            df = df.with_columns(pl.col("depth_team").cast(pl.Int32, strict=False).fill_null(99))
        keep = [c for c in ["season", "week", "team", "gsis_id", "full_name", "position", "depth_team"] if c in df.columns]
        return df.select(keep).unique(subset=["team", "gsis_id", "position"], keep="last")

    if "club_code" in cols and "team" not in cols:
        df = df.rename({"club_code": "team"})
        cols = set(df.columns)

    depth_col = "depth_team" if "depth_team" in cols else None
    pos_col = "position" if "position" in cols else ("pos_abb" if "pos_abb" in cols else None)
    if depth_col and depth_col in df.columns:
        df = df.with_columns(
            pl.col(depth_col).cast(pl.Int32, strict=False).fill_null(99).alias(depth_col)
        )
        df = df.filter(pl.col(depth_col) <= 2)
    keep = [
        c
        for c in ["season", "week", "team", "gsis_id", "full_name", "player_name", pos_col or "position", depth_col or "depth_team"]
        if c and c in df.columns
    ]
    out = df.select(keep)
    if "player_name" in out.columns and "full_name" not in out.columns:
        out = out.rename({"player_name": "full_name"})
    if pos_col and pos_col != "position" and pos_col in out.columns:
        out = out.rename({pos_col: "position"})
    if depth_col and depth_col != "depth_team" and depth_col in out.columns:
        out = out.rename({depth_col: "depth_team"})
    if "depth_team" in out.columns:
        out = out.with_columns(pl.col("depth_team").cast(pl.Int32, strict=False).fill_null(99))
    return out.unique()


def recent_starting_qbs(player_stats: pl.DataFrame, *, season: int, week: int) -> dict[str, dict]:
    """
    Map team -> last week's primary passer, plus any QB who started a prior week.

    Depth charts often flip to the backup before report_status says Out.
    Week-2 early exits (Dart) leave the backup with more attempts that week,
    so we also keep every prior weekly max-att passer as a 'known starter'.
    """
    if player_stats is None or player_stats.is_empty():
        return {}
    df = player_stats
    team_col = "recent_team" if "recent_team" in df.columns else ("team" if "team" in df.columns else None)
    if not team_col or "season" not in df.columns or "week" not in df.columns:
        return {}
    att_col = "attempts" if "attempts" in df.columns else (
        "passing_attempts" if "passing_attempts" in df.columns else None
    )
    if att_col is None and "passing_yards" not in df.columns:
        return {}
    name_col = (
        "player_display_name"
        if "player_display_name" in df.columns
        else ("player_name" if "player_name" in df.columns else ("full_name" if "full_name" in df.columns else None))
    )
    id_col = "gsis_id" if "gsis_id" in df.columns else ("player_id" if "player_id" in df.columns else None)
    prior = df.filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )
    if prior.is_empty():
        return {}
    metric = pl.col(att_col) if att_col else pl.col("passing_yards")
    prior = prior.with_columns(metric.fill_null(0).alias("_att"))
    if "position" in prior.columns:
        prior = prior.filter(
            pl.col("position").is_null()
            | pl.col("position").cast(pl.Utf8).str.to_uppercase().str.starts_with("QB")
            | (pl.col("_att") >= 10)
        )
    prior = prior.filter(pl.col("_att") >= 5)
    if prior.is_empty():
        return {}

    # One row per team-week: the passer with the most attempts that week.
    weekly = prior.sort("_att", descending=True).unique(subset=[team_col, "season", "week"], keep="first")
    weekly = weekly.sort([team_col, "season", "week"])

    def _pid(row: dict) -> str:
        return str(row.get(id_col) or "") if id_col else ""

    def _pname(row: dict) -> str:
        return str(row.get(name_col) or row.get("full_name") or "") if name_col else ""

    # Only the current starter is QB1. A backup who started a game in a prior
    # season (Bagent, Mariota, O'Connell) used to stay in known_starter_ids
    # forever, so his DNP was priced as the starter being out. Week number
    # also restarts each season, so max(week) picked last year's week 18
    # instead of this year's last start.
    out: dict[str, dict] = {}
    current: dict | None = None
    current_team: str | None = None

    def _commit(team: str, row: dict) -> None:
        gid = _pid(row)
        nm = _pname(row) or "QB1"
        names: dict[str, str] = {}
        if nm:
            names[nm.lower()] = nm
        if gid and nm:
            names[gid] = nm
        out[team] = {
            "gsis_id": gid,
            "name": nm,
            "attempts": float(row.get("_att") or 0),
            "known_starter_ids": {gid} if gid else set(),
            "known_starter_names": names,
        }

    for r in weekly.iter_rows(named=True):
        team = str(r[team_col])
        if team != current_team:
            if current_team is not None and current is not None:
                _commit(current_team, current)
            current_team = team
            current = None
        att = float(r.get("_att") or 0)
        # A short relief appearance does not take the job. A real start does.
        if current is None or att >= 12:
            current = r
    if current_team is not None and current is not None:
        _commit(current_team, current)
    return out


def team_injury_impact(
    *,
    season: int,
    week: int,
    team: str,
    injuries: pl.DataFrame,
    depth_charts: pl.DataFrame,
    weights: dict | None = None,
    recent_qb: dict | None = None,
) -> dict[str, float | str]:
    """
    Estimate point / pass-yard impacts for a team entering a week.
    Returns keys: points, pass_yards_pct, rush_yards_pct, qb_out, qb_note.
    """
    w = (weights or load_weights())["injuries"]
    impact: dict[str, float | str] = {
        "points": 0.0,
        "pass_yards_pct": 0.0,
        "rush_yards_pct": 0.0,
        "qb_out": 0.0,
        "qb_note": "",
        "catchers_out": 0.0,
        "catchers_n": 0,
    }

    if injuries.is_empty():
        return impact

    week_inj = injuries.filter(
        (pl.col("season") == season) & (pl.col("week") == week) & (pl.col("team") == team)
    )
    if week_inj.is_empty():
        return impact

    starters: set[str] = set()
    starter_pos: dict[str, str] = {}
    starter_rank: dict[str, int] = {}
    depth_by_pos: dict[str, dict[str, int]] = {}
    name_by_id: dict[str, str] = {}

    dc = depth_charts
    if not dc.is_empty():
        if (
            "season" in dc.columns
            and dc["season"].dtype != pl.Null
            and dc["season"].null_count() < dc.height
        ):
            team_dc = dc.filter(
                (pl.col("season") == season) & (pl.col("week") == week) & (pl.col("team") == team)
            )
        else:
            team_dc = dc.filter(pl.col("team") == team)

        for row in team_dc.iter_rows(named=True):
            gid = row.get("gsis_id")
            if not gid:
                continue
            gid = str(gid)
            pos = str(row.get("position") or "").upper()
            rank = int(row.get("depth_team") or 99)
            slot = depth_by_pos.setdefault(gid, {})
            if pos not in slot or rank < slot[pos]:
                slot[pos] = rank
            name_by_id[gid] = str(row.get("full_name") or "")
            if rank <= 2 and pos not in _SPECIAL and rank < starter_rank.get(gid, 99):
                starters.add(gid)
                starter_pos[gid] = pos
                starter_rank[gid] = rank

    recent_gid = str((recent_qb or {}).get("gsis_id") or "")
    recent_name = str((recent_qb or {}).get("name") or "")
    known_ids: set[str] = set((recent_qb or {}).get("known_starter_ids") or set())
    if recent_gid:
        known_ids.add(recent_gid)
    known_names: dict[str, str] = dict((recent_qb or {}).get("known_starter_names") or {})
    has_known_starter = bool(known_ids or known_names)
    qb1_applied = False

    for row in week_inj.iter_rows(named=True):
        gid = str(row.get("gsis_id") or "")
        pname = str(row.get("full_name") or "")
        pos = str(row.get("position") or starter_pos.get(gid) or "").upper()
        is_qb = pos.startswith("QB")
        sev = _row_severity(row, is_qb=is_qb)
        if sev <= 0:
            continue
        slot = depth_by_pos.get(gid) or {}
        if pos in slot:
            depth = slot[pos]
        else:
            real = [r for p, r in slot.items() if p not in _SPECIAL]
            depth = min(real) if real else (1 if gid in starters else 3)
        was_starter = bool(
            (gid and gid in known_ids)
            or (pname and pname.lower() in known_names)
            or (pname and any(pname.lower() == str(v).lower() for v in known_names.values()))
        )
        # The current starter stays QB1 even if the depth chart has already flipped
        # to the backup. A former starter who is not the current one does not.
        if was_starter:
            depth = 1
            is_qb = True

        if depth > 2 and not is_qb:
            impact["points"] = float(impact["points"]) + float(w.get("inactive_depth_gt_2", 0.0)) * sev
            continue

        if is_qb:
            # QB1 only when this is the current starter. With no starter history,
            # fall back to depth-chart QB1. Never stack a second QB1 penalty.
            is_qb1 = was_starter or (not has_known_starter and depth <= 1)
            if is_qb1 and not qb1_applied:
                qb1_applied = True
                impact["points"] = float(impact["points"]) + float(w["qb1_out_points"]) * sev
                impact["pass_yards_pct"] = float(impact["pass_yards_pct"]) - 0.25 * sev
                # Only raise qb_out / note for true starter (don't let backup DNP overwrite)
                if sev >= float(impact["qb_out"]):
                    impact["qb_out"] = sev
                    nm = str(
                        pname
                        or known_names.get(gid)
                        or known_names.get(pname.lower())
                        or recent_name
                        or name_by_id.get(gid)
                        or "QB1"
                    )
                    status = "OUT" if sev >= 0.95 else ("out/DNP" if sev >= 0.75 else "questionable")
                    impact["qb_note"] = f"{nm} {status} (backup starting)"
            else:
                impact["points"] = float(impact["points"]) + float(w["qb2_out_points"]) * sev
                # Track backup status separately for notes
                if sev >= 0.5:
                    nm = str(pname or name_by_id.get(gid) or "QB2")
                    prev = str(impact.get("qb2_note") or "")
                    bit = f"{nm} {'OUT/DNP' if sev >= 0.75 else 'questionable'}"
                    impact["qb2_note"] = f"{prev}; {bit}" if prev else bit
        elif pos.startswith("WR") or pos == "TE" or pos.startswith("TE"):
            wr_scale = 1.0 if pos.startswith("WR") else 0.7
            impact["points"] = float(impact["points"]) + float(w["wr1_out_points"]) * wr_scale * sev * (1.0 if depth == 1 else 0.4)
            impact["pass_yards_pct"] = float(impact["pass_yards_pct"]) - (0.05 if pos.startswith("WR") else 0.04) * sev * (1.0 if depth == 1 else 0.3)
            if sev >= 0.6 and depth <= 2:
                impact["catchers_n"] = int(impact["catchers_n"]) + 1
                impact["catchers_out"] = float(impact["catchers_out"]) + (1.0 if depth <= 1 else 0.7)
        elif pos in {"RB", "HB", "FB"} or pos.startswith("RB"):
            impact["points"] = float(impact["points"]) + float(w["rb1_out_points"]) * sev * (1.0 if depth == 1 else 0.3)
            impact["rush_yards_pct"] = float(impact["rush_yards_pct"]) - 0.08 * sev * (1.0 if depth == 1 else 0.3)
        elif pos in {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"}:
            impact["points"] = float(impact["points"]) + float(w["ol_starter_out_points"]) * sev
        elif pos in {"CB", "DB", "S", "FS", "SS"} or pos.startswith("CB") or pos.endswith("S"):
            impact["points"] = float(impact["points"]) - float(w["cb1_out_vs_pass"]) * sev * (1.0 if depth == 1 else 0.4)

    return impact


def build_injury_features(
    schedules: pl.DataFrame,
    injuries: pl.DataFrame,
    depth_charts: pl.DataFrame,
    weights: dict | None = None,
    player_stats: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-game home/away injury point and yard adjustments."""
    inj = latest_injuries_by_week(injuries) if not injuries.is_empty() else injuries
    dc = normalize_depth_charts(depth_charts) if not depth_charts.is_empty() else depth_charts
    rows = []
    qb_cache: dict[tuple[int, int], dict[str, dict]] = {}
    for r in schedules.iter_rows(named=True):
        season = int(r["season"])
        week = int(r["week"])
        key = (season, week)
        if player_stats is not None and key not in qb_cache:
            qb_cache[key] = recent_starting_qbs(player_stats, season=season, week=week)
        starters = qb_cache.get(key) or {}
        home = team_injury_impact(
            season=season,
            week=week,
            team=r["home_team"],
            injuries=inj,
            depth_charts=dc,
            weights=weights,
            recent_qb=starters.get(str(r["home_team"])),
        )
        away = team_injury_impact(
            season=season,
            week=week,
            team=r["away_team"],
            injuries=inj,
            depth_charts=dc,
            weights=weights,
            recent_qb=starters.get(str(r["away_team"])),
        )
        rows.append(
            {
                "game_id": r["game_id"],
                "home_inj_points": home["points"],
                "away_inj_points": away["points"],
                "home_inj_pass_pct": home["pass_yards_pct"],
                "away_inj_pass_pct": away["pass_yards_pct"],
                "home_inj_rush_pct": home["rush_yards_pct"],
                "away_inj_rush_pct": away["rush_yards_pct"],
                "home_qb_out": home["qb_out"],
                "away_qb_out": away["qb_out"],
                "home_catchers_out": home["catchers_out"],
                "away_catchers_out": away["catchers_out"],
                "home_catchers_n": home["catchers_n"],
                "away_catchers_n": away["catchers_n"],
                "home_inj_note": _inj_note(r["home_team"], home),
                "away_inj_note": _inj_note(r["away_team"], away),
            }
        )
    if not rows:
        return pl.DataFrame({"game_id": []})
    return pl.DataFrame(rows)


def _inj_note(team: str, impact: dict) -> str:
    bits = []
    qb_note = str(impact.get("qb_note") or "")
    qb2_note = str(impact.get("qb2_note") or "")
    if float(impact.get("qb_out") or 0) >= 0.75:
        bits.append(qb_note or "QB1 OUT/Doubtful/DNP")
    elif float(impact.get("qb_out") or 0) >= 0.35:
        bits.append(qb_note or "QB Questionable")
    if qb2_note:
        bits.append(qb2_note)
    catchers_n = int(impact.get("catchers_n") or 0)
    if catchers_n:
        label = "pass catcher" if catchers_n == 1 else "pass catchers"
        bits.append(f"{catchers_n} {label} out")
    pts = float(impact.get("points") or 0)
    if pts <= -2.5:
        bits.append(f"skill/OL hit ({pts:+.1f} pts)")
    elif pts <= -1.0:
        bits.append(f"injury drag ({pts:+.1f})")
    elif pts >= 1.0:
        bits.append(f"opp secondary thinned ({pts:+.1f})")
    return f"{team}: " + (", ".join(bits) if bits else "healthy-ish")
