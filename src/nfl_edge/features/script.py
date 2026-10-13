"""Game script: this offense against this defense, then injuries that change the plan.

Pass yards are the offense's recent production times how many pass yards that
defense has allowed, not a blend of the offense with itself. When starting
receivers are out, the vacated throws become runs. They are not handed to the
players who are still on the field.
"""

from __future__ import annotations

import polars as pl

LEAGUE_PASS = 215.0
LEAGUE_RUSH = 120.0
LEAGUE_TE = 48.0


def defense_index(allowed: float | None, league: float, *, lo: float = 0.78, hi: float = 1.22) -> float:
    """How this defense compares with a normal defense. 1.0 means league average."""
    try:
        value = float(allowed)  # type: ignore[arg-type]
        base = float(league)
    except (TypeError, ValueError):
        return 1.0
    if value != value or value <= 0 or base != base or base <= 0:
        return 1.0
    return max(lo, min(hi, value / base))


def catcher_pass_cut(catchers_out: float, pass_index: float) -> float:
    """Share of the pass game that becomes runs when catchers are missing.

    A defense that already takes the pass away makes that shift larger. The
    offense knows the throws are not there.
    """
    units = max(float(catchers_out or 0.0), 0.0)
    cut = min(0.21, 0.07 * units)
    if pass_index < 0.90 and units >= 1:
        cut = min(0.28, cut + 0.07)
    return cut


def scripted_pass_yards(
    off_yards: float | None,
    def_allowed: float | None,
    *,
    catchers_out: float = 0.0,
    league: float = LEAGUE_PASS,
) -> tuple[float, float, float]:
    """Return (yards after the run shift, fraction moved to the run, defense index)."""
    try:
        base = float(off_yards)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        base = league
    if base != base or base <= 0:
        base = league
    index = defense_index(def_allowed, league)
    before = base * index
    cut = catcher_pass_cut(catchers_out, index)
    return before * (1.0 - cut), cut, index


def scripted_rush_yards(
    off_yards: float | None,
    def_allowed: float | None,
    *,
    pass_before_cut: float,
    cut: float,
    league: float = LEAGUE_RUSH,
) -> float:
    """Rush yards after the opponent's run defense, plus carries taken from the pass game."""
    try:
        base = float(off_yards)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        base = league
    if base != base or base <= 0:
        base = league
    yards = base * defense_index(def_allowed, league)
    moved_attempts = max(float(pass_before_cut), 0.0) / 7.1 * max(float(cut), 0.0)
    return yards + moved_attempts * 4.4


def te_volume_multiplier(
    te_allowed: float | None,
    pass_index: float,
    *,
    league_te: float = LEAGUE_TE,
) -> float:
    """Extra tight-end cut when this defense allows fewer TE yards than its overall pass defense."""
    if te_allowed is None:
        return 1.0
    te_index = defense_index(te_allowed, league_te, lo=0.55, hi=1.40)
    # An average tight-end defense is not an extra fade, even if the overall
    # pass defense is leaky. Only a defense that actually takes tight ends away
    # changes their volume, and only by the gap the team pass total missed.
    if te_index >= 0.90:
        return 1.0
    denom = pass_index if pass_index and pass_index > 0 else 1.0
    if denom < 0.95:
        return max(0.60, min(1.0, te_index / denom))
    return max(0.60, min(1.0, te_index))


def script_note(team: str, catchers_n: int, pass_index: float, te_mult: float) -> str:
    """One sentence on why the plan changed. Empty when nothing in the matchup moves it."""
    bits: list[str] = []
    n = int(catchers_n or 0)
    label = "pass catcher" if n == 1 else "pass catchers"
    if n >= 1 and pass_index < 0.90:
        bits.append(
            f"{n} {label} out and this defense allows {pass_index:.0%} of a normal pass game, so the plan leans run"
        )
    elif n >= 1:
        bits.append(f"{n} {label} out, so some throws become runs instead of extra targets")
    elif pass_index < 0.90:
        bits.append(f"this defense allows {pass_index:.0%} of a normal pass game, so the pass volume comes down")
    if te_mult <= 0.85:
        bits.append("tight ends see even less, because this defense takes them away")
    if not bits:
        return ""
    return f"{team}: " + ". ".join(bits)


def build_te_allowed_lookup(
    player_stats: pl.DataFrame | None,
    schedules: pl.DataFrame | None,
) -> tuple[dict[tuple[str, int, int], float], float]:
    """TE yards each defense allowed per game before that week, plus the league TE rate."""
    if (
        player_stats is None
        or schedules is None
        or player_stats.is_empty()
        or schedules.is_empty()
    ):
        return {}, LEAGUE_TE
    pcols = set(player_stats.columns)
    team_col = "recent_team" if "recent_team" in pcols else ("team" if "team" in pcols else None)
    if (
        team_col is None
        or "position" not in pcols
        or "receiving_yards" not in pcols
        or "season" not in pcols
        or "week" not in pcols
        or "home_team" not in schedules.columns
        or "away_team" not in schedules.columns
    ):
        return {}, LEAGUE_TE

    gained = (
        player_stats.filter(pl.col("position").cast(pl.Utf8).str.to_uppercase() == "TE")
        .group_by(["season", "week", team_col])
        .agg(pl.col("receiving_yards").fill_null(0).sum().alias("te_yards"))
        .rename({team_col: "offense"})
    )
    if gained.is_empty():
        return {}, LEAGUE_TE
    league_te = float(gained["te_yards"].mean() or LEAGUE_TE)
    if league_te <= 0:
        league_te = LEAGUE_TE

    games = pl.concat(
        [
            schedules.select(
                "season",
                "week",
                pl.col("home_team").alias("defense"),
                pl.col("away_team").alias("offense"),
            ),
            schedules.select(
                "season",
                "week",
                pl.col("away_team").alias("defense"),
                pl.col("home_team").alias("offense"),
            ),
        ]
    )
    allowed = (
        games.join(gained, on=["season", "week", "offense"], how="left")
        .with_columns(pl.col("te_yards").fill_null(0.0))
        .group_by(["defense", "season", "week"])
        .agg(pl.col("te_yards").sum())
        .sort(["defense", "season", "week"])
        .with_columns(
            pl.col("te_yards").shift(1).rolling_mean(window_size=6, min_samples=2).over("defense").alias("roll")
        )
    )
    lookup: dict[tuple[str, int, int], float] = {}
    for row in allowed.iter_rows(named=True):
        roll = row.get("roll")
        if roll is None:
            continue
        lookup[(str(row["defense"]), int(row["season"]), int(row["week"]))] = float(roll)
    return lookup, league_te
