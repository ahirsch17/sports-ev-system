"""Same-game parlays from one passing total, for every game.

A leg is kept only when it clears both the defense's recent yardage and the
blended total. New starters who have not played a snap for this defense yet
trim that total. They do not rewrite it.
"""

from __future__ import annotations

import math
import re
from datetime import datetime

import polars as pl

from nfl_edge.models.props import blend_pass_yards, defense_shape_rooms

DL = {"LDE", "RDE", "LDT", "RDT", "NT", "DE", "DT"}
DB = {"LCB", "RCB", "NB", "FS", "SS", "CB", "S"}
_CUSHION = 8.0


def _half_floor(value: float) -> float:
    return math.floor(value * 2) / 2


def _half_ceil(value: float) -> float:
    return math.ceil(value * 2) / 2


def _stated_line(line: float, side: str) -> float:
    """The half-point a book prints.

    Yards land on whole numbers. Over 95 is the book's over 95.5. Under 240 is
    the book's under 239.5. A line that is already on the half stays there.
    """
    line = float(line)
    half = int(round(line * 2))
    if abs(line * 2 - half) > 1e-6 or half % 2 == 1:
        return line
    return line + 0.5 if side == "over" else line - 0.5


_BET_UNIT = {
    "pass": "passing yards",
    "rec": "receiving yards",
    "rush": "rushing yards",
    "receptions": "receptions",
    "carries": "carries",
}


def _bet_label(player: str, side: str, line: float, market: str) -> str:
    word = "over" if side == "over" else "under"
    return f"{player} {word} {line:.1f} {_BET_UNIT[market]}."


_NAMES = {
    "ARI": "Arizona", "ATL": "Atlanta", "BAL": "Baltimore", "BUF": "Buffalo",
    "CAR": "Carolina", "CHI": "Chicago", "CIN": "Cincinnati", "CLE": "Cleveland",
    "DAL": "Dallas", "DEN": "Denver", "DET": "Detroit", "GB": "Green Bay",
    "HOU": "Houston", "IND": "Indianapolis", "JAX": "Jacksonville", "KC": "Kansas City",
    "LA": "Los Angeles", "LAC": "the Chargers", "LV": "Las Vegas", "MIA": "Miami",
    "MIN": "Minnesota", "NE": "New England", "NO": "New Orleans", "NYG": "the Giants",
    "NYJ": "the Jets", "PHI": "Philadelphia", "PIT": "Pittsburgh", "SEA": "Seattle",
    "SF": "San Francisco", "TB": "Tampa Bay", "TEN": "Tennessee", "WAS": "Washington",
}


def _name(team: str) -> str:
    return _NAMES.get(team, team)


def _side_phrase(margin: float, home: str, away: str) -> str:
    if abs(margin) < 0.5:
        return "a pick"
    who = _name(home) if margin > 0 else _name(away)
    return f"{who} by {abs(margin):.0f}"


def _points(value: float) -> str:
    value = float(value)
    if abs(value * 2 - round(value * 2)) < 0.05 and abs(value - round(value)) >= 0.2:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _posted(game: dict, line_key: str, model_key: str):
    """The number on the board. The blended forecast is only a fallback."""
    posted = game.get(line_key)
    return posted if posted is not None else game.get(model_key)


def _cleared_majority(values: list[float], line: float) -> bool:
    """An over has to land in more than half the games that produced the average."""
    if not values:
        return False
    hits = sum(float(value) > float(line) for value in values)
    return hits * 2 > len(values)


def _rest_sentence(note: str | None, home: str, away: str) -> str | None:
    if not note or note == "normal rest":
        return None
    found = re.search(r"rest (\d+)/(\d+)", note)
    if found:
        return (
            f"{_name(home)} is on {int(found.group(1))} days of rest and "
            f"{_name(away)} is on {int(found.group(2))}. That gap is only a few tenths of a point."
        )
    return f"Rest is {note}, which is already in the margin."


def _injury_sentence(note: str | None) -> str | None:
    if not note or ": " not in str(note):
        return None
    team, rest = str(note).split(": ", 1)
    name = _name(team)
    if rest == "healthy-ish":
        return f"{name} is healthy enough that injuries do not move the score."
    points = re.search(r"([+-]?\d+(?:\.\d+)?)", rest)
    amount = f"{abs(float(points.group(1))):.1f} points" if points else "the score"
    if "skill/OL" in rest:
        return f"{name} is missing skill players or linemen, and the injury model docks them {amount}."
    if "injury drag" in rest:
        return f"{name} has an injury drag of {amount} already in the score."
    return f"{name}: {rest}."


def _absence_sentence(injuries: pl.DataFrame | None, team: str, season: int, week: int) -> str | None:
    if injuries is None or injuries.is_empty() or "full_name" not in injuries.columns:
        return None
    rows = injuries.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("team") == team)
        & pl.col("report_status").is_in(["Out", "Doubtful"])
        & pl.col("position").is_in(["QB", "RB", "FB", "WR", "TE", "T", "G", "C", "OT", "OG", "OL"])
    )
    if rows.is_empty():
        return None
    bits = [f"{row['full_name']} ({row['position']})" for row in rows.iter_rows(named=True)]
    return f"{_name(team)} is without {', '.join(bits)}."


def write_script(
    game: dict,
    tickets: list[dict],
    defender_notes: list[str],
    injuries: pl.DataFrame | None = None,
    play: str | None = None,
) -> str:
    """Plain description of the model score, then what the parlays are using."""
    home = _name(game["home_team"])
    away = _name(game["away_team"])
    bits = [f"{away} at {home}."]
    if play:
        bits.append(play)
    ours = game.get("model_margin")
    book = _posted(game, "spread_line", "pred_margin")
    our_total = game.get("model_total")
    book_total = _posted(game, "total_line", "pred_total")
    if ours is not None:
        score = ""
        if our_total is not None:
            home_pts = (float(our_total) + float(ours)) / 2.0
            away_pts = (float(our_total) - float(ours)) / 2.0
            score = f", {home_pts:.0f}–{away_pts:.0f}"
        bits.append(f"Our number is {_side_phrase(float(ours), game['home_team'], game['away_team'])}{score}.")
    if our_total is not None:
        bits.append(f"Our total is {float(our_total):.0f}.")
    if book is not None:
        book_line = f"The book is {_side_phrase(float(book), game['home_team'], game['away_team'])}"
        if book_total is not None:
            book_line += f" with a total of {_points(book_total)}"
        total_is_lower = (
            our_total is not None
            and book_total is not None
            and float(our_total) <= float(book_total) - 3
        )
        if total_is_lower:
            bits.append(book_line + ". The side stays off. The total is lower, so the under is a leg.")
        else:
            bits.append(book_line + ". The side and the total stay off the card.")
    roof = str(game.get("roof") or "")
    wind, temp = game.get("wind"), game.get("temp")
    if roof.lower() in {"outdoors", "open", "outdoor"} and wind in (None, "") and temp in (None, ""):
        bits.append("It is outdoors and the feed has no wind or temperature, so weather is not in the number.")
    elif wind not in (None, "") and float(wind) >= 15:
        bits.append(f"Wind is {float(wind):.0f} mph, and that is already in the total.")
    rest = _rest_sentence(game.get("rest_note"), game["home_team"], game["away_team"])
    if rest:
        bits.append(rest)
    deep = game.get("deep_net_margin_adj")
    if deep is not None:
        moved = abs(float(deep))
        if moved < 1:
            bits.append("The player matchups move the margin by less than a point.")
        else:
            bits.append(f"The player matchups move the margin by {float(deep):+.1f}.")
    for key in ("away_inj_note", "home_inj_note"):
        sentence = _injury_sentence(game.get(key))
        if sentence:
            bits.append(sentence)
    if injuries is not None and game.get("season") is not None and game.get("week") is not None:
        for team in (game["away_team"], game["home_team"]):
            missing = _absence_sentence(injuries, team, int(game["season"]), int(game["week"]))
            if missing:
                bits.append(missing)
    bits.extend(defender_notes)
    if tickets and {ticket.get("team") for ticket in tickets} <= {"Props", "Scores"}:
        bits.append("The parlays are the props that cleared, then the scorers.")
    elif tickets and tickets[0].get("team") == "FanDuel":
        legs = tickets[0]["legs"]
        scorers = [leg["player"] for leg in legs if leg.get("market") == "anytime_td"]
        others = [leg["player"] for leg in legs if leg.get("market") != "anytime_td"]
        sentence = "On FanDuel, take " + ", ".join(others) + "." if others else ""
        if scorers:
            extra = "anytime touchdowns from " + ", ".join(scorers)
            sentence = (sentence[:-1] + ", plus " + extra + ".") if sentence else extra[0].upper() + extra[1:] + "."
        if sentence:
            bits.append(sentence)
    elif tickets:
        parts = []
        for ticket in tickets:
            who = _name(ticket["team"])
            names = list(dict.fromkeys(leg["player"] for leg in ticket["legs"]))
            parts.append(f"{who}: {', '.join(names)}")
        bits.append("The parlays that clear both numbers are " + "; ".join(parts) + ".")
    else:
        bits.append("Nothing cleared both the defense's recent yards and the blend, so there is no parlay.")
    return " ".join(bits)


def _leg_decimal(price: int) -> float:
    if price < 0:
        return 1 + 100 / abs(price)
    return 1 + price / 100


def _ticket_odds(legs: list[dict]) -> int:
    decimal = 1.0
    for leg in legs:
        decimal *= _leg_decimal(int(leg.get("price", -110)))
    if decimal < 2:
        return int(round(-100 / (decimal - 1)))
    return int(round((decimal - 1) * 100))


def starter_yard_cuts(
    player_stats: pl.DataFrame,
    depth_charts: pl.DataFrame | None,
    team: str,
    season: int,
    week: int,
) -> dict:
    """Trim allowed yards for starters who have not played for this defense yet.

    A new tackle changes pressure and the run, not the whole secondary.
    The cut is capped so one player cannot replace seven games of team totals.
    """
    empty = {"pass": 0.0, "rush": 0.0, "wr": 0.0, "names": []}
    if (
        depth_charts is None
        or depth_charts.is_empty()
        or "dt" not in depth_charts.columns
        or player_stats.is_empty()
    ):
        return empty
    latest = depth_charts.filter(pl.col("dt") == depth_charts["dt"].max())
    starters = latest.filter(
        (pl.col("team") == team)
        & (pl.col("pos_rank") == 1)
        & pl.col("pos_abb").is_in(list(DL | DB))
    )
    if starters.is_empty() or "player_display_name" not in player_stats.columns:
        return empty

    prior = player_stats.filter(
        (pl.col("season") == season)
        & (pl.col("week") < week)
        & (pl.col("season_type") == "REG")
    )
    pass_cut = rush_cut = wr_cut = 0.0
    names: list[str] = []
    for row in starters.iter_rows(named=True):
        name = str(row.get("player_name") or "")
        if not name:
            continue
        his = prior.filter(pl.col("player_display_name") == name)
        if his.is_empty():
            continue
        on_team = his.filter(pl.col("team") == team)
        if on_team.height >= 2:
            continue
        played = his.filter(pl.col("team") != team)
        if played.is_empty():
            continue
        games = played["week"].n_unique()
        if games < 3:
            continue
        sacks = float(played["def_sacks"].fill_null(0).sum()) if "def_sacks" in played.columns else 0.0
        tfl = float(played["def_tackles_for_loss"].fill_null(0).sum()) if "def_tackles_for_loss" in played.columns else 0.0
        pos = str(row.get("pos_abb") or "")
        old = played["team"].unique().to_list()
        old_team = old[0] if len(old) == 1 else "another team"
        if pos in DL:
            pass_cut += min(8.0, max(0.0, sacks / games - 0.25) * 10)
            rush_cut += min(10.0, max(0.0, tfl / games - 0.40) * 8)
            names.append(f"{name} ({old_team}, {pos})")
        elif pos in DB:
            wr_cut += min(6.0, max(0.0, sacks / games) * 4)
            names.append(f"{name} ({old_team}, {pos})")
    return {
        "pass": min(12.0, pass_cut),
        "rush": min(14.0, rush_cut),
        "wr": min(8.0, wr_cut),
        "names": names,
    }


def _prior_allowed(stats: pl.DataFrame, defense: str, season: int, week: int) -> dict[str, float] | None:
    prior = stats.filter(
        (pl.col("opponent_team") == defense)
        & (pl.col("season") == season)
        & (pl.col("week") < week)
        & (pl.col("season_type") == "REG")
    )
    if prior.is_empty():
        return None
    weeks = []
    for w in prior["week"].unique().sort().to_list():
        g = prior.filter(pl.col("week") == w)
        qb = g.filter((pl.col("position") == "QB") & (pl.col("attempts") >= 8))
        weeks.append(
            {
                "pass": float(qb["passing_yards"].sum()) if qb.height else 0.0,
                "wr": float(g.filter(pl.col("position") == "WR")["receiving_yards"].sum()),
                "te": float(g.filter(pl.col("position") == "TE")["receiving_yards"].sum()),
                "rb": float(g.filter(pl.col("position").is_in(["RB", "FB"]))["receiving_yards"].sum()),
                "rush": float(g.filter(pl.col("position").is_in(["RB", "FB"]))["rushing_yards"].sum()),
            }
        )
    if len(weeks) < 3:
        return None
    keys = ("pass", "wr", "te", "rb", "rush")
    return {k: sum(row[k] for row in weeks) / len(weeks) for k in keys}


def _checked_and_left_off(
    stats: pl.DataFrame,
    team: str,
    season: int,
    week: int,
    ticket_names: set[str],
    injuries: pl.DataFrame | None = None,
) -> str | None:
    """Skill players with a real target share who did not clear a leg.

    The same rule every week: anyone with 15 targets or 80 receiving yards
    before this game is checked. His last four games decide it, not an early
    spike. Four names at most, the ones with the most targets first.
    """
    if stats.is_empty() or "receiving_yards" not in stats.columns:
        return None
    group = stats.filter(
        (pl.col("team") == team)
        & (pl.col("season") == season)
        & (pl.col("week") < week)
        & (pl.col("season_type") == "REG")
        & pl.col("position").is_in(["WR", "TE", "RB", "FB"])
    )
    if group.is_empty():
        return None
    out = _out_names(injuries, team, season, week)
    ranked = []
    for name in group["player_display_name"].unique().to_list():
        if name in ticket_names or str(name).lower() in out:
            continue
        his = group.filter(pl.col("player_display_name") == name).sort("week")
        if his.height < 3:
            continue
        targets = int(his["targets"].sum()) if "targets" in his.columns else 0
        yards = float(his["receiving_yards"].sum())
        if targets < 15 and yards < 80:
            continue
        last = [int(y) for y in his.tail(4)["receiving_yards"].to_list()]
        ranked.append((targets, yards, name, last))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    if not ranked:
        return None
    bits = [f"{name} ({', '.join(str(y) for y in last)})" for _, _, name, last in ranked[:4]]
    return (
        f"Also checked {', '.join(bits)}. "
        "The last four games are in the parentheses. None of them cleared a leg."
    )


def _corners_already_counted(
    stats: pl.DataFrame,
    depth_charts: pl.DataFrame | None,
    defense: str,
    offense: str,
    season: int,
    week: int,
) -> str | None:
    """Corners of the defense those receivers actually face.

    Their snaps are inside the yards that defense allowed. Naming them is the
    consideration. Cutting the receiver again would count the same corner twice.
    """
    if (
        depth_charts is None
        or depth_charts.is_empty()
        or "dt" not in depth_charts.columns
        or stats.is_empty()
    ):
        return None
    latest = depth_charts.filter(pl.col("dt") == depth_charts["dt"].max())
    starters = latest.filter(
        (pl.col("team") == defense)
        & (pl.col("pos_rank") == 1)
        & pl.col("pos_abb").is_in(["LCB", "RCB", "CB"])
    )
    if starters.is_empty():
        return None
    played = []
    for row in starters.iter_rows(named=True):
        name = str(row.get("player_name") or "")
        if not name:
            continue
        his = stats.filter(
            (pl.col("player_display_name") == name)
            & (pl.col("team") == defense)
            & (pl.col("season") == season)
            & (pl.col("week") < week)
            & (pl.col("season_type") == "REG")
        )
        games = his["week"].n_unique() if not his.is_empty() else 0
        if games >= 3:
            played.append((games, name))
    if not played:
        return None
    played.sort(reverse=True)
    bits = [f"{name} ({games} games)" for games, name in played]
    who = _name(defense)
    room = _name(offense)
    if len(bits) == 1:
        return (
            f"{who}'s starting corner is already in {room}'s receiver room: {bits[0]}. "
            "He already played in the games that built that room, so he is not a second cut."
        )
    return (
        f"{who}'s starting corners are already in {room}'s receiver room: {', '.join(bits)}. "
        "They already played in the games that built that room, so they are not a second cut."
    )


def _bars(*pairs: tuple[str, float]) -> list[dict]:
    return [{"name": name, "value": round(float(value), 1)} for name, value in pairs]


def _scoreboard(game: dict) -> dict | None:
    ours, our_total = game.get("model_margin"), game.get("model_total")
    if ours is None or our_total is None:
        return None
    book, book_total = _posted(game, "spread_line", "pred_margin"), _posted(game, "total_line", "pred_total")

    def split(margin: float, total: float) -> tuple[float, float]:
        return round((total - margin) / 2.0, 1), round((total + margin) / 2.0, 1)

    away_pts, home_pts = split(float(ours), float(our_total))
    out = {
        "away": game.get("away_team"),
        "home": game.get("home_team"),
        "our_away": away_pts,
        "our_home": home_pts,
        "our_total": round(float(our_total), 1),
    }
    if book is not None and book_total is not None:
        book_away, book_home = split(float(book), float(book_total))
        out["book_away"] = book_away
        out["book_home"] = book_home
        out["book_total"] = round(float(book_total), 1)
    return out


def _typical_yards(values: list[float]) -> tuple[float, float | None]:
    """The average, with one game set aside when it stands apart from the rest.

    A single long play shows up here only as that game's total. There is no
    play log. The count legs do not use this. They already require every game.
    """
    vals = [float(v) for v in values]
    if not vals:
        return 0.0, None
    if len(vals) < 4:
        return sum(vals) / len(vals), None
    spike = max(vals)
    rest = list(vals)
    rest.remove(spike)
    ordered = sorted(rest)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    gap = spike - median
    stands_apart = gap >= 40 or (median > 0 and spike >= median * 1.5 and gap >= 25)
    if stands_apart:
        return sum(rest) / len(rest), spike
    return sum(vals) / len(vals), None


def _per_game(value: float, games: int, unit: str, spike: float | None = None) -> str:
    """His average per game, over the same games the leg is using."""
    sentence = f"He averages {value:.0f} {unit} a game over his last {games}."
    if spike is None:
        return sentence
    return sentence[:-1] + f", with the {spike:.0f}-yard game set aside."


def _td_per_game(frame: pl.DataFrame) -> str:
    cols = [c for c in ("passing_tds", "rushing_tds", "receiving_tds") if c in frame.columns]
    if not cols or frame.is_empty():
        return ""
    total = sum(float(frame[c].fill_null(0).sum()) for c in cols)
    avg = total / frame.height
    shown = f"{avg:.1f}".rstrip("0").rstrip(".")
    word = "touchdown" if shown == "1" else "touchdowns"
    return f"He averages {shown} {word} a game over his last {frame.height}."


def _out_names(injuries: pl.DataFrame | None, team: str, season: int, week: int) -> set[str]:
    if injuries is None or injuries.is_empty() or "full_name" not in injuries.columns:
        return set()
    rows = injuries.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("team") == team)
        & pl.col("report_status").is_in(["Out", "Doubtful"])
    )
    return {str(n).lower() for n in rows["full_name"].to_list() if n}


def _posted_count_line(values: list[float], slack: float) -> float | None:
    """A half-point just under the average, if more than half the recent games cleared it.

    Slack is how far under the average a book still hangs the number: 1.5 for
    catches and carries. A line at the minimum, several units under the average,
    is not on the board.
    """
    if len(values) < 4:
        return None
    mean = sum(values) / len(values)
    line = int((mean - slack) * 2) / 2
    if line <= 0:
        return None
    clears = sum(1 for value in values if float(value) > line)
    if clears * 2 <= len(values):
        return None
    return float(line)


def _count_locks(legs: list[dict], offense: pl.DataFrame) -> list[dict]:
    """Add a carries or receptions leg when that count is a real price.

    The line sits just under the recent average, where a book actually hangs it,
    and more than half the recent games have to clear it. Pass attempts are not
    a posted market.
    """
    have = {(leg["player"], leg["market"]) for leg in legs}
    added: list[dict] = []

    def games(player: str, n: int) -> pl.DataFrame:
        return offense.filter(pl.col("player_display_name") == player).sort("week").tail(n)

    for leg in legs:
        player = leg["player"]
        if leg["market"] == "rec" and (player, "receptions") not in have and "receptions" in offense.columns:
            recent = games(player, 6)
            catches = [float(value) for value in recent["receptions"].to_list()]
            line = _posted_count_line(catches, 1.5)
            if line is not None:
                line = _stated_line(line, "over")
                added.append(
                    {
                        "player": player,
                        "market": "receptions",
                        "side": "over",
                        "line": line,
                        "price": -150,
                        "bars": _bars(("Avg", float(recent["receptions"].mean())), ("Line", line)),
                        "label": _bet_label(player, "over", line, "receptions"),
                        "detail": (
                            f"{_per_game(float(recent['receptions'].mean()), recent.height, 'catches')} "
                            f"More than half of those games cleared {line:.1f}. Take -150 or longer."
                        ),
                    }
                )
    return added


def _receptions_when_yards_move(legs: list[dict], offense: pl.DataFrame) -> list[dict]:
    """One catch leg when the yards swing and the catches do not.

    A receiver whose last six games average at least 5 catches still gets to 3
    the next week about 4 times in 5, even when his yardage is noisy. The yards
    version of that same spot hits about half the time. Only the steadiest
    catch total is used, and only if his yards did not already clear a leg.
    """
    if offense.is_empty() or "receptions" not in offense.columns:
        return []
    already = {leg["player"] for leg in legs if leg["market"] in {"receptions", "rec"}}
    best: tuple[float, str, pl.DataFrame] | None = None
    skill = offense.filter(pl.col("position").is_in(["WR", "TE"]))
    for name in skill["player_display_name"].unique().to_list():
        if name in already:
            continue
        recent = skill.filter(pl.col("player_display_name") == name).sort("week").tail(6)
        if recent.height < 4:
            continue
        mean = float(recent["receptions"].mean())
        if mean < 5:
            continue
        if best is None or mean > best[0]:
            best = (mean, name, recent)
    if best is None:
        return []
    mean, name, recent = best
    catches = [float(value) for value in recent["receptions"].to_list()]
    line = _posted_count_line(catches, 1.5)
    if line is None:
        return []
    line = _stated_line(line, "over")
    targets = ""
    if "targets" in recent.columns:
        targets = f" on {float(recent['targets'].mean()):.0f} targets"
    return [
        {
            "player": name,
            "market": "receptions",
            "side": "over",
            "line": line,
            "price": -150,
            "bars": _bars(
                ("Avg", mean),
                *((("Targets", float(recent["targets"].mean())),) if "targets" in recent.columns else ()),
                ("Line", line),
            ),
            "label": _bet_label(name, "over", line, "receptions"),
            "detail": " ".join(
                part
                for part in (
                    f"He averages {mean:.0f} catches a game over his last {recent.height}{targets}.",
                    _td_per_game(recent),
                    f"More than half of those games cleared {line:.1f}. The yards move around more than the catches. Take -150 or longer.",
                )
                if part
            ),
        }
    ]


def _returning_rb1(
    stats: pl.DataFrame,
    depth: pl.DataFrame | None,
    injuries: pl.DataFrame | None,
    team: str,
    season: int,
    week: int,
) -> str | None:
    """Depth-chart RB1 who was out recently, is active now, and has barely played.

    The carries in that window belong to the absence. On the 24 returns from
    2023 through 2025, the fill-in kept about half the yards and half the
    carries, and he was still at 8 carries in 11 of the 24. The same gates
    run on that half. A number that still clears stays.
    """
    if (
        depth is None
        or depth.is_empty()
        or "dt" not in depth.columns
        or injuries is None
        or injuries.is_empty()
        or "full_name" not in injuries.columns
    ):
        return None
    latest = depth.filter(pl.col("dt") == depth["dt"].max())
    rbs = latest.filter((pl.col("team") == team) & (pl.col("pos_abb") == "RB") & (pl.col("pos_rank") == 1))
    if rbs.is_empty():
        return None
    name = str(rbs["player_name"][0] or "")
    if not name or name.lower() in _out_names(injuries, team, season, week):
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
    played = stats.filter(
        (pl.col("season") == season)
        & (pl.col("week") < week)
        & (pl.col("season_type") == "REG")
        & (pl.col("team") == team)
        & (pl.col("player_display_name").str.to_lowercase() == name.lower())
        & (pl.col("carries") > 0)
    )
    if played.height >= 2:
        return None
    return name


def _team_ticket(
    stats: pl.DataFrame,
    *,
    team: str,
    opponent: str,
    season: int,
    week: int,
    pace: float,
    cuts: dict,
    injuries: pl.DataFrame | None,
    depth: pl.DataFrame | None = None,
) -> dict | None:
    allowed = _prior_allowed(stats, opponent, season, week)
    if not allowed or allowed["pass"] <= 0:
        return None
    out = _out_names(injuries, team, season, week)
    raw_pass = max(80.0, allowed["pass"] - cuts["pass"])
    raw_rush = max(40.0, allowed["rush"] - cuts["rush"])
    scale = raw_pass / allowed["pass"]
    raw_wr = max(0.0, allowed["wr"] * scale - cuts["wr"])
    raw_te = max(0.0, allowed["te"] * scale)
    raw_rb = max(0.0, allowed["rb"] * scale)

    offense = stats.filter(
        (pl.col("team") == team)
        & (pl.col("season") == season)
        & (pl.col("week") < week)
        & (pl.col("season_type") == "REG")
    )
    if offense.is_empty():
        return None

    qbs = offense.filter((pl.col("position") == "QB") & (pl.col("attempts") >= 12))
    legs: list[dict] = []
    count_anchors: list[dict] = []
    blend = None
    roll = None
    if not qbs.is_empty():
        starter_name = qbs.sort("week").tail(1)["player_display_name"][0]
        if str(starter_name).lower() not in out:
            starts = qbs.filter(pl.col("player_display_name") == starter_name).sort("week").tail(6)
            if starts.height >= 2:
                roll, pass_spike = _typical_yards([float(y) for y in starts["passing_yards"].to_list()])
                blend = blend_pass_yards(roll, raw_pass, pace)
                anchor = max(blend, raw_pass)
                if roll - anchor >= 20:
                    # The middle of his average and the projection is the starting
                    # floor. A normal QB game also has one completion around 35
                    # yards, so the floor has to clear the projection plus that
                    # play. Since 2023 that higher floor cashed about 3 in 5.
                    line = _stated_line(_half_ceil(max((roll + anchor) / 2, anchor + 35)), "under")
                    legs.append(
                        {
                            "player": starter_name,
                            "market": "pass",
                            "side": "under",
                            "line": line,
                            "bars": _bars(("Avg", roll), ("Ours", blend), ("Defense", raw_pass), ("Line", line)),
                            "label": _bet_label(starter_name, "under", line, "pass"),
                            "detail": " ".join(
                                part
                                for part in (
                                    _per_game(roll, starts.height, "passing yards", pass_spike),
                                    _td_per_game(starts),
                                    f"Model {blend:.0f}. The defense has allowed {raw_pass:.0f}. "
                                    "The under clears that number plus one long gain. Take -110 or longer.",
                                )
                                if part
                            ),
                        }
                    )

    rooms_low = {"WR": raw_wr, "TE": raw_te, "RB": raw_rb}
    shape = defense_shape_rooms(blend if blend else raw_pass, raw_wr, raw_te, raw_rb)
    skill_all = offense.filter(pl.col("position").is_in(["WR", "TE", "RB", "FB"]))
    # An absent teammate stays in the past share. Deleting him hands his old
    # yards to whoever is left, and those catches were never theirs.
    skill = (
        skill_all.filter(~pl.col("player_display_name").str.to_lowercase().is_in(list(out)))
        if out
        else skill_all
    )
    for pos, key in (("WR", "WR"), ("TE", "TE"), ("RB", "RB")):
        pos_filter = pl.col("position") == pos if pos != "RB" else pl.col("position").is_in(["RB", "FB"])
        history = skill_all.filter(pos_filter)
        group = skill.filter(pos_filter)
        if group.is_empty() or history.is_empty():
            continue
        def _room(frame: pl.DataFrame) -> pl.DataFrame:
            return frame.group_by("player_display_name").agg(
                pl.col("receiving_yards").sum().alias("yds"), pl.len().alias("n")
            ).filter(pl.col("n") >= 3)

        totals = _room(group)
        room_sum = float(_room(history)["yds"].sum()) or 1.0
        low_room = rooms_low[key]
        high_room = shape[key]
        for player in totals.iter_rows(named=True):
            share = float(player["yds"]) / room_sum
            if share < 0.45:
                continue
            own = group.filter(pl.col("player_display_name") == player["player_display_name"]).sort("week").tail(6)
            own_mean, own_spike = _typical_yards([float(y) for y in own["receiving_yards"].to_list()])
            low = share * low_room
            high = share * high_room
            # The room can be large while this player is not. His own games count.
            floor = min(own_mean, low, high)
            if floor < 45:
                continue
            line = _stated_line(_half_floor(floor), "over")
            if line < 40:
                continue
            count_anchors.append({"player": player["player_display_name"], "market": "rec"})
            if not _cleared_majority([float(y) for y in own["receiving_yards"].to_list()], line):
                continue
            legs.append(
                {
                    "player": player["player_display_name"],
                    "market": "rec",
                    "side": "over",
                    "line": line,
                    "bars": _bars(("Avg", own_mean), ("Low end", floor), ("Line", line)),
                    "label": _bet_label(player["player_display_name"], "over", line, "rec"),
                    "detail": " ".join(
                        part
                        for part in (
                            _per_game(own_mean, own.height, "receiving yards", own_spike),
                            _td_per_game(own),
                            f"The low end is {floor:.0f}. Take -110 or longer.",
                        )
                        if part
                    ),
                }
            )

    role_note = None
    backs = skill.filter(pl.col("position").is_in(["RB", "FB"]))
    if not backs.is_empty():
        recent_weeks = sorted(backs["week"].unique().to_list())[-4:]
        recent = backs.filter(pl.col("week").is_in(recent_weeks))
        carries = recent.group_by("player_display_name").agg(
            pl.col("carries").mean().alias("car"),
            pl.col("rushing_yards").mean().alias("rush"),
            pl.len().alias("n"),
        )
        carries = carries.filter(pl.col("car") >= 8).sort("car", descending=True)
        if not carries.is_empty():
            lead = carries.row(0, named=True)
            car_sum = float(carries["car"].sum()) or 1.0
            share = min(0.85, float(lead["car"]) / car_sum)
            his = recent.filter(pl.col("player_display_name") == lead["player_display_name"])
            his_yards = [float(y) for y in his["rushing_yards"].to_list()]
            raw_mean = sum(his_yards) / len(his_yards) if his_yards else 0.0
            typical_rush, rush_spike = _typical_yards(his_yards)
            room = share * raw_rush
            fair = min(typical_rush, room)
            starter_back = _returning_rb1(stats, depth, injuries, team, season, week)
            fill_in = bool(starter_back) and str(lead["player_display_name"]).lower() != starter_back.lower()
            # One big yardage game does not erase the carries. The carries
            # already had to clear in every recent game.
            if fill_in:
                # Half is the middle of those 24 returns, not a deletion.
                before_yards = typical_rush
                before_carries = float(his["carries"].min())
                his_yards = [yards * 0.5 for yards in his_yards]
                typical_rush *= 0.5
                if rush_spike is not None:
                    rush_spike *= 0.5
                share *= 0.5
                room = share * raw_rush
                fair = min(typical_rush, room)
                raw_mean = sum(his_yards) / len(his_yards) if his_yards else 0.0
                role_note = (
                    f"{lead['player_display_name']} averaged {before_yards:.0f} rushing yards and at least "
                    f"{before_carries:.0f} carries while {starter_back} was out. {starter_back} is active. "
                    f"Fill-ins kept about half on the returns since 2023, so the check uses {typical_rush:.0f} yards "
                    f"and a low game of {before_carries * 0.5:.0f} carries."
                )
            if min(raw_mean, room) >= 50 or fair >= 50:
                if not fill_in or float(his["carries"].min()) * 0.5 >= 8:
                    count_anchors.append({"player": lead["player_display_name"], "market": "rush"})
            if fair >= 50:
                line = _stated_line(_half_floor(fair - _CUSHION), "over")
                if line >= 35 and _cleared_majority(his_yards, line):
                    legs.append(
                        {
                            "player": lead["player_display_name"],
                            "market": "rush",
                            "side": "over",
                            "line": line,
                            "bars": _bars(("Avg", typical_rush), ("Room", raw_rush), ("Line", line)),
                            "label": _bet_label(lead["player_display_name"], "over", line, "rush"),
                            "detail": " ".join(
                                part
                                for part in (
                                    _per_game(typical_rush, int(lead["n"]), "rushing yards", rush_spike),
                                    _td_per_game(recent.filter(pl.col("player_display_name") == lead["player_display_name"])),
                                    f"Rush room {raw_rush:.0f}. Take -140 or longer.",
                                )
                                if part
                            ),
                        }
                    )

    # One market per player. A count leg can sit next to his yards when the count has cleared every recent game.
    seen: set[tuple[str, str]] = set()
    unique = []
    for leg in legs:
        key = (leg["player"], leg["market"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(leg)
    already = {(leg["player"], leg["market"]) for leg in unique}
    unique.extend(_count_locks(unique + [a for a in count_anchors if (a["player"], a["market"]) not in already], offense))
    unique.extend(_receptions_when_yards_move(unique, offense))
    if not unique:
        if role_note:
            return {"team": team, "title": team, "odds": "", "legs": [], "why": "", "role_note": role_note}
        return None
    unique = unique[:5]
    catches = any(leg["market"] == "receptions" and leg["line"] == 2.5 for leg in unique)
    counted = any(leg["market"] in {"carries", "receptions", "attempts"} for leg in unique)
    if catches:
        tail = "Catches are the leg when the yards move around more than the targets."
    elif counted:
        tail = "The count legs sit just under his average."
    else:
        tail = "Every leg clears both numbers."
    return {
        "team": team,
        "title": team,
        "odds": _ticket_odds(unique),
        "legs": unique,
        "why": (
            f"One passing total"
            + (f" of {blend:.0f}" if blend else "")
            + f". The defense has allowed {raw_pass:.0f} through the air and {raw_rush:.0f} on the ground. "
            + tail
        ),
        "role_note": role_note,
    }


def _pack_ticket(name: str, legs: list[dict], why: str) -> dict:
    return {
        "team": name,
        "title": name,
        "odds": _ticket_odds(legs),
        "legs": legs,
        "why": why,
    }


def _paired_combo(yards: list[dict], counts: list[dict]) -> dict | None:
    """A yards leg with the count that was added because of it.

    Swift's carries and Burden's catches have no yards leg, so they stay on
    the counts ticket. The combo is the pairs.
    """
    mate = {"rec": "receptions", "rush": "carries"}
    by_player: dict[str, dict[str, dict]] = {}
    for leg in counts:
        by_player.setdefault(leg["player"], {})[leg["market"]] = leg
    paired: list[dict] = []
    for leg in yards:
        other = by_player.get(leg["player"], {}).get(mate.get(leg["market"], ""))
        if other is None:
            continue
        paired.append(dict(leg))
        paired.append(dict(other))
        if len(paired) >= 4:
            break
    if len(paired) < 2:
        return None
    return _pack_ticket(
        "Combo",
        paired,
        "Each yards leg is paired with that player's count. The count was added because of the yards leg.",
    )


def _mix_tickets(team_tickets: list[dict]) -> list[dict]:
    """Legs can come from either team. Same-team is not the grouping.

    Yardage that spends different piles goes on one ticket. Counts that cleared
    every recent game go on another. The third pairs a yards leg with the count
    that belongs to that same player.
    """
    pool: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for ticket in team_tickets:
        for leg in ticket["legs"]:
            key = (leg["player"], leg["market"])
            if key in seen:
                continue
            seen.add(key)
            pool.append(leg)
    yards = [leg for leg in pool if leg["market"] in {"pass", "rec", "rush"}][:5]
    counts = [leg for leg in pool if leg["market"] in {"attempts", "carries", "receptions"}][:5]
    if len(yards) >= 2 and len(counts) >= 2:
        tickets = [
            _pack_ticket(
                "Yards",
                yards,
                "These yards are different piles, so they can land in the same game.",
            ),
            _pack_ticket(
                "Counts",
                counts,
                "Each of these counts sits just under the average.",
            ),
        ]
        combo = _paired_combo(yards, counts)
        if combo:
            tickets.append(combo)
        return tickets
    if len(yards) >= 2:
        legs = (yards + counts)[:5]
        return [_pack_ticket("Yards", legs, "These yards are different piles, so they can land in the same game.")]
    if len(counts) >= 2 and yards:
        return [_pack_ticket("Counts", (counts + yards)[:5], "Each of these counts sits just under the average.")]
    if len(counts) >= 2:
        return [_pack_ticket("Counts", counts, "Each of these counts sits just under the average.")]
    return team_tickets


def _fanduel_ticket(props: list[dict]) -> dict | None:
    """The FanDuel side that has actually cashed.

    A passing number 20 or more yards off his average went under that average
    about 2 in 3, on a few hundred games. Receiving and rushing gaps did not
    have a sample big enough to parlay, so they stay off the ticket.
    """
    legs = []
    for prop in props or []:
        if str(prop.get("market") or "") != "player_pass_yds":
            continue
        try:
            mu = float(prop.get("mu"))
            roll = float(prop.get("roll"))
        except (TypeError, ValueError):
            continue
        if mu != mu or roll != roll or roll < 140 or abs(mu - roll) < 20:
            continue
        side = "under" if mu < roll else "over"
        player = str(prop.get("player") or "")
        if not player:
            continue
        legs.append(
            {
                "player": player,
                "market": "player_pass_yds",
                "side": side,
                "line": None,
                "price": -110,
                "bars": _bars(("Ours", mu), ("Avg", roll)),
                "_gap": abs(mu - roll),
                "label": f"{player} passing yards {side}.",
                "detail": (
                    f"We have {mu:.0f}. He averages {roll:.0f}. "
                    f"Take FanDuel's {side} while their number is still up near that average. "
                    "That side cashed about 2 in 3."
                ),
            }
        )
    legs.sort(key=lambda leg: leg["_gap"], reverse=True)
    for leg in legs:
        leg.pop("_gap", None)
    if not legs:
        return None
    return _pack_ticket(
        "FanDuel",
        legs[:1],
        "This is the passing side. It cashed about 2 in 3 when the number was this far under his average.",
    )


def _anytime_legs(
    stats: pl.DataFrame,
    depth: pl.DataFrame | None,
    injuries: pl.DataFrame | None,
    teams: list[str],
    season: int,
    week: int,
) -> list[dict]:
    """Players who have been scoring, priced at the rate that actually repeats.

    Someone who scored in at least 4 of his last 8 still scores the next game
    about 2 in 5. That is plus money, not the hot streak. A fill-in running
    back is left off once the starter is active again.
    """
    needed = {
        "player_display_name",
        "team",
        "position",
        "season",
        "week",
        "rushing_tds",
        "receiving_tds",
        "carries",
        "receptions",
    }
    if stats is None or stats.is_empty() or not needed <= set(stats.columns):
        return []
    prior = stats.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    )
    if "season_type" in prior.columns:
        prior = prior.filter(pl.col("season_type") == "REG")
    if prior.is_empty():
        return []
    # The last team he played for. An old stint on Chicago does not put him in this game.
    current = (
        prior.sort(["player_display_name", "season", "week"])
        .group_by("player_display_name", maintain_order=True)
        .agg([pl.col("team").last(), pl.col("position").last()])
    )
    eligible = current.filter(
        pl.col("team").is_in(teams) & pl.col("position").is_in(["RB", "WR", "TE", "FB"])
    )
    if eligible.is_empty():
        return []
    frame = prior.filter(pl.col("player_display_name").is_in(eligible["player_display_name"].to_list()))
    recent = (
        frame.sort(["player_display_name", "season", "week"])
        .group_by("player_display_name", maintain_order=True)
        .tail(8)
    )
    summary = recent.group_by("player_display_name").agg(
        [
            pl.col("team").last(),
            pl.col("position").last(),
            pl.len().alias("games"),
            pl.col("season").last().alias("last_season"),
            ((pl.col("rushing_tds").fill_null(0) + pl.col("receiving_tds").fill_null(0)) > 0)
            .sum()
            .alias("scored"),
            (pl.col("carries").fill_null(0) + pl.col("receptions").fill_null(0)).mean().alias("touches"),
        ]
    )
    returning = {}
    for team in teams:
        name = _returning_rb1(stats, depth, injuries, team, season, week)
        if name:
            returning[team] = name.lower()
    legs = []
    for row in summary.iter_rows(named=True):
        games = int(row["games"])
        scored = int(row["scored"])
        if games < 4 or scored < 2 or scored / games < 0.40 or float(row["touches"] or 0) < 4:
            continue
        team = str(row["team"])
        player = str(row["player_display_name"])
        if str(row["position"]) == "RB" and team in returning and player.lower() != returning[team]:
            continue
        short = games < 6
        detail = f"He scored in {scored} of his last {games}."
        if short:
            detail += " That is a short run."
        if int(row["last_season"]) < season:
            detail += f" His last game was in {int(row['last_season'])}."
        detail += " Players scoring this often hit the next game about 2 in 5, so the price has to be plus money."
        legs.append(
            {
                "player": player,
                "team": team,
                "position": str(row["position"]),
                "market": "anytime_td",
                "side": "over",
                "line": None,
                "price": 130,
                "bars": [],
                "scored": scored,
                "games": games,
                "label": f"{player} anytime touchdown.",
                "_rate": scored / games,
                "detail": detail,
            }
        )
    legs.sort(key=lambda leg: (leg["_rate"], leg["scored"]), reverse=True)
    for leg in legs:
        leg.pop("_rate", None)
    return legs


def _play_line(game: dict) -> str | None:
    """The passing projections, stated before any parlay is built."""
    passes = []
    for prop in game.get("player_props") or []:
        if str(prop.get("market") or "") != "player_pass_yds":
            continue
        edge = _prop_edge(prop)
        if edge is None or not prop.get("player"):
            continue
        passes.append((edge, prop))
    if not passes:
        return None
    passes.sort()
    parts = [
        f"{prop['player']} is {float(prop['mu']):.0f} against his average of {float(prop['roll']):.0f}"
        for _, prop in passes
    ]
    line = "The passing numbers are " + " and ".join(parts) + "."
    notes = []
    for _, prop in passes:
        note = prop.get("allocation")
        if note and note not in notes:
            notes.append(note)
    if notes:
        line = f"{line} {' '.join(notes)}"
    return line


def _prop_edge(prop: dict) -> float | None:
    try:
        mu = float(prop.get("mu"))
        roll = float(prop.get("roll"))
    except (TypeError, ValueError):
        return None
    if mu != mu or roll != roll:
        return None
    return mu - roll


_PROP_GATES = {
    "player_pass_yds": ("passing yards", 20.0, 140.0),
    "player_reception_yds": ("receiving yards", 15.0, 35.0),
    "player_rush_yds": ("rushing yards", 12.0, 40.0),
}


def _yard_leg(prop: dict) -> dict | None:
    market = str(prop.get("market") or "")
    spec = _PROP_GATES.get(market)
    edge = _prop_edge(prop)
    if spec is None or edge is None or abs(edge) < spec[1]:
        return None
    unit, _gap, floor = spec
    roll = float(prop["roll"])
    if roll < floor:
        return None
    player = str(prop.get("player") or "")
    if not player:
        return None
    mu = float(prop["mu"])
    # A back the model has at a handful of yards is not a posted under. The
    # carries already moved. 5 yards is not a line.
    if market == "player_rush_yds" and mu < 20:
        return None
    side = "under" if edge < 0 else "over"
    return {
        "player": player,
        "team": prop.get("team"),
        "market": market,
        "side": side,
        "line": None,
        "price": -110,
        "bars": _bars(("Ours", mu), ("Avg", roll)),
        "_gap": abs(edge),
        "label": f"{player} {unit} {side}.",
        "detail": f"We have {mu:.0f}. He averages {roll:.0f}.",
    }


# FanDuel's measured tax on a simple two-prop same-game pair (Pikkit, NFL
# quarterback yards with a running back). They price each -110 leg nearer -125,
# which pays +224 instead of the +264 two separate bets pay. It is not a live
# line. It is the floor: shorter than that, the tax ate the legs.
_SGP_FLOOR = 224


def _family(leg: dict) -> str:
    market = str(leg.get("market") or "")
    if market in {"player_pass_yds", "player_reception_yds", "pass", "rec"}:
        return "pass"
    if market in {"player_rush_yds", "rush"}:
        return "rush"
    return market


def _same_outcome(a: dict, b: dict) -> bool:
    """True when the two legs are one prediction the book already prices as one."""
    if a.get("player") and a.get("player") == b.get("player"):
        return True
    same_team = a.get("team") and a.get("team") == b.get("team")
    if same_team and _family(a) == _family(b):
        return True
    families = {_family(a), _family(b)}
    if "total" in families and "pass" in families:
        return True
    return False


def _best_scorers(scorers: list[dict]) -> list[dict]:
    """Two scorers. The second is the other team when that team has one."""
    if len(scorers) < 2:
        return []
    first = scorers[0]
    second = next((leg for leg in scorers[1:] if leg.get("team") != first.get("team")), scorers[1])
    return [first, second]


def _total_script(game: dict) -> str | None:
    """Over or under when our total is at least 3 points off the book. Otherwise no script."""
    our_total = game.get("model_total")
    book_total = _posted(game, "total_line", "pred_total")
    if our_total is None or book_total is None:
        return None
    if abs(float(our_total) - float(book_total)) < 3:
        return None
    return "under" if float(our_total) < float(book_total) else "over"


def _stack_rank(left: dict, right: dict, script: str | None) -> int | None:
    """Higher means the two legs belong in one game. None means they fight.

    A pass under with that team's back over is one offense. A back and the
    total only go together on the same side. Two unders in a game we have
    going over do not.
    """
    if _same_outcome(left, right):
        return None
    families = {_family(left), _family(right)}
    sides = {left.get("side"), right.get("side")}
    same_team = bool(left.get("team") and left.get("team") == right.get("team"))
    if "total" in families:
        other = right if _family(left) == "total" else left
        total = left if _family(left) == "total" else right
        if other.get("side") != total.get("side"):
            return None
        return 1
    if same_team and families == {"pass", "rush"} and sides == {"over", "under"}:
        return 3
    if len(sides) == 1:
        side = next(iter(sides))
        if script is not None and side != script:
            return None
        return 2
    return None


def _collapsed(leg: dict) -> bool:
    """A projection under half of the player's own average is not a bettable line."""
    if leg.get("market") == "total":
        return False
    ours = next((bar.get("value") for bar in leg.get("bars") or [] if bar.get("name") == "Ours"), None)
    avg = next((bar.get("value") for bar in leg.get("bars") or [] if bar.get("name") == "Avg"), None)
    if ours is None or not avg:
        return False
    return float(ours) < 0.45 * float(avg)


def _best_pair(legs: list[dict], script: str | None = None) -> list[dict]:
    """The pair that shares a script. A fight is not posted."""
    best: tuple[dict, dict] | None = None
    best_score = -1.0
    for i, left in enumerate(legs):
        for right in legs[i + 1 :]:
            if _collapsed(left) or _collapsed(right):
                continue
            rank = _stack_rank(left, right, script)
            if rank is None:
                continue
            score = rank * 1000 + float(left.get("_gap") or 0) + float(right.get("_gap") or 0)
            if score > best_score:
                best_score = score
                best = (left, right)
    if best is None:
        return []
    pair = list(best)
    pair.sort(key=lambda leg: float(leg.get("_gap") or 0), reverse=True)
    return pair


def _pair_blurb(legs: list[dict]) -> str:
    families = {_family(leg) for leg in legs}
    sides = {leg.get("side") for leg in legs}
    same_team = len({leg.get("team") for leg in legs if leg.get("team")}) == 1
    if "total" in families:
        return "The run game and the total go the same way."
    if same_team and sides == {"over", "under"} and families == {"pass", "rush"}:
        rush = next(leg for leg in legs if _family(leg) == "rush")
        throw = next(leg for leg in legs if _family(leg) == "pass")
        if rush.get("side") == "over":
            return f"{rush['player']} gets the carries. {throw['player']} is under his average."
        return f"{throw['player']} is over his average. {rush['player']} is under his."
    if len(sides) == 1:
        return "Both are under their own averages." if "under" in sides else "Both are over their own averages."
    return "The two legs go the same way."


def _pair_why(legs: list[dict]) -> str:
    return (
        f"{_pair_blurb(legs)} "
        f"Two separate bets at -110 pay +{_ticket_odds(legs)}. "
        f"FanDuel has been pricing a pair like this nearer +{_SGP_FLOOR}, "
        "so take the parlay only when it is longer than that."
    )


def _aligned_tickets(
    game: dict,
    props: list[dict],
    scorers: list[dict],
    returning: dict | None = None,
) -> list[dict]:
    """One pair from predictions that already cleared. The numbers are not moved to fit it.

    Once the starting back is active again, the fill-in is not a posted rushing
    prop. His yards stay in the team split. They are not a leg.
    """
    starters = {team: str(name).lower() for team, name in (returning or {}).items() if name}
    tickets = []
    props_legs = []
    for prop in props or []:
        starter = starters.get(prop.get("team"))
        if (
            starter
            and str(prop.get("market") or "") == "player_rush_yds"
            and str(prop.get("player") or "").lower() != starter
        ):
            continue
        leg = _yard_leg(prop)
        if leg:
            props_legs.append(leg)
    props_legs.sort(key=lambda leg: leg["_gap"], reverse=True)
    our_total = game.get("model_total")
    book_total = _posted(game, "total_line", "pred_total")
    total_leg = None
    if (
        our_total is not None
        and book_total is not None
        and abs(float(our_total) - float(book_total)) >= 3
    ):
        side = "under" if float(our_total) < float(book_total) else "over"
        total_leg = {
            "player": "Full game",
            "market": "total",
            "side": side,
            "line": None,
            "price": -110,
            "bars": _bars(("Ours", float(our_total)), ("Book", float(book_total))),
            "_gap": abs(float(our_total) - float(book_total)),
            "label": f"Full game total {side}.",
            "detail": f"The model total is {float(our_total):.0f}. The book has {float(book_total):.0f}.",
        }
        if props_legs:
            props_legs.append(total_leg)
    pair = _best_pair(props_legs, _total_script(game))
    for leg in props_legs:
        leg.pop("_gap", None)
    if len(pair) == 2:
        ticket = _pack_ticket("Props", pair, _pair_why(pair))
        ticket["blurb"] = _pair_blurb(pair)
        ticket["price_label"] = "two bets"
        tickets.append(ticket)
    paired_total = any(leg.get("market") == "total" for leg in pair)
    if total_leg is not None and not paired_total:
        total_leg.pop("_gap", None)
        side = str(total_leg["side"])
        tickets.append(
            _pack_ticket(
                "Total",
                [total_leg],
                f"On its own. We have {float(our_total):.0f}. The book has {float(book_total):.0f}.",
            )
        )
        tickets[-1]["blurb"] = f"Game {side}, on its own."
    scorers = _best_scorers(scorers)
    if len(scorers) == 2:
        tickets.append(
            _pack_ticket(
                "Scores",
                scorers,
                "Two touchdowns, from different teams when both have one. Separate from the yard props.",
            )
        )
    return tickets


def build_matchup_parlays(
    games: list[dict],
    player_stats: pl.DataFrame,
    depth_charts: pl.DataFrame | None = None,
    injuries: pl.DataFrame | None = None,
) -> list[dict]:
    """Same-game tickets. Legs from either team, grouped by the yards they spend."""
    if player_stats.is_empty():
        return []
    cards = []
    for game in games:
        if game.get("actual_home_score") is not None or game.get("home_score") is not None:
            continue
        if game.get("season") is None or game.get("week") is None:
            continue
        season = int(game["season"])
        week = int(game["week"])
        home = game["home_team"]
        away = game["away_team"]
        home_cuts = starter_yard_cuts(player_stats, depth_charts, home, season, week)
        away_cuts = starter_yard_cuts(player_stats, depth_charts, away, season, week)
        tickets = []
        for team, opp, pace_key, cuts in (
            (away, home, "pred_away_pass_yds", home_cuts),
            (home, away, "pred_home_pass_yds", away_cuts),
        ):
            pace = float(game.get(pace_key) or 230)
            ticket = _team_ticket(
                player_stats,
                team=team,
                opponent=opp,
                season=season,
                week=week,
                pace=pace,
                cuts=cuts,
                injuries=injuries,
                depth=depth_charts,
            )
            if ticket:
                tickets.append(ticket)
        notes = []
        kept = []
        for ticket in tickets:
            note = ticket.pop("role_note", None)
            if note and game.get("player_props"):
                # The rushing split above is the number. Don't cite the older check.
                reason = note.split(" Fill-ins kept")[0].strip()
                notes.append(reason + " That work is split in the rushing total above.")
            elif note:
                notes.append(note)
            if ticket["legs"]:
                kept.append(ticket)
        tickets = _mix_tickets(kept)
        if game.get("player_props"):
            starters = {
                team: name
                for team in (home, away)
                if (name := _returning_rb1(player_stats, depth_charts, injuries, team, season, week))
            }
            tickets = _aligned_tickets(
                game,
                game.get("player_props") or [],
                _anytime_legs(player_stats, depth_charts, injuries, [home, away], season, week),
                starters,
            )
        for side, cuts in ((home, home_cuts), (away, away_cuts)):
            if cuts["names"]:
                notes.append(
                    f"{_name(side)} added {', '.join(cuts['names'])}. "
                    f"Against them, the passing total is {cuts['pass']:.0f} lower and the rushing total is {cuts['rush']:.0f} lower. "
                    "A new tackle is pressure and the run, not a new corner."
                )
        ticket_names = {leg["player"] for ticket in tickets for leg in ticket["legs"]}
        for offense, defense in ((away, home), (home, away)):
            checked = _checked_and_left_off(player_stats, offense, season, week, ticket_names, injuries)
            if checked:
                notes.append(checked)
            corners = _corners_already_counted(player_stats, depth_charts, defense, offense, season, week)
            if corners:
                notes.append(corners)
        when = ""
        gameday = game.get("gameday")
        if gameday:
            try:
                when = datetime.strptime(str(gameday)[:10], "%Y-%m-%d").strftime("%A")
            except ValueError:
                when = str(gameday)
        if game.get("gametime"):
            when = f"{when} {game['gametime']} ET".strip()
        cards.append(
            {
                "game_id": game.get("game_id"),
                "matchup": f"{away} at {home}",
                "when": when,
                "script": write_script(
                    game,
                    tickets,
                    notes,
                    injuries,
                    _play_line(game),
                ),
                "score": _scoreboard(game),
                "note": "",
                "tickets": tickets,
            }
        )
    return cards
