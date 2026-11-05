"""Figures for one game. Drawn as SVG so they match the page, not a plot library."""

from __future__ import annotations

import polars as pl

INK = "#1c1916"
MUTED = "#5e584e"
TRACK = "#e7e0d4"
OURS = "#1d4e89"
BOOK = "#8a8175"
AWAY = "#1d4e89"
AWAY_SOFT = "#c5d4ea"
HOME = "#8d2f24"
HOME_SOFT = "#e7cfc9"
RULE = "#e4ddd2"
FONT = "Georgia, 'Iowan Old Style', Palatino, serif"


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _pts(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 0.05:
        return str(int(round(number)))
    return f"{number:.1f}"


def _svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" font-family="{FONT}">{body}</svg>'
    )


def _bar(parts: list[str], x: float, y: float, width: float, value: float, peak: float, color: str) -> None:
    length = width * (float(value) / peak) if peak else 0
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="12" rx="2" fill="{TRACK}"/>')
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(length, 1.5):.1f}" height="12" rx="2" fill="{color}"/>'
    )
    parts.append(
        f'<text x="{x + width + 10:.1f}" y="{y + 10.5:.1f}" font-size="13" fill="{INK}">{_pts(value)}</text>'
    )


def score_chart(
    *,
    away: str,
    home: str,
    our_away: float,
    our_home: float,
    book_away: float | None,
    book_home: float | None,
) -> str:
    """Our points beside the book's points. Each row is scaled on its own."""
    has_book = book_away is not None and book_home is not None
    team_peak = max(our_away, our_home, book_away or 0, book_home or 0, 1)
    total_peak = max(our_away + our_home, (book_away or 0) + (book_home or 0), 1)
    rows = [
        (away, our_away, book_away, team_peak),
        (home, our_home, book_home, team_peak),
        (
            "Total",
            our_away + our_home,
            (book_away + book_home) if has_book else None,
            total_peak,
        ),
    ]
    width = 640
    bar_x = 96
    bar_w = 430
    row_h = 46 if has_book else 30
    height = 28 + row_h * len(rows) + 6
    parts: list[str] = [
        f'<rect x="{bar_x}" y="6" width="11" height="11" rx="2" fill="{OURS}"/>',
        f'<text x="{bar_x + 16}" y="16" font-size="13" fill="{INK}">Ours</text>',
    ]
    if has_book:
        parts.append(f'<rect x="{bar_x + 78}" y="6" width="11" height="11" rx="2" fill="{BOOK}"/>')
        parts.append(f'<text x="{bar_x + 94}" y="16" font-size="13" fill="{INK}">Book</text>')
    for index, (label, ours, book, peak) in enumerate(rows):
        y = 28 + index * row_h
        label_y = y + (22 if has_book else 12)
        parts.append(
            f'<text x="8" y="{label_y}" font-size="14" font-weight="600" fill="{INK}">{_esc(label)}</text>'
        )
        _bar(parts, bar_x, y, bar_w, ours, peak, OURS)
        if has_book and book is not None:
            _bar(parts, bar_x, y + 16, bar_w, book, peak, BOOK)
    return _svg(width, height, "".join(parts))


def _hex(value: str | None, fallback: str) -> str:
    text = str(value or "").strip()
    if len(text) == 7 and text[0] == "#":
        try:
            int(text[1:], 16)
        except ValueError:
            return fallback
        return text.upper()
    return fallback


def _mix(color: str, other: str, amount: float) -> str:
    def parts(value: str) -> tuple[int, int, int]:
        return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))

    left, right = parts(color), parts(other)
    mixed = tuple(round(left[i] + (right[i] - left[i]) * amount) for i in range(3))
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def trend_chart(
    away: str,
    home: str,
    away_pts: list[float],
    home_pts: list[float],
    away_color: str | None = None,
    home_color: str | None = None,
) -> str:
    """Points scored in each team's recent games, as columns. Oldest at the left."""
    away_fill = _hex(away_color, AWAY)
    home_fill = _hex(home_color, HOME)
    series = [
        (name, pts, color, soft)
        for name, pts, color, soft in (
            (away, away_pts, away_fill, _mix(away_fill, "#F3EFE6", 0.5)),
            (home, home_pts, home_fill, _mix(home_fill, "#F3EFE6", 0.5)),
        )
        if pts
    ]
    if not series:
        return _svg(640, 40, f'<text x="8" y="24" font-size="14" fill="{MUTED}">No recent games</text>')
    ceiling = max(max(pts) for _, pts, _, _ in series) * 1.28
    ceiling = max(ceiling, 1)
    width = 640
    row_h = 128
    plot_h = 70
    col_w = 28
    gap = 16
    height = row_h * len(series) + 16
    parts: list[str] = []
    for index, (name, pts, color, soft) in enumerate(series):
        y0 = index * row_h
        average = sum(pts) / len(pts)
        count = len(pts)
        used = count * col_w + max(count - 1, 0) * gap
        origin = (width - used) / 2
        baseline = y0 + 112
        parts.append(
            f'<text x="{origin:.1f}" y="{y0 + 18}" font-size="13" font-weight="600" fill="{color}">{_esc(name)}</text>'
        )
        parts.append(
            f'<text x="{origin + used:.1f}" y="{y0 + 18}" font-size="12" fill="{MUTED}" text-anchor="end">avg {_pts(average)}</text>'
        )
        parts.append(
            f'<line x1="{origin:.1f}" y1="{baseline:.1f}" x2="{origin + used:.1f}" y2="{baseline:.1f}" stroke="{RULE}" stroke-width="1"/>'
        )
        for spot, value in enumerate(pts):
            height_px = max(plot_h * (float(value) / ceiling), 2)
            x = origin + spot * (col_w + gap)
            y = baseline - height_px
            fill = color if spot == count - 1 else soft
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{col_w}" height="{height_px:.1f}" rx="2" fill="{fill}"/>'
            )
            parts.append(
                f'<text x="{x + col_w / 2:.1f}" y="{y - 6:.1f}" font-size="13" fill="{INK}" text-anchor="middle">{_pts(value)}</text>'
            )
    parts.append(
        f'<text x="8" y="{height - 4}" font-size="12" fill="{MUTED}">Oldest on the left. Latest game is the solid column.</text>'
    )
    return _svg(width, height, "".join(parts))


def _implied(american: int) -> float:
    price = int(american)
    if price >= 0:
        return 100.0 / (price + 100.0)
    return abs(price) / (abs(price) + 100.0)


def payout_chart(*, label: str, payout: int, floor: int) -> str:
    """The chance implied by our price, next to the chance implied by the book's tax."""
    ours = _implied(payout) * 100
    book = _implied(floor) * 100
    label = label[:1].upper() + label[1:] if label else label
    scale = 50.0
    width = 640
    bar_x = 200.0
    bar_w = 320.0
    parts: list[str] = []
    for tick, anchor in ((0, "start"), (25, "middle"), (50, "end")):
        x = bar_x + bar_w * (tick / scale)
        parts.append(
            f'<text x="{x:.1f}" y="16" font-size="12" fill="{MUTED}" text-anchor="{anchor}">{tick:.0f}%</text>'
        )
    for name, value, color, y in (
        (f"{label} {payout:+d}", ours, OURS, 32),
        (f"FanDuel tax {floor:+d}", book, BOOK, 68),
    ):
        length = bar_w * min(value, scale) / scale
        parts.append(f'<text x="8" y="{y + 13}" font-size="14" fill="{INK}">{_esc(name)}</text>')
        parts.append(f'<rect x="{bar_x:.1f}" y="{y}" width="{bar_w:.1f}" height="14" rx="2" fill="{TRACK}"/>')
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{y}" width="{max(length, 1.5):.1f}" height="14" rx="2" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{bar_x + bar_w + 12:.1f}" y="{y + 12}" font-size="14" font-weight="600" fill="{INK}">{value:.0f}%</text>'
        )
    parts.append(
        f'<text x="8" y="112" font-size="12" fill="{MUTED}">Shorter price, higher chance.</text>'
    )
    return _svg(width, 126, "".join(parts))


def recent_points(schedules: pl.DataFrame, team: str, season: int, week: int, n: int = 8) -> list[float]:
    """Points this team scored before this week, oldest first."""
    if schedules.is_empty() or "home_score" not in schedules.columns:
        return []
    done = schedules.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    rows: list[tuple[int, int, float]] = []
    for game in done.iter_rows(named=True):
        g_season = int(game["season"])
        g_week = int(game["week"])
        if (g_season, g_week) >= (season, week):
            continue
        if game["home_team"] == team:
            rows.append((g_season, g_week, float(game["home_score"])))
        elif game["away_team"] == team:
            rows.append((g_season, g_week, float(game["away_score"])))
    rows.sort()
    return [points for _, _, points in rows[-n:]]
