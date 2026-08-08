from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from sports_ev.db.models import Game


def normalize_team(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for token in ("the ", "fc ", "football club"):
        if text.startswith(token):
            text = text[len(token) :]
    return text


def teams_match(stored: str, scraped: str) -> bool:
    a = normalize_team(stored)
    b = normalize_team(scraped)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    return bool(a_tokens & b_tokens)


def find_game_for_matchup(
    session: Session,
    *,
    sport: str = "nfl",
    home_team: str,
    away_team: str,
    kickoff_time: datetime | None = None,
    event_date: date | None = None,
) -> Game | None:
    candidates = session.query(Game).filter(Game.sport == sport.lower()).all()
    matches: list[Game] = []
    for game in candidates:
        if teams_match(game.home_team, home_team) and teams_match(game.away_team, away_team):
            matches.append(game)

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    target_date = event_date
    if target_date is None and kickoff_time is not None:
        target_date = kickoff_time.astimezone(timezone.utc).date()

    if target_date is not None:
        for game in matches:
            if game.kickoff_time.astimezone(timezone.utc).date() == target_date:
                return game

    if kickoff_time is not None:
        kickoff_date = kickoff_time.astimezone(timezone.utc).date()
        for game in matches:
            if game.kickoff_time.astimezone(timezone.utc).date() == kickoff_date:
                return game

    return matches[0]
