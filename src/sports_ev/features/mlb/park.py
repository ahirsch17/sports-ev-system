from __future__ import annotations

# Approximate home-run / run-scoring park factors (1.0 = neutral).
# Source: public park factor estimates; coarse v1 lookup by home team name.
PARK_FACTORS: dict[str, float] = {
    "arizona diamondbacks": 1.05,
    "atlanta braves": 1.00,
    "baltimore orioles": 1.03,
    "boston red sox": 1.06,
    "chicago cubs": 1.02,
    "chicago white sox": 1.04,
    "cincinnati reds": 1.05,
    "cleveland guardians": 0.98,
    "colorado rockies": 1.12,
    "detroit tigers": 0.99,
    "houston astros": 1.01,
    "kansas city royals": 0.98,
    "los angeles angels": 0.99,
    "los angeles dodgers": 0.97,
    "miami marlins": 0.96,
    "milwaukee brewers": 1.01,
    "minnesota twins": 1.00,
    "new york mets": 0.98,
    "new york yankees": 1.04,
    "oakland athletics": 0.95,
    "philadelphia phillies": 1.02,
    "pittsburgh pirates": 0.97,
    "san diego padres": 0.96,
    "san francisco giants": 0.94,
    "seattle mariners": 0.97,
    "st. louis cardinals": 1.00,
    "tampa bay rays": 0.98,
    "texas rangers": 1.05,
    "toronto blue jays": 1.03,
    "washington nationals": 1.00,
}


def park_factor_for_team(home_team: str) -> float:
    key = home_team.lower().strip()
    return PARK_FACTORS.get(key, 1.0)
