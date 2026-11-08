"""
odds.py
=======
Sportsbook odds from The Odds API (https://the-odds-api.com), for the market
comparison (v2.0b item 1).

Home: src/data/odds.py

API key: put ODDS_API_KEY=your_key in a .env file in the project root (.env is
gitignored), or set the ODDS_API_KEY environment variable.

Cost: each call uses (markets x regions) credits; h2h in the us region = 1 credit.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from config.settings import PROJECT_ROOT

API_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"
EVENTS_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/events"     # start times; costs no credits

# The Odds API uses full team names; aliases cover renames and spellings.
TEAM_ABBR = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
    "Montréal Canadiens": "MTL", "Nashville Predators": "NSH", "New Jersey Devils": "NJD",
    "New York Islanders": "NYI", "New York Rangers": "NYR", "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA", "St Louis Blues": "STL", "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR", "Utah Mammoth": "UTA",
    "Utah Hockey Club": "UTA", "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}


def api_key() -> str:
    """ODDS_API_KEY from the environment or the project's .env file."""
    key = os.environ.get("ODDS_API_KEY")
    env = PROJECT_ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ODDS_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("No API key. Get a free key at https://the-odds-api.com and add the line\n"
                         "    ODDS_API_KEY=your_key\n"
                         f"to {env} (it is gitignored).")
    return key


def fetch_odds(markets: str = "h2h", regions: str = "us", bookmakers: str | None = None) -> tuple[list, dict]:
    """Current NHL odds. Returns (events, quota headers)."""
    params = {"apiKey": api_key(), "regions": regions, "markets": markets, "oddsFormat": "american"}
    if bookmakers:
        params["bookmakers"] = bookmakers
    r = requests.get(API_URL, params=params, timeout=30)
    if r.status_code == 401:
        raise SystemExit("The Odds API rejected the key (401). Check ODDS_API_KEY in .env.")
    r.raise_for_status()
    quota = {k: r.headers.get(f"x-requests-{k}") for k in ("remaining", "used", "last")}
    return r.json(), quota


def fetch_events() -> list:
    """Upcoming NHL events (id, home_team, away_team, commence_time). Free: no credits used."""
    r = requests.get(EVENTS_URL, params={"apiKey": api_key()}, timeout=30)
    if r.status_code == 401:
        raise SystemExit("The Odds API rejected the key (401). Check ODDS_API_KEY in .env.")
    r.raise_for_status()
    return r.json()


def american_to_prob(ml) -> float:
    """Implied probability of American odds, with the bookmaker's margin still in."""
    ml = float(ml)
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def events_to_rows(events: list, snapshot_at: str | None = None) -> pd.DataFrame:
    """One row per game x bookmaker: moneylines, no-vig P(home), and totals / puck
    line when those markets were requested."""
    snapshot_at = snapshot_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, unknown = [], set()
    for ev in events:
        home, away = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
        if home is None or away is None:
            unknown |= {ev["home_team"], ev["away_team"]} - set(TEAM_ABBR)
            continue
        for book in ev.get("bookmakers", []):
            row = {"snapshot_at": snapshot_at, "commence_time": ev["commence_time"], "event_id": ev["id"],
                   "home": home, "away": away, "book": book["key"], "book_updated": book.get("last_update")}
            for m in book.get("markets", []):
                out = {o["name"]: o for o in m.get("outcomes", [])}
                if m["key"] == "h2h" and ev["home_team"] in out and ev["away_team"] in out:
                    row["home_ml"] = out[ev["home_team"]]["price"]
                    row["away_ml"] = out[ev["away_team"]]["price"]
                elif m["key"] == "totals" and "Over" in out and "Under" in out:
                    row["total_line"] = out["Over"].get("point")
                    row["over_price"], row["under_price"] = out["Over"]["price"], out["Under"]["price"]
                elif m["key"] == "spreads" and ev["home_team"] in out and ev["away_team"] in out:
                    row["home_spread"] = out[ev["home_team"]].get("point")
                    row["home_spread_price"] = out[ev["home_team"]]["price"]
                    row["away_spread_price"] = out[ev["away_team"]]["price"]
            if "home_ml" in row:
                ph, pa = american_to_prob(row["home_ml"]), american_to_prob(row["away_ml"])
                row["overround"] = ph + pa                      # about 1.04-1.05: the book's margin
                row["p_home_novig"] = ph / (ph + pa)            # margin removed
            rows.append(row)
    if unknown:
        print(f"WARNING  unknown team names (add them to TEAM_ABBR in src/data/odds.py): {sorted(unknown)}")
    return pd.DataFrame(rows)


def match_games(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach game_id, season and game_date from the schedule (same home/away, date
    within a day of the start time in UTC)."""
    if df.empty:
        return df
    sched = schedule.copy()
    sched["game_date"] = pd.to_datetime(sched["game_date"])
    start = pd.to_datetime(df["commence_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.assign(_start=start.dt.normalize())
    m = df.merge(sched, on=["home", "away"], how="left")
    m = m[(m["game_date"] - m["_start"]).abs() <= pd.Timedelta(days=1)]
    m = m.sort_values("game_date").drop_duplicates(["event_id", "book"])
    return m.drop(columns=["_start"])


def consensus(odds: pd.DataFrame, before_start: bool = True) -> pd.DataFrame:
    """One market P(home win) per game: the median no-vig probability across books
    in the last snapshot taken before the game started (closest to the closing line)."""
    o = odds.dropna(subset=["p_home_novig"]).copy()
    if o.empty:
        return pd.DataFrame(columns=["game_id", "p_market", "books", "market_snapshot", "overround", "minutes_before"])
    o["snap"] = pd.to_datetime(o["snapshot_at"], utc=True)
    o["start"] = pd.to_datetime(o["commence_time"], utc=True)
    if before_start:
        o = o[o["snap"] < o["start"]]
    last = o.groupby("game_id")["snap"].transform("max")
    o = o[o["snap"] == last]
    o["minutes_before"] = (o["start"] - o["snap"]).dt.total_seconds() / 60
    out = o.groupby("game_id").agg(p_market=("p_home_novig", "median"), books=("book", "nunique"),
                                   market_snapshot=("snapshot_at", "max"), overround=("overround", "median"),
                                   minutes_before=("minutes_before", "first"))
    return out.reset_index()