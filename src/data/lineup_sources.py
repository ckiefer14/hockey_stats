"""
lineup_sources.py
=================
v2.2 live lineups: the two pregame sources, as tested with scripts/probe_lineup_sources.py.

Home: src/data/lineup_sources.py

  ESPN injuries (unofficial public feed, no key): every team's injured players with a status:
      "Injured Reserve", "Out", "Day-To-Day". Team abbreviations match the NHL's except
      LA, NJ, SJ, TB (mapped below).
  NHL game-day roster (gamecenter/<id>/play-by-play -> rosterSpots): about 23 players per
      team on game day: the active roster, without injured-reserve players but with healthy
      scratches and "Out" / day-to-day players. If it lists 20 per team (12 F, 6 D, 2 G),
      that is the dressed lineup.
Both can fail or change format: callers treat a failure as "no information" and say so.
"""

import pandas as pd
import requests

TIMEOUT = 20
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
NHL_PBP = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
ESPN_TEAM = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL"}
HEADERS = {"User-Agent": "hockey-stats"}


def fetch_injuries() -> pd.DataFrame:
    """ESPN's injury list. Columns: team (NHL abbreviation), first, last, name, status, return_date, note."""
    r = requests.get(ESPN_INJURIES, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    rows = []
    for t in r.json().get("injuries", []):
        for inj in t.get("injuries", []):
            ath = inj.get("athlete", {})
            ab = ath.get("team", {}).get("abbreviation", "")
            rows.append({"team": ESPN_TEAM.get(ab, ab), "first": ath.get("firstName", ""), "last": ath.get("lastName", ""),
                         "name": ath.get("displayName", ""), "status": inj.get("status", ""),
                         "return_date": (inj.get("details") or {}).get("returnDate"),
                         "note": inj.get("shortComment", "")})
    return pd.DataFrame(rows, columns=["team", "first", "last", "name", "status", "return_date", "note"])


def fetch_roster_spots(game_id: int) -> tuple[pd.DataFrame, str | None]:
    """The NHL's game-day roster for one game. Returns (DataFrame team, player_id, first, last, pos; game state)."""
    r = requests.get(NHL_PBP.format(gid=int(game_id)), timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    d = r.json()
    ab = {d.get("homeTeam", {}).get("id"): d.get("homeTeam", {}).get("abbrev"),
          d.get("awayTeam", {}).get("id"): d.get("awayTeam", {}).get("abbrev")}
    rows = [{"team": ab.get(x.get("teamId")), "player_id": x.get("playerId"),
             "first": (x.get("firstName") or {}).get("default", ""), "last": (x.get("lastName") or {}).get("default", ""),
             "pos": x.get("positionCode")} for x in d.get("rosterSpots", [])]
    return pd.DataFrame(rows, columns=["team", "player_id", "first", "last", "pos"]), d.get("gameState")