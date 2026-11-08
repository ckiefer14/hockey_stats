"""
probe_lineup_sources.py
=======================
v2.2: check which pregame lineup / injury sources answer from this computer, and save a
sample of each so the parser can be written against the real format.

Home: scripts/probe_lineup_sources.py

Usage (from the project root, on a game day: once in the morning, once about an hour
before the first game):
    py .\\scripts\\probe_lineup_sources.py

Tries:
  1. ESPN's public injuries feed (unofficial, no key):
     site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries
     -> team abbreviations, statuses, number of injuries
  2. The NHL's own game-day feed for today's first game (the API load_all already uses):
     gamecenter/<id>/play-by-play -> rosterSpots: how many players per team and position
     (a dressed lineup is 20: 12 F, 6 D, 2 G), game state, and whether the ESPN-injured
     players of these two teams are on it
Saves the raw answers to output/probe/ and prints the summary. Send the printout.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from config.settings import OUTPUT_DIR
from src.data.db import get_connection

OUT = OUTPUT_DIR / "probe"
TIMEOUT = 20
ESPN = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
NHL = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"


def get(url):
    t0 = time.time()
    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "hockey-stats-probe"})
        ok = "json" in r.headers.get("content-type", "")
        return r.status_code, (r.json() if ok else None), time.time() - t0
    except Exception as e:
        return f"error: {type(e).__name__}: {e}", None, time.time() - t0


def espn_entries(data):
    """(team abbreviation, first, last, display name, status) for every ESPN injury."""
    out = []
    for t in (data or {}).get("injuries", []):
        for inj in t.get("injuries", []):
            ath = inj.get("athlete", {})
            out.append((ath.get("team", {}).get("abbreviation", "?"), ath.get("firstName", ""),
                        ath.get("lastName", ""), ath.get("displayName", ""), inj.get("status", "?")))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    print(f"Probe at {time.strftime('%Y-%m-%d %H:%M')}\n")

    # 1. ESPN injuries
    status, espn, sec = get(ESPN)
    print(f"1. ESPN injuries: status {status} ({sec:.1f} s)")
    ents = espn_entries(espn)
    if espn:
        (OUT / f"espn_injuries_{stamp}.json").write_text(json.dumps(espn, indent=1))
        sts = {}
        for e in ents:
            sts[e[4]] = sts.get(e[4], 0) + 1
        print(f"   {len(espn.get('injuries', []))} teams, {len(ents)} injuries; statuses {sts}")
        print(f"   team abbreviations: {sorted({e[0] for e in ents})}")

    # 2. NHL game-day roster for today's first game
    conn = get_connection()
    today = time.strftime("%Y-%m-%d")
    row = conn.execute("SELECT game_id, home_team_abbrev, away_team_abbrev FROM schedule WHERE game_date = ? "
                       "ORDER BY game_id LIMIT 1", (today,)).fetchone()
    conn.close()
    if not row:
        print(f"\n2. NHL: no game in the schedule for {today}; run again on a game day.")
        return
    gid, h, a = row
    status, data, sec = get(NHL.format(gid=gid))
    print(f"\n2. NHL play-by-play for {a} @ {h} (game {gid}): status {status} ({sec:.1f} s)")
    if not data:
        return
    (OUT / f"nhl_play-by-play_{gid}_{stamp}.json").write_text(json.dumps(data, indent=1))
    spots = data.get("rosterSpots", [])
    print(f"   game state {data.get('gameState')}, start (UTC) {data.get('startTimeUTC')}, rosterSpots {len(spots)}")
    teams = {data.get("homeTeam", {}).get("id"): h, data.get("awayTeam", {}).get("id"): a}
    for tid, ab in teams.items():
        mine = [x for x in spots if x.get("teamId") == tid]
        pos = {}
        for x in mine:
            pos[x.get("positionCode")] = pos.get(x.get("positionCode"), 0) + 1
        print(f"   {ab}: {len(mine)} players {pos}   (a dressed lineup is 20: 12 F, 6 D, 2 G)")
    names = {(x.get("firstName", {}).get("default", "").lower(), x.get("lastName", {}).get("default", "").lower())
             for x in spots}
    mine = [e for e in ents if e[0].upper()[:2] in (h[:2], a[:2])]
    print(f"   ESPN-injured players of these two teams ({len(mine)}), and whether they are in rosterSpots:")
    for ab, first, last, disp, st in mine:
        print(f"     {disp} ({ab}, {st}): {'IN rosterSpots' if (first.lower(), last.lower()) in names else 'not in rosterSpots'}")
    print(f"\nRaw answers saved in {OUT}")


if __name__ == "__main__":
    main()