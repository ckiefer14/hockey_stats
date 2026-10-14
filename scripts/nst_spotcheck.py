# scripts/nst_spotcheck.py
"""
EXTERNAL VALIDATION: compare our rush/rebound flags against Natural Stat Trick.

This is the one check that tests our flags against something outside our own
database. Everything in data_sanity.ipynb tests internal consistency.

HOW TO USE
----------
1. Pick 3-4 games from the current season and put their game_ids in GAMES below.
2. For each, open the NST game report:
       naturalstattrick.com -> Games -> pick the date -> the game's report
   Set the report to ALL SITUATIONS (not 5v5) to match our period<=4,
   all-strengths derivation.
3. In the Individual tables (one per team), sum the "Rush Attempts" column for
   each team and enter it in the nst_rush dict. Do the same with
   "Rebounds Created" -> nst_rebounds_created if you want the rebound row too.
4. Run:  python scripts/nst_spotcheck.py

WHY THE NUMBERS WON'T MATCH EXACTLY (and shouldn't)
---------------------------------------------------
* NST's Rush Attempts counts goals, shots on net, misses AND BLOCKS. Our
  shot_context only generates rows for unblocked shots, because blocked shots
  are excluded from xG (their coordinates are the block location, not the
  shot location). Blocks are ~26% of attempts, so OUR COUNT SHOULD RUN LOWER,
  roughly 70-80% of theirs.
* NST's "Rebounds Created" counts the attempt that CAUSED a rebound; our
  is_rebound counts the follow-up shot itself. Near 1:1 conceptually, but
  theirs includes blocked follow-ups, so again ours runs lower.

So we are checking SHAPE, not equality:
  - does the same team lead in each game?
  - is our/NST ratio stable across games (not swinging wildly)?
A stable ratio in the 0.6-0.9 band with matching leaders = pass.
Reversed leaders, or a ratio jumping between e.g. 0.3 and 1.2 = investigate.
"""
import sys

sys.path.insert(0, ".")

import pandas as pd

from src.data.db import get_connection

# ---------------------------------------------------------------------------
# FILL THIS IN. Keys are game_ids; inner keys are team abbreviations.
# Leave a game's dict empty to just print our numbers for it first.
# ---------------------------------------------------------------------------
GAMES = {
    2025020025: {"nst_rush": {"DET": 0, "TOR": 2}, "nst_rebounds_created": {"DET": 3, "TOR": 0}},
    2025020742: {"nst_rush": {"SJS": 0, "WSH": 2}, "nst_rebounds_created": {"SJS": 3, "WSH": 2}},
    2025020993: {"nst_rush": {"PHI": 1, "PIT": 3}, "nst_rebounds_created": {"PHI": 2, "PIT": 4}},
    # 2025020034: {"nst_rush": {"DAL": 0, "COL": 0}, "nst_rebounds_created": {}},
}
# Example once filled:
#   2025020971: {"nst_rush": {"WPG": 11, "COL": 14},
#                "nst_rebounds_created": {"WPG": 8, "COL": 12}},


def our_counts(conn, game_id):
    return pd.read_sql("""
        SELECT t.abbreviation AS team,
               SUM(sc.is_rush)     AS our_rush,
               SUM(sc.is_rebound)  AS our_rebound,
               COUNT(*)            AS unblocked_shots
        FROM shot_context sc
        JOIN shot_events se ON se.game_id = sc.game_id AND se.event_id = sc.event_id
        JOIN teams t        ON t.team_id  = se.team_id
        WHERE sc.game_id = ?
        GROUP BY t.abbreviation
        ORDER BY t.abbreviation
    """, conn, params=(game_id,))


def main():
    conn = get_connection()
    rows = []

    for game_id, ref in GAMES.items():
        meta = conn.execute("""
            SELECT game_date, away_team_abbrev, home_team_abbrev
            FROM schedule WHERE game_id = ?""", (game_id,)).fetchone()
        if meta is None:
            print(f"game_id {game_id} not in schedule — skipping\n")
            continue
        date, away, home = meta
        ours = our_counts(conn, game_id)
        print(f"\n=== {game_id}  {away} @ {home}  {date} ===")
        print(ours.to_string(index=False))

        nst_rush = ref.get("nst_rush") or {}
        nst_reb = ref.get("nst_rebounds_created") or {}
        if not nst_rush:
            print("  (no NST numbers entered yet — look up Rush Attempts "
                  "per team and add them to GAMES)")
            continue

        for r in ours.itertuples():
            rows.append({
                "game_id": game_id, "team": r.team,
                "our_rush": int(r.our_rush), "nst_rush": nst_rush.get(r.team),
                "our_reb": int(r.our_rebound),
                "nst_reb_created": nst_reb.get(r.team),
            })

    conn.close()

    if not rows:
        print("\nNo NST numbers entered yet. Fill in GAMES and re-run.")
        return

    df = pd.DataFrame(rows).dropna(subset=["nst_rush"])
    df["rush_ratio"] = (df.our_rush / df.nst_rush).round(2)
    print("\n\n=== COMPARISON ===")
    print(df.to_string(index=False))

    print("\nSHAPE CHECKS")
    # 1. same team leads within each game
    agree = []
    for gid, g in df.groupby("game_id"):
        if len(g) == 2:
            ours_lead = g.our_rush.idxmax()
            nst_lead = g.nst_rush.idxmax()
            tie = g.our_rush.nunique() == 1 or g.nst_rush.nunique() == 1
            agree.append(tie or ours_lead == nst_lead)
    if agree:
        print(f"  same team leads: {sum(agree)}/{len(agree)} games "
              f"({'OK' if all(agree) else 'INVESTIGATE the disagreeing game'})")

    # 2. ratio stability
    lo, hi = df.rush_ratio.min(), df.rush_ratio.max()
    print(f"  our/NST rush ratio: min {lo:.2f}, max {hi:.2f}, "
          f"mean {df.rush_ratio.mean():.2f}")
    if 0.55 <= df.rush_ratio.mean() <= 0.95 and (hi - lo) <= 0.35:
        print("    OK — stable and in the expected band (we exclude blocked attempts)")
    else:
        print("    INVESTIGATE — expected a stable ratio around 0.6-0.9")

    # 3. correlation, if enough team-games
    if len(df) >= 6:
        print(f"  correlation across team-games: r = {df.our_rush.corr(df.nst_rush):.2f} "
              f"(want > 0.8)")


if __name__ == "__main__":
    main()