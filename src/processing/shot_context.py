"""
shot_context.py
===============
Derive shot-context flags (rebound, rush) from full play-by-play events.

Home: src/processing/shot_context.py

Design per the project data contract:
  - `game_events` is the raw, complete record (every event type, every
    period). This module DERIVES; it never mutates raw tables.
  - Output goes to a `shot_context` table keyed by (game_id, event_id) —
    the same key as shot_events — so the xG feature builder (Phase 2)
    joins it on at query time.
  - Chronological order comes from sort_order (the API's eventId is NOT
    in play order, so never sort by it).
  - Shootout (period 5) is excluded, consistent with the period <= 4
    filtering rule. Blocked shots inform rebound state but never receive
    flags themselves (not xG rows).

Blocked-shot frames (both verified against this database, not assumed):
  A blocked-shot record mixes two frames of reference, so each is
  handled by its own switch.

  TEAM — scripts/check_block_attribution.py compares the shooter's team
  against the stored team_id and found the owner IS the shooting team
  (90/10). So blocked_shots_owned_by_blocker defaults to False and no
  team re-attribution happens. If a future API change inverts this, set
  it True.

  ZONE — scripts/check_block_zone_frame.py compares blocked-shot
  coordinates against the same team's shots on goal and found blocks sit
  on that team's OFFENSIVE end 99.9% of the time, despite carrying
  zone_code 'D'. The zone is therefore written from the DEFENDING side
  while the owner is the shooter. blocked_zone_in_defending_frame
  defaults to True, flipping O<->D so the zone matches the owner.

  Without the zone flip the RUSH rule inverts on blocks: a shooter's own
  follow-up reads as a rush (it is a rebound), while the blocking team's
  genuine counterattack does not. REBOUNDS are unaffected either way,
  since that rule never reads zone codes.

Definitions (NST conventions, configurable):
  rebound: unblocked shot attempt by team T within REBOUND_WINDOW seconds
           of any prior shot attempt by T (incl. blocked), same period.
  rush:    unblocked shot attempt within RUSH_WINDOW seconds of any event
           in the neutral zone or the shooting team's defensive zone.

Plausibility notes for sanity checks: rebounds land around ~8-13% of
unblocked shots depending on the window (NST uses 3s; MoneyPuck 2s).
Rush rates with this event-sparse definition land in the mid single
digits; validate against NST per-game rush attempt counts rather than a
fixed band.

Typical usage (end of a load session):
    from src.data.db import get_connection
    from src.processing.shot_context import build_shot_context
    conn = get_connection()
    build_shot_context(conn)

After changing windows or attribution settings, rebuild from scratch:
    build_shot_context(conn, rebuild=True)
"""

from __future__ import annotations

import sqlite3

import pandas as pd

REBOUND_WINDOW = 3.0   # s; MoneyPuck uses 2.0 — set to 2.0 to validate against them
RUSH_WINDOW = 4.0      # s

SHOT_EVENTS = {"shot-on-goal", "missed-shot", "goal"}
ATTEMPT_EVENTS = SHOT_EVENTS | {"blocked-shot"}

# Events that break the play chain. Both the NST rebound and rush definitions
# require "no stoppage in play in between", so after one of these the rebound
# and rush context is cleared exactly as it is at a period boundary.
# A goal is included: it is evaluated as a shot first (a goal can itself be a
# rebound), then clears the chain, since play stops after it.
STOPPAGE_EVENTS = {
    "stoppage", "faceoff", "goal", "penalty",
    "period-start", "period-end", "game-end", "shootout-complete",
}

_ZONE_FLIP = {"O": "D", "D": "O"}


def derive_shot_context(
    events: pd.DataFrame,
    rebound_window: float = REBOUND_WINDOW,
    rush_window: float = RUSH_WINDOW,
    blocked_shots_owned_by_blocker: bool = False,
    blocked_zone_in_defending_frame: bool = True,
    team_pair: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """
    Pure function: one game's events in, one row per unblocked non-shootout
    shot out, with derived context columns.

    Required input columns (game_events schema):
        game_id, event_id, sort_order, period, game_seconds,
        event_type, team_id, zone_code

    blocked_shots_owned_by_blocker
        If True (default, matches the NHL API), re-attribute each
        blocked-shot to the opposing team and flip its zone before it
        enters the bookkeeping.
    team_pair
        The two team_ids in this game, used to resolve "the opposing
        team" for re-attribution. Inferred from the data if omitted
        (build_shot_context passes it explicitly from the schedule).

    Returns columns:
        game_id, event_id, is_rebound, is_rush,
        time_since_last_event, last_event_type, last_event_zone
    """
    ev = events[events["period"] <= 4].sort_values("sort_order")

    if blocked_shots_owned_by_blocker and team_pair is None:
        teams = ev["team_id"].dropna().unique()
        team_pair = tuple(int(t) for t in teams[:2]) if len(teams) >= 2 else None

    def other_team(t):
        if team_pair is None or t not in team_pair:
            return None
        return team_pair[1] if t == team_pair[0] else team_pair[0]

    out = []
    last_attempt_time: dict[int, float] = {}   # team_id -> game_seconds
    prev_time = prev_zone = prev_team = prev_type = None
    prev_period = None

    for row in ev.itertuples(index=False):
        if row.period != prev_period:
            # period boundary: no rebound/rush context carries over
            last_attempt_time.clear()
            prev_time = prev_zone = prev_team = prev_type = None
            prev_period = row.period

        # Normalize blocked shots so team AND zone are both expressed from the
        # shooting team's perspective. The API mixes frames on these events:
        # the owner is the shooter but the zone is written from the defending
        # side (verified by scripts/check_block_zone_frame.py: blocked shots sit
        # on the same end of the ice as that team's shots on goal, 99.9%).
        eff_team, eff_zone = row.team_id, row.zone_code
        if row.event_type == "blocked-shot":
            if blocked_zone_in_defending_frame:
                # 'D' in the defending frame is the shooter's OFFENSIVE zone
                eff_zone = _ZONE_FLIP.get(row.zone_code, row.zone_code)
            if blocked_shots_owned_by_blocker and pd.notna(row.team_id):
                flipped = other_team(row.team_id)
                if flipped is not None:
                    eff_team = flipped

        if row.event_type in SHOT_EVENTS and pd.notna(row.team_id):
            # rebound: prior same-team attempt within window
            prior = last_attempt_time.get(row.team_id)
            is_rebound = prior is not None and (row.game_seconds - prior) <= rebound_window

            # rush: prior event in N or shooter's D zone within window
            is_rush = False
            if prev_time is not None and (row.game_seconds - prev_time) <= rush_window:
                zone = prev_zone
                if prev_team is not None and prev_team != row.team_id:
                    zone = _ZONE_FLIP.get(zone, zone)
                is_rush = zone in ("N", "D")

            out.append({
                "game_id": row.game_id,
                "event_id": row.event_id,
                "is_rebound": bool(is_rebound),
                "is_rush": bool(is_rush),
                "time_since_last_event": (
                    row.game_seconds - prev_time if prev_time is not None else None
                ),
                "last_event_type": prev_type,
                "last_event_zone": prev_zone,
            })

        # update state AFTER evaluating the current event,
        # using the EFFECTIVE (re-attributed) team and zone
        if row.event_type in ATTEMPT_EVENTS and pd.notna(eff_team):
            last_attempt_time[eff_team] = row.game_seconds
        prev_time, prev_zone = row.game_seconds, eff_zone
        prev_team, prev_type = eff_team, row.event_type

        # no-stoppage rule: a whistle severs the chain for both flags
        if row.event_type in STOPPAGE_EVENTS:
            last_attempt_time.clear()
            prev_time = prev_zone = prev_team = prev_type = None

    return pd.DataFrame(out)


def build_shot_context(conn: sqlite3.Connection, rebuild: bool = False) -> int:
    """
    Materialize the shot_context table from game_events for every game not
    yet processed (or all games if rebuild=True). Returns rows written.

    Idempotent: safe to re-run after each load_all.py / backfill session.
    Team pairs for blocked-shot re-attribution come from the schedule table.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shot_context (
            game_id INTEGER,
            event_id INTEGER,
            is_rebound INTEGER,
            is_rush INTEGER,
            time_since_last_event REAL,
            last_event_type TEXT,
            last_event_zone TEXT,
            PRIMARY KEY (game_id, event_id),
            FOREIGN KEY (game_id) REFERENCES games(id)
        )
    """)
    if rebuild:
        conn.execute("DELETE FROM shot_context")
        conn.commit()

    todo = pd.read_sql(
        """SELECT DISTINCT game_id FROM game_events
           WHERE game_id NOT IN (SELECT DISTINCT game_id FROM shot_context)""",
        conn,
    )["game_id"].tolist()

    written = 0
    for gid in todo:
        events = pd.read_sql(
            """SELECT game_id, event_id, sort_order, period, game_seconds,
                      event_type, team_id, zone_code
               FROM game_events WHERE game_id = ?""",
            conn, params=(gid,),
        )
        pair_row = conn.execute(
            "SELECT home_team_id, away_team_id FROM schedule WHERE game_id = ?",
            (gid,),
        ).fetchone()
        team_pair = tuple(pair_row) if pair_row else None

        ctx = derive_shot_context(events, team_pair=team_pair)
        if len(ctx):
            ctx.to_sql("shot_context", conn, if_exists="append", index=False)
            written += len(ctx)
    conn.commit()
    return written