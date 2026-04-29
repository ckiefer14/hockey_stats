# src/processing/corsi.py
from src.data.db import get_connection, get_team_id, get_finished_game_ids

def time_to_seconds(time_str: str) -> int:
    """Convert 'MM:SS' to total seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])

def calculate_player_game_fenwick(player_id: int, game_id: int, team_abbrev: str) -> dict | None:
    """
    Calculate Fenwick for a single player in a single game using shift data.
    Counts all unblocked shot attempts that occurred while the player was on ice.
    """
    conn = get_connection()
    team_id = get_team_id(conn, team_abbrev)

    if team_id is None:
        conn.close()
        return None

    shifts = conn.execute("""
        SELECT period, start_time, end_time FROM shifts
        WHERE game_id = ? AND player_id = ?
    """, (game_id, player_id)).fetchall()

    if not shifts:
        conn.close()
        return None
    
    shot_events = conn.execute("""
        SELECT period, time_in_period, team_id FROM shot_events
        WHERE game_id = ? AND event_type != 'blocked-shot'
    """, (game_id,)).fetchall()

    conn.close()

    ff = 0
    fa = 0

    for shot_period, shot_time_str, shot_team_id in shot_events:
        shot_seconds = time_to_seconds(shot_time_str)

        # Check if this player was on ice during this shot
        on_ice = False
        for shift_period, shift_start, shift_end in shifts:
            if shift_period != shot_period:
                continue
            start_sec = time_to_seconds(shift_start)
            end_sec = time_to_seconds(shift_end)
            if start_sec <= shot_seconds <= end_sec:
                on_ice = True
                break

        if on_ice:
            if shot_team_id == team_id:
                ff += 1
            else:
                fa += 1

    total = ff + fa
    ff_pct = (ff / total) * 100 if total > 0 else 0

    return {
        "player_id": player_id,
        "game_id": game_id,
        "team_abbrev": team_abbrev,
        "ff": ff,
        "fa": fa,
        "ff_pct": round(ff_pct, 1),
        "ff_diff": ff - fa,
    }


def calculate_player_season_fenwick(player_id: int, team_abbrev: str) -> dict | None:
    """
    Calculate season Fenwick for a single player.
    Builds on calculate_player_game_corsi.
    """

    conn = get_connection()
    game_ids = get_finished_game_ids(conn, team_abbrev)
    conn.close()

    if not game_ids:
        return None

    total_ff = 0
    total_fa = 0
    games_played = 0

    for game_id in game_ids:
        result = calculate_player_game_fenwick(player_id, game_id, team_abbrev)
        if result:
            total_ff += result["ff"]
            total_fa += result["fa"]
            games_played += 1

    total = total_ff + total_fa
    ff_pct = (total_ff / total) * 100 if total > 0 else 0

    return {
        "player_id": player_id,
        "team_abbrev": team_abbrev,
        "games_played": games_played,
        "total_ff": total_ff,
        "total_fa": total_fa,
        "ff_pct": round(ff_pct, 1),
        "ff_diff": total_ff - total_fa,
    }

def calculate_game_fenwick(game_id: int, team_abbrev: str) -> dict | None:
    """
    Calculate Corsi stats for a team in a single game.
    This is the base function everything else builds on.
    """
    conn = get_connection()
    team_id = get_team_id(conn, team_abbrev)

    if team_id is None:
        conn.close()
        return None

    ff = conn.execute("""
        SELECT COUNT(*) FROM shot_events
        WHERE game_id = ? AND team_id = ? AND event_type != 'blocked-shot'
    """, (game_id, team_id)).fetchone()[0]

    fa = conn.execute("""
        SELECT COUNT(*) FROM shot_events
        WHERE game_id = ? AND team_id != ? AND event_type != 'blocked-shot'
    """, (game_id, team_id)).fetchone()[0]

    game = conn.execute("""
        SELECT home_team_abbrev, away_team_abbrev, home_score, away_score
        FROM games WHERE id = ?
    """, (game_id,)).fetchone()

    conn.close()

    ff_pct = (ff / (ff + fa) * 100) if (ff + fa) > 0 else 0

    if game:
        if game[0] == team_abbrev:
            # Team is home
            team_score = game[2]
            opp_score = game[3]
        else:
            # team is away
            team_score = game[3]
            opp_score = game[2]
        won = team_score > opp_score
    else:
        team_score = 0
        opp_score = 0
        won = False
    

    return {
        "game_id": game_id,
        "team_abbrev": team_abbrev,
        "ff": ff,
        "fa": fa,
        "ff_pct": round(ff_pct, 1),
        "ff_diff": ff - fa,
        "won": won,
        "score": f"{team_score}-{opp_score}",
    }


def calculate_season_fenwick(team_abbrev: str) -> dict | None:
    """
    Calculate season Fenwick by summing all game Corsi values.
    Builds on calculate_game_corsi.
    """
    conn = get_connection()
    game_ids = get_finished_game_ids(conn, team_abbrev)
    conn.close()

    if not game_ids:
        return None

    total_ff = 0
    total_fa = 0
    game_results = []

    for game_id in game_ids:
        result = calculate_game_fenwick(game_id, team_abbrev)
        if result:
            total_ff += result["ff"]
            total_fa += result["fa"]
            game_results.append(result)

    ff_pct = (total_ff / (total_ff + total_fa) * 100) if (total_ff + total_fa) > 0 else 0

    return {
        "team_abbrev": team_abbrev,
        "games_played": len(game_results),
        "total_ff": total_ff,
        "total_fa": total_fa,
        "ff_pct": round(ff_pct, 1),
        "ff_diff": total_ff - total_fa,
        "per_game": game_results,
    }