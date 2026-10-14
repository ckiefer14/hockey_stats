# Database Schema

_Auto-generated from the live database on 2026-10-14 10:25._
_Regenerate with `python scripts/generate_schema_doc.py`._

**10 tables.**

## `game_events`

2,060,996 rows.

| Column | Type | Primary Key | Not Null |
|---|---|---|---|
| `game_id` | INTEGER | ✓ |  |
| `event_id` | INTEGER | ✓ |  |
| `sort_order` | INTEGER |  |  |
| `period` | INTEGER |  |  |
| `period_type` | TEXT |  |  |
| `time_in_period` | TEXT |  |  |
| `game_seconds` | REAL |  |  |
| `event_type` | TEXT |  |  |
| `team_id` | INTEGER |  |  |
| `zone_code` | TEXT |  |  |
| `x_coord` | REAL |  |  |
| `y_coord` | REAL |  |  |
| `shot_type` | TEXT |  |  |
| `shooting_player_id` | INTEGER |  |  |
| `goalie_id` | INTEGER |  |  |
| `situation_code` | TEXT |  |  |

**Foreign keys:**

- `game_id` → `games.id`

## `games`

6,560 rows.

| Column | Type | Primary Key | Not Null |
|---|---|---|---|
| `id` | INTEGER | ✓ |  |
| `season` | INTEGER |  |  |
| `game_type` | INTEGER |  |  |
| `game_date` | TEXT |  |  |
| `venue` | TEXT |  |  |
| `home_team_abbrev` | TEXT |  |  |
| `away_team_abbrev` | TEXT |  |  |
| `home_score` | INTEGER |  |  |
| `away_score` | INTEGER |  |  |
| `home_sog` | INTEGER |  |  |
| `away_sog` | INTEGER |  |  |
| `game_state` | TEXT |  |  |
| `period_type` | TEXT |  |  |

**Foreign keys:**

- `away_team_abbrev` → `teams.abbreviation`
- `home_team_abbrev` → `teams.abbreviation`

## `goalie_game_stats`

26,238 rows.

| Column | Type | Primary Key | Not Null |
|---|---|---|---|
| `game_id` | INTEGER | ✓ |  |
| `player_id` | INTEGER | ✓ |  |
| `team_abbreviation` | TEXT |  |  |
| `name` | TEXT |  |  |
| `saves` | INTEGER |  |  |
| `shots_against` | INTEGER |  |  |
| `goals_against` | INTEGER |  |  |
| `save_percentage` | REAL |  |  |
| `toi` | TEXT |  |  |
| `starter` | INTEGER |  |  |
| `decision` | TEXT |  |  |
| `even_strength_goals_against` | INTEGER |  |  |
| `power_play_goals_against` | INTEGER |  |  |
| `shorthanded_goals_against` | INTEGER |  |  |

**Foreign keys:**

- `player_id` → `players.id`
- `game_id` → `games.id`

## `players`

876 rows.

| Column | Type | Primary Key | Not Null |
|---|---|---|---|
| `id` | INTEGER | ✓ |  |
| `first_name` | TEXT |  | ✓ |
| `last_name` | TEXT |  | ✓ |
| `team_abbreviation` | TEXT |  |  |
| `headshot` | TEXT |  |  |
| `sweaterNumber` | INTEGER |  |  |
| `shoots_catches` | TEXT |  |  |
| `position_code` | TEXT |  |  |
| `height_inches` | INTEGER |  |  |
| `weight_pounds` | INTEGER |  |  |
| `birth_date` | TEXT |  |  |
| `age` | INTEGER |  |  |
| `birth_city` | TEXT |  |  |
| `birth_country` | TEXT |  |  |
| `birth_state_province` | TEXT |  |  |

**Foreign keys:**

- `team_abbreviation` → `teams.abbreviation`

## `schedule`

6,560 rows.

| Column | Type | Primary Key | Not Null |
|---|---|---|---|
| `game_id` | INTEGER | ✓ |  |
| `season` | INTEGER |  |  |
| `game_type` | INTEGER |  |  |
| `game_date` | TEXT |  |  |
| `home_team_abbrev` | TEXT |  |  |
| `away_team_abbrev` | TEXT |  |  |
| `game_state` | TEXT |  |  |
| `home_team_id` | INTEGER |  |  |
| `away_team_id` | INTEGER |  |  |

**Foreign keys:**

- `away_team_abbrev` → `teams.abbreviation`
- `home_team_abbrev` → `teams.abbreviation`

## `shifts`

4,578,542 rows.

| Column | Type | Primary Key | Not Null |
|---|---|---|---|
| `id` | INTEGER | ✓ |  |
| `game_id` | INTEGER |  |  |
| `player_id` | INTEGER |  |  |
| `team_id` | INTEGER |  |  |
| `team_abbrev` | TEXT |  |  |
| `first_name` | TEXT |  |  |
| `last_name` | TEXT |  |  |
| `period` | INTEGER |  |  |
| `shift_number` | INTEGER |  |  |
| `start_time` | TEXT |  |  |
| `end_time` | TEXT |  |  |
| `duration` | TEXT |  |  |

**Foreign keys:**

- `game_id` → `games.id`

## `shot_context`

566,545 rows.

| Column | Type | Primary Key | Not Null |
|---|---|---|---|
| `game_id` | INTEGER | ✓ |  |
| `event_id` | INTEGER | ✓ |  |
| `is_rebound` | INTEGER |  |  |
| `is_rush` | INTEGER |  |  |
| `time_since_last_event` | REAL |  |  |
| `last_event_type` | TEXT |  |  |
| `last_event_zone` | TEXT |  |  |

**Foreign keys:**

- `game_id` → `games.id`

## `shot_events`

773,259 rows.

| Column | Type | Primary Key | Not Null |
|---|---|---|---|
| `game_id` | INTEGER | ✓ |  |
| `event_id` | INTEGER | ✓ |  |
| `period` | INTEGER |  |  |
| `period_type` | TEXT |  |  |
| `time_in_period` | TEXT |  |  |
| `event_type` | TEXT |  |  |
| `team_id` | INTEGER |  |  |
| `shooting_player_id` | INTEGER |  |  |
| `goalie_id` | INTEGER |  |  |
| `x_coord` | REAL |  |  |
| `y_coord` | REAL |  |  |
| `zone_code` | TEXT |  |  |
| `shot_type` | TEXT |  |  |
| `reason` | TEXT |  |  |
| `blocking_player_id` | INTEGER |  |  |

**Foreign keys:**

- `game_id` → `games.id`

## `skater_game_stats`

236,101 rows.

| Column | Type | Primary Key | Not Null |
|---|---|---|---|
| `game_id` | INTEGER | ✓ |  |
| `player_id` | INTEGER | ✓ |  |
| `team_abbreviation` | TEXT |  |  |
| `name` | TEXT |  |  |
| `position` | TEXT |  |  |
| `goals` | INTEGER |  |  |
| `assists` | INTEGER |  |  |
| `points` | INTEGER |  |  |
| `plus_minus` | INTEGER |  |  |
| `pim` | INTEGER |  |  |
| `hits` | INTEGER |  |  |
| `power_play_goals` | INTEGER |  |  |
| `sog` | INTEGER |  |  |
| `faceoff_win_percentage` | INTEGER |  |  |
| `toi` | TEXT |  |  |
| `blocked_shots` | INTEGER |  |  |
| `shifts` | INTEGER |  |  |
| `giveaways` | INTEGER |  |  |
| `takeaways` | INTEGER |  |  |

**Foreign keys:**

- `player_id` → `players.id`
- `game_id` → `games.id`

## `teams`

32 rows.

| Column | Type | Primary Key | Not Null |
|---|---|---|---|
| `abbreviation` | TEXT | ✓ |  |
| `name` | TEXT |  | ✓ |
| `common_name` | TEXT |  |  |
| `logo_url` | TEXT |  |  |
| `place_name` | TEXT |  |  |
| `division` | TEXT |  |  |
| `division_abbrev` | TEXT |  |  |
| `conference` | TEXT |  |  |
| `conference_abbrev` | TEXT |  |  |
| `primary_color` | TEXT |  |  |
| `secondary_color` | TEXT |  |  |
| `team_id` | INTEGER |  |  |
