SQLite
Pydantic
API


## To do:
X: Pull in data for Avalanche only
Start with SQLite and then move to SQLAlchemy/SQLModel once it works

All game events

## Planned Structure
- `src/processing/` — Corsi, xG, game score, RAPM (coming later)
- `src/viz/` — game summaries, player cards, heat maps (coming later)
- `src/export/` — PNG/PDF output, social media formatting (coming later)


## Explanations
pyproject.toml is the standard way Python projects declare themselves.

Current structure:
hockey-stats/
├── pyproject.toml
├── README.md
├── TODO.md
├── .gitignore
├── .env.example
│
│
├── config/
│   ├── settings.py
│   └── teams.py
│
├── data/
│
├── scripts/
│   ├── load_games.py
│   ├── load_teams.py
│   └── test_boxscore.py
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   └── schemas.py      
│   └── data/
│       ├── nhl_api.py
│       └── db.py


## Planned
- visualization
- Processing/analysis

Shot location data and standings over time


hockey-stats/
├── pyproject.toml
├── README.md
├── .gitignore
├── .env.example
│
├── config/
│   ├── settings.py
│   └── teams.py
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── data/
│       ├── __init__.py
│       ├── nhl_api.py
│       └── db.py

cache


Stats:

Corsi
Fenwick
PDO
Expected Goals
Goals saved above expected
Game Score
RAPM

Predictive:

###
Corsi % = CF /  (CF + CA)

CF = shots on goal for + missed shots for + blocked shots for

CA = shots on goal against + missed shots against + blocked shots against


Relative Corsi = (Player's on-ice Corsi) - (teams Corsi when off ice)

CA/60 = CA*60

###
Fenwick FF/ (FF + FA)

FF = shots on goal for + missed shots for
###
PDO = SH% + SV%

###
Gradient Boost for xGF

GSAx = xGA - GA

