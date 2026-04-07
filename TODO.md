SQLite
Pydantic
API


## To do:
Pull in data for Avalanche only
Start with SQLite and then move to SQLAlchemy/SQLModel once it works

## Planned Structure
- `src/processing/` — Corsi, xG, game score, RAPM (coming later)
- `src/viz/` — game summaries, player cards, heat maps (coming later)
- `src/export/` — PNG/PDF output, social media formatting (coming later)


## Explanations
pyproject.toml is the standard way Python projects declare themselves.


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

write scripts/load_teams.py