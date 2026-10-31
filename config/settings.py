# config/settings.py
from pathlib import Path

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent

# Database
DB_PATH = PROJECT_ROOT / "data" / "hockey_stats.db"

# NHL API
NHL_API_BASE_URL = "https://api-web.nhle.com"

# Current Season (Format: "YYYYYYYY")
CURRENT_SEASON = "20252026"

# Last Season
LAST_SEASON = "20242025"

# Seasons the data loader pulls; add a season here when it starts
SEASONS = ["20212022", "20222023", "20232024", "20242025", "20252026","20262027"]

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'output'
# Active xG model: Phase 3+ code loads it with load_active_model()
# (src/models/deterministic/xg.py). Change this one line to switch models.
XG_MODEL_PATH = OUTPUT_DIR / "models" / "xg_lgbm_v1_live.pkl"