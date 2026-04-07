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

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'output'