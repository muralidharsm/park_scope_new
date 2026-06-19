"""
PARK-SCOPE configuration.

All tunable knobs live here so the logic stays transparent and auditable —
which matters because the end users are police officers who need to trust
why a zone is ranked the way it is. Every signal below is derived ONLY from
the two datasets HackerEarth provides (parking violations + Astram events);
no external data is read anywhere, keeping the submission rule-compliant.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

# Canonical raw input names. The loader also accepts the original
# hash-suffixed filenames as downloaded from HackerEarth via the patterns
# below, so you can drop the CSVs in /data exactly as you received them.
VIOLATION_CSV = DATA_DIR / "jan_to_may_police_violation_anonymized791b166.csv"
EVENT_CSV = DATA_DIR / "Astram_event_data_anonymized_-_Astram_event_data_anonymizedb40ac87.csv"
VIOLATION_PATTERN = "*police_violation*.csv"
EVENT_PATTERN = "*Astram*event*.csv"

# Generated artifacts (small, safe to commit / ship in the zip)
ZONES_CSV = ARTIFACT_DIR / "zones.csv"                 # one row per ~111m zone
HEAT_CSV = ARTIFACT_DIR / "heat_points.csv"            # lat/long/hour for the map
HOURLY_CSV = ARTIFACT_DIR / "zone_hour_weekday.csv"    # forecasting base table
META_JSON = ARTIFACT_DIR / "meta.json"                 # headline numbers + metrics

# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------
# The raw created_datetime is tagged "+00". We convert to India Standard Time
# for all hour-of-day / weekday analysis. If you later confirm the timestamps
# were already stored in local time, set ASSUME_TIMESTAMPS_ARE_UTC = False.
ASSUME_TIMESTAMPS_ARE_UTC = True
IST_OFFSET_HOURS = 5.5

# ---------------------------------------------------------------------------
# Spatial binning
# ---------------------------------------------------------------------------
# 3 decimal places of lat/long ~= 111m cells. This is the "zone" granularity.
GRID_DECIMALS = 3

# ---------------------------------------------------------------------------
# Congestion Impact Score weights (must sum to ~1.0; shown to judges as-is)
# ---------------------------------------------------------------------------
IMPACT_WEIGHTS = {
    "violation_density": 0.45,   # how many tickets the zone generates
    "event_congestion": 0.25,    # overlap with congestion-type events nearby
    "junction_criticality": 0.20,  # named BTP junction vs unmonitored stretch
    "strategic_proximity": 0.10,   # metro / market / hospital / school / bus stop
}

# Astram event_cause values we treat as genuine congestion-impact signals
CONGESTION_EVENT_CAUSES = {
    "congestion", "construction", "water_logging", "road_conditions",
    "pot_holes", "accident", "tree_fall", "procession", "public_event",
    "vip_movement", "protest", "debris", "vehicle_breakdown",
}

# Keywords found inside the provided junction_name labels that flag a
# strategically sensitive location. (Derived from BTP's own labels — not an
# external dataset.)
STRATEGIC_KEYWORDS = ["metro", "market", "hospital", "school", "bus", "station", "circle"]

# ---------------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------------
APP_NAME = "PARK-SCOPE"
APP_TAGLINE = "Parking Risk & Congestion Optimization for Smart Patrol Enforcement"


def resolve_csv(canonical: Path, pattern: str) -> Path:
    """Return the canonical path if it exists, else the first file in /data
    matching the glob pattern. Keeps the pipeline working whether the user
    renames the files or drops them in with the original hash suffix."""
    if canonical.exists():
        return canonical
    matches = sorted(DATA_DIR.glob(pattern))
    if matches:
        return matches[0]
    return canonical  # let pandas raise a clear FileNotFoundError
