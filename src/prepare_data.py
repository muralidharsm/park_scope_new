"""
PARK-SCOPE data pipeline.

Reads the two raw HackerEarth CSVs from data/, cleans them, builds the zone
table with Congestion Impact Scores, the forecasting base table, a downsampled
heat-point table for the map, and a meta.json of headline numbers + model
metrics. All outputs are small and live in artifacts/ so the dashboard loads
instantly and the shipped zip stays well under the 50MB limit.

Run:  python -m src.prepare_data    (from the project root)
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from . import config
from .impact import compute_impact
from .forecast import build_hour_weekday_table, train_validation_model
from .optimizer import allocate_patrols  # noqa: F401  (re-exported for app)

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]


def _to_ist(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if config.ASSUME_TIMESTAMPS_ARE_UTC:
        dt = dt + pd.Timedelta(hours=config.IST_OFFSET_HOURS)
    return dt.dt.tz_localize(None)


def load_violations() -> pd.DataFrame:
    path = config.resolve_csv(config.VIOLATION_CSV, config.VIOLATION_PATTERN)
    v = pd.read_csv(path, low_memory=False)
    v = v.dropna(subset=["latitude", "longitude", "created_datetime"]).copy()
    v["dt_ist"] = _to_ist(v["created_datetime"])
    v = v.dropna(subset=["dt_ist"])
    v["hour"] = v["dt_ist"].dt.hour
    v["weekday"] = v["dt_ist"].dt.day_name()
    v["cell"] = (
        v["latitude"].round(config.GRID_DECIMALS).astype(str)
        + ","
        + v["longitude"].round(config.GRID_DECIMALS).astype(str)
    )
    v["junction_name"] = v["junction_name"].fillna("No Junction")
    return v


def build_zone_table(v: pd.DataFrame) -> pd.DataFrame:
    g = v.groupby("cell")
    zones = g.size().rename("violations").reset_index()
    zones["latitude"] = g["latitude"].mean().values
    zones["longitude"] = g["longitude"].mean().values

    # dominant named junction in the cell (ignore the generic 'No Junction')
    def dominant_junction(s: pd.Series) -> str:
        named = s[s != "No Junction"]
        if len(named):
            return named.value_counts().index[0]
        return "No Junction"

    jt = g["junction_name"].apply(dominant_junction).rename("junction_label")
    zones = zones.merge(jt, on="cell")
    pol = g["police_station"].apply(
        lambda s: s.dropna().value_counts().index[0] if s.notna().any() else "Unknown"
    ).rename("police_station")
    zones = zones.merge(pol, on="cell")

    zones["is_named_junction"] = (zones["junction_label"] != "No Junction").astype(int)
    zones["label"] = np.where(
        zones["is_named_junction"] == 1,
        zones["junction_label"],
        "Zone " + zones["latitude"].round(3).astype(str)
        + ", " + zones["longitude"].round(3).astype(str),
    )
    return zones


def enrich_zones(v: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """Add a per-zone violation-type mix and peak hour, for deep-dive tooltips."""
    import re
    g = v.groupby("cell")

    # violation_type is stored as a JSON-ish list, e.g. ["WRONG PARKING","NO PARKING"]
    def parse_types(s: str) -> list[str]:
        toks = re.findall(r'"([^"]+)"', str(s))
        return toks if toks else [str(s).strip("[]\" ").title()]

    long = (v[["cell", "violation_type"]]
            .assign(vt=lambda d: d["violation_type"].map(parse_types))
            .explode("vt"))

    def top_mix(s: pd.Series, k: int = 2) -> str:
        vc = s.value_counts(normalize=True).head(k)
        return " · ".join(f"{name.title()} {pct*100:.0f}%" for name, pct in vc.items())

    mix = long.groupby("cell")["vt"].apply(top_mix).rename("top_violations")
    peak = g["hour"].apply(lambda s: int(s.value_counts().index[0])).rename("peak_hour")
    zones = zones.merge(mix, on="cell", how="left").merge(peak, on="cell", how="left")
    zones["top_violations"] = zones["top_violations"].fillna("n/a")
    zones["peak_hour"] = zones["peak_hour"].fillna(0).astype(int)
    zones["peak_hour_label"] = zones["peak_hour"].map(lambda h: f"{h:02d}:00–{(h+1)%24:02d}:00 IST")
    return zones


def pareto(counts: pd.Series, pct: float) -> float:
    counts = counts.sort_values(ascending=False)
    k = max(1, int(len(counts) * pct))
    return round(counts.head(k).sum() / counts.sum() * 100, 1)


def main() -> None:
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading violations ...")
    v = load_violations()
    print(f"  {len(v):,} usable violation records")

    print("Loading events ...")
    ev_path = config.resolve_csv(config.EVENT_CSV, config.EVENT_PATTERN)
    events = pd.read_csv(ev_path, low_memory=False)

    print("Building zone table ...")
    zones = build_zone_table(v)
    print(f"  {len(zones):,} zones (~111m)")

    print("Computing Congestion Impact Scores (fusing event data) ...")
    zones = compute_impact(zones, events)
    zones = enrich_zones(v, zones)

    print("Building forecast base table ...")
    hour_table = build_hour_weekday_table(v)

    print("Validating learnability with GBM vs naive baseline ...")
    metrics = train_validation_model(v)
    print(f"  {metrics}")

    # ---- downsampled heat points for the map (cap so the artifact stays small)
    heat = v[["latitude", "longitude", "hour", "weekday"]].copy()
    if len(heat) > 60000:
        heat = heat.sample(60000, random_state=42)

    # ---- headline numbers
    cell_counts = v["cell"].value_counts()
    named = v[v["junction_name"] != "No Junction"]["junction_name"].value_counts()
    meta = {
        "total_violations": int(len(v)),
        "n_zones": int(len(zones)),
        "n_named_junctions": int(named.shape[0]),
        "n_police_stations": int(v["police_station"].nunique()),
        "date_min": str(v["dt_ist"].min().date()),
        "date_max": str(v["dt_ist"].max().date()),
        "pareto_top1pct_cells": pareto(cell_counts, 0.01),
        "pareto_top5pct_cells": pareto(cell_counts, 0.05),
        "pareto_top10pct_cells": pareto(cell_counts, 0.10),
        "pareto_top5pct_junctions": pareto(named, 0.05),
        "top_junctions": named.head(8).to_dict(),
        "peak_hours": v["hour"].value_counts().sort_values(ascending=False).head(3).index.tolist(),
        "model_metrics": metrics,
        "impact_weights": config.IMPACT_WEIGHTS,
        "timezone_assumption": "created_datetime treated as UTC, converted to IST (+5:30)"
        if config.ASSUME_TIMESTAMPS_ARE_UTC else "created_datetime treated as local IST",
    }

    print("Writing artifacts ...")
    zones.sort_values("impact_score", ascending=False).to_csv(config.ZONES_CSV, index=False)
    hour_table.to_csv(config.HOURLY_CSV, index=False)
    heat.to_csv(config.HEAT_CSV, index=False)
    with open(config.META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone. Headline numbers:")
    print(json.dumps({k: meta[k] for k in
                      ["total_violations", "n_zones", "pareto_top1pct_cells",
                       "pareto_top5pct_cells", "pareto_top10pct_cells",
                       "pareto_top5pct_junctions", "peak_hours"]}, indent=2))


if __name__ == "__main__":
    main()
