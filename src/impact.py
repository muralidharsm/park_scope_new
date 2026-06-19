"""
Congestion Impact Score.

This is PARK-SCOPE's core differentiator. Most parking analyses rank zones by
raw ticket count. That over-weights places that are merely easy to ticket and
under-weights places where a violation genuinely chokes traffic. We instead
score each zone on four transparent, min-max normalized components:

    impact = w1 * violation_density
           + w2 * event_congestion      (fused from the Astram event dataset)
           + w3 * junction_criticality
           + w4 * strategic_proximity   (metro / market / hospital / school...)

The weights live in config.py and are shown to judges verbatim. Nothing is a
black box: every zone's score can be decomposed back into these four drivers.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def event_congestion_per_zone(events: pd.DataFrame) -> pd.Series:
    """Count congestion-type events snapped to each violation grid cell."""
    ev = events.copy()
    ev["cause"] = ev["event_cause"].astype(str).str.lower().str.strip()
    ev = ev[ev["cause"].isin(config.CONGESTION_EVENT_CAUSES)]
    ev = ev.dropna(subset=["latitude", "longitude"])
    ev["cell"] = (
        ev["latitude"].round(config.GRID_DECIMALS).astype(str)
        + ","
        + ev["longitude"].round(config.GRID_DECIMALS).astype(str)
    )
    # weight road-closure / high-priority events more heavily
    w = pd.Series(1.0, index=ev.index)
    if "requires_road_closure" in ev.columns:
        w += ev["requires_road_closure"].astype(str).str.lower().eq("true") * 1.5
    if "priority" in ev.columns:
        w += ev["priority"].astype(str).str.lower().eq("high") * 0.5
    ev["w"] = w
    return ev.groupby("cell")["w"].sum().rename("event_congestion_raw")


def strategic_flag(label: str) -> int:
    s = str(label).lower()
    return int(any(k in s for k in config.STRATEGIC_KEYWORDS))


def compute_impact(zones: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """zones must contain: cell, violations, label, is_named_junction."""
    z = zones.copy()

    ev_zone = event_congestion_per_zone(events)
    z = z.merge(ev_zone, on="cell", how="left")
    z["event_congestion_raw"] = z["event_congestion_raw"].fillna(0)

    z["strategic_raw"] = z["label"].map(strategic_flag)
    # junction criticality: named BTP junction gets a base, scaled by its own load rank
    z["junction_raw"] = z["is_named_junction"].astype(float) * (
        0.5 + 0.5 * _minmax(z["violations"])
    )

    z["c_violation"] = _minmax(z["violations"])
    z["c_event"] = _minmax(z["event_congestion_raw"])
    z["c_junction"] = _minmax(z["junction_raw"])
    z["c_strategic"] = z["strategic_raw"].astype(float)  # already 0/1

    w = config.IMPACT_WEIGHTS
    z["impact_score"] = (
        w["violation_density"] * z["c_violation"]
        + w["event_congestion"] * z["c_event"]
        + w["junction_criticality"] * z["c_junction"]
        + w["strategic_proximity"] * z["c_strategic"]
    )
    # scale to 0-100 for readability
    z["impact_score"] = (z["impact_score"] * 100).round(2)
    return z
