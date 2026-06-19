"""
Patrol Optimizer.

The "so what do I do Monday morning" layer. Two stages:

1. SELECTION (`allocate_patrols`) — weighted maximum coverage. Each unit covers
   one zone for the window; zones are ranked by impact-weighted violation load.

2. SEQUENCING (`sequence_stops`) — given the selected stops, output the optimal
   physical patrol ROUTE. We treat junctions as graph nodes V and drive times as
   edges E (estimated via haversine distance at an assumed patrol speed), then
   solve a Travelling-Salesperson route with a nearest-neighbour construction
   improved by 2-opt. This turns a list of disconnected hotspots into a single
   drivable beat that minimises travel weight while covering the priority zones.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def allocate_patrols(
    zones: pd.DataFrame,
    hour_table: pd.DataFrame,
    n_units: int,
    weekday: str | None = None,
    hour_start: int = 8,
    hour_end: int = 12,
) -> dict:
    """
    Parameters
    ----------
    zones : per-zone table containing 'cell', 'impact_score', 'violations',
            'label', 'latitude', 'longitude'.
    hour_table : zone x weekday x hour expected counts (from forecast layer).
    n_units : number of patrol units available.
    weekday : optional weekday name to focus on (e.g. 'Sunday'); None = all.
    hour_start, hour_end : inclusive-exclusive window of the shift.
    """
    ht = hour_table.copy()
    ht = ht[(ht["hour"] >= hour_start) & (ht["hour"] < hour_end)]
    if weekday:
        ht = ht[ht["weekday"] == weekday]

    window_load = (
        ht.groupby("cell")["count"].sum().rename("window_violations").reset_index()
    )
    z = zones.merge(window_load, on="cell", how="left")
    z["window_violations"] = z["window_violations"].fillna(0)

    # impact-weighted load in the window = expected window violations * impact
    z["window_impact_load"] = z["window_violations"] * z["impact_score"]

    total_load = z["window_impact_load"].sum()
    z = z.sort_values("window_impact_load", ascending=False).reset_index(drop=True)

    n_units = max(0, int(n_units))
    chosen = z.head(n_units).copy()
    covered = chosen["window_impact_load"].sum()
    coverage_pct = (covered / total_load * 100) if total_load > 0 else 0.0

    chosen["rank"] = range(1, len(chosen) + 1)
    plan = chosen[
        [
            "rank", "label", "latitude", "longitude",
            "window_violations", "impact_score", "window_impact_load",
        ]
    ].rename(
        columns={
            "label": "deploy_zone",
            "window_violations": "expected_violations_in_window",
        }
    )

    return {
        "plan": plan,
        "coverage_pct": round(float(coverage_pct), 1),
        "n_units": n_units,
        "n_zones_total": int(len(z)),
        "window": f"{hour_start:02d}:00-{hour_end:02d}:00",
        "weekday": weekday or "All days",
    }


# --------------------------------------------------------------------------- #
# Stage 2 — route sequencing (TSP heuristic over drive-time edges)
# --------------------------------------------------------------------------- #
def _haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance matrix (km) for arrays of coords."""
    R = 6371.0
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _route_minutes(order, M):
    return float(sum(M[order[i], order[i + 1]] for i in range(len(order) - 1)))


def _two_opt(order, M, max_pass=40):
    """Classic 2-opt local search to untangle the nearest-neighbour route."""
    best = order[:]
    improved = True
    passes = 0
    while improved and passes < max_pass:
        improved = False
        passes += 1
        for i in range(1, len(best) - 1):
            for k in range(i + 1, len(best)):
                if k - i == 1:
                    continue
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                if _route_minutes(cand, M) + 1e-9 < _route_minutes(best, M):
                    best = cand
                    improved = True
    return best


def sequence_stops(plan: pd.DataFrame, speed_kmh: float = 20.0) -> dict:
    """
    Order the selected patrol stops into a single optimised beat.

    plan : DataFrame with 'deploy_zone', 'latitude', 'longitude' and an impact
           column ('window_impact_load' or 'impact_score').
    speed_kmh : assumed average city patrol speed used to turn distance into time.

    Returns the ordered stops (with stop_no, leg_min, cumulative_min) plus the
    total travel time and distance for the whole route.
    """
    df = plan.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return {"route": df.assign(stop_no=[], leg_min=[], cumulative_min=[]),
                "total_min": 0.0, "total_km": 0.0, "speed_kmh": speed_kmh}

    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    Dkm = _haversine_km(lat, lon)
    Mmin = Dkm / max(speed_kmh, 1e-6) * 60.0  # edge weight = drive minutes

    # start from the single highest-impact stop (the anchor of the beat)
    impact_col = "window_impact_load" if "window_impact_load" in df.columns else "impact_score"
    start = int(df[impact_col].to_numpy().argmax())

    # nearest-neighbour construction
    unvisited = set(range(n))
    order = [start]
    unvisited.discard(start)
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda j: Mmin[last, j])
        order.append(nxt)
        unvisited.discard(nxt)

    if n >= 4:
        order = _two_opt(order, Mmin)

    legs = [0.0] + [Mmin[order[i - 1], order[i]] for i in range(1, len(order))]
    kms = [0.0] + [Dkm[order[i - 1], order[i]] for i in range(1, len(order))]
    route = df.iloc[order].reset_index(drop=True)
    route["stop_no"] = range(1, n + 1)
    route["leg_min"] = np.round(legs, 1)
    route["cumulative_min"] = np.round(np.cumsum(legs), 1)

    return {
        "route": route,
        "total_min": round(float(np.sum(legs)), 1),
        "total_km": round(float(np.sum(kms)), 1),
        "speed_kmh": speed_kmh,
        "order": order,
    }
