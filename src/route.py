"""
Patrol route sequencing.

The optimizer picks WHICH zones to cover. This module decides the ORDER to
visit them in, so a patrol vehicle drives the shortest sensible loop instead of
zig-zagging. Junctions are nodes; edges are travel cost.

Edge weight = straight-line (haversine) distance between zone centroids. This is
a transparent proxy — in production, swap `distance_matrix()` for a real
drive-time matrix from a routing API (one function, no other changes). The
sequencing itself is a real TSP heuristic: nearest-neighbour construction then
2-opt local improvement.
"""
from __future__ import annotations
from math import radians, sin, cos, asin, sqrt
import numpy as np
import pandas as pd

CITY_SPEED_KMH = 18.0   # average Bengaluru patrol speed proxy
DWELL_MIN_PER_STOP = 8  # minutes spent enforcing at each stop


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    n = len(lats)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            km = haversine_km(lats[i], lons[i], lats[j], lons[j])
            d[i, j] = d[j, i] = km
    return d


def _nearest_neighbour(d: np.ndarray, start: int = 0) -> list[int]:
    n = len(d)
    unvisited = set(range(n))
    order = [start]
    unvisited.discard(start)
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda j: d[last, j])
        order.append(nxt)
        unvisited.discard(nxt)
    return order


def _route_len(order: list[int], d: np.ndarray) -> float:
    return sum(d[order[i], order[i + 1]] for i in range(len(order) - 1))


def _two_opt(order: list[int], d: np.ndarray, max_pass: int = 40) -> list[int]:
    best = order[:]
    improved = True
    passes = 0
    while improved and passes < max_pass:
        improved = False
        passes += 1
        for i in range(1, len(best) - 1):
            for k in range(i + 1, len(best)):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                if _route_len(cand, d) + 1e-9 < _route_len(best, d):
                    best = cand
                    improved = True
    return best


def sequence_route(stops: pd.DataFrame) -> dict:
    """
    Parameters
    ----------
    stops : DataFrame with at least 'label', 'latitude', 'longitude'. The first
            row is treated as the start (highest-priority stop, since the
            optimizer hands stops back already ranked by impact load).

    Returns dict with the ordered stops, total distance, estimated patrol time,
    and the lat/lon path for drawing on the map.
    """
    s = stops.reset_index(drop=True).copy()
    n = len(s)
    if n == 0:
        return {"ordered": s.assign(stop_order=[]), "total_km": 0.0,
                "est_minutes": 0, "path_lat": [], "path_lon": []}
    if n == 1:
        s["stop_order"] = [1]
        return {"ordered": s, "total_km": 0.0, "est_minutes": DWELL_MIN_PER_STOP,
                "path_lat": [s.loc[0, "latitude"]], "path_lon": [s.loc[0, "longitude"]]}

    lats = s["latitude"].to_numpy()
    lons = s["longitude"].to_numpy()
    d = distance_matrix(lats, lons)
    order = _two_opt(_nearest_neighbour(d, start=0), d)

    total_km = round(float(_route_len(order, d)), 2)
    est_minutes = int(round(total_km / CITY_SPEED_KMH * 60 + n * DWELL_MIN_PER_STOP))

    ordered = s.iloc[order].copy()
    ordered["stop_order"] = range(1, n + 1)
    return {
        "ordered": ordered,
        "total_km": total_km,
        "est_minutes": est_minutes,
        "path_lat": ordered["latitude"].tolist(),
        "path_lon": ordered["longitude"].tolist(),
    }
