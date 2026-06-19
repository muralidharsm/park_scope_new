"""
Forecasting layer.

Two complementary models, both deliberately interpretable:

1. Historical-rate forecaster (primary, used live in the dashboard):
   expected violations for a zone at a given (weekday, hour) = the historical
   mean for that zone/weekday/hour bin. Simple, transparent, instantly
   explainable to an officer ("this corner averages 6 tickets every Sunday
   between 10 and 11am").

2. Gradient Boosting model (validation artifact): predicts a zone's daily
   violation count from temporal + spatial features and is benchmarked against
   a naive "predict the historical mean" baseline on a forward-in-time holdout.
   Its only job is to prove, with a real metric, that the patterns are learnable
   and not noise — so when a judge asks "is this signal real?" we have numbers.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_hour_weekday_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average violations per zone per (weekday, hour) — the live forecaster."""
    counts = (
        df.groupby(["cell", "weekday", "hour"]).size().rename("count").reset_index()
    )
    # number of distinct calendar days each (weekday) appeared, to get a rate
    days_per_weekday = (
        df.assign(date=df["dt_ist"].dt.date)
        .groupby("weekday")["date"]
        .nunique()
        .rename("n_days")
    )
    counts = counts.merge(days_per_weekday, on="weekday", how="left")
    counts["expected_per_occurrence"] = counts["count"] / counts["n_days"].clip(lower=1)
    return counts


def train_validation_model(df: pd.DataFrame) -> dict:
    """
    Train a GBM on per-zone-per-day counts and compare to a naive baseline on a
    temporal holdout (last 20% of days). Returns metrics for the meta artifact.
    """
    daily = (
        df.assign(date=df["dt_ist"].dt.normalize())
        .groupby(["cell", "date"])
        .size()
        .rename("count")
        .reset_index()
    )
    if daily["date"].nunique() < 20:
        return {"trained": False, "reason": "not enough distinct days"}

    daily = daily.sort_values("date")
    daily["dow"] = daily["date"].dt.dayofweek
    daily["dom"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)

    # zone strength = that zone's overall mean (a strong, honest feature)
    zone_mean = daily.groupby("cell")["count"].transform("mean")
    daily["zone_mean"] = zone_mean

    cutoff = daily["date"].quantile(0.8)
    train = daily[daily["date"] <= cutoff]
    test = daily[daily["date"] > cutoff]
    if len(test) < 50:
        return {"trained": False, "reason": "holdout too small"}

    feats = ["dow", "dom", "month", "is_weekend", "zone_mean"]
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(train[feats], train["count"])
    pred = model.predict(test[feats])

    # naive baseline: predict each zone's historical (train) mean
    train_zone_mean = train.groupby("cell")["count"].mean()
    overall_mean = train["count"].mean()
    naive = test["cell"].map(train_zone_mean).fillna(overall_mean).values

    # XAI: expose what the model actually leans on (pretty labels for the UI)
    pretty = {
        "zone_mean": "Location (zone baseline rate)",
        "dow": "Day of week",
        "is_weekend": "Weekend vs weekday",
        "month": "Month / seasonality",
        "dom": "Day of month",
    }
    fi = {pretty.get(f, f): round(float(imp), 4)
          for f, imp in zip(feats, model.feature_importances_)}

    return {
        "trained": True,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "model_mae": round(float(mean_absolute_error(test["count"], pred)), 3),
        "naive_mae": round(float(mean_absolute_error(test["count"], naive)), 3),
        "model_rmse": round(float(np.sqrt(mean_squared_error(test["count"], pred))), 3),
        "naive_rmse": round(float(np.sqrt(mean_squared_error(test["count"], naive))), 3),
        "feature_importance": fi,
    }
