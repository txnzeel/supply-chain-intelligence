"""Monitor demand-data drift and rolling forecast-performance degradation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMAND_FILE = PROJECT_ROOT / "data" / "processed" / "daily_demand.csv"
METRICS_FILE = PROJECT_ROOT / "data" / "processed" / "forecasts" / "forecast_metrics.csv"
REGISTRY_FILE = PROJECT_ROOT / "models" / "registry" / "champion_models.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "modeling"

RECENT_DAYS = 60
MEAN_SHIFT_WARNING = 1.0
MEAN_SHIFT_CRITICAL = 2.0
PERFORMANCE_WARNING = 0.20
PERFORMANCE_CRITICAL = 0.50


def severity(score: pd.Series) -> pd.Series:
    return pd.cut(
        score,
        bins=[-np.inf, 0.75, 1.5, np.inf],
        labels=["Stable", "Warning", "Critical"],
        right=False,
    ).astype(str)


def main() -> None:
    for path in (DEMAND_FILE, METRICS_FILE, REGISTRY_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required monitoring input not found: {path}")

    demand = pd.read_csv(DEMAND_FILE, usecols=["product_id", "date", "actual_demand"])
    demand["date"] = pd.to_datetime(demand["date"], errors="raise")
    cutoff = demand["date"].max() - pd.Timedelta(days=RECENT_DAYS - 1)
    baseline = demand[demand["date"] < cutoff]
    recent = demand[demand["date"] >= cutoff]

    def profile(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        result = frame.groupby("product_id")["actual_demand"].agg(
            **{
                f"{prefix}_mean": "mean",
                f"{prefix}_std": "std",
                f"{prefix}_observations": "size",
            }
        )
        zero_rate = frame["actual_demand"].eq(0).groupby(frame["product_id"]).mean()
        result[f"{prefix}_zero_rate"] = zero_rate
        return result.reset_index()

    drift = profile(baseline, "baseline").merge(
        profile(recent, "recent"), on="product_id", how="inner", validate="one_to_one"
    )
    safe_std = drift["baseline_std"].replace(0, np.nan)
    drift["mean_shift_z"] = (
        (drift["recent_mean"] - drift["baseline_mean"]).abs() / safe_std
    ).fillna(0)
    drift["volatility_ratio"] = (
        drift["recent_std"] / safe_std
    ).replace([np.inf, -np.inf], np.nan).fillna(1)
    drift["volatility_shift"] = np.log(drift["volatility_ratio"].clip(lower=0.01)).abs()
    drift["zero_rate_shift"] = (
        drift["recent_zero_rate"] - drift["baseline_zero_rate"]
    ).abs()

    registry = pd.read_csv(REGISTRY_FILE)
    metrics = pd.read_csv(METRICS_FILE)
    selected = registry[["product_id", "selected_model", "validation_wape", "model_version"]]
    champion_metrics = metrics.merge(
        selected,
        left_on=["product_id", "model"],
        right_on=["product_id", "selected_model"],
        how="inner",
    )
    latest_window = champion_metrics.groupby("product_id")["validation_window"].transform("max")
    latest = champion_metrics[champion_metrics["validation_window"].eq(latest_window)][
        ["product_id", "wape"]
    ].rename(columns={"wape": "latest_window_wape"})
    prior = champion_metrics[~champion_metrics["validation_window"].eq(latest_window)].groupby(
        "product_id", as_index=False
    )["wape"].mean().rename(columns={"wape": "prior_window_wape"})
    performance = selected.merge(latest, on="product_id", how="left").merge(
        prior, on="product_id", how="left"
    )
    denominator = performance["prior_window_wape"].replace(0, np.nan)
    performance["performance_change_pct"] = (
        (performance["latest_window_wape"] - performance["prior_window_wape"])
        / denominator
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    report = drift.merge(performance, on="product_id", how="left", validate="one_to_one")
    report["data_drift_score"] = (
        0.55 * report["mean_shift_z"].clip(upper=3) / 2
        + 0.25 * report["volatility_shift"].clip(upper=2)
        + 0.20 * report["zero_rate_shift"].clip(upper=0.5) * 2
    )
    report["performance_drift_score"] = report["performance_change_pct"].clip(lower=0)
    report["monitoring_score"] = (
        0.65 * report["data_drift_score"]
        + 0.35 * report["performance_drift_score"].clip(upper=2)
    )
    report["drift_status"] = severity(report["monitoring_score"])
    report["retraining_recommended"] = report["drift_status"].isin(["Warning", "Critical"])
    report["monitoring_date_utc"] = datetime.now(timezone.utc).isoformat()
    report = report.sort_values("monitoring_score", ascending=False).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "model_monitoring.csv"
    alerts_path = OUTPUT_DIR / "model_alerts.csv"
    summary_path = OUTPUT_DIR / "model_monitoring_summary.json"
    report.round(6).to_csv(report_path, index=False)
    report[report["retraining_recommended"]].to_csv(alerts_path, index=False)

    counts = report["drift_status"].value_counts().to_dict()
    summary = {
        "monitored_at_utc": datetime.now(timezone.utc).isoformat(),
        "recent_window_days": RECENT_DAYS,
        "products_monitored": int(len(report)),
        "status_counts": {key: int(value) for key, value in counts.items()},
        "retraining_recommended": int(report["retraining_recommended"].sum()),
        "mean_monitoring_score": float(report["monitoring_score"].mean()),
        "mean_latest_wape": float(report["latest_window_wape"].mean()),
        "mean_performance_change_pct": float(report["performance_change_pct"].mean()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Monitoring report: {report_path}")


if __name__ == "__main__":
    main()
