"""Stress-test optimized reorder points under demand and lead-time uncertainty."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_SUMMARY = PROJECT_ROOT / "data" / "processed" / "forecasts" / "forecast_summary.csv"
OPTIMIZATION_FILE = PROJECT_ROOT / "data" / "processed" / "inventory_optimization.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "modeling"
SIMULATIONS = 5_000
RANDOM_SEED = 42

LEAD_TIME_RANGES = {
    "Stable": (5, 10),
    "Trending": (5, 12),
    "Seasonal": (7, 14),
    "Volatile": (10, 20),
    "Intermittent": (12, 25),
}


def product_seed(product_id: str) -> int:
    token = hashlib.sha256(f"{RANDOM_SEED}:{product_id}".encode()).hexdigest()[:8]
    return int(token, 16)


def simulate(row: pd.Series) -> dict[str, object]:
    rng = np.random.default_rng(product_seed(str(row["product_id"])))
    low, high = LEAD_TIME_RANGES.get(str(row["demand_class"]), (7, 14))
    lead_times = rng.integers(low, high + 1, size=SIMULATIONS)
    daily_mean = max(float(row["average_forecast_daily"]), 0.0)
    daily_sigma = max(float(row["validation_rmse"]), 0.01)
    means = daily_mean * lead_times
    sigmas = daily_sigma * np.sqrt(lead_times)
    lead_time_demand = np.maximum(rng.normal(means, sigmas), 0)
    reorder_point = max(float(row["optimized_reorder_point"]), 0.0)
    shortage = np.maximum(lead_time_demand - reorder_point, 0)
    stockout_probability = float(np.mean(shortage > 0))

    if stockout_probability <= 0.02:
        robustness = "Robust"
    elif stockout_probability <= 0.05:
        robustness = "Acceptable"
    elif stockout_probability <= 0.10:
        robustness = "Watch"
    else:
        robustness = "Fragile"

    return {
        "product_id": row["product_id"],
        "model_version": row.get("model_version", "unregistered"),
        "demand_class": row["demand_class"],
        "selected_model": row["selected_model"],
        "simulations": SIMULATIONS,
        "optimized_reorder_point": reorder_point,
        "mean_simulated_lead_time": float(lead_times.mean()),
        "mean_lead_time_demand": float(lead_time_demand.mean()),
        "p50_lead_time_demand": float(np.quantile(lead_time_demand, 0.50)),
        "p90_lead_time_demand": float(np.quantile(lead_time_demand, 0.90)),
        "p95_lead_time_demand": float(np.quantile(lead_time_demand, 0.95)),
        "p99_lead_time_demand": float(np.quantile(lead_time_demand, 0.99)),
        "stockout_probability": stockout_probability,
        "expected_shortage_units": float(shortage.mean()),
        "shortage_units_p95": float(np.quantile(shortage, 0.95)),
        "policy_robustness": robustness,
    }


def main() -> None:
    if not FORECAST_SUMMARY.is_file() or not OPTIMIZATION_FILE.is_file():
        raise FileNotFoundError("Forecast summary and inventory optimization outputs are required.")

    forecasts = pd.read_csv(FORECAST_SUMMARY)
    optimization = pd.read_csv(OPTIMIZATION_FILE)
    registry_path = PROJECT_ROOT / "models" / "registry" / "champion_models.csv"
    registry = pd.read_csv(registry_path, usecols=["product_id", "model_version"])
    inputs = forecasts.merge(
        optimization[["product_id", "optimized_reorder_point"]],
        on="product_id", how="inner", validate="one_to_one",
    ).merge(registry, on="product_id", how="left", validate="one_to_one")
    if len(inputs) != len(forecasts):
        raise ValueError("Monte Carlo inputs do not cover every forecasted product.")

    results = pd.DataFrame(simulate(row) for _, row in inputs.iterrows())
    results["simulated_at_utc"] = datetime.now(timezone.utc).isoformat()
    results = results.sort_values(
        ["stockout_probability", "expected_shortage_units"], ascending=False
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "monte_carlo_inventory_risk.csv"
    summary_path = OUTPUT_DIR / "monte_carlo_summary.json"
    results.round(6).to_csv(output_path, index=False)
    counts = results["policy_robustness"].value_counts().to_dict()
    summary = {
        "simulated_at_utc": datetime.now(timezone.utc).isoformat(),
        "products": int(len(results)),
        "simulations_per_product": SIMULATIONS,
        "total_scenarios": int(len(results) * SIMULATIONS),
        "policy_robustness": {key: int(value) for key, value in counts.items()},
        "mean_stockout_probability": float(results["stockout_probability"].mean()),
        "products_above_5pct_stockout_risk": int(results["stockout_probability"].gt(0.05).sum()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Monte Carlo report: {output_path}")


if __name__ == "__main__":
    main()
