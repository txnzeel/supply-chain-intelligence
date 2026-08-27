"""Register product-level champion forecasters and their reproducibility lineage."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = PROJECT_ROOT / "data" / "processed" / "forecasts"
SELECTION_FILE = FORECAST_DIR / "model_selection.csv"
FORECAST_FILE = FORECAST_DIR / "demand_forecasts.csv"
DEMAND_FILE = PROJECT_ROOT / "data" / "processed" / "daily_demand.csv"
MODEL_CODE = PROJECT_ROOT / "python" / "forecasting" / "demand_forecaster.py"

REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"
EXPERIMENT_DIR = PROJECT_ROOT / "models" / "experiments"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_version(product_id: str, model: str, data_hash: str, code_hash: str) -> str:
    payload = f"{product_id}|{model}|{data_hash}|{code_hash}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> None:
    for path in (SELECTION_FILE, FORECAST_FILE, DEMAND_FILE, MODEL_CODE):
        if not path.is_file():
            raise FileNotFoundError(f"Required registry input not found: {path}")

    registered_at = datetime.now(timezone.utc)
    run_id = registered_at.strftime("forecast-%Y%m%dT%H%M%SZ")
    data_hash = sha256(DEMAND_FILE)
    code_hash = sha256(MODEL_CODE)
    forecast_hash = sha256(FORECAST_FILE)

    selections = pd.read_csv(SELECTION_FILE)
    forecasts = pd.read_csv(FORECAST_FILE)
    required = {
        "product_id", "selected_model", "validation_wape", "validation_mae",
        "validation_rmse", "validation_bias", "history_days", "demand_class",
    }
    missing = sorted(required - set(selections.columns))
    if missing:
        raise ValueError(f"Model selection schema is missing: {missing}")
    if selections["product_id"].duplicated().any():
        raise ValueError("Model registry requires one champion per product.")

    date_range = forecasts.groupby("product_id")["forecast_date"].agg(
        forecast_start="min", forecast_end="max", forecast_rows="size"
    )
    registry = selections.merge(date_range, on="product_id", how="left", validate="one_to_one")
    registry["model_version"] = [
        stable_version(pid, model, data_hash, code_hash)
        for pid, model in zip(registry["product_id"], registry["selected_model"])
    ]
    registry["registry_stage"] = "champion"
    registry["model_family"] = "statistical_time_series"
    registry["training_data_sha256"] = data_hash
    registry["training_code_sha256"] = code_hash
    registry["forecast_artifact_sha256"] = forecast_hash
    registry["registered_at_utc"] = registered_at.isoformat()
    registry["retrain_policy"] = "drift_or_scheduled"

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    registry_path = REGISTRY_DIR / "champion_models.csv"
    specs_path = REGISTRY_DIR / "champion_model_specs.jsonl"
    manifest_path = REGISTRY_DIR / "registry_manifest.json"
    forecast_artifact = ARTIFACT_DIR / "champion_forecasts.parquet"

    registry.to_csv(registry_path, index=False)
    with duckdb.connect() as connection:
        connection.register("forecast_artifact_source", forecasts)
        artifact_sql_path = str(forecast_artifact).replace("'", "''")
        connection.execute(
            f"COPY forecast_artifact_source TO '{artifact_sql_path}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    with specs_path.open("w", encoding="utf-8") as handle:
        for record in registry.replace({np.nan: None}).to_dict(orient="records"):
            handle.write(json.dumps(record, default=str) + "\n")

    model_counts = registry["selected_model"].value_counts().to_dict()
    manifest = {
        "registry_version": 1,
        "run_id": run_id,
        "registered_at_utc": registered_at.isoformat(),
        "champion_models": int(len(registry)),
        "model_distribution": model_counts,
        "mean_validation_wape": float(registry["validation_wape"].mean()),
        "training_data": str(DEMAND_FILE.relative_to(PROJECT_ROOT)),
        "training_data_sha256": data_hash,
        "training_code": str(MODEL_CODE.relative_to(PROJECT_ROOT)),
        "training_code_sha256": code_hash,
        "forecast_artifact": str(forecast_artifact.relative_to(PROJECT_ROOT)),
        "forecast_artifact_sha256": forecast_hash,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    experiment = manifest | {
        "python": sys.version,
        "platform": platform.platform(),
        "selection_input": str(SELECTION_FILE.relative_to(PROJECT_ROOT)),
        "status": "registered",
    }
    (EXPERIMENT_DIR / f"{run_id}.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )

    print(f"Registered champion models: {len(registry):,}")
    print(f"Registry: {registry_path}")
    print(f"Forecast artifact: {forecast_artifact}")
    print(f"Mean validation WAPE: {registry['validation_wape'].mean():.4f}")


if __name__ == "__main__":
    main()
