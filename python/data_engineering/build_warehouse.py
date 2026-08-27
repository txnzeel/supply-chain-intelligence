"""Build a DuckDB analytical warehouse from validated pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
WAREHOUSE = DATA / "warehouse" / "supply_chain.duckdb"
LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"


SOURCES = {
    "dim_product": DATA / "raw" / "products.csv",
    "fact_demand": DATA / "processed" / "daily_demand.csv",
    "fact_inventory": DATA / "processed" / "inventory_daily.csv",
    "fact_forecast": DATA / "processed" / "forecasts" / "demand_forecasts.csv",
    "mart_inventory_decision": DATA / "processed" / "inventory_optimization.csv",
    "mart_supply_chain_risk": DATA / "processed" / "risk" / "supply_chain_risk.csv",
    "mart_model_monitoring": DATA / "processed" / "modeling" / "model_monitoring.csv",
    "mart_monte_carlo_risk": DATA / "processed" / "modeling" / "monte_carlo_inventory_risk.csv",
    "mart_model_registry": PROJECT_ROOT / "models" / "registry" / "champion_models.csv",
}


def materialized(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        return handle.read(len(LFS_SIGNATURE)) != LFS_SIGNATURE


def quote(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> None:
    missing = [name for name, path in SOURCES.items() if not materialized(path)]
    required_missing = [name for name in ("dim_product", "fact_demand") if name in missing]
    if required_missing:
        raise FileNotFoundError(f"Required warehouse sources are missing: {required_missing}")

    WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(WAREHOUSE)) as connection:
        connection.execute("CREATE SCHEMA IF NOT EXISTS core")
        connection.execute("CREATE SCHEMA IF NOT EXISTS marts")
        connection.execute("CREATE SCHEMA IF NOT EXISTS metadata")

        loaded = []
        for name, path in SOURCES.items():
            if name in missing:
                continue
            schema = "marts" if name.startswith("mart_") else "core"
            connection.execute(
                f"CREATE OR REPLACE TABLE {schema}.{name} AS "
                f"SELECT * FROM read_csv_auto('{quote(path)}', header=true, sample_size=-1)"
            )
            loaded.append(f"{schema}.{name}")

        connection.execute(
            """
            CREATE OR REPLACE TABLE core.dim_date AS
            SELECT DISTINCT
                CAST(date AS DATE) AS date_key,
                EXTRACT(year FROM CAST(date AS DATE))::INTEGER AS year,
                EXTRACT(month FROM CAST(date AS DATE))::INTEGER AS month,
                EXTRACT(quarter FROM CAST(date AS DATE))::INTEGER AS quarter,
                EXTRACT(dow FROM CAST(date AS DATE))::INTEGER AS day_of_week,
                CASE WHEN EXTRACT(dow FROM CAST(date AS DATE)) IN (0, 6)
                     THEN TRUE ELSE FALSE END AS is_weekend
            FROM core.fact_demand
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE metadata.dataset_registry AS
            SELECT
                schema_name,
                table_name,
                estimated_size,
                column_count,
                NOW() AS loaded_at
            FROM duckdb_tables()
            WHERE schema_name IN ('core', 'marts')
            """
        )

    print(f"Warehouse created: {WAREHOUSE}")
    print("Loaded tables:")
    for table in loaded + ["core.dim_date", "metadata.dataset_registry"]:
        print(f"  - {table}")
    if missing:
        print(f"Skipped optional unmaterialized sources: {', '.join(missing)}")


if __name__ == "__main__":
    main()
