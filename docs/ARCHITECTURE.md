# Platform Architecture

This repository is evolving from a collection of analytical scripts into a local,
production-style supply-chain decision intelligence platform.

## Layers

1. **Source and ingestion** — synthetic product, demand, inventory, and supplier data.
2. **Quality and contracts** — schema, completeness, uniqueness, range, and LFS checks.
3. **Analytical warehouse** — DuckDB core facts/dimensions and business-facing marts.
4. **Data science** — forecast selection, residual anomaly detection, and monitoring.
5. **Decision science** — inventory policy, scenarios, economic validation, and risk.
6. **Consumption** — a Streamlit executive control tower backed by governed marts.

## Model lifecycle

Product-level statistical forecasters are governed as reproducible model
specifications. The registry records the champion algorithm, validation metrics,
training-data fingerprint, forecasting-code fingerprint, forecast artifact, and
retraining policy for every product. This avoids fragile cross-version pickle
files while preserving the information required to reproduce and audit a model.

Monitoring separates two kinds of degradation:

- **Data drift:** changes in demand level, volatility, or zero-demand behavior.
- **Performance drift:** deterioration in WAPE across rolling validation windows.

Monte Carlo simulation then uses forecast RMSE and uncertain lead times to test
whether optimized reorder points remain robust under 5,000 scenarios per product.

## Intended warehouse model

| Layer | Assets | Purpose |
|---|---|---|
| `core` | `dim_product`, `dim_date` | Conformed business dimensions |
| `core` | `fact_demand`, `fact_inventory`, `fact_forecast` | Granular analytical facts |
| `marts` | `mart_inventory_decision` | Recommended inventory policies |
| `marts` | `mart_supply_chain_risk` | Portfolio risk and action queue |
| `marts` | `mart_model_monitoring` | Drift status and retraining alerts |
| `marts` | `mart_monte_carlo_risk` | Probabilistic policy robustness |
| `marts` | `mart_model_registry` | Champion models and lineage |
| `metadata` | `dataset_registry` | Load observability and lineage support |

## Engineering principles

- Every pipeline step declares its inputs and outputs.
- Git LFS pointers are never treated as valid data.
- Each run produces timestamped JSON metadata and output fingerprints.
- Raw data remains immutable; transformations produce repeatable downstream assets.
- Models are evaluated before selection and their outputs feed explicit decisions.
- Dashboard pages consume decision marts rather than reproducing business logic.
