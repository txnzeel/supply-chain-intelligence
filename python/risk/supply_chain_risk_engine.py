from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SUPPLY CHAIN RISK INTELLIGENCE ENGINE
# ============================================================
#
# Architecture:
#
#   Product Economics
#          +
#   Inventory Risk
#          +
#   Demand Anomalies
#          +
#   Forecast Changes
#          +
#   Forecast Accuracy
#          +
#   Optimization
#          |
#          v
#   SIGNAL NORMALIZATION
#          |
#          v
#   COMPOUND RISK ENGINE
#          |
#          v
#   RISK SCORE
#          |
#          +----> FINANCIAL IMPACT
#          |
#          +----> CONFIDENCE
#          |
#          v
#   PRIORITY ENGINE
#          |
#          v
#   ACTION INTELLIGENCE
#
#
# Outputs:
#
#   data/processed/risk/supply_chain_risk.csv
#   data/processed/risk/risk_summary.csv
#   data/processed/risk/risk_actions.csv
#   data/processed/risk/risk_executive_summary.csv
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

PRODUCTS_PATH = DATA_DIR / "raw" / "products.csv"

INVENTORY_RISK_PATH = (
    DATA_DIR
    / "processed"
    / "inventory_risk.csv"
)

INVENTORY_OPTIMIZATION_PATH = (
    DATA_DIR
    / "processed"
    / "inventory_optimization.csv"
)

ANOMALIES_PATH = (
    DATA_DIR
    / "processed"
    / "anomalies"
    / "demand_anomalies.csv"
)

FORECAST_CHANGE_PATH = (
    DATA_DIR
    / "processed"
    / "anomalies"
    / "forecast_change_analysis.csv"
)

FORECAST_METRICS_PATH = (
    DATA_DIR
    / "processed"
    / "forecasts"
    / "forecast_metrics.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "processed"
    / "risk"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RISK THRESHOLDS
# ============================================================

RISK_THRESHOLDS = {
    "Critical": 75,
    "High": 50,
    "Moderate": 25,
}


# ============================================================
# PRIORITY THRESHOLDS
# ============================================================

PRIORITY_THRESHOLDS = {
    "P1_RISK": 75,
    "P2_RISK": 50,
    "P3_RISK": 25,

    "HIGH_AT_RISK_VALUE": 500_000,
    "MEDIUM_AT_RISK_VALUE": 100_000,
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_columns(df):
    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

    return df


def ensure_product_id(df):

    if "product_id" not in df.columns:

        raise ValueError(
            "Dataset does not contain required column: product_id"
        )

    df["product_id"] = (
        df["product_id"]
        .astype(str)
        .str.strip()
    )

    return df


def safe_numeric(df, columns):

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def first_existing_column(df, candidates):

    for column in candidates:

        if column in df.columns:
            return column

    return None


def safe_divide(
    numerator,
    denominator,
    default=0.0,
):

    result = np.divide(
        numerator,
        denominator,
        out=np.full(
            np.asarray(numerator).shape,
            default,
            dtype=float,
        ),
        where=np.asarray(denominator) != 0,
    )

    return result


def normalize_percent_series(series):

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = series.dropna()

    if len(valid) == 0:
        return series.fillna(0)

    median_abs = valid.abs().median()

    if median_abs <= 2:
        series = series * 100

    return series.fillna(0)


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

def validate_input_files():

    print_header("VALIDATING INPUT FILES")

    required_files = {
        "Products": PRODUCTS_PATH,
        "Inventory risk": INVENTORY_RISK_PATH,
        "Inventory optimization": INVENTORY_OPTIMIZATION_PATH,
        "Demand anomalies": ANOMALIES_PATH,
        "Forecast changes": FORECAST_CHANGE_PATH,
        "Forecast metrics": FORECAST_METRICS_PATH,
    }

    missing = []

    for name, path in required_files.items():

        if path.exists():

            print(
                f"[OK] {name}: {path}"
            )

        else:

            print(
                f"[MISSING] {name}: {path}"
            )

            missing.append(name)

    if missing:

        raise FileNotFoundError(
            "Required input files are missing: "
            + ", ".join(missing)
        )

    print()
    print("All required inputs are available.")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print_header("LOADING RISK INTELLIGENCE DATA")

    raw_datasets = {

        "products":
            pd.read_csv(PRODUCTS_PATH),

        "inventory_risk":
            pd.read_csv(INVENTORY_RISK_PATH),

        "inventory_optimization":
            pd.read_csv(
                INVENTORY_OPTIMIZATION_PATH
            ),

        "anomalies":
            pd.read_csv(ANOMALIES_PATH),

        "forecast_changes":
            pd.read_csv(FORECAST_CHANGE_PATH),

        "forecast_metrics":
            pd.read_csv(
                FORECAST_METRICS_PATH
            ),
    }

    datasets = {}

    for name, df in raw_datasets.items():

        df = normalize_columns(df)
        df = ensure_product_id(df)

        datasets[name] = df

        print(
            f"{name:<25}"
            f"rows={len(df):>10,} "
            f"products={df['product_id'].nunique():>6,}"
        )

    return datasets


# ============================================================
# PREPARE PRODUCTS
# ============================================================

def prepare_products(df):

    print_header("PREPARING PRODUCT ECONOMICS")

    df = normalize_columns(df)
    df = ensure_product_id(df)

    numeric_columns = [
        "selling_price",
        "unit_cost",
        "base_daily_demand",
        "gross_profit",
        "gross_margin_pct",
    ]

    df = safe_numeric(
        df,
        numeric_columns,
    )

    for column in numeric_columns:

        if column not in df.columns:

            df[column] = 0.0

    # --------------------------------------------------------
    # Derive gross profit if absent
    # --------------------------------------------------------

    if (
        df["gross_profit"].eq(0).all()
        and
        "selling_price" in df.columns
        and
        "unit_cost" in df.columns
    ):

        df["gross_profit"] = (
            df["selling_price"]
            - df["unit_cost"]
        )

    # --------------------------------------------------------
    # Derive margin if absent
    # --------------------------------------------------------

    if df["gross_margin_pct"].eq(0).all():

        df["gross_margin_pct"] = (
            safe_divide(
                df["gross_profit"],
                df["selling_price"],
            )
            * 100
        )

    # --------------------------------------------------------
    # Daily / annual profit
    # --------------------------------------------------------

    df["daily_gross_profit"] = (
        df["base_daily_demand"]
        * df["gross_profit"]
    )

    df["annual_gross_profit"] = (
        df["daily_gross_profit"]
        * 365
    )

    return df


# ============================================================
# PREPARE INVENTORY RISK
# ============================================================

def prepare_inventory_risk(df):

    print_header("PREPARING INVENTORY RISK SIGNALS")

    df = normalize_columns(df)
    df = ensure_product_id(df)

    numeric_columns = [

        "inventory_risk_score",
        "stockout_risk",
        "service_risk",
        "volatility_risk",
        "supplier_risk",
        "excess_inventory_risk",

        "stockout_rate",
        "lost_sales_rate",
        "fill_rate",

        "inventory_days",
        "excess_inventory",

        "average_supplier_delay",
        "supplier_delay_rate",

        "service_level_gap",
        "total_lost_sales",

        "average_daily_demand",
        "order_quantity",
    ]

    df = safe_numeric(
        df,
        numeric_columns,
    )

    required_columns = [

        "inventory_risk_score",
        "stockout_risk",
        "service_risk",
        "volatility_risk",
        "supplier_risk",
        "excess_inventory_risk",

        "total_lost_sales",
        "average_daily_demand",
        "inventory_days",
        "fill_rate",

        "supplier_delay_rate",
        "average_supplier_delay",
    ]

    for column in required_columns:

        if column not in df.columns:

            df[column] = 0.0

    aggregation = {

        "inventory_risk_score": "max",
        "stockout_risk": "max",
        "service_risk": "max",
        "volatility_risk": "max",
        "supplier_risk": "max",
        "excess_inventory_risk": "max",

        "total_lost_sales": "max",

        "average_daily_demand": "mean",
        "inventory_days": "mean",
        "fill_rate": "mean",

        "supplier_delay_rate": "mean",
        "average_supplier_delay": "mean",
    }

    if "risk_class" in df.columns:

        aggregation["risk_class"] = (
            lambda x:
            x.dropna().iloc[0]
            if not x.dropna().empty
            else "Unknown"
        )

    if "primary_risk_driver" in df.columns:

        aggregation["primary_risk_driver"] = (
            lambda x:
            x.dropna().iloc[0]
            if not x.dropna().empty
            else "Unknown"
        )

    grouped = (
        df.groupby("product_id")
        .agg(aggregation)
        .reset_index()
    )

    print(
        f"Inventory risk products: "
        f"{len(grouped):,}"
    )

    return grouped


# ============================================================
# PREPARE OPTIMIZATION
# ============================================================

def prepare_optimization(df):

    print_header("PREPARING OPTIMIZATION SIGNALS")

    df = normalize_columns(df)
    df = ensure_product_id(df)

    numeric_columns = [

        "baseline_total_cost",
        "optimized_total_cost",
        "cost_savings",

        "lost_sales_reduction",
        "baseline_lost_sales",
        "optimized_lost_sales",

        "baseline_fill_rate",
        "optimized_fill_rate",

        "baseline_stockout_rate",
        "optimized_stockout_rate",

        "baseline_inventory_days",
        "optimized_inventory_days",

        "safety_stock_change",
        "inventory_days_change",
    ]

    df = safe_numeric(
        df,
        numeric_columns,
    )

    for column in numeric_columns:

        if column not in df.columns:

            df[column] = 0.0

    grouped = (
        df.groupby("product_id")
        .agg(
            baseline_total_cost=(
                "baseline_total_cost",
                "mean",
            ),

            optimized_total_cost=(
                "optimized_total_cost",
                "mean",
            ),

            cost_savings=(
                "cost_savings",
                "mean",
            ),

            lost_sales_reduction=(
                "lost_sales_reduction",
                "mean",
            ),

            baseline_lost_sales=(
                "baseline_lost_sales",
                "mean",
            ),

            optimized_lost_sales=(
                "optimized_lost_sales",
                "mean",
            ),

            baseline_fill_rate=(
                "baseline_fill_rate",
                "mean",
            ),

            optimized_fill_rate=(
                "optimized_fill_rate",
                "mean",
            ),

            baseline_stockout_rate=(
                "baseline_stockout_rate",
                "mean",
            ),

            optimized_stockout_rate=(
                "optimized_stockout_rate",
                "mean",
            ),

            baseline_inventory_days=(
                "baseline_inventory_days",
                "mean",
            ),

            optimized_inventory_days=(
                "optimized_inventory_days",
                "mean",
            ),

            safety_stock_change=(
                "safety_stock_change",
                "mean",
            ),

            inventory_days_change=(
                "inventory_days_change",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["optimization_opportunity"] = (

        grouped["cost_savings"]
        .clip(lower=0)

        +

        grouped["lost_sales_reduction"]
        .clip(lower=0)
    )

    grouped["optimization_opportunity"] = (
        grouped["optimization_opportunity"]
        .fillna(0)
    )

    print(
        "Total positive optimization opportunity: "
        f"{grouped['optimization_opportunity'].sum():,.2f}"
    )

    return grouped


# ============================================================
# PREPARE ANOMALIES
# ============================================================

def prepare_anomalies(df):

    print_header("PREPARING DEMAND ANOMALY SIGNALS")

    df = normalize_columns(df)
    df = ensure_product_id(df)

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity_map = {

        "none": 0,
        "normal": 0,

        "moderate": 1,
        "medium": 1,

        "high": 2,

        "critical": 3,
    }

    if "anomaly_severity" in df.columns:

        severity_text = (
            df["anomaly_severity"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["anomaly_severity_score"] = (
            severity_text
            .map(severity_map)
            .fillna(0)
        )

    else:

        df["anomaly_severity_score"] = 0

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    numeric_columns = [

        "current_deviation_pct",
        "current_z_score",
        "current_residual",

        "actual",
        "current_baseline",
    ]

    df = safe_numeric(
        df,
        numeric_columns,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The old code attempted:
    #
    # .agg({
    #     "current_deviation_pct":
    #         "max_anomaly_deviation"
    # })
    #
    # which is invalid pandas syntax.
    #
    # We explicitly create source fields first and then
    # aggregate them.
    # --------------------------------------------------------

    if "current_deviation_pct" in df.columns:

        df["anomaly_deviation_value"] = (
            df["current_deviation_pct"]
            .abs()
        )

    else:

        df["anomaly_deviation_value"] = 0.0

    if "current_z_score" in df.columns:

        df["anomaly_z_value"] = (
            df["current_z_score"]
            .abs()
        )

    else:

        df["anomaly_z_value"] = 0.0

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    grouped = (
        df.groupby("product_id")
        .agg(

            anomaly_severity_score=(
                "anomaly_severity_score",
                "max",
            ),

            max_anomaly_deviation=(
                "anomaly_deviation_value",
                "max",
            ),

            max_anomaly_z=(
                "anomaly_z_value",
                "max",
            ),

            anomaly_event_count=(
                "product_id",
                "size",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Direction counts
    # --------------------------------------------------------

    if "anomaly_direction" in df.columns:

        direction_counts = (
            df.groupby("product_id")[
                "anomaly_direction"
            ]
            .value_counts()
            .unstack(
                fill_value=0
            )
        )

        direction_counts.columns = [

            "anomaly_count_"
            + str(column)
            .strip()
            .lower()
            .replace(" ", "_")

            for column
            in direction_counts.columns
        ]

        direction_counts = (
            direction_counts
            .reset_index()
        )

        grouped = grouped.merge(
            direction_counts,
            on="product_id",
            how="left",
        )

    # --------------------------------------------------------
    # Demand anomaly flag
    # --------------------------------------------------------

    grouped["demand_anomaly_flag"] = (
        grouped["anomaly_severity_score"]
        > 0
    ).astype(int)

    print(
        "Products with anomaly signals: "
        f"{grouped['demand_anomaly_flag'].sum():,}"
    )

    print(
        "Maximum anomaly severity: "
        f"{grouped['anomaly_severity_score'].max():.0f}"
    )

    return grouped


# ============================================================
# PREPARE FORECAST CHANGES
# ============================================================

def prepare_forecast_changes(df):

    print_header("PREPARING FORECAST CHANGE SIGNALS")

    df = normalize_columns(df)
    df = ensure_product_id(df)

    candidates = [

        "forecast_change_pct",
        "forecast_change",

        "change_pct",

        "forecast_change_percentage",
    ]

    change_column = first_existing_column(
        df,
        candidates,
    )

    if change_column is None:

        df["forecast_change_pct"] = 0.0

    else:

        df["forecast_change_pct"] = (
            normalize_percent_series(
                df[change_column]
            )
        )

    grouped = (
        df.groupby("product_id")
        .agg(

            forecast_change_pct=(
                "forecast_change_pct",
                "mean",
            ),

            max_forecast_increase_pct=(
                "forecast_change_pct",
                lambda x:
                x[x > 0].max()
                if (x > 0).any()
                else 0,
            ),

            max_forecast_decrease_pct=(
                "forecast_change_pct",
                lambda x:
                x[x < 0].min()
                if (x < 0).any()
                else 0,
            ),

            forecast_change_volatility=(
                "forecast_change_pct",
                "std",
            ),
        )
        .reset_index()
    )

    grouped[
        "forecast_change_volatility"
    ] = (
        grouped[
            "forecast_change_volatility"
        ]
        .fillna(0)
    )

    grouped["forecast_increase_flag"] = (
        grouped[
            "max_forecast_increase_pct"
        ]
        > 10
    ).astype(int)

    grouped["forecast_decrease_flag"] = (
        grouped[
            "max_forecast_decrease_pct"
        ]
        < -10
    ).astype(int)

    print(
        "Products with significant forecast increase: "
        f"{grouped['forecast_increase_flag'].sum():,}"
    )

    print(
        "Products with significant forecast decrease: "
        f"{grouped['forecast_decrease_flag'].sum():,}"
    )

    return grouped


# ============================================================
# PREPARE FORECAST METRICS
# ============================================================

def prepare_forecast_metrics(df):

    print_header("PREPARING FORECAST ACCURACY SIGNALS")

    df = normalize_columns(df)
    df = ensure_product_id(df)

    wape_column = first_existing_column(
        df,
        [
            "wape",
            "mean_wape",
            "validation_wape",
            "average_wape",
        ],
    )

    if wape_column is None:

        df["wape"] = np.nan

    else:

        df["wape"] = (
            normalize_percent_series(
                df[wape_column]
            )
        )

    grouped = (
        df.groupby("product_id")
        .agg(

            wape=(
                "wape",
                "mean",
            ),

            max_wape=(
                "wape",
                "max",
            ),

            forecast_observations=(
                "wape",
                "count",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Forecast accuracy risk
    # --------------------------------------------------------

    def accuracy_score(wape):

        if pd.isna(wape):
            return 0

        if wape <= 10:
            return 0

        if wape <= 20:
            return 20

        if wape <= 30:
            return 40

        if wape <= 50:
            return 70

        return 100

    grouped["forecast_accuracy_risk"] = (
        grouped["wape"]
        .apply(accuracy_score)
    )

    print(
        f"Mean WAPE: "
        f"{grouped['wape'].mean():.2f}%"
    )

    print(
        f"Median WAPE: "
        f"{grouped['wape'].median():.2f}%"
    )

    return grouped


# ============================================================
# BUILD MASTER DATASET
# ============================================================

def build_master_dataset(
    products,
    inventory_risk,
    optimization,
    anomalies,
    forecast_changes,
    forecast_metrics,
):

    print_header("BUILDING UNIFIED RISK DATASET")

    master = products.copy()

    # --------------------------------------------------------
    # Merge helper
    # --------------------------------------------------------

    def merge_dataset(
        base,
        addition,
        label,
    ):

        result = base.merge(
            addition,
            on="product_id",
            how="left",
            suffixes=(
                "",
                f"_{label}",
            ),
        )

        print(
            f"After {label} merge: "
            f"{len(result):,} rows"
        )

        return result

    master = merge_dataset(
        master,
        inventory_risk,
        "inventory",
    )

    master = merge_dataset(
        master,
        optimization,
        "optimization",
    )

    master = merge_dataset(
        master,
        anomalies,
        "anomaly",
    )

    master = merge_dataset(
        master,
        forecast_changes,
        "forecast_change",
    )

    master = merge_dataset(
        master,
        forecast_metrics,
        "forecast",
    )

    # --------------------------------------------------------
    # Duplicate protection
    # --------------------------------------------------------

    if master["product_id"].duplicated().any():

        print(
            "Duplicate product records detected "
            "after merge. Consolidating."
        )

        master = (
            master
            .groupby("product_id")
            .first()
            .reset_index()
        )

    print(
        f"Unified products: "
        f"{master['product_id'].nunique():,}"
    )

    return master


# ============================================================
# NORMALIZE SIGNALS
# ============================================================

def normalize_risk_components(df):

    print_header("NORMALIZING RISK COMPONENTS")

    df = df.copy()

    numeric_columns = [

        "inventory_risk_score",
        "stockout_risk",
        "service_risk",
        "volatility_risk",
        "supplier_risk",
        "excess_inventory_risk",

        "forecast_accuracy_risk",

        "anomaly_severity_score",

        "max_anomaly_deviation",
        "max_anomaly_z",

        "forecast_change_pct",
    ]

    df = safe_numeric(
        df,
        numeric_columns,
    )

    for column in numeric_columns:

        if column not in df.columns:

            df[column] = 0.0

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    df["inventory_component"] = (
        df["inventory_risk_score"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Stockout
    # --------------------------------------------------------

    df["stockout_component"] = (
        df["stockout_risk"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Service
    # --------------------------------------------------------

    df["service_component"] = (
        df["service_risk"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Supplier
    # --------------------------------------------------------

    df["supplier_component"] = (
        df["supplier_risk"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["volatility_component"] = (
        df["volatility_risk"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Excess inventory
    # --------------------------------------------------------

    df["excess_inventory_component"] = (
        df["excess_inventory_risk"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Forecast accuracy
    # --------------------------------------------------------

    df["forecast_accuracy_component"] = (
        df["forecast_accuracy_risk"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Demand anomaly
    # --------------------------------------------------------

    severity_component = (
        df["anomaly_severity_score"]
        .map(
            {
                0: 0,
                1: 35,
                2: 70,
                3: 100,
            }
        )
        .fillna(0)
    )

    deviation_component = (
        df["max_anomaly_deviation"]
        .abs()
        .clip(
            0,
            100,
        )
    )

    z_component = (
        (
            df["max_anomaly_z"]
            .abs()
            / 4
        )
        .clip(
            0,
            1,
        )
        * 100
    )

    df["anomaly_component"] = (

        severity_component * 0.50

        +

        deviation_component * 0.30

        +

        z_component * 0.20
    ).clip(0, 100)

    # --------------------------------------------------------
    # Forecast change
    # --------------------------------------------------------

    increase = (
        df["forecast_change_pct"]
        .clip(lower=0)
        .abs()
    )

    decrease = (
        df["forecast_change_pct"]
        .clip(upper=0)
        .abs()
    )

    df["forecast_increase_component"] = (
        increase
        .clip(
            0,
            100,
        )
    )

    df["forecast_decrease_component"] = (
        decrease
        .clip(
            0,
            100,
        )
    )

    # --------------------------------------------------------
    # Overall forecast-change component
    # --------------------------------------------------------

    df["forecast_change_component"] = (

        df["forecast_increase_component"]
        * 0.80

        +

        df["forecast_decrease_component"]
        * 0.20
    ).clip(0, 100)

    return df


# ============================================================
# COMPOUND RISK ENGINE
# ============================================================

def create_interaction_signals(df):

    print_header("RUNNING COMPOUND RISK ENGINE")

    df = df.copy()

    # ========================================================
    # SIGNAL DEFINITIONS
    # ========================================================

    # --------------------------------------------------------
    # 1. Demand spike + low inventory
    # --------------------------------------------------------

    df["demand_supply_mismatch"] = (

        (df["anomaly_severity_score"] >= 2)

        &

        (df["inventory_days"] < 14)

    ).astype(int)

    # --------------------------------------------------------
    # 2. Demand spike + supplier risk
    # --------------------------------------------------------

    df["demand_supplier_collision"] = (

        (df["anomaly_severity_score"] >= 2)

        &

        (df["supplier_risk"] >= 50)

    ).astype(int)

    # --------------------------------------------------------
    # 3. Forecast increase + low inventory
    # --------------------------------------------------------

    df["forecast_inventory_gap"] = (

        (df["forecast_change_pct"] > 20)

        &

        (df["inventory_days"] < 30)

    ).astype(int)

    # --------------------------------------------------------
    # 4. Forecast uncertainty + supplier risk
    # --------------------------------------------------------

    df["uncertainty_supplier_exposure"] = (

        (df["forecast_accuracy_risk"] >= 40)

        &

        (df["supplier_risk"] >= 50)

    ).astype(int)

    # --------------------------------------------------------
    # 5. Low inventory + poor service
    # --------------------------------------------------------

    df["inventory_service_collision"] = (

        (df["inventory_days"] < 14)

        &

        (df["service_risk"] >= 50)

    ).astype(int)

    # --------------------------------------------------------
    # 6. Supplier risk + low inventory
    # --------------------------------------------------------

    df["supplier_inventory_collision"] = (

        (df["supplier_risk"] >= 50)

        &

        (df["inventory_days"] < 21)

    ).astype(int)

    # --------------------------------------------------------
    # 7. Forecast error + volatility
    # --------------------------------------------------------

    df["forecast_uncertainty_collision"] = (

        (df["forecast_accuracy_risk"] >= 40)

        &

        (df["volatility_risk"] >= 50)

    ).astype(int)

    # --------------------------------------------------------
    # 8. Excess inventory + declining forecast
    # --------------------------------------------------------

    df["obsolescence_exposure"] = (

        (df["excess_inventory_risk"] >= 50)

        &

        (df["forecast_change_pct"] < -10)

    ).astype(int)

    interaction_columns = [

        "demand_supply_mismatch",
        "demand_supplier_collision",
        "forecast_inventory_gap",

        "uncertainty_supplier_exposure",

        "inventory_service_collision",
        "supplier_inventory_collision",

        "forecast_uncertainty_collision",

        "obsolescence_exposure",
    ]

    df["compound_risk_count"] = (
        df[interaction_columns]
        .sum(axis=1)
    )

    df["compound_risk_flag"] = (
        df["compound_risk_count"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Weighted compound score
    # --------------------------------------------------------

    interaction_weights = {

        "demand_supply_mismatch": 18,

        "demand_supplier_collision": 20,

        "forecast_inventory_gap": 15,

        "uncertainty_supplier_exposure": 10,

        "inventory_service_collision": 15,

        "supplier_inventory_collision": 12,

        "forecast_uncertainty_collision": 10,

        "obsolescence_exposure": 10,
    }

    df["compound_risk_score"] = 0.0

    for column, weight in interaction_weights.items():

        df["compound_risk_score"] += (
            df[column]
            * weight
        )

    df["compound_risk_score"] = (
        df["compound_risk_score"]
        .clip(
            0,
            35,
        )
    )

    print(
        "Products with compound risk: "
        f"{df['compound_risk_flag'].sum():,}"
    )

    print(
        "Maximum compound risk score: "
        f"{df['compound_risk_score'].max():.2f}"
    )

    return df


# ============================================================
# CALCULATE BASE RISK SCORE
# ============================================================

def calculate_risk_score(df):

    print_header("CALCULATING UNIFIED RISK SCORE")

    df = df.copy()

    # ========================================================
    # Base architecture
    #
    # Inventory / service     25%
    # Demand anomaly          20%
    # Supplier                15%
    # Volatility              10%
    # Forecast accuracy       10%
    # Forecast change         10%
    # Excess inventory          5%
    # Stockout                 5%
    #
    # Total                   100%
    # ========================================================

    df["risk_score_base"] = (

        df["inventory_component"]
        * 0.25

        +

        df["anomaly_component"]
        * 0.20

        +

        df["supplier_component"]
        * 0.15

        +

        df["volatility_component"]
        * 0.10

        +

        df["forecast_accuracy_component"]
        * 0.10

        +

        df["forecast_change_component"]
        * 0.10

        +

        df["excess_inventory_component"]
        * 0.05

        +

        df["stockout_component"]
        * 0.05
    )

    # --------------------------------------------------------
    # Compound adjustment
    # --------------------------------------------------------

    df["risk_score_pre_compound"] = (
        df["risk_score_base"]
    )

    df["risk_score"] = (

        df["risk_score_base"]

        +

        df["compound_risk_score"]

    ).clip(
        0,
        100,
    )

    df["risk_score"] = (
        df["risk_score"]
        .round(2)
    )

    # --------------------------------------------------------
    # Risk class
    # --------------------------------------------------------

    df["risk_class"] = np.select(

        [
            df["risk_score"]
            >= RISK_THRESHOLDS["Critical"],

            df["risk_score"]
            >= RISK_THRESHOLDS["High"],

            df["risk_score"]
            >= RISK_THRESHOLDS["Moderate"],
        ],

        [
            "Critical",
            "High",
            "Moderate",
        ],

        default="Low",
    )

    return df


# ============================================================
# PRIMARY RISK DRIVER
# ============================================================

def identify_primary_driver(df):

    print_header("IDENTIFYING PRIMARY RISK DRIVERS")

    df = df.copy()

    component_columns = {

        "Inventory":
            "inventory_component",

        "Demand Anomaly":
            "anomaly_component",

        "Supplier":
            "supplier_component",

        "Demand Volatility":
            "volatility_component",

        "Forecast Accuracy":
            "forecast_accuracy_component",

        "Forecast Change":
            "forecast_change_component",

        "Excess Inventory":
            "excess_inventory_component",

        "Stockout":
            "stockout_component",
    }

    component_frame = pd.DataFrame(

        {
            name: df[column]

            for name, column
            in component_columns.items()
        },

        index=df.index,
    )

    df["primary_risk_driver"] = (
        component_frame
        .idxmax(axis=1)
    )

    df["primary_risk_driver_score"] = (
        component_frame
        .max(axis=1)
        .round(2)
    )

    # --------------------------------------------------------
    # Second driver
    # --------------------------------------------------------

    sorted_components = np.sort(
        component_frame.values,
        axis=1,
    )

    df["secondary_risk_driver_score"] = (
        sorted_components[:, -2]
        if component_frame.shape[1] >= 2
        else 0
    )

    return df


# ============================================================
# FINANCIAL EXPOSURE
# ============================================================

def calculate_financial_exposure(df):

    print_header("CALCULATING FINANCIAL EXPOSURE")

    df = df.copy()

    # --------------------------------------------------------
    # Normalize required fields
    # --------------------------------------------------------

    numeric_columns = [

        "total_lost_sales",
        "gross_profit",

        "base_daily_demand",

        "average_supplier_delay",

        "forecast_change_pct",
    ]

    df = safe_numeric(
        df,
        numeric_columns,
    )

    for column in numeric_columns:

        if column not in df.columns:

            df[column] = 0.0

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    # --------------------------------------------------------
    # Lost sales
    # --------------------------------------------------------

    df["lost_sales_exposure"] = (

        df["total_lost_sales"]
        *
        df["gross_profit"]
    ).clip(lower=0)

    # --------------------------------------------------------
    # Daily profit
    # --------------------------------------------------------

    df["daily_profit_exposure"] = (

        df["base_daily_demand"]
        *
        df["gross_profit"]
    ).clip(lower=0)

    # --------------------------------------------------------
    # 30-day profit exposure
    # --------------------------------------------------------

    df["thirty_day_profit_exposure"] = (

        df["daily_profit_exposure"]
        * 30
    )

    # --------------------------------------------------------
    # Supplier disruption exposure
    # --------------------------------------------------------

    df["supplier_delay_exposure"] = (

        df["average_supplier_delay"]
        *
        df["daily_profit_exposure"]
    ).clip(lower=0)

    # --------------------------------------------------------
    # Forecast increase exposure
    # --------------------------------------------------------

    increase_pct = (

        df["forecast_change_pct"]
        .clip(lower=0)
        / 100
    )

    df["forecast_increase_exposure"] = (

        df["thirty_day_profit_exposure"]
        *
        increase_pct
    ).clip(lower=0)

    # --------------------------------------------------------
    # Excess inventory / obsolescence exposure
    # --------------------------------------------------------

    if "inventory_days" not in df.columns:

        df["inventory_days"] = 0.0

    if "excess_inventory" not in df.columns:

        df["excess_inventory"] = 0.0

    excess_days = (
        df["inventory_days"]
        .clip(lower=0)
        - 30
    ).clip(lower=0)

    df["inventory_obsolescence_exposure"] = (

        excess_days
        *
        df["base_daily_demand"]
        *
        df["gross_profit"]
        *
        (
            df["excess_inventory_component"]
            / 100
        )
    ).clip(lower=0)

    # --------------------------------------------------------
    # Total exposure
    # --------------------------------------------------------

    df["estimated_financial_exposure"] = (

        df["lost_sales_exposure"]

        +

        df["supplier_delay_exposure"]

        +

        df["forecast_increase_exposure"]

        +

        df["inventory_obsolescence_exposure"]

    ).clip(lower=0)

    # --------------------------------------------------------
    # Risk-weighted exposure
    #
    # We do NOT use risk_score / 100 directly.
    # That would make even low-risk products contribute heavily.
    #
    # Instead, apply a nonlinear risk probability curve.
    # --------------------------------------------------------

    risk_probability = (

        (
            df["risk_score"]
            / 100
        )
        ** 1.5

    ).clip(
        0,
        1,
    )

    df["risk_probability"] = (
        risk_probability
        .round(4)
    )

    df["at_risk_value"] = (

        df["estimated_financial_exposure"]
        *
        df["risk_probability"]

    ).clip(lower=0)

    # --------------------------------------------------------
    # Service exposure
    # --------------------------------------------------------

    if "fill_rate" not in df.columns:

        df["fill_rate"] = 1.0

    df["fill_rate"] = (
        pd.to_numeric(
            df["fill_rate"],
            errors="coerce",
        )
        .fillna(1)
        .clip(0, 1)
    )

    service_gap = (
        1
        -
        df["fill_rate"]
    ).clip(
        0,
        1,
    )

    df["service_risk_value"] = (

        df["base_daily_demand"]
        *
        30
        *
        service_gap
        *
        df["gross_profit"]

    ).clip(lower=0)

    return df


# ============================================================
# BUSINESS IMPACT
# ============================================================

def calculate_business_impact(df):

    print_header("CALCULATING BUSINESS IMPACT")

    df = df.copy()

    # --------------------------------------------------------
    # Product value percentile
    # --------------------------------------------------------

    annual_profit = (
        df["annual_gross_profit"]
        .fillna(0)
        .clip(lower=0)
    )

    if annual_profit.nunique() > 1:

        df["business_value_percentile"] = (
            annual_profit
            .rank(
                pct=True,
                method="average",
            )
            * 100
        )

    else:

        df["business_value_percentile"] = 50.0

    # --------------------------------------------------------
    # Exposure percentile
    # --------------------------------------------------------

    exposure = (
        df["estimated_financial_exposure"]
        .fillna(0)
        .clip(lower=0)
    )

    if exposure.nunique() > 1:

        df["exposure_percentile"] = (
            exposure
            .rank(
                pct=True,
                method="average",
            )
            * 100
        )

    else:

        df["exposure_percentile"] = 50.0

    # --------------------------------------------------------
    # Business impact score
    # --------------------------------------------------------

    df["business_impact_score"] = (

        df["business_value_percentile"]
        * 0.40

        +

        df["exposure_percentile"]
        * 0.60

    ).clip(
        0,
        100,
    ).round(2)

    # --------------------------------------------------------
    # Business impact class
    # --------------------------------------------------------

    df["business_impact_class"] = np.select(

        [
            df["business_impact_score"] >= 80,
            df["business_impact_score"] >= 60,
            df["business_impact_score"] >= 40,
        ],

        [
            "Very High",
            "High",
            "Medium",
        ],

        default="Low",
    )

    return df


# ============================================================
# CONFIDENCE ENGINE
# ============================================================

def calculate_confidence(df):

    print_header("CALCULATING RISK CONFIDENCE")

    df = df.copy()

    # --------------------------------------------------------
    # Evidence availability
    # --------------------------------------------------------

    evidence_columns = [

        "inventory_risk_score",
        "supplier_risk",
        "volatility_risk",
        "forecast_accuracy_risk",
        "anomaly_severity_score",
    ]

    evidence_present = pd.DataFrame(

        {
            column:
            df[column].fillna(0) > 0

            for column
            in evidence_columns
            if column in df.columns
        },

        index=df.index,
    )

    df["evidence_count"] = (
        evidence_present.sum(axis=1)
    )

    # --------------------------------------------------------
    # Forecast observations
    # --------------------------------------------------------

    observations = (
        df.get(
            "forecast_observations",
            pd.Series(
                0,
                index=df.index,
            ),
        )
        .fillna(0)
    )

    observation_confidence = (
        observations
        .clip(
            0,
            12,
        )
        / 12
        * 100
    )

    evidence_confidence = (
        df["evidence_count"]
        /
        max(
            len(evidence_columns),
            1,
        )
        *
        100
    )

    # --------------------------------------------------------
    # Base confidence
    # --------------------------------------------------------

    df["confidence_score"] = (

        evidence_confidence
        * 0.60

        +

        observation_confidence
        * 0.40

    ).clip(
        0,
        100,
    )

    # --------------------------------------------------------
    # Strong anomaly evidence
    # --------------------------------------------------------

    df.loc[
        df["anomaly_severity_score"] >= 2,
        "confidence_score",
    ] += 10

    # --------------------------------------------------------
    # Compound evidence
    # --------------------------------------------------------

    df.loc[
        df["compound_risk_count"] >= 2,
        "confidence_score",
    ] += 5

    df["confidence_score"] = (
        df["confidence_score"]
        .clip(
            0,
            100,
        )
        .round(2)
    )

    df["confidence_class"] = np.select(

        [
            df["confidence_score"] >= 80,
            df["confidence_score"] >= 60,
            df["confidence_score"] >= 40,
        ],

        [
            "High",
            "Medium",
            "Low",
        ],

        default="Very Low",
    )

    return df


# ============================================================
# FINAL RISK SCORE ADJUSTMENT
# ============================================================

def apply_business_adjustment(df):

    print_header("APPLYING BUSINESS IMPACT ADJUSTMENT")

    df = df.copy()

    # --------------------------------------------------------
    # We intentionally keep business impact from dominating
    # risk classification.
    #
    # Maximum adjustment = +10 points.
    # --------------------------------------------------------

    impact_adjustment = (

        (
            df["business_impact_score"]
            / 100
        )

        * 10

        *

        (
            df["risk_probability"]
            .clip(
                0,
                1,
            )
        )

    )

    df["business_impact_adjustment"] = (
        impact_adjustment
        .round(2)
    )

    df["risk_score_before_business"] = (
        df["risk_score"]
    )

    df["risk_score"] = (

        df["risk_score"]
        +

        df["business_impact_adjustment"]

    ).clip(
        0,
        100,
    ).round(2)

    df["risk_class"] = np.select(

        [
            df["risk_score"] >= 75,
            df["risk_score"] >= 50,
            df["risk_score"] >= 25,
        ],

        [
            "Critical",
            "High",
            "Moderate",
        ],

        default="Low",
    )

    return df


# ============================================================
# MANAGEMENT PRIORITY
# ============================================================

def assign_priority(df):

    print_header("ASSIGNING MANAGEMENT PRIORITY")

    df = df.copy()

    at_risk = (
        df["at_risk_value"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # P1:
    #
    # Critical risk
    # OR
    # high risk + major financial exposure
    # OR
    # severe compound disruption
    # --------------------------------------------------------

    p1 = (

        (df["risk_class"] == "Critical")

        |

        (
            (df["risk_score"] >= 60)
            &
            (at_risk >= 500_000)
        )

        |

        (
            (df["compound_risk_count"] >= 2)
            &
            (df["risk_score"] >= 50)
        )
    )

    # --------------------------------------------------------
    # P2
    # --------------------------------------------------------

    p2 = (

        (~p1)

        &

        (
            (df["risk_class"] == "High")

            |

            (
                (df["risk_score"] >= 40)
                &
                (at_risk >= 100_000)
            )
        )
    )

    # --------------------------------------------------------
    # P3
    # --------------------------------------------------------

    p3 = (

        (~p1)
        &
        (~p2)

        &

        (
            (df["risk_class"] == "Moderate")

            |

            (
                df["compound_risk_flag"] == 1
            )
        )
    )

    df["management_priority"] = np.select(

        [
            p1,
            p2,
            p3,
        ],

        [
            "P1 - Immediate",
            "P2 - High",
            "P3 - Monitor",
        ],

        default="P4 - Routine",
    )

    priority_map = {

        "P1 - Immediate": 1,
        "P2 - High": 2,
        "P3 - Monitor": 3,
        "P4 - Routine": 4,
    }

    df["priority_rank"] = (
        df["management_priority"]
        .map(priority_map)
    )

    return df


# ============================================================
# ACTION ENGINE
# ============================================================

def generate_recommended_action(row):

    driver = row.get(
        "primary_risk_driver",
        "Unknown",
    )

    risk_score = float(
        row.get(
            "risk_score",
            0,
        )
        or 0
    )

    anomaly = float(
        row.get(
            "anomaly_severity_score",
            0,
        )
        or 0
    )

    inventory_days = float(
        row.get(
            "inventory_days",
            0,
        )
        or 0
    )

    supplier_risk = float(
        row.get(
            "supplier_risk",
            0,
        )
        or 0
    )

    forecast_change = float(
        row.get(
            "forecast_change_pct",
            0,
        )
        or 0
    )

    forecast_accuracy = float(
        row.get(
            "forecast_accuracy_risk",
            0,
        )
        or 0
    )

    excess_risk = float(
        row.get(
            "excess_inventory_risk",
            0,
        )
        or 0
    )

    compound_count = int(
        row.get(
            "compound_risk_count",
            0,
        )
        or 0
    )

    # ========================================================
    # COMPOUND RISKS FIRST
    # ========================================================

    if (
        anomaly >= 2
        and inventory_days < 14
        and supplier_risk >= 50
    ):

        return (
            "Activate supply disruption response; "
            "expedite replenishment and evaluate alternate sourcing"
        )

    if (
        anomaly >= 2
        and inventory_days < 14
    ):

        return (
            "Expedite replenishment; "
            "validate demand spike and protect near-term service levels"
        )

    if (
        anomaly >= 2
        and supplier_risk >= 50
    ):

        return (
            "Validate demand signal and supplier capacity; "
            "prepare alternate sourcing if demand persists"
        )

    if (
        forecast_change > 20
        and inventory_days < 30
    ):

        return (
            "Increase replenishment coverage; "
            "validate forecast uplift against available inventory"
        )

    if (
        excess_risk >= 50
        and forecast_change < -10
    ):

        return (
            "Reduce replenishment exposure; "
            "review excess stock and potential obsolescence"
        )

    # ========================================================
    # DRIVER-SPECIFIC
    # ========================================================

    if driver == "Supplier":

        if supplier_risk >= 75:

            return (
                "Escalate supplier risk; "
                "secure contingency supply and review alternate vendors"
            )

        if supplier_risk >= 50:

            return (
                "Review supplier reliability and "
                "establish contingency replenishment coverage"
            )

        return (
            "Monitor supplier performance and lead-time stability"
        )

    if driver == "Inventory":

        if inventory_days < 7:

            return (
                "Immediate replenishment review; "
                "inventory coverage is below one week"
            )

        if inventory_days < 14:

            return (
                "Prioritize replenishment; "
                "inventory coverage is below two weeks"
            )

        return (
            "Review replenishment policy and inventory service level"
        )

    if driver == "Stockout":

        return (
            "Review stockout drivers and "
            "prioritize replenishment to protect service"
        )

    if driver == "Demand Anomaly":

        if anomaly >= 3:

            return (
                "Immediately validate demand anomaly; "
                "protect supply availability and investigate root cause"
            )

        return (
            "Investigate demand anomaly and "
            "increase short-term demand monitoring"
        )

    if driver == "Demand Volatility":

        return (
            "Review demand variability and "
            "increase forecast monitoring frequency"
        )

    if driver == "Forecast Accuracy":

        if forecast_accuracy >= 70:

            return (
                "Review forecast model performance; "
                "recalibrate demand assumptions before next planning cycle"
            )

        return (
            "Monitor forecast accuracy and "
            "review model error drivers"
        )

    if driver == "Forecast Change":

        if forecast_change > 20:

            return (
                "Validate forecast uplift and "
                "increase replenishment coverage"
            )

        if forecast_change < -20:

            return (
                "Validate demand decline and "
                "review inventory overstock exposure"
            )

        return (
            "Validate forecast revision against current demand signals"
        )

    if driver == "Excess Inventory":

        return (
            "Reduce replenishment and "
            "review excess inventory liquidation options"
        )

    # ========================================================
    # COMPOUND FALLBACK
    # ========================================================

    if compound_count >= 2:

        return (
            "Cross-functional review required; "
            "multiple risk signals are interacting"
        )

    # ========================================================
    # RISK FALLBACK
    # ========================================================

    if risk_score >= 75:

        return (
            "Immediate management intervention required"
        )

    if risk_score >= 50:

        return (
            "Prioritize for management review and corrective action"
        )

    if risk_score >= 25:

        return (
            "Monitor risk indicators during the next planning cycle"
        )

    return (
        "Maintain current policy and monitor"
    )


def generate_actions(df):

    print_header("GENERATING MANAGEMENT ACTIONS")

    df = df.copy()

    df["recommended_action"] = (
        df.apply(
            generate_recommended_action,
            axis=1,
        )
    )

    return df


# ============================================================
# RISK EXECUTIVE LABEL
# ============================================================

def create_risk_label(df):

    df = df.copy()

    df["risk_label"] = (

        df["risk_class"]

        +

        " | "

        +

        df["management_priority"]

    )

    return df


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def create_executive_summary(df):

    print_header("BUILDING EXECUTIVE RISK SUMMARY")

    total_products = (
        df["product_id"]
        .nunique()
    )

    critical = int(
        (
            df["risk_class"]
            == "Critical"
        ).sum()
    )

    high = int(
        (
            df["risk_class"]
            == "High"
        ).sum()
    )

    moderate = int(
        (
            df["risk_class"]
            == "Moderate"
        ).sum()
    )

    low = int(
        (
            df["risk_class"]
            == "Low"
        ).sum()
    )

    p1 = int(
        (
            df["management_priority"]
            == "P1 - Immediate"
        ).sum()
    )

    p2 = int(
        (
            df["management_priority"]
            == "P2 - High"
        ).sum()
    )

    p3 = int(
        (
            df["management_priority"]
            == "P3 - Monitor"
        ).sum()
    )

    anomaly_products = int(
        (
            df["demand_anomaly_flag"]
            > 0
        ).sum()
    )

    compound_products = int(
        (
            df["compound_risk_flag"]
            > 0
        ).sum()
    )

    financial_exposure = (
        df["estimated_financial_exposure"]
        .sum()
    )

    at_risk_value = (
        df["at_risk_value"]
        .sum()
    )

    lost_sales_exposure = (
        df["lost_sales_exposure"]
        .sum()
    )

    optimization_opportunity = (
        df["optimization_opportunity"]
        .sum()
    )

    average_risk = (
        df["risk_score"]
        .mean()
    )

    average_confidence = (
        df["confidence_score"]
        .mean()
    )

    # --------------------------------------------------------
    # Executive summary
    # --------------------------------------------------------

    summary = pd.DataFrame(

        [

            {
                "metric":
                    "Products monitored",
                "value":
                    total_products,
            },

            {
                "metric":
                    "Critical risk products",
                "value":
                    critical,
            },

            {
                "metric":
                    "High risk products",
                "value":
                    high,
            },

            {
                "metric":
                    "Moderate risk products",
                "value":
                    moderate,
            },

            {
                "metric":
                    "Low risk products",
                "value":
                    low,
            },

            {
                "metric":
                    "P1 immediate actions",
                "value":
                    p1,
            },

            {
                "metric":
                    "P2 high priority actions",
                "value":
                    p2,
            },

            {
                "metric":
                    "P3 monitoring actions",
                "value":
                    p3,
            },

            {
                "metric":
                    "Products with demand anomalies",
                "value":
                    anomaly_products,
            },

            {
                "metric":
                    "Products with compound risk",
                "value":
                    compound_products,
            },

            {
                "metric":
                    "Average risk score",
                "value":
                    round(
                        average_risk,
                        2,
                    ),
            },

            {
                "metric":
                    "Average risk confidence",
                "value":
                    round(
                        average_confidence,
                        2,
                    ),
            },

            {
                "metric":
                    "Estimated financial exposure",
                "value":
                    round(
                        financial_exposure,
                        2,
                    ),
            },

            {
                "metric":
                    "Risk-weighted at-risk value",
                "value":
                    round(
                        at_risk_value,
                        2,
                    ),
            },

            {
                "metric":
                    "Estimated lost-sales exposure",
                "value":
                    round(
                        lost_sales_exposure,
                        2,
                    ),
            },

            {
                "metric":
                    "Optimization opportunity",
                "value":
                    round(
                        optimization_opportunity,
                        2,
                    ),
            },
        ]
    )

    return summary


# ============================================================
# RISK SUMMARY
# ============================================================

def create_risk_summary(df):

    summary = (

        df.groupby(

            [
                "risk_class",
                "primary_risk_driver",
            ],

            dropna=False,
        )

        .agg(

            products=(
                "product_id",
                "nunique",
            ),

            average_risk_score=(
                "risk_score",
                "mean",
            ),

            average_confidence=(
                "confidence_score",
                "mean",
            ),

            financial_exposure=(
                "estimated_financial_exposure",
                "sum",
            ),

            at_risk_value=(
                "at_risk_value",
                "sum",
            ),

            lost_sales_exposure=(
                "lost_sales_exposure",
                "sum",
            ),

            optimization_opportunity=(
                "optimization_opportunity",
                "sum",
            ),

            compound_risk_products=(
                "compound_risk_flag",
                "sum",
            ),
        )

        .reset_index()
    )

    numeric_columns = [

        "average_risk_score",
        "average_confidence",

        "financial_exposure",
        "at_risk_value",

        "lost_sales_exposure",
        "optimization_opportunity",

        "compound_risk_products",
    ]

    for column in numeric_columns:

        summary[column] = (
            pd.to_numeric(
                summary[column],
                errors="coerce",
            )
            .fillna(0)
            .round(2)
        )

    return summary


# ============================================================
# ACTION QUEUE
# ============================================================

def create_action_queue(df):

    action_columns = [

        "priority_rank",
        "management_priority",

        "product_id",
        "product_name",
        "category",
        "demand_class",

        "risk_score",
        "risk_class",

        "confidence_score",
        "confidence_class",

        "business_impact_score",
        "business_impact_class",

        "primary_risk_driver",
        "primary_risk_driver_score",

        "anomaly_severity_score",
        "max_anomaly_deviation",
        "max_anomaly_z",

        "forecast_change_pct",
        "wape",

        "inventory_risk_score",
        "stockout_risk",
        "service_risk",

        "supplier_risk",
        "volatility_risk",
        "excess_inventory_risk",

        "inventory_days",
        "fill_rate",

        "estimated_financial_exposure",
        "at_risk_value",

        "lost_sales_exposure",
        "supplier_delay_exposure",

        "inventory_obsolescence_exposure",

        "optimization_opportunity",

        "compound_risk_count",
        "compound_risk_score",

        "recommended_action",
    ]

    available = [

        column
        for column
        in action_columns
        if column in df.columns
    ]

    actions = (

        df[available]

        .sort_values(

            [
                "priority_rank",
                "risk_score",
                "at_risk_value",
            ],

            ascending=[
                True,
                False,
                False,
            ],
        )

        .reset_index(
            drop=True
        )
    )

    actions.insert(

        0,
        "action_rank",

        np.arange(
            1,
            len(actions) + 1,
        ),
    )

    return actions


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

def run_data_quality_checks(df):

    print_header("RUNNING FINAL DATA QUALITY CHECKS")

    expected_products = (
        df["product_id"]
        .nunique()
    )

    duplicate_ids = int(
        df["product_id"]
        .duplicated()
        .sum()
    )

    missing_scores = int(
        df["risk_score"]
        .isna()
        .sum()
    )

    missing_classes = int(
        df["risk_class"]
        .isna()
        .sum()
    )

    missing_actions = int(
        df["recommended_action"]
        .isna()
        .sum()
    )

    infinite_values = int(
        np.isinf(
            df.select_dtypes(
                include=np.number
            )
        )
        .sum()
        .sum()
    )

    checks = [

        (
            "Product rows",
            len(df),
            expected_products,
        ),

        (
            "Unique products",
            df["product_id"].nunique(),
            expected_products,
        ),

        (
            "Duplicate product IDs",
            duplicate_ids,
            0,
        ),

        (
            "Missing risk scores",
            missing_scores,
            0,
        ),

        (
            "Missing risk classes",
            missing_classes,
            0,
        ),

        (
            "Missing recommended actions",
            missing_actions,
            0,
        ),

        (
            "Infinite numeric values",
            infinite_values,
            0,
        ),
    ]

    passed = True

    for name, actual, expected in checks:

        print(
            f"{name:<40}"
            f"{actual:>8} "
            f"(expected {expected})"
        )

        if actual != expected:

            passed = False

    if not passed:

        raise ValueError(
            "Final data quality checks failed."
        )

    print()
    print(
        "[OK] Final data quality checks passed."
    )


# ============================================================
# FINAL CLEANUP
# ============================================================

def finalize_dataset(df):

    print_header("FINALIZING RISK DATASET")

    df = df.copy()

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [

        "risk_score",
        "risk_score_base",
        "risk_score_pre_compound",
        "risk_score_before_business",

        "compound_risk_score",
        "compound_risk_adjustment",
        "business_impact_adjustment",

        "inventory_risk_score",
        "stockout_risk",
        "service_risk",

        "supplier_risk",
        "volatility_risk",
        "excess_inventory_risk",

        "forecast_accuracy_risk",

        "anomaly_severity_score",
        "max_anomaly_deviation",
        "max_anomaly_z",

        "forecast_change_pct",

        "wape",

        "inventory_days",
        "fill_rate",

        "estimated_financial_exposure",
        "at_risk_value",

        "risk_probability",

        "lost_sales_exposure",
        "supplier_delay_exposure",
        "forecast_increase_exposure",

        "inventory_obsolescence_exposure",

        "service_risk_value",

        "optimization_opportunity",

        "confidence_score",

        "business_impact_score",

        "compound_risk_count",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = (

                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

                .fillna(0)

                .round(4)
            )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (

        df.sort_values(

            [
                "priority_rank",
                "risk_score",
                "at_risk_value",
            ],

            ascending=[
                True,
                False,
                False,
            ],
        )

        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# WRITE OUTPUTS
# ============================================================

def write_outputs(
    risk_df,
    risk_summary,
    action_queue,
    executive_summary,
):

    print_header("WRITING RISK INTELLIGENCE OUTPUTS")

    risk_path = (
        OUTPUT_DIR
        / "supply_chain_risk.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "risk_summary.csv"
    )

    actions_path = (
        OUTPUT_DIR
        / "risk_actions.csv"
    )

    executive_path = (
        OUTPUT_DIR
        / "risk_executive_summary.csv"
    )

    risk_df.to_csv(
        risk_path,
        index=False,
    )

    risk_summary.to_csv(
        summary_path,
        index=False,
    )

    action_queue.to_csv(
        actions_path,
        index=False,
    )

    executive_summary.to_csv(
        executive_path,
        index=False,
    )

    print(
        f"Unified risk dataset:\n{risk_path}"
    )

    print(
        f"\nRisk summary:\n{summary_path}"
    )

    print(
        f"\nAction queue:\n{actions_path}"
    )

    print(
        f"\nExecutive summary:\n{executive_path}"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    risk_df,
    risk_summary,
    executive_summary,
):

    print_header(
        "SUPPLY CHAIN RISK INTELLIGENCE SUMMARY"
    )

    print(
        executive_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("RISK DISTRIBUTION")
    print("-" * 80)

    distribution = (

        risk_df["risk_class"]

        .value_counts()

        .rename_axis(
            "risk_class"
        )

        .reset_index(
            name="products"
        )
    )

    print(
        distribution.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Priority distribution
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("MANAGEMENT PRIORITY")
    print("-" * 80)

    priority = (

        risk_df["management_priority"]

        .value_counts()

        .rename_axis(
            "management_priority"
        )

        .reset_index(
            name="products"
        )
    )

    print(
        priority.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Primary drivers
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("PRIMARY RISK DRIVERS")
    print("-" * 80)

    drivers = (

        risk_df["primary_risk_driver"]

        .value_counts()

        .rename_axis(
            "risk_driver"
        )

        .reset_index(
            name="products"
        )
    )

    print(
        drivers.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Top exposures
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("TOP 20 AT-RISK EXPOSURES")
    print("-" * 80)

    top_columns = [

        "product_id",
        "product_name",

        "risk_score",
        "risk_class",

        "management_priority",

        "primary_risk_driver",

        "confidence_score",

        "business_impact_score",

        "estimated_financial_exposure",

        "at_risk_value",

        "inventory_days",

        "forecast_change_pct",

        "compound_risk_count",

        "recommended_action",
    ]

    top_columns = [

        column
        for column in top_columns
        if column in risk_df.columns
    ]

    top = (

        risk_df[top_columns]

        .sort_values(
            "at_risk_value",
            ascending=False,
        )

        .head(20)
    )

    print(
        top.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Risk summary
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("RISK SUMMARY BY DRIVER")
    print("-" * 80)

    print(
        risk_summary.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("SUPPLY CHAIN RISK INTELLIGENCE ENGINE")
    print("=" * 80)

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_input_files()

    # ========================================================
    # LOAD
    # ========================================================

    datasets = load_data()

    # ========================================================
    # PREPARE INPUTS
    # ========================================================

    products = prepare_products(
        datasets["products"]
    )

    inventory_risk = prepare_inventory_risk(
        datasets["inventory_risk"]
    )

    optimization = prepare_optimization(
        datasets["inventory_optimization"]
    )

    anomalies = prepare_anomalies(
        datasets["anomalies"]
    )

    forecast_changes = prepare_forecast_changes(
        datasets["forecast_changes"]
    )

    forecast_metrics = prepare_forecast_metrics(
        datasets["forecast_metrics"]
    )

    # ========================================================
    # MASTER DATASET
    # ========================================================

    master = build_master_dataset(

        products=products,

        inventory_risk=inventory_risk,

        optimization=optimization,

        anomalies=anomalies,

        forecast_changes=forecast_changes,

        forecast_metrics=forecast_metrics,
    )

    # ========================================================
    # SIGNAL NORMALIZATION
    # ========================================================

    master = normalize_risk_components(
        master
    )

    # ========================================================
    # COMPOUND RISK ENGINE
    # ========================================================

    master = create_interaction_signals(
        master
    )

    # ========================================================
    # BASE RISK
    # ========================================================

    master = calculate_risk_score(
        master
    )

    # ========================================================
    # PRIMARY DRIVER
    # ========================================================

    master = identify_primary_driver(
        master
    )

    # ========================================================
    # FINANCIAL EXPOSURE
    # ========================================================

    master = calculate_financial_exposure(
        master
    )

    # ========================================================
    # BUSINESS IMPACT
    # ========================================================

    master = calculate_business_impact(
        master
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    master = calculate_confidence(
        master
    )

    # ========================================================
    # BUSINESS IMPACT ADJUSTMENT
    # ========================================================

    master = apply_business_adjustment(
        master
    )

    # ========================================================
    # MANAGEMENT PRIORITY
    # ========================================================

    master = assign_priority(
        master
    )

    # ========================================================
    # ACTION ENGINE
    # ========================================================

    master = generate_actions(
        master
    )

    # ========================================================
    # LABEL
    # ========================================================

    master = create_risk_label(
        master
    )

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    master = finalize_dataset(
        master
    )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    run_data_quality_checks(
        master
    )

    # ========================================================
    # SUMMARIES
    # ========================================================

    executive_summary = (
        create_executive_summary(
            master
        )
    )

    risk_summary = (
        create_risk_summary(
            master
        )
    )

    action_queue = (
        create_action_queue(
            master
        )
    )

    # ========================================================
    # WRITE
    # ========================================================

    write_outputs(

        risk_df=master,

        risk_summary=risk_summary,

        action_queue=action_queue,

        executive_summary=executive_summary,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_results(

        risk_df=master,

        risk_summary=risk_summary,

        executive_summary=executive_summary,
    )

    print()
    print("=" * 80)
    print(
        "SUPPLY CHAIN RISK INTELLIGENCE ENGINE COMPLETE"
    )
    print("=" * 80)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()