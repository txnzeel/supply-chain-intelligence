from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEMAND_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_demand.csv"
)

FORECAST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasts"
    / "demand_forecasts.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "anomalies"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Detection configuration
# -----------------------------------------------------------------------------

MIN_HISTORY_DAYS = 90
ROLLING_WINDOW = 28

# Historical backtesting
VALIDATION_HORIZON = 30
BACKTEST_WINDOWS = 3

# Anomaly thresholds
Z_THRESHOLD = 3.0
MODERATE_Z_THRESHOLD = 2.0

# Minimum demand needed before percentage-based residual calculations
MIN_DEMAND_FOR_PERCENT = 1.0

# Winsorization prevents one extreme anomaly from destroying the baseline
RESIDUAL_WINSOR_LOWER = 0.01
RESIDUAL_WINSOR_UPPER = 0.99


# =============================================================================
# GLOBAL
# =============================================================================

print("=" * 80)
print("SUPPLY CHAIN DEMAND ANOMALY DETECTION ENGINE")
print("=" * 80)


# =============================================================================
# HELPERS
# =============================================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_divide(a, b):
    """
    Safe element-wise division.
    """
    return np.divide(
        a,
        b,
        out=np.zeros_like(
            np.asarray(a, dtype=float),
            dtype=float,
        ),
        where=np.asarray(b, dtype=float) != 0,
    )


def wape(actual, forecast):
    """
    Weighted Absolute Percentage Error.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.nan

    return np.sum(np.abs(actual - forecast)) / denominator


def normalize_column_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def detect_column(df, candidates):
    """
    Find a column from a list of possible names.
    """

    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:

        key = normalize_column_name(candidate)

        if key in normalized:
            return normalized[key]

    return None


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print_section("LOADING DATA")

    if not DEMAND_PATH.exists():
        raise FileNotFoundError(
            f"Demand dataset not found:\n{DEMAND_PATH}"
        )

    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n{FORECAST_PATH}"
        )

    demand = pd.read_csv(
        DEMAND_PATH,
        parse_dates=["date"],
    )

    forecasts = pd.read_csv(
        FORECAST_PATH,
    )

    print(f"Demand rows: {len(demand):,}")
    print(f"Forecast rows: {len(forecasts):,}")

    # -------------------------------------------------------------------------
    # Demand schema
    # -------------------------------------------------------------------------

    demand_product = detect_column(
        demand,
        ["product_id"],
    )

    demand_date = detect_column(
        demand,
        ["date"],
    )

    demand_value = detect_column(
        demand,
        [
            "actual_demand",
            "demand",
            "actual",
        ],
    )

    if demand_product is None:
        raise ValueError(
            "Demand dataset is missing product_id."
        )

    if demand_date is None:
        raise ValueError(
            "Demand dataset is missing date."
        )

    if demand_value is None:
        raise ValueError(
            "Demand dataset is missing actual demand."
        )

    demand = demand.rename(
        columns={
            demand_product: "product_id",
            demand_date: "date",
            demand_value: "actual",
        }
    )

    demand["date"] = pd.to_datetime(
        demand["date"],
        errors="coerce",
    )

    demand["actual"] = pd.to_numeric(
        demand["actual"],
        errors="coerce",
    )

    demand = demand.dropna(
        subset=[
            "product_id",
            "date",
            "actual",
        ]
    )

    demand["product_id"] = (
        demand["product_id"]
        .astype(str)
        .str.strip()
    )

    demand = demand.sort_values(
        ["product_id", "date"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Forecast schema
    # -------------------------------------------------------------------------

    forecast_product = detect_column(
        forecasts,
        ["product_id"],
    )

    forecast_date = detect_column(
        forecasts,
        [
            "forecast_date",
            "date",
            "ds",
        ],
    )

    forecast_value = detect_column(
        forecasts,
        [
            "forecast_demand",
            "forecast",
            "predicted_demand",
            "prediction",
            "yhat",
        ],
    )

    if forecast_product is None:
        raise ValueError(
            "Forecast dataset is missing product_id."
        )

    if forecast_date is None:
        raise ValueError(
            "Forecast dataset is missing forecast date."
        )

    if forecast_value is None:
        raise ValueError(
            "Forecast dataset is missing forecast demand."
        )

    forecasts = forecasts.rename(
        columns={
            forecast_product: "product_id",
            forecast_date: "date",
            forecast_value: "forecast",
        }
    )

    forecasts["date"] = pd.to_datetime(
        forecasts["date"],
        errors="coerce",
    )

    forecasts["forecast"] = pd.to_numeric(
        forecasts["forecast"],
        errors="coerce",
    )

    forecasts = forecasts.dropna(
        subset=[
            "product_id",
            "date",
            "forecast",
        ]
    )

    forecasts["product_id"] = (
        forecasts["product_id"]
        .astype(str)
        .str.strip()
    )

    forecasts = forecasts.sort_values(
        ["product_id", "date"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    print()
    print("DEMAND DATA")
    print("-" * 80)
    print(f"Date column: date")
    print(
        f"Date range: "
        f"{demand['date'].min().date()} → "
        f"{demand['date'].max().date()}"
    )
    print(
        f"Products: {demand['product_id'].nunique():,}"
    )

    print()
    print("FORECAST DATA")
    print("-" * 80)
    print("Forecast date column detected.")
    print("Forecast value column detected.")
    print(
        f"Date range: "
        f"{forecasts['date'].min().date()} → "
        f"{forecasts['date'].max().date()}"
    )
    print(
        f"Products: {forecasts['product_id'].nunique():,}"
    )
    print(
        f"Forecast rows: {len(forecasts):,}"
    )

    demand_products = set(
        demand["product_id"].unique()
    )

    forecast_products = set(
        forecasts["product_id"].unique()
    )

    missing = demand_products - forecast_products
    unexpected = forecast_products - demand_products

    print()
    print("PRODUCT COVERAGE")
    print("-" * 80)
    print(
        f"Historical demand products: "
        f"{len(demand_products):,}"
    )
    print(
        f"Forecast products: "
        f"{len(forecast_products):,}"
    )
    print(
        f"Products missing forecasts: "
        f"{len(missing):,}"
    )
    print(
        f"Unexpected forecast products: "
        f"{len(unexpected):,}"
    )

    print()
    print("NORMALIZED FORECAST SCHEMA")
    print("-" * 80)
    print("product_id → product_id")
    print("forecast_date/date → date")
    print("forecast_demand/forecast → forecast")

    print()
    print("Data loading complete.")

    return demand, forecasts


# =============================================================================
# HISTORICAL FORECASTING
# =============================================================================

def seasonal_naive_forecast(history, horizon, season_length=7):
    """
    Seasonal naive forecast using the last available seasonal cycle.
    """

    history = np.asarray(
        history,
        dtype=float,
    )

    if len(history) == 0:
        return np.zeros(horizon)

    if len(history) < season_length:
        return np.repeat(
            np.mean(history),
            horizon,
        )

    last_cycle = history[-season_length:]

    repeats = int(
        np.ceil(
            horizon / season_length
        )
    )

    forecast = np.tile(
        last_cycle,
        repeats,
    )

    return forecast[:horizon]


def rolling_mean_forecast(history, horizon, window=28):
    """
    Rolling mean baseline.
    """

    history = np.asarray(
        history,
        dtype=float,
    )

    if len(history) == 0:
        return np.zeros(horizon)

    window = min(
        window,
        len(history),
    )

    value = np.mean(
        history[-window:]
    )

    return np.repeat(
        value,
        horizon,
    )


def choose_historical_model(history):
    """
    Lightweight model selection for historical backtesting.

    We intentionally keep this computationally cheap because the
    dataset contains 2,000 products × 730 observations.
    """

    if len(history) < 35:
        return "RollingMean28"

    recent = np.asarray(
        history[-28:],
        dtype=float,
    )

    mean = np.mean(recent)

    if mean <= 0:
        return "RollingMean28"

    cv = np.std(recent) / mean

    # Intermittent demand
    zero_ratio = np.mean(
        np.asarray(history) == 0
    )

    if zero_ratio >= 0.40:
        return "SeasonalNaive7"

    # Strong weekly pattern / volatility
    if cv >= 0.50:
        return "SeasonalNaive7"

    return "SeasonalNaive7"


def historical_forecast(
    history,
    horizon,
    model,
):
    """
    Produce historical backtest forecast.
    """

    if model == "SeasonalNaive7":

        return seasonal_naive_forecast(
            history,
            horizon,
            season_length=7,
        )

    if model == "RollingMean28":

        return rolling_mean_forecast(
            history,
            horizon,
            window=28,
        )

    return seasonal_naive_forecast(
        history,
        horizon,
        season_length=7,
    )


# =============================================================================
# BUILD HISTORICAL RESIDUALS
# =============================================================================

def build_historical_residuals(
    demand,
    validation_horizon=30,
    backtest_windows=3,
):
    """
    Build actual-vs-forecast residuals using historical rolling
    backtesting.

    This is the critical part of the anomaly system.

    We DO NOT compare 2027 forecasts to 2025/2026 actuals.

    Instead:

        historical data
              ↓
        pretend we are at an earlier date
              ↓
        generate forecast
              ↓
        compare forecast with actual future observations
              ↓
        calculate residual distribution
              ↓
        use distribution to identify anomalies
    """

    print_section(
        "BUILDING HISTORICAL FORECAST RESIDUAL DATASET"
    )

    records = []

    products = demand["product_id"].unique()

    total_products = len(products)

    print(
        f"Products: {total_products:,}"
    )

    for index, product_id in enumerate(products, start=1):

        product_df = demand[
            demand["product_id"] == product_id
        ][
            [
                "product_id",
                "date",
                "actual",
            ]
        ].sort_values("date")

        values = (
            product_df["actual"]
            .to_numpy(dtype=float)
        )

        dates = (
            product_df["date"]
            .to_numpy()
        )

        n = len(values)

        minimum_required = (
            MIN_HISTORY_DAYS
            + validation_horizon
        )

        if n < minimum_required:
            continue

        # ---------------------------------------------------------------------
        # Backtest origins
        # ---------------------------------------------------------------------

        latest_origin = (
            n - validation_horizon
        )

        origins = []

        for window_number in range(
            backtest_windows
        ):

            origin = (
                latest_origin
                - (
                    window_number
                    * validation_horizon
                )
            )

            if (
                origin >= MIN_HISTORY_DAYS
            ):
                origins.append(origin)

        origins = sorted(origins)

        for origin in origins:

            history = values[:origin]

            actual_future = values[
                origin:
                origin + validation_horizon
            ]

            future_dates = dates[
                origin:
                origin + validation_horizon
            ]

            if len(actual_future) != validation_horizon:
                continue

            model = choose_historical_model(
                history
            )

            forecast = historical_forecast(
                history,
                validation_horizon,
                model,
            )

            for day_index in range(
                validation_horizon
            ):

                actual = float(
                    actual_future[day_index]
                )

                predicted = float(
                    forecast[day_index]
                )

                residual = (
                    actual
                    - predicted
                )

                absolute_error = abs(
                    residual
                )

                denominator = max(
                    abs(predicted),
                    MIN_DEMAND_FOR_PERCENT,
                )

                residual_pct = (
                    residual
                    / denominator
                )

                records.append(
                    {
                        "product_id": product_id,
                        "date": pd.Timestamp(
                            future_dates[
                                day_index
                            ]
                        ),
                        "actual": actual,
                        "forecast": predicted,
                        "residual": residual,
                        "absolute_error": absolute_error,
                        "residual_pct": residual_pct,
                        "backtest_origin": pd.Timestamp(
                            dates[origin - 1]
                        ),
                        "model": model,
                    }
                )

        if index % 100 == 0:
            print(
                f"Processed "
                f"{index:,}/{total_products:,} "
                f"products"
            )

    residuals = pd.DataFrame(records)

    if residuals.empty:
        raise ValueError(
            "Historical residual dataset is empty."
        )

    residuals = residuals.sort_values(
        [
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)

    print()
    print(
        f"Historical residual rows: "
        f"{len(residuals):,}"
    )

    print(
        f"Products with residual history: "
        f"{residuals['product_id'].nunique():,}"
    )

    return residuals


# =============================================================================
# RESIDUAL BASELINES
# =============================================================================

def build_residual_baselines(
    historical_residuals,
):
    """
    Build product-level residual distributions.

    Robust statistics are used because anomalies should not
    redefine their own baseline.
    """

    print_section(
        "BUILDING PRODUCT RESIDUAL BASELINES"
    )

    grouped = []

    for product_id, group in historical_residuals.groupby(
        "product_id"
    ):

        residual = (
            group["residual"]
            .astype(float)
            .to_numpy()
        )

        residual_pct = (
            group["residual_pct"]
            .astype(float)
            .to_numpy()
        )

        if len(residual) < 5:
            continue

        residual_lower = np.quantile(
            residual,
            RESIDUAL_WINSOR_LOWER,
        )

        residual_upper = np.quantile(
            residual,
            RESIDUAL_WINSOR_UPPER,
        )

        residual_pct_lower = np.quantile(
            residual_pct,
            RESIDUAL_WINSOR_LOWER,
        )

        residual_pct_upper = np.quantile(
            residual_pct,
            RESIDUAL_WINSOR_UPPER,
        )

        clipped_residual = np.clip(
            residual,
            residual_lower,
            residual_upper,
        )

        clipped_pct = np.clip(
            residual_pct,
            residual_pct_lower,
            residual_pct_upper,
        )

        median_residual = np.median(
            clipped_residual
        )

        median_pct = np.median(
            clipped_pct
        )

        mad_residual = np.median(
            np.abs(
                clipped_residual
                - median_residual
            )
        )

        mad_pct = np.median(
            np.abs(
                clipped_pct
                - median_pct
            )
        )

        # Convert MAD to a standard-deviation-like scale.
        robust_std = (
            1.4826
            * mad_residual
        )

        robust_pct_std = (
            1.4826
            * mad_pct
        )

        # Prevent division by zero.
        robust_std = max(
            robust_std,
            1.0,
        )

        robust_pct_std = max(
            robust_pct_std,
            0.01,
        )

        grouped.append(
            {
                "product_id": product_id,
                "residual_median": median_residual,
                "residual_robust_std": robust_std,
                "residual_pct_median": median_pct,
                "residual_pct_robust_std": robust_pct_std,
                "historical_residual_count": len(
                    residual
                ),
            }
        )

    baselines = pd.DataFrame(
        grouped
    )

    if baselines.empty:
        raise ValueError(
            "Could not build residual baselines."
        )

    print(
        f"Residual baselines: "
        f"{len(baselines):,} products"
    )

    return baselines


# =============================================================================
# CURRENT DEMAND ANOMALY DETECTION
# =============================================================================

def detect_current_anomalies(
    demand,
    baselines,
):
    """
    Detect anomalies in the latest historical demand period.

    Each product gets:
      - rolling baseline
      - residual
      - robust z-score
      - percentage deviation
      - anomaly severity
      - anomaly direction
    """

    print_section(
        "DETECTING CURRENT DEMAND ANOMALIES"
    )

    df = demand.copy()

    df = df.sort_values(
        [
            "product_id",
            "date",
        ]
    )

    # -------------------------------------------------------------------------
    # Rolling demand statistics
    # -------------------------------------------------------------------------

    grouped = df.groupby(
        "product_id",
        group_keys=False,
    )

    df["rolling_mean_28"] = grouped[
        "actual"
    ].transform(
        lambda x: x.shift(1)
        .rolling(
            ROLLING_WINDOW,
            min_periods=7,
        )
        .mean()
    )

    df["rolling_std_28"] = grouped[
        "actual"
    ].transform(
        lambda x: x.shift(1)
        .rolling(
            ROLLING_WINDOW,
            min_periods=7,
        )
        .std()
    )

    df["rolling_mean_7"] = grouped[
        "actual"
    ].transform(
        lambda x: x.shift(1)
        .rolling(
            7,
            min_periods=3,
        )
        .mean()
    )

    df["rolling_std_7"] = grouped[
        "actual"
    ].transform(
        lambda x: x.shift(1)
        .rolling(
            7,
            min_periods=3,
        )
        .std()
    )

    # -------------------------------------------------------------------------
    # Merge residual baselines
    # -------------------------------------------------------------------------

    df = df.merge(
        baselines,
        on="product_id",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Current residual
    # -------------------------------------------------------------------------

    df["current_baseline"] = (
        df["rolling_mean_28"]
    )

    df["current_residual"] = (
        df["actual"]
        - df["current_baseline"]
    )

    df["current_deviation_pct"] = safe_divide(
        df["current_residual"],
        df["current_baseline"].clip(
            lower=MIN_DEMAND_FOR_PERCENT
        ),
    )

    df["current_z_score"] = safe_divide(
        df["current_residual"]
        - df["residual_median"],
        df["residual_robust_std"],
    )

    df["abs_z_score"] = abs(
        df["current_z_score"]
    )

    # -------------------------------------------------------------------------
    # Latest observation per product
    # -------------------------------------------------------------------------

    latest = (
        df.sort_values(
            [
                "product_id",
                "date",
            ]
        )
        .groupby(
            "product_id",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    # -------------------------------------------------------------------------
    # Direction
    # -------------------------------------------------------------------------

    latest["anomaly_direction"] = np.where(
        latest["current_residual"] > 0,
        "Demand Spike",
        np.where(
            latest["current_residual"] < 0,
            "Demand Drop",
            "Normal",
        ),
    )

    # -------------------------------------------------------------------------
    # Severity
    # -------------------------------------------------------------------------

    latest["anomaly_severity"] = np.select(
        [
            latest["abs_z_score"] >= 4.0,
            latest["abs_z_score"] >= 3.0,
            latest["abs_z_score"] >= 2.0,
        ],
        [
            "Critical",
            "High",
            "Moderate",
        ],
        default="Normal",
    )

    latest["is_anomaly"] = (
        latest["abs_z_score"]
        >= MODERATE_Z_THRESHOLD
    )

    # -------------------------------------------------------------------------
    # Business interpretation
    # -------------------------------------------------------------------------

    latest["recommended_action"] = np.select(
        [
            (
                latest["anomaly_severity"]
                == "Critical"
            )
            & (
                latest["anomaly_direction"]
                == "Demand Spike"
            ),

            (
                latest["anomaly_severity"]
                == "High"
            )
            & (
                latest["anomaly_direction"]
                == "Demand Spike"
            ),

            (
                latest["anomaly_severity"]
                == "Critical"
            )
            & (
                latest["anomaly_direction"]
                == "Demand Drop"
            ),

            (
                latest["anomaly_severity"]
                == "High"
            )
            & (
                latest["anomaly_direction"]
                == "Demand Drop"
            ),

            latest["is_anomaly"],
        ],
        [
            "Immediate supply review; assess stockout exposure",
            "Review replenishment and near-term inventory position",
            "Review forecast; assess excess inventory exposure",
            "Review demand forecast and reduce replenishment risk",
            "Monitor demand deviation",
        ],
        default="No immediate action",
    )

    return df, latest


# =============================================================================
# FORECAST VS HISTORICAL BASELINE
# =============================================================================

def analyze_future_forecast(
    forecasts,
    demand,
):
    """
    Compare the 2027 forecast with recent historical demand.

    This is NOT anomaly detection against actual 2027 demand.
    It is a forecast plausibility / change analysis.
    """

    print_section(
        "ANALYZING FUTURE FORECAST AGAINST HISTORICAL BASELINE"
    )

    historical = (
        demand.sort_values(
            [
                "product_id",
                "date",
            ]
        )
        .groupby(
            "product_id",
            as_index=False,
        )
        .tail(56)
    )

    recent_stats = (
        historical.groupby(
            "product_id"
        )["actual"]
        .agg(
            recent_mean_56="mean",
            recent_std_56="std",
            recent_total_56="sum",
        )
        .reset_index()
    )

    recent_28 = (
        historical
        .sort_values(
            [
                "product_id",
                "date",
            ]
        )
        .groupby(
            "product_id",
            as_index=False,
        )
        .tail(28)
        .groupby("product_id")["actual"]
        .mean()
        .reset_index(
            name="recent_mean_28"
        )
    )

    stats = recent_stats.merge(
        recent_28,
        on="product_id",
        how="left",
    )

    forecast_summary = (
        forecasts.groupby(
            "product_id"
        )["forecast"]
        .agg(
            forecast_mean_30="mean",
            forecast_total_30="sum",
            forecast_min_30="min",
            forecast_max_30="max",
        )
        .reset_index()
    )

    comparison = forecast_summary.merge(
        stats,
        on="product_id",
        how="left",
    )

    comparison["forecast_vs_recent_pct"] = (
        safe_divide(
            comparison["forecast_mean_30"]
            - comparison["recent_mean_28"],
            comparison["recent_mean_28"].clip(
                lower=MIN_DEMAND_FOR_PERCENT
            ),
        )
    )

    comparison["forecast_change_abs"] = (
        comparison["forecast_mean_30"]
        - comparison["recent_mean_28"]
    )

    comparison["forecast_change_direction"] = np.where(
        comparison["forecast_vs_recent_pct"] > 0,
        "Forecast Increase",
        np.where(
            comparison["forecast_vs_recent_pct"] < 0,
            "Forecast Decrease",
            "Stable",
        ),
    )

    comparison["forecast_change_severity"] = np.select(
        [
            abs(
                comparison["forecast_vs_recent_pct"]
            ) >= 0.50,

            abs(
                comparison["forecast_vs_recent_pct"]
            ) >= 0.30,

            abs(
                comparison["forecast_vs_recent_pct"]
            ) >= 0.20,
        ],
        [
            "Critical",
            "High",
            "Moderate",
        ],
        default="Normal",
    )

    return comparison


# =============================================================================
# NETWORK SUMMARY
# =============================================================================

def create_network_summary(
    anomalies,
    forecast_comparison,
):
    """
    Create executive-level anomaly summary.
    """

    print_section(
        "BUILDING ANOMALY SUMMARY"
    )

    total_products = len(
        anomalies
    )

    anomalous_products = int(
        anomalies["is_anomaly"].sum()
    )

    critical = int(
        (
            anomalies["anomaly_severity"]
            == "Critical"
        ).sum()
    )

    high = int(
        (
            anomalies["anomaly_severity"]
            == "High"
        ).sum()
    )

    moderate = int(
        (
            anomalies["anomaly_severity"]
            == "Moderate"
        ).sum()
    )

    demand_spikes = int(
        (
            anomalies["anomaly_direction"]
            == "Demand Spike"
        ).sum()
    )

    demand_drops = int(
        (
            anomalies["anomaly_direction"]
            == "Demand Drop"
        ).sum()
    )

    forecast_increases = int(
        (
            forecast_comparison[
                "forecast_change_direction"
            ]
            == "Forecast Increase"
        ).sum()
    )

    forecast_decreases = int(
        (
            forecast_comparison[
                "forecast_change_direction"
            ]
            == "Forecast Decrease"
        ).sum()
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "Products monitored",
                "value": total_products,
            },
            {
                "metric": "Products with anomalies",
                "value": anomalous_products,
            },
            {
                "metric": "Critical anomalies",
                "value": critical,
            },
            {
                "metric": "High anomalies",
                "value": high,
            },
            {
                "metric": "Moderate anomalies",
                "value": moderate,
            },
            {
                "metric": "Demand spikes",
                "value": demand_spikes,
            },
            {
                "metric": "Demand drops",
                "value": demand_drops,
            },
            {
                "metric": "Products with forecast increases",
                "value": forecast_increases,
            },
            {
                "metric": "Products with forecast decreases",
                "value": forecast_decreases,
            },
        ]
    )

    return summary


# =============================================================================
# ANOMALY RANKING
# =============================================================================

def create_anomaly_ranking(anomalies):
    """
    Rank anomalies by business importance.

    Magnitude alone is not enough; a 3x deviation on a product
    with tiny demand should not necessarily outrank a major
    deviation on a high-volume product.
    """

    ranked = anomalies.copy()

    ranked["business_impact_score"] = (
        ranked["abs_z_score"]
        * np.log1p(
            ranked["actual"].clip(
                lower=0
            )
        )
    )

    severity_weight = {
        "Critical": 4,
        "High": 3,
        "Moderate": 2,
        "Normal": 0,
    }

    ranked["severity_weight"] = (
        ranked["anomaly_severity"]
        .map(severity_weight)
        .fillna(0)
    )

    ranked["priority_score"] = (
        ranked["business_impact_score"]
        * ranked["severity_weight"]
    )

    ranked = ranked.sort_values(
        [
            "priority_score",
            "abs_z_score",
        ],
        ascending=False,
    )

    ranked["priority_rank"] = (
        np.arange(len(ranked))
        + 1
    )

    return ranked


# =============================================================================
# OUTPUTS
# =============================================================================

def save_outputs(
    historical_residuals,
    baselines,
    detailed_analysis,
    anomalies,
    forecast_comparison,
    summary,
):
    print_section(
        "WRITING ANOMALY OUTPUTS"
    )

    residual_path = (
        OUTPUT_DIR
        / "historical_forecast_residuals.csv"
    )

    baseline_path = (
        OUTPUT_DIR
        / "residual_baselines.csv"
    )

    detailed_path = (
        OUTPUT_DIR
        / "demand_anomaly_analysis.csv"
    )

    anomaly_path = (
        OUTPUT_DIR
        / "demand_anomalies.csv"
    )

    forecast_path = (
        OUTPUT_DIR
        / "forecast_change_analysis.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "anomaly_summary.csv"
    )

    historical_residuals.to_csv(
        residual_path,
        index=False,
    )

    baselines.to_csv(
        baseline_path,
        index=False,
    )

    detailed_analysis.to_csv(
        detailed_path,
        index=False,
    )

    anomalies.to_csv(
        anomaly_path,
        index=False,
    )

    forecast_comparison.to_csv(
        forecast_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"Historical residuals:\n"
        f"{residual_path}"
    )

    print(
        f"\nResidual baselines:\n"
        f"{baseline_path}"
    )

    print(
        f"\nDetailed analysis:\n"
        f"{detailed_path}"
    )

    print(
        f"\nDemand anomalies:\n"
        f"{anomaly_path}"
    )

    print(
        f"\nForecast change analysis:\n"
        f"{forecast_path}"
    )

    print(
        f"\nSummary:\n"
        f"{summary_path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    demand, forecasts = load_data()

    # -------------------------------------------------------------------------
    # Historical residuals
    # -------------------------------------------------------------------------

    historical_residuals = (
        build_historical_residuals(
            demand=demand,
            validation_horizon=VALIDATION_HORIZON,
            backtest_windows=BACKTEST_WINDOWS,
        )
    )

    # -------------------------------------------------------------------------
    # Residual baseline
    # -------------------------------------------------------------------------

    baselines = (
        build_residual_baselines(
            historical_residuals
        )
    )

    # -------------------------------------------------------------------------
    # Current anomalies
    # -------------------------------------------------------------------------

    detailed_analysis, anomalies = (
        detect_current_anomalies(
            demand=demand,
            baselines=baselines,
        )
    )

    # -------------------------------------------------------------------------
    # Future forecast analysis
    # -------------------------------------------------------------------------

    forecast_comparison = (
        analyze_future_forecast(
            forecasts=forecasts,
            demand=demand,
        )
    )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    anomalies = create_anomaly_ranking(
        anomalies
    )

    # -------------------------------------------------------------------------
    # Network summary
    # -------------------------------------------------------------------------

    summary = create_network_summary(
        anomalies=anomalies,
        forecast_comparison=forecast_comparison,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_outputs(
        historical_residuals=historical_residuals,
        baselines=baselines,
        detailed_analysis=detailed_analysis,
        anomalies=anomalies,
        forecast_comparison=forecast_comparison,
        summary=summary,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print_section(
        "ANOMALY DETECTION SUMMARY"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("-" * 80)

    print(
        f"Products monitored: "
        f"{len(anomalies):,}"
    )

    print(
        f"Anomalous products: "
        f"{int(anomalies['is_anomaly'].sum()):,}"
    )

    print(
        f"Critical anomalies: "
        f"{int((anomalies['anomaly_severity'] == 'Critical').sum()):,}"
    )

    print(
        f"High anomalies: "
        f"{int((anomalies['anomaly_severity'] == 'High').sum()):,}"
    )

    print(
        f"Moderate anomalies: "
        f"{int((anomalies['anomaly_severity'] == 'Moderate').sum()):,}"
    )

    print()
    print(
        "TOP 20 ANOMALIES"
    )
    print("-" * 80)

    top_columns = [
        "priority_rank",
        "product_id",
        "date",
        "actual",
        "current_baseline",
        "current_residual",
        "current_deviation_pct",
        "current_z_score",
        "anomaly_direction",
        "anomaly_severity",
        "recommended_action",
    ]

    available_columns = [
        column
        for column in top_columns
        if column in anomalies.columns
    ]

    print(
        anomalies[
            available_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "SUPPLY CHAIN DEMAND ANOMALY DETECTION COMPLETE"
    )
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()