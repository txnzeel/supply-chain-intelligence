from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    SimpleExpSmoothing,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEMAND_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_demand.csv"
)

FORECAST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasts"
)

FORECAST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Forecast horizon
FORECAST_HORIZON = 30

# Number of validation periods
VALIDATION_WINDOWS = 3

# Validation horizon
VALIDATION_HORIZON = 30

# Minimum amount of history required
MIN_HISTORY = 120

# Weekly seasonal period
SEASONAL_PERIOD = 7

# Minimum non-zero observations for Croston
MIN_NONZERO_DEMAND = 5


# ============================================================
# GLOBAL RESULTS
# ============================================================

metrics_results = []
selection_results = []
forecast_results = []


# ============================================================
# LOGGING
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# METRICS
# ============================================================

def mae(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    return float(
        np.mean(
            np.abs(actual - forecast)
        )
    )


def rmse(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (actual - forecast) ** 2
            )
        )
    )


def wape(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.sum(
            np.abs(actual - forecast)
        )
        / denominator
    )


def bias(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.sum(forecast - actual)
        / denominator
    )


def mean_absolute_percentage_error(
    actual,
    forecast,
):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    mask = actual != 0

    if not np.any(mask):
        return 0.0

    return float(
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - forecast[mask]
                )
                / actual[mask]
            )
        )
    )


# ============================================================
# DEMAND CLASSIFICATION
# ============================================================

def classify_demand_series(series):
    """
    Additional behavioral classification used by the
    forecasting engine.

    This does not replace the demand_class supplied in
    products/daily_demand.

    Used to identify intermittent demand.
    """

    values = np.asarray(
        series,
        dtype=float,
    )

    if len(values) == 0:
        return "Unknown"

    nonzero = values[values > 0]

    zero_ratio = (
        np.mean(values == 0)
    )

    if (
        len(nonzero) >= MIN_NONZERO_DEMAND
        and zero_ratio >= 0.30
    ):
        return "Intermittent"

    return "Continuous"


# ============================================================
# FORECASTING MODELS
# ============================================================

def forecast_naive(
    train,
    horizon,
):
    """
    Last-value forecast.
    """

    if len(train) == 0:
        return np.zeros(horizon)

    value = float(train.iloc[-1])

    return np.repeat(
        max(value, 0),
        horizon,
    )


def forecast_mean(
    train,
    horizon,
    window=28,
):
    """
    Rolling mean forecast.
    """

    if len(train) == 0:
        return np.zeros(horizon)

    window = min(
        window,
        len(train),
    )

    value = float(
        train.iloc[-window:].mean()
    )

    return np.repeat(
        max(value, 0),
        horizon,
    )


def forecast_seasonal_naive(
    train,
    horizon,
    season_length=7,
):
    """
    Seasonal naive forecast.

    Repeats the most recent seasonal pattern.
    """

    if len(train) == 0:
        return np.zeros(horizon)

    if len(train) < season_length:
        return forecast_naive(
            train,
            horizon,
        )

    pattern = np.asarray(
        train.iloc[-season_length:],
        dtype=float,
    )

    repetitions = int(
        np.ceil(
            horizon
            / season_length
        )
    )

    forecast = np.tile(
        pattern,
        repetitions,
    )

    return np.maximum(
        forecast[:horizon],
        0,
    )


def forecast_ses(
    train,
    horizon,
):
    """
    Simple Exponential Smoothing.
    """

    if len(train) < 2:
        return forecast_naive(
            train,
            horizon,
        )

    try:

        model = (
            SimpleExpSmoothing(
                train.astype(float),
                initialization_method="estimated",
            )
            .fit(
                optimized=True
            )
        )

        forecast = model.forecast(
            horizon
        )

        return np.maximum(
            np.asarray(
                forecast,
                dtype=float,
            ),
            0,
        )

    except Exception:

        return forecast_naive(
            train,
            horizon,
        )


def forecast_holt(
    train,
    horizon,
):
    """
    Holt's trend method.
    """

    if len(train) < 3:
        return forecast_naive(
            train,
            horizon,
        )

    try:

        model = (
            ExponentialSmoothing(
                train.astype(float),
                trend="add",
                damped_trend=True,
                initialization_method="estimated",
            )
            .fit(
                optimized=True
            )
        )

        forecast = model.forecast(
            horizon
        )

        return np.maximum(
            np.asarray(
                forecast,
                dtype=float,
            ),
            0,
        )

    except Exception:

        return forecast_naive(
            train,
            horizon,
        )


def forecast_holt_weekly(
    train,
    horizon,
):
    """
    Holt-Winters model with weekly seasonality.

    Used only when sufficient history exists.
    """

    if len(train) < 2 * SEASONAL_PERIOD:
        return forecast_holt(
            train,
            horizon,
        )

    try:

        model = (
            ExponentialSmoothing(
                train.astype(float),
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=SEASONAL_PERIOD,
                initialization_method="estimated",
            )
            .fit(
                optimized=True
            )
        )

        forecast = model.forecast(
            horizon
        )

        return np.maximum(
            np.asarray(
                forecast,
                dtype=float,
            ),
            0,
        )

    except Exception:

        return forecast_seasonal_naive(
            train,
            horizon,
            SEASONAL_PERIOD,
        )


# ============================================================
# CROSTON
# ============================================================

def croston_forecast(
    train,
    horizon,
    alpha=0.1,
):
    """
    Croston's method for intermittent demand.

    Estimates:
        - demand size
        - interval between non-zero demands

    Forecast is constant over the horizon.
    """

    values = np.asarray(
        train,
        dtype=float,
    )

    values = np.maximum(
        values,
        0,
    )

    nonzero_indices = np.flatnonzero(
        values > 0
    )

    if len(nonzero_indices) == 0:
        return np.zeros(horizon)

    if len(nonzero_indices) == 1:
        demand_estimate = (
            values[
                nonzero_indices[0]
            ]
        )

        interval_estimate = max(
            len(values),
            1,
        )

        forecast_value = (
            demand_estimate
            / interval_estimate
        )

        return np.repeat(
            forecast_value,
            horizon,
        )

    first_index = nonzero_indices[0]

    demand_estimate = float(
        values[first_index]
    )

    interval_estimate = float(
        nonzero_indices[1]
        - nonzero_indices[0]
    )

    previous_index = (
        nonzero_indices[0]
    )

    for index in nonzero_indices[1:]:

        demand = float(
            values[index]
        )

        interval = float(
            index
            - previous_index
        )

        demand_estimate = (
            alpha * demand
            + (1 - alpha)
            * demand_estimate
        )

        interval_estimate = (
            alpha * interval
            + (1 - alpha)
            * interval_estimate
        )

        previous_index = index

    if interval_estimate <= 0:
        interval_estimate = 1

    forecast_value = (
        demand_estimate
        / interval_estimate
    )

    return np.repeat(
        max(forecast_value, 0),
        horizon,
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_FUNCTIONS = {
    "Naive": forecast_naive,
    "RollingMean28": lambda x, h: forecast_mean(
        x,
        h,
        window=28,
    ),
    "SeasonalNaive7": lambda x, h: forecast_seasonal_naive(
        x,
        h,
        season_length=7,
    ),
    "SeasonalNaive28": lambda x, h: forecast_seasonal_naive(
        x,
        h,
        season_length=28,
    ),
    "SES": forecast_ses,
    "Holt": forecast_holt,
    "HoltWinters7": forecast_holt_weekly,
    "Croston": croston_forecast,
}


# ============================================================
# MODEL AVAILABILITY
# ============================================================

def get_candidate_models(
    demand_class,
    behavioral_class,
):
    """
    Choose reasonable candidate models.

    We don't run every model blindly for every product.
    """

    if behavioral_class == "Intermittent":

        return [
            "Naive",
            "RollingMean28",
            "Croston",
            "SeasonalNaive7",
        ]

    if demand_class == "Stable":

        return [
            "Naive",
            "RollingMean28",
            "SeasonalNaive7",
            "SES",
            "HoltWinters7",
        ]

    if demand_class == "Seasonal":

        return [
            "Naive",
            "SeasonalNaive7",
            "SeasonalNaive28",
            "SES",
            "HoltWinters7",
        ]

    if demand_class == "Trending":

        return [
            "Naive",
            "RollingMean28",
            "SES",
            "Holt",
            "HoltWinters7",
        ]

    if demand_class == "Volatile":

        return [
            "Naive",
            "RollingMean28",
            "SeasonalNaive7",
            "SES",
        ]

    return [
        "Naive",
        "RollingMean28",
        "SeasonalNaive7",
        "SES",
        "Holt",
        "HoltWinters7",
    ]


# ============================================================
# VALIDATION SPLITS
# ============================================================

def create_validation_splits(
    series,
    horizon=VALIDATION_HORIZON,
    windows=VALIDATION_WINDOWS,
):
    """
    Creates rolling chronological validation windows.

    Example:

    TRAIN ---------------- VALID
    TRAIN ---------------------- VALID
    TRAIN ---------------------------- VALID
    """

    n = len(series)

    splits = []

    required = (
        horizon
        * windows
    )

    if n < (
        MIN_HISTORY
        + required
    ):
        return splits

    for window_number in range(
        windows,
        0,
        -1,
    ):

        validation_end = (
            n
            - (
                (window_number - 1)
                * horizon
            )
        )

        validation_start = (
            validation_end
            - horizon
        )

        train_end = validation_start

        if train_end < MIN_HISTORY:
            continue

        train = series.iloc[
            :train_end
        ]

        validation = series.iloc[
            validation_start:
            validation_end
        ]

        splits.append(
            {
                "window": (
                    windows
                    - window_number
                    + 1
                ),
                "train": train,
                "validation": validation,
            }
        )

    return splits


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    product_id,
    demand_class,
    behavioral_class,
    model_name,
    series,
):
    """
    Evaluate one model using rolling-origin validation.
    """

    splits = create_validation_splits(
        series
    )

    if not splits:
        return None

    model_function = (
        MODEL_FUNCTIONS[
            model_name
        ]
    )

    window_results = []

    for split in splits:

        train = split["train"]
        actual = split["validation"]

        try:

            predicted = model_function(
                train,
                len(actual),
            )

            predicted = np.asarray(
                predicted,
                dtype=float,
            )

            if len(predicted) != len(actual):

                predicted = np.resize(
                    predicted,
                    len(actual),
                )

            predicted = np.maximum(
                predicted,
                0,
            )

            actual_values = (
                actual.to_numpy(
                    dtype=float
                )
            )

            window_results.append(
                {
                    "product_id": product_id,
                    "demand_class": demand_class,
                    "behavioral_class": behavioral_class,
                    "model": model_name,
                    "validation_window": split[
                        "window"
                    ],
                    "mae": mae(
                        actual_values,
                        predicted,
                    ),
                    "rmse": rmse(
                        actual_values,
                        predicted,
                    ),
                    "wape": wape(
                        actual_values,
                        predicted,
                    ),
                    "mape": mean_absolute_percentage_error(
                        actual_values,
                        predicted,
                    ),
                    "bias": bias(
                        actual_values,
                        predicted,
                    ),
                }
            )

        except Exception as exc:

            window_results.append(
                {
                    "product_id": product_id,
                    "demand_class": demand_class,
                    "behavioral_class": behavioral_class,
                    "model": model_name,
                    "validation_window": split[
                        "window"
                    ],
                    "mae": np.nan,
                    "rmse": np.nan,
                    "wape": np.nan,
                    "mape": np.nan,
                    "bias": np.nan,
                }
            )

    if not window_results:
        return None

    return window_results


# ============================================================
# MODEL SELECTION
# ============================================================

def select_best_model(
    product_id,
    demand_class,
    behavioral_class,
    series,
):
    """
    Select the model with the best average WAPE.

    Tie-breakers:
        1. WAPE
        2. MAE
        3. absolute bias
    """

    candidates = get_candidate_models(
        demand_class,
        behavioral_class,
    )

    all_results = []

    for model_name in candidates:

        results = evaluate_model(
            product_id,
            demand_class,
            behavioral_class,
            model_name,
            series,
        )

        if results:
            all_results.extend(
                results
            )

    if not all_results:
        return (
            "Naive",
            pd.DataFrame(),
        )

    metrics_df = pd.DataFrame(
        all_results
    )

    summary = (
        metrics_df
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            mean_wape=("wape", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            mean_bias=("bias", "mean"),
            validation_windows=(
                "validation_window",
                "count",
            ),
        )
    )

    summary[
        "abs_bias"
    ] = summary[
        "mean_bias"
    ].abs()

    summary = summary.sort_values(
        [
            "mean_wape",
            "mean_mae",
            "abs_bias",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )

    best_model = (
        summary.iloc[0]["model"]
    )

    return (
        best_model,
        metrics_df,
    )


# ============================================================
# FORECAST CONFIDENCE
# ============================================================

def calculate_forecast_interval(
    series,
    forecast,
):
    """
    Simple empirical forecast interval.

    Uses historical residual volatility around a
    rolling mean.

    This is intentionally transparent rather than
    pretending to provide mathematically exact
    probabilistic intervals.
    """

    values = np.asarray(
        series,
        dtype=float,
    )

    if len(values) < 30:

        std = (
            np.std(values)
            if len(values)
            else 0
        )

    else:

        rolling_mean = (
            pd.Series(values)
            .rolling(
                28,
                min_periods=7,
            )
            .mean()
        )

        residuals = (
            pd.Series(values)
            - rolling_mean
        )

        std = float(
            residuals
            .dropna()
            .std()
        )

    if not np.isfinite(std):
        std = 0.0

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    lower = np.maximum(
        forecast
        - 1.96 * std,
        0,
    )

    upper = np.maximum(
        forecast
        + 1.96 * std,
        0,
    )

    return lower, upper


# ============================================================
# FORECAST ONE PRODUCT
# ============================================================

def forecast_product(
    product_id,
    product_df,
):
    """
    Select the best model and produce a 30-day forecast.
    """

    product_df = (
        product_df
        .sort_values("date")
        .copy()
    )

    series = (
        product_df[
            "actual_demand"
        ]
        .astype(float)
        .reset_index(drop=True)
    )

    demand_class = str(
        product_df[
            "demand_class"
        ].iloc[0]
    )

    behavioral_class = (
        classify_demand_series(
            series
        )
    )

    if len(series) < MIN_HISTORY:

        best_model = "Naive"

        metrics_df = pd.DataFrame()

    else:

        best_model, metrics_df = (
            select_best_model(
                product_id,
                demand_class,
                behavioral_class,
                series,
            )
        )

    model_function = (
        MODEL_FUNCTIONS[
            best_model
        ]
    )

    try:

        forecast = model_function(
            series,
            FORECAST_HORIZON,
        )

    except Exception:

        best_model = "Naive"

        forecast = forecast_naive(
            series,
            FORECAST_HORIZON,
        )

    forecast = np.maximum(
        np.asarray(
            forecast,
            dtype=float,
        ),
        0,
    )

    lower, upper = (
        calculate_forecast_interval(
            series,
            forecast,
        )
    )

    last_date = (
        product_df[
            "date"
        ].max()
    )

    future_dates = pd.date_range(
        start=(
            last_date
            + pd.Timedelta(days=1)
        ),
        periods=FORECAST_HORIZON,
        freq="D",
    )

    rows = []

    for index, forecast_date in enumerate(
        future_dates
    ):

        rows.append(
            {
                "product_id": product_id,
                "forecast_date": forecast_date,
                "horizon_day": index + 1,
                "forecast_demand": float(
                    forecast[index]
                ),
                "lower_bound": float(
                    lower[index]
                ),
                "upper_bound": float(
                    upper[index]
                ),
                "model": best_model,
                "demand_class": demand_class,
                "behavioral_class": behavioral_class,
            }
        )

    forecast_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Selection summary
    # --------------------------------------------------------

    if metrics_df.empty:

        selection = {
            "product_id": product_id,
            "demand_class": demand_class,
            "behavioral_class": behavioral_class,
            "selected_model": best_model,
            "validation_wape": np.nan,
            "validation_mae": np.nan,
            "validation_rmse": np.nan,
            "validation_mape": np.nan,
            "validation_bias": np.nan,
            "validation_windows": 0,
            "history_days": len(series),
            "total_demand": float(
                series.sum()
            ),
            "average_daily_demand": float(
                series.mean()
            ),
            "demand_std": float(
                series.std()
            ),
            "zero_demand_rate": float(
                np.mean(series == 0)
            ),
        }

    else:

        model_summary = (
            metrics_df
            .groupby(
                "model",
                as_index=False,
            )
            .agg(
                validation_wape=(
                    "wape",
                    "mean",
                ),
                validation_mae=(
                    "mae",
                    "mean",
                ),
                validation_rmse=(
                    "rmse",
                    "mean",
                ),
                validation_mape=(
                    "mape",
                    "mean",
                ),
                validation_bias=(
                    "bias",
                    "mean",
                ),
                validation_windows=(
                    "validation_window",
                    "count",
                ),
            )
        )

        selected = (
            model_summary[
                model_summary[
                    "model"
                ]
                == best_model
            ]
            .iloc[0]
        )

        selection = {
            "product_id": product_id,
            "demand_class": demand_class,
            "behavioral_class": behavioral_class,
            "selected_model": best_model,
            "validation_wape": float(
                selected[
                    "validation_wape"
                ]
            ),
            "validation_mae": float(
                selected[
                    "validation_mae"
                ]
            ),
            "validation_rmse": float(
                selected[
                    "validation_rmse"
                ]
            ),
            "validation_mape": float(
                selected[
                    "validation_mape"
                ]
            ),
            "validation_bias": float(
                selected[
                    "validation_bias"
                ]
            ),
            "validation_windows": int(
                selected[
                    "validation_windows"
                ]
            ),
            "history_days": len(series),
            "total_demand": float(
                series.sum()
            ),
            "average_daily_demand": float(
                series.mean()
            ),
            "demand_std": float(
                series.std()
            ),
            "zero_demand_rate": float(
                np.mean(series == 0)
            ),
        }

    return (
        forecast_df,
        selection,
        metrics_df,
    )


# ============================================================
# FORECAST SUMMARY
# ============================================================

def create_forecast_summary(
    forecast_df,
    selection_df,
):
    """
    Creates executive-level forecasting summary.
    """

    if forecast_df.empty:
        return pd.DataFrame()

    summary = (
        forecast_df
        .groupby(
            "product_id",
            as_index=False,
        )
        .agg(
            forecast_30d=(
                "forecast_demand",
                "sum",
            ),
            average_forecast_daily=(
                "forecast_demand",
                "mean",
            ),
            forecast_min=(
                "forecast_demand",
                "min",
            ),
            forecast_max=(
                "forecast_demand",
                "max",
            ),
            uncertainty_lower_30d=(
                "lower_bound",
                "sum",
            ),
            uncertainty_upper_30d=(
                "upper_bound",
                "sum",
            ),
        )
    )

    summary = summary.merge(
        selection_df[
            [
                "product_id",
                "demand_class",
                "behavioral_class",
                "selected_model",
                "validation_wape",
                "validation_mae",
                "validation_rmse",
                "validation_bias",
                "average_daily_demand",
                "demand_std",
                "zero_demand_rate",
            ]
        ],
        on="product_id",
        how="left",
    )

    summary[
        "forecast_growth_vs_history"
    ] = np.where(
        summary[
            "average_daily_demand"
        ] > 0,
        (
            summary[
                "average_forecast_daily"
            ]
            / summary[
                "average_daily_demand"
            ]
            - 1
        ),
        np.nan,
    )

    summary[
        "forecast_uncertainty"
    ] = (
        summary[
            "uncertainty_upper_30d"
        ]
        - summary[
            "uncertainty_lower_30d"
        ]
    )

    summary[
        "forecast_confidence"
    ] = np.select(
        [
            summary[
                "validation_wape"
            ]
            <= 0.15,
            summary[
                "validation_wape"
            ]
            <= 0.30,
            summary[
                "validation_wape"
            ]
            <= 0.50,
        ],
        [
            "High",
            "Medium",
            "Low",
        ],
        default="Very Low",
    )

    return summary


# ============================================================
# MAIN FORECASTING ENGINE
# ============================================================

def main():

    print_header(
        "SUPPLY CHAIN DEMAND FORECASTING ENGINE"
    )

    print(
        f"Input:\n{DEMAND_PATH}"
    )

    print(
        f"\nForecast horizon: "
        f"{FORECAST_HORIZON} days"
    )

    print(
        f"Validation windows: "
        f"{VALIDATION_WINDOWS}"
    )

    print(
        f"Validation horizon: "
        f"{VALIDATION_HORIZON} days"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print_header(
        "LOADING DEMAND DATA"
    )

    df = pd.read_csv(
        DEMAND_PATH,
        parse_dates=["date"],
    )

    required_columns = [
        "product_id",
        "date",
        "actual_demand",
        "demand_class",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df = df[
        required_columns
    ].copy()

    df[
        "actual_demand"
    ] = pd.to_numeric(
        df[
            "actual_demand"
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "product_id",
            "date",
            "actual_demand",
        ]
    )

    df[
        "actual_demand"
    ] = np.maximum(
        df[
            "actual_demand"
        ],
        0,
    )

    df = (
        df
        .sort_values(
            [
                "product_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Products: "
        f"{df['product_id'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"-> "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Process products
    # --------------------------------------------------------

    print_header(
        "PRODUCT-LEVEL FORECASTING"
    )

    product_groups = (
        df.groupby(
            "product_id",
            sort=False,
        )
    )

    total_products = (
        df["product_id"]
        .nunique()
    )

    processed = 0

    for product_id, product_df in product_groups:

        processed += 1

        try:

            (
                product_forecast,
                selection,
                product_metrics,
            ) = forecast_product(
                product_id,
                product_df,
            )

            forecast_results.append(
                product_forecast
            )

            selection_results.append(
                selection
            )

            if (
                product_metrics is not None
                and not product_metrics.empty
            ):

                metrics_results.append(
                    product_metrics
                )

        except Exception as exc:

            print(
                f"\nWARNING: "
                f"{product_id} failed: "
                f"{exc}"
            )

            # ------------------------------------------------
            # Fail-safe naive forecast
            # ------------------------------------------------

            series = (
                product_df[
                    "actual_demand"
                ]
                .astype(float)
            )

            forecast = forecast_naive(
                series,
                FORECAST_HORIZON,
            )

            lower, upper = (
                calculate_forecast_interval(
                    series,
                    forecast,
                )
            )

            last_date = (
                product_df[
                    "date"
                ].max()
            )

            future_dates = pd.date_range(
                start=(
                    last_date
                    + pd.Timedelta(days=1)
                ),
                periods=FORECAST_HORIZON,
                freq="D",
            )

            fallback_rows = []

            for index, date in enumerate(
                future_dates
            ):

                fallback_rows.append(
                    {
                        "product_id": product_id,
                        "forecast_date": date,
                        "horizon_day": index + 1,
                        "forecast_demand": float(
                            forecast[index]
                        ),
                        "lower_bound": float(
                            lower[index]
                        ),
                        "upper_bound": float(
                            upper[index]
                        ),
                        "model": "Naive",
                        "demand_class": str(
                            product_df[
                                "demand_class"
                            ].iloc[0]
                        ),
                        "behavioral_class":
                            classify_demand_series(
                                series
                            ),
                    }
                )

            forecast_results.append(
                pd.DataFrame(
                    fallback_rows
                )
            )

            selection_results.append(
                {
                    "product_id": product_id,
                    "demand_class": str(
                        product_df[
                            "demand_class"
                        ].iloc[0]
                    ),
                    "behavioral_class":
                        classify_demand_series(
                            series
                        ),
                    "selected_model": "Naive",
                    "validation_wape": np.nan,
                    "validation_mae": np.nan,
                    "validation_rmse": np.nan,
                    "validation_mape": np.nan,
                    "validation_bias": np.nan,
                    "validation_windows": 0,
                    "history_days": len(series),
                    "total_demand": float(
                        series.sum()
                    ),
                    "average_daily_demand": float(
                        series.mean()
                    ),
                    "demand_std": float(
                        series.std()
                    ),
                    "zero_demand_rate": float(
                        np.mean(series == 0)
                    ),
                }
            )

        if (
            processed % 100 == 0
            or processed == total_products
        ):

            print(
                f"Processed "
                f"{processed:,}/"
                f"{total_products:,} products"
            )

    # --------------------------------------------------------
    # Combine outputs
    # --------------------------------------------------------

    print_header(
        "BUILDING FORECAST OUTPUTS"
    )

    if forecast_results:

        forecast_df = pd.concat(
            forecast_results,
            ignore_index=True,
        )

    else:

        forecast_df = pd.DataFrame()

    if selection_results:

        selection_df = pd.DataFrame(
            selection_results
        )

    else:

        selection_df = pd.DataFrame()

    if metrics_results:

        metrics_df = pd.concat(
            metrics_results,
            ignore_index=True,
        )

    else:

        metrics_df = pd.DataFrame()

    # --------------------------------------------------------
    # Forecast summary
    # --------------------------------------------------------

    summary_df = create_forecast_summary(
        forecast_df,
        selection_df,
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    forecast_path = (
        FORECAST_DIR
        / "demand_forecasts.csv"
    )

    metrics_path = (
        FORECAST_DIR
        / "forecast_metrics.csv"
    )

    selection_path = (
        FORECAST_DIR
        / "model_selection.csv"
    )

    summary_path = (
        FORECAST_DIR
        / "forecast_summary.csv"
    )

    forecast_df.to_csv(
        forecast_path,
        index=False,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    selection_df.to_csv(
        selection_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    print_header(
        "FORECASTING SUMMARY"
    )

    print(
        f"Products forecast: "
        f"{len(selection_df):,}"
    )

    print(
        f"Forecast rows: "
        f"{len(forecast_df):,}"
    )

    print(
        f"Expected: "
        f"{total_products * FORECAST_HORIZON:,}"
    )

    if not selection_df.empty:

        print()

        print(
            "SELECTED MODELS:"
        )

        model_counts = (
            selection_df[
                "selected_model"
            ]
            .value_counts()
        )

        print(
            model_counts.to_string()
        )

        print()

        print(
            "DEMAND BEHAVIOR:"
        )

        behavior_counts = (
            selection_df[
                "behavioral_class"
            ]
            .value_counts()
        )

        print(
            behavior_counts.to_string()
        )

        valid_wape = (
            selection_df[
                "validation_wape"
            ]
            .dropna()
        )

        if not valid_wape.empty:

            print()

            print(
                "VALIDATION PERFORMANCE:"
            )

            print(
                f"Mean WAPE: "
                f"{valid_wape.mean():.2%}"
            )

            print(
                f"Median WAPE: "
                f"{valid_wape.median():.2%}"
            )

            print(
                f"Products with WAPE ≤ 20%: "
                f"{(
                    valid_wape <= 0.20
                ).mean():.1%}"
            )

            print(
                f"Products with WAPE ≤ 30%: "
                f"{(
                    valid_wape <= 0.30
                ).mean():.1%}"
            )

    # --------------------------------------------------------
    # Forecast demand
    # --------------------------------------------------------

    if not forecast_df.empty:

        total_forecast = (
            forecast_df[
                "forecast_demand"
            ].sum()
        )

        total_lower = (
            forecast_df[
                "lower_bound"
            ].sum()
        )

        total_upper = (
            forecast_df[
                "upper_bound"
            ].sum()
        )

        print()

        print(
            "30-DAY NETWORK FORECAST:"
        )

        print(
            f"Expected demand: "
            f"{total_forecast:,.0f}"
        )

        print(
            f"Lower bound: "
            f"{total_lower:,.0f}"
        )

        print(
            f"Upper bound: "
            f"{total_upper:,.0f}"
        )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    print_header(
        "FORECAST OUTPUTS"
    )

    print(
        f"Demand forecasts:\n"
        f"{forecast_path}"
    )

    print(
        f"\nForecast metrics:\n"
        f"{metrics_path}"
    )

    print(
        f"\nModel selection:\n"
        f"{selection_path}"
    )

    print(
        f"\nForecast summary:\n"
        f"{summary_path}"
    )

    print()

    print("=" * 80)
    print(
        "FORECASTING ENGINE COMPLETE"
    )
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()