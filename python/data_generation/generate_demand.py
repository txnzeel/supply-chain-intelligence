import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

np.random.seed(42)

START_DATE = "2025-01-01"
END_DATE = "2026-12-31"

INTERMITTENT_DEMAND_PROBABILITY = 0.35

DEMAND_CLASS_CONFIG = {
    "Stable": {
        "trend_range": (-0.02, 0.02),
        "noise_std": 0.05,
        "seasonality_strength": 0.50,
    },

    "Trending": {
        "trend_range": (0.10, 0.30),
        "noise_std": 0.08,
        "seasonality_strength": 0.70,
    },

    "Seasonal": {
        "trend_range": (-0.05, 0.10),
        "noise_std": 0.06,
        "seasonality_strength": 1.00,
    },

    "Volatile": {
        "trend_range": (-0.10, 0.15),
        "noise_std": 0.20,
        "seasonality_strength": 0.50,
    },

    "Intermittent": {
        "trend_range": (-0.05, 0.05),
        "noise_std": 0.10,
        "seasonality_strength": 0.30,
    }
}


WEEKDAY_MULTIPLIERS = {
    0: 1.05,  # Monday
    1: 1.08,  # Tuesday
    2: 1.05,  # Wednesday
    3: 1.02,  # Thursday
    4: 1.15,  # Friday
    5: 0.90,  # Saturday
    6: 0.75,  # Sunday
}


EVENT_CONFIG = [
    {
        "event_id": "E001",
        "event_type": "Republic Sale",
        "start_date": "2025-01-20",
        "end_date": "2025-01-26",
        "category_multipliers": {
            "Electronics": 1.35,
            "Accessories": 1.25,
            "Personal Care": 1.15,
            "Home Appliances": 1.20,
            "Office Supplies": 1.05,
        },
    },

    {
        "event_id": "E002",
        "event_type": "Summer Sale",
        "start_date": "2025-05-01",
        "end_date": "2025-05-10",
        "category_multipliers": {
            "Electronics": 1.50,
            "Accessories": 1.30,
            "Personal Care": 1.20,
            "Home Appliances": 1.60,
            "Office Supplies": 1.10,
        },
    },

    {
        "event_id": "E003",
        "event_type": "Monsoon Slowdown",
        "start_date": "2025-07-15",
        "end_date": "2025-07-25",
        "category_multipliers": {
            "Electronics": 0.85,
            "Accessories": 0.90,
            "Personal Care": 0.95,
            "Home Appliances": 0.80,
            "Office Supplies": 0.90,
        },
    },

    {
        "event_id": "E004",
        "event_type": "Festival Sale",
        "start_date": "2025-10-10",
        "end_date": "2025-10-25",
        "category_multipliers": {
            "Electronics": 1.60,
            "Accessories": 1.40,
            "Personal Care": 1.25,
            "Home Appliances": 1.50,
            "Office Supplies": 1.15,
        },
    },

    {
        "event_id": "E005",
        "event_type": "Electronics Promo",
        "start_date": "2026-11-01",
        "end_date": "2026-11-15",
        "category_multipliers": {
            "Electronics": 1.80,
            "Accessories": 1.35,
            "Personal Care": 1.00,
            "Home Appliances": 1.20,
            "Office Supplies": 1.00,
        },
    },
]


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "products.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_demand.csv"
)


# =========================================================
# LOAD PRODUCTS
# =========================================================

products = pd.read_csv(PRODUCT_FILE)

required_columns = [
    "product_id",
    "category",
    "demand_class",
    "base_daily_demand",
]

missing_columns = [
    column
    for column in required_columns
    if column not in products.columns
]

assert not missing_columns, (
    f"Missing required columns: {missing_columns}"
)

assert products["product_id"].notna().all()
assert products["product_id"].is_unique

assert products["base_daily_demand"].notna().all()
assert (products["base_daily_demand"] > 0).all()

assert products["demand_class"].isin(
    DEMAND_CLASS_CONFIG.keys()
).all()


print(f"Products loaded: {len(products):,}")


# =========================================================
# PRODUCT PARAMETERS
# =========================================================

product_parameters = products[
    [
        "product_id",
        "category",
        "demand_class",
        "base_daily_demand",
    ]
].copy()


product_parameters["annual_growth_rate"] = (
    product_parameters["demand_class"]
    .apply(
        lambda demand_class:
        np.random.uniform(
            *DEMAND_CLASS_CONFIG[demand_class]["trend_range"]
        )
    )
)


product_parameters["noise_std"] = (
    product_parameters["demand_class"]
    .map(
        lambda demand_class:
        DEMAND_CLASS_CONFIG[demand_class]["noise_std"]
    )
)


product_parameters["seasonality_strength"] = (
    product_parameters["demand_class"]
    .map(
        lambda demand_class:
        DEMAND_CLASS_CONFIG[demand_class]["seasonality_strength"]
    )
)


# =========================================================
# CALENDAR
# =========================================================

dates = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq="D",
)

calendar = pd.DataFrame({
    "date": dates,
})


calendar["year"] = calendar["date"].dt.year
calendar["month"] = calendar["date"].dt.month
calendar["quarter"] = calendar["date"].dt.quarter
calendar["week"] = calendar["date"].dt.isocalendar().week.astype(int)
calendar["day_of_week"] = calendar["date"].dt.dayofweek
calendar["day_of_month"] = calendar["date"].dt.day

calendar["days_since_start"] = (
    calendar["date"] - calendar["date"].min()
).dt.days

calendar["is_weekend"] = (
    calendar["day_of_week"] >= 5
)

calendar["month_name"] = (
    calendar["date"].dt.month_name()
)


# =========================================================
# WEEKDAY SEASONALITY
# =========================================================

calendar["weekday_multiplier"] = (
    calendar["day_of_week"]
    .map(WEEKDAY_MULTIPLIERS)
)


assert calendar["weekday_multiplier"].notna().all()
assert (calendar["weekday_multiplier"] > 0).all()


# =========================================================
# ANNUAL SEASONALITY
# =========================================================

calendar["day_of_year"] = (
    calendar["date"].dt.dayofyear
)


calendar["annual_seasonality"] = (
    1
    + 0.20
    * np.sin(
        2
        * np.pi
        * calendar["day_of_year"]
        / 365
    )
)


assert calendar["annual_seasonality"].notna().all()
assert calendar["annual_seasonality"].between(
    0.80,
    1.20,
).all()


# =========================================================
# COMBINED SEASONALITY
# =========================================================

calendar["combined_seasonality"] = (
    calendar["weekday_multiplier"]
    * calendar["annual_seasonality"]
)


assert calendar["combined_seasonality"].notna().all()
assert (
    calendar["combined_seasonality"] > 0
).all()


# =========================================================
# PRODUCT × DATE DATASET
# =========================================================

product_dates = (
    product_parameters.assign(key=1)
    .merge(
        calendar.assign(key=1),
        on="key",
    )
    .drop(columns="key")
)


expected_rows = (
    len(products)
    * len(calendar)
)

assert len(product_dates) == expected_rows


# =========================================================
# PRODUCT TREND
# =========================================================

product_dates["product_trend_multiplier"] = (
    (
        1
        + product_dates["annual_growth_rate"]
    )
    ** (
        product_dates["days_since_start"]
        / 365
    )
)


assert (
    product_dates["product_trend_multiplier"]
    .notna()
    .all()
)

assert (
    product_dates["product_trend_multiplier"]
    > 0
).all()


# =========================================================
# PRODUCT SEASONALITY
# =========================================================

product_dates["product_seasonality_multiplier"] = (
    1
    + (
        product_dates["combined_seasonality"]
        - 1
    )
    * product_dates["seasonality_strength"]
)


assert (
    product_dates["product_seasonality_multiplier"]
    .notna()
    .all()
)

assert (
    product_dates["product_seasonality_multiplier"]
    > 0
).all()


# =========================================================
# EXPECTED DEMAND
# =========================================================

product_dates["expected_demand"] = (
    product_dates["base_daily_demand"]
    * product_dates["product_trend_multiplier"]
    * product_dates["product_seasonality_multiplier"]
)


assert product_dates["expected_demand"].notna().all()
assert (product_dates["expected_demand"] > 0).all()


# =========================================================
# EVENTS
# =========================================================

events = []

for event in EVENT_CONFIG:

    for category, multiplier in (
        event["category_multipliers"].items()
    ):

        events.append({
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "start_date": pd.Timestamp(
                event["start_date"]
            ),
            "end_date": pd.Timestamp(
                event["end_date"]
            ),
            "category": category,
            "event_multiplier": multiplier,
        })


events = pd.DataFrame(events)


assert len(events) == 25

assert events["event_multiplier"].notna().all()

assert (
    events["event_multiplier"] > 0
).all()


# =========================================================
# MATCH EVENTS TO PRODUCT-DATE ROWS
# =========================================================

event_matches = (
    product_dates[
        [
            "product_id",
            "category",
            "date",
        ]
    ]
    .merge(
        events,
        on="category",
        how="left",
    )
)


event_matches = event_matches[
    (
        event_matches["date"]
        >= event_matches["start_date"]
    )
    &
    (
        event_matches["date"]
        <= event_matches["end_date"]
    )
]


# =========================================================
# APPLY EVENT MULTIPLIERS
# =========================================================

product_dates["event_multiplier"] = 1.0


event_index = (
    event_matches
    .set_index(
        [
            "product_id",
            "date",
        ]
    )["event_multiplier"]
)


product_dates = (
    product_dates
    .set_index(
        [
            "product_id",
            "date",
        ]
    )
)


product_dates["event_multiplier"] = (
    event_index
    .reindex(product_dates.index)
    .fillna(1.0)
)


product_dates = (
    product_dates
    .reset_index()
)


# =========================================================
# EVENT-ADJUSTED DEMAND
# =========================================================

product_dates["event_adjusted_demand"] = (
    product_dates["expected_demand"]
    * product_dates["event_multiplier"]
)


assert (
    product_dates["event_adjusted_demand"]
    .notna()
    .all()
)

assert (
    product_dates["event_adjusted_demand"]
    > 0
).all()


# =========================================================
# RANDOM DEMAND NOISE
# =========================================================

product_dates["noise_factor"] = np.exp(
    np.random.normal(
        loc=0,
        scale=product_dates["noise_std"],
    )
)


assert product_dates["noise_factor"].notna().all()

assert (
    product_dates["noise_factor"] > 0
).all()


# =========================================================
# ACTUAL DEMAND
# =========================================================

product_dates["actual_demand"] = (
    product_dates["event_adjusted_demand"]
    * product_dates["noise_factor"]
)


assert product_dates["actual_demand"].notna().all()

assert (
    product_dates["actual_demand"] >= 0
).all()


product_dates["actual_demand"] = (
    product_dates["actual_demand"]
    .round()
    .astype(int)
)


# =========================================================
# INTERMITTENT DEMAND
# =========================================================

intermittent_mask = (
    product_dates["demand_class"]
    == "Intermittent"
)


demand_occurs = (
    np.random.random(
        len(product_dates)
    )
    < INTERMITTENT_DEMAND_PROBABILITY
)


product_dates.loc[
    intermittent_mask & ~demand_occurs,
    "actual_demand",
] = 0


# =========================================================
# FINAL VALIDATION
# =========================================================

assert product_dates["actual_demand"].notna().all()

assert (
    product_dates["actual_demand"] >= 0
).all()


intermittent_demand = product_dates[
    intermittent_mask
]


zero_rate = (
    intermittent_demand["actual_demand"]
    == 0
).mean()


# =========================================================
# SAVE DATASET
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


product_dates.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 60)
print("DEMAND GENERATION COMPLETE")
print("=" * 60)

print()
print(f"Products: {len(products):,}")

print(
    f"Dates: {len(calendar):,}"
)

print(
    f"Product-date rows: {len(product_dates):,}"
)

print(
    f"Event matches: {len(event_matches):,}"
)

print(
    f"Intermittent zero-demand rate: "
    f"{zero_rate:.2%}"
)

print()
print("Expected demand summary:")

print(
    product_dates["expected_demand"]
    .describe()
)


print()
print("Actual demand summary:")

print(
    product_dates["actual_demand"]
    .describe()
)


print()
print("Event multiplier distribution:")

print(
    product_dates["event_multiplier"]
    .value_counts()
    .sort_index()
)


print()
print("Sample:")

print(
    product_dates[
        [
            "product_id",
            "date",
            "demand_class",
            "expected_demand",
            "event_multiplier",
            "noise_factor",
            "actual_demand",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print()
print(f"Saved to:")

print(OUTPUT_FILE)