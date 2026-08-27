import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

SERVICE_LEVEL_TARGET = 0.95

RISK_THRESHOLDS = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
}


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_daily.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_risk.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

inventory = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"],
)


# =========================================================
# VALIDATION
# =========================================================

required_columns = [
    "product_id",
    "category",
    "demand_class",
    "actual_demand",
    "opening_inventory",
    "closing_inventory",
    "demand_fulfilled",
    "lost_sales",
    "purchase_order",
    "purchase_order_arrival",
    "min_lead_time_days",
"max_lead_time_days",
"average_lead_time_days",
"lead_time_std",
    "actual_lead_time",
    "supplier_delay_days",
    "inventory_position",
    "stockout_flag",
    "fill_rate",
]


missing_columns = [
    column
    for column in required_columns
    if column not in inventory.columns
]


assert not missing_columns, (
    f"Missing required columns: {missing_columns}"
)


assert inventory["product_id"].notna().all()

assert inventory["date"].notna().all()

assert (
    inventory["actual_demand"] >= 0
).all()

assert (
    inventory["opening_inventory"] >= 0
).all()

assert (
    inventory["closing_inventory"] >= 0
).all()

assert (
    inventory["lost_sales"] >= 0
).all()

assert (
    inventory["supplier_delay_days"] >= 0
).all()


# =========================================================
# SORT
# =========================================================

inventory = inventory.sort_values(
    [
        "product_id",
        "date",
    ]
).reset_index(drop=True)


# =========================================================
# PRODUCT-LEVEL AGGREGATION
# =========================================================

risk = (
    inventory
    .groupby(
        [
            "product_id",
            "category",
            "demand_class",
        ],
        as_index=False,
    )
    .agg(
        total_demand=(
            "actual_demand",
            "sum",
        ),

        total_demand_fulfilled=(
            "demand_fulfilled",
            "sum",
        ),

        total_lost_sales=(
            "lost_sales",
            "sum",
        ),

        stockout_days=(
            "stockout_flag",
            "sum",
        ),

        average_inventory=(
            "closing_inventory",
            "mean",
        ),

        minimum_inventory=(
            "closing_inventory",
            "min",
        ),

        maximum_inventory=(
            "closing_inventory",
            "max",
        ),

        average_inventory_position=(
            "inventory_position",
            "mean",
        ),

        purchase_quantity=(
            "purchase_order",
            "sum",
        ),

        purchase_orders=(
            "purchase_order",
            lambda x: (x > 0).sum(),
        ),

        purchase_order_arrivals=(
            "purchase_order_arrival",
            lambda x: (x > 0).sum(),
        ),

        delayed_orders=(
            "supplier_delay_days",
            lambda x: (x > 0).sum(),
        ),

        average_lead_time=(
            "actual_lead_time",
            lambda x: x[x > 0].mean(),
        ),

        maximum_lead_time=(
            "actual_lead_time",
            "max",
        ),

        average_supplier_delay=(
            "supplier_delay_days",
            lambda x: x[x > 0].mean(),
        ),

        maximum_supplier_delay=(
            "supplier_delay_days",
            "max",
        ),

        average_daily_demand=(
            "actual_demand",
            "mean",
        ),

        demand_std=(
            "actual_demand",
            "std",
        ),

        base_daily_demand=(
            "base_daily_demand",
            "first",
        ),

        reorder_point=(
            "reorder_point",
            "first",
        ),

        safety_stock=(
            "safety_stock",
            "first",
        ),

        order_quantity=(
            "order_quantity",
            "first",
        ),

        min_lead_time_days=(
    "min_lead_time_days",
    "first",
),

max_lead_time_days=(
    "max_lead_time_days",
    "first",
),

average_lead_time_days=(
    "average_lead_time_days",
    "first",
),

lead_time_std=(
    "lead_time_std",
    "first",
)   ,
    )
)


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

risk["demand_std"] = (
    risk["demand_std"]
    .fillna(0)
)


risk["average_lead_time"] = (
    risk["average_lead_time"]
    .fillna(risk["average_lead_time_days"])
)


risk["average_supplier_delay"] = (
    risk["average_supplier_delay"]
    .fillna(0)
)


# =========================================================
# BASIC PERFORMANCE METRICS
# =========================================================

risk["fill_rate"] = np.where(
    risk["total_demand"] > 0,

    risk["total_demand_fulfilled"]
    / risk["total_demand"],

    1.0,
)


risk["stockout_rate"] = (
    risk["stockout_days"]
    / inventory["date"].nunique()
)


risk["lost_sales_rate"] = np.where(
    risk["total_demand"] > 0,

    risk["total_lost_sales"]
    / risk["total_demand"],

    0,
)


# =========================================================
# DEMAND VOLATILITY
# =========================================================

risk["coefficient_of_variation"] = np.where(
    risk["average_daily_demand"] > 0,

    risk["demand_std"]
    / risk["average_daily_demand"],

    0,
)


# =========================================================
# INVENTORY DAYS
# =========================================================

risk["inventory_days"] = np.where(
    risk["average_daily_demand"] > 0,

    risk["average_inventory"]
    / risk["average_daily_demand"],

    0,
)


# =========================================================
# EXCESS INVENTORY
# =========================================================
#
# Inventory above the intended operating requirement:
#
# demand during lead time
# + safety stock
# + one order cycle
#
# =========================================================

risk["target_inventory"] = (
    risk["average_daily_demand"]
    * risk["average_lead_time_days"]
    +
    risk["safety_stock"]
    +
    risk["average_daily_demand"]
    * 30
)


risk["excess_inventory"] = (
    risk["average_inventory"]
    - risk["target_inventory"]
).clip(lower=0)


risk["excess_inventory_rate"] = np.where(
    risk["average_inventory"] > 0,

    risk["excess_inventory"]
    / risk["average_inventory"],

    0,
)


# =========================================================
# INVENTORY TURNOVER
# =========================================================

risk["inventory_turnover"] = np.where(
    risk["average_inventory"] > 0,

    risk["total_demand"]
    / risk["average_inventory"],

    0,
)


# =========================================================
# SUPPLIER METRICS
# =========================================================

risk["supplier_delay_rate"] = np.where(
    risk["purchase_orders"] > 0,

    risk["delayed_orders"]
    / risk["purchase_orders"],

    0,
)


risk["lead_time_variability"] = (
    risk["average_lead_time"]
    - risk["average_lead_time_days"]
)


# =========================================================
# SERVICE LEVEL GAP
# =========================================================

risk["service_level_gap"] = (
    SERVICE_LEVEL_TARGET
    - risk["fill_rate"]
).clip(lower=0)


# =========================================================
# RISK COMPONENT 1 — STOCKOUT RISK
# =========================================================

risk["stockout_risk"] = (
    risk["stockout_rate"] * 100
)


# Lost sales increase the severity.

risk["stockout_risk"] += (
    risk["lost_sales_rate"] * 100
)


risk["stockout_risk"] = (
    risk["stockout_risk"]
    .clip(upper=100)
)


# =========================================================
# RISK COMPONENT 2 — SERVICE RISK
# =========================================================

risk["service_risk"] = (
    risk["service_level_gap"] * 100
)


risk["service_risk"] = (
    risk["service_risk"]
    .clip(upper=100)
)


# =========================================================
# RISK COMPONENT 3 — DEMAND VOLATILITY RISK
# =========================================================

risk["volatility_risk"] = (
    risk["coefficient_of_variation"]
    * 50
)


risk["volatility_risk"] = (
    risk["volatility_risk"]
    .clip(upper=100)
)


# =========================================================
# RISK COMPONENT 4 — SUPPLIER RISK
# =========================================================

risk["supplier_risk"] = (
    risk["supplier_delay_rate"]
    * 100
)


risk["supplier_risk"] += np.where(
    risk["lead_time_variability"] > 0,

    risk["lead_time_variability"]
    * 5,

    0,
)


risk["supplier_risk"] = (
    risk["supplier_risk"]
    .clip(upper=100)
)


# =========================================================
# RISK COMPONENT 5 — EXCESS INVENTORY RISK
# =========================================================

risk["excess_inventory_risk"] = (
    risk["excess_inventory_rate"]
    * 100
)


risk["excess_inventory_risk"] = (
    risk["excess_inventory_risk"]
    .clip(upper=100)
)


# =========================================================
# OVERALL INVENTORY RISK SCORE
# =========================================================
#
# Stockout/service problems receive the highest weight.
#
# =========================================================

risk["inventory_risk_score"] = (
    risk["stockout_risk"] * 0.30
    +
    risk["service_risk"] * 0.25
    +
    risk["volatility_risk"] * 0.15
    +
    risk["supplier_risk"] * 0.15
    +
    risk["excess_inventory_risk"] * 0.15
)


risk["inventory_risk_score"] = (
    risk["inventory_risk_score"]
    .clip(0, 100)
    .round(2)
)


# =========================================================
# RISK CLASSIFICATION
# =========================================================

def classify_risk(score):

    if score < RISK_THRESHOLDS["Low"]:

        return "Low"

    elif score < RISK_THRESHOLDS["Medium"]:

        return "Medium"

    elif score < RISK_THRESHOLDS["High"]:

        return "High"

    else:

        return "Critical"


# =========================================================
# RISK CLASSIFICATION
# =========================================================

risk["risk_class"] = np.select(
    [
        risk["inventory_risk_score"] >= 16,
        risk["inventory_risk_score"] >= 11,
        risk["inventory_risk_score"] >= 6,
    ],
    [
        "Critical",
        "High",
        "Medium",
    ],
    default="Low",
)

# =========================================================
# PRIMARY RISK DRIVER
# =========================================================

risk_components = {
    "Stockout": "stockout_risk",
    "Service": "service_risk",
    "Demand Volatility": "volatility_risk",
    "Supplier": "supplier_risk",
    "Excess Inventory": "excess_inventory_risk",
}

risk["primary_risk_driver"] = (
    risk[
        list(risk_components.values())
    ]
    .idxmax(axis=1)
    .map(
        {
            value: key
            for key, value in risk_components.items()
        }
    )
)


# =========================================================
# RECOMMENDED ACTION
# =========================================================

def recommend_action(row):

    driver = row["primary_risk_driver"]
    risk_class = row["risk_class"]

    if driver == "Stockout":

        return "Increase safety stock / review reorder point"

    if driver == "Service":

        return "Increase inventory protection"

    if driver == "Demand Volatility":

        return "Review forecast and demand variability"

    if driver == "Supplier":

        return "Review supplier lead time and reliability"

    if driver == "Excess Inventory":

        return "Reduce inventory / optimize order quantity"

    if risk_class == "Low":

        return "Maintain current policy"

    return "Monitor"


risk["recommended_action"] = (
    risk.apply(
        recommend_action,
        axis=1,
    )
)


# =========================================================
# INVENTORY HEALTH
# =========================================================

def inventory_health(row):

    stockout_risk = row["stockout_risk"]
    excess_risk = row["excess_inventory_risk"]
    supplier_risk = row["supplier_risk"]
    volatility_risk = row["volatility_risk"]
    service_risk = row["service_risk"]

    # -----------------------------------------------------
    # Critical operational conditions
    # -----------------------------------------------------

    if (
        stockout_risk >= 70
        and service_risk >= 70
    ):
        return "Critical Service Risk"

    if (
        stockout_risk >= 70
        and supplier_risk >= 60
    ):
        return "Supply Disruption"

    # -----------------------------------------------------
    # Inventory imbalance
    # -----------------------------------------------------

    if stockout_risk >= 60:
        return "Understocked"

    if excess_risk >= 60:
        return "Overstocked"

    # -----------------------------------------------------
    # Supplier / demand problems
    # -----------------------------------------------------

    if supplier_risk >= 60:
        return "Supplier Risk"

    if volatility_risk >= 60:
        return "Demand Risk"

    # -----------------------------------------------------
    # Moderate operational warning
    # -----------------------------------------------------

    if (
        stockout_risk >= 40
        or service_risk >= 40
        or supplier_risk >= 40
        or volatility_risk >= 40
        or excess_risk >= 40
    ):
        return "Monitor"

    return "Healthy"


risk["inventory_health"] = (
    risk.apply(
        inventory_health,
        axis=1,
    )
)

# =========================================================
# FINAL VALIDATION
# =========================================================

assert len(risk) == inventory["product_id"].nunique()

assert (
    risk["inventory_risk_score"]
    .between(0, 100)
).all()

assert (
    risk["stockout_risk"]
    .between(0, 100)
).all()

assert (
    risk["supplier_risk"]
    .between(0, 100)
).all()

assert (
    risk["excess_inventory_risk"]
    .between(0, 100)
).all()

assert (
    risk["fill_rate"]
    .between(0, 1)
).all()

assert (
    risk["stockout_rate"]
    .between(0, 1)
).all()


# =========================================================
# SORT BY RISK
# =========================================================

risk = risk.sort_values(
    "inventory_risk_score",
    ascending=False,
).reset_index(drop=True)


# =========================================================
# SAVE
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


risk.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 60)
print("INVENTORY RISK ENGINE COMPLETE")
print("=" * 60)

print()

print(
    f"Products analyzed: "
    f"{len(risk):,}"
)

print()
print("Risk score distribution:")
print(
    risk["inventory_risk_score"]
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

print()
print("Risk distribution:")

print(
    risk["risk_class"]
    .value_counts()
    .reindex(
        [
            "Critical",
            "High",
            "Medium",
            "Low",
        ],
        fill_value=0,
    )
)


print()

print(
    f"Average risk score: "
    f"{risk['inventory_risk_score'].mean():.2f}"
)

print(
    f"Average fill rate: "
    f"{risk['fill_rate'].mean():.2%}"
)

print(
    f"Average stockout rate: "
    f"{risk['stockout_rate'].mean():.2%}"
)

print(
    f"Average supplier delay rate: "
    f"{risk['supplier_delay_rate'].mean():.2%}"
)

print()

print("Primary risk drivers:")

print(
    risk["primary_risk_driver"]
    .value_counts()
)


print()

print("Inventory health:")

print(
    risk["inventory_health"]
    .value_counts()
)


print()

print("Top 20 highest-risk products:")

print(
    risk[
        [
            "product_id",
            "category",
            "demand_class",
            "inventory_risk_score",
            "risk_class",
            "primary_risk_driver",
            "inventory_health",
            "fill_rate",
            "stockout_rate",
            "supplier_delay_rate",
            "inventory_days",
            "recommended_action",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

# =========================================================
# RISK CLASS VALIDATION
# =========================================================

print()
print("=" * 60)
print("RISK CLASS VALIDATION")
print("=" * 60)

risk_validation = (
    risk
    .groupby("risk_class", observed=True)
    .agg(
        products=("product_id", "count"),
        avg_risk_score=("inventory_risk_score", "mean"),
        avg_fill_rate=("fill_rate", "mean"),
        avg_stockout_rate=("stockout_rate", "mean"),
        avg_supplier_delay_rate=("supplier_delay_rate", "mean"),
        avg_inventory_days=("inventory_days", "mean"),
    )
    .reindex(
        ["Critical", "High", "Medium", "Low"]
    )
)

print()
print(
    risk_validation
    .round(4)
    .to_string()
)

print()

print("Saved to:")

print(OUTPUT_FILE)