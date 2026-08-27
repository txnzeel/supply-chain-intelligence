import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scenario_analysis.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scenario_summary.csv"
)

OPPORTUNITIES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scenario_opportunities.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

optimization = pd.read_csv(INPUT_FILE)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "product_id",
    "category",
    "demand_class",
    "risk_class",
    "primary_risk_driver",

    "baseline_service_level",
    "baseline_order_coverage_days",
    "baseline_safety_stock",
    "baseline_reorder_point",
    "baseline_order_quantity",
    "baseline_fill_rate",
    "baseline_stockout_rate",
    "baseline_lost_sales",
    "baseline_inventory_days",
    "baseline_total_cost",

    "optimized_service_level",
    "optimized_safety_stock",
    "optimized_reorder_point",
    "optimized_order_quantity",
    "optimized_order_coverage_days",
    "optimized_fill_rate",
    "optimized_stockout_rate",
    "optimized_lost_sales",
    "optimized_inventory_days",
    "optimized_total_cost",

    "optimized_purchase_orders",
    "optimized_purchase_quantity",
    "optimized_supplier_delay_rate",
    "optimized_average_supplier_delay",

    "lost_sales_reduction",
    "cost_change",
    "cost_savings",
    "inventory_days_change",
    "fill_rate_change",
    "safety_stock_change",

    "recommended_action",
]


missing_columns = [
    column
    for column in required_columns
    if column not in optimization.columns
]

assert not missing_columns, (
    f"Missing required columns: {missing_columns}"
)


# =========================================================
# BASIC VALIDATION
# =========================================================

assert len(optimization) == 2000, (
    f"Expected 2000 products, found {len(optimization)}"
)

assert optimization["product_id"].notna().all()

assert optimization["product_id"].is_unique


# =========================================================
# NUMERIC CONVERSION
# =========================================================

numeric_columns = [
    column
    for column in required_columns
    if column not in [
        "product_id",
        "category",
        "demand_class",
        "risk_class",
        "primary_risk_driver",
        "recommended_action",
    ]
]


for column in numeric_columns:

    optimization[column] = pd.to_numeric(
        optimization[column],
        errors="coerce",
    )


assert optimization[numeric_columns].notna().all().all(), (
    "Numeric columns contain missing or invalid values."
)


# =========================================================
# BASELINE / OPTIMIZED DATA
# =========================================================

df = optimization.copy()


# =========================================================
# SCENARIO DEFINITIONS
# =========================================================
#
# Scenario 1:
# Current optimized policy
#
# Scenario 2:
# Demand surge
# +20% demand
#
# Scenario 3:
# Supplier disruption
# +30% supplier delay
#
# Scenario 4:
# Combined stress
# +20% demand
# +30% supplier delay
#
# These are sensitivity scenarios, not new optimization runs.
#


SCENARIOS = {

    "Optimized Policy": {
        "demand_multiplier": 1.00,
        "supplier_delay_multiplier": 1.00,
        "safety_stock_multiplier": 1.00,
    },

    "Demand Surge": {
        "demand_multiplier": 1.20,
        "supplier_delay_multiplier": 1.00,
        "safety_stock_multiplier": 1.10,
    },

    "Supplier Disruption": {
        "demand_multiplier": 1.00,
        "supplier_delay_multiplier": 1.30,
        "safety_stock_multiplier": 1.15,
    },

    "Combined Stress": {
        "demand_multiplier": 1.20,
        "supplier_delay_multiplier": 1.30,
        "safety_stock_multiplier": 1.25,
    },

}


# =========================================================
# BASELINE METRICS
# =========================================================

baseline_cost = df["baseline_total_cost"]

baseline_lost_sales = df["baseline_lost_sales"]

baseline_fill_rate = df["baseline_fill_rate"]

baseline_inventory_days = df["baseline_inventory_days"]


# =========================================================
# SCENARIO CALCULATION FUNCTION
# =========================================================

def calculate_scenario(data, scenario_name, config):

    result = data[
        [
            "product_id",
            "category",
            "demand_class",
            "risk_class",
            "primary_risk_driver",
        ]
    ].copy()

    demand_multiplier = config["demand_multiplier"]

    supplier_multiplier = (
        config["supplier_delay_multiplier"]
    )

    safety_multiplier = (
        config["safety_stock_multiplier"]
    )


    # -----------------------------------------------------
    # Demand
    # -----------------------------------------------------

    result["scenario_demand_multiplier"] = (
        demand_multiplier
    )


    # -----------------------------------------------------
    # Supplier delay
    # -----------------------------------------------------

    result["scenario_supplier_delay"] = (

        data["optimized_average_supplier_delay"]
        *
        supplier_multiplier

    )


    result["scenario_supplier_delay_rate"] = (

        data["optimized_supplier_delay_rate"]

        *
        supplier_multiplier

    )


    # -----------------------------------------------------
    # Safety stock
    # -----------------------------------------------------

    result["scenario_safety_stock"] = (

        data["optimized_safety_stock"]

        *
        safety_multiplier

    )


    # -----------------------------------------------------
    # Reorder point
    # -----------------------------------------------------

    result["scenario_reorder_point"] = (

        data["optimized_reorder_point"]

        *
        demand_multiplier

        *
        supplier_multiplier

    )


    # -----------------------------------------------------
    # Order quantity
    # -----------------------------------------------------

    result["scenario_order_quantity"] = (

        data["optimized_order_quantity"]

        *
        demand_multiplier

    )


    # -----------------------------------------------------
    # Purchase quantity
    # -----------------------------------------------------

    result["scenario_purchase_quantity"] = (

        data["optimized_purchase_quantity"]

        *
        demand_multiplier

    )


    # -----------------------------------------------------
    # Purchase orders
    # -----------------------------------------------------

    result["scenario_purchase_orders"] = np.ceil(

        data["optimized_purchase_orders"]

        *
        demand_multiplier

    )


    # -----------------------------------------------------
    # Inventory days
    # -----------------------------------------------------
    #
    # Higher demand reduces days of coverage.
    # Higher safety stock increases coverage.
    # Supplier disruption increases buffer requirement.
    #

    demand_effect = (
        1 / demand_multiplier
    )

    safety_effect = (
        safety_multiplier
    )

    supplier_effect = (
        1
        +
        (
            supplier_multiplier - 1
        )
        * 0.50
    )

    result["scenario_inventory_days"] = (

        data["optimized_inventory_days"]

        *
        demand_effect

        *
        safety_effect

        *
        supplier_effect

    )


    # -----------------------------------------------------
    # Fill rate
    # -----------------------------------------------------
    #
    # Demand and supplier stress reduce service.
    # Additional safety stock offsets part of the damage.
    #

    demand_penalty = (

        (demand_multiplier - 1)

        * 0.08

    )

    supplier_penalty = (

        (supplier_multiplier - 1)

        * 0.10

    )

    safety_protection = (

        (safety_multiplier - 1)

        * 0.06

    )

    scenario_fill_rate = (

        data["optimized_fill_rate"]

        -
        demand_penalty

        -
        supplier_penalty

        +
        safety_protection

    )

    result["scenario_fill_rate"] = (
        scenario_fill_rate.clip(
            lower=0,
            upper=1,
        )
    )


    # -----------------------------------------------------
    # Stockout rate
    # -----------------------------------------------------

    fill_rate_difference = (

        data["optimized_fill_rate"]

        -
        result["scenario_fill_rate"]

    )

    result["scenario_stockout_rate"] = (

        data["optimized_stockout_rate"]

        +
        fill_rate_difference.clip(
            lower=0
        )

    ).clip(
        lower=0,
        upper=1,
    )


    # -----------------------------------------------------
    # Lost sales
    # -----------------------------------------------------
    #
    # Demand increases potential lost sales.
    # Lower fill rate increases lost sales further.
    #

    baseline_optimized_lost_sales = (
        data["optimized_lost_sales"]
    )

    demand_scaled_lost_sales = (

        baseline_optimized_lost_sales

        *
        demand_multiplier

    )

    service_loss_factor = (

        1
        +
        (
            data["optimized_fill_rate"]
            -
            result["scenario_fill_rate"]
        ).clip(lower=0)
        *
        5

    )

    result["scenario_lost_sales"] = (

        demand_scaled_lost_sales
        *
        service_loss_factor

    ).round(0)


    # -----------------------------------------------------
    # Inventory cost impact
    # -----------------------------------------------------

    inventory_ratio = (

        result["scenario_inventory_days"]

        /
        data["optimized_inventory_days"]

    )

    inventory_cost = (

        data["optimized_total_cost"]

        *
        (
            1
            +
            (
                inventory_ratio - 1
            )
            * 0.25
        )

    )


    # -----------------------------------------------------
    # Ordering / purchasing cost impact
    # -----------------------------------------------------

    purchase_ratio = (

        result["scenario_purchase_quantity"]

        /
        data["optimized_purchase_quantity"]

    )

    purchasing_cost = (

        data["optimized_total_cost"]

        *
        (
            purchase_ratio - 1
        )
        *
        0.35

    )


    # -----------------------------------------------------
    # Supplier disruption cost
    # -----------------------------------------------------

    supplier_cost = (

        data["optimized_total_cost"]

        *
        (
            supplier_multiplier - 1
        )
        *
        0.10

    )


    # -----------------------------------------------------
    # Scenario total cost
    # -----------------------------------------------------

    result["scenario_total_cost"] = (

        inventory_cost

        +
        purchasing_cost

        +
        supplier_cost

    )


    # Prevent unrealistic negative cost

    result["scenario_total_cost"] = (

        result["scenario_total_cost"]
        .clip(lower=0)

    )


    # -----------------------------------------------------
    # Cost change vs optimized policy
    # -----------------------------------------------------

    result["cost_change_vs_optimized"] = (

        result["scenario_total_cost"]

        -
        data["optimized_total_cost"]

    )


    result["cost_savings_vs_optimized"] = (

        -result["cost_change_vs_optimized"]

    )


    # -----------------------------------------------------
    # Lost-sales change
    # -----------------------------------------------------

    result["lost_sales_change_vs_optimized"] = (

        result["scenario_lost_sales"]

        -
        data["optimized_lost_sales"]

    )


    result["lost_sales_increase_vs_optimized"] = (

        result[
            "lost_sales_change_vs_optimized"
        ]

        .clip(lower=0)

    )


    # -----------------------------------------------------
    # Fill-rate change
    # -----------------------------------------------------

    result["fill_rate_change_vs_optimized"] = (

        result["scenario_fill_rate"]

        -
        data["optimized_fill_rate"]

    )


    # -----------------------------------------------------
    # Inventory change
    # -----------------------------------------------------

    result["inventory_days_change_vs_optimized"] = (

        result["scenario_inventory_days"]

        -
        data["optimized_inventory_days"]

    )


    # -----------------------------------------------------
    # Scenario severity
    # -----------------------------------------------------

    result["scenario_severity"] = np.select(

        [

            result[
                "lost_sales_increase_vs_optimized"
            ]
            > data["optimized_lost_sales"] * 0.20,

            result[
                "lost_sales_increase_vs_optimized"
            ]
            > data["optimized_lost_sales"] * 0.10,

            result[
                "lost_sales_increase_vs_optimized"
            ]
            > 0,

        ],

        [
            "Severe",
            "High",
            "Moderate",
        ],

        default="Low",

    )


    # -----------------------------------------------------
    # Scenario recommendation
    # -----------------------------------------------------

    def recommendation(row):

        if (
            row["scenario_severity"]
            == "Severe"
        ):

            return (
                "Immediate contingency planning"
            )

        if (
            row["scenario_severity"]
            == "High"
        ):

            return (
                "Increase inventory protection"
            )

        if (
            row["scenario_severity"]
            == "Moderate"
        ):

            return (
                "Monitor and prepare response"
            )

        return (
            "Current policy remains resilient"
        )


    result["scenario_recommendation"] = (
        result.apply(
            recommendation,
            axis=1,
        )
    )


    result["scenario"] = scenario_name


    return result


# =========================================================
# RUN ALL SCENARIOS
# =========================================================

scenario_results = []


for scenario_name, config in SCENARIOS.items():

    scenario_result = calculate_scenario(
        df,
        scenario_name,
        config,
    )

    scenario_results.append(
        scenario_result
    )


scenario_analysis = pd.concat(
    scenario_results,
    ignore_index=True,
)


# =========================================================
# ROUND NUMERIC OUTPUT
# =========================================================

round_columns = [
    "scenario_supplier_delay",
    "scenario_supplier_delay_rate",
    "scenario_safety_stock",
    "scenario_reorder_point",
    "scenario_order_quantity",
    "scenario_purchase_quantity",
    "scenario_inventory_days",
    "scenario_fill_rate",
    "scenario_stockout_rate",
    "scenario_total_cost",
    "cost_change_vs_optimized",
    "cost_savings_vs_optimized",
    "fill_rate_change_vs_optimized",
    "inventory_days_change_vs_optimized",
]


for column in round_columns:

    if column in scenario_analysis.columns:

        scenario_analysis[column] = (
            pd.to_numeric(
                scenario_analysis[column],
                errors="coerce",
            )
            .round(4)
        )


# =========================================================
# VALIDATION
# =========================================================

print()
print("=" * 70)
print("SCENARIO ANALYSIS VALIDATION")
print("=" * 70)

print()

print(
    f"Products loaded: {len(df):,}"
)

print(
    f"Scenarios generated: "
    f"{scenario_analysis['scenario'].nunique()}"
)

assert (
    scenario_analysis["scenario"].nunique()
    == len(SCENARIOS)
)

assert (
    len(scenario_analysis)
    ==
    len(df) * len(SCENARIOS)
)

assert (
    scenario_analysis["product_id"]
    .notna()
    .all()
)

assert (
    scenario_analysis["scenario_total_cost"]
    >= 0
).all()

assert (
    scenario_analysis["scenario_fill_rate"]
    .between(0, 1)
    .all()
)

assert (
    scenario_analysis["scenario_stockout_rate"]
    .between(0, 1)
    .all()
)

print()

print("Validation checks passed.")


# =========================================================
# PORTFOLIO SCENARIO SUMMARY
# =========================================================

summary = (
    scenario_analysis
    .groupby(
        "scenario",
        as_index=False,
    )
    .agg(

        products=(
            "product_id",
            "count",
        ),

        total_scenario_cost=(
            "scenario_total_cost",
            "sum",
        ),

        total_cost_change=(
            "cost_change_vs_optimized",
            "sum",
        ),

        total_cost_savings=(
            "cost_savings_vs_optimized",
            "sum",
        ),

        total_lost_sales=(
            "scenario_lost_sales",
            "sum",
        ),

        total_lost_sales_increase=(
            "lost_sales_increase_vs_optimized",
            "sum",
        ),

        avg_fill_rate=(
            "scenario_fill_rate",
            "mean",
        ),

        avg_fill_rate_change=(
            "fill_rate_change_vs_optimized",
            "mean",
        ),

        avg_inventory_days=(
            "scenario_inventory_days",
            "mean",
        ),

        avg_inventory_days_change=(
            "inventory_days_change_vs_optimized",
            "mean",
        ),

    )
)


# =========================================================
# SCENARIO STATUS
# =========================================================

def scenario_status(row):

    if (
        row["total_cost_savings"] > 0
        and
        row["total_lost_sales_increase"] <= 0
    ):

        return "Highly resilient"

    if (
        row["total_cost_savings"] > 0
        and
        row["total_lost_sales_increase"] > 0
    ):

        return "Economically favorable but service exposed"

    if (
        row["total_cost_savings"] <= 0
        and
        row["total_lost_sales_increase"] <= 0
    ):

        return "Service resilient but more expensive"

    return "High exposure"


summary["scenario_status"] = (
    summary.apply(
        scenario_status,
        axis=1,
    )
)


# =========================================================
# PORTFOLIO PRIORITY
# =========================================================

def scenario_priority(row):

    if (
        row["total_lost_sales_increase"]
        >
        df["optimized_lost_sales"].sum()
        * 0.20
    ):

        return "P1 - Severe"

    if (
        row["total_lost_sales_increase"]
        >
        df["optimized_lost_sales"].sum()
        * 0.10
    ):

        return "P2 - High"

    if (
        row["total_lost_sales_increase"]
        > 0
    ):

        return "P3 - Moderate"

    return "P4 - Resilient"


summary["scenario_priority"] = (
    summary.apply(
        scenario_priority,
        axis=1,
    )
)


# =========================================================
# SCENARIO OPPORTUNITIES
# =========================================================

opportunities = (
    scenario_analysis[
        scenario_analysis["scenario"]
        != "Optimized Policy"
    ]
    .copy()
)


# =========================================================
# OPPORTUNITY SCORE
# =========================================================

lost_sales_exposure = (
    opportunities[
        "lost_sales_increase_vs_optimized"
    ]
    .clip(lower=0)
)


cost_exposure = (
    opportunities[
        "cost_change_vs_optimized"
    ]
    .clip(lower=0)
)


risk_score_map = {
    "Critical": 1.00,
    "High": 0.75,
    "Medium": 0.50,
    "Low": 0.25,
}


risk_score = (
    opportunities["risk_class"]
    .map(risk_score_map)
    .fillna(0)
)


def normalize(series):

    maximum = series.max()

    if maximum <= 0:

        return pd.Series(
            0,
            index=series.index,
        )

    return (
        series
        /
        maximum
    )


opportunities["scenario_opportunity_score"] = (

    0.45
    *
    normalize(
        lost_sales_exposure
    )

    +

    0.25
    *
    normalize(
        cost_exposure
    )

    +

    0.30
    *
    risk_score

)


# =========================================================
# OPPORTUNITY CLASSIFICATION
# =========================================================

def opportunity_class(row):

    score = row[
        "scenario_opportunity_score"
    ]

    if score >= 0.75:

        return "Critical exposure"

    if score >= 0.50:

        return "High exposure"

    if score >= 0.25:

        return "Moderate exposure"

    return "Low exposure"


opportunities["opportunity_class"] = (
    opportunities.apply(
        opportunity_class,
        axis=1,
    )
)


# =========================================================
# SORT OPPORTUNITIES
# =========================================================

opportunities = (
    opportunities
    .sort_values(
        [
            "scenario_opportunity_score",
            "lost_sales_increase_vs_optimized",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# SAVE OUTPUTS
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


scenario_analysis.to_csv(
    OUTPUT_FILE,
    index=False,
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


opportunities.to_csv(
    OPPORTUNITIES_FILE,
    index=False,
)


# =========================================================
# EXECUTIVE OUTPUT
# =========================================================

print()
print("=" * 70)
print("SCENARIO ANALYSIS COMPLETE")
print("=" * 70)

print()

print(
    f"Products analyzed: "
    f"{len(df):,}"
)

print(
    f"Scenarios analyzed: "
    f"{len(SCENARIOS)}"
)

print()

print(
    "Portfolio scenario results:"
)

print()

for _, row in summary.iterrows():

    print(
        f"{row['scenario']:<25}"
        f"Cost: ${row['total_scenario_cost']:>14,.2f}   "
        f"Lost sales: {row['total_lost_sales']:>10,.0f}   "
        f"Fill rate: {row['avg_fill_rate']:.2%}   "
        f"Status: {row['scenario_status']}"
    )


# =========================================================
# TOP SCENARIO EXPOSURES
# =========================================================

print()

print("=" * 70)
print("TOP 20 SCENARIO EXPOSURES")
print("=" * 70)

print()

print(
    opportunities[
        [
            "product_id",
            "scenario",
            "risk_class",
            "primary_risk_driver",
            "scenario_opportunity_score",
            "opportunity_class",
            "lost_sales_increase_vs_optimized",
            "cost_change_vs_optimized",
            "fill_rate_change_vs_optimized",
            "scenario_recommendation",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# =========================================================
# OUTPUT PATHS
# =========================================================

print()

print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print()

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)

print(
    OPPORTUNITIES_FILE
)

print()