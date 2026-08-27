import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SCENARIO_DECISIONS_FILE = (
    DATA_DIR
    / "scenario_decisions.csv"
)

SCENARIO_SUMMARY_FILE = (
    DATA_DIR
    / "scenario_decision_summary.csv"
)

OPTIMIZATION_FILE = (
    DATA_DIR
    / "inventory_optimization.csv"
)

OUTPUT_FILE = (
    DATA_DIR
    / "contingency_plans.csv"
)

SUMMARY_OUTPUT_FILE = (
    DATA_DIR
    / "contingency_summary.csv"
)

EXPOSURE_OUTPUT_FILE = (
    DATA_DIR
    / "contingency_exposures.csv"
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def require_columns(
    dataframe,
    required_columns,
    dataframe_name,
):

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    assert not missing, (
        f"{dataframe_name} is missing required "
        f"columns: {missing}"
    )


def numeric_series(
    dataframe,
    column,
    default=0.0,
):

    if column not in dataframe.columns:

        return pd.Series(
            default,
            index=dataframe.index,
            dtype=float,
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).fillna(default)


# =========================================================
# LOAD INPUTS
# =========================================================

scenario_decisions = pd.read_csv(
    SCENARIO_DECISIONS_FILE
)

scenario_summary = pd.read_csv(
    SCENARIO_SUMMARY_FILE
)

optimization = pd.read_csv(
    OPTIMIZATION_FILE
)


# =========================================================
# HEADER
# =========================================================

print()
print("=" * 70)
print("CONTINGENCY PLANNING ENGINE")
print("=" * 70)

print()

print(
    f"Scenario decisions loaded: "
    f"{len(scenario_decisions):,}"
)

print(
    f"Scenario summary records loaded: "
    f"{len(scenario_summary):,}"
)

print(
    f"Optimization decisions loaded: "
    f"{len(optimization):,}"
)


# =========================================================
# INPUT VALIDATION
# =========================================================

print()
print("=" * 70)
print("INPUT VALIDATION")
print("=" * 70)


require_columns(
    scenario_decisions,

    [
        "product_id",
        "scenario",
        "risk_class",
        "primary_risk_driver",

        "exposure_class",
        "action_priority",

        "final_decision_score",

        "lost_sales_increase_vs_optimized",
        "cost_change_vs_optimized",
        "fill_rate_change_vs_optimized",

        "additional_safety_stock_required",

        "scenario_recommendation",
        "decision_confidence",
    ],

    "scenario_decisions",
)


require_columns(
    scenario_summary,

    [
        "scenario",
        "products",
    ],

    "scenario_summary",
)


require_columns(
    optimization,

    [
        "product_id",
        "category",
        "demand_class",
        "risk_class",
        "primary_risk_driver",

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

        "optimized_supplier_delay_rate",
        "optimized_average_supplier_delay",
    ],

    "inventory_optimization",
)


assert scenario_decisions["product_id"].notna().all()

assert optimization["product_id"].notna().all()

assert optimization["product_id"].is_unique, (
    "inventory_optimization.csv must contain "
    "one row per product."
)


assert scenario_decisions["scenario"].notna().all()

assert scenario_decisions["scenario"].nunique() >= 2


print()
print("Input validation passed.")


# =========================================================
# COPY SCENARIO DATA
# =========================================================

plans = scenario_decisions.copy()


# =========================================================
# MERGE OPTIMIZED INVENTORY POLICY
# =========================================================
#
# scenario_decisions.csv does NOT contain all of the
# optimized inventory-policy columns.
#
# Therefore we explicitly retrieve them from
# inventory_optimization.csv.
#


optimization_policy = optimization[
    [
        "product_id",

        "category",
        "demand_class",

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

        "optimized_supplier_delay_rate",
        "optimized_average_supplier_delay",
    ]
].copy()


# ---------------------------------------------------------
# Prevent duplicate columns during merge
# ---------------------------------------------------------

policy_columns = [
    "category",
    "demand_class",

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

    "optimized_supplier_delay_rate",
    "optimized_average_supplier_delay",
]


for column in policy_columns:

    if column in plans.columns:

        plans = plans.drop(
            columns=[column]
        )


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

plans = plans.merge(

    optimization_policy,

    on="product_id",

    how="left",

    validate="many_to_one",
)


# =========================================================
# MERGE VALIDATION
# =========================================================

assert plans["optimized_safety_stock"].notna().all(), (
    "Optimization merge failed: "
    "optimized_safety_stock contains missing values."
)

assert plans["optimized_reorder_point"].notna().all(), (
    "Optimization merge failed: "
    "optimized_reorder_point contains missing values."
)

assert plans["optimized_order_quantity"].notna().all(), (
    "Optimization merge failed: "
    "optimized_order_quantity contains missing values."
)

assert plans[
    "optimized_order_coverage_days"
].notna().all(), (
    "Optimization merge failed: "
    "optimized_order_coverage_days contains missing values."
)


print()
print(
    "Optimization policy successfully merged."
)

print(
    f"Scenario-product records: "
    f"{len(plans):,}"
)

print(
    f"Products matched: "
    f"{plans['product_id'].nunique():,}"
)


# =========================================================
# NUMERIC CONVERSION
# =========================================================

numeric_columns = [

    "final_decision_score",

    "lost_sales_increase_vs_optimized",

    "cost_change_vs_optimized",

    "fill_rate_change_vs_optimized",

    "additional_safety_stock_required",

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

    "optimized_supplier_delay_rate",

    "optimized_average_supplier_delay",
]


for column in numeric_columns:

    plans[column] = pd.to_numeric(

        plans[column],

        errors="coerce",

    ).fillna(0)


# =========================================================
# CONTINGENCY SEVERITY
# =========================================================

def contingency_severity(row):

    exposure = str(
        row["exposure_class"]
    )

    priority = str(
        row["action_priority"]
    )

    relationship = str(
        row.get(
            "risk_scenario_relationship",
            "",
        )
    )

    score = float(
        row["final_decision_score"]
    )

    if (
        exposure == "Critical exposure"
        or
        priority == "P1 - Immediate"
        or
        score >= 0.65
    ):

        return "Critical"

    if (
        exposure == "High exposure"
        or
        priority == "P2 - High"
        or
        score >= 0.45
    ):

        return "High"

    if (
        exposure == "Moderate exposure"
        or
        priority == "P3 - Planned"
        or
        relationship == "Scenario-sensitive"
    ):

        return "Moderate"

    return "Low"


plans["contingency_severity"] = (
    plans.apply(
        contingency_severity,
        axis=1,
    )
)


# =========================================================
# SCENARIO TYPE
# =========================================================

def scenario_type(row):

    scenario = str(
        row["scenario"]
    )

    if scenario == "Demand Surge":

        return "Demand stress"

    if scenario == "Supplier Disruption":

        return "Supplier stress"

    if scenario == "Combined Stress":

        return "Combined stress"

    if scenario == "Optimized Policy":

        return "Baseline optimized policy"

    return "Other"


plans["scenario_type"] = (
    plans.apply(
        scenario_type,
        axis=1,
    )
)


# =========================================================
# CONTINGENCY MULTIPLIER
# =========================================================
#
# This determines how aggressively the optimized policy
# should be buffered under each scenario.
#


def contingency_multiplier(row):

    scenario = str(
        row["scenario"]
    )

    risk_class = str(
        row["risk_class"]
    )

    driver = str(
        row["primary_risk_driver"]
    ).lower()

    severity = str(
        row["contingency_severity"]
    )

    multiplier = 0.10


    # -----------------------------------------------------
    # Scenario stress
    # -----------------------------------------------------

    if scenario == "Demand Surge":

        multiplier += 0.10

    elif scenario == "Supplier Disruption":

        multiplier += 0.10

    elif scenario == "Combined Stress":

        multiplier += 0.20


    # -----------------------------------------------------
    # Risk driver
    # -----------------------------------------------------

    if "demand" in driver:

        multiplier += 0.05

    if "supplier" in driver:

        multiplier += 0.05


    # -----------------------------------------------------
    # Baseline risk
    # -----------------------------------------------------

    if risk_class == "Critical":

        multiplier += 0.15

    elif risk_class == "High":

        multiplier += 0.10

    elif risk_class == "Medium":

        multiplier += 0.05


    # -----------------------------------------------------
    # Exposure severity
    # -----------------------------------------------------

    if severity == "Critical":

        multiplier += 0.10

    elif severity == "High":

        multiplier += 0.05


    return multiplier


plans["contingency_multiplier"] = (
    plans.apply(
        contingency_multiplier,
        axis=1,
    )
)


# ---------------------------------------------------------
# Cap multiplier
# ---------------------------------------------------------

plans["contingency_multiplier"] = (

    plans["contingency_multiplier"]
    .clip(
        lower=0.10,
        upper=0.75,
    )

)


# =========================================================
# LOST SALES EXPOSURE
# =========================================================

plans["lost_sales_exposure"] = (

    plans[
        "lost_sales_increase_vs_optimized"
    ]
    .clip(
        lower=0
    )

)


# =========================================================
# ADDITIONAL SAFETY STOCK
# =========================================================
#
# Existing scenario_decisions already contains an initial
# additional_safety_stock_required value.
#
# We recompute it here using the actual optimized safety
# stock from inventory_optimization.csv.
#


base_safety_stock = (

    plans[
        "optimized_safety_stock"
    ]

)


scenario_buffer = (

    base_safety_stock
    *
    plans[
        "contingency_multiplier"
    ]

)


lost_sales_buffer = (

    plans[
        "lost_sales_exposure"
    ]
    *
    0.10

)


plans["calculated_additional_safety_stock"] = (

    scenario_buffer
    +
    lost_sales_buffer

)


plans[
    "calculated_additional_safety_stock"
] = (

    np.ceil(
        plans[
            "calculated_additional_safety_stock"
        ]
    )
    .astype(int)

)


# Use the larger of the scenario engine requirement
# and the recalculated contingency requirement.

plans["additional_safety_stock_required"] = (

    np.maximum(

        plans[
            "additional_safety_stock_required"
        ],

        plans[
            "calculated_additional_safety_stock"
        ],

    )

)


plans[
    "additional_safety_stock_required"
] = (

    np.ceil(
        plans[
            "additional_safety_stock_required"
        ]
    )
    .astype(int)

)


# =========================================================
# CONTINGENCY REORDER POINT
# =========================================================

plans["contingency_reorder_point"] = (

    plans[
        "optimized_reorder_point"
    ]
    +
    plans[
        "additional_safety_stock_required"
    ]

)


plans["contingency_reorder_point"] = (

    np.ceil(
        plans[
            "contingency_reorder_point"
        ]
    )
    .astype(int)

)


# =========================================================
# CONTINGENCY ORDER QUANTITY
# =========================================================

plans["contingency_order_quantity"] = (

    plans[
        "optimized_order_quantity"
    ]
    *
    (
        1
        +
        plans[
            "contingency_multiplier"
        ]
    )

)


plans["contingency_order_quantity"] = (

    np.ceil(
        plans[
            "contingency_order_quantity"
        ]
    )
    .astype(int)

)


# =========================================================
# CONTINGENCY ORDER COVERAGE
# =========================================================

plans[
    "contingency_order_coverage_days"
] = (

    plans[
        "optimized_order_coverage_days"
    ]
    *
    (
        1
        +
        plans[
            "contingency_multiplier"
        ]
    )

).round(2)


# =========================================================
# INVENTORY BUFFER
# =========================================================

plans["inventory_buffer_days"] = (

    plans[
        "contingency_order_coverage_days"
    ]
    -
    plans[
        "optimized_order_coverage_days"
    ]

).round(2)


# =========================================================
# SUPPLIER CONTINGENCY
# =========================================================

supplier_delay_rate = (

    plans[
        "optimized_supplier_delay_rate"
    ]
    .clip(
        lower=0
    )

)


supplier_delay = (

    plans[
        "optimized_average_supplier_delay"
    ]
    .clip(
        lower=0
    )

)


plans["supplier_exposure_score"] = (

    supplier_delay_rate
    *
    supplier_delay

)


maximum_supplier_score = (

    plans[
        "supplier_exposure_score"
    ].max()

)


if maximum_supplier_score > 0:

    plans["supplier_exposure_score"] = (

        plans[
            "supplier_exposure_score"
        ]
        /
        maximum_supplier_score

    )

else:

    plans["supplier_exposure_score"] = 0.0


# =========================================================
# CONTINGENCY ACTION
# =========================================================

def contingency_action(row):

    scenario = str(
        row["scenario"]
    )

    driver = str(
        row["primary_risk_driver"]
    ).lower()

    severity = str(
        row["contingency_severity"]
    )

    supplier_score = float(
        row["supplier_exposure_score"]
    )

    if severity == "Critical":

        if (
            scenario == "Supplier Disruption"
            or
            "supplier" in driver
        ):

            return (
                "Activate supplier contingency "
                "and increase inventory buffer"
            )

        if (
            scenario == "Demand Surge"
            or
            "demand" in driver
        ):

            return (
                "Activate demand contingency "
                "and increase safety stock"
            )

        return (
            "Activate full contingency plan "
            "and increase inventory protection"
        )


    if severity == "High":

        if (
            scenario == "Supplier Disruption"
            or
            supplier_score >= 0.50
            or
            "supplier" in driver
        ):

            return (
                "Prepare alternate supplier "
                "and increase inventory buffer"
            )

        if (
            scenario == "Demand Surge"
            or
            "demand" in driver
        ):

            return (
                "Increase safety stock "
                "for demand stress"
            )

        return (
            "Increase contingency inventory "
            "and monitor service"
        )


    if severity == "Moderate":

        if (
            "supplier" in driver
        ):

            return (
                "Monitor supplier exposure "
                "and maintain contingency buffer"
            )

        if (
            "demand" in driver
        ):

            return (
                "Monitor demand volatility "
                "and maintain inventory buffer"
            )

        return (
            "Monitor scenario exposure "
            "before changing policy"
        )


    return (
        "Monitor scenario exposure "
        "without immediate policy change"
    )


plans["contingency_action"] = (
    plans.apply(
        contingency_action,
        axis=1,
    )
)


# =========================================================
# CONTINGENCY RECOMMENDATION
# =========================================================

def contingency_recommendation(row):

    severity = str(
        row["contingency_severity"]
    )

    scenario = str(
        row["scenario"]
    )

    action = str(
        row["contingency_action"]
    )

    additional_stock = int(
        row[
            "additional_safety_stock_required"
        ]
    )

    if severity == "Critical":

        return (
            f"IMMEDIATE: {action}. "
            f"Prepare approximately "
            f"{additional_stock:,} additional units "
            f"of safety stock for {scenario}."
        )

    if severity == "High":

        return (
            f"HIGH PRIORITY: {action}. "
            f"Prepare approximately "
            f"{additional_stock:,} additional units "
            f"of contingency safety stock."
        )

    if severity == "Moderate":

        return (
            f"PLANNED: {action}. "
            f"Contingency buffer requirement is "
            f"approximately {additional_stock:,} units."
        )

    return (
        f"MONITOR: {action}."
    )


plans["contingency_recommendation"] = (
    plans.apply(
        contingency_recommendation,
        axis=1,
    )
)


# =========================================================
# DECISION URGENCY
# =========================================================

def decision_urgency(row):

    severity = str(
        row["contingency_severity"]
    )

    if severity == "Critical":

        return "Immediate"

    if severity == "High":

        return "High"

    if severity == "Moderate":

        return "Planned"

    return "Monitor"


plans["decision_urgency"] = (
    plans.apply(
        decision_urgency,
        axis=1,
    )
)


# =========================================================
# CONTINGENCY CONFIDENCE
# =========================================================

def contingency_confidence(row):

    score = 0

    severity = str(
        row["contingency_severity"]
    )

    relationship = str(
        row.get(
            "risk_scenario_relationship",
            "",
        )
    )

    lost_sales = float(
        row[
            "lost_sales_exposure"
        ]
    )

    final_score = float(
        row[
            "final_decision_score"
        ]
    )


    if severity == "Critical":

        score += 3

    elif severity == "High":

        score += 2

    elif severity == "Moderate":

        score += 1


    if relationship in {
        "Persistent high risk",
        "Hidden scenario vulnerability",
    }:

        score += 2


    if lost_sales > 0:

        score += 1


    if final_score >= 0.50:

        score += 1


    if score >= 5:

        return "High"

    if score >= 3:

        return "Medium"

    return "Low"


plans["contingency_confidence"] = (
    plans.apply(
        contingency_confidence,
        axis=1,
    )
)


# =========================================================
# EXECUTIVE CONTINGENCY DECISION
# =========================================================

def executive_decision(row):

    severity = str(
        row["contingency_severity"]
    )

    scenario = str(
        row["scenario"]
    )

    action = str(
        row["contingency_action"]
    )

    lost_sales = float(
        row[
            "lost_sales_exposure"
        ]
    )

    fill_change = float(
        row[
            "fill_rate_change_vs_optimized"
        ]
    )


    if severity == "Critical":

        return (
            f"URGENT CONTINGENCY DECISION: "
            f"{action}. "
            f"Under {scenario}, the product is exposed "
            f"to approximately "
            f"{lost_sales:,.0f} additional lost-sales units "
            f"and a {fill_change:+.2%} fill-rate change."
        )


    if severity == "High":

        return (
            f"HIGH PRIORITY CONTINGENCY DECISION: "
            f"{action}. "
            f"Prepare the recommended buffer before "
            f"{scenario} conditions materialize."
        )


    if severity == "Moderate":

        return (
            f"PLANNED CONTINGENCY DECISION: "
            f"{action}. "
            f"Monitor the product as {scenario} conditions "
            f"develop."
        )


    return (
        f"MONITORING DECISION: "
        f"Maintain current optimized policy while "
        f"monitoring {scenario} exposure."
    )


plans["executive_contingency_decision"] = (
    plans.apply(
        executive_decision,
        axis=1,
    )
)


# =========================================================
# CONTINGENCY COST EXPOSURE
# =========================================================

plans["contingency_inventory_cost"] = (

    plans[
        "additional_safety_stock_required"
    ]
    *
    (
        plans[
            "optimized_total_cost"
        ]
        /
        plans[
            "optimized_order_quantity"
        ].replace(
            0,
            np.nan,
        )
    )

)


plans["contingency_inventory_cost"] = (

    plans[
        "contingency_inventory_cost"
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .fillna(0)
    .clip(
        lower=0
    )

)


# =========================================================
# SCENARIO IMPACT SCORE
# =========================================================

lost_sales_component = (

    plans[
        "lost_sales_exposure"
    ]
    .clip(
        lower=0
    )

)


max_lost_sales = (

    lost_sales_component.max()

)


if max_lost_sales > 0:

    lost_sales_component = (

        lost_sales_component
        /
        max_lost_sales

    )


cost_component = (

    plans[
        "cost_change_vs_optimized"
    ]
    .clip(
        lower=0
    )

)


max_cost = (

    cost_component.max()

)


if max_cost > 0:

    cost_component = (

        cost_component
        /
        max_cost

    )


fill_rate_component = (

    plans[
        "fill_rate_change_vs_optimized"
    ]
    .abs()

)


max_fill_rate = (

    fill_rate_component.max()

)


if max_fill_rate > 0:

    fill_rate_component = (

        fill_rate_component
        /
        max_fill_rate

    )


plans["contingency_impact_score"] = (

    0.45 * lost_sales_component
    +
    0.25 * cost_component
    +
    0.20 * fill_rate_component
    +
    0.10 * plans[
        "final_decision_score"
    ].clip(
        lower=0,
        upper=1,
    )

)


plans["contingency_impact_score"] = (

    plans[
        "contingency_impact_score"
    ]
    .clip(
        lower=0,
        upper=1,
    )
    .round(4)

)


# =========================================================
# POLICY CHANGE FLAG
# =========================================================

plans["contingency_policy_change"] = np.where(

    plans[
        "contingency_severity"
    ].isin(
        [
            "Critical",
            "High",
        ]
    ),

    1,

    0,

)


# =========================================================
# VALIDATION
# =========================================================

print()
print("=" * 70)
print("CONTINGENCY PLANNING VALIDATION")
print("=" * 70)


assert len(plans) == len(
    scenario_decisions
)


assert plans[
    "product_id"
].nunique() == optimization[
    "product_id"
].nunique()


assert plans[
    "scenario"
].nunique() == scenario_decisions[
    "scenario"
].nunique()


assert plans[
    "optimized_safety_stock"
].notna().all()


assert plans[
    "contingency_reorder_point"
].notna().all()


assert plans[
    "contingency_order_quantity"
].notna().all()


assert plans[
    "additional_safety_stock_required"
].ge(0).all()


assert plans[
    "contingency_multiplier"
].between(
    0.10,
    0.75,
).all()


assert plans[
    "contingency_impact_score"
].between(
    0,
    1,
).all()


assert plans[
    "contingency_severity"
].isin(
    [
        "Critical",
        "High",
        "Moderate",
        "Low",
    ]
).all()


assert plans[
    "contingency_confidence"
].isin(
    [
        "High",
        "Medium",
        "Low",
    ]
).all()


assert plans[
    "contingency_policy_change"
].isin(
    [
        0,
        1,
    ]
).all()


print()
print("Validation checks passed.")

print(
    f"Scenario-product records: "
    f"{len(plans):,}"
)

print(
    f"Products represented: "
    f"{plans['product_id'].nunique():,}"
)

print(
    f"Scenarios represented: "
    f"{plans['scenario'].nunique():,}"
)


# =========================================================
# SORT
# =========================================================

severity_order = {

    "Critical": 1,
    "High": 2,
    "Moderate": 3,
    "Low": 4,

}


plans["severity_rank"] = (

    plans[
        "contingency_severity"
    ]
    .map(
        severity_order
    )

)


plans = (

    plans
    .sort_values(

        [
            "severity_rank",
            "contingency_impact_score",
            "lost_sales_exposure",
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


# =========================================================
# SAVE MAIN OUTPUT
# =========================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


plans.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# CONTINGENCY SUMMARY
# =========================================================

summary = (

    plans
    .groupby(
        "scenario",
        as_index=False,
    )
    .agg(

        products=(
            "product_id",
            "nunique",
        ),

        critical_exposures=(
            "contingency_severity",
            lambda x: (
                x == "Critical"
            ).sum(),
        ),

        high_exposures=(
            "contingency_severity",
            lambda x: (
                x == "High"
            ).sum(),
        ),

        moderate_exposures=(
            "contingency_severity",
            lambda x: (
                x == "Moderate"
            ).sum(),
        ),

        low_exposures=(
            "contingency_severity",
            lambda x: (
                x == "Low"
            ).sum(),
        ),

        total_lost_sales_exposure=(
            "lost_sales_exposure",
            "sum",
        ),

        total_contingency_safety_stock=(
            "additional_safety_stock_required",
            "sum",
        ),

        avg_contingency_safety_stock=(
            "additional_safety_stock_required",
            "mean",
        ),

        total_contingency_cost=(
            "contingency_inventory_cost",
            "sum",
        ),

        avg_fill_rate_change=(
            "fill_rate_change_vs_optimized",
            "mean",
        ),

        avg_contingency_impact=(
            "contingency_impact_score",
            "mean",
        ),

        policy_changes=(
            "contingency_policy_change",
            "sum",
        ),

    )

)


summary = (

    summary
    .sort_values(
        "total_lost_sales_exposure",
        ascending=False,
    )
    .reset_index(
        drop=True
    )

)


summary.to_csv(
    SUMMARY_OUTPUT_FILE,
    index=False,
)


# =========================================================
# TOP CONTINGENCY EXPOSURES
# =========================================================

top_columns = [

    "product_id",
    "scenario",

    "risk_class",
    "primary_risk_driver",

    "contingency_severity",

    "contingency_impact_score",

    "lost_sales_exposure",

    "cost_change_vs_optimized",

    "fill_rate_change_vs_optimized",

    "additional_safety_stock_required",

    "contingency_reorder_point",

    "contingency_order_quantity",

    "contingency_order_coverage_days",

    "contingency_action",

    "contingency_confidence",

]


top_exposures = (

    plans[
        top_columns
    ]
    .head(100)
    .copy()

)


top_exposures.to_csv(
    EXPOSURE_OUTPUT_FILE,
    index=False,
)


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("=" * 70)
print("CONTINGENCY PLANNING ENGINE COMPLETE")
print("=" * 70)

print()

print(
    f"Scenario-product records: "
    f"{len(plans):,}"
)

print(
    f"Products analyzed: "
    f"{plans['product_id'].nunique():,}"
)

print(
    f"Scenarios analyzed: "
    f"{plans['scenario'].nunique():,}"
)


# =========================================================
# EXPOSURE DISTRIBUTION
# =========================================================

print()
print("CONTINGENCY EXPOSURE DISTRIBUTION:")

print(

    plans[
        "contingency_severity"
    ]
    .value_counts()
    .reindex(
        [
            "Critical",
            "High",
            "Moderate",
            "Low",
        ],
        fill_value=0,
    )
    .to_string()

)


# =========================================================
# SCENARIO SUMMARY
# =========================================================

print()
print("SCENARIO CONTINGENCY SUMMARY:")
print()

print(

    summary[
        [
            "scenario",
            "products",
            "critical_exposures",
            "high_exposures",
            "moderate_exposures",
            "total_lost_sales_exposure",
            "total_contingency_safety_stock",
            "total_contingency_cost",
            "avg_fill_rate_change",
            "policy_changes",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )

)


# =========================================================
# TOP 20
# =========================================================

print()
print("=" * 70)
print("TOP 20 CONTINGENCY EXPOSURES")
print("=" * 70)

print()

print(

    plans[
        top_columns
    ]
    .head(20)
    .round(
        {
            "contingency_impact_score": 4,
            "cost_change_vs_optimized": 2,
            "fill_rate_change_vs_optimized": 4,
        }
    )
    .to_string(
        index=False
    )

)


# =========================================================
# CRITICAL EXPOSURES
# =========================================================

critical = plans[
    plans[
        "contingency_severity"
    ]
    ==
    "Critical"
]


print()
print("=" * 70)
print("CRITICAL CONTINGENCY EXPOSURES")
print("=" * 70)

print()

print(
    f"Critical scenario-product exposures: "
    f"{len(critical):,}"
)

print(
    f"Products affected: "
    f"{critical['product_id'].nunique():,}"
)

print(
    f"Additional safety stock required: "
    f"{critical['additional_safety_stock_required'].sum():,.0f}"
)

print(
    f"Lost-sales exposure: "
    f"{critical['lost_sales_exposure'].sum():,.0f}"
)


# =========================================================
# SUPPLIER CONTINGENCY
# =========================================================

supplier_contingency = plans[
    (
        plans["scenario"]
        ==
        "Supplier Disruption"
    )
    |
    (
        plans[
            "primary_risk_driver"
        ]
        .astype(str)
        .str.contains(
            "supplier",
            case=False,
            na=False,
        )
    )
]


print()
print("=" * 70)
print("SUPPLIER CONTINGENCY")
print("=" * 70)

print()

print(
    f"Supplier-exposed records: "
    f"{len(supplier_contingency):,}"
)

print(
    f"Supplier contingency safety stock: "
    f"{supplier_contingency['additional_safety_stock_required'].sum():,.0f}"
)

print(
    f"Supplier lost-sales exposure: "
    f"{supplier_contingency['lost_sales_exposure'].sum():,.0f}"
)


# =========================================================
# DEMAND CONTINGENCY
# =========================================================

demand_contingency = plans[
    (
        plans["scenario"]
        ==
        "Demand Surge"
    )
    |
    (
        plans[
            "primary_risk_driver"
        ]
        .astype(str)
        .str.contains(
            "demand",
            case=False,
            na=False,
        )
    )
]


print()
print("=" * 70)
print("DEMAND CONTINGENCY")
print("=" * 70)

print()

print(
    f"Demand-exposed records: "
    f"{len(demand_contingency):,}"
)

print(
    f"Demand contingency safety stock: "
    f"{demand_contingency['additional_safety_stock_required'].sum():,.0f}"
)

print(
    f"Demand lost-sales exposure: "
    f"{demand_contingency['lost_sales_exposure'].sum():,.0f}"
)


# =========================================================
# OUTPUT FILES
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
    SUMMARY_OUTPUT_FILE
)

print(
    EXPOSURE_OUTPUT_FILE
)

print()