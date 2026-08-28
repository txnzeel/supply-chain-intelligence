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
    / "optimization_decisions.csv"
)


# =========================================================
# LOAD OPTIMIZATION RESULTS
# =========================================================

optimization = pd.read_csv(INPUT_FILE)


required_columns = [
    "product_id",
    "category",
    "demand_class",
    "risk_class",
    "primary_risk_driver",

    "baseline_service_level",
    "optimized_service_level",

    "baseline_safety_stock",
    "optimized_safety_stock",

    "baseline_reorder_point",
    "optimized_reorder_point",

    "baseline_order_quantity",
    "optimized_order_quantity",

    "optimized_order_coverage_days",

    "baseline_fill_rate",
    "optimized_fill_rate",

    "baseline_stockout_rate",
    "optimized_stockout_rate",

    "baseline_lost_sales",
    "optimized_lost_sales",

    "baseline_inventory_days",
    "optimized_inventory_days",

    "baseline_total_cost",
    "optimized_total_cost",

    "lost_sales_reduction",
    "cost_change",
    "inventory_days_change",
    "fill_rate_change",

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


assert optimization["product_id"].notna().all()
assert optimization["product_id"].is_unique


# =========================================================
# COPY DATA
# =========================================================

decisions = optimization.copy()


# =========================================================
# COST SAVINGS
# =========================================================

decisions["cost_savings"] = (
    -decisions["cost_change"]
)


# =========================================================
# SERVICE IMPROVEMENT
# =========================================================

decisions["service_level_improvement"] = (
    decisions["fill_rate_change"]
)


# =========================================================
# LOST SALES IMPROVEMENT
# =========================================================

decisions["lost_sales_reduction_pct"] = np.where(

    decisions["baseline_lost_sales"] > 0,

    decisions["lost_sales_reduction"]
    /
    decisions["baseline_lost_sales"],

    0,
)


# =========================================================
# INVENTORY IMPACT
# =========================================================

decisions["inventory_days_change_pct"] = np.where(

    decisions["baseline_inventory_days"] > 0,

    decisions["inventory_days_change"]
    /
    decisions["baseline_inventory_days"],

    0,
)


# =========================================================
# ORDER QUANTITY CHANGE
# =========================================================

decisions["order_quantity_change"] = (

    decisions["optimized_order_quantity"]
    -
    decisions["baseline_order_quantity"]

)


decisions["order_quantity_change_pct"] = np.where(

    decisions["baseline_order_quantity"] > 0,

    decisions["order_quantity_change"]
    /
    decisions["baseline_order_quantity"],

    0,
)


# =========================================================
# SAFETY STOCK CHANGE
# =========================================================

decisions["safety_stock_change"] = (

    decisions["optimized_safety_stock"]
    -
    decisions["baseline_safety_stock"]

)


decisions["safety_stock_change_pct"] = np.where(

    decisions["baseline_safety_stock"] > 0,

    decisions["safety_stock_change"]
    /
    decisions["baseline_safety_stock"],

    0,
)


# =========================================================
# NEW — INVENTORY EFFICIENCY
# =========================================================

decisions["inventory_efficiency"] = np.where(

    decisions["inventory_days_change"] < 0,
    "Inventory reduction",

    np.where(
        decisions["inventory_days_change"] > 0,
        "Inventory increase",
        "Inventory unchanged",
    ),
)


# =========================================================
# NEW — SERVICE RISK
# =========================================================

decisions["service_risk"] = np.where(

    decisions["fill_rate_change"] >= 0.02,
    "Strong service improvement",

    np.where(
        decisions["fill_rate_change"] > 0,
        "Moderate service improvement",

        np.where(
            decisions["fill_rate_change"] < 0,
            "Service deterioration",
            "Service unchanged",
        ),
    ),
)


# =========================================================
# NEW — FINANCIAL EXPOSURE
# =========================================================

positive_cost_impact = (
    pd.to_numeric(decisions["cost_savings"], errors="coerce")
    .fillna(0)
    .gt(0)
)

positive_service_impact = (
    pd.to_numeric(decisions["lost_sales_reduction"], errors="coerce")
    .fillna(0)
    .gt(0)
)


decisions["financial_exposure"] = np.select(

    [
        positive_cost_impact & positive_service_impact,
        positive_service_impact & ~positive_cost_impact,
        positive_cost_impact,
    ],

    [
        "Positive economic + service impact",

        "Service improvement with additional cost",

        "Cost reduction",
    ],

    default="Limited financial benefit",
)

# =========================================================
# NEW — SUPPLIER RISK SCORE
# =========================================================

supplier_delay_rate = pd.to_numeric(

    decisions.get(
        "optimized_supplier_delay_rate",
        pd.Series(
            0,
            index=decisions.index,
        ),
    ),

    errors="coerce",
).fillna(0)


supplier_delay = pd.to_numeric(

    decisions.get(
        "optimized_average_supplier_delay",
        pd.Series(
            0,
            index=decisions.index,
        ),
    ),

    errors="coerce",
).fillna(0)


decisions["supplier_risk_score"] = (

    supplier_delay_rate
    *
    supplier_delay

)


max_supplier_score = (
    decisions["supplier_risk_score"].max()
)


if max_supplier_score > 0:

    decisions["supplier_risk_score"] = (

        decisions["supplier_risk_score"]
        /
        max_supplier_score

    )


# =========================================================
# NEW — RISK EXPLANATION
# =========================================================

def explain_risk(row):

    driver = str(
        row["primary_risk_driver"]
    ).lower()

    risk = str(
        row["risk_class"]
    )

    if "supplier" in driver:

        return (
            f"{risk} risk driven primarily by "
            "supplier reliability and lead-time uncertainty."
        )

    if "demand" in driver:

        return (
            f"{risk} risk driven primarily by "
            "demand variability and forecast uncertainty."
        )

    if "stockout" in driver:

        return (
            f"{risk} risk driven primarily by "
            "stockout exposure and insufficient inventory coverage."
        )

    if "inventory" in driver:

        return (
            f"{risk} risk driven primarily by "
            "inventory policy imbalance."
        )

    if "service" in driver:

        return (
            f"{risk} risk driven primarily by "
            "service-level exposure."
        )

    return (
        f"{risk} risk requiring inventory policy review."
    )


decisions["risk_explanation"] = (
    decisions.apply(
        explain_risk,
        axis=1,
    )
)


# =========================================================
# DECISION IMPACT SCORE
# =========================================================
#
# Measures overall business importance.
#
# Components:
#
# 35% lost-sales reduction
# 25% cost savings
# 20% service improvement
# 10% risk class
# 10% supplier exposure
#


def normalize_positive(series):

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)

    maximum = series.max()

    if maximum <= 0:
        return pd.Series(
            0,
            index=series.index,
            dtype=float,
        )

    return (
        series.clip(lower=0)
        /
        maximum
    )


lost_sales_score = normalize_positive(
    decisions["lost_sales_reduction"]
)


cost_savings_score = normalize_positive(
    decisions["cost_savings"]
)


service_score = normalize_positive(
    decisions["service_level_improvement"]
)


risk_score_map = {
    "Critical": 1.00,
    "High": 0.75,
    "Medium": 0.50,
    "Low": 0.25,
}


risk_score = (
    decisions["risk_class"]
    .map(risk_score_map)
    .fillna(0)
)


supplier_score = (
    decisions["supplier_risk_score"]
)


decisions["decision_impact_score"] = (

    0.35 * lost_sales_score
    +
    0.25 * cost_savings_score
    +
    0.20 * service_score
    +
    0.10 * risk_score
    +
    0.10 * supplier_score

)


# =========================================================
# PRIORITY CLASSIFICATION
# =========================================================

lost_sales_90th = (
    decisions[
        "lost_sales_reduction"
    ].quantile(0.90)
)


def assign_priority(row):

    if (
        row["risk_class"] == "Critical"
        and
        row["lost_sales_reduction"] > 0
    ):
        return "P1 - Critical"

    if (
        row["risk_class"] == "High"
        and
        row["lost_sales_reduction"] > 0
    ):
        return "P1 - High Impact"

    if (
        row["lost_sales_reduction"]
        >= lost_sales_90th
        and
        row["lost_sales_reduction"] > 0
    ):
        return "P1 - High Impact"

    if (
        row["risk_class"] == "Medium"
        and
        row["lost_sales_reduction"] > 0
    ):
        return "P2 - Medium"

    if row["lost_sales_reduction"] > 0:
        return "P3 - Low Impact"

    return "P4 - Monitor"


decisions["priority"] = (
    decisions.apply(
        assign_priority,
        axis=1,
    )
)


# =========================================================
# MANAGEMENT DECISION
# =========================================================

def assign_management_decision(row):

    action = str(
        row["recommended_action"]
    )

    if (
        row["priority"].startswith("P1")
        and
        row["lost_sales_reduction"] > 0
    ):
        return "Immediate intervention"

    if (
        action
        ==
        "Increase service level and adjust safety stock"
    ):
        return "Increase safety stock"

    if (
        action
        ==
        "Increase order quantity to reduce ordering frequency"
    ):
        return "Increase order quantity"

    if (
        action
        ==
        "Reduce order quantity and increase ordering frequency"
    ):
        return "Reduce order quantity"

    return "Maintain policy"


decisions["management_decision"] = (
    decisions.apply(
        assign_management_decision,
        axis=1,
    )
)


# =========================================================
# BUSINESS IMPACT CATEGORY
# =========================================================

def classify_impact(row):

    cost_savings = float(row["cost_savings"])
    lost_sales_reduction = float(row["lost_sales_reduction"])

    if cost_savings > 0 and lost_sales_reduction > 0:
        return "Cost saving + service improvement"

    elif cost_savings > 0 and lost_sales_reduction <= 0:
        return "Cost saving"

    elif cost_savings <= 0 and lost_sales_reduction > 0:
        return "Service improvement"

    else:
        return "Limited benefit"


decisions["business_impact"] = decisions.apply(
    classify_impact,
    axis=1,
)

# =========================================================
# POLICY CHANGE FLAG
# =========================================================

decisions["policy_change_flag"] = np.where(

    decisions["recommended_action"]
    ==
    "Maintain current inventory policy",

    0,

    1,
)


# =========================================================
# NEW — URGENCY SCORE
# =========================================================
#
# Combines:
# - risk
# - lost sales
# - service deterioration
# - supplier exposure
#


service_risk_score = np.where(

    decisions["fill_rate_change"] < 0,

    np.abs(
        decisions["fill_rate_change"]
    ),

    0,
)


service_risk_score = normalize_positive(
    pd.Series(
        service_risk_score,
        index=decisions.index,
    )
)


decisions["urgency_score"] = (

    0.40 * risk_score
    +
    0.30 * lost_sales_score
    +
    0.20 * service_risk_score
    +
    0.10 * supplier_score

)


# =========================================================
# NEW — RECOMMENDATION CONFIDENCE
# =========================================================

def confidence_level(row):

    score = 0

    if row["risk_class"] in {
        "Critical",
        "High",
    }:
        score += 2

    elif row["risk_class"] == "Medium":
        score += 1

    if row["lost_sales_reduction"] > 0:
        score += 1

    if row["fill_rate_change"] > 0:
        score += 1

    if row["cost_savings"] > 0:
        score += 1

    if (
        row["recommended_action"]
        !=
        "Maintain current inventory policy"
    ):
        score += 1

    if score >= 5:
        return "High"

    if score >= 3:
        return "Medium"

    return "Low"


decisions["recommendation_confidence"] = (
    decisions.apply(
        confidence_level,
        axis=1,
    )
)


# =========================================================
# NEW — DECISION RATIONALE
# =========================================================

def generate_rationale(row):

    action = row["recommended_action"]

    lost_sales = row[
        "lost_sales_reduction"
    ]

    cost = row[
        "cost_savings"
    ]

    fill_change = row[
        "fill_rate_change"
    ]

    inventory_change = row[
        "inventory_days_change"
    ]

    if (
        cost > 0
        and
        lost_sales > 0
    ):

        return (
            f"{action}. The optimization is expected to "
            f"reduce lost sales by {lost_sales:,.0f} units "
            f"while reducing modeled cost by "
            f"₹{cost:,.0f}. "
            f"Fill rate changes by {fill_change:+.2%}."
        )

    if lost_sales > 0:

        return (
            f"{action}. The primary benefit is improved "
            f"service, with expected lost-sales reduction "
            f"of {lost_sales:,.0f} units. "
            f"Modeled cost impact is "
            f"₹{cost:,.0f}."
        )

    if inventory_change < 0:

        return (
            f"{action}. The policy reduces inventory "
            f"exposure by {abs(inventory_change):.2f} "
            f"inventory days while maintaining the modeled "
            f"service outcome."
        )

    return (
        f"{action}. The modeled economics do not indicate "
        "a strong requirement for immediate intervention."
    )


decisions["decision_rationale"] = (
    decisions.apply(
        generate_rationale,
        axis=1,
    )
)


# =========================================================
# NEW — EXPECTED OUTCOME
# =========================================================

def expected_outcome(row):

    if (
        row["cost_savings"] > 0
        and
        row["lost_sales_reduction"] > 0
    ):

        return (
            "Lower cost with improved service"
        )

    if (
        row["lost_sales_reduction"] > 0
        and
        row["cost_savings"] <= 0
    ):

        return (
            "Improved service at additional cost"
        )

    if (
        row["inventory_days_change"] < 0
        and
        row["cost_savings"] > 0
    ):

        return (
            "Lean inventory with lower cost"
        )

    if row["cost_savings"] > 0:

        return (
            "Lower modeled inventory cost"
        )

    return (
        "Limited modeled improvement"
    )


decisions["expected_outcome"] = (
    decisions.apply(
        expected_outcome,
        axis=1,
    )
)


# =========================================================
# NEW — EXECUTIVE RECOMMENDATION
# =========================================================

def executive_recommendation(row):

    priority = row["priority"]
    action = row["management_decision"]

    if priority == "P1 - Critical":

        return (
            f"URGENT: {action}. "
            f"Review immediately due to Critical risk "
            f"and material service exposure."
        )

    if priority == "P1 - High Impact":

        return (
            f"HIGH PRIORITY: {action}. "
            f"Intervention is expected to materially "
            f"improve portfolio performance."
        )

    if priority == "P2 - Medium":

        return (
            f"PLANNED ACTION: {action}. "
            f"Include in the next inventory policy review."
        )

    if priority == "P3 - Low Impact":

        return (
            f"LOW PRIORITY: {action}. "
            f"Monitor performance before intervention."
        )

    return (
        "MONITOR: Maintain current policy unless "
        "operating conditions materially change."
    )


decisions["executive_recommendation"] = (
    decisions.apply(
        executive_recommendation,
        axis=1,
    )
)


# =========================================================
# VALIDATION
# =========================================================

print()
print("=" * 70)
print("DECISION ANALYSIS VALIDATION")
print("=" * 70)


assert len(optimization) == 2000, (
    f"Expected 2000 products, found {len(optimization)}"
)


valid_actions = {
    "Increase order quantity to reduce ordering frequency",
    "Maintain current inventory policy",
    "Reduce order quantity and increase ordering frequency",
    "Increase service level and adjust safety stock",
}


invalid_actions = set(
    optimization[
        "recommended_action"
    ].dropna().unique()
) - valid_actions


assert not invalid_actions, (
    f"Unexpected recommended actions: {invalid_actions}"
)


numeric_validation_columns = [
    "lost_sales_reduction",
    "cost_change",
    "inventory_days_change",
    "fill_rate_change",
]


for column in numeric_validation_columns:

    assert pd.api.types.is_numeric_dtype(
        optimization[column]
    ), (
        f"{column} must be numeric"
    )

    assert optimization[column].notna().all(), (
        f"{column} contains missing values"
    )


assert (
    optimization[
        "fill_rate_change"
    ].abs() <= 1
).all(), (
    "fill_rate_change contains invalid values"
)


# New validation

assert decisions[
    "decision_impact_score"
].between(
    0,
    1,
).all()


assert decisions[
    "urgency_score"
].between(
    0,
    1,
).all()


assert decisions[
    "recommendation_confidence"
].isin(
    [
        "High",
        "Medium",
        "Low",
    ]
).all()


assert decisions[
    "risk_explanation"
].notna().all()


assert decisions[
    "decision_rationale"
].notna().all()


assert decisions[
    "executive_recommendation"
].notna().all()


print()
print("Validation checks passed:")
print(
    f"Products analyzed: "
    f"{len(optimization):,}"
)
print(
    f"Risk classes: "
    f"{optimization['risk_class'].nunique()}"
)
print(
    f"Recommended actions: "
    f"{optimization['recommended_action'].nunique()}"
)
print(
    f"Decision columns: "
    f"{len(decisions.columns)}"
)

print()
print("Validation complete.")


# =========================================================
# ROUNDING
# =========================================================

round_columns = [
    "lost_sales_reduction_pct",
    "inventory_days_change_pct",
    "order_quantity_change_pct",
    "safety_stock_change_pct",
    "decision_impact_score",
    "supplier_risk_score",
    "urgency_score",
]


for column in round_columns:

    if column in decisions.columns:

        decisions[column] = (
            pd.to_numeric(
                decisions[column],
                errors="coerce",
            )
            .round(4)
        )


# =========================================================
# PRIORITY RANK
# =========================================================

priority_order = {
    "P1 - Critical": 1,
    "P1 - High Impact": 2,
    "P2 - Medium": 3,
    "P3 - Low Impact": 4,
    "P4 - Monitor": 5,
}


decisions["priority_rank"] = (
    decisions["priority"]
    .map(priority_order)
)


# =========================================================
# SORT
# =========================================================

decisions = (
    decisions
    .sort_values(
        [
            "priority_rank",
            "decision_impact_score",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# SAVE
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


decisions.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 70)
print("OPTIMIZATION DECISION ANALYSIS COMPLETE")
print("=" * 70)
print()

print(
    f"Products analyzed: "
    f"{len(decisions):,}"
)

print()

print("Priority distribution:")

print(
    decisions[
        "priority"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print()

print("Management decisions:")

print(
    decisions[
        "management_decision"
    ]
    .value_counts()
    .to_string()
)

print()

print("Business impact:")

print(
    decisions[
        "business_impact"
    ]
    .value_counts()
    .to_string()
)

print()

print("Recommendation confidence:")

print(
    decisions[
        "recommendation_confidence"
    ]
    .value_counts()
    .to_string()
)


# =========================================================
# TOP OPPORTUNITIES
# =========================================================

print()

print(
    "Top 20 optimization opportunities:"
)

print()

print(
    decisions[
        [
            "product_id",
            "risk_class",
            "primary_risk_driver",
            "priority",
            "decision_impact_score",
            "urgency_score",
            "lost_sales_reduction",
            "cost_savings",
            "fill_rate_change",
            "management_decision",
            "recommendation_confidence",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# =========================================================
# RISK CLASS SUMMARY
# =========================================================

print()

print(
    "Decision summary by risk class:"
)

print()

print(
    decisions
    .groupby(
        "risk_class",
        as_index=True,
    )
    .agg(

        products=(
            "product_id",
            "count",
        ),

        total_lost_sales_reduction=(
            "lost_sales_reduction",
            "sum",
        ),

        total_cost_savings=(
            "cost_savings",
            "sum",
        ),

        avg_fill_rate_change=(
            "fill_rate_change",
            "mean",
        ),

        avg_inventory_days_change=(
            "inventory_days_change",
            "mean",
        ),

        avg_decision_impact=(
            "decision_impact_score",
            "mean",
        ),

        avg_urgency=(
            "urgency_score",
            "mean",
        ),

        policy_changes=(
            "policy_change_flag",
            "sum",
        ),

    )
    .round(2)
    .sort_values(
        "total_lost_sales_reduction",
        ascending=False,
    )
    .to_string()
)


# =========================================================
# OUTPUT
# =========================================================

print()

print(
    "Saved to:"
)

print()

print(
    OUTPUT_FILE
)

print()
