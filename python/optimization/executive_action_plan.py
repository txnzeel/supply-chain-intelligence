import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# EXECUTIVE ACTION PLAN ENGINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# INPUT FILES
# ============================================================

OPTIMIZATION_FILE = (
    PROCESSED_DIR / "optimization_decisions.csv"
)

ECONOMIC_FILE = (
    PROCESSED_DIR / "economic_validation.csv"
)

SCENARIO_DECISIONS_FILE = (
    PROCESSED_DIR / "scenario_decisions.csv"
)

SCENARIO_SUMMARY_FILE = (
    PROCESSED_DIR / "scenario_decision_summary.csv"
)

CONTINGENCY_PLANS_FILE = (
    PROCESSED_DIR / "contingency_plans.csv"
)

CONTINGENCY_SUMMARY_FILE = (
    PROCESSED_DIR / "contingency_summary.csv"
)

CONTINGENCY_EXPOSURES_FILE = (
    PROCESSED_DIR / "contingency_exposures.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

ACTION_PLAN_FILE = (
    PROCESSED_DIR / "executive_action_plan.csv"
)

ACTION_SUMMARY_FILE = (
    PROCESSED_DIR / "executive_action_summary.csv"
)

EXECUTIVE_PRIORITIES_FILE = (
    PROCESSED_DIR / "executive_priorities.csv"
)

MANAGEMENT_BRIEF_FILE = (
    PROCESSED_DIR / "executive_management_brief.csv"
)


# ============================================================
# HELPERS
# ============================================================

def require_columns(df, columns, name):

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    assert not missing, (
        f"{name} is missing required columns: {missing}"
    )


def numeric_series(df, column, default=0):

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


def normalize_positive(series):

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)

    maximum = series.max()

    if maximum <= 0:

        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        series.clip(lower=0)
        /
        maximum
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("EXECUTIVE ACTION PLAN ENGINE")
print("=" * 70)
print()


# ============================================================
# LOAD DATA
# ============================================================

optimization = pd.read_csv(
    OPTIMIZATION_FILE
)

economic = pd.read_csv(
    ECONOMIC_FILE
)

scenario_decisions = pd.read_csv(
    SCENARIO_DECISIONS_FILE
)

scenario_summary = pd.read_csv(
    SCENARIO_SUMMARY_FILE
)

contingency_plans = pd.read_csv(
    CONTINGENCY_PLANS_FILE
)

contingency_summary = pd.read_csv(
    CONTINGENCY_SUMMARY_FILE
)

contingency_exposures = pd.read_csv(
    CONTINGENCY_EXPOSURES_FILE
)


print(
    f"Optimization decisions loaded: "
    f"{len(optimization):,}"
)

print(
    f"Economic validation records loaded: "
    f"{len(economic):,}"
)

print(
    f"Scenario decisions loaded: "
    f"{len(scenario_decisions):,}"
)

print(
    f"Scenario summary records loaded: "
    f"{len(scenario_summary):,}"
)

print(
    f"Contingency plans loaded: "
    f"{len(contingency_plans):,}"
)

print(
    f"Contingency summary records loaded: "
    f"{len(contingency_summary):,}"
)

print(
    f"Contingency exposures loaded: "
    f"{len(contingency_exposures):,}"
)


# ============================================================
# INPUT VALIDATION
# ============================================================

print()
print("=" * 70)
print("INPUT VALIDATION")
print("=" * 70)
print()


require_columns(
    optimization,
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
        "decision_rationale",
        "executive_recommendation",
    ],
    "optimization_decisions",
)


require_columns(
    scenario_decisions,
    [
        "product_id",
        "scenario",
        "risk_class",
        "primary_risk_driver",
        "exposure_class",
        "final_decision_score",
        "lost_sales_increase_vs_optimized",
        "cost_change_vs_optimized",
        "fill_rate_change_vs_optimized",
        "scenario_recommendation",
        "decision_confidence",
    ],
    "scenario_decisions",
)


require_columns(
    contingency_plans,
    [
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
    ],
    "contingency_plans",
)


assert optimization["product_id"].notna().all()

assert optimization["product_id"].is_unique

assert len(optimization) == 2000, (
    f"Expected 2,000 products, "
    f"found {len(optimization)}"
)

assert scenario_decisions["product_id"].nunique() == 2000

assert contingency_plans["product_id"].nunique() == 2000


print("Input validation passed.")


# ============================================================
# PREPARE CONTINGENCY DATA
# ============================================================

contingency = contingency_plans.copy()


contingency["contingency_impact_score"] = numeric_series(
    contingency,
    "contingency_impact_score",
)

contingency["lost_sales_exposure"] = numeric_series(
    contingency,
    "lost_sales_exposure",
)

contingency["cost_change_vs_optimized"] = numeric_series(
    contingency,
    "cost_change_vs_optimized",
)

contingency["fill_rate_change_vs_optimized"] = numeric_series(
    contingency,
    "fill_rate_change_vs_optimized",
)

contingency["additional_safety_stock_required"] = numeric_series(
    contingency,
    "additional_safety_stock_required",
)

contingency["contingency_order_quantity"] = numeric_series(
    contingency,
    "contingency_order_quantity",
)

contingency["contingency_order_coverage_days"] = numeric_series(
    contingency,
    "contingency_order_coverage_days",
)


# ============================================================
# SELECT WORST SCENARIO PER PRODUCT
# ============================================================

severity_rank = {
    "Critical": 4,
    "High": 3,
    "Moderate": 2,
    "Low": 1,
}


contingency["severity_rank"] = (
    contingency["contingency_severity"]
    .map(severity_rank)
    .fillna(0)
)


contingency = (
    contingency
    .sort_values(
        [
            "product_id",
            "severity_rank",
            "contingency_impact_score",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )
)


worst_contingency = (
    contingency
    .drop_duplicates(
        subset=["product_id"],
        keep="first",
    )
    .copy()
)


worst_contingency = worst_contingency[
    [
        "product_id",
        "scenario",
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
        "primary_risk_driver",
    ]
]


# ============================================================
# MERGE OPTIMIZATION + CONTINGENCY
# ============================================================

actions = optimization.copy()


actions = actions.merge(
    worst_contingency,
    on="product_id",
    how="left",
    suffixes=(
        "",
        "_contingency",
    ),
)


# ============================================================
# CONTINGENCY DEFAULTS
# ============================================================

actions["contingency_severity"] = (
    actions["contingency_severity"]
    .fillna("Low")
)


actions["contingency_impact_score"] = numeric_series(
    actions,
    "contingency_impact_score",
)


actions["lost_sales_exposure"] = numeric_series(
    actions,
    "lost_sales_exposure",
)


actions["cost_change_vs_optimized"] = numeric_series(
    actions,
    "cost_change_vs_optimized",
)


actions["fill_rate_change_vs_optimized"] = numeric_series(
    actions,
    "fill_rate_change_vs_optimized",
)


actions["additional_safety_stock_required"] = numeric_series(
    actions,
    "additional_safety_stock_required",
)


actions["contingency_reorder_point"] = numeric_series(
    actions,
    "contingency_reorder_point",
)


actions["contingency_order_quantity"] = numeric_series(
    actions,
    "contingency_order_quantity",
)


actions["contingency_order_coverage_days"] = numeric_series(
    actions,
    "contingency_order_coverage_days",
)


actions["contingency_action"] = (
    actions["contingency_action"]
    .fillna("Monitor scenario exposure")
)


actions["contingency_confidence"] = (
    actions["contingency_confidence"]
    .fillna("Low")
)


print()
print(
    "Optimization policy successfully merged "
    "with contingency exposure."
)

print(
    f"Products represented: "
    f"{actions['product_id'].nunique():,}"
)


# ============================================================
# ACTION TYPE
# ============================================================

def determine_action_type(row):

    severity = row["contingency_severity"]

    driver = str(
        row["primary_risk_driver"]
    ).lower()

    lost_sales = row["lost_sales_exposure"]

    safety_stock = row[
        "additional_safety_stock_required"
    ]

    if severity == "Critical":

        if "supplier" in driver:
            return "Supplier contingency"

        if "demand" in driver:
            return "Demand contingency"

        return "Critical inventory contingency"

    if severity == "High":

        if "supplier" in driver:
            return "Supplier risk mitigation"

        if "demand" in driver:
            return "Demand risk mitigation"

        return "Inventory risk mitigation"

    if lost_sales > 0 and safety_stock > 0:
        return "Inventory buffer planning"

    if row["management_decision"] != "Maintain policy":
        return "Inventory policy action"

    return "Monitor"


actions["action_type"] = actions.apply(
    determine_action_type,
    axis=1,
)


# ============================================================
# ACTION PRIORITY
# ============================================================

def determine_action_priority(row):

    severity = row["contingency_severity"]

    risk = row["risk_class"]

    decision_score = row[
        "decision_impact_score"
    ]

    urgency = row[
        "urgency_score"
    ]

    if severity == "Critical":
        return "P1 - Immediate"

    if (
        severity == "High"
        or risk in {"Critical", "High"}
        or urgency >= 0.60
        or decision_score >= 0.60
    ):
        return "P2 - High"

    if (
        severity == "Moderate"
        or urgency >= 0.30
        or decision_score >= 0.30
    ):
        return "P3 - Planned"

    return "P4 - Monitor"


actions["action_priority"] = actions.apply(
    determine_action_priority,
    axis=1,
)


# ============================================================
# ACTION OWNER
# ============================================================

def determine_owner(row):

    action_type = row["action_type"]

    if "Supplier" in action_type:
        return "Procurement"

    if "Demand" in action_type:
        return "Demand Planning"

    if "Inventory" in action_type:
        return "Inventory Planning"

    if "policy" in action_type.lower():
        return "Inventory Planning"

    return "Supply Chain Management"


actions["action_owner"] = actions.apply(
    determine_owner,
    axis=1,
)


# ============================================================
# ACTION HORIZON
# ============================================================

def determine_horizon(row):

    priority = row["action_priority"]

    if priority == "P1 - Immediate":
        return "0-7 days"

    if priority == "P2 - High":
        return "7-30 days"

    if priority == "P3 - Planned":
        return "30-90 days"

    return "Ongoing monitoring"


actions["action_horizon"] = actions.apply(
    determine_horizon,
    axis=1,
)


# ============================================================
# ACTION SCORE
# ============================================================

priority_score_map = {
    "P1 - Immediate": 1.00,
    "P2 - High": 0.75,
    "P3 - Planned": 0.50,
    "P4 - Monitor": 0.25,
}


priority_score = (
    actions["action_priority"]
    .map(priority_score_map)
    .fillna(0)
)


decision_score = normalize_positive(
    actions["decision_impact_score"]
)


urgency_score = normalize_positive(
    actions["urgency_score"]
)


contingency_score = normalize_positive(
    actions["contingency_impact_score"]
)


lost_sales_exposure_score = normalize_positive(
    actions["lost_sales_exposure"]
)


actions["executive_action_score"] = (

    0.30 * priority_score
    +
    0.25 * decision_score
    +
    0.20 * urgency_score
    +
    0.15 * contingency_score
    +
    0.10 * lost_sales_exposure_score

)


# ============================================================
# EXECUTIVE ACTION
# ============================================================

def generate_executive_action(row):

    severity = row[
        "contingency_severity"
    ]

    driver = str(
        row["primary_risk_driver"]
    ).lower()

    safety_stock = row[
        "additional_safety_stock_required"
    ]

    lost_sales = row[
        "lost_sales_exposure"
    ]

    action = row[
        "management_decision"
    ]

    if severity == "Critical":

        if "supplier" in driver:

            return (
                "Activate supplier contingency, "
                "secure alternate supply, and increase "
                f"inventory buffer by approximately "
                f"{safety_stock:,.0f} units."
            )

        if "demand" in driver:

            return (
                "Activate demand contingency and increase "
                f"safety stock by approximately "
                f"{safety_stock:,.0f} units."
            )

        return (
            "Activate immediate contingency response and "
            f"increase inventory buffer by approximately "
            f"{safety_stock:,.0f} units."
        )

    if severity == "High":

        if "supplier" in driver:

            return (
                "Prepare alternate supplier coverage and "
                f"increase inventory buffer by approximately "
                f"{safety_stock:,.0f} units."
            )

        if "demand" in driver:

            return (
                "Prepare demand contingency and increase "
                f"safety stock by approximately "
                f"{safety_stock:,.0f} units."
            )

        return (
            "Increase inventory buffer and monitor "
            "service performance closely."
        )

    if action != "Maintain policy":

        return (
            f"Execute optimization recommendation: {action}."
        )

    if lost_sales > 0:

        return (
            "Monitor scenario exposure and maintain "
            "contingency readiness."
        )

    return (
        "Maintain current policy and monitor operating "
        "conditions."
    )


actions["executive_action"] = actions.apply(
    generate_executive_action,
    axis=1,
)


# ============================================================
# EXPECTED BUSINESS OUTCOME
# ============================================================

def expected_business_outcome(row):

    cost_savings = numeric_series(
        pd.DataFrame([row]),
        "cost_savings",
    ).iloc[0]

    lost_sales_reduction = numeric_series(
        pd.DataFrame([row]),
        "lost_sales_reduction",
    ).iloc[0]

    contingency_exposure = row[
        "lost_sales_exposure"
    ]

    fill_change = row[
        "fill_rate_change"
    ]

    if (
        cost_savings > 0
        and lost_sales_reduction > 0
        and contingency_exposure <= 0
    ):

        return (
            "Lower cost and improved service"
        )

    if contingency_exposure > 0:

        return (
            "Protect service under adverse scenario"
        )

    if (
        lost_sales_reduction > 0
        and fill_change >= 0
    ):

        return (
            "Improved service"
        )

    if cost_savings > 0:

        return (
            "Lower modeled cost"
        )

    return (
        "Limited modeled improvement"
    )


actions["expected_business_outcome"] = actions.apply(
    expected_business_outcome,
    axis=1,
)


# ============================================================
# EXECUTIVE RATIONALE
# ============================================================

def generate_executive_rationale(row):

    priority = row[
        "action_priority"
    ]

    scenario = row[
        "scenario"
    ]

    severity = row[
        "contingency_severity"
    ]

    lost_sales = row[
        "lost_sales_exposure"
    ]

    safety_stock = row[
        "additional_safety_stock_required"
    ]

    cost = row[
        "cost_change_vs_optimized"
    ]

    fill_change = row[
        "fill_rate_change_vs_optimized"
    ]

    driver = row[
        "primary_risk_driver"
    ]

    if severity == "Critical":

        return (
            f"{priority} exposure under {scenario}. "
            f"Primary driver: {driver}. "
            f"Potential lost-sales exposure is "
            f"{lost_sales:,.0f} units with modeled "
            f"fill-rate deterioration of "
            f"{abs(fill_change):.2%}. "
            f"Approximately {safety_stock:,.0f} additional "
            "units of safety stock are indicated."
        )

    if severity == "High":

        return (
            f"{priority} scenario exposure under "
            f"{scenario}. Primary driver: {driver}. "
            f"Potential lost-sales exposure is "
            f"{lost_sales:,.0f} units. "
            f"Recommended contingency buffer is "
            f"{safety_stock:,.0f} units."
        )

    if row["management_decision"] != "Maintain policy":

        return (
            f"Optimization analysis recommends "
            f"{row['management_decision'].lower()}. "
            f"Modeled cost impact is "
            f"₹{row['cost_savings']:,.2f}, with lost-sales "
            f"change of "
            f"{row['lost_sales_reduction']:,.0f} units."
        )

    return (
        "No immediate intervention is required. "
        "Maintain policy and monitor scenario exposure."
    )


actions["executive_rationale"] = actions.apply(
    generate_executive_rationale,
    axis=1,
)


# ============================================================
# CONFIDENCE
# ============================================================

def determine_confidence(row):

    scores = 0

    if row["recommendation_confidence"] == "High":
        scores += 2

    elif row["recommendation_confidence"] == "Medium":
        scores += 1

    if row["contingency_confidence"] == "High":
        scores += 2

    elif row["contingency_confidence"] == "Medium":
        scores += 1

    if row["action_priority"] in {
        "P1 - Immediate",
        "P2 - High",
    }:
        scores += 1

    if row["contingency_impact_score"] > 0:
        scores += 1

    if scores >= 5:
        return "High"

    if scores >= 3:
        return "Medium"

    return "Low"


actions["executive_confidence"] = actions.apply(
    determine_confidence,
    axis=1,
)


# ============================================================
# SCENARIO RESPONSE
# ============================================================

def scenario_response(row):

    severity = row[
        "contingency_severity"
    ]

    driver = str(
        row["primary_risk_driver"]
    ).lower()

    if severity == "Critical":

        if "supplier" in driver:
            return "Activate supplier contingency"

        if "demand" in driver:
            return "Activate demand contingency"

        return "Activate inventory contingency"

    if severity == "High":

        if "supplier" in driver:
            return "Prepare alternate supplier"

        if "demand" in driver:
            return "Prepare demand response"

        return "Increase inventory buffer"

    if severity == "Moderate":
        return "Monitor scenario exposure"

    return "No immediate contingency action"


actions["scenario_response"] = actions.apply(
    scenario_response,
    axis=1,
)


# ============================================================
# POLICY CHANGE FLAG
# ============================================================

actions["policy_change_required"] = np.where(

    (
        actions["management_decision"]
        !=
        "Maintain policy"
    )
    |
    (
        actions["contingency_severity"]
        .isin(
            [
                "Critical",
                "High",
            ]
        )
    ),

    1,
    0,
)


# ============================================================
# ROUNDING
# ============================================================

round_columns = [

    "decision_impact_score",
    "urgency_score",
    "contingency_impact_score",
    "executive_action_score",

    "fill_rate_change",
    "fill_rate_change_vs_optimized",

    "cost_savings",
    "cost_change_vs_optimized",

    "lost_sales_reduction",
    "lost_sales_exposure",

]


for column in round_columns:

    if column in actions.columns:

        actions[column] = (
            pd.to_numeric(
                actions[column],
                errors="coerce",
            )
            .round(4)
        )


# ============================================================
# PRIORITY RANK
# ============================================================

actions["priority_rank"] = (
    actions["action_priority"]
    .map(
        {
            "P1 - Immediate": 1,
            "P2 - High": 2,
            "P3 - Planned": 3,
            "P4 - Monitor": 4,
        }
    )
    .fillna(5)
)


# ============================================================
# SORT
# ============================================================

actions = (
    actions
    .sort_values(
        [
            "priority_rank",
            "executive_action_score",
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


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("EXECUTIVE ACTION PLAN VALIDATION")
print("=" * 70)
print()


assert len(actions) == 2000, (
    f"Expected 2,000 action records, "
    f"found {len(actions)}"
)


assert actions["product_id"].is_unique


assert (
    actions["executive_action_score"]
    .between(0, 1)
    .all()
)


assert actions[
    "action_priority"
].isin(
    [
        "P1 - Immediate",
        "P2 - High",
        "P3 - Planned",
        "P4 - Monitor",
    ]
).all()


assert actions[
    "executive_confidence"
].isin(
    [
        "High",
        "Medium",
        "Low",
    ]
).all()


assert actions[
    "executive_action"
].notna().all()


assert actions[
    "executive_rationale"
].notna().all()


assert actions[
    "action_owner"
].notna().all()


assert actions[
    "action_horizon"
].notna().all()


print("Validation checks passed.")

print(
    f"Products represented: "
    f"{actions['product_id'].nunique():,}"
)

print(
    f"Action records: "
    f"{len(actions):,}"
)


# ============================================================
# SAVE MAIN ACTION PLAN
# ============================================================

actions.to_csv(
    ACTION_PLAN_FILE,
    index=False,
)


# ============================================================
# EXECUTIVE ACTION SUMMARY
# ============================================================

summary = (
    actions
    .groupby(
        "action_priority",
        as_index=False,
    )
    .agg(
        products=(
            "product_id",
            "count",
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

        total_lost_sales_exposure=(
            "lost_sales_exposure",
            "sum",
        ),

        total_additional_safety_stock=(
            "additional_safety_stock_required",
            "sum",
        ),

        total_cost_exposure=(
            "cost_change_vs_optimized",
            "sum",
        ),

        avg_action_score=(
            "executive_action_score",
            "mean",
        ),

        policy_changes=(
            "policy_change_required",
            "sum",
        ),
    )
)


summary["priority_rank"] = (
    summary["action_priority"]
    .map(
        {
            "P1 - Immediate": 1,
            "P2 - High": 2,
            "P3 - Planned": 3,
            "P4 - Monitor": 4,
        }
    )
)


summary = (
    summary
    .sort_values(
        "priority_rank"
    )
    .drop(
        columns=[
            "priority_rank"
        ]
    )
)


summary.to_csv(
    ACTION_SUMMARY_FILE,
    index=False,
)


# ============================================================
# EXECUTIVE PRIORITIES
# ============================================================

executive_priorities = (
    actions[
        [
            "product_id",
            "category",
            "risk_class",
            "primary_risk_driver",
            "scenario",
            "contingency_severity",
            "action_priority",
            "executive_action_score",
            "action_type",
            "action_owner",
            "action_horizon",
            "executive_action",
            "scenario_response",
            "additional_safety_stock_required",
            "lost_sales_exposure",
            "cost_change_vs_optimized",
            "executive_confidence",
            "executive_rationale",
        ]
    ]
    .head(100)
    .copy()
)


executive_priorities.to_csv(
    EXECUTIVE_PRIORITIES_FILE,
    index=False,
)


# ============================================================
# MANAGEMENT BRIEF
# ============================================================

total_products = len(actions)


p1_count = int(
    (
        actions["action_priority"]
        ==
        "P1 - Immediate"
    ).sum()
)


p2_count = int(
    (
        actions["action_priority"]
        ==
        "P2 - High"
    ).sum()
)


critical_count = int(
    (
        actions["contingency_severity"]
        ==
        "Critical"
    ).sum()
)


high_count = int(
    (
        actions["contingency_severity"]
        ==
        "High"
    ).sum()
)


total_safety_stock = float(
    actions[
        "additional_safety_stock_required"
    ].sum()
)


total_lost_sales_exposure = float(
    actions[
        "lost_sales_exposure"
    ].sum()
)


total_cost_exposure = float(
    actions[
        "cost_change_vs_optimized"
    ].sum()
)


supplier_actions = int(
    actions[
        "action_type"
    ]
    .isin(
        [
            "Supplier contingency",
            "Supplier risk mitigation",
        ]
    )
    .sum()
)


demand_actions = int(
    actions[
        "action_type"
    ]
    .isin(
        [
            "Demand contingency",
            "Demand risk mitigation",
        ]
    )
    .sum()
)


inventory_actions = int(
    actions[
        "action_type"
    ]
    .isin(
        [
            "Critical inventory contingency",
            "Inventory risk mitigation",
            "Inventory buffer planning",
            "Inventory policy action",
        ]
    )
    .sum()
)


management_brief = pd.DataFrame(
    [
        {
            "metric": "Products analyzed",
            "value": total_products,
        },

        {
            "metric": "P1 immediate actions",
            "value": p1_count,
        },

        {
            "metric": "P2 high-priority actions",
            "value": p2_count,
        },

        {
            "metric": "Critical contingency exposures",
            "value": critical_count,
        },

        {
            "metric": "High contingency exposures",
            "value": high_count,
        },

        {
            "metric": "Additional safety stock required",
            "value": round(
                total_safety_stock,
                2,
            ),
        },

        {
            "metric": "Potential lost-sales exposure",
            "value": round(
                total_lost_sales_exposure,
                2,
            ),
        },

        {
            "metric": "Total contingency cost exposure",
            "value": round(
                total_cost_exposure,
                2,
            ),
        },

        {
            "metric": "Supplier-focused actions",
            "value": supplier_actions,
        },

        {
            "metric": "Demand-focused actions",
            "value": demand_actions,
        },

        {
            "metric": "Inventory-focused actions",
            "value": inventory_actions,
        },
    ]
)


management_brief.to_csv(
    MANAGEMENT_BRIEF_FILE,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("EXECUTIVE ACTION PLAN COMPLETE")
print("=" * 70)
print()


print(
    f"Products analyzed:              "
    f"{total_products:,}"
)

print(
    f"P1 immediate actions:           "
    f"{p1_count:,}"
)

print(
    f"P2 high-priority actions:       "
    f"{p2_count:,}"
)

print(
    f"Critical exposures:             "
    f"{critical_count:,}"
)

print(
    f"High exposures:                 "
    f"{high_count:,}"
)

print(
    f"Additional safety stock:        "
    f"{total_safety_stock:,.0f}"
)

print(
    f"Potential lost-sales exposure:  "
    f"{total_lost_sales_exposure:,.0f}"
)

print(
    f"Contingency cost exposure:      "
    f"₹{total_cost_exposure:,.2f}"
)


# ============================================================
# ACTION DISTRIBUTION
# ============================================================

print()
print("ACTION PRIORITY DISTRIBUTION:")
print()

print(
    actions[
        "action_priority"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("ACTION TYPE DISTRIBUTION:")
print()

print(
    actions[
        "action_type"
    ]
    .value_counts()
    .to_string()
)


print()
print("ACTION OWNER DISTRIBUTION:")
print()

print(
    actions[
        "action_owner"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# TOP EXECUTIVE PRIORITIES
# ============================================================

print()
print("=" * 70)
print("TOP 20 EXECUTIVE PRIORITIES")
print("=" * 70)
print()


print(
    actions[
        [
            "product_id",
            "risk_class",
            "primary_risk_driver",
            "scenario",
            "contingency_severity",
            "action_priority",
            "executive_action_score",
            "action_owner",
            "executive_action",
            "executive_confidence",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# CRITICAL ACTIONS
# ============================================================

critical_actions = actions[
    actions[
        "action_priority"
    ]
    ==
    "P1 - Immediate"
]


print()
print("=" * 70)
print("CRITICAL MANAGEMENT ACTIONS")
print("=" * 70)
print()


print(
    f"Immediate action records: "
    f"{len(critical_actions):,}"
)


if len(critical_actions) > 0:

    print()

    print(
        critical_actions[
            [
                "product_id",
                "scenario",
                "risk_class",
                "primary_risk_driver",
                "contingency_severity",
                "additional_safety_stock_required",
                "lost_sales_exposure",
                "action_owner",
                "action_horizon",
                "executive_action",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)
print()

print(
    ACTION_PLAN_FILE
)

print(
    ACTION_SUMMARY_FILE
)

print(
    EXECUTIVE_PRIORITIES_FILE
)

print(
    MANAGEMENT_BRIEF_FILE
)

print()