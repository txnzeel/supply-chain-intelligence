import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

DECISION_INPUT = DATA_DIR / "optimization_decisions.csv"
SCENARIO_INPUT = DATA_DIR / "scenario_analysis.csv"
OPPORTUNITY_INPUT = DATA_DIR / "scenario_opportunities.csv"

OUTPUT_FILE = DATA_DIR / "scenario_decisions.csv"
SUMMARY_FILE = DATA_DIR / "scenario_decision_summary.csv"


# =========================================================
# HELPERS
# =========================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


def numeric(df, column, default=0):
    """
    Safely convert a column to numeric.
    If the column does not exist, return a zero Series.
    """

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


def first_existing(df, columns, default=0):
    """
    Return the first existing column from a list.
    """

    for column in columns:
        if column in df.columns:
            return df[column]

    return pd.Series(
        default,
        index=df.index,
    )


def normalize_positive(series):
    """
    Normalize positive values between 0 and 1.
    """

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
        / maximum
    )


# =========================================================
# VALIDATE INPUT FILES
# =========================================================

require_file(DECISION_INPUT)
require_file(SCENARIO_INPUT)
require_file(OPPORTUNITY_INPUT)


# =========================================================
# LOAD DATA
# =========================================================

decisions = pd.read_csv(
    DECISION_INPUT
)

scenario = pd.read_csv(
    SCENARIO_INPUT
)

opportunities = pd.read_csv(
    OPPORTUNITY_INPUT
)


print()
print("=" * 70)
print("SCENARIO DECISION ENGINE")
print("=" * 70)

print()
print(
    f"Optimization decisions loaded: "
    f"{len(decisions):,}"
)

print(
    f"Scenario records loaded: "
    f"{len(scenario):,}"
)

print(
    f"Scenario opportunities loaded: "
    f"{len(opportunities):,}"
)


# =========================================================
# BASIC VALIDATION
# =========================================================

if "product_id" not in decisions.columns:
    raise ValueError(
        "optimization_decisions.csv must contain product_id"
    )

if "product_id" not in scenario.columns:
    raise ValueError(
        "scenario_analysis.csv must contain product_id"
    )

if "scenario" not in scenario.columns:
    raise ValueError(
        "scenario_analysis.csv must contain scenario"
    )


if decisions["product_id"].duplicated().any():
    raise ValueError(
        "optimization_decisions.csv contains duplicate product_id values."
    )


# =========================================================
# PREPARE BASELINE DECISION DATA
# =========================================================

decision_columns = [
    "product_id",
    "category",
    "demand_class",
    "risk_class",
    "primary_risk_driver",
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
    "decision_impact_score",
    "urgency_score",
    "recommendation_confidence",
    "management_decision",
]


decision_columns = [
    column
    for column in decision_columns
    if column in decisions.columns
]


base = decisions[
    decision_columns
].copy()


# =========================================================
# MERGE BASELINE DECISIONS INTO SCENARIO DATA
# =========================================================

scenario_decisions = scenario.merge(
    base,
    on="product_id",
    how="left",
    suffixes=("", "_decision"),
)


# =========================================================
# REMOVE DUPLICATE / CONFLICTING COLUMNS
# =========================================================

for column in [
    "category",
    "demand_class",
    "risk_class",
    "primary_risk_driver",
]:

    decision_version = f"{column}_decision"

    if (
        column not in scenario_decisions.columns
        and
        decision_version in scenario_decisions.columns
    ):

        scenario_decisions[column] = (
            scenario_decisions[
                decision_version
            ]
        )

    if decision_version in scenario_decisions.columns:

        scenario_decisions.drop(
            columns=[decision_version],
            inplace=True,
        )


# =========================================================
# SCENARIO METRICS
# =========================================================

scenario_decisions[
    "scenario_cost"
] = numeric(
    scenario_decisions,
    "scenario_total_cost",
)

if (
    scenario_decisions[
        "scenario_cost"
    ].eq(0).all()
):

    scenario_decisions[
        "scenario_cost"
    ] = numeric(
        scenario_decisions,
        "cost",
    )


scenario_decisions[
    "scenario_lost_sales"
] = numeric(
    scenario_decisions,
    "scenario_lost_sales",
)

if (
    scenario_decisions[
        "scenario_lost_sales"
    ].eq(0).all()
):

    scenario_decisions[
        "scenario_lost_sales"
    ] = numeric(
        scenario_decisions,
        "lost_sales",
    )


scenario_decisions[
    "scenario_fill_rate"
] = numeric(
    scenario_decisions,
    "scenario_fill_rate",
)

if (
    scenario_decisions[
        "scenario_fill_rate"
    ].eq(0).all()
):

    scenario_decisions[
        "scenario_fill_rate"
    ] = numeric(
        scenario_decisions,
        "fill_rate",
    )


# =========================================================
# OPTIMIZED BASELINE
# =========================================================

scenario_decisions[
    "optimized_total_cost_base"
] = numeric(
    scenario_decisions,
    "optimized_total_cost",
)


scenario_decisions[
    "optimized_lost_sales_base"
] = numeric(
    scenario_decisions,
    "optimized_lost_sales",
)


scenario_decisions[
    "optimized_fill_rate_base"
] = numeric(
    scenario_decisions,
    "optimized_fill_rate",
)


scenario_decisions[
    "optimized_safety_stock_base"
] = numeric(
    scenario_decisions,
    "optimized_safety_stock",
)


scenario_decisions[
    "optimized_order_quantity_base"
] = numeric(
    scenario_decisions,
    "optimized_order_quantity",
)


scenario_decisions[
    "optimized_inventory_days_base"
] = numeric(
    scenario_decisions,
    "optimized_inventory_days",
)


# =========================================================
# SCENARIO IMPACT
# =========================================================

scenario_decisions[
    "lost_sales_increase_vs_optimized"
] = (
    scenario_decisions[
        "scenario_lost_sales"
    ]
    -
    scenario_decisions[
        "optimized_lost_sales_base"
    ]
)


scenario_decisions[
    "cost_change_vs_optimized"
] = (
    scenario_decisions[
        "scenario_cost"
    ]
    -
    scenario_decisions[
        "optimized_total_cost_base"
    ]
)


scenario_decisions[
    "fill_rate_change_vs_optimized"
] = (
    scenario_decisions[
        "scenario_fill_rate"
    ]
    -
    scenario_decisions[
        "optimized_fill_rate_base"
    ]
)


# =========================================================
# RELATIVE IMPACT
# =========================================================

scenario_decisions[
    "lost_sales_increase_pct"
] = np.where(

    scenario_decisions[
        "optimized_lost_sales_base"
    ] > 0,

    scenario_decisions[
        "lost_sales_increase_vs_optimized"
    ]
    /
    scenario_decisions[
        "optimized_lost_sales_base"
    ],

    0,
)


scenario_decisions[
    "cost_increase_pct"
] = np.where(

    scenario_decisions[
        "optimized_total_cost_base"
    ] > 0,

    scenario_decisions[
        "cost_change_vs_optimized"
    ]
    /
    scenario_decisions[
        "optimized_total_cost_base"
    ],

    0,
)


# =========================================================
# SUPPLIER EXPOSURE
# =========================================================

supplier_delay_rate = numeric(
    scenario_decisions,
    "optimized_supplier_delay_rate",
)

supplier_delay = numeric(
    scenario_decisions,
    "optimized_average_supplier_delay",
)


scenario_decisions[
    "supplier_exposure"
] = (
    supplier_delay_rate
    *
    supplier_delay
)


scenario_decisions[
    "supplier_exposure_score"
] = normalize_positive(
    scenario_decisions[
        "supplier_exposure"
    ]
)


# =========================================================
# SCENARIO SEVERITY COMPONENTS
# =========================================================

lost_sales_component = normalize_positive(
    scenario_decisions[
        "lost_sales_increase_vs_optimized"
    ]
)


cost_component = normalize_positive(
    scenario_decisions[
        "cost_change_vs_optimized"
    ]
)


service_component = normalize_positive(
    -scenario_decisions[
        "fill_rate_change_vs_optimized"
    ]
)


risk_map = {
    "Critical": 1.00,
    "High": 0.75,
    "Medium": 0.50,
    "Low": 0.25,
}


scenario_decisions[
    "baseline_risk_score"
] = (
    scenario_decisions[
        "risk_class"
    ]
    .map(risk_map)
    .fillna(0)
)


# =========================================================
# SCENARIO EXPOSURE SCORE
# =========================================================
#
# 35% lost-sales exposure
# 25% service deterioration
# 15% incremental cost
# 15% baseline risk
# 10% supplier exposure
#

scenario_decisions[
    "scenario_exposure_score"
] = (

    0.35
    *
    lost_sales_component

    +

    0.25
    *
    service_component

    +

    0.15
    *
    cost_component

    +

    0.15
    *
    scenario_decisions[
        "baseline_risk_score"
    ]

    +

    0.10
    *
    scenario_decisions[
        "supplier_exposure_score"
    ]

)


scenario_decisions[
    "scenario_exposure_score"
] = (
    scenario_decisions[
        "scenario_exposure_score"
    ]
    .clip(
        lower=0,
        upper=1,
    )
)


# =========================================================
# EXPOSURE CLASSIFICATION
# =========================================================

def classify_exposure(score):

    if score >= 0.70:
        return "Critical exposure"

    if score >= 0.45:
        return "High exposure"

    if score >= 0.20:
        return "Moderate exposure"

    return "Low exposure"


scenario_decisions[
    "exposure_class"
] = (
    scenario_decisions[
        "scenario_exposure_score"
    ]
    .apply(
        classify_exposure
    )
)


# =========================================================
# ECONOMIC EXPOSURE
# =========================================================
#
# Positive = additional cost
# Negative = modeled cost reduction
#
# Lost sales are also converted into a relative
# exposure measure.
#


scenario_decisions[
    "economic_exposure"
] = (

    scenario_decisions[
        "cost_change_vs_optimized"
    ]
)


# =========================================================
# CONTINGENCY BUFFER
# =========================================================
#
# We don't pretend this is a mathematically optimal
# safety stock calculation. It is a scenario buffer
# recommendation based on stress severity.
#
# Critical  = 30%
# High      = 20%
# Moderate  = 10%
# Low       = 0%
#


buffer_rate = np.select(

    [
        scenario_decisions[
            "exposure_class"
        ]
        == "Critical exposure",

        scenario_decisions[
            "exposure_class"
        ]
        == "High exposure",

        scenario_decisions[
            "exposure_class"
        ]
        == "Moderate exposure",
    ],

    [
        0.30,
        0.20,
        0.10,
    ],

    default=0.0,
)


scenario_decisions[
    "recommended_buffer_rate"
] = buffer_rate


scenario_decisions[
    "stress_adjusted_safety_stock"
] = (

    scenario_decisions[
        "optimized_safety_stock_base"
    ]
    *
    (
        1
        +
        scenario_decisions[
            "recommended_buffer_rate"
        ]
    )

)


scenario_decisions[
    "additional_safety_stock_required"
] = (

    scenario_decisions[
        "stress_adjusted_safety_stock"
    ]
    -
    scenario_decisions[
        "optimized_safety_stock_base"
    ]

)


scenario_decisions[
    "additional_safety_stock_required"
] = (
    scenario_decisions[
        "additional_safety_stock_required"
    ]
    .clip(lower=0)
    .round()
)


# =========================================================
# CONTINGENCY ORDER QUANTITY
# =========================================================

scenario_decisions[
    "stress_adjusted_order_quantity"
] = (

    scenario_decisions[
        "optimized_order_quantity_base"
    ]
    *
    (
        1
        +
        scenario_decisions[
            "recommended_buffer_rate"
        ]
    )

)


scenario_decisions[
    "additional_order_quantity_required"
] = (

    scenario_decisions[
        "stress_adjusted_order_quantity"
    ]
    -
    scenario_decisions[
        "optimized_order_quantity_base"
    ]

)


scenario_decisions[
    "additional_order_quantity_required"
] = (
    scenario_decisions[
        "additional_order_quantity_required"
    ]
    .clip(lower=0)
    .round()
)


# =========================================================
# SCENARIO-SPECIFIC MITIGATION
# =========================================================

def recommend_mitigation(row):

    scenario_name = str(
        row["scenario"]
    ).lower()

    driver = str(
        row.get(
            "primary_risk_driver",
            "",
        )
    ).lower()

    exposure = row[
        "exposure_class"
    ]

    lost_sales = row[
        "lost_sales_increase_vs_optimized"
    ]

    service_change = row[
        "fill_rate_change_vs_optimized"
    ]

    supplier_score = row[
        "supplier_exposure_score"
    ]

    # -----------------------------------------------------
    # CRITICAL
    # -----------------------------------------------------

    if exposure == "Critical exposure":

        if (
            "supplier" in driver
            or
            "supplier" in scenario_name
        ):

            return (
                "Immediate supplier contingency "
                "planning"
            )

        if (
            "demand" in driver
            or
            "surge" in scenario_name
        ):

            return (
                "Increase safety stock and "
                "prepare demand contingency"
            )

        if service_change < -0.02:

            return (
                "Immediate service protection "
                "and inventory buffer"
            )

        return (
            "Immediate contingency planning"
        )

    # -----------------------------------------------------
    # HIGH
    # -----------------------------------------------------

    if exposure == "High exposure":

        if (
            supplier_score >= 0.50
            or
            "supplier" in scenario_name
        ):

            return (
                "Increase supplier monitoring "
                "and establish backup supply"
            )

        if (
            "demand" in driver
            or
            "surge" in scenario_name
        ):

            return (
                "Increase safety stock "
                "for demand stress"
            )

        if lost_sales > 0:

            return (
                "Increase inventory buffer "
                "and monitor service"
            )

        return (
            "Planned contingency review"
        )

    # -----------------------------------------------------
    # MODERATE
    # -----------------------------------------------------

    if exposure == "Moderate exposure":

        if (
            supplier_score >= 0.50
        ):

            return (
                "Increase supplier monitoring"
            )

        if lost_sales > 0:

            return (
                "Review safety-stock buffer"
            )

        return (
            "Monitor scenario exposure"
        )

    # -----------------------------------------------------
    # LOW
    # -----------------------------------------------------

    return (
        "Maintain policy and monitor"
    )


scenario_decisions[
    "scenario_recommendation"
] = (
    scenario_decisions.apply(
        recommend_mitigation,
        axis=1,
    )
)


# =========================================================
# ACTION PRIORITY
# =========================================================

def assign_action_priority(row):

    exposure = row[
        "exposure_class"
    ]

    scenario_name = str(
        row["scenario"]
    ).lower()

    if exposure == "Critical exposure":

        return "P1 - Immediate"

    if exposure == "High exposure":

        return "P2 - High"

    if exposure == "Moderate exposure":

        return "P3 - Planned"

    if "combined" in scenario_name:

        return "P3 - Planned"

    return "P4 - Monitor"


scenario_decisions[
    "action_priority"
] = (
    scenario_decisions.apply(
        assign_action_priority,
        axis=1,
    )
)


# =========================================================
# BASELINE RISK VS SCENARIO EXPOSURE
# =========================================================

def risk_scenario_relationship(row):

    baseline = row[
        "risk_class"
    ]

    exposure = row[
        "exposure_class"
    ]

    if (
        baseline == "Low"
        and
        exposure
        in {
            "Critical exposure",
            "High exposure",
        }
    ):

        return (
            "Hidden scenario vulnerability"
        )

    if (
        baseline in {
            "Critical",
            "High",
        }
        and
        exposure
        in {
            "Critical exposure",
            "High exposure",
        }
    ):

        return (
            "Persistent high risk"
        )

    if (
        baseline in {
            "Critical",
            "High",
        }
        and
        exposure == "Low exposure"
    ):

        return (
            "Scenario resilient"
        )

    if (
        baseline == "Low"
        and
        exposure == "Low exposure"
    ):

        return (
            "Low baseline and scenario risk"
        )

    return (
        "Scenario-sensitive"
    )


scenario_decisions[
    "risk_scenario_relationship"
] = (
    scenario_decisions.apply(
        risk_scenario_relationship,
        axis=1,
    )
)


# =========================================================
# SCENARIO RATIONALE
# =========================================================

def generate_rationale(row):

    scenario_name = row[
        "scenario"
    ]

    exposure = row[
        "exposure_class"
    ]

    lost_sales = row[
        "lost_sales_increase_vs_optimized"
    ]

    cost = row[
        "cost_change_vs_optimized"
    ]

    fill_change = row[
        "fill_rate_change_vs_optimized"
    ]

    buffer = row[
        "additional_safety_stock_required"
    ]

    recommendation = row[
        "scenario_recommendation"
    ]

    relationship = row[
        "risk_scenario_relationship"
    ]

    if lost_sales < 0:
        lost_sales_text = (
            "lost sales are lower than the optimized baseline"
        )
    else:
        lost_sales_text = (
            f"lost sales increase by "
            f"{lost_sales:,.0f} units"
        )

    if cost >= 0:
        cost_text = (
            f"modeled cost increases by "
            f"${cost:,.0f}"
        )
    else:
        cost_text = (
            f"modeled cost decreases by "
            f"${abs(cost):,.0f}"
        )

    return (
        f"{scenario_name} creates {exposure.lower()} "
        f"for this product. {lost_sales_text}, while "
        f"{cost_text}. Fill rate changes by "
        f"{fill_change:+.2%}. The product is classified "
        f"as '{relationship}'. Recommended action: "
        f"{recommendation}. Stress-adjusted safety stock "
        f"requires approximately {buffer:,.0f} additional "
        f"units."
    )


scenario_decisions[
    "scenario_rationale"
] = (
    scenario_decisions.apply(
        generate_rationale,
        axis=1,
    )
)


# =========================================================
# DECISION CONFIDENCE
# =========================================================

def decision_confidence(row):

    score = 0

    if row[
        "scenario_exposure_score"
    ] >= 0.70:

        score += 3

    elif row[
        "scenario_exposure_score"
    ] >= 0.45:

        score += 2

    elif row[
        "scenario_exposure_score"
    ] >= 0.20:

        score += 1


    if row[
        "lost_sales_increase_vs_optimized"
    ] > 0:

        score += 1


    if row[
        "fill_rate_change_vs_optimized"
    ] < 0:

        score += 1


    if row[
        "cost_change_vs_optimized"
    ] > 0:

        score += 1


    if row[
        "risk_scenario_relationship"
    ] == "Hidden scenario vulnerability":

        score += 1


    if score >= 5:
        return "High"

    if score >= 3:
        return "Medium"

    return "Low"


scenario_decisions[
    "decision_confidence"
] = (
    scenario_decisions.apply(
        decision_confidence,
        axis=1,
    )
)


# =========================================================
# OPPORTUNITY DATA
# =========================================================

opportunity_columns = [
    "product_id",
    "scenario",
    "scenario_opportunity_score",
]


available_opportunity_columns = [
    column
    for column in opportunity_columns
    if column in opportunities.columns
]


if (
    "product_id" in available_opportunity_columns
    and
    "scenario" in available_opportunity_columns
):

    opportunity_data = opportunities[
        available_opportunity_columns
    ].copy()

    opportunity_data = (
        opportunity_data
        .drop_duplicates(
            subset=[
                "product_id",
                "scenario",
            ]
        )
    )

    scenario_decisions = scenario_decisions.merge(
        opportunity_data,
        on=[
            "product_id",
            "scenario",
        ],
        how="left",
    )

else:

    scenario_decisions[
        "scenario_opportunity_score"
    ] = 0


scenario_decisions[
    "scenario_opportunity_score"
] = numeric(
    scenario_decisions,
    "scenario_opportunity_score",
)


# =========================================================
# FINAL DECISION SCORE
# =========================================================

scenario_decisions[
    "final_decision_score"
] = (

    0.60
    *
    scenario_decisions[
        "scenario_exposure_score"
    ]

    +

    0.20
    *
    normalize_positive(
        scenario_decisions[
            "scenario_opportunity_score"
        ]
    )

    +

    0.10
    *
    scenario_decisions[
        "baseline_risk_score"
    ]

    +

    0.10
    *
    scenario_decisions[
        "supplier_exposure_score"
    ]

)


scenario_decisions[
    "final_decision_score"
] = (
    scenario_decisions[
        "final_decision_score"
    ]
    .clip(
        lower=0,
        upper=1,
    )
)


# =========================================================
# SCENARIO ORDER
# =========================================================

scenario_order = {
    "Optimized Policy": 1,
    "Demand Surge": 2,
    "Supplier Disruption": 3,
    "Combined Stress": 4,
}


scenario_decisions[
    "scenario_rank"
] = (
    scenario_decisions[
        "scenario"
    ]
    .map(scenario_order)
    .fillna(99)
)


# =========================================================
# VALIDATION
# =========================================================

print()
print("=" * 70)
print("SCENARIO DECISION ENGINE VALIDATION")
print("=" * 70)


assert (
    scenario_decisions[
        "product_id"
    ].notna().all()
)


assert (
    scenario_decisions[
        "scenario"
    ].notna().all()
)


assert (
    scenario_decisions[
        "scenario_exposure_score"
    ].between(
        0,
        1,
    ).all()
)


assert (
    scenario_decisions[
        "baseline_risk_score"
    ].between(
        0,
        1,
    ).all()
)


assert (
    scenario_decisions[
        "supplier_exposure_score"
    ].between(
        0,
        1,
    ).all()
)


assert (
    scenario_decisions[
        "final_decision_score"
    ].between(
        0,
        1,
    ).all()
)


assert (
    scenario_decisions[
        "exposure_class"
    ].isin(
        [
            "Critical exposure",
            "High exposure",
            "Moderate exposure",
            "Low exposure",
        ]
    ).all()
)


assert (
    scenario_decisions[
        "decision_confidence"
    ].isin(
        [
            "High",
            "Medium",
            "Low",
        ]
    ).all()
)


assert (
    scenario_decisions[
        "scenario_recommendation"
    ].notna().all()
)


assert (
    scenario_decisions[
        "scenario_rationale"
    ].notna().all()
)


print()
print("Validation checks passed.")

print(
    f"Scenario-product records: "
    f"{len(scenario_decisions):,}"
)

print(
    f"Products represented: "
    f"{scenario_decisions['product_id'].nunique():,}"
)

print(
    f"Scenarios represented: "
    f"{scenario_decisions['scenario'].nunique():,}"
)


# =========================================================
# ROUNDING
# =========================================================

round_columns = [
    "scenario_exposure_score",
    "supplier_exposure_score",
    "baseline_risk_score",
    "final_decision_score",
    "lost_sales_increase_pct",
    "cost_increase_pct",
    "fill_rate_change_vs_optimized",
    "recommended_buffer_rate",
]


for column in round_columns:

    if column in scenario_decisions.columns:

        scenario_decisions[column] = (
            pd.to_numeric(
                scenario_decisions[column],
                errors="coerce",
            )
            .round(4)
        )


scenario_decisions[
    "scenario_cost"
] = scenario_decisions[
    "scenario_cost"
].round(2)


scenario_decisions[
    "cost_change_vs_optimized"
] = scenario_decisions[
    "cost_change_vs_optimized"
].round(2)


# =========================================================
# SORT
# =========================================================

scenario_decisions = (
    scenario_decisions
    .sort_values(
        [
            "action_priority",
            "final_decision_score",
            "scenario_rank",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# SAVE MAIN OUTPUT
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


scenario_decisions.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# SUMMARY BY SCENARIO
# =========================================================

scenario_summary = (
    scenario_decisions
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
            "exposure_class",
            lambda x: (
                x == "Critical exposure"
            ).sum(),
        ),

        high_exposures=(
            "exposure_class",
            lambda x: (
                x == "High exposure"
            ).sum(),
        ),

        moderate_exposures=(
            "exposure_class",
            lambda x: (
                x == "Moderate exposure"
            ).sum(),
        ),

        total_lost_sales_exposure=(
            "lost_sales_increase_vs_optimized",
            "sum",
        ),

        total_incremental_cost=(
            "cost_change_vs_optimized",
            "sum",
        ),

        avg_fill_rate_change=(
            "fill_rate_change_vs_optimized",
            "mean",
        ),

        avg_exposure_score=(
            "scenario_exposure_score",
            "mean",
        ),

        avg_final_decision_score=(
            "final_decision_score",
            "mean",
        ),

        total_additional_safety_stock=(
            "additional_safety_stock_required",
            "sum",
        ),

        hidden_vulnerabilities=(
            "risk_scenario_relationship",
            lambda x: (
                x
                ==
                "Hidden scenario vulnerability"
            ).sum(),
        ),

    )
)


scenario_summary[
    "avg_fill_rate_change"
] = (
    scenario_summary[
        "avg_fill_rate_change"
    ]
    .round(4)
)


scenario_summary[
    "avg_exposure_score"
] = (
    scenario_summary[
        "avg_exposure_score"
    ]
    .round(4)
)


scenario_summary[
    "avg_final_decision_score"
] = (
    scenario_summary[
        "avg_final_decision_score"
    ]
    .round(4)
)


scenario_summary[
    "total_incremental_cost"
] = (
    scenario_summary[
        "total_incremental_cost"
    ]
    .round(2)
)


scenario_summary = (
    scenario_summary
    .sort_values(
        "avg_final_decision_score",
        ascending=False,
    )
)


# =========================================================
# SAVE SUMMARY
# =========================================================

scenario_summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# =========================================================
# EXECUTIVE OUTPUT
# =========================================================

print()
print("=" * 70)
print("SCENARIO DECISION ENGINE COMPLETE")
print("=" * 70)

print()

print(
    f"Scenario-product records: "
    f"{len(scenario_decisions):,}"
)

print(
    f"Products analyzed: "
    f"{scenario_decisions['product_id'].nunique():,}"
)

print(
    f"Scenarios analyzed: "
    f"{scenario_decisions['scenario'].nunique():,}"
)


print()
print("Exposure distribution:")

print(
    scenario_decisions[
        "exposure_class"
    ]
    .value_counts()
    .to_string()
)


print()
print("Action priority:")

print(
    scenario_decisions[
        "action_priority"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("Risk vs scenario relationship:")

print(
    scenario_decisions[
        "risk_scenario_relationship"
    ]
    .value_counts()
    .to_string()
)


print()
print("Scenario summary:")

print()


display_columns = [
    "scenario",
    "products",
    "critical_exposures",
    "high_exposures",
    "total_lost_sales_exposure",
    "total_incremental_cost",
    "avg_fill_rate_change",
    "hidden_vulnerabilities",
]


print(
    scenario_summary[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# =========================================================
# TOP 20 CONTINGENCY RISKS
# =========================================================

print()
print("=" * 70)
print("TOP 20 CONTINGENCY RISKS")
print("=" * 70)

print()

top_columns = [
    "product_id",
    "scenario",
    "risk_class",
    "primary_risk_driver",
    "risk_scenario_relationship",
    "exposure_class",
    "final_decision_score",
    "lost_sales_increase_vs_optimized",
    "cost_change_vs_optimized",
    "fill_rate_change_vs_optimized",
    "additional_safety_stock_required",
    "scenario_recommendation",
    "decision_confidence",
]


available_top_columns = [
    column
    for column in top_columns
    if column in scenario_decisions.columns
]


print(
    scenario_decisions[
        available_top_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# =========================================================
# HIDDEN VULNERABILITIES
# =========================================================

hidden = (
    scenario_decisions[
        scenario_decisions[
            "risk_scenario_relationship"
        ]
        ==
        "Hidden scenario vulnerability"
    ]
)


print()
print("=" * 70)
print("HIDDEN SCENARIO VULNERABILITIES")
print("=" * 70)

print()

print(
    f"Products identified: "
    f"{hidden['product_id'].nunique():,}"
)

print(
    f"Scenario exposures: "
    f"{len(hidden):,}"
)


if len(hidden) > 0:

    print()

    print(
        hidden[
            [
                "product_id",
                "scenario",
                "risk_class",
                "primary_risk_driver",
                "exposure_class",
                "final_decision_score",
                "scenario_recommendation",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# =========================================================
# OUTPUT
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

print()
