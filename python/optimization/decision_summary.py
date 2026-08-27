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
    / "optimization_decisions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "decision_summary.csv"
)


# =========================================================
# LOAD DECISION DATA
# =========================================================

decisions = pd.read_csv(INPUT_FILE)


print()
print("=" * 70)
print("DECISION SUMMARY ANALYSIS")
print("=" * 70)


# =========================================================
# VALIDATION
# =========================================================

required_columns = [
    "product_id",
    "risk_class",
    "primary_risk_driver",
    "priority",
    "management_decision",
    "business_impact",
    "recommendation_confidence",
    "decision_impact_score",
    "urgency_score",
    "lost_sales_reduction",
    "cost_savings",
    "fill_rate_change",
    "inventory_days_change",
    "policy_change_flag",
]


missing_columns = [
    column
    for column in required_columns
    if column not in decisions.columns
]


assert not missing_columns, (
    f"Missing required columns: {missing_columns}"
)


assert decisions["product_id"].notna().all()

assert decisions["product_id"].is_unique


print()
print("Input validation passed.")
print(
    f"Products loaded: {len(decisions):,}"
)


# =========================================================
# NUMERIC CLEANING
# =========================================================

numeric_columns = [
    "decision_impact_score",
    "urgency_score",
    "lost_sales_reduction",
    "cost_savings",
    "fill_rate_change",
    "inventory_days_change",
    "policy_change_flag",
]


for column in numeric_columns:

    decisions[column] = pd.to_numeric(
        decisions[column],
        errors="coerce",
    )


# =========================================================
# PORTFOLIO METRICS
# =========================================================

total_products = len(decisions)

total_lost_sales_reduction = (
    decisions["lost_sales_reduction"]
    .sum()
)

total_cost_savings = (
    decisions["cost_savings"]
    .sum()
)

average_fill_rate_change = (
    decisions["fill_rate_change"]
    .mean()
)

average_inventory_days_change = (
    decisions["inventory_days_change"]
    .mean()
)

total_policy_changes = int(
    decisions["policy_change_flag"]
    .sum()
)


# =========================================================
# BUSINESS IMPACT COUNTS
# =========================================================

cost_service_products = int(
    (
        decisions["business_impact"]
        == "Cost saving + service improvement"
    )
    .sum()
)


cost_saving_products = int(
    (
        decisions["business_impact"]
        == "Cost saving"
    )
    .sum()
)


service_improvement_products = int(
    (
        decisions["business_impact"]
        == "Service improvement"
    )
    .sum()
)


limited_benefit_products = int(
    (
        decisions["business_impact"]
        == "Limited benefit"
    )
    .sum()
)


# =========================================================
# PRIORITY COUNTS
# =========================================================

p1_critical = int(
    (
        decisions["priority"]
        == "P1 - Critical"
    )
    .sum()
)


p1_high = int(
    (
        decisions["priority"]
        == "P1 - High Impact"
    )
    .sum()
)


p2_medium = int(
    (
        decisions["priority"]
        == "P2 - Medium"
    )
    .sum()
)


p3_low = int(
    (
        decisions["priority"]
        == "P3 - Low Impact"
    )
    .sum()
)


p4_monitor = int(
    (
        decisions["priority"]
        == "P4 - Monitor"
    )
    .sum()
)


# =========================================================
# CONFIDENCE COUNTS
# =========================================================

high_confidence = int(
    (
        decisions["recommendation_confidence"]
        == "High"
    )
    .sum()
)


medium_confidence = int(
    (
        decisions["recommendation_confidence"]
        == "Medium"
    )
    .sum()
)


low_confidence = int(
    (
        decisions["recommendation_confidence"]
        == "Low"
    )
    .sum()
)


# =========================================================
# MANAGEMENT DECISION COUNTS
# =========================================================

immediate_intervention = int(
    (
        decisions["management_decision"]
        == "Immediate intervention"
    )
    .sum()
)


increase_order_quantity = int(
    (
        decisions["management_decision"]
        == "Increase order quantity"
    )
    .sum()
)


reduce_order_quantity = int(
    (
        decisions["management_decision"]
        == "Reduce order quantity"
    )
    .sum()
)


increase_safety_stock = int(
    (
        decisions["management_decision"]
        == "Increase safety stock"
    )
    .sum()
)


maintain_policy = int(
    (
        decisions["management_decision"]
        == "Maintain policy"
    )
    .sum()
)


# =========================================================
# RISK COUNTS
# =========================================================

critical_products = int(
    (
        decisions["risk_class"]
        == "Critical"
    )
    .sum()
)


high_risk_products = int(
    (
        decisions["risk_class"]
        == "High"
    )
    .sum()
)


medium_risk_products = int(
    (
        decisions["risk_class"]
        == "Medium"
    )
    .sum()
)


low_risk_products = int(
    (
        decisions["risk_class"]
        == "Low"
    )
    .sum()
)


# =========================================================
# TOP OPPORTUNITIES
# =========================================================

top_opportunities = (
    decisions
    .sort_values(
        [
            "decision_impact_score",
            "urgency_score",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(20)
    .copy()
)


# =========================================================
# SUMMARY DATASET
# =========================================================

summary = pd.DataFrame(
    [
        {
            "metric": "Total products",
            "value": total_products,
        },

        {
            "metric": "Total lost sales reduction",
            "value": total_lost_sales_reduction,
        },

        {
            "metric": "Total cost savings",
            "value": total_cost_savings,
        },

        {
            "metric": "Average fill rate change",
            "value": average_fill_rate_change,
        },

        {
            "metric": "Average inventory days change",
            "value": average_inventory_days_change,
        },

        {
            "metric": "Total policy changes",
            "value": total_policy_changes,
        },

        {
            "metric": "Cost saving + service improvement",
            "value": cost_service_products,
        },

        {
            "metric": "Cost saving",
            "value": cost_saving_products,
        },

        {
            "metric": "Service improvement",
            "value": service_improvement_products,
        },

        {
            "metric": "Limited benefit",
            "value": limited_benefit_products,
        },

        {
            "metric": "P1 Critical",
            "value": p1_critical,
        },

        {
            "metric": "P1 High Impact",
            "value": p1_high,
        },

        {
            "metric": "P2 Medium",
            "value": p2_medium,
        },

        {
            "metric": "P3 Low Impact",
            "value": p3_low,
        },

        {
            "metric": "P4 Monitor",
            "value": p4_monitor,
        },

        {
            "metric": "High confidence",
            "value": high_confidence,
        },

        {
            "metric": "Medium confidence",
            "value": medium_confidence,
        },

        {
            "metric": "Low confidence",
            "value": low_confidence,
        },

        {
            "metric": "Immediate intervention",
            "value": immediate_intervention,
        },

        {
            "metric": "Increase order quantity",
            "value": increase_order_quantity,
        },

        {
            "metric": "Reduce order quantity",
            "value": reduce_order_quantity,
        },

        {
            "metric": "Increase safety stock",
            "value": increase_safety_stock,
        },

        {
            "metric": "Maintain policy",
            "value": maintain_policy,
        },

        {
            "metric": "Critical risk products",
            "value": critical_products,
        },

        {
            "metric": "High risk products",
            "value": high_risk_products,
        },

        {
            "metric": "Medium risk products",
            "value": medium_risk_products,
        },

        {
            "metric": "Low risk products",
            "value": low_risk_products,
        },
    ]
)


# =========================================================
# VALIDATION OF COUNTS
# =========================================================

business_impact_total = (
    cost_service_products
    + cost_saving_products
    + service_improvement_products
    + limited_benefit_products
)


priority_total = (
    p1_critical
    + p1_high
    + p2_medium
    + p3_low
    + p4_monitor
)


confidence_total = (
    high_confidence
    + medium_confidence
    + low_confidence
)


management_total = (
    immediate_intervention
    + increase_order_quantity
    + reduce_order_quantity
    + increase_safety_stock
    + maintain_policy
)


risk_total = (
    critical_products
    + high_risk_products
    + medium_risk_products
    + low_risk_products
)


assert business_impact_total == total_products, (
    "Business impact counts do not equal total products."
)


assert priority_total == total_products, (
    "Priority counts do not equal total products."
)


assert confidence_total == total_products, (
    "Confidence counts do not equal total products."
)


assert management_total == total_products, (
    "Management decision counts do not equal total products."
)


assert risk_total == total_products, (
    "Risk class counts do not equal total products."
)


# =========================================================
# PRINT EXECUTIVE SUMMARY
# =========================================================

print()
print("=" * 70)
print("EXECUTIVE PORTFOLIO SUMMARY")
print("=" * 70)

print()
print(
    f"Products analyzed:              "
    f"{total_products:,}"
)

print(
    f"Lost sales reduction:           "
    f"{total_lost_sales_reduction:,.0f}"
)

print(
    f"Modeled cost savings:           "
    f"${total_cost_savings:,.2f}"
)

print(
    f"Average fill-rate change:       "
    f"{average_fill_rate_change:+.2%}"
)

print(
    f"Average inventory-days change:  "
    f"{average_inventory_days_change:+.2f}"
)

print(
    f"Policy changes:                 "
    f"{total_policy_changes:,}"
)


# =========================================================
# BUSINESS IMPACT
# =========================================================

print()
print("=" * 70)
print("BUSINESS IMPACT")
print("=" * 70)

print()
print(
    f"Cost + service improvement:     "
    f"{cost_service_products:,}"
)

print(
    f"Cost saving:                    "
    f"{cost_saving_products:,}"
)

print(
    f"Service improvement:            "
    f"{service_improvement_products:,}"
)

print(
    f"Limited benefit:                "
    f"{limited_benefit_products:,}"
)


# =========================================================
# PRIORITY
# =========================================================

print()
print("=" * 70)
print("PRIORITY DISTRIBUTION")
print("=" * 70)

print()
print(
    f"P1 Critical:                    "
    f"{p1_critical:,}"
)

print(
    f"P1 High Impact:                 "
    f"{p1_high:,}"
)

print(
    f"P2 Medium:                      "
    f"{p2_medium:,}"
)

print(
    f"P3 Low Impact:                  "
    f"{p3_low:,}"
)

print(
    f"P4 Monitor:                     "
    f"{p4_monitor:,}"
)


# =========================================================
# MANAGEMENT DECISIONS
# =========================================================

print()
print("=" * 70)
print("MANAGEMENT DECISIONS")
print("=" * 70)

print()
print(
    f"Immediate intervention:         "
    f"{immediate_intervention:,}"
)

print(
    f"Increase order quantity:        "
    f"{increase_order_quantity:,}"
)

print(
    f"Reduce order quantity:          "
    f"{reduce_order_quantity:,}"
)

print(
    f"Increase safety stock:           "
    f"{increase_safety_stock:,}"
)

print(
    f"Maintain policy:                "
    f"{maintain_policy:,}"
)


# =========================================================
# CONFIDENCE
# =========================================================

print()
print("=" * 70)
print("RECOMMENDATION CONFIDENCE")
print("=" * 70)

print()
print(
    f"High:                           "
    f"{high_confidence:,}"
)

print(
    f"Medium:                         "
    f"{medium_confidence:,}"
)

print(
    f"Low:                            "
    f"{low_confidence:,}"
)


# =========================================================
# RISK
# =========================================================

print()
print("=" * 70)
print("RISK DISTRIBUTION")
print("=" * 70)

print()
print(
    f"Critical:                       "
    f"{critical_products:,}"
)

print(
    f"High:                           "
    f"{high_risk_products:,}"
)

print(
    f"Medium:                         "
    f"{medium_risk_products:,}"
)

print(
    f"Low:                            "
    f"{low_risk_products:,}"
)


# =========================================================
# TOP 20 OPPORTUNITIES
# =========================================================

print()
print("=" * 70)
print("TOP 20 OPTIMIZATION OPPORTUNITIES")
print("=" * 70)

print()

top_columns = [
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


print(
    top_opportunities[
        top_columns
    ]
    .to_string(
        index=False
    )
)


# =========================================================
# SAVE SUMMARY
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


summary.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 70)
print("DECISION SUMMARY COMPLETE")
print("=" * 70)

print()
print(
    f"Summary saved to:"
)

print(
    OUTPUT_FILE
)

print()
print("All validation checks passed.")
