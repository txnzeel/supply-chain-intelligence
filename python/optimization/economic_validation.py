import pandas as pd
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
    / "economic_validation.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

decisions = pd.read_csv(INPUT_FILE)


print()
print("=" * 70)
print("ECONOMIC VALIDATION")
print("=" * 70)
print()


# =========================================================
# VALIDATION
# =========================================================

required_columns = [
    "product_id",

    "baseline_total_cost",
    "optimized_total_cost",

    "baseline_lost_sales",
    "optimized_lost_sales",

    "baseline_inventory_days",
    "optimized_inventory_days",

    "baseline_fill_rate",
    "optimized_fill_rate",

    "cost_savings",
    "lost_sales_reduction",
    "fill_rate_change",
    "inventory_days_change",

    "business_impact",
    "priority",
    "management_decision",
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


numeric_columns = [
    "baseline_total_cost",
    "optimized_total_cost",
    "baseline_lost_sales",
    "optimized_lost_sales",
    "baseline_inventory_days",
    "optimized_inventory_days",
    "baseline_fill_rate",
    "optimized_fill_rate",
    "cost_savings",
    "lost_sales_reduction",
    "fill_rate_change",
    "inventory_days_change",
]


for column in numeric_columns:

    decisions[column] = pd.to_numeric(
        decisions[column],
        errors="coerce",
    )


assert decisions[numeric_columns].notna().all().all(), (
    "Numeric validation failed: missing or invalid values detected."
)


print("Input validation passed.")
print(
    f"Products loaded: {len(decisions):,}"
)


# =========================================================
# PORTFOLIO TOTALS
# =========================================================

baseline_total_cost = (
    decisions["baseline_total_cost"].sum()
)

optimized_total_cost = (
    decisions["optimized_total_cost"].sum()
)

calculated_cost_savings = (
    baseline_total_cost
    -
    optimized_total_cost
)


baseline_lost_sales = (
    decisions["baseline_lost_sales"].sum()
)

optimized_lost_sales = (
    decisions["optimized_lost_sales"].sum()
)

calculated_lost_sales_reduction = (
    baseline_lost_sales
    -
    optimized_lost_sales
)


baseline_inventory_days = (
    decisions["baseline_inventory_days"].mean()
)

optimized_inventory_days = (
    decisions["optimized_inventory_days"].mean()
)

calculated_inventory_days_change = (
    optimized_inventory_days
    -
    baseline_inventory_days
)


baseline_fill_rate = (
    decisions["baseline_fill_rate"].mean()
)

optimized_fill_rate = (
    decisions["optimized_fill_rate"].mean()
)

calculated_fill_rate_change = (
    optimized_fill_rate
    -
    baseline_fill_rate
)


# =========================================================
# RECONCILIATION
# =========================================================

reported_cost_savings = (
    decisions["cost_savings"].sum()
)

reported_lost_sales_reduction = (
    decisions["lost_sales_reduction"].sum()
)

reported_fill_rate_change = (
    decisions["fill_rate_change"].mean()
)

reported_inventory_days_change = (
    decisions["inventory_days_change"].mean()
)


cost_difference = (
    calculated_cost_savings
    -
    reported_cost_savings
)

lost_sales_difference = (
    calculated_lost_sales_reduction
    -
    reported_lost_sales_reduction
)

fill_rate_difference = (
    calculated_fill_rate_change
    -
    reported_fill_rate_change
)

inventory_days_difference = (
    calculated_inventory_days_change
    -
    reported_inventory_days_change
)


# =========================================================
# ECONOMIC STATUS
# =========================================================

if calculated_cost_savings > 0:

    economic_result = (
        "Positive economic impact"
    )

elif calculated_cost_savings < 0:

    economic_result = (
        "Negative economic impact"
    )

else:

    economic_result = (
        "Neutral economic impact"
    )


# =========================================================
# SERVICE STATUS
# =========================================================

if calculated_fill_rate_change > 0:

    service_result = (
        "Service improvement"
    )

elif calculated_fill_rate_change < 0:

    service_result = (
        "Service deterioration"
    )

else:

    service_result = (
        "Service unchanged"
    )


# =========================================================
# LOST SALES STATUS
# =========================================================

if calculated_lost_sales_reduction > 0:

    lost_sales_result = (
        "Lost-sales reduction"
    )

elif calculated_lost_sales_reduction < 0:

    lost_sales_result = (
        "Lost-sales increase"
    )

else:

    lost_sales_result = (
        "Lost sales unchanged"
    )


# =========================================================
# INVENTORY STATUS
# =========================================================

if calculated_inventory_days_change > 0:

    inventory_result = (
        "Inventory increased"
    )

elif calculated_inventory_days_change < 0:

    inventory_result = (
        "Inventory reduced"
    )

else:

    inventory_result = (
        "Inventory unchanged"
    )


# =========================================================
# OVERALL OUTCOME
# =========================================================

if (
    calculated_cost_savings > 0
    and
    calculated_lost_sales_reduction > 0
    and
    calculated_fill_rate_change > 0
):

    overall_outcome = (
        "Strong optimization outcome"
    )

elif (
    calculated_cost_savings > 0
    and
    calculated_lost_sales_reduction > 0
):

    overall_outcome = (
        "Economic and service improvement"
    )

elif (
    calculated_lost_sales_reduction > 0
    and
    calculated_fill_rate_change > 0
):

    overall_outcome = (
        "Service improvement with economic trade-off"
    )

elif calculated_cost_savings > 0:

    overall_outcome = (
        "Cost improvement"
    )

elif calculated_fill_rate_change > 0:

    overall_outcome = (
        "Service improvement"
    )

else:

    overall_outcome = (
        "Limited optimization benefit"
    )


# =========================================================
# PRINT PORTFOLIO VALIDATION
# =========================================================

print()
print("=" * 70)
print("PORTFOLIO ECONOMIC RESULTS")
print("=" * 70)
print()


print(
    f"Baseline total cost:       "
    f"₹{baseline_total_cost:,.2f}"
)

print(
    f"Optimized total cost:      "
    f"₹{optimized_total_cost:,.2f}"
)

print(
    f"Calculated cost savings:   "
    f"₹{calculated_cost_savings:,.2f}"
)

print(
    f"Reported cost savings:     "
    f"₹{reported_cost_savings:,.2f}"
)

print(
    f"Cost reconciliation gap:   "
    f"₹{cost_difference:,.6f}"
)


print()

print(
    f"Baseline lost sales:       "
    f"{baseline_lost_sales:,.0f}"
)

print(
    f"Optimized lost sales:      "
    f"{optimized_lost_sales:,.0f}"
)

print(
    f"Lost sales reduction:      "
    f"{calculated_lost_sales_reduction:,.0f}"
)

print(
    f"Reported reduction:        "
    f"{reported_lost_sales_reduction:,.0f}"
)

print(
    f"Lost-sales reconciliation: "
    f"{lost_sales_difference:,.6f}"
)


print()

print(
    f"Baseline average fill rate: "
    f"{baseline_fill_rate:.2%}"
)

print(
    f"Optimized average fill rate: "
    f"{optimized_fill_rate:.2%}"
)

print(
    f"Calculated fill-rate change: "
    f"{calculated_fill_rate_change:+.2%}"
)

print(
    f"Reported fill-rate change:   "
    f"{reported_fill_rate_change:+.2%}"
)

print(
    f"Fill-rate reconciliation:    "
    f"{fill_rate_difference:+.8f}"
)


print()

print(
    f"Baseline inventory days:     "
    f"{baseline_inventory_days:.2f}"
)

print(
    f"Optimized inventory days:    "
    f"{optimized_inventory_days:.2f}"
)

print(
    f"Inventory-days change:       "
    f"{calculated_inventory_days_change:+.2f}"
)

print(
    f"Reported inventory change:   "
    f"{reported_inventory_days_change:+.2f}"
)

print(
    f"Inventory reconciliation:    "
    f"{inventory_days_difference:+.8f}"
)


# =========================================================
# OUTCOME
# =========================================================

print()
print("=" * 70)
print("OPTIMIZATION OUTCOME")
print("=" * 70)
print()


print(
    f"Economic result:     {economic_result}"
)

print(
    f"Service result:      {service_result}"
)

print(
    f"Lost-sales result:   {lost_sales_result}"
)

print(
    f"Inventory result:    {inventory_result}"
)

print(
    f"Overall outcome:     {overall_outcome}"
)


# =========================================================
# PRODUCT-LEVEL ECONOMIC CLASSIFICATION
# =========================================================

def classify_product(row):

    cost = row["cost_savings"]
    lost_sales = row["lost_sales_reduction"]
    service = row["fill_rate_change"]

    if (
        cost > 0
        and
        lost_sales > 0
        and
        service > 0
    ):
        return "Triple improvement"

    if (
        cost > 0
        and
        lost_sales > 0
    ):
        return "Cost + service improvement"

    if (
        cost > 0
        and
        service > 0
    ):
        return "Cost + fill-rate improvement"

    if (
        lost_sales > 0
        and
        service > 0
    ):
        return "Service improvement"

    if cost > 0:
        return "Cost improvement"

    if lost_sales > 0:
        return "Lost-sales improvement"

    if service > 0:
        return "Fill-rate improvement"

    return "Limited benefit"


decisions["economic_classification"] = (
    decisions.apply(
        classify_product,
        axis=1,
    )
)


# =========================================================
# RECONCILIATION VALIDATION
# =========================================================

# =========================================================
# RECONCILIATION TOLERANCE
# =========================================================

# Small differences can occur because of floating-point
# arithmetic and aggregation across thousands of products.

cost_tolerance = 1.00
rate_tolerance = 1e-5
# Allows harmless differences caused by two-decimal CSV rounding.
inventory_tolerance = 1e-3


assert abs(cost_difference) <= cost_tolerance, (
    f"Cost savings reconciliation failed. "
    f"Difference: ₹{abs(cost_difference):.6f}, "
    f"Allowed tolerance: ₹{cost_tolerance:.2f}"
)

assert abs(lost_sales_difference) <= rate_tolerance, (
    f"Lost-sales reconciliation failed. "
    f"Difference: {abs(lost_sales_difference):.6f}, "
    f"Allowed tolerance: {rate_tolerance:.6f}"
)

assert abs(fill_rate_difference) <= rate_tolerance, (
    f"Fill-rate reconciliation failed. "
    f"Difference: {abs(fill_rate_difference):.6f}, "
    f"Allowed tolerance: {rate_tolerance:.6f}"
)

assert abs(inventory_days_difference) <= inventory_tolerance, (
    f"Inventory-days reconciliation failed. "
    f"Difference: {abs(inventory_days_difference):.6f}, "
    f"Allowed tolerance: {inventory_tolerance:.6f}"
)


# =========================================================
# SUMMARY TABLE
# =========================================================

summary = pd.DataFrame({

    "metric": [

        "Products analyzed",

        "Baseline total cost",
        "Optimized total cost",
        "Cost savings",

        "Baseline lost sales",
        "Optimized lost sales",
        "Lost sales reduction",

        "Baseline average fill rate",
        "Optimized average fill rate",
        "Fill rate change",

        "Baseline average inventory days",
        "Optimized average inventory days",
        "Inventory days change",

        "Policy changes",

    ],

    "value": [

        len(decisions),

        baseline_total_cost,
        optimized_total_cost,
        calculated_cost_savings,

        baseline_lost_sales,
        optimized_lost_sales,
        calculated_lost_sales_reduction,

        baseline_fill_rate,
        optimized_fill_rate,
        calculated_fill_rate_change,

        baseline_inventory_days,
        optimized_inventory_days,
        calculated_inventory_days_change,

        int(
            (
                decisions["management_decision"]
                !=
                "Maintain policy"
            ).sum()
        ),

    ],
})


# =========================================================
# ECONOMIC CLASS SUMMARY
# =========================================================

economic_summary = (

    decisions
    .groupby(
        "economic_classification",
        as_index=False,
    )
    .agg(

        products=(
            "product_id",
            "count",
        ),

        cost_savings=(
            "cost_savings",
            "sum",
        ),

        lost_sales_reduction=(
            "lost_sales_reduction",
            "sum",
        ),

        avg_fill_rate_change=(
            "fill_rate_change",
            "mean",
        ),

    )
    .sort_values(
        "cost_savings",
        ascending=False,
    )

)


print()
print("=" * 70)
print("ECONOMIC CLASSIFICATION")
print("=" * 70)
print()

print(
    economic_summary
    .round(4)
    .to_string(index=False)
)


# =========================================================
# SAVE PRODUCT-LEVEL VALIDATION
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
# FINAL
# =========================================================

print()
print("=" * 70)
print("ECONOMIC VALIDATION COMPLETE")
print("=" * 70)
print()

print(
    f"Products validated: {len(decisions):,}"
)

print(
    f"Overall outcome: {overall_outcome}"
)

print(
    f"Validated cost savings: "
    f"₹{calculated_cost_savings:,.2f}"
)

print(
    f"Validated lost-sales reduction: "
    f"{calculated_lost_sales_reduction:,.0f}"
)

print(
    f"Validated fill-rate change: "
    f"{calculated_fill_rate_change:+.2%}"
)

print(
    f"Validated inventory-days change: "
    f"{calculated_inventory_days_change:+.2f}"
)

print()

print(
    f"Saved to:"
)

print(
    OUTPUT_FILE
)

print()
