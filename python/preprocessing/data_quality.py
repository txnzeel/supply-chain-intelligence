from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTS_PATH = PROJECT_ROOT / "data" / "raw" / "products.csv"
DEMAND_PATH = PROJECT_ROOT / "data" / "processed" / "daily_demand.csv"
INVENTORY_PATH = PROJECT_ROOT / "data" / "processed" / "inventory_daily.csv"

QUALITY_DIR = PROJECT_ROOT / "data" / "processed" / "quality"

QUALITY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL RESULTS
# ============================================================

results = []


# ============================================================
# HELPERS
# ============================================================

def add_result(
    dataset,
    check,
    status,
    value,
    expected="",
    severity="INFO",
):
    results.append(
        {
            "dataset": dataset,
            "check": check,
            "status": status,
            "value": value,
            "expected": expected,
            "severity": severity,
        }
    )


def check_required_columns(df, dataset, required_columns):
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        add_result(
            dataset=dataset,
            check="Required columns",
            status="FAIL",
            value=", ".join(missing),
            expected="All required columns present",
            severity="CRITICAL",
        )
    else:
        add_result(
            dataset=dataset,
            check="Required columns",
            status="PASS",
            value=len(required_columns),
            expected="All required columns present",
        )


def check_nulls(df, dataset):
    null_count = int(df.isnull().sum().sum())

    if null_count == 0:
        add_result(
            dataset=dataset,
            check="Missing values",
            status="PASS",
            value=0,
            expected="0",
        )
    else:
        add_result(
            dataset=dataset,
            check="Missing values",
            status="FAIL",
            value=null_count,
            expected="0",
            severity="HIGH",
        )

        null_details = (
            df.isnull()
            .sum()
            .loc[lambda x: x > 0]
            .sort_values(ascending=False)
        )

        for column, count in null_details.items():
            add_result(
                dataset=dataset,
                check=f"Missing values: {column}",
                status="FAIL",
                value=int(count),
                expected="0",
                severity="HIGH",
            )


def check_duplicates(df, dataset, subset=None):
    duplicate_count = int(
        df.duplicated(subset=subset).sum()
    )

    if subset:
        label = f"Duplicate rows by {', '.join(subset)}"
    else:
        label = "Duplicate rows"

    if duplicate_count == 0:
        add_result(
            dataset=dataset,
            check=label,
            status="PASS",
            value=0,
            expected="0",
        )
    else:
        add_result(
            dataset=dataset,
            check=label,
            status="FAIL",
            value=duplicate_count,
            expected="0",
            severity="HIGH",
        )


def check_negative_values(df, dataset, columns):
    for column in columns:

        if column not in df.columns:
            continue

        negative_count = int(
            (df[column] < 0).sum()
        )

        if negative_count == 0:
            add_result(
                dataset=dataset,
                check=f"Negative values: {column}",
                status="PASS",
                value=0,
                expected="0",
            )
        else:
            add_result(
                dataset=dataset,
                check=f"Negative values: {column}",
                status="FAIL",
                value=negative_count,
                expected="0",
                severity="HIGH",
            )


def check_numeric_columns(df, dataset, columns):
    for column in columns:

        if column not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            add_result(
                dataset=dataset,
                check=f"Numeric type: {column}",
                status="PASS",
                value=str(df[column].dtype),
                expected="Numeric",
            )
        else:
            add_result(
                dataset=dataset,
                check=f"Numeric type: {column}",
                status="FAIL",
                value=str(df[column].dtype),
                expected="Numeric",
                severity="HIGH",
            )


# ============================================================
# PRODUCT QUALITY
# ============================================================

def validate_products():

    print("\n" + "=" * 70)
    print("PRODUCT DATA QUALITY")
    print("=" * 70)

    df = pd.read_csv(PRODUCTS_PATH)

    required_columns = [
        "product_id",
        "product_name",
        "category",
        "subcategory",
        "brand",
        "demand_class",
        "selling_price",
        "unit_cost",
        "base_daily_demand",
        "gross_profit",
        "gross_margin_pct",
    ]

    check_required_columns(
        df,
        "products",
        required_columns,
    )

    add_result(
        dataset="products",
        check="Row count",
        status="PASS",
        value=len(df),
        expected="> 0",
    )

    check_nulls(
        df,
        "products",
    )

    check_duplicates(
        df,
        "products",
        ["product_id"],
    )

    check_numeric_columns(
        df,
        "products",
        [
            "selling_price",
            "unit_cost",
            "base_daily_demand",
            "gross_profit",
            "gross_margin_pct",
        ],
    )

    check_negative_values(
        df,
        "products",
        [
            "selling_price",
            "unit_cost",
            "base_daily_demand",
            "gross_profit",
        ],
    )

    # --------------------------------------------------------
    # Product ID uniqueness
    # --------------------------------------------------------

    if "product_id" in df.columns:

        unique_products = df["product_id"].nunique()

        if unique_products == len(df):

            add_result(
                dataset="products",
                check="Product ID uniqueness",
                status="PASS",
                value=unique_products,
                expected=len(df),
            )

        else:

            add_result(
                dataset="products",
                check="Product ID uniqueness",
                status="FAIL",
                value=unique_products,
                expected=len(df),
                severity="CRITICAL",
            )

    # --------------------------------------------------------
    # Gross profit calculation
    # --------------------------------------------------------

    if {
        "selling_price",
        "unit_cost",
        "gross_profit",
    }.issubset(df.columns):

        expected_profit = (
            df["selling_price"]
            - df["unit_cost"]
        )

        mismatch_mask = ~np.isclose(
            df["gross_profit"],
            expected_profit,
            rtol=0.001,
            atol=0.1,
        )

        mismatches = int(
            mismatch_mask.sum()
        )

        if mismatches == 0:

            add_result(
                dataset="products",
                check="Gross profit calculation",
                status="PASS",
                value=0,
                expected="0 mismatches",
            )

        else:

            add_result(
                dataset="products",
                check="Gross profit calculation",
                status="FAIL",
                value=mismatches,
                expected="0 mismatches",
                severity="HIGH",
            )

    # --------------------------------------------------------
    # Gross margin calculation
    # --------------------------------------------------------

    if {
        "selling_price",
        "unit_cost",
        "gross_margin_pct",
    }.issubset(df.columns):

        expected_margin = (
            (
                df["selling_price"]
                - df["unit_cost"]
            )
            / df["selling_price"]
        )

        mismatch_mask = ~np.isclose(
            df["gross_margin_pct"],
            expected_margin,
            rtol=0.01,
            atol=0.001,
        )

        mismatches = int(
            mismatch_mask.sum()
        )

        if mismatches == 0:

            add_result(
                dataset="products",
                check="Gross margin calculation",
                status="PASS",
                value=0,
                expected="0 mismatches",
            )

        else:

            add_result(
                dataset="products",
                check="Gross margin calculation",
                status="FAIL",
                value=mismatches,
                expected="0 mismatches",
                severity="HIGH",
            )

    # --------------------------------------------------------
    # Gross margin bounds
    # --------------------------------------------------------

    if "gross_margin_pct" in df.columns:

        invalid = int(
            (
                (df["gross_margin_pct"] < 0)
                | (df["gross_margin_pct"] > 1)
            ).sum()
        )

        if invalid == 0:

            add_result(
                dataset="products",
                check="Gross margin bounds",
                status="PASS",
                value=0,
                expected="0 ≤ margin ≤ 1",
            )

        else:

            add_result(
                dataset="products",
                check="Gross margin bounds",
                status="FAIL",
                value=invalid,
                expected="0 ≤ margin ≤ 1",
                severity="HIGH",
            )

    # --------------------------------------------------------
    # Demand classes
    # --------------------------------------------------------

    if "demand_class" in df.columns:

        valid_classes = {
            "Stable",
            "Seasonal",
            "Intermittent",
            "Volatile",
            "Trending",
        }

        actual_classes = set(
            df["demand_class"]
            .dropna()
            .unique()
        )

        invalid_classes = (
            actual_classes - valid_classes
        )

        if not invalid_classes:

            add_result(
                dataset="products",
                check="Demand class validity",
                status="PASS",
                value=0,
                expected="Known demand classes",
            )

        else:

            add_result(
                dataset="products",
                check="Demand class validity",
                status="FAIL",
                value=", ".join(
                    sorted(invalid_classes)
                ),
                expected="Known demand classes",
                severity="HIGH",
            )

    return df


# ============================================================
# DEMAND QUALITY
# ============================================================

def validate_demand(products_df):

    print("\n" + "=" * 70)
    print("DEMAND DATA QUALITY")
    print("=" * 70)

    df = pd.read_csv(
        DEMAND_PATH,
        parse_dates=["date"],
    )

    required_columns = [
        "product_id",
        "date",
        "actual_demand",
        "expected_demand",
        "category",
        "demand_class",
    ]

    check_required_columns(
        df,
        "daily_demand",
        required_columns,
    )

    add_result(
        dataset="daily_demand",
        check="Row count",
        status="PASS",
        value=len(df),
        expected="> 0",
    )

    check_nulls(
        df,
        "daily_demand",
    )

    check_duplicates(
        df,
        "daily_demand",
        ["product_id", "date"],
    )

    check_numeric_columns(
        df,
        "daily_demand",
        [
            "actual_demand",
            "expected_demand",
            "base_daily_demand",
        ],
    )

    check_negative_values(
        df,
        "daily_demand",
        [
            "actual_demand",
            "expected_demand",
            "base_daily_demand",
        ],
    )

    # --------------------------------------------------------
    # Product coverage
    # --------------------------------------------------------

    if "product_id" in df.columns:

        demand_products = df["product_id"].nunique()
        master_products = products_df["product_id"].nunique()

        if demand_products == master_products:

            add_result(
                dataset="daily_demand",
                check="Product coverage",
                status="PASS",
                value=demand_products,
                expected=master_products,
            )

        else:

            add_result(
                dataset="daily_demand",
                check="Product coverage",
                status="FAIL",
                value=demand_products,
                expected=master_products,
                severity="CRITICAL",
            )

        missing_products = (
            set(products_df["product_id"])
            - set(df["product_id"])
        )

        if not missing_products:

            add_result(
                dataset="daily_demand",
                check="Master product referential integrity",
                status="PASS",
                value=0,
                expected="0 missing products",
            )

        else:

            add_result(
                dataset="daily_demand",
                check="Master product referential integrity",
                status="FAIL",
                value=len(missing_products),
                expected="0 missing products",
                severity="CRITICAL",
            )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    if "date" in df.columns:

        add_result(
            dataset="daily_demand",
            check="Minimum date",
            status="PASS",
            value=str(df["date"].min().date()),
            expected="",
        )

        add_result(
            dataset="daily_demand",
            check="Maximum date",
            status="PASS",
            value=str(df["date"].max().date()),
            expected="",
        )

    # --------------------------------------------------------
    # Product observation consistency
    # --------------------------------------------------------

    if {
        "product_id",
        "date",
    }.issubset(df.columns):

        observations = (
            df.groupby("product_id")["date"]
            .nunique()
        )

        expected_observations = int(
            observations.median()
        )

        inconsistent = int(
            (
                observations
                != expected_observations
            ).sum()
        )

        if inconsistent == 0:

            add_result(
                dataset="daily_demand",
                check="Product date coverage",
                status="PASS",
                value=expected_observations,
                expected="Consistent observations per product",
            )

        else:

            add_result(
                dataset="daily_demand",
                check="Product date coverage",
                status="FAIL",
                value=inconsistent,
                expected="Consistent observations per product",
                severity="HIGH",
            )

    return df


# ============================================================
# INVENTORY QUALITY
# ============================================================

def validate_inventory(products_df):

    print("\n" + "=" * 70)
    print("INVENTORY DATA QUALITY")
    print("=" * 70)

    df = pd.read_csv(
        INVENTORY_PATH,
        parse_dates=["date"],
    )

    required_columns = [
        "product_id",
        "date",
        "actual_demand",
        "opening_inventory",
        "closing_inventory",
        "purchase_order",
        "purchase_order_arrival",
        "inventory_position",
        "stockout_flag",
        "fill_rate",
        "inventory_utilization",
    ]

    check_required_columns(
        df,
        "inventory_daily",
        required_columns,
    )

    add_result(
        dataset="inventory_daily",
        check="Row count",
        status="PASS",
        value=len(df),
        expected="> 0",
    )

    check_nulls(
        df,
        "inventory_daily",
    )

    check_duplicates(
        df,
        "inventory_daily",
        ["product_id", "date"],
    )

    check_numeric_columns(
        df,
        "inventory_daily",
        [
            "actual_demand",
            "opening_inventory",
            "closing_inventory",
            "purchase_order",
            "inventory_position",
            "safety_stock",
            "reorder_point",
            "order_quantity",
            "fill_rate",
            "inventory_utilization",
        ],
    )

    check_negative_values(
        df,
        "inventory_daily",
        [
            "actual_demand",
            "opening_inventory",
            "closing_inventory",
            "purchase_order",
            "inventory_position",
            "safety_stock",
            "reorder_point",
            "order_quantity",
        ],
    )

    # --------------------------------------------------------
    # Product coverage
    # --------------------------------------------------------

    if "product_id" in df.columns:

        inventory_products = (
            df["product_id"].nunique()
        )

        master_products = (
            products_df["product_id"].nunique()
        )

        if inventory_products == master_products:

            add_result(
                dataset="inventory_daily",
                check="Product coverage",
                status="PASS",
                value=inventory_products,
                expected=master_products,
            )

        else:

            add_result(
                dataset="inventory_daily",
                check="Product coverage",
                status="FAIL",
                value=inventory_products,
                expected=master_products,
                severity="CRITICAL",
            )

        missing_products = (
            set(products_df["product_id"])
            - set(df["product_id"])
        )

        if not missing_products:

            add_result(
                dataset="inventory_daily",
                check="Master product referential integrity",
                status="PASS",
                value=0,
                expected="0 missing products",
            )

        else:

            add_result(
                dataset="inventory_daily",
                check="Master product referential integrity",
                status="FAIL",
                value=len(missing_products),
                expected="0 missing products",
                severity="CRITICAL",
            )

    # --------------------------------------------------------
    # Fill rate bounds
    # --------------------------------------------------------

    if "fill_rate" in df.columns:

        invalid = int(
            (
                (df["fill_rate"] < 0)
                | (df["fill_rate"] > 1)
            ).sum()
        )

        if invalid == 0:

            add_result(
                dataset="inventory_daily",
                check="Fill rate bounds",
                status="PASS",
                value=0,
                expected="0 ≤ fill_rate ≤ 1",
            )

        else:

            add_result(
                dataset="inventory_daily",
                check="Fill rate bounds",
                status="FAIL",
                value=invalid,
                expected="0 ≤ fill_rate ≤ 1",
                severity="HIGH",
            )

    # --------------------------------------------------------
    # Inventory utilization bounds
    # --------------------------------------------------------

    if "inventory_utilization" in df.columns:

        invalid = int(
            (
                (df["inventory_utilization"] < 0)
                | (df["inventory_utilization"] > 1)
            ).sum()
        )

        if invalid == 0:

            add_result(
                dataset="inventory_daily",
                check="Inventory utilization bounds",
                status="PASS",
                value=0,
                expected="0 ≤ utilization ≤ 1",
            )

        else:

            add_result(
                dataset="inventory_daily",
                check="Inventory utilization bounds",
                status="FAIL",
                value=invalid,
                expected="0 ≤ utilization ≤ 1",
                severity="HIGH",
            )

    # --------------------------------------------------------
    # Stockout flag validity
    # --------------------------------------------------------

    if "stockout_flag" in df.columns:

        valid_values = {
            0,
            1,
            True,
            False,
        }

        invalid = int(
            (
                ~df["stockout_flag"]
                .isin(valid_values)
            ).sum()
        )

        if invalid == 0:

            add_result(
                dataset="inventory_daily",
                check="Stockout flag validity",
                status="PASS",
                value=0,
                expected="Boolean / 0 / 1",
            )

        else:

            add_result(
                dataset="inventory_daily",
                check="Stockout flag validity",
                status="FAIL",
                value=invalid,
                expected="Boolean / 0 / 1",
                severity="HIGH",
            )

    # --------------------------------------------------------
    # Lead time validation
    # --------------------------------------------------------

    lead_time_columns = [
        "min_lead_time_days",
        "max_lead_time_days",
        "average_lead_time_days",
        "actual_lead_time",
        "supplier_delay_days",
    ]

    check_negative_values(
        df,
        "inventory_daily",
        lead_time_columns,
    )

    # --------------------------------------------------------
    # Normal lead time consistency
    # --------------------------------------------------------

    if {
        "min_lead_time_days",
        "max_lead_time_days",
    }.issubset(df.columns):

        invalid = int(
            (
                df["min_lead_time_days"]
                > df["max_lead_time_days"]
            ).sum()
        )

        if invalid == 0:

            add_result(
                dataset="inventory_daily",
                check="Lead time consistency",
                status="PASS",
                value=0,
                expected="min ≤ max",
            )

        else:

            add_result(
                dataset="inventory_daily",
                check="Lead time consistency",
                status="FAIL",
                value=invalid,
                expected="min ≤ max",
                severity="HIGH",
            )

    # --------------------------------------------------------
    # Average lead time consistency
    # --------------------------------------------------------

    if {
        "min_lead_time_days",
        "max_lead_time_days",
        "average_lead_time_days",
    }.issubset(df.columns):

        invalid = int(
            (
                (df["average_lead_time_days"]
                 < df["min_lead_time_days"])
                |
                (df["average_lead_time_days"]
                 > df["max_lead_time_days"])
            ).sum()
        )

        if invalid == 0:

            add_result(
                dataset="inventory_daily",
                check="Average lead time bounds",
                status="PASS",
                value=0,
                expected="min ≤ average ≤ max",
            )

        else:

            add_result(
                dataset="inventory_daily",
                check="Average lead time bounds",
                status="FAIL",
                value=invalid,
                expected="min ≤ average ≤ max",
                severity="HIGH",
            )

    # --------------------------------------------------------
    # Actual lead time validation
    #
    # actual_lead_time includes supplier delays.
    #
    # Therefore:
    #
    #   actual >= normal minimum
    #
    # and:
    #
    #   actual <= normal maximum + supplier delay
    #
    # A value of zero means there is no active lead-time
    # event and is therefore valid.
    # --------------------------------------------------------

    if {
        "actual_lead_time",
        "min_lead_time_days",
        "max_lead_time_days",
        "supplier_delay_days",
    }.issubset(df.columns):

        active = (
            df["actual_lead_time"] > 0
        )

        lower_bound_invalid = (
            active
            & (
                df["actual_lead_time"]
                < df["min_lead_time_days"]
            )
        )

        upper_bound_invalid = (
            active
            & (
                df["actual_lead_time"]
                >
                (
                    df["max_lead_time_days"]
                    + df["supplier_delay_days"]
                )
            )
        )

        invalid_mask = (
            lower_bound_invalid
            | upper_bound_invalid
        )

        invalid = int(
            invalid_mask.sum()
        )

        if invalid == 0:

            add_result(
                dataset="inventory_daily",
                check="Actual lead time validation",
                status="PASS",
                value=0,
                expected=(
                    "For active lead times: "
                    "min ≤ actual ≤ max + supplier delay"
                ),
            )

        else:

            add_result(
                dataset="inventory_daily",
                check="Actual lead time validation",
                status="FAIL",
                value=invalid,
                expected=(
                    "For active lead times: "
                    "min ≤ actual ≤ max + supplier delay"
                ),
                severity="MEDIUM",
            )

    # --------------------------------------------------------
    # Inventory balance diagnostic
    #
    # Purchase-order arrivals can increase inventory, so
    # closing inventory cannot be validated using only:
    #
    # opening inventory - demand fulfilled
    #
    # Therefore this remains a diagnostic rather than a
    # hard data-quality failure.
    # --------------------------------------------------------

    balance_columns = {
        "opening_inventory",
        "demand_fulfilled",
        "closing_inventory",
    }

    if balance_columns.issubset(df.columns):

        expected_closing = (
            df["opening_inventory"]
            - df["demand_fulfilled"]
        )

        mismatch_mask = (
            abs(
                df["closing_inventory"]
                - expected_closing
            ) > 1
        )

        mismatch_count = int(
            mismatch_mask.sum()
        )

        add_result(
            dataset="inventory_daily",
            check="Inventory balance diagnostic",
            status="INFO",
            value=mismatch_count,
            expected="Review mismatches against purchase arrivals",
            severity=(
                "MEDIUM"
                if mismatch_count > 0
                else "INFO"
            ),
        )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    if "date" in df.columns:

        add_result(
            dataset="inventory_daily",
            check="Minimum date",
            status="PASS",
            value=str(
                df["date"].min().date()
            ),
            expected="",
        )

        add_result(
            dataset="inventory_daily",
            check="Maximum date",
            status="PASS",
            value=str(
                df["date"].max().date()
            ),
            expected="",
        )

    # --------------------------------------------------------
    # Product observation consistency
    # --------------------------------------------------------

    if {
        "product_id",
        "date",
    }.issubset(df.columns):

        observations = (
            df.groupby("product_id")["date"]
            .nunique()
        )

        expected_observations = int(
            observations.median()
        )

        inconsistent = int(
            (
                observations
                != expected_observations
            ).sum()
        )

        if inconsistent == 0:

            add_result(
                dataset="inventory_daily",
                check="Product date coverage",
                status="PASS",
                value=expected_observations,
                expected="Consistent observations per product",
            )

        else:

            add_result(
                dataset="inventory_daily",
                check="Product date coverage",
                status="FAIL",
                value=inconsistent,
                expected="Consistent observations per product",
                severity="HIGH",
            )

    return df


# ============================================================
# QUALITY SUMMARY
# ============================================================

def create_summary():

    quality_df = pd.DataFrame(results)

    summary = (
        quality_df
        .groupby(
            ["dataset", "status"],
            dropna=False,
        )
        .size()
        .reset_index(name="checks")
    )

    quality_df.to_csv(
        QUALITY_DIR / "data_quality_report.csv",
        index=False,
    )

    summary.to_csv(
        QUALITY_DIR / "data_quality_summary.csv",
        index=False,
    )

    failures = quality_df[
        quality_df["status"] == "FAIL"
    ].copy()

    failures.to_csv(
        QUALITY_DIR / "data_quality_issues.csv",
        index=False,
    )

    return quality_df, summary, failures


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SUPPLY CHAIN DATA QUALITY ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # File existence
    # --------------------------------------------------------

    required_files = {
        "Products": PRODUCTS_PATH,
        "Daily demand": DEMAND_PATH,
        "Inventory": INVENTORY_PATH,
    }

    for name, path in required_files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    products = validate_products()

    validate_demand(
        products
    )

    validate_inventory(
        products
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    quality_df, summary, failures = (
        create_summary()
    )

    print()
    print("=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)

    if failures.empty:

        print("STATUS: PASS")
        print(
            "No data-quality failures detected."
        )

    else:

        print("STATUS: FAIL")
        print(
            f"{len(failures)} quality checks require attention."
        )

        print()
        print("FAILURES:")
        print()

        print(
            failures[
                [
                    "dataset",
                    "check",
                    "value",
                    "expected",
                    "severity",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("REPORTS")
    print("=" * 70)

    print(
        f"Detailed report:\n"
        f"{QUALITY_DIR / 'data_quality_report.csv'}"
    )

    print(
        f"\nSummary:\n"
        f"{QUALITY_DIR / 'data_quality_summary.csv'}"
    )

    print(
        f"\nIssues:\n"
        f"{QUALITY_DIR / 'data_quality_issues.csv'}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()