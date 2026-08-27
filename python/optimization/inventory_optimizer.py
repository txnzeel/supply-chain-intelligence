import numpy as np
import pandas as pd
from pathlib import Path
from statistics import NormalDist


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEMAND_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_demand.csv"
)

RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_risk.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_optimization.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "optimization_summary.csv"
)


# =========================================================
# BUSINESS COST ASSUMPTIONS
# =========================================================

HOLDING_COST_PER_UNIT = 1.00

ORDERING_COST_PER_ORDER = 25.00

LOST_SALES_COST_PER_UNIT = 10.00


# =========================================================
# BASELINE POLICY
# =========================================================

BASELINE_SERVICE_LEVEL = 0.95

BASELINE_ORDER_COVERAGE_DAYS = 30


# =========================================================
# OPTIMIZATION POLICY CANDIDATES
# =========================================================

SERVICE_LEVELS = {

    "Low": [
        0.90,
    ],

    "Medium": [
        0.95,
    ],

    "High": [
        0.975,
    ],

    "Critical": [
        0.99,
    ],

}


ORDER_COVERAGE_DAYS = [

    15,
    30,
    45,

]


# =========================================================
# LEAD TIME CONFIGURATION
# =========================================================

LEAD_TIME_CONFIG = {

    "Stable": (
        5,
        10,
    ),

    "Trending": (
        5,
        12,
    ),

    "Seasonal": (
        7,
        14,
    ),

    "Volatile": (
        10,
        20,
    ),

    "Intermittent": (
        12,
        25,
    ),

}


SUPPLIER_DELAY_PROBABILITY = {

    "Stable": 0.05,

    "Trending": 0.08,

    "Seasonal": 0.10,

    "Volatile": 0.15,

    "Intermittent": 0.20,

}


SUPPLIER_DELAY_RANGE = {

    "Stable": (
        1,
        2,
    ),

    "Trending": (
        1,
        3,
    ),

    "Seasonal": (
        1,
        4,
    ),

    "Volatile": (
        2,
        5,
    ),

    "Intermittent": (
        2,
        7,
    ),

}


# =========================================================
# LOAD DATA
# =========================================================

demand = pd.read_csv(
    DEMAND_FILE,
    parse_dates=[
        "date",
    ],
)


risk = pd.read_csv(
    RISK_FILE,
)


# =========================================================
# VALIDATION — INPUT DATA
# =========================================================

required_demand_columns = [

    "product_id",

    "category",

    "demand_class",

    "actual_demand",

    "date",

]


missing_demand_columns = [

    column

    for column in required_demand_columns

    if column not in demand.columns

]


assert not missing_demand_columns, (
    "Missing demand columns: "
    f"{missing_demand_columns}"
)


required_risk_columns = [

    "product_id",

    "risk_class",

    "primary_risk_driver",

]


missing_risk_columns = [

    column

    for column in required_risk_columns

    if column not in risk.columns

]


assert not missing_risk_columns, (
    "Missing risk columns: "
    f"{missing_risk_columns}"
)


assert demand["product_id"].notna().all()

assert demand["date"].notna().all()

assert demand["actual_demand"].notna().all()

assert (
    demand["actual_demand"] >= 0
).all()


assert (
    demand["product_id"].nunique()
    ==
    len(
        demand["product_id"]
        .unique()
    )
)


# =========================================================
# SORT DEMAND
# =========================================================

demand = (

    demand

    .sort_values(
        [
            "product_id",
            "date",
        ]
    )

    .reset_index(
        drop=True
    )

)


# =========================================================
# PRODUCT PARAMETERS
# =========================================================

product_parameters = (

    demand

    .groupby(
        [
            "product_id",
            "category",
            "demand_class",
        ],
        as_index=False,
    )

    .agg(

        average_daily_demand=(
            "actual_demand",
            "mean",
        ),

        demand_std=(
            "actual_demand",
            "std",
        ),

    )

)


product_parameters["demand_std"] = (

    product_parameters[
        "demand_std"
    ]

    .fillna(0)

)


# =========================================================
# LEAD TIME PARAMETERS
# =========================================================

product_parameters["min_lead_time"] = (

    product_parameters[
        "demand_class"
    ]

    .map(
        {
            key: value[0]
            for key, value
            in LEAD_TIME_CONFIG.items()
        }
    )

)


product_parameters["max_lead_time"] = (

    product_parameters[
        "demand_class"
    ]

    .map(
        {
            key: value[1]
            for key, value
            in LEAD_TIME_CONFIG.items()
        }
    )

)


assert (
    product_parameters[
        "min_lead_time"
    ]
    .notna()
    .all()
)


assert (
    product_parameters[
        "max_lead_time"
    ]
    .notna()
    .all()
)


product_parameters["average_lead_time"] = (

    product_parameters[
        "min_lead_time"
    ]

    +

    product_parameters[
        "max_lead_time"
    ]

) / 2


product_parameters["lead_time_std"] = (

    product_parameters[
        "max_lead_time"
    ]

    -

    product_parameters[
        "min_lead_time"
    ]

) / 4


# =========================================================
# MERGE RISK
# =========================================================

risk_columns = [

    "product_id",

    "risk_class",

    "primary_risk_driver",

]


risk_subset = (

    risk[
        risk_columns
    ]

    .drop_duplicates(
        "product_id"
    )

)


assert (
    risk_subset[
        "product_id"
    ]
    .is_unique
)


product_parameters = (

    product_parameters

    .merge(
        risk_subset,
        on="product_id",
        how="left",
        validate="one_to_one",
    )

)


assert (
    product_parameters[
        "risk_class"
    ]
    .notna()
    .all()
)


assert (
    product_parameters[
        "primary_risk_driver"
    ]
    .notna()
    .all()
)


# =========================================================
# SAFETY STOCK
# =========================================================

def calculate_safety_stock(
    average_demand,
    demand_std,
    average_lead_time,
    lead_time_std,
    service_level,
):

    z_score = NormalDist().inv_cdf(
        service_level
    )


    variance = (

        average_lead_time
        *
        demand_std ** 2

        +

        average_demand ** 2
        *
        lead_time_std ** 2

    )


    safety_stock = (

        z_score
        *
        np.sqrt(
            max(
                0,
                variance,
            )
        )

    )


    return max(
        0,
        round(
            safety_stock
        ),
    )


# =========================================================
# POLICY SIMULATION
# =========================================================

def simulate_policy(
    group,
    parameters,
    service_level,
    order_coverage_days,
    random_seed,
):

    rng = np.random.default_rng(
        random_seed
    )


    average_demand = (

        parameters[
            "average_daily_demand"
        ]

    )


    demand_std = (

        parameters[
            "demand_std"
        ]

    )


    average_lead_time = (

        parameters[
            "average_lead_time"
        ]

    )


    lead_time_std = (

        parameters[
            "lead_time_std"
        ]

    )


    demand_class = (

        parameters[
            "demand_class"
        ]

    )


    # =====================================================
    # SAFETY STOCK
    # =====================================================

    safety_stock = calculate_safety_stock(

        average_demand,

        demand_std,

        average_lead_time,

        lead_time_std,

        service_level,

    )


    # =====================================================
    # REORDER POINT
    # =====================================================

    reorder_point = (

        average_demand
        *
        average_lead_time

        +

        safety_stock

    )


    reorder_point = max(
        0,
        round(
            reorder_point
        ),
    )


    # =====================================================
    # ORDER QUANTITY
    # =====================================================

    order_quantity = (

        average_demand
        *
        order_coverage_days

    )


    order_quantity = max(
        1,
        round(
            order_quantity
        ),
    )


    # =====================================================
    # INITIAL INVENTORY
    # =====================================================

    initial_inventory = (

        average_demand
        *
        BASELINE_ORDER_COVERAGE_DAYS

    )


    initial_inventory = max(
        0,
        round(
            initial_inventory
        ),
    )


    # =====================================================
    # INVENTORY STATE
    # =====================================================

    inventory = initial_inventory

    outstanding_orders = []


    total_demand = 0

    total_fulfilled = 0

    total_lost_sales = 0

    stockout_days = 0

    total_inventory = 0

    total_purchase_quantity = 0

    purchase_orders = 0

    total_supplier_delay_days = 0

    delayed_orders = 0


    # =====================================================
    # DAILY SIMULATION
    # =====================================================

    for _, row in group.iterrows():

        current_date = row["date"]


        demand_value = int(
            row["actual_demand"]
        )


        # =================================================
        # RECEIVE ORDERS
        # =================================================

        arrivals_today = sum(

            quantity

            for arrival_date, quantity

            in outstanding_orders

            if arrival_date == current_date

        )


        inventory += arrivals_today


        outstanding_orders = [

            (
                arrival_date,
                quantity,
            )

            for arrival_date, quantity

            in outstanding_orders

            if arrival_date != current_date

        ]


        # =================================================
        # FULFILL DEMAND
        # =================================================

        fulfilled = min(
            inventory,
            demand_value,
        )


        inventory -= fulfilled


        lost_sales = (

            demand_value
            -
            fulfilled

        )


        if lost_sales > 0:

            stockout_days += 1


        # =================================================
        # INVENTORY POSITION
        # =================================================

        outstanding_quantity = sum(

            quantity

            for _, quantity

            in outstanding_orders

        )


        inventory_position = (

            inventory
            +
            outstanding_quantity

        )


        # =================================================
        # REORDER
        # =================================================

        if inventory_position <= reorder_point:

            purchase_order = order_quantity


            min_lead_time = int(
                parameters[
                    "min_lead_time"
                ]
            )


            max_lead_time = int(
                parameters[
                    "max_lead_time"
                ]
            )


            actual_lead_time = rng.integers(

                min_lead_time,

                max_lead_time + 1,

            )


            # ---------------------------------------------
            # SUPPLIER DELAY
            # ---------------------------------------------

            delay_probability = (

                SUPPLIER_DELAY_PROBABILITY[
                    demand_class
                ]

            )


            supplier_delay_days = 0


            if (

                rng.random()
                <
                delay_probability

            ):

                delay_min, delay_max = (

                    SUPPLIER_DELAY_RANGE[
                        demand_class
                    ]

                )


                supplier_delay_days = rng.integers(

                    delay_min,

                    delay_max + 1,

                )


                actual_lead_time += (

                    supplier_delay_days

                )


                delayed_orders += 1


            total_supplier_delay_days += (

                supplier_delay_days

            )


            # ---------------------------------------------
            # ARRIVAL
            # ---------------------------------------------

            arrival_date = (

                current_date

                +

                pd.Timedelta(
                    days=int(
                        actual_lead_time
                    )
                )

            )


            outstanding_orders.append(

                (
                    arrival_date,
                    purchase_order,
                )

            )


            purchase_orders += 1


            total_purchase_quantity += (

                purchase_order

            )


        # =================================================
        # METRICS
        # =================================================

        total_demand += demand_value

        total_fulfilled += fulfilled

        total_lost_sales += lost_sales

        total_inventory += inventory


    # =====================================================
    # FINAL METRICS
    # =====================================================

    days = len(group)


    fill_rate = (

        total_fulfilled
        /
        total_demand

        if total_demand > 0

        else 1.0

    )


    stockout_rate = (

        stockout_days
        /
        days

        if days > 0

        else 0

    )


    average_inventory = (

        total_inventory
        /
        days

        if days > 0

        else 0

    )


    inventory_days = (

        average_inventory
        /
        average_demand

        if average_demand > 0

        else 0

    )


    average_supplier_delay = (

        total_supplier_delay_days
        /
        purchase_orders

        if purchase_orders > 0

        else 0

    )


    supplier_delay_rate = (

        delayed_orders
        /
        purchase_orders

        if purchase_orders > 0

        else 0

    )


    # =====================================================
    # COST
    # =====================================================

    holding_cost = (

        average_inventory
        *
        HOLDING_COST_PER_UNIT
        *
        days
        /
        365

    )


    ordering_cost = (

        purchase_orders
        *
        ORDERING_COST_PER_ORDER

    )


    lost_sales_cost = (

        total_lost_sales
        *
        LOST_SALES_COST_PER_UNIT

    )


    total_cost = (

        holding_cost
        +
        ordering_cost
        +
        lost_sales_cost

    )


    return {

        "service_level":
            service_level,

        "order_coverage_days":
            order_coverage_days,

        "safety_stock":
            safety_stock,

        "reorder_point":
            reorder_point,

        "order_quantity":
            order_quantity,

        "fill_rate":
            fill_rate,

        "stockout_rate":
            stockout_rate,

        "lost_sales":
            total_lost_sales,

        "average_inventory":
            average_inventory,

        "inventory_days":
            inventory_days,

        "purchase_orders":
            purchase_orders,

        "purchase_quantity":
            total_purchase_quantity,

        "supplier_delay_rate":
            supplier_delay_rate,

        "average_supplier_delay":
            average_supplier_delay,

        "holding_cost":
            holding_cost,

        "ordering_cost":
            ordering_cost,

        "lost_sales_cost":
            lost_sales_cost,

        "total_cost":
            total_cost,

    }


# =========================================================
# OPTIMIZE PRODUCT
# =========================================================

def optimize_product(
    product_id,
    group,
    parameters,
):

    risk_class = parameters[
        "risk_class"
    ]


    candidate_service_levels = (

        SERVICE_LEVELS[
            risk_class
        ]

    )


    candidates = []


    # =====================================================
    # TEST EVERY POLICY
    # =====================================================

    for service_level in candidate_service_levels:

        for order_coverage_days in ORDER_COVERAGE_DAYS:

            # -------------------------------------------------
            # Deterministic seed for fair comparison
            # -------------------------------------------------

            policy_seed = (

                RANDOM_SEED
                +
                abs(
                    hash(
                        (
                            product_id,
                            service_level,
                            order_coverage_days,
                        )
                    )
                )
                % 1_000_000

            )


            result = simulate_policy(

                group,

                parameters,

                service_level,

                order_coverage_days,

                policy_seed,

            )


            candidates.append(
                result
            )


    candidates = pd.DataFrame(
        candidates
    )


    # =====================================================
    # SERVICE CONSTRAINT
    # =====================================================

    minimum_service_level = {

        "Critical": 0.99,

        "High": 0.975,

        "Medium": 0.95,

        "Low": 0.90,

    }[risk_class]


    feasible = candidates[

        candidates[
            "fill_rate"
        ]
        >=
        minimum_service_level

    ]


    # =====================================================
    # SELECT BEST POLICY
    # =====================================================

    if len(feasible) > 0:

        best = (

            feasible

            .sort_values(

                [
                    "total_cost",
                    "inventory_days",
                    "purchase_orders",
                ],

                ascending=[
                    True,
                    True,
                    True,
                ],

            )

            .iloc[0]

        )

    else:

        best = (

            candidates

            .sort_values(

                [
                    "fill_rate",
                    "total_cost",
                    "inventory_days",
                ],

                ascending=[
                    False,
                    True,
                    True,
                ],

            )

            .iloc[0]

        )


    # =====================================================
    # BASELINE
    # =====================================================

    baseline_seed = (

        RANDOM_SEED
        +
        abs(
            hash(
                (
                    product_id,
                    "baseline",
                )
            )
        )
        % 1_000_000

    )


    baseline = simulate_policy(

        group,

        parameters,

        BASELINE_SERVICE_LEVEL,

        BASELINE_ORDER_COVERAGE_DAYS,

        baseline_seed,

    )


    # =====================================================
    # CHANGE METRICS
    # =====================================================

    lost_sales_reduction = (

        baseline["lost_sales"]
        -
        best["lost_sales"]

    )


    cost_change = (

        best["total_cost"]
        -
        baseline["total_cost"]

    )


    cost_savings = (

        baseline["total_cost"]
        -
        best["total_cost"]

    )


    inventory_days_change = (

        best["inventory_days"]
        -
        baseline["inventory_days"]

    )


    fill_rate_change = (

        best["fill_rate"]
        -
        baseline["fill_rate"]

    )


    safety_stock_change = (

        best["safety_stock"]
        -
        baseline["safety_stock"]

    )


    # =====================================================
    # RECOMMENDATION LOGIC
    # =====================================================

    service_level_changed = (

        best["service_level"]
        >
        BASELINE_SERVICE_LEVEL

    )


    order_quantity_increased = (

        best["order_coverage_days"]
        >
        BASELINE_ORDER_COVERAGE_DAYS

    )


    order_quantity_decreased = (

        best["order_coverage_days"]
        <
        BASELINE_ORDER_COVERAGE_DAYS

    )


    if service_level_changed:

        recommended_action = (

            "Increase service level and "
            "adjust safety stock"

        )


    elif order_quantity_increased:

        recommended_action = (

            "Increase order quantity to "
            "reduce ordering frequency"

        )


    elif order_quantity_decreased:

        recommended_action = (

            "Reduce order quantity and "
            "increase ordering frequency"

        )


    else:

        recommended_action = (

            "Maintain current inventory policy"

        )


    # =====================================================
    # RESULT
    # =====================================================

    result = {

        "product_id":
            product_id,

        "category":
            parameters["category"],

        "demand_class":
            parameters["demand_class"],

        "risk_class":
            risk_class,

        "primary_risk_driver":
            parameters[
                "primary_risk_driver"
            ],


        # -------------------------------------------------
        # BASELINE
        # -------------------------------------------------

        "baseline_service_level":
            BASELINE_SERVICE_LEVEL,

        "baseline_order_coverage_days":
            BASELINE_ORDER_COVERAGE_DAYS,

        "baseline_safety_stock":
            baseline["safety_stock"],

        "baseline_reorder_point":
            baseline["reorder_point"],

        "baseline_order_quantity":
            baseline["order_quantity"],

        "baseline_fill_rate":
            baseline["fill_rate"],

        "baseline_stockout_rate":
            baseline["stockout_rate"],

        "baseline_lost_sales":
            baseline["lost_sales"],

        "baseline_inventory_days":
            baseline["inventory_days"],

        "baseline_total_cost":
            baseline["total_cost"],


        # -------------------------------------------------
        # OPTIMIZED
        # -------------------------------------------------

        "optimized_service_level":
            best["service_level"],

        "optimized_safety_stock":
            best["safety_stock"],

        "optimized_reorder_point":
            best["reorder_point"],

        "optimized_order_quantity":
            best["order_quantity"],

        "optimized_order_coverage_days":
            best["order_coverage_days"],

        "optimized_fill_rate":
            best["fill_rate"],

        "optimized_stockout_rate":
            best["stockout_rate"],

        "optimized_lost_sales":
            best["lost_sales"],

        "optimized_inventory_days":
            best["inventory_days"],

        "optimized_total_cost":
            best["total_cost"],

        "optimized_purchase_orders":
            best["purchase_orders"],

        "optimized_purchase_quantity":
            best["purchase_quantity"],

        "optimized_supplier_delay_rate":
            best["supplier_delay_rate"],

        "optimized_average_supplier_delay":
            best["average_supplier_delay"],


        # -------------------------------------------------
        # CHANGE METRICS
        # -------------------------------------------------

        "lost_sales_reduction":
            lost_sales_reduction,

        "cost_change":
            cost_change,

        "cost_savings":
            cost_savings,

        "inventory_days_change":
            inventory_days_change,

        "fill_rate_change":
            fill_rate_change,

        "safety_stock_change":
            safety_stock_change,


        # -------------------------------------------------
        # RECOMMENDATION
        # -------------------------------------------------

        "recommended_action":
            recommended_action,

    }


    return result


# =========================================================
# RUN OPTIMIZATION
# =========================================================

print()

print("=" * 60)

print(
    "RUNNING INVENTORY OPTIMIZATION"
)

print("=" * 60)

print()


optimization_results = []


for product_id, group in demand.groupby(
    "product_id",
    sort=False,
):

    parameters = (

        product_parameters[

            product_parameters[
                "product_id"
            ]
            ==
            product_id

        ]

        .iloc[0]

    )


    result = optimize_product(

        product_id,

        group,

        parameters,

    )


    optimization_results.append(
        result
    )


optimization_results = pd.DataFrame(
    optimization_results
)


# =========================================================
# VALIDATION
# =========================================================

assert (

    len(
        optimization_results
    )

    ==

    demand[
        "product_id"
    ].nunique()

)


assert (

    optimization_results[
        "product_id"
    ].is_unique

)


assert (

    optimization_results[
        "optimized_safety_stock"
    ]
    >= 0
).all()


assert (

    optimization_results[
        "optimized_reorder_point"
    ]
    >= 0
).all()


assert (

    optimization_results[
        "optimized_order_quantity"
    ]
    > 0
).all()


assert (

    optimization_results[
        "optimized_fill_rate"
    ]
    >= 0
).all()


assert (

    optimization_results[
        "optimized_fill_rate"
    ]
    <= 1
).all()


assert (

    optimization_results[
        "baseline_fill_rate"
    ]
    >= 0
).all()


assert (

    optimization_results[
        "baseline_fill_rate"
    ]
    <= 1
).all()


assert (

    optimization_results[
        "optimized_total_cost"
    ]
    >= 0
).all()


assert (

    optimization_results[
        "baseline_total_cost"
    ]
    >= 0
).all()


# =========================================================
# ROUNDING
# =========================================================

percentage_columns = [

    "baseline_service_level",

    "optimized_service_level",

    "baseline_fill_rate",

    "baseline_stockout_rate",

    "optimized_fill_rate",

    "optimized_stockout_rate",

    "fill_rate_change",

    "optimized_supplier_delay_rate",

]


for column in percentage_columns:

    optimization_results[column] = (

        optimization_results[column]

        .round(4)

    )


numeric_columns = [

    "baseline_order_coverage_days",

    "baseline_safety_stock",

    "baseline_reorder_point",

    "baseline_order_quantity",

    "baseline_lost_sales",

    "baseline_inventory_days",

    "baseline_total_cost",

    "optimized_safety_stock",

    "optimized_reorder_point",

    "optimized_order_quantity",

    "optimized_order_coverage_days",

    "optimized_lost_sales",

    "optimized_inventory_days",

    "optimized_total_cost",

    "optimized_purchase_orders",

    "optimized_purchase_quantity",

    "optimized_average_supplier_delay",

    "lost_sales_reduction",

    "cost_change",

    "cost_savings",

    "inventory_days_change",

    "safety_stock_change",

]


for column in numeric_columns:

    optimization_results[column] = (

        optimization_results[column]

        .round(2)

    )


# =========================================================
# SUMMARY BY RISK CLASS
# =========================================================

optimization_summary = (

    optimization_results

    .groupby(
        "risk_class",
        as_index=False,
    )

    .agg(

        products=(
            "product_id",
            "nunique",
        ),

        avg_baseline_fill_rate=(
            "baseline_fill_rate",
            "mean",
        ),

        avg_optimized_fill_rate=(
            "optimized_fill_rate",
            "mean",
        ),

        avg_baseline_inventory_days=(
            "baseline_inventory_days",
            "mean",
        ),

        avg_optimized_inventory_days=(
            "optimized_inventory_days",
            "mean",
        ),

        avg_inventory_days_change=(
            "inventory_days_change",
            "mean",
        ),

        total_baseline_lost_sales=(
            "baseline_lost_sales",
            "sum",
        ),

        total_optimized_lost_sales=(
            "optimized_lost_sales",
            "sum",
        ),

        total_lost_sales_reduction=(
            "lost_sales_reduction",
            "sum",
        ),

        total_baseline_cost=(
            "baseline_total_cost",
            "sum",
        ),

        total_optimized_cost=(
            "optimized_total_cost",
            "sum",
        ),

        total_cost_savings=(
            "cost_savings",
            "sum",
        ),

        avg_safety_stock_change=(
            "safety_stock_change",
            "mean",
        ),

    )

)


# =========================================================
# ROUND SUMMARY
# =========================================================

summary_percentage_columns = [

    "avg_baseline_fill_rate",

    "avg_optimized_fill_rate",

]


for column in summary_percentage_columns:

    optimization_summary[column] = (

        optimization_summary[column]

        .round(4)

    )


summary_numeric_columns = [

    "avg_baseline_inventory_days",

    "avg_optimized_inventory_days",

    "avg_inventory_days_change",

    "total_baseline_lost_sales",

    "total_optimized_lost_sales",

    "total_lost_sales_reduction",

    "total_baseline_cost",

    "total_optimized_cost",

    "total_cost_savings",

    "avg_safety_stock_change",

]


for column in summary_numeric_columns:

    optimization_summary[column] = (

        optimization_summary[column]

        .round(2)

    )


# =========================================================
# SAVE OUTPUT
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


optimization_results.to_csv(
    OUTPUT_FILE,
    index=False,
)


optimization_summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# =========================================================
# OUTPUT — MAIN SUMMARY
# =========================================================

print()

print("=" * 60)

print(
    "INVENTORY OPTIMIZATION ENGINE COMPLETE"
)

print("=" * 60)

print()


print(
    f"Products optimized: "
    f"{len(optimization_results):,}"
)


print()

print(
    "Optimization by risk class:"
)

print()


print(

    optimization_summary

    .sort_values(
        "risk_class"
    )

    .to_string(
        index=False
    )

)


# =========================================================
# TOP POLICY CHANGES
# =========================================================

print()

print(
    "Top 20 products requiring policy changes:"
)

print()


top_changes = (

    optimization_results

    .sort_values(

        [
            "cost_savings",
            "lost_sales_reduction",
        ],

        ascending=[
            False,
            False,
        ],

    )

    .head(20)

)


print(

    top_changes[

        [

            "product_id",

            "risk_class",

            "primary_risk_driver",

            "baseline_safety_stock",

            "optimized_safety_stock",

            "baseline_reorder_point",

            "optimized_reorder_point",

            "baseline_order_quantity",

            "optimized_order_quantity",

            "optimized_service_level",

            "baseline_inventory_days",

            "optimized_inventory_days",

            "baseline_total_cost",

            "optimized_total_cost",

            "lost_sales_reduction",

            "cost_savings",

            "recommended_action",

        ]

    ]

    .to_string(
        index=False
    )

)


# =========================================================
# LARGEST LOST-SALES REDUCTIONS
# =========================================================

print()

print(
    "Largest lost-sales reductions:"
)

print()


print(

    optimization_results[

        [

            "product_id",

            "risk_class",

            "baseline_lost_sales",

            "optimized_lost_sales",

            "lost_sales_reduction",

            "cost_change",

        ]

    ]

    .sort_values(

        "lost_sales_reduction",

        ascending=False,

    )

    .head(10)

    .to_string(
        index=False
    )

)


# =========================================================
# FILE OUTPUT
# =========================================================

print()

print(
    "Saved to:"
)

print()

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)


# =========================================================
# OPTIMIZATION VALIDATION
# =========================================================

print()

print("=" * 60)

print(
    "OPTIMIZATION VALIDATION"
)

print("=" * 60)

print()


# =========================================================
# OVERALL PERFORMANCE
# =========================================================

baseline_fill_rate = (

    optimization_results[
        "baseline_fill_rate"
    ].mean()

)


optimized_fill_rate = (

    optimization_results[
        "optimized_fill_rate"
    ].mean()

)


baseline_lost_sales = (

    optimization_results[
        "baseline_lost_sales"
    ].sum()

)


optimized_lost_sales = (

    optimization_results[
        "optimized_lost_sales"
    ].sum()

)


baseline_cost = (

    optimization_results[
        "baseline_total_cost"
    ].sum()

)


optimized_cost = (

    optimization_results[
        "optimized_total_cost"
    ].sum()

)


lost_sales_reduction = (

    baseline_lost_sales
    -
    optimized_lost_sales

)


cost_change = (

    optimized_cost
    -
    baseline_cost

)


print(
    "Overall performance:"
)

print()


print(
    f"Baseline fill rate: "
    f"{baseline_fill_rate:.2%}"
)


print(
    f"Optimized fill rate: "
    f"{optimized_fill_rate:.2%}"
)


print(
    f"Fill rate improvement: "
    f"{optimized_fill_rate - baseline_fill_rate:+.2%}"
)


print()


print(
    f"Baseline lost sales: "
    f"{baseline_lost_sales:,.0f}"
)


print(
    f"Optimized lost sales: "
    f"{optimized_lost_sales:,.0f}"
)


print(
    f"Lost-sales reduction: "
    f"{lost_sales_reduction:,.0f}"
)


print()


print(
    f"Baseline cost: "
    f"{baseline_cost:,.2f}"
)


print(
    f"Optimized cost: "
    f"{optimized_cost:,.2f}"
)


print(
    f"Cost change: "
    f"{cost_change:+,.2f}"
)


# =========================================================
# POLICY CHANGES
# =========================================================

print()

print(
    "Policy changes:"
)

print()


policy_change_counts = (

    optimization_results[
        "recommended_action"
    ]

    .value_counts()

)


print(
    policy_change_counts.to_string()
)


# =========================================================
# RISK CLASS IMPACT
# =========================================================

print()

print(
    "Lost-sales reduction by risk class:"
)

print()


risk_class_impact = (

    optimization_results

    .groupby(
        "risk_class"
    )

    .agg(

        products=(
            "product_id",
            "nunique",
        ),

        baseline_lost_sales=(
            "baseline_lost_sales",
            "sum",
        ),

        optimized_lost_sales=(
            "optimized_lost_sales",
            "sum",
        ),

        cost_savings=(
            "cost_savings",
            "sum",
        ),

    )

)


risk_class_impact[
    "lost_sales_reduction"
] = (

    risk_class_impact[
        "baseline_lost_sales"
    ]

    -

    risk_class_impact[
        "optimized_lost_sales"
    ]

)


print(

    risk_class_impact

    .sort_values(
        "lost_sales_reduction",
        ascending=False,
    )

    .to_string()

)


# =========================================================
# TOP OPTIMIZATION WINS
# =========================================================

print()

print(
    "Top 10 optimization wins:"
)

print()


print(

    optimization_results[

        [

            "product_id",

            "risk_class",

            "primary_risk_driver",

            "lost_sales_reduction",

            "cost_savings",

            "fill_rate_change",

            "inventory_days_change",

            "recommended_action",

        ]

    ]

    .sort_values(

        [

            "lost_sales_reduction",

            "cost_savings",

        ],

        ascending=[
            False,
            False,
        ],

    )

    .head(10)

    .to_string(
        index=False
    )

)


# =========================================================
# FINAL VALIDATION
# =========================================================

print()

print(
    "=" * 60
)

print(
    "VALIDATION COMPLETE"
)

print(
    "=" * 60
)