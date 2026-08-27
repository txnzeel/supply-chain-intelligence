import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEMAND_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_demand.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scenario_results.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scenario_summary.csv"
)


# =========================================================
# BASE CONFIGURATION
# =========================================================

INITIAL_INVENTORY_DAYS = 30

ORDER_COVERAGE_DAYS = 30

SERVICE_LEVEL_Z = 1.65


# =========================================================
# LEAD TIME CONFIGURATION
# =========================================================

LEAD_TIME_CONFIG = {

    "Stable": (5, 10),

    "Trending": (5, 12),

    "Seasonal": (7, 14),

    "Volatile": (10, 20),

    "Intermittent": (12, 25),

}


# =========================================================
# SUPPLIER DELAY CONFIGURATION
# =========================================================

SUPPLIER_DELAY_PROBABILITY = {

    "Stable": 0.05,

    "Trending": 0.08,

    "Seasonal": 0.10,

    "Volatile": 0.15,

    "Intermittent": 0.20,

}


SUPPLIER_DELAY_RANGE = {

    "Stable": (1, 2),

    "Trending": (1, 3),

    "Seasonal": (1, 4),

    "Volatile": (2, 5),

    "Intermittent": (2, 7),

}


# =========================================================
# SCENARIOS
# =========================================================

SCENARIOS = {

    "Baseline": {

        "safety_stock_multiplier": 1.00,

        "lead_time_multiplier": 1.00,

        "order_quantity_multiplier": 1.00,

        "demand_multiplier": 1.00,

        "supplier_delay_multiplier": 1.00,

    },


    "Safety Stock +20%": {

        "safety_stock_multiplier": 1.20,

        "lead_time_multiplier": 1.00,

        "order_quantity_multiplier": 1.00,

        "demand_multiplier": 1.00,

        "supplier_delay_multiplier": 1.00,

    },


    "Safety Stock +40%": {

        "safety_stock_multiplier": 1.40,

        "lead_time_multiplier": 1.00,

        "order_quantity_multiplier": 1.00,

        "demand_multiplier": 1.00,

        "supplier_delay_multiplier": 1.00,

    },


    "Lead Time -20%": {

        "safety_stock_multiplier": 1.00,

        "lead_time_multiplier": 0.80,

        "order_quantity_multiplier": 1.00,

        "demand_multiplier": 1.00,

        "supplier_delay_multiplier": 1.00,

    },


    "Lead Time -40%": {

        "safety_stock_multiplier": 1.00,

        "lead_time_multiplier": 0.60,

        "order_quantity_multiplier": 1.00,

        "demand_multiplier": 1.00,

        "supplier_delay_multiplier": 1.00,

    },


    "Order Quantity +20%": {

        "safety_stock_multiplier": 1.00,

        "lead_time_multiplier": 1.00,

        "order_quantity_multiplier": 1.20,

        "demand_multiplier": 1.00,

        "supplier_delay_multiplier": 1.00,

    },


    "Supplier Reliability +20%": {

        "safety_stock_multiplier": 1.00,

        "lead_time_multiplier": 1.00,

        "order_quantity_multiplier": 1.00,

        "demand_multiplier": 1.00,

        "supplier_delay_multiplier": 0.80,

    },


    "Demand +15%": {

        "safety_stock_multiplier": 1.00,

        "lead_time_multiplier": 1.00,

        "order_quantity_multiplier": 1.00,

        "demand_multiplier": 1.15,

        "supplier_delay_multiplier": 1.00,

    },


    "Combined Improvement": {

        "safety_stock_multiplier": 1.20,

        "lead_time_multiplier": 0.80,

        "order_quantity_multiplier": 1.00,

        "demand_multiplier": 1.00,

        "supplier_delay_multiplier": 0.80,

    },

}


# =========================================================
# LOAD DEMAND
# =========================================================

demand = pd.read_csv(
    DEMAND_FILE,
    parse_dates=["date"],
)


# =========================================================
# VALIDATION
# =========================================================

required_columns = [

    "product_id",

    "category",

    "demand_class",

    "base_daily_demand",

    "expected_demand",

    "actual_demand",

]


missing_columns = [

    column

    for column in required_columns

    if column not in demand.columns

]


assert not missing_columns, (

    f"Missing required columns: {missing_columns}"

)


assert demand["product_id"].notna().all()

assert demand["date"].notna().all()

assert demand["actual_demand"].notna().all()

assert (
    demand["actual_demand"] >= 0
).all()


# =========================================================
# SORT
# =========================================================

demand = (

    demand

    .sort_values(
        [
            "product_id",
            "date",
        ]
    )

    .reset_index(drop=True)

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

    product_parameters["demand_std"]

    .fillna(0)

)


# =========================================================
# LEAD TIME PARAMETERS
# =========================================================

product_parameters["min_lead_time"] = (

    product_parameters["demand_class"]

    .map(
        {
            key: value[0]

            for key, value
            in LEAD_TIME_CONFIG.items()
        }
    )

)


product_parameters["max_lead_time"] = (

    product_parameters["demand_class"]

    .map(
        {
            key: value[1]

            for key, value
            in LEAD_TIME_CONFIG.items()
        }
    )

)


product_parameters["average_lead_time"] = (

    (

        product_parameters["min_lead_time"]

        +

        product_parameters["max_lead_time"]

    )

    / 2

)


# =========================================================
# SAFETY STOCK
# =========================================================

product_parameters["base_safety_stock"] = (

    SERVICE_LEVEL_Z

    *

    product_parameters["demand_std"]

    *

    np.sqrt(
        product_parameters["average_lead_time"]
    )

)


product_parameters["base_safety_stock"] = (

    product_parameters["base_safety_stock"]

    .round()

    .astype(int)

)


# =========================================================
# ORDER QUANTITY
# =========================================================

product_parameters["base_order_quantity"] = (

    product_parameters["average_daily_demand"]

    *

    ORDER_COVERAGE_DAYS

)


product_parameters["base_order_quantity"] = (

    product_parameters["base_order_quantity"]

    .round()

    .astype(int)

    .clip(lower=1)

)


# =========================================================
# SCENARIO SIMULATION
# =========================================================

def run_scenario(
    scenario_name,
    scenario_config,
):

    print(
        f"Running scenario: {scenario_name}"
    )


    results = []


    for product_id, group in demand.groupby(
        "product_id",
        sort=False,
    ):

        parameters = (

            product_parameters[

                product_parameters["product_id"]
                == product_id

            ]

            .iloc[0]

        )


        demand_class = (
            parameters["demand_class"]
        )


        # =================================================
        # SCENARIO PARAMETERS
        # =================================================

        demand_multiplier = (
            scenario_config[
                "demand_multiplier"
            ]
        )


        safety_stock_multiplier = (
            scenario_config[
                "safety_stock_multiplier"
            ]
        )


        lead_time_multiplier = (
            scenario_config[
                "lead_time_multiplier"
            ]
        )


        order_quantity_multiplier = (
            scenario_config[
                "order_quantity_multiplier"
            ]
        )


        supplier_delay_multiplier = (
            scenario_config[
                "supplier_delay_multiplier"
            ]
        )


        average_demand = (

            parameters[
                "average_daily_demand"
            ]

            *

            demand_multiplier

        )


        safety_stock = (

            parameters[
                "base_safety_stock"
            ]

            *

            safety_stock_multiplier

        )


        safety_stock = max(
            0,
            round(safety_stock),
        )


        average_lead_time = (

            parameters[
                "average_lead_time"
            ]

            *

            lead_time_multiplier

        )


        reorder_point = (

            average_demand

            *

            average_lead_time

            +

            safety_stock

        )


        reorder_point = max(
            0,
            round(reorder_point),
        )


        order_quantity = (

            parameters[
                "base_order_quantity"
            ]

            *

            order_quantity_multiplier

        )


        order_quantity = max(
            1,
            round(order_quantity),
        )


        initial_inventory = (

            average_demand

            *

            INITIAL_INVENTORY_DAYS

        )


        initial_inventory = max(
            0,
            round(initial_inventory),
        )


        # =================================================
        # INVENTORY STATE
        # =================================================

        inventory = initial_inventory

        outstanding_orders = []


        # =================================================
        # METRICS
        # =================================================

        total_demand = 0

        total_fulfilled = 0

        total_lost_sales = 0

        stockout_days = 0

        total_inventory = 0

        total_purchase_quantity = 0

        purchase_orders = 0

        delayed_orders = 0

        total_actual_lead_time = 0

        total_supplier_delay = 0


        # =================================================
        # DAILY SIMULATION
        # =================================================

        for _, row in group.iterrows():

            current_date = row["date"]


            demand_value = round(

                row["actual_demand"]

                *

                demand_multiplier

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


            if arrivals_today > 0:

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


            # =================================================
            # STOCKOUT
            # =================================================

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

            purchase_order = 0

            actual_lead_time = 0

            supplier_delay_days = 0


            if (

                inventory_position
                <=
                reorder_point

                and

                len(outstanding_orders) == 0

            ):

                purchase_order = (
                    order_quantity
                )


                # -----------------------------------------
                # BASE LEAD TIME
                # -----------------------------------------

                base_min = (

                    parameters[
                        "min_lead_time"
                    ]

                    *

                    lead_time_multiplier

                )


                base_max = (

                    parameters[
                        "max_lead_time"
                    ]

                    *

                    lead_time_multiplier

                )


                min_lead = max(
                    1,
                    round(base_min),
                )


                max_lead = max(
                    min_lead,
                    round(base_max),
                )


                actual_lead_time = (

                    np.random.randint(
                        min_lead,
                        max_lead + 1,
                    )

                )


                # -----------------------------------------
                # SUPPLIER DELAY
                # -----------------------------------------

                delay_probability = (

                    SUPPLIER_DELAY_PROBABILITY[
                        demand_class
                    ]

                    *

                    supplier_delay_multiplier

                )


                delay_probability = min(
                    1.0,
                    max(
                        0.0,
                        delay_probability,
                    ),
                )


                if (

                    np.random.random()
                    <
                    delay_probability

                ):

                    delay_min, delay_max = (

                        SUPPLIER_DELAY_RANGE[
                            demand_class
                        ]

                    )


                    supplier_delay_days = (

                        np.random.randint(
                            delay_min,
                            delay_max + 1,
                        )

                    )


                    actual_lead_time += (
                        supplier_delay_days
                    )


                    delayed_orders += 1


                # -----------------------------------------
                # ARRIVAL
                # -----------------------------------------

                arrival_date = (

                    current_date

                    +

                    pd.Timedelta(
                        days=actual_lead_time
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


                total_actual_lead_time += (
                    actual_lead_time
                )


                total_supplier_delay += (
                    supplier_delay_days
                )


            # =================================================
            # METRICS
            # =================================================

            total_demand += demand_value

            total_fulfilled += fulfilled

            total_lost_sales += lost_sales

            total_inventory += inventory


        # =================================================
        # PRODUCT METRICS
        # =================================================

        days = len(group)


        if total_demand > 0:

            fill_rate = (

                total_fulfilled
                /
                total_demand

            )

        else:

            fill_rate = 1.0


        stockout_rate = (

            stockout_days
            /
            days

        )


        average_inventory = (

            total_inventory
            /
            days

        )


        inventory_days = (

            average_inventory
            /
            average_demand

            if average_demand > 0

            else 0

        )


        if purchase_orders > 0:

            supplier_delay_rate = (

                delayed_orders
                /
                purchase_orders

            )


            average_actual_lead_time = (

                total_actual_lead_time
                /
                purchase_orders

            )


            average_supplier_delay = (

                total_supplier_delay
                /
                purchase_orders

            )

        else:

            supplier_delay_rate = 0

            average_actual_lead_time = 0

            average_supplier_delay = 0


        # =================================================
        # STORE PRODUCT RESULT
        # =================================================

        results.append({

            "scenario":
                scenario_name,

            "product_id":
                product_id,

            "category":
                parameters["category"],

            "demand_class":
                demand_class,

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

            "average_actual_lead_time":
                average_actual_lead_time,

            "average_supplier_delay":
                average_supplier_delay,

        })


    return pd.DataFrame(results)


# =========================================================
# RUN ALL SCENARIOS
# =========================================================

scenario_results = []


for scenario_name, scenario_config in SCENARIOS.items():

    result = run_scenario(

        scenario_name,

        scenario_config,

    )


    scenario_results.append(
        result
    )


scenario_results = pd.concat(

    scenario_results,

    ignore_index=True,

)


# =========================================================
# SCENARIO SUMMARY
# =========================================================

scenario_summary = (

    scenario_results

    .groupby(
        "scenario",
        as_index=False,
    )

    .agg(

        products=(
            "product_id",
            "nunique",
        ),

        avg_fill_rate=(
            "fill_rate",
            "mean",
        ),

        avg_stockout_rate=(
            "stockout_rate",
            "mean",
        ),

        total_lost_sales=(
            "lost_sales",
            "sum",
        ),

        avg_inventory=(
            "average_inventory",
            "mean",
        ),

        avg_inventory_days=(
            "inventory_days",
            "mean",
        ),

        total_purchase_orders=(
            "purchase_orders",
            "sum",
        ),

        total_purchase_quantity=(
            "purchase_quantity",
            "sum",
        ),

        avg_supplier_delay_rate=(
            "supplier_delay_rate",
            "mean",
        ),

        avg_actual_lead_time=(
            "average_actual_lead_time",
            "mean",
        ),

        avg_supplier_delay=(
            "average_supplier_delay",
            "mean",
        ),

    )

)


# =========================================================
# BASELINE COMPARISON
# =========================================================

baseline = (

    scenario_summary[

        scenario_summary["scenario"]
        ==
        "Baseline"

    ]

    .iloc[0]

)


scenario_summary["fill_rate_change"] = (

    scenario_summary["avg_fill_rate"]

    -

    baseline["avg_fill_rate"]

)


scenario_summary["stockout_rate_change"] = (

    scenario_summary["avg_stockout_rate"]

    -

    baseline["avg_stockout_rate"]

)


scenario_summary["lost_sales_change"] = (

    scenario_summary["total_lost_sales"]

    -

    baseline["total_lost_sales"]

)


scenario_summary["inventory_days_change"] = (

    scenario_summary["avg_inventory_days"]

    -

    baseline["avg_inventory_days"]

)


scenario_summary["lost_sales_reduction"] = (

    baseline["total_lost_sales"]

    -

    scenario_summary["total_lost_sales"]

)


scenario_summary["lost_sales_reduction_pct"] = (

    np.where(

        baseline["total_lost_sales"] > 0,

        (

            baseline["total_lost_sales"]

            -

            scenario_summary[
                "total_lost_sales"
            ]

        )

        /

        baseline["total_lost_sales"],

        0,

    )

)


# =========================================================
# ROUNDING
# =========================================================

scenario_summary[

    [
        "avg_fill_rate",

        "avg_stockout_rate",

        "fill_rate_change",

        "stockout_rate_change",

        "avg_supplier_delay_rate",

        "lost_sales_reduction_pct",

    ]

] = scenario_summary[

    [
        "avg_fill_rate",

        "avg_stockout_rate",

        "fill_rate_change",

        "stockout_rate_change",

        "avg_supplier_delay_rate",

        "lost_sales_reduction_pct",

    ]

].round(4)


scenario_summary[

    [
        "avg_inventory",

        "avg_inventory_days",

        "inventory_days_change",

        "avg_actual_lead_time",

        "avg_supplier_delay",

    ]

] = scenario_summary[

    [
        "avg_inventory",

        "avg_inventory_days",

        "inventory_days_change",

        "avg_actual_lead_time",

        "avg_supplier_delay",

    ]

].round(2)


# =========================================================
# SAVE
# =========================================================

OUTPUT_FILE.parent.mkdir(

    parents=True,

    exist_ok=True,

)


scenario_results.to_csv(

    OUTPUT_FILE,

    index=False,

)


scenario_summary.to_csv(

    SUMMARY_FILE,

    index=False,

)


# =========================================================
# OUTPUT
# =========================================================

print()

print("=" * 60)

print(
    "INVENTORY SCENARIO ENGINE COMPLETE"
)

print("=" * 60)

print()


print(

    scenario_summary[

        [

            "scenario",

            "avg_fill_rate",

            "avg_stockout_rate",

            "total_lost_sales",

            "avg_inventory_days",

            "total_purchase_quantity",

            "avg_supplier_delay_rate",

        ]

    ]

    .sort_values(

        "avg_fill_rate",

        ascending=False,

    )

    .to_string(

        index=False

    )

)


# =========================================================
# BASELINE
# =========================================================

print()

print(
    "Baseline:"
)

print(

    f"Fill rate: "
    f"{baseline['avg_fill_rate']:.2%}"

)


print(

    f"Stockout rate: "
    f"{baseline['avg_stockout_rate']:.2%}"

)


print(

    f"Lost sales: "
    f"{baseline['total_lost_sales']:,.0f}"

)


print(

    f"Inventory days: "
    f"{baseline['avg_inventory_days']:.2f}"

)


print()

print(
    "Best scenario by fill rate:"
)


best_fill_rate = (

    scenario_summary

    .sort_values(
        "avg_fill_rate",
        ascending=False,
    )

    .iloc[0]

)


print(

    f"{best_fill_rate['scenario']} "
    f"({best_fill_rate['avg_fill_rate']:.2%})"

)


print()

print(
    "Best scenario by lost-sales reduction:"
)


best_lost_sales = (

    scenario_summary

    .sort_values(
        "lost_sales_reduction",
        ascending=False,
    )

    .iloc[0]

)


print(

    f"{best_lost_sales['scenario']} "
    f"("
    f"{best_lost_sales['lost_sales_reduction']:,.0f} "
    f"fewer lost sales"
    f")"

)


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