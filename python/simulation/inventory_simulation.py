import numpy as np
import pandas as pd
from pathlib import Path
from statistics import NormalDist


# =========================================================
# CONFIGURATION
# =========================================================

np.random.seed(42)

INITIAL_INVENTORY_DAYS = 30

SERVICE_LEVEL_TARGET = 0.95

ORDER_COVERAGE_DAYS = 30

MIN_ORDER_QUANTITY = 1


# =========================================================
# PATHS
# =========================================================

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
    / "inventory_daily.csv"
)


# =========================================================
# LOAD DEMAND DATA
# =========================================================

demand = pd.read_csv(
    DEMAND_FILE,
    parse_dates=["date"],
)


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

assert demand["product_id"].notna().all()

assert demand["date"].notna().all()

assert demand["actual_demand"].notna().all()

assert (
    demand["actual_demand"] >= 0
).all()


# =========================================================
# SORT DATA
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
        base_daily_demand=(
            "base_daily_demand",
            "first",
        ),

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
# SUPPLIER LEAD TIME
# =========================================================

LEAD_TIME_CONFIG = {
    "Stable": (5, 10),
    "Trending": (5, 12),
    "Seasonal": (7, 14),
    "Volatile": (10, 20),
    "Intermittent": (12, 25),
}


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


# Minimum possible supplier lead time

product_parameters["min_lead_time_days"] = (
    product_parameters["demand_class"]
    .map(
        lambda demand_class:
        LEAD_TIME_CONFIG[demand_class][0]
    )
)


# Maximum possible supplier lead time

product_parameters["max_lead_time_days"] = (
    product_parameters["demand_class"]
    .map(
        lambda demand_class:
        LEAD_TIME_CONFIG[demand_class][1]
    )
)


assert (
    product_parameters["min_lead_time_days"]
    > 0
).all()


assert (
    product_parameters["max_lead_time_days"]
    >=
    product_parameters["min_lead_time_days"]
).all()


# =========================================================
# SERVICE LEVEL / Z-SCORE
# =========================================================

SERVICE_LEVEL_Z = NormalDist().inv_cdf(
    SERVICE_LEVEL_TARGET
)


# =========================================================
# LEAD TIME STATISTICS
# =========================================================

product_parameters["average_lead_time_days"] = (
    (
        product_parameters["min_lead_time_days"]
        +
        product_parameters["max_lead_time_days"]
    )
    / 2
)


product_parameters["lead_time_std"] = (
    (
        product_parameters["max_lead_time_days"]
        -
        product_parameters["min_lead_time_days"]
    )
    / 4
)


assert (
    product_parameters["average_lead_time_days"]
    > 0
).all()


assert (
    product_parameters["lead_time_std"]
    >= 0
).all()


# =========================================================
# SAFETY STOCK
# =========================================================

product_parameters["safety_stock"] = (
    SERVICE_LEVEL_Z
    *
    np.sqrt(
        (
            product_parameters["average_lead_time_days"]
            *
            product_parameters["demand_std"] ** 2
        )
        +
        (
            product_parameters["average_daily_demand"] ** 2
            *
            product_parameters["lead_time_std"] ** 2
        )
    )
)


product_parameters["safety_stock"] = (
    product_parameters["safety_stock"]
    .round()
    .astype(int)
)


assert (
    product_parameters["safety_stock"]
    >= 0
).all()


# =========================================================
# REORDER POINT
# =========================================================

product_parameters["reorder_point"] = (
    product_parameters["average_daily_demand"]
    *
    product_parameters["average_lead_time_days"]
    +
    product_parameters["safety_stock"]
)


product_parameters["reorder_point"] = (
    product_parameters["reorder_point"]
    .round()
    .astype(int)
)


assert (
    product_parameters["reorder_point"]
    >= 0
).all()


# =========================================================
# ORDER QUANTITY
# =========================================================

product_parameters["order_quantity"] = (
    product_parameters["average_daily_demand"]
    *
    ORDER_COVERAGE_DAYS
)


product_parameters["order_quantity"] = (
    product_parameters["order_quantity"]
    .round()
    .astype(int)
    .clip(
        lower=MIN_ORDER_QUANTITY
    )
)


assert (
    product_parameters["order_quantity"]
    > 0
).all()


# =========================================================
# INITIAL INVENTORY
# =========================================================

product_parameters["initial_inventory"] = (
    product_parameters["average_daily_demand"]
    *
    INITIAL_INVENTORY_DAYS
)


product_parameters["initial_inventory"] = (
    product_parameters["initial_inventory"]
    .round()
    .astype(int)
)


assert (
    product_parameters["initial_inventory"]
    >= 0
).all()


# =========================================================
# MERGE PARAMETERS INTO DAILY DATA
# =========================================================

inventory_data = demand.merge(
    product_parameters[
        [
            "product_id",
            "average_daily_demand",
            "demand_std",
            "min_lead_time_days",
            "max_lead_time_days",
            "average_lead_time_days",
            "lead_time_std",
            "safety_stock",
            "reorder_point",
            "order_quantity",
            "initial_inventory",
        ]
    ],
    on="product_id",
    how="left",
)


# =========================================================
# VALIDATION
# =========================================================

assert (
    inventory_data["min_lead_time_days"]
    .notna()
    .all()
)


assert (
    inventory_data["max_lead_time_days"]
    .notna()
    .all()
)


assert (
    inventory_data["average_lead_time_days"]
    .notna()
    .all()
)


assert (
    inventory_data["reorder_point"]
    .notna()
    .all()
)


assert (
    inventory_data["order_quantity"]
    > 0
).all()


# =========================================================
# INVENTORY STATE COLUMNS
# =========================================================

inventory_data["opening_inventory"] = 0

inventory_data["demand_fulfilled"] = 0

inventory_data["lost_sales"] = 0

inventory_data["backorder_quantity"] = 0

inventory_data["closing_inventory"] = 0

inventory_data["purchase_order"] = 0

inventory_data["purchase_order_arrival"] = 0

inventory_data["actual_lead_time"] = 0

inventory_data["supplier_delay_days"] = 0

inventory_data["inventory_position"] = 0

inventory_data["stockout_flag"] = 0

inventory_data["days_until_arrival"] = 0


# =========================================================
# SIMULATION
# =========================================================

results = []


for product_id, group in inventory_data.groupby(
    "product_id",
    sort=False,
):

    group = (
        group
        .sort_values("date")
        .copy()
    )


    # -----------------------------------------------------
    # PRODUCT PARAMETERS
    # -----------------------------------------------------

    initial_inventory = int(
        group["initial_inventory"].iloc[0]
    )


    reorder_point = float(
        group["reorder_point"].iloc[0]
    )


    order_quantity = int(
        group["order_quantity"].iloc[0]
    )


    min_lead_time = int(
        group["min_lead_time_days"].iloc[0]
    )


    max_lead_time = int(
        group["max_lead_time_days"].iloc[0]
    )


    demand_class = (
        group["demand_class"].iloc[0]
    )


    # -----------------------------------------------------
    # INVENTORY STATE
    # -----------------------------------------------------

    inventory = initial_inventory

    backorder = 0

    outstanding_orders = []


    # -----------------------------------------------------
    # DAILY SIMULATION
    # -----------------------------------------------------

    for index, row in group.iterrows():

        current_date = row["date"]

        demand_value = int(
            row["actual_demand"]
        )


        # =================================================
        # RESET DAILY VALUES
        # =================================================

        purchase_order = 0

        actual_lead_time = 0

        supplier_delay_days = 0


        # =================================================
        # RECEIVE PURCHASE ORDERS
        # =================================================

        arrivals_today = sum(
            quantity

            for arrival_date, quantity
            in outstanding_orders

            if arrival_date == current_date
        )


        if arrivals_today > 0:

            inventory += arrivals_today


        # Remove received orders

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
        # OPENING INVENTORY
        # =================================================

        opening_inventory = inventory


        # =================================================
        # FULFILL EXISTING BACKORDER
        # =================================================

        backorder_fulfilled = min(
            inventory,
            backorder,
        )


        inventory -= backorder_fulfilled

        backorder -= backorder_fulfilled


        # =================================================
        # FULFILL CURRENT DEMAND
        # =================================================

        demand_fulfilled = min(
            inventory,
            demand_value,
        )


        inventory -= demand_fulfilled


        # =================================================
        # STOCKOUT / LOST SALES
        # =================================================

        lost_sales = (
            demand_value
            -
            demand_fulfilled
        )


        stockout_flag = int(
            lost_sales > 0
        )


        # =================================================
        # INVENTORY POSITION BEFORE REORDER
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
            -
            backorder
        )


        # =================================================
        # REORDER DECISION
        # =================================================

        if inventory_position <= reorder_point:

            purchase_order = order_quantity


            # ---------------------------------------------
            # BASE SUPPLIER LEAD TIME
            # ---------------------------------------------

            actual_lead_time = np.random.randint(
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


            # ---------------------------------------------
            # ARRIVAL DATE
            # ---------------------------------------------

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


        # =================================================
        # FINAL INVENTORY POSITION
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
            -
            backorder
        )


        # =================================================
        # DAYS UNTIL NEXT ARRIVAL
        # =================================================

        if outstanding_orders:

            next_arrival_date = min(
                arrival_date

                for arrival_date, _
                in outstanding_orders
            )


            days_until_arrival = max(
                0,
                (
                    next_arrival_date
                    -
                    current_date
                ).days,
            )

        else:

            days_until_arrival = 0


        # =================================================
        # STORE RESULTS
        # =================================================

        group.at[
            index,
            "opening_inventory"
        ] = opening_inventory


        group.at[
            index,
            "demand_fulfilled"
        ] = demand_fulfilled


        group.at[
            index,
            "lost_sales"
        ] = lost_sales


        group.at[
            index,
            "backorder_quantity"
        ] = backorder


        group.at[
            index,
            "closing_inventory"
        ] = inventory


        group.at[
            index,
            "purchase_order"
        ] = purchase_order


        group.at[
            index,
            "purchase_order_arrival"
        ] = arrivals_today


        group.at[
            index,
            "actual_lead_time"
        ] = (
            actual_lead_time
            if purchase_order > 0
            else 0
        )


        group.at[
            index,
            "supplier_delay_days"
        ] = (
            supplier_delay_days
            if purchase_order > 0
            else 0
        )


        group.at[
            index,
            "inventory_position"
        ] = inventory_position


        group.at[
            index,
            "stockout_flag"
        ] = stockout_flag


        group.at[
            index,
            "days_until_arrival"
        ] = days_until_arrival


    # -----------------------------------------------------
    # SAVE PRODUCT RESULTS
    # -----------------------------------------------------

    results.append(group)


# =========================================================
# COMBINE RESULTS
# =========================================================

inventory_data = pd.concat(
    results,
    ignore_index=True,
)


# =========================================================
# FINAL METRICS
# =========================================================

inventory_data["fill_rate"] = np.where(
    inventory_data["actual_demand"] > 0,

    inventory_data["demand_fulfilled"]
    /
    inventory_data["actual_demand"],

    1.0,
)


inventory_data["inventory_utilization"] = np.where(
    inventory_data["opening_inventory"] > 0,

    inventory_data["demand_fulfilled"]
    /
    inventory_data["opening_inventory"],

    0,
)


# =========================================================
# VALIDATION
# =========================================================

assert (
    inventory_data["opening_inventory"]
    >= 0
).all()


assert (
    inventory_data["closing_inventory"]
    >= 0
).all()


assert (
    inventory_data["demand_fulfilled"]
    >= 0
).all()


assert (
    inventory_data["lost_sales"]
    >= 0
).all()


assert (
    inventory_data["purchase_order"]
    >= 0
).all()


assert (
    inventory_data["stockout_flag"]
    .isin([0, 1])
    .all()
)


assert (
    inventory_data["actual_lead_time"]
    >= 0
).all()


assert (
    inventory_data["supplier_delay_days"]
    >= 0
).all()


assert (
    inventory_data["days_until_arrival"]
    >= 0
).all()


# =========================================================
# SAVE
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


inventory_data.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

print()

print("=" * 60)

print(
    "INVENTORY SIMULATION COMPLETE"
)

print("=" * 60)

print()


print(
    f"Products: "
    f"{inventory_data['product_id'].nunique():,}"
)


print(
    f"Days: "
    f"{inventory_data['date'].nunique():,}"
)


print(
    f"Rows: "
    f"{len(inventory_data):,}"
)


print()


print(
    f"Stockout rate: "
    f"{inventory_data['stockout_flag'].mean():.2%}"
)


overall_fill_rate = (
    inventory_data["demand_fulfilled"].sum()
    /
    inventory_data["actual_demand"].sum()
)


print(
    f"Overall fill rate: "
    f"{overall_fill_rate:.2%}"
)


print(
    f"Total lost sales: "
    f"{inventory_data['lost_sales'].sum():,.0f}"
)


print(
    f"Total purchase quantity: "
    f"{inventory_data['purchase_order'].sum():,.0f}"
)


# =========================================================
# SUPPLIER METRICS
# =========================================================

purchase_orders = inventory_data[
    inventory_data["purchase_order"] > 0
].copy()


print()


print(
    f"Purchase orders: "
    f"{len(purchase_orders):,}"
)


delayed_orders = (
    purchase_orders[
        purchase_orders["supplier_delay_days"] > 0
    ]
)


print(
    f"Delayed orders: "
    f"{len(delayed_orders):,}"
)


if len(purchase_orders) > 0:

    supplier_delay_rate = (
        (
            purchase_orders[
                "supplier_delay_days"
            ]
            > 0
        )
        .mean()
    )


    average_actual_lead_time = (
        purchase_orders[
            "actual_lead_time"
        ]
        .mean()
    )


    average_supplier_delay = (
        purchase_orders[
            "supplier_delay_days"
        ]
        .mean()
    )

else:

    supplier_delay_rate = 0

    average_actual_lead_time = 0

    average_supplier_delay = 0


print(
    f"Supplier delay rate: "
    f"{supplier_delay_rate:.2%}"
)


print(
    f"Average actual lead time: "
    f"{average_actual_lead_time:.2f} days"
)


print(
    f"Average supplier delay: "
    f"{average_supplier_delay:.2f} days"
)


# =========================================================
# INVENTORY SUMMARY
# =========================================================

print()

print("Inventory summary:")

print()


print(
    inventory_data[
        [
            "opening_inventory",
            "closing_inventory",
            "inventory_position",
            "actual_demand",
            "demand_fulfilled",
            "lost_sales",
            "average_lead_time_days",
            "actual_lead_time",
            "supplier_delay_days",
        ]
    ]
    .describe()
)


# =========================================================
# SAMPLE
# =========================================================

print()

print("Sample:")

print()


print(
    inventory_data[
        [
            "product_id",
            "date",
            "actual_demand",
            "opening_inventory",
            "demand_fulfilled",
            "lost_sales",
            "closing_inventory",
            "purchase_order",
            "purchase_order_arrival",
            "actual_lead_time",
            "supplier_delay_days",
            "days_until_arrival",
            "inventory_position",
            "stockout_flag",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# =========================================================
# OUTPUT
# =========================================================

print()

print("Saved to:")

print()

print(OUTPUT_FILE)