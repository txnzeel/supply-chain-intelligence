import numpy as np
import pandas as pd
from faker import Faker
from pathlib import Path

# Reproducibility
np.random.seed(42)

fake = Faker()
fake.seed_instance(42)


def generate_products(n=2000):

    # Product configuration by category
    category_config = {
        "Electronics": {
    "subcategories": ["Smartphones", "Laptops", "TVs", "Audio"],
    "brands": ["NovaTech", "Vertex", "Apex", "Zenith"],
    "demand_classes": ["Stable", "Trending", "Seasonal", "Volatile"],
    "price_range": (5000, 150000),
    "demand_range": (2, 30)
},

        "Home Appliances": {
            "subcategories": [
                "Refrigerators",
                "Washing Machines",
                "Air Conditioners",
                "Microwaves"
            ],
            "brands": ["HomePro", "Arctic", "ComfortMax", "EverHome"],
            "demand_classes": ["Stable", "Seasonal", "Volatile"],
            "price_range": (3000, 120000),
            "demand_range": (2, 30)
        },

        "Personal Care": {
            "subcategories": [
                "Skincare",
                "Haircare",
                "Oral Care",
                "Grooming"
            ],
            "brands": ["PureLife", "Glow", "CarePlus", "DermaX"],
            "demand_classes": ["Stable", "Trending", "Seasonal"],
            "price_range": (100, 1000),
            "demand_range": (20, 300)
        },

        "Office Supplies": {
            "subcategories": [
                "Paper",
                "Writing",
                "Desk Accessories",
                "Storage"
            ],
            "brands": ["OfficePro", "WriteWell", "DeskMate", "PaperWorks"],
            "demand_classes": ["Stable", "Trending", "Intermittent"],
            "price_range": (50, 5000),
            "demand_range": (30, 500)
        },

        "Accessories": {
            "subcategories": [
                "Bags",
                "Chargers",
                "Cables",
                "Mobile Accessories"
            ],
            "brands": ["UrbanGear", "ConnectX", "CarryPro", "TechMate"],
            "demand_classes": ["Stable", "Trending", "Volatile"],
            "price_range": (100, 15000),
            "demand_range": (10, 250)
        }
    }

    # Category probabilities
    categories = list(category_config.keys())

    category_probabilities = [
        0.25,
        0.20,
        0.20,
        0.15,
        0.20
    ]

    # Generate categories
    product_categories = np.random.choice(
        categories,
        size=n,
        p=category_probabilities
    )

    # Generate product IDs
    product_ids = [
        f"P{i:05d}"
        for i in range(1, n + 1)
    ]

    # Generate subcategories based on category
    product_subcategories = [
        np.random.choice(
            category_config[category]["subcategories"]
        )
        for category in product_categories
    ]

    # Generate brands based on category
    product_brands = [
        np.random.choice(
            category_config[category]["brands"]
        )
        for category in product_categories
    ]

    # Generate demand classes based on category
    product_demand_classes = [
        np.random.choice(
            category_config[category]["demand_classes"]
        )
        for category in product_categories
    ]
    print(category_config)
    base_demands = []
    
    for category in product_categories:
    
            min_demand, max_demand = category_config[category]["demand_range"]
    
            demand = np.random.uniform(
                min_demand,
                max_demand
            )
    
            base_demands.append(round(demand, 2))


    selling_prices = []

    for category in product_categories:

        min_price, max_price = category_config[category]["price_range"]

        price = np.random.lognormal(
            mean=np.log((min_price + max_price) / 2),
            sigma=0.6
        )

        price = np.clip(price, min_price, max_price)

        selling_prices.append(round(price, 2))


    margin_rates = np.random.uniform(0.15, 0.40, size=n)

    unit_costs = selling_prices * (1 - margin_rates)

    unit_costs = np.round(unit_costs, 2)


    # Generate product names
    product_names = [
        f"{brand} {subcategory} {i:05d}"
        for i, (brand, subcategory) in enumerate(
            zip(product_brands, product_subcategories),
            start=1
        )
    ]

    # Create product DataFrame
    products = pd.DataFrame({
        "product_id": product_ids,
        "product_name": product_names,
        "category": product_categories,
        "subcategory": product_subcategories,
        "brand": product_brands,
        "demand_class": product_demand_classes,
        "selling_price": selling_prices,
        "unit_cost": unit_costs,
        "base_daily_demand": base_demands,
    })
# Validate demand ranges by category
    for category, config in category_config.items():

        min_demand, max_demand = config["demand_range"]

    category_demand = products.loc[
        products["category"] == category,
        "base_daily_demand"
    ]

    assert (category_demand >= min_demand).all(), (
        f"{category}: demand below minimum"
    )

    assert (category_demand <= max_demand).all(), (
        f"{category}: demand above maximum"
    )

    products["gross_profit"] = (
    products["selling_price"] - products["unit_cost"]
)

    products["gross_margin_pct"] = (
    products["gross_profit"] / products["selling_price"]
)
    products["gross_profit"] = products["gross_profit"].round(2)
    products["gross_margin_pct"] = products["gross_margin_pct"].round(4)

    return products

if __name__ == "__main__":

    products = generate_products()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    products.to_csv(
    OUTPUT_DIR / "products.csv",
    index=False)

    # Data validation
    assert len(products) == 2000
    assert products["product_id"].is_unique
    assert products["category"].notna().all()
    assert (products["unit_cost"] > 0).all()
    assert (products["selling_price"] > 0).all()
    assert (products["selling_price"] >= products["unit_cost"]).all()
    assert (products["gross_profit"] >= 0).all()
    assert (products["gross_margin_pct"] >= 0).all()
    assert (products["gross_margin_pct"] <= 1).all()
    # Preview
    # print(products.head(10).to_string(index=False))

    # print()

    # # Category distribution
    # print(products["category"].value_counts())

    print(
    products[
        [
            "product_id",
            "category",
            "demand_class",
            "base_daily_demand"
        ]
    ].head(10).to_string(index=False)
)
    print()
    print("Average gross margin:")
    print(f"{products['gross_margin_pct'].mean():.2%}")
    print()
    print("Gross margin by category:")
    print(
    products.groupby("category")["gross_margin_pct"]
    .mean()
    .sort_values(ascending=False)
)
