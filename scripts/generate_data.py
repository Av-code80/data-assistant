"""Generate the ModaNova e-commerce database (DuckDB) with realistic, reproducible data.

ModaNova is a fictional fashion e-commerce company selling in Europe.
The schema mirrors Google's public `thelook_ecommerce` dataset, so the same SQL
will later run on BigQuery.

Run (from data-assistant/):  uv run python scripts/generate_data.py
"""

import bisect
import csv
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

SEED = 42
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "shop.duckdb"
START = datetime(2024, 1, 1)
USERS_SINCE = datetime(2023, 1, 1)  # existing customer base when orders start
END = datetime(2026, 10, 24, 23, 59)

N_USERS = 8_000
N_PRODUCTS = 400

COUNTRIES = {  # country: (weight, cities)
    "France": (40, ["Paris", "Lyon", "Marseille", "Toulouse", "Lille", "Bordeaux", "Nantes"]),
    "Germany": (15, ["Berlin", "Munich", "Hamburg", "Cologne"]),
    "Spain": (12, ["Madrid", "Barcelona", "Valencia", "Seville"]),
    "Italy": (12, ["Rome", "Milan", "Turin", "Naples"]),
    "Belgium": (11, ["Brussels", "Antwerp", "Liege"]),
    "Switzerland": (10, ["Geneva", "Zurich", "Lausanne"]),
}
TRAFFIC_SOURCES = {"Search": 45, "Organic": 20, "Email": 15, "Facebook": 12, "Display": 8}
CATEGORIES = {  # category: (min retail price EUR, max retail price EUR, return probability)
    "Jeans": (45, 140, 0.12), "Sweaters": (35, 120, 0.08), "Outerwear & Coats": (90, 320, 0.10),
    "Dresses": (40, 180, 0.15), "Tops & Tees": (12, 45, 0.06), "Shoes": (55, 210, 0.14),
    "Accessories": (10, 90, 0.04), "Swim": (25, 80, 0.09), "Active": (25, 110, 0.07),
}
BRANDS = ["Atelier Nord", "Lumen", "Basalte", "Riviera", "Kova", "Oslo Studio", "Terra", "Velin"]
# Fashion seasonality: January sales, autumn collection, November-December peak.
MONTH_WEIGHT = {1: 1.1, 2: 0.8, 3: 0.9, 4: 0.9, 5: 1.0, 6: 1.0, 7: 0.9,
                8: 0.7, 9: 1.0, 10: 1.1, 11: 1.6, 12: 1.8}
ORDER_STATUS = {"Complete": 70, "Shipped": 8, "Processing": 4, "Cancelled": 9, "Returned": 9}


def pick(rng: random.Random, weights: dict) -> str:
    return rng.choices(list(weights), weights=[w[0] if isinstance(w, tuple) else w
                                               for w in weights.values()])[0]


def random_day(rng: random.Random) -> datetime:
    """A day between START and END, weighted by season and by business growth."""
    total_days = (END - START).days
    while True:
        day = START + timedelta(days=rng.randrange(total_days + 1),
                                seconds=rng.randrange(86_400))
        growth = 1 + 0.25 * (day - START).days / 365          # +25 % per year
        if rng.random() < MONTH_WEIGHT[day.month] * growth / (1.8 * 1.8):
            return day


def generate(rng: random.Random) -> dict[str, tuple[list[str], list[list]]]:
    users = []
    for user_id in range(1, N_USERS + 1):
        country = pick(rng, COUNTRIES)
        users.append([user_id, rng.choice(["F", "M"]), rng.randint(18, 75), country,
                      rng.choice(COUNTRIES[country][1]), pick(rng, TRAFFIC_SOURCES),
                      USERS_SINCE + timedelta(days=rng.randrange((END - USERS_SINCE).days))])
    users.sort(key=lambda user: user[6])
    signup_dates = [user[6] for user in users]

    products = []
    for product_id in range(1, N_PRODUCTS + 1):
        category = rng.choice(list(CATEGORIES))
        low, high, _ = CATEGORIES[category]
        retail = round(rng.uniform(low, high), 2)
        brand = rng.choice(BRANDS)
        products.append([product_id, f"{brand} {category} {product_id}", category, brand,
                         rng.choice(["Women", "Men"]), round(retail * rng.uniform(0.35, 0.55), 2),
                         retail])

    orders, items = [], []
    item_id = 0
    for order_id in range(1, 30_001):
        created = random_day(rng)
        # Only customers who had already signed up can place this order.
        user = rng.choice(users[: max(1, bisect.bisect_left(signup_dates, created))])
        status = pick(rng, ORDER_STATUS)
        shipped = created + timedelta(days=rng.randint(1, 3)) if status != "Cancelled" else None
        delivered = (shipped + timedelta(days=rng.randint(2, 6))
                     if status in ("Complete", "Returned") else None)
        returned = (delivered + timedelta(days=rng.randint(3, 25))
                    if status == "Returned" else None)
        # Recent orders cannot be delivered yet: keep dates consistent with "now".
        if delivered and delivered > END:
            status, delivered, returned = "Shipped", None, None
        n_items = rng.choices([1, 2, 3, 4], weights=[55, 28, 12, 5])[0]
        orders.append([order_id, user[0], status, created, shipped, delivered, returned, n_items])

        for _ in range(n_items):
            item_id += 1
            product = rng.choice(products)
            discount = rng.choice([1.0, 1.0, 1.0, 0.9, 0.8, 0.7]) if created.month in (1, 7) else 1.0
            item_status = status
            item_returned = returned
            # Inside a completed order, single items can still be returned.
            if status == "Complete" and rng.random() < CATEGORIES[product[2]][2]:
                item_status = "Returned"
                item_returned = delivered + timedelta(days=rng.randint(3, 25))
            items.append([item_id, order_id, user[0], product[0], item_status, created,
                          item_returned, round(product[6] * discount, 2)])

    return {
        "users": (["id", "gender", "age", "country", "city", "traffic_source", "created_at"], users),
        "products": (["id", "name", "category", "brand", "department", "cost", "retail_price"],
                     products),
        "orders": (["order_id", "user_id", "status", "created_at", "shipped_at", "delivered_at",
                    "returned_at", "num_of_item"], orders),
        "order_items": (["id", "order_id", "user_id", "product_id", "status", "created_at",
                         "returned_at", "sale_price"], items),
    }


def write_database(tables: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    with duckdb.connect(str(DB_PATH)) as db, tempfile.TemporaryDirectory() as tmp:
        for name, (columns, rows) in tables.items():
            path = Path(tmp) / f"{name}.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                writer.writerows(rows)
            db.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv('{path}', header=true)")
            count = db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            print(f"{name:12} {count:>7,} rows")


if __name__ == "__main__":
    write_database(generate(random.Random(SEED)))
    print(f"Database written to {DB_PATH}")
