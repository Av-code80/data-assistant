"""View layer: the analytics marts, the only data the SQL agent is allowed to read.

Built from the source tables (orders, order_items, products, users) into a separate
database file. The agent connects to that file only, so it physically cannot read the
source tables or sensitive columns (product cost), and every business rule from
knowledge/rules.md is applied once, here:
- cancelled items are excluded everywhere;
- returned items stay (they count in the return rate) but bring no revenue.

Materialized as tables, not views: a DuckDB view cannot outlive the database it reads
from. In BigQuery, the same layer would be authorized views over the source dataset.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class Mart:
    name: str
    description: str   # shown to the LLM: what one row is and how to use it
    sql: str


AGE_GROUP = """CASE
            WHEN u.age < 25 THEN '18-24' WHEN u.age < 35 THEN '25-34'
            WHEN u.age < 45 THEN '35-44' WHEN u.age < 55 THEN '45-54' ELSE '55+'
        END"""

MARTS: tuple[Mart, ...] = (
    Mart("sales",
         "One row per item of a non-cancelled order, with product and customer attributes. "
         "revenue is 0 for returned items; use sum(revenue) for revenue, "
         "count(DISTINCT order_id) for orders, avg(is_returned::INT) for the return rate.",
         f"""
        SELECT
            oi.id                                              AS item_id,
            oi.order_id,
            oi.user_id                                         AS customer_id,
            CAST(oi.created_at AS DATE)                        AS order_date,
            CAST(date_trunc('week', oi.created_at) AS DATE)    AS order_week,
            CAST(date_trunc('month', oi.created_at) AS DATE)   AS order_month,
            oi.status,
            oi.status = 'Returned'                             AS is_returned,
            CAST(oi.sale_price AS DECIMAL(10, 2))              AS sale_price,
            CAST(CASE WHEN oi.status = 'Returned' THEN 0 ELSE oi.sale_price END
                 AS DECIMAL(10, 2))                            AS revenue,
            p.category, p.brand, p.department,
            u.country, u.city, u.gender, u.traffic_source,
            {AGE_GROUP}                                        AS age_group
        FROM shop.order_items AS oi
        JOIN shop.products AS p ON p.id = oi.product_id
        JOIN shop.users AS u ON u.id = oi.user_id
        WHERE oi.status <> 'Cancelled'"""),
    Mart("customers",
         "One row per customer, with signup date and acquisition channel. No personal data.",
         f"""
        SELECT
            u.id                          AS customer_id,
            CAST(u.created_at AS DATE)    AS signup_date,
            u.country, u.city, u.gender, u.traffic_source,
            {AGE_GROUP}                   AS age_group
        FROM shop.users AS u"""),
    Mart("products",
         "One row per product of the catalog, with its list price. Product cost is not exposed.",
         """
        SELECT
            p.id                                       AS product_id,
            p.name, p.category, p.brand, p.department,
            CAST(p.retail_price AS DECIMAL(10, 2))     AS retail_price
        FROM shop.products AS p"""),
)


def build_marts(db: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Create every mart in the current database from the attached `shop` catalog.

    Returns the row count of each mart. The caller attaches the source database.
    """
    counts = {}
    for mart in MARTS:
        db.execute(f"CREATE OR REPLACE TABLE {mart.name} AS {mart.sql}")
        db.execute(f"COMMENT ON TABLE {mart.name} IS '{mart.description.replace(chr(39), chr(39) * 2)}'")
        counts[mart.name] = db.execute(f"SELECT count(*) FROM {mart.name}").fetchone()[0]
    return counts


def build_analytics_database(source: Path, target: Path) -> dict[str, int]:
    """Rebuild the analytics file from scratch: the source is attached read-only, then detached."""
    target.unlink(missing_ok=True)
    with duckdb.connect(str(target)) as db:
        db.execute(f"ATTACH '{source}' AS shop (READ_ONLY)")
        counts = build_marts(db)
        db.execute("DETACH shop")
    return counts
