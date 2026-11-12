"""View layer tests: business rules on a tiny hand-made source database (fast, no files)."""

from decimal import Decimal

import duckdb
import pytest

from app.analytics import build_analytics_database, build_marts

SOURCE_TABLES = """
CREATE TABLE {schema}users (id BIGINT, gender VARCHAR, age BIGINT, country VARCHAR, city VARCHAR,
                            traffic_source VARCHAR, created_at TIMESTAMP);
CREATE TABLE {schema}products (id BIGINT, name VARCHAR, category VARCHAR, brand VARCHAR,
                               department VARCHAR, cost DOUBLE, retail_price DOUBLE);
CREATE TABLE {schema}orders (order_id BIGINT, user_id BIGINT, status VARCHAR, created_at TIMESTAMP);
CREATE TABLE {schema}order_items (id BIGINT, order_id BIGINT, user_id BIGINT, product_id BIGINT,
                                  status VARCHAR, created_at TIMESTAMP, returned_at TIMESTAMP,
                                  sale_price DOUBLE);
INSERT INTO {schema}users VALUES (1, 'F', 30, 'France', 'Paris', 'Search', '2024-01-01'),
                                 (2, 'M', 60, 'Germany', 'Berlin', 'Email', '2024-02-01');
INSERT INTO {schema}products VALUES (10, 'Wool Sweater', 'Sweaters', 'Basalte', 'Women', 20.0, 59.9);
INSERT INTO {schema}orders VALUES (100, 1, 'Complete', '2026-03-04 10:00'),
                                  (101, 2, 'Cancelled', '2026-03-05 10:00');
INSERT INTO {schema}order_items VALUES
    (1, 100, 1, 10, 'Complete', '2026-03-04 10:00', NULL, 59.9),
    (2, 100, 1, 10, 'Returned', '2026-03-04 10:00', '2026-03-20', 59.9),
    (3, 101, 2, 10, 'Cancelled', '2026-03-05 10:00', NULL, 59.9);
"""


@pytest.fixture
def analytics():
    db = duckdb.connect()
    db.execute("ATTACH ':memory:' AS shop")
    db.execute(SOURCE_TABLES.format(schema="shop."))
    build_marts(db)
    return db


def test_cancelled_items_are_excluded(analytics):
    assert analytics.execute("SELECT count(*) FROM sales WHERE status = 'Cancelled'").fetchone()[0] == 0
    assert analytics.execute("SELECT count(*) FROM sales").fetchone()[0] == 2


def test_returned_items_bring_no_revenue_but_count_in_return_rate(analytics):
    revenue, return_rate = analytics.execute(
        "SELECT sum(revenue), avg(is_returned::INT) FROM sales").fetchone()
    assert revenue == Decimal("59.90")
    assert return_rate == 0.5


def test_age_groups_and_iso_weeks(analytics):
    rows = dict(analytics.execute("SELECT customer_id, age_group FROM customers").fetchall())
    assert rows == {1: "25-34", 2: "55+"}
    week = analytics.execute("SELECT DISTINCT order_week FROM sales").fetchone()[0]
    assert week.isoformat() == "2026-03-02"             # Wednesday 4 March -> Monday 2 March


def test_product_cost_is_not_exposed(analytics):
    columns = [row[0] for row in analytics.execute("DESCRIBE products").fetchall()]
    assert "cost" not in columns and "retail_price" in columns


def test_marts_document_themselves(analytics):
    comment = analytics.execute(
        "SELECT comment FROM duckdb_tables() WHERE table_name = 'sales'").fetchone()[0]
    assert "revenue is 0 for returned items" in comment


def test_analytics_file_cannot_reach_the_source_tables(tmp_path):
    source, target = tmp_path / "shop.duckdb", tmp_path / "analytics.duckdb"
    with duckdb.connect(str(source)) as db:
        db.execute(SOURCE_TABLES.format(schema=""))
    assert build_analytics_database(source, target) == {"sales": 2, "customers": 2, "products": 1}

    with duckdb.connect(str(target), read_only=True) as db:
        assert {row[0] for row in db.execute("SHOW TABLES").fetchall()} == {"sales", "customers",
                                                                            "products"}
        with pytest.raises(duckdb.Error):
            db.execute("SELECT * FROM order_items")
