"""Chart rules, on the raw types DuckDB returns (date, Decimal, int)."""

from datetime import date
from decimal import Decimal

from app.charts import column_kinds, suggest_chart


def test_time_series_becomes_a_line_chart():
    rows = [(date(2025, 1, 1), Decimal("100.00")), (date(2025, 2, 1), Decimal("120.50"))]
    chart = suggest_chart(["order_month", "revenue"], rows)
    assert (chart.type, chart.x, chart.y) == ("line", "order_month", ["revenue"])


def test_categories_become_a_bar_chart():
    rows = [("Jeans", Decimal("100.00"), 12), ("Shoes", Decimal("250.50"), 30)]
    chart = suggest_chart(["category", "revenue", "order_count"], rows)
    assert (chart.type, chart.x, chart.y) == ("bar", "category", ["revenue", "order_count"])


def test_time_wins_over_category():
    rows = [(date(2025, 1, 1), "Jeans", 10.0), (date(2025, 2, 1), "Jeans", 12.0)]
    assert suggest_chart(["order_month", "category", "revenue"], rows).type == "line"


def test_a_year_number_is_a_time_axis_not_a_series():
    chart = suggest_chart(["year", "revenue"], [(2024, 10.0), (2025, 20.0)])
    assert (chart.type, chart.x, chart.y) == ("line", "year", ["revenue"])


def test_single_figure_has_no_chart():
    assert suggest_chart(["order_count"], [(1234,)]) is None


def test_no_numeric_column_has_no_chart():
    assert suggest_chart(["category", "brand"], [("Jeans", "Kova"), ("Shoes", "Terra")]) is None


def test_identifiers_are_never_plotted():
    rows = [("Jeans", 101), ("Shoes", 102)]
    assert suggest_chart(["category", "product_id"], rows) is None


def test_too_many_categories_have_no_chart():
    rows = [(f"city {i}", float(i)) for i in range(31)]
    assert suggest_chart(["city", "revenue"], rows) is None


def test_null_values_do_not_hide_the_column_type():
    rows = [("Jeans", None), ("Shoes", Decimal("5.00"))]
    assert column_kinds(["category", "revenue"], rows) == {"category": "text", "revenue": "number"}


def test_at_most_three_series():
    rows = [("a", 1, 2, 3, 4), ("b", 5, 6, 7, 8)]
    assert suggest_chart(["category", "w", "x", "y", "z"], rows).y == ["w", "x", "y"]
