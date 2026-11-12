"""The eval's comparison logic is code too: a wrong matcher gives a wrong score."""

from decimal import Decimal

from app.sql_tool import SqlTool
from app.config import PROJECT_DIR
from evals.agent import CASES, rows_match

REFERENCE = [("Jeans", Decimal("100.00")), ("Shoes", Decimal("250.50"))]


def test_order_extra_columns_and_rounding_are_ignored():
    answer = [("shoes", 250.5, 12), ("Jeans", 100.004, 7)]
    assert rows_match(REFERENCE, answer)


def test_a_wrong_number_fails():
    assert not rows_match(REFERENCE, [("Jeans", 100.0), ("Shoes", 251.0)])


def test_values_swapped_between_rows_fail():
    assert not rows_match(REFERENCE, [("Jeans", 250.5), ("Shoes", 100.0)])


def test_a_missing_row_fails_and_an_empty_answer_fails():
    assert not rows_match(REFERENCE, [("Jeans", 100.0)])
    assert not rows_match(REFERENCE, [])


def test_comparison_as_one_row_or_two_rows_both_pass():
    reference = [(Decimal("10.00"),), (Decimal("20.00"),)]
    assert rows_match(reference, [(2025, 20.0), (2024, 10.0)])
    assert rows_match(reference, [(20.0, 10.0, 100.0)])


def test_every_reference_query_runs_and_returns_rows():
    tool = SqlTool(PROJECT_DIR / "data" / "analytics.duckdb")
    for question, sql in CASES:
        result = tool.run(sql)
        assert result.ok and result.rows, question
