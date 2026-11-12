"""SQL tool tests: every protection layer, on a small database file (fast, no LLM)."""

import duckdb
import pytest

from app.sql_tool import MAX_SQL_CHARS, SqlTool


@pytest.fixture
def tool(tmp_path):
    path = tmp_path / "analytics.duckdb"
    with duckdb.connect(str(path)) as db:
        db.execute("CREATE TABLE sales AS SELECT range AS item_id, "
                   "CASE WHEN range % 2 = 0 THEN 'Jeans' ELSE 'Shoes' END AS category, "
                   "10.0 AS revenue FROM range(50)")
        db.execute("COMMENT ON TABLE sales IS 'One row per item.'")
    return SqlTool(path, max_rows=10, timeout_s=1.0)


def test_select_returns_columns_and_rows(tool):
    result = tool.run("SELECT category, sum(revenue) AS revenue FROM sales GROUP BY 1 ORDER BY 1")
    assert result.ok
    assert result.columns == ["category", "revenue"]
    assert result.rows == [("Jeans", 250.0), ("Shoes", 250.0)]
    assert not result.truncated


def test_large_results_are_truncated_and_flagged(tool):
    result = tool.run("SELECT item_id FROM sales")
    assert len(result.rows) == 10
    assert result.truncated


@pytest.mark.parametrize("sql, reason", [
    ("DELETE FROM sales", "only SELECT"),
    ("DROP TABLE sales", "only SELECT"),
    ("UPDATE sales SET revenue = 0", "only SELECT"),
    ("SELECT 1; DROP TABLE sales", "exactly one statement"),
    ("ATTACH 'other.duckdb' AS other", "only SELECT"),
    ("PRAGMA show_tables", "only SELECT"),
    ("", "exactly one statement"),
    ("SELECT " + "1, " * MAX_SQL_CHARS + "1", "too long"),
])
def test_dangerous_queries_are_rejected_before_running(tool, sql, reason):
    result = tool.run(sql)
    assert not result.ok
    assert result.error.startswith("Rejected") and reason in result.error


def test_files_cannot_be_read_even_with_a_select(tool):
    result = tool.run("SELECT * FROM read_csv('/etc/hosts')")
    assert not result.ok and "Permission" in result.error


def test_the_connection_itself_is_read_only(tool):
    # Second layer: even if the parser check were bypassed, the write would fail.
    with tool._connect() as db, pytest.raises(duckdb.Error):
        db.execute("DELETE FROM sales")


def test_runaway_queries_are_interrupted(tool):
    result = tool.run("SELECT count(*) FROM range(100000000000)")
    assert not result.ok and result.error.startswith("Timeout")
    assert result.elapsed_ms < 3_000


def test_errors_are_returned_with_hints_for_the_agent(tool):
    result = tool.run("SELECT categry FROM sales")
    assert not result.ok
    assert "categry" in result.error and "category" in result.error   # DuckDB suggests the fix


def test_schema_lists_documentation_columns_and_short_value_lists(tool):
    schema = tool.describe_schema()
    assert "Table sales: One row per item." in schema
    assert "category VARCHAR (values: Jeans, Shoes)" in schema
    assert "item_id BIGINT" in schema
