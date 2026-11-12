"""The SQL tool: runs the agent's SQL against the view layer, safely.

Defense in depth: each layer alone stops most problems; together they make a write,
a file read or a runaway query impossible, whatever the LLM writes.
1. parse:   exactly one statement, and it must be a SELECT (DuckDB's own parser, no regex);
2. connect: read-only file; external access (files, network, ATTACH) disabled and locked;
3. execute: interrupted after a timeout; at most max_rows rows fetched.

Errors are returned, never raised: the agent reads them and fixes its query.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

logger = logging.getLogger("app.sql")

MAX_SQL_CHARS = 5_000
MAX_LISTED_VALUES = 12   # text columns with at most this many values get them listed in the schema


@dataclass(frozen=True)
class QueryResult:
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False      # more rows existed than max_rows
    elapsed_ms: int = 0
    error: str | None = None     # set when the query was rejected or failed

    @property
    def ok(self) -> bool:
        return self.error is None


class SqlTool:
    def __init__(self, db_path: Path, max_rows: int = 200, timeout_s: float = 5.0):
        self.db_path = db_path
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    def _connect(self) -> duckdb.DuckDBPyConnection:
        # One connection per query: no state shared between concurrent requests.
        return duckdb.connect(str(self.db_path), read_only=True, config={
            "enable_external_access": False,   # no read_csv('/etc/...'), no ATTACH, no COPY
            "lock_configuration": True,        # and a query cannot SET it back
        })

    @staticmethod
    def check(sql: str) -> str | None:
        """Why the query must not run, or None if it may."""
        if len(sql) > MAX_SQL_CHARS:
            return f"query too long ({len(sql)} characters, max {MAX_SQL_CHARS})"
        try:
            statements = duckdb.extract_statements(sql)
        except duckdb.Error as error:
            return f"syntax error: {error}"
        if len(statements) != 1:
            return f"exactly one statement is allowed, got {len(statements)}"
        if statements[0].type != duckdb.StatementType.SELECT:
            return f"only SELECT queries are allowed, got {statements[0].type.name}: the data is read-only"
        return None

    def run(self, sql: str) -> QueryResult:
        sql = sql.strip()
        start = time.perf_counter()
        problem = self.check(sql)
        if problem:
            return self._finish(sql, start, error=f"Rejected: {problem}.")

        db = self._connect()
        timer = threading.Timer(self.timeout_s, db.interrupt)   # DuckDB has no statement timeout
        timer.start()
        try:
            cursor = db.execute(sql)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchmany(self.max_rows + 1)              # one extra row reveals truncation
        except duckdb.InterruptException:
            return self._finish(sql, start, error=f"Timeout: the query ran longer than {self.timeout_s} s. "
                                                  "Filter or aggregate more.")
        except duckdb.Error as error:
            return self._finish(sql, start, error=f"{type(error).__name__}: {error}")
        finally:
            timer.cancel()
            db.close()
        return self._finish(sql, start, columns, rows[: self.max_rows], len(rows) > self.max_rows)

    def _finish(self, sql: str, start: float, columns=None, rows=None, truncated=False,
                error: str | None = None) -> QueryResult:
        result = QueryResult(sql, columns or [], rows or [], truncated,
                             int((time.perf_counter() - start) * 1000), error)
        log = logger.warning if error else logger.info
        log("sql_query", extra={"data": {"sql": sql, "ok": result.ok, "rows": len(result.rows),
                                         "truncated": truncated, "elapsed_ms": result.elapsed_ms,
                                         "error": error}})
        return result

    def describe_schema(self) -> str:
        """Tables, their documentation, columns and the values of short text columns, for the LLM."""
        lines = []
        with self._connect() as db:
            tables = db.execute("SELECT table_name, comment FROM duckdb_tables() "
                                "ORDER BY table_name").fetchall()
            for table, comment in tables:
                lines.append(f"Table {table}: {comment}")
                columns = db.execute("SELECT column_name, data_type FROM duckdb_columns() "
                                     "WHERE table_name = ? ORDER BY column_index", [table]).fetchall()
                for column, data_type in columns:
                    line = f"  - {column} {data_type}"
                    if data_type == "VARCHAR":
                        values = db.execute(f'SELECT DISTINCT "{column}" FROM "{table}" '
                                            f"ORDER BY 1 LIMIT {MAX_LISTED_VALUES + 1}").fetchall()
                        if len(values) <= MAX_LISTED_VALUES:
                            line += f" (values: {', '.join(str(v[0]) for v in values)})"
                    lines.append(line)
        return "\n".join(lines)
