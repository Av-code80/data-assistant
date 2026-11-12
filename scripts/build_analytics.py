"""Build data/analytics.duckdb (the view layer) from data/shop.duckdb (the source tables).

Run after generate_data.py, from data-assistant/:  uv run python scripts/build_analytics.py
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # make `app` importable

from app.analytics import build_analytics_database                # noqa: E402
from app.config import get_settings                               # noqa: E402

settings = get_settings()
counts = build_analytics_database(settings.source_db_path, settings.analytics_db_path)
for name, count in counts.items():
    print(f"{name:10} {count:>7,} rows")

# Reconciliation: the mart must give the same revenue as the source tables.
with duckdb.connect(str(settings.source_db_path), read_only=True) as source:
    expected = source.execute("""SELECT round(sum(sale_price), 2) FROM order_items
                                 WHERE status NOT IN ('Cancelled', 'Returned')""").fetchone()[0]
with duckdb.connect(str(settings.analytics_db_path), read_only=True) as analytics:
    actual = float(analytics.execute("SELECT sum(revenue) FROM sales").fetchone()[0])
    tables = [row[0] for row in analytics.execute("SHOW TABLES").fetchall()]
status = "OK" if abs(actual - expected) < 0.01 else "MISMATCH"
print(f"revenue check: source {expected:,.2f} / mart {actual:,.2f} -> {status}")
print(f"tables visible in the analytics file: {tables}")
print(f"Database written to {settings.analytics_db_path}")
sys.exit(0 if status == "OK" else 1)
