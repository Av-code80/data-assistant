"""Chart suggestion: which chart shows a result table best, decided by plain rules.

The server sends a chart SPEC (type, x column, y columns), not an image: the frontend
draws it with its own library (Recharts, Chart.js) from the rows it already has.
No LLM here: the rules are free, instant, and give the same answer every time.

Pandas is used for one job: recognising what each column holds (dates, numbers, text),
whatever Python type the database returned (Decimal, date, int...).
"""

import pandas as pd

from .schemas import ChartSpec

MAX_BARS = 30          # more categories than this is unreadable as a bar chart
MAX_SERIES = 3         # numeric columns drawn together
NUMERIC = {"integer", "floating", "decimal", "mixed-integer-float"}
TEMPORAL = {"date", "datetime", "datetime64"}
PERIOD_NAMES = {"year", "quarter", "month", "week", "day"}   # numbers that are really a time axis


def column_kinds(columns: list[str], rows: list[tuple]) -> dict[str, str]:
    """Each column -> 'time', 'number', 'text' or 'other'."""
    frame = pd.DataFrame(rows, columns=columns)
    kinds = {}
    for name in columns:
        inferred = pd.api.types.infer_dtype(frame[name], skipna=True)
        if inferred in TEMPORAL or name.lower() in PERIOD_NAMES:
            kinds[name] = "time"
        elif inferred in NUMERIC:
            # An identifier is a number, but adding or plotting it means nothing.
            kinds[name] = "other" if name.lower().endswith("_id") else "number"
        elif inferred == "string":
            kinds[name] = "text"
        else:
            kinds[name] = "other"
    return kinds


def suggest_chart(columns: list[str], rows: list[tuple]) -> ChartSpec | None:
    """A line chart over time, a bar chart per category, or None when a table is better."""
    if len(rows) < 2:
        return None                       # one row is a single figure, not a chart
    kinds = column_kinds(columns, rows)
    numbers = [name for name in columns if kinds[name] == "number"][:MAX_SERIES]
    if not numbers:
        return None

    time_columns = [name for name in columns if kinds[name] == "time"]
    if time_columns:
        return ChartSpec(type="line", x=time_columns[0], y=numbers)

    text_columns = [name for name in columns if kinds[name] == "text"]
    if text_columns and len(rows) <= MAX_BARS:
        return ChartSpec(type="bar", x=text_columns[0], y=numbers)
    return None
