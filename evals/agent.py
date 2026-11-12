"""End-to-end evaluation: does the pipeline return the right NUMBERS for real questions?

Each case is a question a business user could ask, with a reference query written and
checked by a human. The pipeline (retrieve -> interpret -> SQL agent) answers the
question; its rows are compared with the reference rows. The SQL text is never compared:
two different queries can both be right.

Metric: execution accuracy = share of questions whose result matches the reference.
Also reported: tool calls, cost and latency per question.

Real Claude calls (about 0.03 USD per question). Run from data-assistant/:
    uv run python -m evals.agent
Exits with code 1 under the accuracy threshold, so CI can block a bad prompt change.
"""

import asyncio
import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

TODAY = date(2026, 10, 25)      # fixed, so relative periods resolve the same way every run
MIN_ACCURACY = 0.8
MAX_CONCURRENCY = 3             # questions answered in parallel, within API rate limits
TOLERANCE = 0.011               # one cent / 0.01 point: absorbs rounding differences

# (question, reference SQL). References list only the values that must be found:
# extra columns in the agent's answer (labels, counts) are fine.
CASES: list[tuple[str, str]] = [
    ("Quel est le chiffre d'affaires par catégorie au troisième trimestre 2026 ?",
     "SELECT category, sum(revenue) FROM sales "
     "WHERE order_date BETWEEN '2026-07-01' AND '2026-09-30' GROUP BY 1"),
    ("Combien de commandes en septembre 2026 ?",
     "SELECT count(DISTINCT order_id) FROM sales "
     "WHERE order_date BETWEEN '2026-09-01' AND '2026-09-30'"),
    ("Quel est le panier moyen en Allemagne en 2025 ?",
     "SELECT sum(revenue) / count(DISTINCT order_id) FROM sales "
     "WHERE country = 'Germany' AND year(order_date) = 2025"),
    ("Quel est le taux de retour par catégorie en 2025 ?",
     "SELECT category, 100 * avg(is_returned::INT) FROM sales "
     "WHERE year(order_date) = 2025 GROUP BY 1"),
    ("Quelles sont les 5 marques qui ont fait le plus de chiffre d'affaires en 2025 ?",
     "SELECT brand, sum(revenue) FROM sales WHERE year(order_date) = 2025 "
     "GROUP BY 1 ORDER BY 2 DESC LIMIT 5"),
    ("Quel est le chiffre d'affaires mensuel en 2025 ?",
     "SELECT sum(revenue) FROM sales WHERE year(order_date) = 2025 GROUP BY order_month"),
    ("Combien de clients ont acheté en 2025, par pays ?",
     "SELECT country, count(DISTINCT customer_id) FROM sales "
     "WHERE year(order_date) = 2025 GROUP BY 1"),
    ("Combien d'articles vendus par canal d'acquisition au premier semestre 2026 ?",
     "SELECT traffic_source, count(*) FROM sales WHERE NOT is_returned "
     "AND order_date BETWEEN '2026-01-01' AND '2026-06-30' GROUP BY 1"),
    ("Quel est le chiffre d'affaires en France en 2025, comparé à 2024 ?",
     "SELECT sum(revenue) FROM sales WHERE country = 'France' "
     "AND year(order_date) IN (2024, 2025) GROUP BY year(order_date)"),
    ("Quel est le chiffre d'affaires des pulls pour femme en Italie en 2025 ?",
     "SELECT sum(revenue) FROM sales WHERE category = 'Sweaters' AND department = 'Women' "
     "AND country = 'Italy' AND year(order_date) = 2025"),
]


# --- Comparison (pure functions, unit-tested for free) --------------------------
def values_match(expected, actual) -> bool:
    numbers = (int, float, Decimal)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, numbers) and isinstance(actual, numbers):
        return abs(float(expected) - float(actual)) <= TOLERANCE
    return str(expected).casefold() == str(actual).casefold()


def rows_match(reference: list[tuple], answer: list[tuple]) -> bool:
    """Every reference row must be found inside one answer row (values kept together).

    Row order and extra columns are ignored: "top 5" as rows or a comparison as two
    columns are both accepted, as long as the right values are there.
    """
    return all(
        any(all(any(values_match(expected, actual) for actual in row) for expected in wanted)
            for row in answer)
        for wanted in reference
    )


@dataclass(frozen=True)
class CaseReport:
    question: str
    passed: bool
    status: str             # pipeline outcome: answered, failed, needs_clarification...
    tool_calls: int
    cost_usd: float
    latency_ms: int
    sql: str | None


# --- Running the pipeline ---------------------------------------------------------
async def run_case(question: str, reference_sql: str, index, llm, agent, sql_tool,
                   limit: asyncio.Semaphore) -> CaseReport:
    async with limit:
        reference = sql_tool.run(reference_sql)
        assert reference.ok, f"broken reference query: {reference.error}"

        context = [hit.section for hit in index.search(question)]
        interpreted = await llm.interpret_question(question, context=context, today=TODAY)
        if interpreted.status != "ready":
            return CaseReport(question, False, interpreted.status, 0,
                              interpreted.usage.cost_usd, interpreted.usage.latency_ms, None)

        outcome = await agent.answer(question, interpreted.interpretation, context)
        passed = bool(outcome.result) and rows_match(reference.rows, outcome.result.rows)
        return CaseReport(question, passed, outcome.status, len(outcome.steps),
                          interpreted.usage.cost_usd + outcome.usage.cost_usd,
                          interpreted.usage.latency_ms + outcome.usage.latency_ms,
                          outcome.result.sql if outcome.result else None)


async def evaluate() -> list[CaseReport]:
    from app.agent import SqlAgent
    from app.config import get_settings
    from app.knowledge import load_sections
    from app.llm import LLMService
    from app.retrieval import FastEmbedEmbedder, KnowledgeIndex
    from app.sql_tool import SqlTool

    settings = get_settings()
    index = KnowledgeIndex(load_sections(settings.knowledge_dir),
                           FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir))
    llm = LLMService(settings)
    sql_tool = SqlTool(settings.analytics_db_path, settings.sql_max_rows, settings.sql_timeout_s)
    agent = SqlAgent(llm, sql_tool)
    limit = asyncio.Semaphore(MAX_CONCURRENCY)
    # gather = Promise.all: the cases run concurrently and come back in order.
    return await asyncio.gather(*(run_case(question, sql, index, llm, agent, sql_tool, limit)
                                  for question, sql in CASES))


def print_report(reports: list[CaseReport]) -> float:
    for report in reports:
        mark = "OK  " if report.passed else "FAIL"
        print(f"{mark} {report.tool_calls} calls  {report.cost_usd:.3f} USD  "
              f"{report.latency_ms / 1000:4.1f} s  {report.question}")
        if not report.passed:
            print(f"     status: {report.status}\n     sql:    {report.sql}")
    n = len(reports)
    accuracy = sum(r.passed for r in reports) / n
    print(f"\n{n} questions | execution accuracy {accuracy:.0%} | "
          f"avg {sum(r.tool_calls for r in reports) / n:.1f} tool calls | "
          f"avg {sum(r.cost_usd for r in reports) / n:.3f} USD | "
          f"avg {sum(r.latency_ms for r in reports) / n / 1000:.1f} s | "
          f"total {sum(r.cost_usd for r in reports):.2f} USD")
    return accuracy


if __name__ == "__main__":
    sys.exit(0 if print_report(asyncio.run(evaluate())) >= MIN_ACCURACY else 1)
