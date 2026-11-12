"""SQL agent: a ReAct loop written by hand (reason -> act with a tool -> observe -> repeat).

Each step, the LLM either runs an exploratory query (run_sql) or submits its final one
(submit_answer). The server stays in control:
- it executes every query itself, through the SQL tool (read-only, SELECT only, timeout);
- it re-runs the submitted query, so the rows shown to users always come from the
  database, never from the LLM's text;
- it caps the number of steps, which bounds cost and latency;
- it logs every step and sums the cost of the whole loop.
"""

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from . import prompts
from .knowledge import Section
from .llm import TRANSIENT_ERRORS, LLMService, LLMUnavailable, format_context, format_table
from .schemas import QuestionInterpretation, Usage
from .sql_tool import QueryResult, SqlTool

logger = logging.getLogger("app.agent")

SQL_INPUT = {
    "type": "object",
    "properties": {"sql": {"type": "string", "description": "One DuckDB SELECT query."}},
    "required": ["sql"],
    "additionalProperties": False,
}

# strict: the API guarantees the input matches the schema (always a "sql" string).
TOOLS = [
    {"name": "run_sql", "strict": True, "input_schema": SQL_INPUT,
     "description": "Run a read-only SELECT on the analytics database. Returns the columns, "
                    "the row count and the first rows, or the error to fix."},
    {"name": "submit_answer", "strict": True, "input_schema": SQL_INPUT,
     "description": "Submit the final SELECT that answers the question. The server runs it "
                    "and shows the result to the user. Call it exactly once, at the end."},
]

NUDGE = "Call submit_answer with your final query."


@dataclass(frozen=True)
class AgentStep:
    """One tool call: the trace shown to developers (logs) and, later, to users."""
    tool: str
    sql: str
    ok: bool
    rows: int
    error: str | None


@dataclass(frozen=True)
class AgentResult:
    status: Literal["answered", "failed"]
    result: QueryResult | None   # the submitted query, run by the server
    steps: list[AgentStep]
    usage: Usage                 # summed over every LLM call of the loop
    reason: str | None = None    # why the agent failed


def sum_usage(usages: Sequence[Usage], latency_ms: int) -> Usage:
    """One cost line for the whole loop: what the question really cost."""
    return Usage(
        model=",".join(sorted({u.model for u in usages})) or "none",
        prompt_version=prompts.SQL_AGENT_VERSION,
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cost_usd=round(sum(u.cost_usd for u in usages), 6),
        latency_ms=latency_ms,
    )


def format_observation(result: QueryResult, preview_rows: int) -> str:
    """What the agent sees after a query: small, so a big result cannot flood its context."""
    if not result.ok:
        return result.error
    shown = result.rows[:preview_rows]
    count = f"more than {len(result.rows)}" if result.truncated else str(len(result.rows))
    note = f" (first {len(shown)} shown)" if len(shown) < len(result.rows) else ""
    return f"{count} rows{note}, {result.elapsed_ms} ms\n{format_table(result.columns, shown)}"


def format_plan(interpretation: QuestionInterpretation) -> str:
    plan = interpretation.model_dump(include={"metric", "dimensions", "filters", "time_range",
                                              "assumptions"})
    return json.dumps(plan, ensure_ascii=False)


class SqlAgent:
    def __init__(self, llm: LLMService, sql_tool: SqlTool):
        self.llm = llm
        self.sql_tool = sql_tool
        self.settings = llm.settings
        # The schema is read once: it only changes when the view layer is rebuilt.
        self.system = (f"{prompts.SQL_AGENT_SYSTEM}\n\n"
                       f"{prompts.sql_agent_schema(sql_tool.describe_schema())}")

    async def answer(self, question: str, interpretation: QuestionInterpretation,
                     context: Sequence[Section] = ()) -> AgentResult:
        """Loop until the agent submits a query that runs, or the step budget is spent."""
        start = time.perf_counter()
        messages: list[dict] = [{"role": "user", "content":
                                 f"<question>{question}</question>\n\n"
                                 f"<plan>{format_plan(interpretation)}</plan>\n\n"
                                 f"{format_context(list(context))}"}]
        steps: list[AgentStep] = []
        usages: list[Usage] = []

        for _ in range(self.settings.agent_max_steps):
            response = await self._call_llm(messages, usages)
            # The whole content goes back, thinking blocks included: the API requires it.
            messages.append({"role": "assistant", "content": response.content})

            tool_calls = [block for block in response.content if block.type == "tool_use"]
            if not tool_calls:
                messages.append({"role": "user", "content": NUDGE})
                continue

            # All results go back in ONE user message, in the order of the calls.
            tool_results, submitted = [], None
            for call in tool_calls:
                result, observation = await self._execute(call)
                steps.append(AgentStep(call.name, result.sql, result.ok, len(result.rows),
                                       result.error))
                if call.name == "submit_answer" and result.ok:
                    submitted = result
                tool_results.append({"type": "tool_result", "tool_use_id": call.id,
                                     "content": observation, "is_error": not result.ok})

            if submitted:
                return self._finish("answered", submitted, steps, usages, start)
            messages.append({"role": "user", "content": tool_results})

        return self._finish("failed", None, steps, usages, start,
                            reason=f"No valid query after {self.settings.agent_max_steps} steps.")

    async def _call_llm(self, messages: list[dict], usages: list[Usage]):
        call_start = time.perf_counter()
        try:
            response = await self.llm.client.beta.messages.create(
                **self.llm.request_params(effort=self.settings.agent_effort),
                system=self.system,
                tools=TOOLS,
                messages=messages,
            )
        except TRANSIENT_ERRORS as error:
            logger.error("llm_unavailable", extra={"data": {"error": type(error).__name__}})
            raise LLMUnavailable from error
        usages.append(self.llm.record_usage(response, call_start, prompts.SQL_AGENT_VERSION))
        self.llm.check_stop_reason(response.stop_reason)   # refusal or truncation: stop now
        return response

    async def _execute(self, call) -> tuple[QueryResult, str]:
        """Run one tool call. Errors become observations the agent can fix, never exceptions."""
        sql = call.input.get("sql") if isinstance(call.input, dict) else None
        if call.name not in ("run_sql", "submit_answer"):
            result = QueryResult(sql or "", error=f"Unknown tool {call.name!r}.")
        elif not isinstance(sql, str):
            result = QueryResult("", error="Missing 'sql' string.")
        else:
            # DuckDB is synchronous: run it in a thread so one slow query does not
            # freeze the event loop (and with it every other user's request).
            result = await asyncio.to_thread(self.sql_tool.run, sql)
        logger.info("agent_step", extra={"data": {"tool": call.name, "ok": result.ok,
                                                  "rows": len(result.rows)}})
        return result, format_observation(result, self.settings.agent_preview_rows)

    def _finish(self, status, result, steps, usages, start, reason=None) -> AgentResult:
        usage = sum_usage(usages, int((time.perf_counter() - start) * 1000))
        log = logger.info if status == "answered" else logger.warning
        log("agent_done", extra={"data": {"status": status, "steps": len(steps),
                                          "llm_calls": len(usages), "reason": reason,
                                          **usage.model_dump()}})
        return AgentResult(status, result, steps, usage, reason)
