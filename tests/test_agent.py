"""SQL agent tests: the loop logic, with a scripted LLM and a small real database (free, fast)."""

import anthropic
import duckdb
import httpx2
import pytest

from app import prompts
from app.agent import NUDGE, TOOLS, SqlAgent, format_observation
from app.config import Settings
from app.llm import LLMRefusal, LLMService, LLMTruncated, LLMUnavailable
from app.sql_tool import QueryResult, SqlTool
from tests.fakes import ScriptedClient, agent_turn, make_interpretation, text_block, tool_call

QUESTION = "Quel est le CA par catégorie ?"
GOOD_SQL = "SELECT category, sum(revenue) AS revenue FROM sales GROUP BY 1 ORDER BY 2 DESC"


@pytest.fixture
def sql_tool(tmp_path):
    path = tmp_path / "analytics.duckdb"
    with duckdb.connect(str(path)) as db:
        db.execute("CREATE TABLE sales AS SELECT range AS item_id, "
                   "CASE WHEN range % 3 = 0 THEN 'Jeans' ELSE 'Shoes' END AS category, "
                   "10.0 AS revenue FROM range(30)")
        db.execute("COMMENT ON TABLE sales IS 'One row per item.'")
    return SqlTool(path, max_rows=50, timeout_s=1.0)


def make_agent(client, sql_tool, **settings) -> SqlAgent:
    llm = LLMService(Settings(anthropic_api_key="sk-test", _env_file=None, **settings),
                     client=client)
    return SqlAgent(llm, sql_tool)


async def run(agent: SqlAgent):
    return await agent.answer(QUESTION, make_interpretation())


def last_tool_results(client: ScriptedClient, request: int) -> list[dict]:
    return client.requests[request]["messages"][-1]["content"]


# --- Happy paths ----------------------------------------------------------------
async def test_submit_on_first_step_returns_rows_run_by_the_server(sql_tool):
    client = ScriptedClient(agent_turn(tool_call("submit_answer", GOOD_SQL)))
    result = await run(make_agent(client, sql_tool))

    assert result.status == "answered"
    assert result.result.columns == ["category", "revenue"]
    assert result.result.rows == [("Shoes", 200.0), ("Jeans", 100.0)]
    assert len(client.requests) == 1


async def test_explore_then_submit(sql_tool):
    client = ScriptedClient(
        agent_turn(tool_call("run_sql", "SELECT DISTINCT category FROM sales", "toolu_a")),
        agent_turn(tool_call("submit_answer", GOOD_SQL, "toolu_b")),
    )
    result = await run(make_agent(client, sql_tool))

    assert result.status == "answered"
    assert [step.tool for step in result.steps] == ["run_sql", "submit_answer"]
    observation = last_tool_results(client, 1)[0]
    assert observation["tool_use_id"] == "toolu_a" and not observation["is_error"]
    assert "2 rows" in observation["content"] and "Jeans" in observation["content"]


# --- Errors are observations, not crashes ---------------------------------------
async def test_sql_error_is_sent_back_and_the_agent_fixes_it(sql_tool):
    client = ScriptedClient(
        agent_turn(tool_call("submit_answer", "SELECT categry FROM sales", "toolu_a")),
        agent_turn(tool_call("submit_answer", GOOD_SQL, "toolu_b")),
    )
    result = await run(make_agent(client, sql_tool))

    assert result.status == "answered"
    error = last_tool_results(client, 1)[0]
    assert error["is_error"] and "categry" in error["content"]
    assert [step.ok for step in result.steps] == [False, True]


async def test_write_query_is_rejected_and_never_returned(sql_tool):
    client = ScriptedClient(
        agent_turn(tool_call("submit_answer", "DELETE FROM sales")),
        agent_turn(tool_call("submit_answer", "SELECT count(*) AS n FROM sales")),
    )
    result = await run(make_agent(client, sql_tool))

    assert "Rejected" in last_tool_results(client, 1)[0]["content"]
    assert result.result.rows == [(30,)]   # the table is intact


async def test_parallel_calls_get_all_results_in_one_message(sql_tool):
    client = ScriptedClient(
        agent_turn(tool_call("run_sql", "SELECT 1 AS a", "toolu_a"),
                   tool_call("run_sql", "SELECT nope", "toolu_b")),
        agent_turn(tool_call("submit_answer", GOOD_SQL, "toolu_c")),
    )
    await run(make_agent(client, sql_tool))

    results = last_tool_results(client, 1)
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
    assert [r["is_error"] for r in results] == [False, True]


async def test_text_without_tool_call_gets_a_nudge(sql_tool):
    client = ScriptedClient(
        agent_turn(text_block("Le CA est de 300 €."), stop_reason="end_turn"),
        agent_turn(tool_call("submit_answer", GOOD_SQL)),
    )
    result = await run(make_agent(client, sql_tool))

    assert result.status == "answered"
    assert client.requests[1]["messages"][-1] == {"role": "user", "content": NUDGE}


# --- Budget and LLM failures ----------------------------------------------------
async def test_step_budget_stops_the_loop(sql_tool):
    bad = [agent_turn(tool_call("submit_answer", "SELECT nope", f"toolu_{i}")) for i in range(3)]
    client = ScriptedClient(*bad)
    result = await run(make_agent(client, sql_tool, agent_max_steps=3))

    assert result.status == "failed" and result.result is None
    assert "3 steps" in result.reason
    assert len(client.requests) == 3


async def test_cost_is_summed_over_every_call(sql_tool):
    client = ScriptedClient(agent_turn(tool_call("run_sql", "SELECT 1")),
                            agent_turn(tool_call("submit_answer", GOOD_SQL)))
    result = await run(make_agent(client, sql_tool))

    # each fake call: 1000 input + 500 output tokens = 0.004 + 0.01 USD
    assert (result.usage.input_tokens, result.usage.output_tokens) == (2000, 1000)
    assert result.usage.cost_usd == pytest.approx(0.028)
    assert result.usage.prompt_version == prompts.SQL_AGENT_VERSION


@pytest.mark.parametrize("stop_reason, error", [("refusal", LLMRefusal),
                                                ("max_tokens", LLMTruncated)])
async def test_unusable_turn_stops_immediately(sql_tool, stop_reason, error):
    client = ScriptedClient(agent_turn(tool_call("submit_answer", GOOD_SQL),
                                       stop_reason=stop_reason))
    with pytest.raises(error):
        await run(make_agent(client, sql_tool))


async def test_network_failure_becomes_llm_unavailable(sql_tool):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = ScriptedClient(error=anthropic.APIConnectionError(request=request))
    with pytest.raises(LLMUnavailable):
        await run(make_agent(client, sql_tool))


# --- What the LLM receives ------------------------------------------------------
async def test_request_carries_tools_schema_plan_and_agent_effort(sql_tool):
    client = ScriptedClient(agent_turn(tool_call("submit_answer", GOOD_SQL)))
    await run(make_agent(client, sql_tool))

    request = client.requests[0]
    assert request["tools"] == TOOLS
    assert "Table sales: One row per item." in request["system"]
    assert request["output_config"] == {"effort": "medium"}
    first = request["messages"][0]["content"]
    assert QUESTION in first and '"metric": "revenue"' in first


def test_observation_shows_a_preview_only():
    result = QueryResult("SELECT ...", ["n"], [(i,) for i in range(50)], truncated=True)
    text = format_observation(result, preview_rows=5)
    assert text.startswith("more than 50 rows (first 5 shown)")
    assert len(text.splitlines()) == 7   # summary + CSV header + 5 rows
