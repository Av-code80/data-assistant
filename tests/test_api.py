"""HTTP route tests: the service is replaced by a fake via dependency_overrides."""

from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.knowledge import Section
from app.llm import InterpretResult, LLMRefusal, LLMUnavailable
from app.agent import AgentResult, AgentStep
from app.main import app, get_knowledge_index, get_llm_service, get_sql_agent
from app.retrieval import SearchHit
from app.schemas import Usage
from app.sql_tool import QueryResult
from tests.fakes import make_interpretation

USAGE = Usage(model="claude-opus-5-5", prompt_version="interpret-question/v1",
              input_tokens=1000, output_tokens=500, cost_usd=0.014, latency_ms=900)


GLOSSARY = Section("glossary/revenue", "Chiffre d'affaires", "glossary.md", "Somme des ventes.")


class FakeIndex:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k=3, min_vector_score=0.0):
        self.queries.append(query)
        return [SearchHit(GLOSSARY, score=0.03, vector=0.71)]


class FakeService:
    def __init__(self, error=None, status="ready"):
        self.error = error
        self.status = status

    async def interpret_question(self, question, context=None):
        self.context = context
        if self.error:
            raise self.error
        return InterpretResult(make_interpretation(), self.status, ["Période incomplète."], USAGE)

    async def stream_answer(self, request):
        for chunk in ["Le CA ", "est de 10 €."]:
            yield chunk


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_service(service, index=None):
    app.dependency_overrides[get_llm_service] = lambda: service
    app.dependency_overrides[get_knowledge_index] = lambda: index or FakeIndex()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_interpret_success(client):
    use_service(FakeService())
    response = client.post("/questions/interpret", json={"question": "CA par catégorie ?"})
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["interpretation"]["metric"] == "revenue"
    assert response.json()["warnings"] == ["Période incomplète."]
    assert response.json()["sources"] == [{"id": "glossary/revenue", "title": "Chiffre d'affaires",
                                           "file": "glossary.md", "score": 0.71}]


def test_retrieved_sections_reach_the_llm(client):
    service, index = FakeService(), FakeIndex()
    use_service(service, index)
    client.post("/questions/interpret", json={"question": "CA par catégorie ?"})
    assert index.queries == ["CA par catégorie ?"]
    assert service.context == [GLOSSARY]


def test_knowledge_search(client):
    use_service(FakeService())
    response = client.get("/knowledge/search", params={"q": "chiffre d'affaires"})
    assert response.json()[0]["id"] == "glossary/revenue"


def test_too_long_question_rejected_without_calling_llm(client):
    use_service(FakeService(error=AssertionError("the LLM should not be called")))
    response = client.post("/questions/interpret", json={"question": "x" * 1_001})
    assert response.status_code == 422


@pytest.mark.parametrize("error, status_code", [(LLMUnavailable(), 503), (LLMRefusal(), 422)])
def test_domain_errors_map_to_http(client, error, status_code):
    use_service(FakeService(error=error))
    response = client.post("/questions/interpret", json={"question": "CA par catégorie ?"})
    assert response.status_code == status_code


def test_answer_streaming(client):
    use_service(FakeService())
    response = client.post("/answers/stream",
                           json={"question": "CA total ?", "columns": ["revenue"], "rows": [[10]]})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"text": "Le CA "}' in response.text
    assert "event: done" in response.text


def test_answer_rejects_ragged_rows(client):
    use_service(FakeService())
    response = client.post("/answers/stream", json={
        "question": "CA ?", "columns": ["city", "revenue"], "rows": [["Paris", 10], ["Lyon"]]})
    assert response.status_code == 422


# --- POST /questions/ask ---------------------------------------------------------
SQL = "SELECT order_month, sum(revenue) AS revenue FROM sales GROUP BY 1"
AGENT_USAGE = Usage(model="claude-opus-5-5", prompt_version="sql-agent/v1",
                    input_tokens=3000, output_tokens=400, cost_usd=0.02, latency_ms=4000)


class FakeAgent:
    def __init__(self, status="answered"):
        self.status = status
        self.calls = 0

    async def answer(self, question, interpretation, context=()):
        self.calls += 1
        self.context = context
        steps = [AgentStep("submit_answer", SQL, self.status == "answered", 1, None)]
        if self.status == "failed":
            return AgentResult("failed", None, steps, AGENT_USAGE, "No valid query after 6 steps.")
        # Raw database types, as DuckDB returns them.
        result = QueryResult(SQL, ["order_month", "revenue"],
                             [(date(2026, 9, 1), Decimal("1234.50"))])
        return AgentResult("answered", result, steps, AGENT_USAGE)


def use_agent(agent, service=None):
    use_service(service or FakeService())
    app.dependency_overrides[get_sql_agent] = lambda: agent


def ask(client):
    return client.post("/questions/ask", json={"question": "CA par mois ?"})


def test_ask_returns_rows_sql_trace_and_total_cost(client):
    agent = FakeAgent()
    use_agent(agent)
    body = ask(client).json()

    assert body["status"] == "answered"
    assert body["sql"] == SQL
    assert body["columns"] == ["order_month", "revenue"]
    assert body["rows"] == [["2026-09-01", 1234.5]]          # date and Decimal became JSON
    assert body["steps"][0]["tool"] == "submit_answer"
    assert body["chart"] is None                             # one row: a figure, not a chart
    assert [u["prompt_version"] for u in body["usage"]] == ["interpret-question/v1", "sql-agent/v1"]
    assert body["total_cost_usd"] == pytest.approx(0.034)
    assert agent.context == [GLOSSARY]                       # the agent gets the retrieved knowledge


@pytest.mark.parametrize("status", ["needs_clarification", "out_of_scope"])
def test_ask_does_not_run_the_agent_unless_ready(client, status):
    agent = FakeAgent()
    use_agent(agent, FakeService(status=status))
    body = ask(client).json()

    assert body["status"] == status
    assert agent.calls == 0
    assert body["rows"] == [] and body["sql"] is None
    assert body["total_cost_usd"] == pytest.approx(0.014)    # interpretation only


def test_ask_reports_an_agent_failure_without_rows(client):
    use_agent(FakeAgent(status="failed"))
    response = ask(client)

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["reason"] == "No valid query after 6 steps."
    assert response.json()["rows"] == []


def test_ask_translates_llm_errors(client):
    use_agent(FakeAgent(), FakeService(error=LLMUnavailable()))
    assert ask(client).status_code == 503
