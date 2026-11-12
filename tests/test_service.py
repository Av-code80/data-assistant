"""Service-layer tests: business rules and LLM handling, with a fake client (free)."""

import logging
from datetime import date

import anthropic
import httpx2
import pytest

from app import prompts
from app.config import Settings
from app.llm import (NO_DATA_ANSWER, LLMRefusal, LLMService, LLMTruncated, LLMUnavailable,
                     apply_data_coverage, decide_status, format_context, format_table,
                     validate_time_range)
from app.knowledge import Section
from app.schemas import AnswerRequest, QuestionInterpretation, TimeRange
from tests.fakes import FakeClient, fake_message, make_interpretation

TODAY = date(2026, 10, 25)


def make_service(client, **settings):
    return LLMService(Settings(anthropic_api_key="sk-test", _env_file=None, **settings),
                      client=client)


# --- Pure business rules --------------------------------------------------------
def test_status_ready():
    assert decide_status(make_interpretation()) == "ready"


@pytest.mark.parametrize("overrides", [
    {"metric": "unknown"},
    {"ambiguities": ["CA brut ou net ?"]},
    {"clarifying_question": "Sur quelle période ?"},
])
def test_status_needs_clarification(overrides):
    assert decide_status(make_interpretation(**overrides)) == "needs_clarification"


def test_assumptions_do_not_block():
    interpretation = make_interpretation(assumptions=["« Ventes » lu comme le chiffre d'affaires."])
    assert decide_status(interpretation) == "ready"


def test_out_of_scope_wins_over_everything():
    interpretation = make_interpretation(answerable=False, reason="read-only", metric="unknown")
    assert decide_status(interpretation) == "out_of_scope"


def test_inverted_period_asks_for_clarification():
    interpretation = make_interpretation(
        time_range=TimeRange(start="2026-09-30", end="2026-07-01", comparison="none"))
    checked = validate_time_range(interpretation)
    assert decide_status(checked) == "needs_clarification"
    assert checked.clarifying_question


def test_malformed_date_asks_for_clarification():
    interpretation = make_interpretation(
        time_range=TimeRange(start="Q3 2026", end=None, comparison="none"))
    assert decide_status(validate_time_range(interpretation)) == "needs_clarification"


def test_open_ended_period_is_accepted():
    interpretation = make_interpretation(
        time_range=TimeRange(start="2026-01-01", end=None, comparison="none"))
    assert decide_status(validate_time_range(interpretation)) == "ready"


def test_format_table_escapes_commas():
    assert format_table(["city"], [["Paris, France"]]) == 'city\r\n"Paris, France"\r\n'


def test_format_context_cites_sources():
    text = format_context([Section("g/ca", "Chiffre d'affaires", "glossary.md", "Somme des ventes.")])
    assert text.startswith("<context>")
    assert "### Chiffre d'affaires (source: glossary.md)" in text


def test_no_context_adds_no_empty_tag():
    assert format_context([]) == ""


# --- Data coverage: the database holds 2024-01-01 .. 2026-10-24 ---------------------
FROM, UNTIL = date(2024, 1, 1), date(2026, 10, 24)


def with_period(start, end, comparison="none"):
    return make_interpretation(time_range=TimeRange(start=start, end=end, comparison=comparison))


def test_period_inside_the_data_is_untouched():
    interpretation = with_period("2026-07-01", "2026-09-30")
    assert apply_data_coverage(interpretation, FROM, UNTIL) == (interpretation, [])


def test_period_past_the_last_date_is_clamped_and_flagged():
    checked, warnings = apply_data_coverage(with_period("2026-10-01", "2026-12-31", "previous_year"),
                                            FROM, UNTIL)
    assert checked.time_range.end == "2026-10-24"
    assert len(warnings) == 2                     # incomplete period + fair comparison
    assert decide_status(checked) == "ready"      # still answerable, with a warning


def test_period_before_history_is_clamped():
    checked, warnings = apply_data_coverage(with_period("2023-06-01", "2024-03-31"), FROM, UNTIL)
    assert checked.time_range.start == "2024-01-01"
    assert warnings


@pytest.mark.parametrize("start, end", [("2026-12-01", "2026-12-31"), ("2022-01-01", "2022-12-31")])
def test_period_entirely_outside_the_data_is_out_of_scope(start, end):
    checked, _ = apply_data_coverage(with_period(start, end), FROM, UNTIL)
    assert decide_status(checked) == "out_of_scope"
    assert checked.reason


def test_comparison_before_history_is_flagged():
    _, warnings = apply_data_coverage(with_period("2024-03-01", "2024-03-31", "previous_year"),
                                      FROM, UNTIL)
    assert any("comparaison" in w for w in warnings)


# --- Interpretation through the (fake) LLM ----------------------------------------
async def test_interpret_returns_status_and_cost():
    client = FakeClient(fake_message(parsed_output=make_interpretation()))
    result = await make_service(client).interpret_question("CA par catégorie au T3 ?", today=TODAY)
    assert result.status == "ready"
    assert result.warnings == []
    usage = result.usage
    assert usage.cost_usd == 0.014          # 1000 x $4/M + 500 x $20/M
    assert usage.prompt_version == prompts.INTERPRET_VERSION


async def test_today_is_sent_in_the_message_not_the_system_prompt():
    client = FakeClient(fake_message(parsed_output=make_interpretation()))
    await make_service(client).interpret_question("CA ce mois-ci ?", today=TODAY)
    assert "2026-10-25" in client.received["messages"][0]["content"]
    assert "2026-10-25" not in client.received["system"]       # keeps the prompt cacheable
    assert client.received["output_format"] is QuestionInterpretation


async def test_data_coverage_is_sent_with_the_question():
    client = FakeClient(fake_message(parsed_output=make_interpretation()))
    service = make_service(client, data_available_until=date(2026, 10, 24))
    await service.interpret_question("CA ce trimestre ?", today=TODAY)
    assert "Data available: 2024-01-01 to 2026-10-24" in client.received["messages"][0]["content"]


async def test_context_is_sent_before_the_question():
    client = FakeClient(fake_message(parsed_output=make_interpretation()))
    context = [Section("g/ca", "Chiffre d'affaires", "glossary.md", "Ventes = CA par défaut.")]
    await make_service(client).interpret_question("Comment vont les ventes ?", context=context,
                                                  today=TODAY)
    content = client.received["messages"][0]["content"]
    assert content.index("<context>") < content.index("<question>")
    assert "Ventes = CA par défaut." in content


@pytest.mark.parametrize("stop_reason, expected_error", [("refusal", LLMRefusal),
                                                         ("max_tokens", LLMTruncated)])
async def test_abnormal_stop_reasons(stop_reason, expected_error):
    client = FakeClient(fake_message(stop_reason, make_interpretation()))
    with pytest.raises(expected_error):
        await make_service(client).interpret_question("CA ?", today=TODAY)


async def test_network_failure_becomes_domain_error():
    outage = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    with pytest.raises(LLMUnavailable):
        await make_service(FakeClient(error=outage)).interpret_question("CA ?", today=TODAY)


# --- Answer streaming ----------------------------------------------------------------
async def collect(stream):
    return "".join([chunk async for chunk in stream])


async def test_empty_result_answers_without_calling_llm():
    client = FakeClient()
    request = AnswerRequest(question="CA en Islande ?", columns=["revenue"], rows=[])
    assert await collect(make_service(client).stream_answer(request)) == NO_DATA_ANSWER
    assert client.calls == 0


async def test_answer_streams_chunks():
    request = AnswerRequest(question="CA total ?", columns=["revenue"], rows=[[10]])
    assert await collect(make_service(FakeClient()).stream_answer(request)) == "Le CA est de 10 €."


async def test_large_result_is_truncated_and_flagged():
    client = FakeClient()
    request = AnswerRequest(question="CA par jour ?", columns=["day", "revenue"],
                            rows=[[f"2026-01-{d:02d}", d] for d in range(1, 31)])
    await collect(make_service(client, max_answer_rows=5).stream_answer(request))
    content = client.received["messages"][0]["content"]
    assert "2026-01-05" in content and "2026-01-06" not in content
    assert "first 5 of 30 rows" in content


async def test_client_disconnect_is_logged(caplog):
    caplog.set_level(logging.WARNING, logger="app.llm")
    request = AnswerRequest(question="CA total ?", columns=["revenue"], rows=[[10]])
    stream = make_service(FakeClient()).stream_answer(request)
    await anext(stream)        # the user receives one chunk...
    await stream.aclose()      # ...then closes the page
    assert any(r.getMessage() == "llm_stream_aborted" for r in caplog.records)


def test_rows_must_match_columns():
    with pytest.raises(ValueError, match="row 1 has 1 values, expected 2"):
        AnswerRequest(question="CA ?", columns=["city", "revenue"], rows=[["Paris", 10], ["Lyon"]])
