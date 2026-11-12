"""Service layer: the only code that talks to the LLM.

HTTP routes never import the Anthropic SDK. Switching providers means changing
this file only.
"""

import csv
import io
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date
from typing import Literal

import anthropic

from . import prompts
from .config import Settings
from .knowledge import Section
from .schemas import AnswerRequest, QuestionInterpretation, Usage

logger = logging.getLogger("app.llm")

# Server-side fallback: if the model declines, the API retries on another model.
FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

TRANSIENT_ERRORS = (anthropic.APIConnectionError, anthropic.RateLimitError,
                    anthropic.InternalServerError)

NO_DATA_ANSWER = "Aucune donnée ne correspond à cette question sur la période demandée."

Status = Literal["ready", "needs_clarification", "out_of_scope"]


# --- Domain errors: routes translate them into HTTP status codes -------------
class LLMUnavailable(Exception):
    """Network failure, overload or outage, after the SDK's own retries. -> 503"""


class LLMRefusal(Exception):
    """The model and its fallback declined the request. -> 422"""


class LLMTruncated(Exception):
    """max_tokens was reached: the answer is incomplete and unusable. -> 502"""


# --- Pure business rules (no LLM, easy to test) --------------------------------
def validate_time_range(interpretation: QuestionInterpretation) -> QuestionInterpretation:
    """Reject periods the LLM got wrong instead of trusting them."""
    period = interpretation.time_range
    problem = None
    try:
        start = date.fromisoformat(period.start) if period.start else None
        end = date.fromisoformat(period.end) if period.end else None
        if start and end and start > end:
            problem = "La date de début est après la date de fin."
    except ValueError:
        problem = "La période n'a pas pu être interprétée."

    if problem is None:
        return interpretation
    return interpretation.model_copy(update={
        "ambiguities": [*interpretation.ambiguities, problem],
        "clarifying_question": interpretation.clarifying_question
        or "Pouvez-vous préciser la période (date de début et de fin) ?",
    })


def one_year_before(day: date) -> date:
    try:
        return day.replace(year=day.year - 1)
    except ValueError:                      # 29 February
        return day.replace(year=day.year - 1, day=28)


def apply_data_coverage(interpretation: QuestionInterpretation, available_from: date,
                        available_until: date) -> tuple[QuestionInterpretation, list[str]]:
    """Fit the period to the dates the database actually covers.

    A period partly outside the data is clamped and flagged; a period entirely outside
    cannot be answered. Without this, "ce trimestre" on 24 October would compare three
    weeks of this year with a full quarter of last year: a wrong number shown as true.
    """
    period = interpretation.time_range
    try:
        start = date.fromisoformat(period.start) if period.start else None
        end = date.fromisoformat(period.end) if period.end else None
    except ValueError:
        return interpretation, []           # already reported by validate_time_range

    def out_of_scope(reason: str):
        return interpretation.model_copy(update={"answerable": False, "reason": reason}), []

    if start and start > available_until:
        return out_of_scope(f"Aucune donnée après le {available_until:%d/%m/%Y}.")
    if end and end < available_from:
        return out_of_scope(f"L'historique commence le {available_from:%d/%m/%Y}.")

    warnings = []
    if start and start < available_from:
        start = available_from
        warnings.append(f"L'historique commence le {available_from:%d/%m/%Y} : "
                        "la période a été réduite.")
    if end and end > available_until:
        end = available_until
        warnings.append(f"Période incomplète : les données s'arrêtent au "
                        f"{available_until:%d/%m/%Y}.")
        if period.comparison != "none":
            warnings.append("La comparaison portera sur une durée identique, pour rester juste.")
    if period.comparison == "previous_year" and start and one_year_before(start) < available_from:
        warnings.append("La période de comparaison est antérieure au début de l'historique.")

    if not warnings:
        return interpretation, []
    new_period = period.model_copy(update={"start": start and start.isoformat(),
                                           "end": end and end.isoformat()})
    return interpretation.model_copy(update={"time_range": new_period}), warnings


def decide_status(interpretation: QuestionInterpretation) -> Status:
    """The server, not the LLM, decides whether the next step may run."""
    if not interpretation.answerable:
        return "out_of_scope"
    if (interpretation.metric == "unknown" or interpretation.ambiguities
            or interpretation.clarifying_question):
        return "needs_clarification"
    return "ready"


def format_context(sections: list[Section]) -> str:
    """Retrieved sections as tagged text; empty string when nothing relevant was found."""
    if not sections:
        return ""
    body = "\n\n".join(f"### {s.title} (source: {s.source})\n{s.text}" for s in sections)
    return f"<context>\n{body}\n</context>\n\n"


def format_table(columns: list[str], rows: list[list]) -> str:
    """CSV, so commas or line breaks inside values cannot break the table."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue()


@dataclass(frozen=True)
class InterpretResult:
    interpretation: QuestionInterpretation
    status: Status
    warnings: list[str]
    usage: Usage


class LLMService:
    def __init__(self, settings: Settings, client: anthropic.AsyncAnthropic | None = None):
        self.settings = settings
        # The client is injectable so tests can pass a fake one (no paid calls).
        self.client = client or anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    def request_params(self, effort: str | None = None) -> dict:
        return {
            "model": self.settings.llm_model,
            "max_tokens": self.settings.llm_max_tokens,
            "output_config": {"effort": effort or self.settings.llm_effort},
            **FALLBACK,
        }

    # -----------------------------------------------------------------------------
    async def interpret_question(self, question: str, context: list[Section] | None = None,
                                 today: date | None = None) -> InterpretResult:
        """Business question in French + retrieved knowledge -> structured query plan + decision."""
        today = today or date.today()
        start = time.perf_counter()
        try:
            response = await self.client.beta.messages.parse(
                **self.request_params(),
                system=prompts.INTERPRET_SYSTEM,
                messages=[{"role": "user", "content":
                           f"Today's date: {today.isoformat()}\n"
                           f"Data available: {self.settings.data_available_from.isoformat()} "
                           f"to {self.settings.data_available_until.isoformat()}\n\n"
                           f"{format_context(context or [])}<question>{question}</question>"}],
                output_format=QuestionInterpretation,
            )
        except TRANSIENT_ERRORS as error:
            logger.error("llm_unavailable", extra={"data": {"error": type(error).__name__}})
            raise LLMUnavailable from error

        usage = self.record_usage(response, start, prompts.INTERPRET_VERSION)
        self.check_stop_reason(response.stop_reason)
        interpretation = validate_time_range(response.parsed_output)
        interpretation, warnings = apply_data_coverage(interpretation,
                                                       self.settings.data_available_from,
                                                       self.settings.data_available_until)
        return InterpretResult(interpretation, decide_status(interpretation), warnings, usage)

    # -----------------------------------------------------------------------------
    async def stream_answer(self, request: AnswerRequest) -> AsyncIterator[str]:
        """Explain a result table to a business user, chunk by chunk."""
        if not request.rows:
            # No rows: answer without the LLM. Free, instant, and nothing to hallucinate.
            yield NO_DATA_ANSWER
            return

        rows = request.rows[: self.settings.max_answer_rows]
        sample_note = (f"\nOnly the first {len(rows)} of {len(request.rows)} rows are shown."
                       if len(rows) < len(request.rows) else "")
        content = (f"<question>{request.question}</question>\n\n"
                   f"<data>\n{format_table(request.columns, rows)}</data>{sample_note}")

        start = time.perf_counter()
        streamed_chars = 0
        finished = False
        try:
            async with self.client.beta.messages.stream(
                **self.request_params(),
                system=prompts.ANSWER_SYSTEM,
                messages=[{"role": "user", "content": content}],
            ) as stream:
                async for chunk in stream.text_stream:
                    streamed_chars += len(chunk)
                    yield chunk
                final_message = await stream.get_final_message()
            finished = True
        except TRANSIENT_ERRORS as error:
            logger.error("llm_unavailable", extra={"data": {"error": type(error).__name__}})
            raise LLMUnavailable from error
        finally:
            if not finished:
                # Client closed the page, or an error occurred mid-stream: the tokens
                # already generated are still billed, so the event must be logged.
                logger.warning("llm_stream_aborted", extra={"data": {
                    "prompt_version": prompts.ANSWER_VERSION,
                    "streamed_chars": streamed_chars,
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                }})

        self.record_usage(final_message, start, prompts.ANSWER_VERSION)
        self.check_stop_reason(final_message.stop_reason)

    # -----------------------------------------------------------------------------
    def check_stop_reason(self, stop_reason: str | None) -> None:
        # Always check why the model stopped before using its answer.
        if stop_reason == "refusal":
            raise LLMRefusal
        if stop_reason == "max_tokens":
            raise LLMTruncated

    def record_usage(self, response, start: float, prompt_version: str) -> Usage:
        """Log one JSON line per call: cost, latency, model, prompt version."""
        tokens = response.usage
        usage = Usage(
            model=response.model,  # differs from the requested model if the fallback answered
            prompt_version=prompt_version,
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            cost_usd=round(
                tokens.input_tokens / 1e6 * self.settings.input_price_per_million
                + tokens.output_tokens / 1e6 * self.settings.output_price_per_million,
                6,
            ),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
        logger.info("llm_call", extra={"data": {
            **usage.model_dump(),
            "stop_reason": response.stop_reason,
            "request_id": getattr(response, "_request_id", None),  # quote it to Anthropic support
        }})
        return usage
