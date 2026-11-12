"""Pydantic contracts: what the frontend sends, what the API returns,
and the shape the LLM must produce (structured output)."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .catalog import DIMENSION_NAMES, METRIC_NAMES

Metric = Literal[*METRIC_NAMES, "unknown"]
Dimension = Literal[*DIMENSION_NAMES]


# --- Shape the LLM must produce ------------------------------------------------
class Filter(BaseModel):
    dimension: Dimension
    values: list[str]


class TimeRange(BaseModel):
    start: str | None = Field(description="ISO date YYYY-MM-DD, or null")
    end: str | None = Field(description="ISO date YYYY-MM-DD, inclusive, or null")
    comparison: Literal["none", "previous_period", "previous_year"]


class QuestionInterpretation(BaseModel):
    answerable: bool
    reason: str | None = Field(description="Why the question cannot be answered, if answerable is false")
    metric: Metric
    dimensions: list[Dimension]
    filters: list[Filter]
    time_range: TimeRange
    assumptions: list[str] = Field(
        description="Interpretations you applied, shown to the user; they do not block the answer")
    ambiguities: list[str] = Field(
        description="Points you cannot resolve alone; if not empty, clarifying_question is required")
    clarifying_question: str | None


# --- HTTP contracts ------------------------------------------------------------
class Usage(BaseModel):
    """Returned with every answer: the basis of cost tracking (LLMOps)."""
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


class InterpretRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1_000)


class Source(BaseModel):
    """A knowledge-base section given to the LLM: shown to users as a citation."""
    id: str
    title: str
    file: str
    score: float


class InterpretResponse(BaseModel):
    status: Literal["ready", "needs_clarification", "out_of_scope"]
    interpretation: QuestionInterpretation
    warnings: list[str]      # shown to the user next to the answer (e.g. incomplete period)
    sources: list[Source]
    usage: Usage


Cell = str | int | float | bool | None


class AnswerRequest(BaseModel):
    """A question and the table that answers it (produced by the SQL agent)."""
    question: str = Field(min_length=3, max_length=1_000)
    columns: list[str] = Field(min_length=1, max_length=50)
    rows: list[list[Cell]] = Field(max_length=10_000)

    @model_validator(mode="after")
    def rows_match_columns(self):
        width = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"row {index} has {len(row)} values, expected {width}")
        return self


class AgentStepOut(BaseModel):
    """One tool call of the SQL agent: the trace that makes the answer auditable."""
    tool: str
    sql: str
    ok: bool
    rows: int
    error: str | None


class ChartSpec(BaseModel):
    """How the frontend should draw the rows: column names refer to `columns`."""
    type: Literal["line", "bar"]
    x: str                 # the axis: a date or a category
    y: list[str]           # one series per numeric column


class AskResponse(BaseModel):
    """The whole pipeline in one call: interpretation, then the SQL agent's result."""
    status: Literal["answered", "needs_clarification", "out_of_scope", "failed"]
    interpretation: QuestionInterpretation
    warnings: list[str]
    sources: list[Source]
    sql: str | None = None              # the query that produced the rows, shown to the user
    columns: list[str] = []
    rows: list[list[Cell]] = []
    truncated: bool = False             # more rows existed than the server returns
    chart: ChartSpec | None = None      # null when a table reads better than a chart
    steps: list[AgentStepOut] = []
    reason: str | None = None           # why the agent failed
    usage: list[Usage]                  # one entry per stage: interpretation, then agent
    total_cost_usd: float
