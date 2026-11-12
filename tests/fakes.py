"""Fake Anthropic client for tests: no network, no cost, and it records what it receives."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

from app.schemas import QuestionInterpretation, TimeRange


def make_interpretation(**overrides) -> QuestionInterpretation:
    """A valid 'ready' interpretation; override any field to build edge cases."""
    fields = dict(
        answerable=True, reason=None, metric="revenue", dimensions=["product_category"],
        filters=[], assumptions=[], ambiguities=[], clarifying_question=None,
        time_range=TimeRange(start="2026-07-01", end="2026-09-30", comparison="none"),
    )
    fields.update(overrides)
    return QuestionInterpretation(**fields)


def fake_message(stop_reason="end_turn", parsed_output=None, content=()):
    return SimpleNamespace(parsed_output=parsed_output, stop_reason=stop_reason, content=list(content),
                           model="claude-opus-5-5", _request_id="req_test",
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=500))


class FakeClient:
    """Mimics client.beta.messages.parse and client.beta.messages.stream."""

    def __init__(self, message=None, error=None, chunks=("Le CA ", "est de 10 €.")):
        self.received = None
        self.calls = 0

        async def parse(**params):
            self.calls += 1
            self.received = params
            if error:
                raise error
            return message

        @asynccontextmanager
        async def stream(**params):
            self.calls += 1
            self.received = params
            if error:
                raise error

            async def text_stream():
                for chunk in chunks:
                    yield chunk

            async def get_final_message():
                return message or fake_message()

            yield SimpleNamespace(text_stream=text_stream(), get_final_message=get_final_message)

        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=parse, stream=stream))


# --- SQL agent ------------------------------------------------------------------
def tool_call(name: str, sql: str, call_id: str = "toolu_1"):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input={"sql": sql})


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def agent_turn(*blocks, stop_reason="tool_use"):
    return fake_message(stop_reason=stop_reason, content=blocks)


class ScriptedClient:
    """Mimics client.beta.messages.create: returns the scripted turns one by one,
    and keeps a snapshot of every request so tests can read what the agent sent."""

    def __init__(self, *turns, error=None):
        self.turns = list(turns)
        self.requests: list[dict] = []

        async def create(**params):
            self.requests.append({**params, "messages": list(params["messages"])})
            if error:
                raise error
            return self.turns.pop(0)

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))
