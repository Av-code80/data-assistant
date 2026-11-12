"""Structured JSON logs, one line per event, written to stdout.

In production (Cloud Run), Cloud Logging ingests them and lets you filter by
field: cost, model, prompt version, latency.
"""

import json
import logging


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {"level": record.levelname, "event": record.getMessage()}
        line.update(getattr(record, "data", {}))
        return json.dumps(line, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("app")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
