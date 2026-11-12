"""Shared pytest setup, loaded automatically."""

import os

from pydantic import ValidationError

from app.config import Settings

# Unit tests never call the API, but importing the app needs a key. When neither
# the environment nor .env provides one, use a placeholder. A real key is kept.
try:
    Settings()
except ValidationError:
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-placeholder"
