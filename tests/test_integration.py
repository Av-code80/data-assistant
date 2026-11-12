"""Integration tests: the real pipeline (local retrieval + real Claude calls, ~1 cent each).

Skipped automatically without a real key. In a company they run before each
release to check that the prompts and the knowledge base still produce correct results.
"""

from datetime import date
from functools import lru_cache

import pytest

from app.config import Settings, get_settings
from app.knowledge import load_sections
from app.llm import LLMService
from app.retrieval import FastEmbedEmbedder, KnowledgeIndex

# Settings reads the environment first, then .env.
has_real_key = Settings().anthropic_api_key.get_secret_value().startswith("sk-ant-")
pytestmark = pytest.mark.skipif(not has_real_key, reason="no real ANTHROPIC_API_KEY set")
TODAY = date(2026, 10, 25)


@lru_cache
def index() -> KnowledgeIndex:
    settings = get_settings()
    return KnowledgeIndex(load_sections(settings.knowledge_dir),
                          FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir))


async def interpret(question: str, today: date = TODAY):
    context = [hit.section for hit in index().search(question)]
    return await LLMService(Settings()).interpret_question(question, context=context, today=today)


async def test_clear_question_is_ready():
    result = await interpret(
        "Quel est le chiffre d'affaires par catégorie de produit au troisième trimestre 2026 ?")
    assert result.status == "ready"
    assert result.interpretation.metric == "revenue"
    assert "product_category" in result.interpretation.dimensions
    assert (result.interpretation.time_range.start, result.interpretation.time_range.end) == ("2026-07-01",
                                                                               "2026-09-30")


async def test_glossary_convention_removes_the_ambiguity():
    # Without the glossary, "le plus vendu" made the assistant ask: revenue or units?
    result = await interpret(
        "Quelles catégories ont le plus vendu en France le mois dernier ?")
    assert result.status == "ready"
    assert result.interpretation.metric == "revenue"


async def test_stored_values_come_from_the_documentation():
    result = await interpret("Combien de pulls vendus en Allemagne en 2025 ?")
    values = {f.dimension: f.values for f in result.interpretation.filters}
    assert values.get("product_category") == ["Sweaters"]
    assert values.get("country") == ["Germany"]


async def test_unavailable_data_is_out_of_scope():
    result = await interpret("Quel est notre niveau de stock sur les manteaux ?")
    assert result.status == "out_of_scope"


async def test_write_request_is_out_of_scope():
    result = await interpret("Supprime toutes les commandes annulées de 2025.")
    assert result.status == "out_of_scope"


async def test_current_quarter_is_clamped_to_available_data():
    result = await interpret("Comment vont les ventes ce trimestre par rapport à l'année dernière ?")
    assert result.status == "ready"
    assert result.interpretation.time_range.end == "2026-10-24"
    assert result.warnings


async def test_assumptions_do_not_contradict_the_coverage_warnings():
    # Data ends on 24/10 but today is 02/11: only the server reports coverage, so the
    # model's assumptions must not claim another end date or comparison length.
    result = await interpret("Comment vont les ventes de pulls en Allemagne ce trimestre par "
                             "rapport à l'an dernier ?", today=date(2026, 11, 2))
    assert result.status == "ready"
    assert result.interpretation.time_range.end == "2026-10-24"
    assert result.warnings
    coverage_words = ("disponible", "s'arrêt", "incomplet", "novembre", "durée")
    assumptions = " ".join(result.interpretation.assumptions).lower()
    assert not any(word in assumptions for word in coverage_words), assumptions
