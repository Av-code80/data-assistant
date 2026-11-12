"""Retrieval tests.

- Unit tests use a fake embedder: fast, deterministic, test our ranking logic.
- The quality gate uses the real local model (free, ~10 s): it fails if a change to the
  knowledge base, the chunking or the search makes retrieval worse.
"""

import zlib

import numpy as np
import pytest

from app.config import get_settings
from app.knowledge import Section, load_sections
from app.retrieval import BM25, FastEmbedEmbedder, KnowledgeIndex, tokenize
from evals.retrieval import evaluate


class FakeEmbedder:
    """Bag of words hashed into 64 dimensions: texts sharing words get close vectors."""

    def embed(self, texts):
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                vectors[row, zlib.crc32(token.encode()) % 64] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


SECTIONS = [
    Section("g/revenue", "Chiffre d'affaires", "g.md", "Somme des ventes en euros."),
    Section("g/returns", "Taux de retour", "g.md", "Part des articles retournés."),
    Section("r/stock", "Stocks", "r.md", "Aucune donnée de stock ni de rupture."),
]


def test_tokenize_folds_accents_and_drops_stopwords():
    assert tokenize("Quel est le taux de retour des Chaussures ?") == ["taux", "retour", "chaussures"]


def test_bm25_rewards_rare_shared_words():
    scores = BM25(["stock rupture", "ventes euros", "ventes retour"]).scores("rupture de stock")
    assert scores[0] > 0 and scores[1] == 0


def test_search_ranks_the_matching_section_first():
    hits = KnowledgeIndex(SECTIONS, FakeEmbedder()).search("niveau de stock", top_k=2)
    assert hits[0].section.id == "r/stock"
    assert len(hits) == 2


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_returns_nothing(query):
    assert KnowledgeIndex(SECTIONS, FakeEmbedder()).search(query) == []


def test_empty_knowledge_base_returns_nothing():
    assert KnowledgeIndex([], FakeEmbedder()).search("ventes") == []


def test_min_vector_score_drops_unrelated_sections():
    hits = KnowledgeIndex(SECTIONS, FakeEmbedder()).search("rupture de stock", top_k=3,
                                                           min_vector_score=0.3)
    assert [h.section.id for h in hits] == ["r/stock"]


def test_retrieval_quality_gate():
    settings = get_settings()
    index = KnowledgeIndex(load_sections(settings.knowledge_dir),
                           FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir))
    report = evaluate(index, top_k=settings.retrieval_top_k)
    assert report.hit_at_k >= 0.95, report.failures
    assert report.mrr >= 0.80, report.failures
