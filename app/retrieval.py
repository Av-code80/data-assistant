"""Hybrid search over the knowledge base.

Two complementary rankings, merged:
- vector search: an embedding turns a text into a vector (here 384 numbers) so that
  texts with a similar meaning get close vectors. Good at paraphrases
  ("combien a-t-on vendu" ~ "chiffre d'affaires"), weak on exact rare words.
- keyword search (BM25): scores passages sharing the question's words, weighting rare
  words more. Good at exact terms ("marge", "Facebook", "T3"), blind to paraphrases.
Reciprocal Rank Fusion (RRF) merges the two rankings using positions only, so the two
very different score scales never have to be compared.

The knowledge base is small (dozens of passages), so everything stays in memory; at
scale this becomes pgvector, Vertex AI Vector Search or BigQuery vector search behind
the same `search` method.
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .knowledge import Section, split_passages

# Frequent French words carry no meaning for keyword matching.
STOPWORDS = set("""le la les l un une des de du d et ou a au aux en dans sur pour par
avec sans ce cet cette ces est sont ont a-t-il quel quelle quels quelles qui que quoi
combien comment on nous notre nos vous votre il elle ils elles se sa son ses leur leurs
plus moins tout tous pas ne y t s qu c n m j""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, split on non-letters, drop stopwords and 1-letter tokens."""
    ascii_text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return [t for t in re.findall(r"[a-z0-9_]+", ascii_text) if len(t) > 1 and t not in STOPWORDS]


class BM25:
    """Okapi BM25, the classic search-engine ranking function."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(tokenize(doc)) for doc in documents]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.avg_length = sum(self.lengths) / len(self.docs) if self.docs else 0
        self.k1, self.b = k1, b
        frequency = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        results = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for term in terms:
                if term in doc:
                    tf = doc[term]
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                    score += self.idf[term] * tf * (self.k1 + 1) / norm
            results.append(score)
        return results


class Embedder(Protocol):
    """Anything that turns texts into L2-normalized vectors (one row per text)."""

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedEmbedder:
    """Local embedding model (ONNX, CPU only): free, offline, no API key."""

    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed import TextEmbedding   # imported here: loading takes a few seconds
        self.model = TextEmbedding(model_name, cache_dir=str(cache_dir))

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@dataclass(frozen=True)
class SearchHit:
    section: Section
    score: float    # fused RRF score: only meaningful to compare hits of the same query
    vector: float   # cosine similarity of the section's best passage, between -1 and 1


RRF_K = 60  # standard constant: dampens the gap between rank 1 and rank 2


def best_per_section(scores, owners: list[int]) -> dict[int, float]:
    """A section scores as well as its best passage."""
    best: dict[int, float] = {}
    for row, score in enumerate(scores):
        best[owners[row]] = max(best.get(owners[row], float("-inf")), float(score))
    return best


def ranks(scores: dict[int, float]) -> dict[int, int]:
    """Section index -> rank (1 = best)."""
    ordered = sorted(scores, key=scores.get, reverse=True)
    return {section: position + 1 for position, section in enumerate(ordered)}


class KnowledgeIndex:
    def __init__(self, sections: list[Section], embedder: Embedder, hybrid: bool = True):
        self.sections = sections
        self.embedder = embedder
        self.hybrid = hybrid
        self.passage_owner: list[int] = []      # passage row -> index of its section
        passages: list[str] = []
        for index, section in enumerate(sections):
            for passage in split_passages(section):
                passages.append(passage)
                self.passage_owner.append(index)
        self.matrix = embedder.embed(passages) if passages else np.zeros((0, 0), np.float32)
        self.bm25 = BM25(passages)

    def search(self, query: str, top_k: int = 3, min_vector_score: float = 0.0) -> list[SearchHit]:
        """Best sections for the query, best first. Can return fewer than top_k, or none.

        min_vector_score drops sections whose meaning is too far from the question,
        even if they share a keyword with it.
        """
        if not query.strip() or len(self.matrix) == 0:
            return []
        vector = best_per_section(self.matrix @ self.embedder.embed([query])[0],
                                  self.passage_owner)
        fused = {i: 1 / (RRF_K + r) for i, r in ranks(vector).items()}
        if self.hybrid:
            keyword = {i: s for i, s in best_per_section(self.bm25.scores(query),
                                                         self.passage_owner).items() if s > 0}
            for i, r in ranks(keyword).items():   # sections with no shared word get nothing
                fused[i] += 1 / (RRF_K + r)
        ordered = sorted(fused, key=fused.get, reverse=True)
        return [SearchHit(self.sections[i], round(fused[i], 5), round(vector[i], 4))
                for i in ordered if vector[i] >= min_vector_score][:top_k]