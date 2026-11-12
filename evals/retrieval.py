"""Retrieval evaluation: does the search find the right section for real user questions?

Each case is a question a business user could ask, with the section that must be
retrieved. Metrics:
- hit@1: the right section is ranked first;
- hit@k: the right section is in the top k (what the LLM actually receives);
- MRR (mean reciprocal rank): 1 if first, 1/2 if second, ... averaged.

Run (from data-assistant/):  uv run python -m evals.retrieval
"""

from dataclasses import dataclass

from app.retrieval import KnowledgeIndex

CASES: list[tuple[str, str]] = [
    ("Quel est le CA de septembre ?", "glossary/chiffre-d-affaires-revenue"),
    ("Combien a-t-on vendu en France cette année ?", "glossary/chiffre-d-affaires-revenue"),
    ("Quelle est la GMV du trimestre ?", "glossary/chiffre-d-affaires-revenue"),
    ("Combien de commandes la semaine dernière ?", "glossary/nombre-de-commandes-order_count"),
    ("Quel est le panier moyen en Allemagne ?", "glossary/panier-moyen-average_order_value"),
    ("Combien de pièces ont été vendues en juillet ?", "glossary/articles-vendus-units_sold"),
    ("Combien de clients ont acheté en 2025 ?", "glossary/clients-customer_count"),
    ("Quelle catégorie a le plus d'articles renvoyés ?", "glossary/taux-de-retour-return_rate"),
    ("Combien de remboursements sur les chaussures ?", "glossary/taux-de-retour-return_rate"),
    ("Les montants sont-ils hors taxes ?", "rules/devise-taxes-et-remises"),
    ("Quelle est notre marge nette ?", "rules/devise-taxes-et-remises"),
    ("Quelles sont les dates du T3 ?", "rules/calendrier-et-periodes"),
    ("Jusqu'à quelle date les données sont-elles disponibles ?", "rules/fraicheur-et-historique-des-donnees"),
    ("Dans quels pays vend-on ?", "rules/perimetre-geographique"),
    ("Quel est notre niveau de stock ?", "rules/donnees-non-disponibles"),
    ("Combien a coûté la dernière campagne Facebook ?", "rules/donnees-non-disponibles"),
    ("Que signifie le statut Shipped ?", "rules/statuts-des-commandes-et-des-articles"),
    ("Les commandes annulées sont-elles comptées ?", "rules/statuts-des-commandes-et-des-articles"),
    ("Quelles catégories de produits existent ?", "tables/table-products-produits"),
    ("D'où viennent nos clients, quel canal d'acquisition ?", "tables/table-users-clients"),
]


@dataclass
class RetrievalReport:
    hit_at_1: float
    hit_at_k: float
    mrr: float
    failures: list[tuple[str, str, list[str]]]   # question, expected, retrieved ids


def evaluate(index: KnowledgeIndex, top_k: int = 3) -> RetrievalReport:
    hits_1 = hits_k = reciprocal_ranks = 0.0
    failures = []
    for question, expected in CASES:
        retrieved = [hit.section.id for hit in index.search(question, top_k=top_k)]
        if expected in retrieved:
            rank = retrieved.index(expected) + 1
            hits_k += 1
            hits_1 += rank == 1
            reciprocal_ranks += 1 / rank
        else:
            failures.append((question, expected, retrieved))
    n = len(CASES)
    return RetrievalReport(hits_1 / n, hits_k / n, reciprocal_ranks / n, failures)


if __name__ == "__main__":
    from app.config import get_settings
    from app.knowledge import load_sections
    from app.retrieval import FastEmbedEmbedder

    settings = get_settings()
    sections = load_sections(settings.knowledge_dir)
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
    for label, hybrid in [("vector only", False), ("hybrid", True)]:
        report = evaluate(KnowledgeIndex(sections, embedder, hybrid=hybrid), settings.retrieval_top_k)
        print(f"{label:12} {len(CASES)} questions | hit@1 {report.hit_at_1:.0%} | "
              f"hit@{settings.retrieval_top_k} {report.hit_at_k:.0%} | MRR {report.mrr:.2f}")
        for question, expected, retrieved in report.failures:
            print(f"  MISS {question!r}\n       expected {expected}\n       got      {retrieved}")
