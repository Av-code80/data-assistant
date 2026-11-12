"""Run the pipeline on one question from the terminal: retrieve -> interpret -> SQL agent.

Real Claude calls (a few cents per question). From data-assistant/:
    uv run python scripts/ask.py "Quel est le CA par catégorie au T3 2026 ?"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import SqlAgent  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.knowledge import load_sections  # noqa: E402
from app.llm import LLMService  # noqa: E402
from app.retrieval import FastEmbedEmbedder, KnowledgeIndex  # noqa: E402
from app.sql_tool import SqlTool  # noqa: E402


async def main(question: str) -> None:
    settings = get_settings()
    index = KnowledgeIndex(load_sections(settings.knowledge_dir),
                           FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir))
    context = [hit.section for hit in index.search(question, settings.retrieval_top_k,
                                                   settings.retrieval_min_vector_score)]
    llm = LLMService(settings)

    interpreted = await llm.interpret_question(question, context=context)
    print(f"Status: {interpreted.status}  warnings: {interpreted.warnings}")
    if interpreted.status != "ready":
        print(interpreted.interpretation.model_dump_json(indent=2))
        return

    agent = SqlAgent(llm, SqlTool(settings.analytics_db_path, settings.sql_max_rows,
                                  settings.sql_timeout_s))
    result = await agent.answer(question, interpreted.interpretation, context)

    print(f"\nAgent: {result.status} in {len(result.steps)} tool calls, "
          f"{result.usage.cost_usd:.4f} USD, {result.usage.latency_ms} ms")
    for number, step in enumerate(result.steps, 1):
        outcome = f"{step.rows} rows" if step.ok else step.error
        print(f"  {number}. {step.tool}: {outcome}")
    if result.result:
        print(f"\n{result.result.sql}\n")
        print(" | ".join(result.result.columns))
        for row in result.result.rows[:15]:
            print(" | ".join(str(value) for value in row))


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Quel est le CA par catégorie au T3 2026 ?"))
