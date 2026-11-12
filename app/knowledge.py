"""Load the business knowledge base (markdown) and split it for retrieval.

Two levels, a technique called small-to-big (or parent-document) retrieval:
- a Section ("## " heading) is what the LLM receives: complete, self-contained context;
- Passages (a few sentences each) are what gets embedded: short enough for the
  embedding model, which truncates long inputs, and precise enough to match a question.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Section:
    id: str        # stable identifier, e.g. "glossary/taux-de-retour-return_rate"
    title: str
    source: str    # file name, shown to users as the citation
    text: str


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9_]+", "-", ascii_text.lower()).strip("-")


def parse_sections(markdown: str, source: str) -> list[Section]:
    """Split a markdown document on '## ' headings. Text before the first one is ignored."""
    sections = []
    for block in re.split(r"^## ", markdown, flags=re.MULTILINE)[1:]:
        title, _, body = block.partition("\n")
        body = " ".join(body.split())          # join wrapped lines, collapse whitespace
        if body:                               # a heading with no content is useless context
            stem = Path(source).stem
            sections.append(Section(f"{stem}/{slugify(title)}", title.strip(), source, body))
    return sections


def load_sections(directory: Path) -> list[Section]:
    sections = [section
                for path in sorted(directory.glob("*.md"))
                for section in parse_sections(path.read_text(encoding="utf-8"), path.name)]
    ids = [section.id for section in sections]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate section titles: {sorted(duplicates)}")
    return sections


def split_passages(section: Section, max_words: int = 40) -> list[str]:
    """Group whole sentences into passages of at most ~max_words words.

    Each passage starts with the section title, so a passage like "Les chaussures ont
    les taux les plus élevés" still says what it is about.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", section.text.strip()) if s]
    passages, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            passages.append(current)
            current = []
        current.append(sentence)
    if current:
        passages.append(current)
    return [f"{section.title}. {' '.join(group)}" for group in passages]
