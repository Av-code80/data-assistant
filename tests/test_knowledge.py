"""Knowledge loading and splitting: pure functions, no model needed."""

import pytest

from app.config import get_settings
from app.knowledge import Section, load_sections, parse_sections, slugify, split_passages


def test_slugify_strips_accents_and_punctuation():
    assert slugify("Chiffre d'affaires (revenue)") == "chiffre-d-affaires-revenue"
    assert slugify("Fraîcheur et historique") == "fraicheur-et-historique"


def test_parse_sections_ignores_intro_and_empty_headings():
    markdown = "# Title\nIntro ignored.\n\n## First\nLine one\nline two.\n\n## Empty\n\n## Last\nText."
    sections = parse_sections(markdown, "doc.md")
    assert [s.id for s in sections] == ["doc/first", "doc/last"]
    assert sections[0].text == "Line one line two."        # wrapped lines are joined


def test_duplicate_titles_are_rejected(tmp_path):
    (tmp_path / "a.md").write_text("## Same\nOne.\n## Same\nTwo.", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_sections(tmp_path)


def test_passages_keep_whole_sentences_and_the_title():
    section = Section("doc/x", "Retours", "doc.md", "Une phrase courte. " * 12)
    passages = split_passages(section, max_words=10)
    assert len(passages) > 1
    assert all(p.startswith("Retours. ") for p in passages)
    assert all(p.endswith(".") for p in passages)            # never cut mid-sentence


def test_real_knowledge_base_loads():
    sections = load_sections(get_settings().knowledge_dir)
    assert len(sections) >= 10
    assert "glossary/chiffre-d-affaires-revenue" in {s.id for s in sections}


def test_empty_section_has_no_passage():
    assert split_passages(Section("doc/x", "Vide", "doc.md", "   ")) == []
