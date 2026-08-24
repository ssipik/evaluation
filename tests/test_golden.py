"""The golden set is data, so a typo has to fail loudly rather than shrink the set."""

from __future__ import annotations

import pytest

from rag_evaluation.golden import DEFAULT_PATH, DocRef, GoldenSetError, load_golden


def write(tmp_path, body):
    p = tmp_path / "golden_set.yaml"
    p.write_text(body, encoding="utf-8")
    return p


class TestShippedSet:
    def test_it_loads(self):
        assert len(load_golden(DEFAULT_PATH)) >= 15

    def test_ids_are_unique(self):
        entries = load_golden(DEFAULT_PATH)
        assert len({e.id for e in entries}) == len(entries)

    def test_it_has_a_negative_control(self):
        """Without one, an agent answering from its own weights is invisible."""
        assert any(e.expect_refusal for e in load_golden(DEFAULT_PATH))

    def test_every_entry_declares_a_language(self):
        assert all(e.language for e in load_golden(DEFAULT_PATH))


class TestValidation:
    def test_duplicate_ids_raise(self, tmp_path):
        p = write(tmp_path, "questions:\n  - {id: a, question: q}\n  - {id: a, question: r}\n")
        with pytest.raises(GoldenSetError, match="duplicate id"):
            load_golden(p)

    def test_unknown_field_raises(self, tmp_path):
        p = write(tmp_path, "questions:\n  - {id: a, question: q, relevent_docs: []}\n")
        with pytest.raises(GoldenSetError, match="unknown field"):
            load_golden(p)

    def test_the_old_relevant_urls_field_says_what_replaced_it(self, tmp_path):
        p = write(tmp_path, 'questions:\n  - {id: a, question: q, relevant_urls: ["u"]}\n')
        with pytest.raises(GoldenSetError, match="replaced by relevant_documents"):
            load_golden(p)

    def test_missing_question_raises(self, tmp_path):
        p = write(tmp_path, "questions:\n  - {id: a}\n")
        with pytest.raises(GoldenSetError, match="missing question"):
            load_golden(p)

    def test_relevant_documents_must_be_a_list(self, tmp_path):
        p = write(
            tmp_path,
            'questions:\n  - {id: a, question: q, relevant_documents: "http://x"}\n',
        )
        with pytest.raises(GoldenSetError, match="must be a list"):
            load_golden(p)

    def test_a_bare_url_is_not_a_document(self, tmp_path):
        """The url alone was the old shape — it must not load as a document."""
        p = write(
            tmp_path,
            'questions:\n  - {id: a, question: q, relevant_documents: ["http://x"]}\n',
        )
        with pytest.raises(GoldenSetError, match="must have exactly"):
            load_golden(p)

    def test_a_document_without_a_source_system_raises(self, tmp_path):
        p = write(
            tmp_path,
            "questions:\n  - {id: a, question: q, "
            'relevant_documents: [{url: "http://x"}]}\n',
        )
        with pytest.raises(GoldenSetError, match="must have exactly"):
            load_golden(p)

    def test_an_empty_field_raises(self, tmp_path):
        p = write(
            tmp_path,
            "questions:\n  - {id: a, question: q, "
            'relevant_documents: [{source_system: "", url: "http://x"}]}\n',
        )
        with pytest.raises(GoldenSetError, match="empty field"):
            load_golden(p)

    def test_the_same_document_twice_raises(self, tmp_path):
        p = write(
            tmp_path,
            "questions:\n  - {id: a, question: q, relevant_documents: ["
            '{source_system: web_rss, url: "http://x"}, '
            '{source_system: web_rss, url: "http://x"}]}\n',
        )
        with pytest.raises(GoldenSetError, match="duplicate relevant_documents"):
            load_golden(p)

    def test_one_url_under_two_source_systems_is_two_documents(self, tmp_path):
        """What the pair-shaped key buys: the url alone would collapse these."""
        p = write(
            tmp_path,
            "questions:\n  - {id: a, question: q, relevant_documents: ["
            '{source_system: web_rss, url: "http://x"}, '
            '{source_system: confluence, url: "http://x"}]}\n',
        )
        (entry,) = load_golden(p)
        assert len(entry.relevant_documents) == 2

    def test_unknown_id_filter_raises(self, tmp_path):
        p = write(tmp_path, "questions:\n  - {id: a, question: q}\n")
        with pytest.raises(GoldenSetError, match="unknown id"):
            load_golden(p, ids=["b"])


class TestScorability:
    def test_an_entry_without_labels_is_not_tier1_scorable(self, tmp_path):
        p = write(tmp_path, "questions:\n  - {id: a, question: q}\n")
        (entry,) = load_golden(p)
        assert not entry.scorable_tier1 and not entry.scorable_tier2

    def test_labels_and_reference_are_independent(self, tmp_path):
        p = write(
            tmp_path,
            "questions:\n  - id: a\n    question: q\n    relevant_documents:\n"
            '      - {source_system: web_rss, url: "http://x"}\n',
        )
        (entry,) = load_golden(p)
        assert entry.relevant_documents == (DocRef("web_rss", "http://x"),)
        assert entry.scorable_tier1 and not entry.scorable_tier2
