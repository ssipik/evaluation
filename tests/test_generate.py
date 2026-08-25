"""Offline: no Postgres, no Qdrant, no API key.

The one test that matters most is the round trip — what `generate` writes has to
load back through `load_golden`, or the entries are hand-editing work rather
than paste-ready output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import yaml

from rag_evaluation.generate import (
    Failure,
    Generated,
    GenerationError,
    Sampled,
    _generate_one,
    document_text,
    sample_documents,
    format_entries,
    format_review,
)
from rag_evaluation.golden import DocRef, load_golden

DOC = DocRef("web_rss", "https://ecb.example/pr.html")
OTHER = DocRef("confluence", "https://ecb.example/pr.html")  # same url, other source


@dataclass
class FakeRecord:
    payload: dict


class FakeSearcher:
    """Stands in for QdrantSearcher.scroll_document, which sorts by node_seq."""

    def __init__(self, records):
        self.records = records

    def scroll_document(self, source_url, source_system=None, limit=1024):
        return sorted(
            (
                r
                for r in self.records
                if r.payload["source_url"] == source_url
                and (source_system is None or r.payload["source_system"] == source_system)
            ),
            key=lambda r: r.payload["node_seq"],
        )


def chunk(seq, text, source_system="web_rss", language="en"):
    return FakeRecord(
        {
            "node_seq": seq,
            "text": text,
            "source_url": DOC.url,
            "source_system": source_system,
            "language": language,
        }
    )


def sampled(doc=DOC, title="ECB press release"):
    return Sampled(doc=doc, title=title, source_id="ecb_press")


class FakeClient:
    """The OpenAI-shaped surface `_generate_one` touches."""

    def __init__(self, content):
        payload = content if isinstance(content, str) else json.dumps(content)
        message = type("M", (), {"content": payload})()
        choice = type("C", (), {"message": message})()
        self.response = type("R", (), {"choices": [choice]})()
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


GOOD = {
    "summary": [
        "The ECB selected 12 payment service providers.",
        "The pilot runs until 2027.",
        "Participants cover eight member states.",
        "No consumer funds are at risk.",
        "A report follows in Q1.",
    ],
    "question": "How many payment service providers did the ECB select for the pilot?",
    "answer": "Twelve.",
    "point_index": 0,
    "language": "en",
}


class TestSampleDocuments:
    """Against SQLite, so the join and the filter are actually executed. The
    real schema is Postgres', but this query is plain SQL."""

    def ledger(self, rows):
        from sqlalchemy import create_engine, insert

        from rag_evaluation.generate import _metadata, ingest_docs, ingest_manifest

        engine = create_engine("sqlite://")
        _metadata.create_all(engine)
        with engine.begin() as conn:
            for manifest_id, source_system, url, status in rows:
                conn.execute(
                    insert(ingest_manifest).values(
                        manifest_id=manifest_id,
                        source_system=source_system,
                        doc_url=url,
                        source_id="src",
                        title=f"doc {manifest_id}",
                    )
                )
                if status:
                    conn.execute(
                        insert(ingest_docs).values(
                            manifest_id=manifest_id, status=status
                        )
                    )
        return engine

    def test_a_document_listed_by_several_runs_appears_once(self):
        """Three sightings of one document, and the ledger points at the newest
        — the join is what collapses them."""
        engine = self.ledger(
            [
                (1, "web_rss", "https://a", None),
                (2, "web_rss", "https://a", None),
                (3, "web_rss", "https://a", "active"),
            ]
        )
        out = sample_documents(n=50, engine=engine)
        assert [s.doc for s in out] == [DocRef("web_rss", "https://a")]

    def test_only_active_documents_are_drawn(self):
        engine = self.ledger(
            [
                (1, "web_rss", "https://a", "active"),
                (2, "web_rss", "https://b", "deleted"),
                (3, "web_rss", "https://c", "empty"),
            ]
        )
        out = sample_documents(n=50, engine=engine)
        assert [s.doc.url for s in out] == ["https://a"]

    def test_one_url_under_two_sources_is_two_documents(self):
        engine = self.ledger(
            [
                (1, "web_rss", "https://a", "active"),
                (2, "confluence", "https://a", "active"),
            ]
        )
        out = sample_documents(n=50, engine=engine)
        assert {s.doc.source_system for s in out} == {"web_rss", "confluence"}

    def test_a_document_with_no_ledger_row_is_not_drawn(self):
        """Discovered but never ingested — no vectors, so nothing to ask about."""
        engine = self.ledger([(1, "web_rss", "https://a", None)])
        assert sample_documents(n=50, engine=engine) == []

    def many(self):
        return self.ledger(
            [(i, "web_rss", f"https://doc{i}", "active") for i in range(1, 21)]
        )

    def test_the_same_seed_draws_the_same_sample(self):
        engine = self.many()
        first = sample_documents(n=5, seed=7, engine=engine)
        second = sample_documents(n=5, seed=7, engine=engine)
        assert [s.doc for s in first] == [s.doc for s in second]

    def test_a_different_seed_draws_a_different_sample(self):
        engine = self.many()
        first = sample_documents(n=5, seed=7, engine=engine)
        second = sample_documents(n=5, seed=8, engine=engine)
        assert [s.doc for s in first] != [s.doc for s in second]

    def test_asking_for_more_than_there_is_returns_everything(self):
        assert len(sample_documents(n=500, engine=self.many())) == 20

    def test_the_title_comes_along(self):
        engine = self.ledger([(1, "web_rss", "https://a", "active")])
        assert sample_documents(n=1, engine=engine)[0].title == "doc 1"


class TestDocumentText:
    def test_chunks_are_joined_in_node_seq_order(self):
        searcher = FakeSearcher([chunk(2, "second"), chunk(0, "first"), chunk(1, "mid")])
        body, language = document_text(searcher, sampled(), max_chars=1000)
        assert body == "first\n\nmid\n\nsecond"
        assert language == "en"

    def test_the_other_source_systems_chunks_are_left_out(self):
        """Same url, two sources — two documents, and mixing them would put text
        the question was never written from into the prompt."""
        searcher = FakeSearcher(
            [chunk(0, "ours"), chunk(1, "theirs", source_system="confluence")]
        )
        body, _ = document_text(searcher, sampled(), max_chars=1000)
        assert body == "ours"

    def test_a_document_with_no_chunks_here_is_empty(self):
        searcher = FakeSearcher([chunk(0, "theirs", source_system="confluence")])
        body, _ = document_text(searcher, sampled(), max_chars=1000)
        assert body == ""

    def test_max_chars_truncates_rather_than_dropping_the_chunk(self):
        searcher = FakeSearcher([chunk(0, "aaaa"), chunk(1, "bbbb")])
        body, _ = document_text(searcher, sampled(), max_chars=6)
        assert body == "aaaa\n\nbb"


class TestGenerateOne:
    def test_a_well_formed_response_parses(self):
        client = FakeClient(GOOD)
        out = _generate_one(client, "m", sampled(), "body")
        assert out.question.startswith("How many")
        assert out.point == GOOD["summary"][0]
        assert out.language == "en"

    def test_the_document_text_reaches_the_model(self):
        client = FakeClient(GOOD)
        _generate_one(client, "m", sampled(), "the body text")
        assert "the body text" in client.calls[0]["messages"][1]["content"]

    def test_a_point_index_past_the_end_raises(self):
        """The contract of the whole script: the answer lives in a point."""
        client = FakeClient({**GOOD, "point_index": 9})
        with pytest.raises(GenerationError, match="outside the 5 summary points"):
            _generate_one(client, "m", sampled(), "body")

    def test_a_negative_point_index_raises(self):
        client = FakeClient({**GOOD, "point_index": -1})
        with pytest.raises(GenerationError, match="outside"):
            _generate_one(client, "m", sampled(), "body")

    def test_prose_instead_of_json_raises(self):
        client = FakeClient("Sure! Here is your summary:")
        with pytest.raises(GenerationError, match="not JSON"):
            _generate_one(client, "m", sampled(), "body")

    def test_an_empty_question_raises(self):
        client = FakeClient({**GOOD, "question": ""})
        with pytest.raises(GenerationError, match="empty question"):
            _generate_one(client, "m", sampled(), "body")


def generated(**overrides):
    base = dict(
        sampled=sampled(),
        summary=tuple(GOOD["summary"]),
        question=GOOD["question"],
        answer=GOOD["answer"],
        point_index=0,
        language="en",
    )
    return Generated(**{**base, **overrides})


class TestFormatEntries:
    def load(self, text, tmp_path):
        p = tmp_path / "generated.yaml"
        p.write_text("questions:\n" + text, encoding="utf-8")
        return load_golden(p)

    def test_the_output_loads_back_as_a_golden_set(self, tmp_path):
        entries = self.load(format_entries([generated()]), tmp_path)
        assert len(entries) == 1
        assert entries[0].question == GOOD["question"]
        assert entries[0].relevant_documents == (DOC,)

    def test_the_label_is_the_document_the_question_came_from(self, tmp_path):
        (entry,) = self.load(format_entries([generated(sampled=sampled(OTHER))]), tmp_path)
        assert entry.relevant_documents == (OTHER,)

    def test_reference_is_empty_by_default(self, tmp_path):
        """EVALUATION.md: references are written by hand, so tier 2 must not
        silently start scoring against machine-written ones."""
        (entry,) = self.load(format_entries([generated()]), tmp_path)
        assert entry.reference == ""

    def test_the_answer_is_still_there_to_read(self):
        assert "Twelve." in format_entries([generated()])

    def test_with_reference_fills_the_field(self, tmp_path):
        (entry,) = self.load(
            format_entries([generated()], with_reference=True), tmp_path
        )
        assert entry.reference == "Twelve."

    def test_ids_do_not_collide_with_the_existing_set(self, tmp_path):
        two = [generated(), generated(sampled=sampled(OTHER))]
        entries = self.load(format_entries(two), tmp_path)
        assert len({e.id for e in entries}) == 2

    def test_an_id_taken_by_the_shipped_set_is_not_reused(self):
        text = format_entries([generated()], existing_ids=["how-many-payment"])
        first_id = [ln for ln in text.splitlines() if "- id:" in ln][0]
        assert "how-many-payment:" not in first_id

    def test_a_multiline_answer_does_not_break_the_yaml(self, tmp_path):
        """A newline inside a `#` comment would end it and leave the rest to be
        parsed as YAML."""
        entries = self.load(
            format_entries([generated(answer="line one\nline two: with a colon")]),
            tmp_path,
        )
        assert len(entries) == 1

    def test_a_url_with_a_fragment_does_not_start_a_comment(self, tmp_path):
        doc = DocRef("confluence", "https://wiki.example/page # notes")
        (entry,) = self.load(format_entries([generated(sampled=sampled(doc))]), tmp_path)
        assert entry.relevant_documents == (doc,)

    def test_the_id_is_cut_on_a_word_boundary(self):
        text = format_entries(
            [generated(question="What was the aggregate monetary developments figure?")]
        )
        (line,) = [ln for ln in text.splitlines() if "- id:" in ln]
        assert not line.strip().endswith("-")
        for word in line.split("id: ")[1].split("-"):
            assert word in "what was the aggregate monetary developments figure"

    def test_a_question_with_quotes_and_umlauts_survives(self, tmp_path):
        question = 'Was bedeutet "Laufzeit" für den Fonds: kurz oder lang?'
        (entry,) = self.load(format_entries([generated(question=question)]), tmp_path)
        assert entry.question == question


class TestFormatReview:
    def test_it_parses_as_yaml_and_carries_the_whole_summary(self):
        report = yaml.safe_load(format_review([generated()], []))
        (entry,) = report["generated"]
        # the whole summary, not just the point the question used
        assert entry["summary"] == list(GOOD["summary"])
        assert entry["point_index"] == 0
        assert entry["question"] == GOOD["question"]

    def test_failures_are_recorded_rather_than_dropped(self):
        report = yaml.safe_load(
            format_review([], [Failure(DOC, "no chunks in the collection")])
        )
        assert report["generated"] == []
        (failure,) = report["failed"]
        assert failure["error"] == "no chunks in the collection"
        assert failure["url"] == DOC.url

    def test_umlauts_stay_readable_rather_than_escaped(self):
        # the reason for YAML over JSON: this file is read by a person
        text = format_review([generated(answer="Die Zinssätze blieben unverändert.")], [])
        assert "Zinssätze" in text
        assert "\\u" not in text
