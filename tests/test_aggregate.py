"""Clustering summary points into multi-document entries — offline, no TEI.

`find_clusters` takes `vectors` so these tests supply their own embeddings; the
filters, not the embedding model, are what this module gets wrong.
"""

from __future__ import annotations

import yaml

from rag_evaluation.aggregate import (
    Aggregated,
    Cluster,
    Point,
    figures_differ,
    find_clusters,
    format_entries,
    load_points,
)
from rag_evaluation.golden import DocRef, load_golden

A = DocRef("web_rss", "https://e.x/md-april")
B = DocRef("web_rss", "https://e.x/md-may")
C = DocRef("web_rss", "https://e.x/md-june")
D = DocRef("web_rss", "https://e.x/md-july")
E = DocRef("web_rss", "https://e.x/md-august")
F = DocRef("web_rss", "https://e.x/md-september")


def point(doc, text, title="Monetary developments"):
    return Point(doc=doc, title=title, text=text)


def near(n):
    """Vectors whose pairwise cosine is ~1.0 within a family, ~0 across."""
    return [1.0 if i == n else 0.0 for i in range(8)]


class TestFiguresDiffer:
    def test_same_wording_different_numbers_is_a_series(self):
        assert figures_differ(
            [point(A, "M3 growth was 3.2% in March"), point(B, "M3 growth was 2.7% in April")]
        )

    def test_identical_text_is_one_fact_duplicated(self):
        # labelling both as required would score retrieving either alone as 50%
        # recall, so this is not the kind of entry this module writes
        same = "Bankgeschäfte dürfen nur mit Erlaubnis der BaFin erbracht werden."
        assert not figures_differ([point(A, same), point(B, same)])

    def test_prose_without_figures_never_qualifies(self):
        # the real case: BaFin's cookie banner, cosine 0.990 across documents
        assert not figures_differ(
            [
                point(A, "Technisch notwendige Cookies sind nicht abwählbar."),
                point(B, "Technisch notwendige Cookies auf der Website sind nicht abwählbar."),
            ]
        )


class TestFindClusters:
    def test_a_series_across_three_documents_is_kept(self):
        pts = [
            point(A, "M3 growth was 3.2% in March"),
            point(B, "M3 growth was 2.7% in April"),
            point(C, "M3 growth was 3.3% in May"),
        ]
        (cluster,) = find_clusters(pts, vectors=[near(0), near(0), near(0)])
        assert len(cluster.documents) == 3

    def test_points_from_one_document_never_pair(self):
        # a document restating its own figure is not a multi-document entry
        pts = [point(A, "M3 growth was 3.2% in March"), point(A, "M3 rose 3.2% that month")]
        assert find_clusters(pts, vectors=[near(0), near(0)]) == []

    def test_a_cluster_wider_than_the_limit_is_boilerplate(self):
        pts = [point(d, f"Rule {i} applies.") for i, d in enumerate((A, B, C, D, E, F), 1)]
        assert find_clusters(pts, max_documents=5, vectors=[near(0)] * 6) == []

    def test_below_the_threshold_nothing_groups(self):
        pts = [point(A, "M3 was 3.2%"), point(B, "M3 was 2.7%")]
        assert find_clusters(pts, vectors=[near(0), near(1)]) == []

    def test_unrelated_families_do_not_merge(self):
        pts = [
            point(A, "M3 growth was 3.2% in March"),
            point(B, "M3 growth was 2.7% in April"),
            point(A, "M1 growth was 4.0% in March"),
            point(B, "M1 growth was 3.8% in April"),
        ]
        clusters = find_clusters(pts, vectors=[near(0), near(0), near(1), near(1)])
        assert len(clusters) == 2
        assert all(len(c.documents) == 2 for c in clusters)

    def test_fewer_than_two_points_is_not_a_cluster(self):
        assert find_clusters([point(A, "M3 was 3.2%")], vectors=[near(0)]) == []


class TestFormatEntries:
    def make(self, question="How did M3 develop between March and May 2026?"):
        cluster = Cluster(
            points=(
                point(A, "M3 growth was 3.2% in March"),
                point(B, "M3 growth was 2.7% in April"),
                point(C, "M3 growth was 3.3% in May"),
            ),
            similarity=0.93,
        )
        return Aggregated(cluster=cluster, question=question, answer="3.2, 2.7, 3.3.", language="en")

    def load(self, text, tmp_path):
        p = tmp_path / "aggregate.yaml"
        p.write_text("questions:\n" + text, encoding="utf-8")
        return load_golden(p)

    def test_the_output_loads_as_a_golden_set(self, tmp_path):
        (entry,) = self.load(format_entries([self.make()]), tmp_path)
        assert entry.question.startswith("How did M3")

    def test_every_document_of_the_cluster_is_labelled(self, tmp_path):
        (entry,) = self.load(format_entries([self.make()]), tmp_path)
        assert {d.url for d in entry.relevant_documents} == {A.url, B.url, C.url}

    def test_the_tests_note_states_the_count_to_check(self, tmp_path):
        (entry,) = self.load(format_entries([self.make()]), tmp_path)
        assert "aggregation across 3 documents" in entry.tests

    def test_ids_do_not_collide_with_the_existing_set(self, tmp_path):
        text = format_entries([self.make()], existing_ids=["how-did-develop-between-march-and-may"])
        (entry,) = self.load(text, tmp_path)
        assert entry.id != "how-did-develop-between-march-and-may"

    def test_reference_is_empty_and_the_answer_is_a_comment(self, tmp_path):
        text = format_entries([self.make()])
        (entry,) = self.load(text, tmp_path)
        assert entry.reference == ""
        assert "# proposed answer: 3.2, 2.7, 3.3." in text


class TestLoadPoints:
    def test_it_reads_a_review_file(self, tmp_path):
        review = tmp_path / "review.yaml"
        review.write_text(
            yaml.dump(
                {
                    "generated": [
                        {
                            "source_system": "web_rss",
                            "url": "https://e.x/d1",
                            "title": "T",
                            "summary": ["one", "two"],
                        }
                    ],
                    "failed": [],
                }
            ),
            encoding="utf-8",
        )
        pts = load_points(review)
        assert [p.text for p in pts] == ["one", "two"]
        assert pts[0].doc == DocRef("web_rss", "https://e.x/d1")

    def test_a_run_with_no_generated_records_yields_nothing(self, tmp_path):
        review = tmp_path / "review.yaml"
        review.write_text(yaml.dump({"generated": [], "failed": []}), encoding="utf-8")
        assert load_points(review) == []
