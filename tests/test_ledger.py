"""The frozen ledger's pure parts. No Postgres: the copy itself is exercised by
running a campaign, not by a unit test."""

from __future__ import annotations

from rag_evaluation.ledger import eval_dsn


class TestEvalDsn:
    def test_it_keeps_the_server_and_changes_only_the_database(self):
        dsn = eval_dsn("postgresql+psycopg://rag:rag@db:5433/rag_ingestion", "a3f19c")
        assert dsn == "postgresql+psycopg://rag:rag@db:5433/rag_eval_a3f19c"

    def test_the_password_survives(self):
        """str(URL) masks it as ***, which psycopg would then send as the
        password. The whole sweep fails at the first connection."""
        assert "***" not in eval_dsn("postgresql://rag:secret@db/rag_ingestion", "a1")

    def test_one_database_per_fingerprint(self):
        source = "postgresql+psycopg://rag@localhost/rag_ingestion"
        assert eval_dsn(source, "a3f19c") != eval_dsn(source, "b71e02")
