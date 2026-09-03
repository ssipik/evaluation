"""The golden set: load it, validate it, and say what it can score.

One file feeds every tier — they read different fields of the same entries:

- tier 1 (retrieval)  `question` + `relevant_documents`
- tier 2 (generation) `question` + `reference`
- tier 3 (behaviour)  `expect_refusal`, and the negative controls generally

`reference` and `relevant_documents` are both filled in gradually, so a harness
must report how many entries it could actually score rather than averaging over
the ones that happened to be labeled. A score over 3 of 15 entries is not a score
of the system.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

__all__ = ["DocRef", "Entry", "load_golden", "GoldenSetError"]

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_set.yaml"


class GoldenSetError(ValueError):
    """The set is malformed. Loud, because a silently skipped entry looks like a
    metric that improved."""


@dataclass(frozen=True)
class DocRef:
    """One labeled document. A url alone does not identify a document — two
    sources can list the same url — so `source_system` is part of the key, and
    retrieval is scored on the pair.
    """

    source_system: str
    url: str

    def __str__(self) -> str:
        return f"{self.source_system}  {self.url}"


@dataclass(frozen=True)
class Entry:
    id: str
    question: str
    language: str | None = None
    reference: str = ""
    relevant_documents: tuple[DocRef, ...] = ()
    expect_refusal: bool = False
    tests: str = ""
    source_system: str | None = None  # for the per-source breakdown, optional

    @property
    def scorable_tier1(self) -> bool:
        return bool(self.relevant_documents)

    @property
    def scorable_tier2(self) -> bool:
        return bool(self.reference)


_KNOWN = {f.name for f in fields(Entry)}
_DOC_FIELDS = {"source_system", "url"}


def _docs(raw: object, where: str) -> tuple[DocRef, ...]:
    if isinstance(raw, str) or not isinstance(raw, list):
        raise GoldenSetError(
            f"{where}: relevant_documents must be a list of "
            "{source_system, url} mappings"
        )
    out: list[DocRef] = []
    for i, doc in enumerate(raw, 1):
        if not isinstance(doc, dict) or set(doc) != _DOC_FIELDS:
            raise GoldenSetError(
                f"{where}: relevant_documents[{i}] must have exactly "
                f"{', '.join(sorted(_DOC_FIELDS))}"
            )
        if not doc["source_system"] or not doc["url"]:
            raise GoldenSetError(f"{where}: relevant_documents[{i}] has an empty field")
        out.append(DocRef(source_system=str(doc["source_system"]), url=str(doc["url"])))
    duplicates = {d for d in out if out.count(d) > 1}
    if duplicates:
        raise GoldenSetError(
            f"{where}: duplicate relevant_documents "
            f"{', '.join(sorted(str(d) for d in duplicates))}"
        )
    return tuple(out)


def _entry(raw: dict, where: str) -> Entry:
    if not isinstance(raw, dict):
        raise GoldenSetError(f"{where}: expected a mapping, got {type(raw).__name__}")
    unknown = set(raw) - _KNOWN
    if "relevant_urls" in unknown:
        raise GoldenSetError(
            f"{where}: relevant_urls was replaced by relevant_documents — a list of "
            "{source_system, url} mappings, because a url alone does not identify a "
            "document"
        )
    if unknown:
        raise GoldenSetError(
            f"{where}: unknown field(s) {', '.join(sorted(unknown))} — known: "
            f"{', '.join(sorted(_KNOWN))}"
        )
    for required in ("id", "question"):
        if not raw.get(required):
            raise GoldenSetError(f"{where}: missing {required}")
    return Entry(
        id=str(raw["id"]),
        question=str(raw["question"]),
        language=raw.get("language"),
        reference=raw.get("reference") or "",
        relevant_documents=_docs(raw.get("relevant_documents") or [], where),
        expect_refusal=bool(raw.get("expect_refusal", False)),
        tests=raw.get("tests") or "",
        source_system=raw.get("source_system"),
    )


def load_golden(
    path: str | Path | None = None, ids: Iterable[str] | None = None
) -> list[Entry]:
    """Every entry, or only the named ones. Unknown ids raise rather than
    silently scoring a smaller set."""
    p = Path(path) if path else DEFAULT_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    questions = raw.get("questions") or []
    if not questions:
        raise GoldenSetError(f"no questions in {p}")

    entries = [_entry(q, f"{p}: entry {i}") for i, q in enumerate(questions, 1)]

    seen: set[str] = set()
    duplicates = {e.id for e in entries if e.id in seen or seen.add(e.id)}
    if duplicates:
        raise GoldenSetError(
            f"{p}: duplicate id(s) {', '.join(sorted(duplicates))} — ids are how a "
            "score is tracked across runs, so they have to be unique"
        )

    if ids is not None:
        wanted = set(ids)
        entries = [e for e in entries if e.id in wanted]
        missing = wanted - {e.id for e in entries}
        if missing:
            raise GoldenSetError(f"{p}: unknown id(s) {', '.join(sorted(missing))}")
    return entries
