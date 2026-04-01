"""Helpers for querying PubMed from trial identifiers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from Bio import Entrez


def search_pubmed_by_code(code: str, email: str | None = None) -> dict[str, Any]:
    """Search PubMed for a trial identifier such as an NCT code."""
    if email:
        Entrez.email = email
    with Entrez.esearch(db="pubmed", term=code) as handle:
        return Entrez.read(handle)


def search_pubmed_for_codes(
    codes: Iterable[str],
    email: str | None = None,
) -> list[dict[str, Any]]:
    """Search PubMed for many identifiers and return a compact summary."""
    results: list[dict[str, Any]] = []
    for raw_code in codes:
        code = raw_code.strip()
        if not code:
            continue
        record = search_pubmed_by_code(code, email=email)
        results.append(
            {
                "code": code,
                "count": int(record.get("Count", 0)),
                "id_list": list(record.get("IdList", [])),
            }
        )
    return results
