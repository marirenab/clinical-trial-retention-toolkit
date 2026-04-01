"""Download helpers for ClinicalTrials.gov study exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_studies(
    *,
    query_cond: str,
    page_size: int = 1000,
) -> dict[str, Any]:
    """Download all pages for a ClinicalTrials.gov condition query."""
    studies: list[dict[str, Any]] = []
    next_page_token: str | None = None
    total_count: int | None = None

    while True:
        params = {
            "query.cond": query_cond,
            "pageSize": page_size,
            "format": "json",
        }
        if next_page_token:
            params["pageToken"] = next_page_token
        if total_count is None:
            params["countTotal"] = "true"

        payload = _fetch_json(f"{BASE_URL}?{urlencode(params)}")
        studies.extend(payload.get("studies", []))
        next_page_token = payload.get("nextPageToken")
        if total_count is None:
            total_count = payload.get("totalCount")
        if not next_page_token:
            break

    return {
        "query": {"query_cond": query_cond, "page_size": page_size},
        "totalCount": total_count if total_count is not None else len(studies),
        "downloadedCount": len(studies),
        "studies": studies,
    }


def save_study_download(payload: dict[str, Any], output_path: str | Path) -> None:
    """Persist a downloaded study payload to disk."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url) as response:  # noqa: S310 - official public API endpoint
        return json.loads(response.read().decode("utf-8"))
