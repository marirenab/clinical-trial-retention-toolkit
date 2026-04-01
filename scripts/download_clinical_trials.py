#!/usr/bin/env python3
"""Download a ClinicalTrials.gov dataset for this project."""

from __future__ import annotations

import argparse
from pathlib import Path

from trial_retention_toolkit.clinical_trials import build_study_summary_frame
from trial_retention_toolkit.download import fetch_studies, save_study_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="mental health")
    parser.add_argument(
        "--raw-output",
        default="data/raw/mental_health_studies.json",
    )
    parser.add_argument(
        "--summary-output",
        default="data/processed/mental_health_study_summary.csv",
    )
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()

    payload = fetch_studies(query_cond=args.condition, page_size=args.page_size)
    save_study_download(payload, args.raw_output)

    summary = build_study_summary_frame(payload["studies"])
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(
        f"Downloaded {payload['downloadedCount']} studies "
        f"for query '{args.condition}' to {args.raw_output}"
    )
    print(f"Wrote summary table to {args.summary_output}")


if __name__ == "__main__":
    main()
