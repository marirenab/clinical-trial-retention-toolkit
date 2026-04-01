#!/usr/bin/env python3
"""Build a clean retention-modelling dataset from ctg-studies.json."""

from __future__ import annotations

import json
from pathlib import Path

from trial_retention_toolkit import build_retention_modeling_frame


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "data/raw/ctg-studies.json"
    output_path = root / "data/processed/retention_modeling_dataset.csv"

    studies = json.loads(input_path.read_text())
    df = build_retention_modeling_frame(studies)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    usable = df["dropout_rate"].notna().sum()
    print(f"Wrote {len(df)} rows to {output_path}")
    print(f"Rows with usable dropout_rate: {usable}")


if __name__ == "__main__":
    main()
