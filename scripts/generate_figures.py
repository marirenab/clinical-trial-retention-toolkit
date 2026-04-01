#!/usr/bin/env python3
"""Generate summary figures from the downloaded ClinicalTrials.gov dataset."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    data_path = ROOT / "data/processed/mental_health_study_summary.csv"
    output_dir = ROOT / "outputs/figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    save_status_counts(df, output_dir / "study_status_counts.png")
    save_study_type_counts(df, output_dir / "study_type_counts.png")
    save_results_rate_by_type(df, output_dir / "results_rate_by_study_type.png")
    save_top_countries(df, output_dir / "top_countries.png")

    print(f"Saved figures to {output_dir}")


def save_status_counts(df: pd.DataFrame, path: Path) -> None:
    counts = df["overallStatus"].fillna("Unknown").value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", color="#2f6b7c", ax=ax)
    ax.set_title("Most Common Study Status Values")
    ax.set_xlabel("Study count")
    ax.set_ylabel("Status")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_study_type_counts(df: pd.DataFrame, path: Path) -> None:
    counts = df["studyType"].fillna("Unknown").value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="barh", color="#bd632f", ax=ax)
    ax.set_title("Study Types")
    ax.set_xlabel("Study count")
    ax.set_ylabel("Type")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_results_rate_by_type(df: pd.DataFrame, path: Path) -> None:
    rates = (
        df.assign(hasResults=df["hasResults"].fillna(False).astype(bool))
        .groupby("studyType", dropna=False)["hasResults"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    (rates * 100).plot(kind="bar", color="#4c8c4a", ax=ax)
    ax.set_title("Studies With Posted Results by Study Type")
    ax.set_xlabel("Study type")
    ax.set_ylabel("Percent with results")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_top_countries(df: pd.DataFrame, path: Path) -> None:
    counts = (
        df["countries"]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.lower() != "nan"]
        .str.split("; ")
        .explode()
        .value_counts()
        .head(12)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    if counts.empty:
        ax.text(0.5, 0.5, "No country data available in current summary export", ha="center", va="center")
        ax.set_title("Top Countries in Mental Health Studies")
        ax.set_axis_off()
    else:
        counts.plot(kind="barh", color="#7a5ea6", ax=ax)
        ax.set_title("Top Countries in Mental Health Studies")
        ax.set_xlabel("Study count")
        ax.set_ylabel("Country")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
