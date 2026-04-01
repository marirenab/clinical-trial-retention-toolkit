#!/usr/bin/env python3
"""Generate descriptive figures for the retention modelling dataset."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    data_path = ROOT / "data/processed/retention_modeling_dataset.csv"
    output_dir = ROOT / "outputs/retention_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df[df["dropout_rate"].notna()].copy()
    df = df[(df["dropout_rate"] >= 0) & (df["dropout_rate"] <= 1)]

    save_dropout_distribution(df, output_dir / "dropout_distribution.png")
    save_dropout_by_phase(df, output_dir / "dropout_by_phase.png")
    save_dropout_by_primary_purpose(df, output_dir / "dropout_by_primary_purpose.png")
    save_dropout_by_study_type(df, output_dir / "dropout_by_study_type.png")
    save_dropout_by_allocation(df, output_dir / "dropout_by_allocation.png")
    save_dropout_by_intervention_model(df, output_dir / "dropout_by_intervention_model.png")
    save_dropout_by_masking(df, output_dir / "dropout_by_masking.png")
    save_enrollment_vs_dropout(df, output_dir / "enrollment_vs_dropout.png")
    save_age_vs_dropout(df, output_dir / "mean_age_vs_dropout.png")
    save_reason_totals(df, output_dir / "withdrawal_reason_totals.png")
    save_intervention_types(df, output_dir / "intervention_type_counts.png")
    save_baseline_feature_coverage(df, output_dir / "baseline_feature_coverage.png")

    print(f"Saved retention figures to {output_dir}")


def save_dropout_distribution(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    df["dropout_rate"].plot(kind="hist", bins=30, ax=ax, color="#2f6b7c")
    ax.set_title("Distribution of Dropout Rate")
    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("Study count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_dropout_by_phase(df: pd.DataFrame, path: Path) -> None:
    plot_df = (
        df.dropna(subset=["phases"])
        .assign(phases=df["phases"].astype(str))
        .groupby("phases")["dropout_rate"]
        .mean()
        .sort_values()
        .tail(12)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="barh", ax=ax, color="#bd632f")
    ax.set_title("Average Dropout Rate by Phase")
    ax.set_xlabel("Average dropout rate")
    ax.set_ylabel("Phase")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_dropout_by_primary_purpose(df: pd.DataFrame, path: Path) -> None:
    plot_df = (
        df.dropna(subset=["primaryPurpose"])
        .groupby("primaryPurpose")["dropout_rate"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="barh", ax=ax, color="#4c8c4a")
    ax.set_title("Average Dropout Rate by Primary Purpose")
    ax.set_xlabel("Average dropout rate")
    ax.set_ylabel("Primary purpose")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_dropout_by_study_type(df: pd.DataFrame, path: Path) -> None:
    plot_df = (
        df.dropna(subset=["studyType"])
        .groupby("studyType")["dropout_rate"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="barh", ax=ax, color="#6b8e23")
    ax.set_title("Average Dropout Rate by Study Type")
    ax.set_xlabel("Average dropout rate")
    ax.set_ylabel("Study type")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_dropout_by_allocation(df: pd.DataFrame, path: Path) -> None:
    plot_df = (
        df.dropna(subset=["allocation"])
        .groupby("allocation")["dropout_rate"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="barh", ax=ax, color="#b56576")
    ax.set_title("Average Dropout Rate by Allocation")
    ax.set_xlabel("Average dropout rate")
    ax.set_ylabel("Allocation")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_dropout_by_intervention_model(df: pd.DataFrame, path: Path) -> None:
    plot_df = (
        df.dropna(subset=["interventionModel"])
        .groupby("interventionModel")["dropout_rate"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(kind="barh", ax=ax, color="#457b9d")
    ax.set_title("Average Dropout Rate by Intervention Model")
    ax.set_xlabel("Average dropout rate")
    ax.set_ylabel("Intervention model")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_dropout_by_masking(df: pd.DataFrame, path: Path) -> None:
    plot_df = (
        df.dropna(subset=["masking"])
        .groupby("masking")["dropout_rate"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="barh", ax=ax, color="#8d99ae")
    ax.set_title("Average Dropout Rate by Masking")
    ax.set_xlabel("Average dropout rate")
    ax.set_ylabel("Masking")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_enrollment_vs_dropout(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.dropna(subset=["enrollmentCount", "dropout_rate"]).copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(plot_df["enrollmentCount"], plot_df["dropout_rate"], alpha=0.4, color="#7a5ea6")
    ax.set_title("Enrollment vs Dropout Rate")
    ax.set_xlabel("Enrollment count")
    ax.set_ylabel("Dropout rate")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_age_vs_dropout(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.dropna(subset=["mean_age", "dropout_rate"]).copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(plot_df["mean_age"], plot_df["dropout_rate"], alpha=0.4, color="#c26a3d")
    ax.set_title("Mean Age vs Dropout Rate")
    ax.set_xlabel("Mean age")
    ax.set_ylabel("Dropout rate")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_reason_totals(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "reason_lost_to_follow_up",
        "reason_adverse_event",
        "reason_withdrawal_by_subject",
        "reason_lack_of_efficacy",
        "reason_protocol_violation",
        "reason_physician_decision",
        "reason_other",
    ]
    labels = [col.replace("reason_", "").replace("_", " ") for col in columns]
    totals = pd.Series({label: df[col].fillna(0).sum() for label, col in zip(labels, columns)}).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    totals.plot(kind="barh", ax=ax, color="#8a3f5d")
    ax.set_title("Withdrawal Reason Totals")
    ax.set_xlabel("Participants")
    ax.set_ylabel("Reason category")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_intervention_types(df: pd.DataFrame, path: Path) -> None:
    counts = (
        df["intervention_type_text"]
        .dropna()
        .astype(str)
        .str.split("; ")
        .explode()
        .value_counts()
        .head(12)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax, color="#3f7f6d")
    ax.set_title("Most Common Intervention Types")
    ax.set_xlabel("Study count")
    ax.set_ylabel("Intervention type")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_baseline_feature_coverage(df: pd.DataFrame, path: Path) -> None:
    coverage = pd.Series(
        {
            "mean_age": df["mean_age"].notna().mean(),
            "std_age": df["std_age"].notna().mean(),
            "num_female": df["num_female"].notna().mean(),
            "num_male": df["num_male"].notna().mean(),
            "bmi flag": df["baseline_has_bmi"].fillna(0).gt(0).mean(),
            "weight flag": df["baseline_has_weight"].fillna(0).gt(0).mean(),
            "height flag": df["baseline_has_height"].fillna(0).gt(0).mean(),
            "depression scale flag": df["baseline_has_depression_scale"].fillna(0).gt(0).mean(),
        }
    ).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    (coverage * 100).plot(kind="barh", ax=ax, color="#5b84c4")
    ax.set_title("Baseline Feature Coverage")
    ax.set_xlabel("Percent of studies with feature")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
