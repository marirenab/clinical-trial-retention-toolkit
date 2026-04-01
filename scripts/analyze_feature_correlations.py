#!/usr/bin/env python3
"""Compute feature associations with dropout rate."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


LEAKY_COLUMNS = {
    "total_started",
    "not_completed_total",
    "completion_rate",
    "reason_count_total",
    "withdraw_reason_categories",
    "reason_lost_to_follow_up",
    "reason_adverse_event",
    "reason_withdrawal_by_subject",
    "reason_lack_of_efficacy",
    "reason_protocol_violation",
    "reason_physician_decision",
    "reason_other",
}


def main() -> None:
    data_path = ROOT / "data/processed/retention_modeling_dataset.csv"
    output_dir = ROOT / "outputs/correlations"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df[df["dropout_rate"].notna()].copy()
    df = df[(df["dropout_rate"] >= 0) & (df["dropout_rate"] <= 1)]

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "dropout_rate"]

    full_corr = compute_correlations(df, numeric_cols)
    clean_cols = [col for col in numeric_cols if col not in LEAKY_COLUMNS]
    clean_corr = compute_correlations(df, clean_cols)

    full_corr.to_csv(output_dir / "dropout_correlations_full.csv", index=False)
    clean_corr.to_csv(output_dir / "dropout_correlations_clean.csv", index=False)

    categorical_summary = summarize_categorical_associations(
        df,
        ["studyType", "phases", "allocation", "interventionModel", "primaryPurpose", "masking", "sex", "leadSponsorClass"],
    )
    categorical_summary.to_csv(output_dir / "dropout_by_category.csv", index=False)

    save_corr_plot(
        clean_corr,
        output_dir / "top_clean_correlations.png",
        title="Top Clean Feature Correlations With Dropout Rate",
    )
    save_corr_plot(
        full_corr,
        output_dir / "top_full_correlations.png",
        title="Top Full Feature Correlations With Dropout Rate",
    )

    print("Saved correlation outputs to", output_dir)


def compute_correlations(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    target = df["dropout_rate"]
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        pair = pd.DataFrame({"x": series, "y": target}).dropna()
        if len(pair) < 20:
            continue
        pearson_value, pearson_p = pearsonr(pair["x"], pair["y"])
        spearman_value, spearman_p = spearmanr(pair["x"], pair["y"])
        rows.append(
            {
                "feature": col,
                "n": len(pair),
                "pearson_r": pearson_value,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_value,
                "spearman_p": spearman_p,
                "abs_pearson_r": abs(pearson_value),
                "abs_spearman_rho": abs(spearman_value),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False)


def summarize_categorical_associations(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        subset = df[[col, "dropout_rate"]].dropna().copy()
        if subset.empty:
            continue
        grouped = subset.groupby(col)["dropout_rate"].agg(["mean", "count"]).reset_index()
        grouped = grouped[grouped["count"] >= 10].sort_values("mean", ascending=False)
        for _, row in grouped.head(15).iterrows():
            rows.append(
                {
                    "feature": col,
                    "level": row[col],
                    "mean_dropout_rate": row["mean"],
                    "count": int(row["count"]),
                }
            )
    return pd.DataFrame(rows)


def save_corr_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    plot_df = df.head(15).sort_values("pearson_r")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#bd632f" if value < 0 else "#2f6b7c" for value in plot_df["pearson_r"]]
    ax.barh(plot_df["feature"], plot_df["pearson_r"], color=colors)
    ax.set_title(title)
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
