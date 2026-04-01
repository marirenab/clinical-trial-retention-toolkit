#!/usr/bin/env python3
"""Train retention and withdrawal-reason models and save outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from trial_retention_toolkit.modeling import (
    FEATURE_COLUMNS,
    build_model_pipeline,
    build_ui_metadata,
)


def main() -> None:
    data_path = ROOT / "data/processed/retention_modeling_dataset.csv"
    output_dir = ROOT / "outputs/retention_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    model_df = df[df["dropout_rate"].notna()].copy()
    model_df = model_df[(model_df["dropout_rate"] >= 0) & (model_df["dropout_rate"] <= 1)]
    X = model_df[FEATURE_COLUMNS]
    y = model_df["dropout_rate"]
    retention_model = build_model_pipeline()

    reason_columns = [
        "reason_lost_to_follow_up",
        "reason_adverse_event",
        "reason_withdrawal_by_subject",
        "reason_lack_of_efficacy",
        "reason_protocol_violation",
        "reason_physician_decision",
        "reason_other",
    ]
    reason_df = model_df.copy()
    for column in reason_columns:
        reason_df[f"{column}_rate"] = (
            pd.to_numeric(reason_df[column], errors="coerce").fillna(0)
            / pd.to_numeric(reason_df["total_started"], errors="coerce").replace(0, np.nan)
        ).fillna(0)
    reason_rate_columns = [f"{column}_rate" for column in reason_columns]
    y_reason = reason_df[reason_rate_columns]
    reason_model = build_model_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_reason_train, X_reason_test, y_reason_train, y_reason_test = train_test_split(
        X, y_reason, test_size=0.2, random_state=42
    )

    retention_model.fit(X_train, y_train)
    dropout_predictions = retention_model.predict(X_test)

    reason_model.fit(X_reason_train, y_reason_train)
    reason_predictions = pd.DataFrame(
        reason_model.predict(X_reason_test),
        columns=reason_rate_columns,
        index=X_reason_test.index,
    )

    retention_metrics = {
        "rows_used": int(len(model_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, dropout_predictions)),
        "r2": float(r2_score(y_test, dropout_predictions)),
        "target_mean": float(y.mean()),
    }
    (output_dir / "metrics.json").write_text(json.dumps(retention_metrics, indent=2))

    dropout_comparison = pd.DataFrame({"actual": y_test, "predicted": dropout_predictions})
    dropout_comparison.to_csv(output_dir / "predictions.csv", index=False)
    save_prediction_scatter(
        actual=y_test,
        predicted=pd.Series(dropout_predictions, index=y_test.index),
        title="Predicted vs Actual Dropout Rate",
        path=output_dir / "predicted_vs_actual_dropout.png",
        color="#2f6b7c",
        xlabel="Actual dropout rate",
        ylabel="Predicted dropout rate",
    )
    save_feature_importance(
        retention_model,
        output_dir / "feature_importance.png",
        output_dir / "feature_importance.csv",
    )

    save_reason_model_outputs(
        output_dir=output_dir,
        reason_model=reason_model,
        y_reason_test=y_reason_test,
        reason_predictions=reason_predictions,
        reason_columns=reason_columns,
        reason_rate_columns=reason_rate_columns,
    )

    ui_metadata = build_ui_metadata(model_df)
    (output_dir / "ui_metadata.json").write_text(json.dumps(ui_metadata, indent=2))
    joblib.dump(
        {
            "retention_model": retention_model,
            "reason_model": reason_model,
            "feature_columns": FEATURE_COLUMNS,
            "reason_columns": reason_columns,
            "metrics": retention_metrics,
        },
        output_dir / "retention_model.joblib",
    )

    print(json.dumps(retention_metrics, indent=2))


def save_reason_model_outputs(
    output_dir: Path,
    reason_model,
    y_reason_test: pd.DataFrame,
    reason_predictions: pd.DataFrame,
    reason_columns: list[str],
    reason_rate_columns: list[str],
) -> None:
    reason_output_dir = output_dir / "reason_model"
    reason_output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "rows_used": int(len(y_reason_test)),
        "overall_mae_mean": float(
            mean_absolute_error(y_reason_test, reason_predictions, multioutput="uniform_average")
        ),
        "overall_r2_mean": float(
            r2_score(y_reason_test, reason_predictions, multioutput="uniform_average")
        ),
        "per_reason": {},
    }

    prediction_rows = []
    for rate_column, raw_column in zip(reason_rate_columns, reason_columns):
        clean_name = raw_column.replace("reason_", "")
        actual = y_reason_test[rate_column]
        predicted = reason_predictions[rate_column]
        metrics["per_reason"][clean_name] = {
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)),
        }
        prediction_rows.append(
            pd.DataFrame(
                {
                    "row_index": actual.index,
                    "reason": clean_name,
                    "actual": actual.values,
                    "predicted": predicted.values,
                }
            )
        )
        save_prediction_scatter(
            actual=actual,
            predicted=predicted,
            title=f"Predicted vs Actual {clean_name.replace('_', ' ').title()} Rate",
            path=reason_output_dir / f"predicted_vs_actual_{clean_name}.png",
            color="#8a3f5d",
            xlabel="Actual rate",
            ylabel="Predicted rate",
        )

    (reason_output_dir / "reason_model_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        reason_output_dir / "reason_model_predictions.csv",
        index=False,
    )
    save_feature_importance(
        reason_model,
        reason_output_dir / "reason_model_feature_importance.png",
        reason_output_dir / "reason_model_feature_importance.csv",
    )


def save_prediction_scatter(
    actual: pd.Series,
    predicted: pd.Series,
    title: str,
    path: Path,
    color: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.6, color=color)
    max_value = max(float(pd.Series(actual).max()), float(pd.Series(predicted).max()), 1e-6)
    ax.plot([0, max_value], [0, max_value], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_feature_importance(
    model,
    figure_path: Path,
    table_path: Path,
) -> None:
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = pd.Series(regressor.feature_importances_, index=feature_names).sort_values(ascending=False)
    importances.head(20).to_csv(table_path, header=["importance"])

    fig, ax = plt.subplots(figsize=(10, 6))
    importances.head(15).sort_values().plot(kind="barh", ax=ax, color="#bd632f")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Encoded feature")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
