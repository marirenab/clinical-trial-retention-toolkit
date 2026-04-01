"""Shared modelling utilities for retention prediction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "enrollmentCount",
    "conditionCount",
    "keywordCount",
    "armGroupCount",
    "interventionCount",
    "locationCount",
    "minimumAgeYears",
    "maximumAgeYears",
    "mean_age",
    "std_age",
    "num_female",
    "num_male",
    "baseline_measure_count",
    "baseline_has_bmi",
    "baseline_has_weight",
    "baseline_has_height",
    "baseline_has_depression_scale",
]

CATEGORICAL_FEATURES = [
    "studyType",
    "phases",
    "allocation",
    "interventionModel",
    "primaryPurpose",
    "masking",
    "sex",
    "healthyVolunteers",
    "leadSponsorClass",
]

TEXT_FEATURES = [
    "condition_text",
    "keyword_text",
    "brief_summary_text",
    "arm_group_labels_text",
    "intervention_names_text",
    "intervention_type_text",
    "group_title_0",
    "group_title_1",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + TEXT_FEATURES


class TextColumnFlattener:
    """Turn a single-column 2D array into a flat string sequence for TF-IDF."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        series = pd.Series(X.ravel()).fillna("").astype(str)
        return series.tolist()

    def get_feature_names_out(self, input_features=None):
        return input_features


def build_model_pipeline() -> Pipeline:
    """Create the preprocessing + regressor pipeline."""
    transformers = [
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            CATEGORICAL_FEATURES,
        ),
    ]
    for column in TEXT_FEATURES:
        transformers.append(
            (
                f"text_{column}",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("flatten", TextColumnFlattener()),
                        ("tfidf", TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
                    ]
                ),
                [column],
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )


def build_ui_metadata(df: pd.DataFrame) -> dict[str, object]:
    """Create defaults and categorical choices for the interactive app."""
    clean_df = df.copy()
    defaults: dict[str, object] = {}
    choices: dict[str, list[str]] = {}

    for column in NUMERIC_FEATURES:
        series = pd.to_numeric(clean_df[column], errors="coerce")
        defaults[column] = float(series.median()) if series.notna().any() else 0.0

    for column in CATEGORICAL_FEATURES:
        series = clean_df[column].dropna().astype(str)
        values = sorted(series.unique().tolist())
        choices[column] = values
        defaults[column] = series.mode().iat[0] if not series.empty else ""

    for column in TEXT_FEATURES:
        series = clean_df[column].dropna().astype(str)
        defaults[column] = series.mode().iat[0] if not series.empty else ""

    labels = {
        "studyType": "Study Type",
        "phases": "Phase",
        "allocation": "Allocation",
        "interventionModel": "Intervention Model",
        "primaryPurpose": "Primary Purpose",
        "masking": "Masking",
        "sex": "Eligible Sex",
        "healthyVolunteers": "Healthy Volunteers",
        "leadSponsorClass": "Lead Sponsor Class",
        "enrollmentCount": "Enrollment Count",
        "conditionCount": "Condition Count",
        "keywordCount": "Keyword Count",
        "armGroupCount": "Arm Group Count",
        "interventionCount": "Intervention Count",
        "locationCount": "Location Count",
        "minimumAgeYears": "Minimum Age (Years)",
        "maximumAgeYears": "Maximum Age (Years)",
        "mean_age": "Mean Age",
        "std_age": "Age SD",
        "num_female": "Female Participants",
        "num_male": "Male Participants",
        "baseline_measure_count": "Baseline Measure Count",
        "baseline_has_bmi": "Has BMI Measure",
        "baseline_has_weight": "Has Weight Measure",
        "baseline_has_height": "Has Height Measure",
        "baseline_has_depression_scale": "Has Depression Scale",
        "condition_text": "Condition Text",
        "keyword_text": "Keyword Text",
        "brief_summary_text": "Brief Summary",
        "arm_group_labels_text": "Arm Group Labels",
        "intervention_names_text": "Intervention Names",
        "intervention_type_text": "Intervention Types",
        "group_title_0": "Primary Group Title",
        "group_title_1": "Second Group Title",
    }

    return {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "text_features": TEXT_FEATURES,
        "defaults": defaults,
        "choices": choices,
        "labels": labels,
    }
