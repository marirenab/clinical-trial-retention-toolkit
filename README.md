# Mental Health Clinical Trial Retention

This project explores whether clinical-trial design and baseline study features can be used to estimate participant dropout and retention in mental health trials using ClinicalTrials.gov data.

The repository now includes:
- cleaned project structure
- reusable Python utilities under `src/`
- processed retention-modelling datasets
- exploratory figures
- saved machine-learning models for retention and withdrawal-reason estimation
- a Gradio app for interactive retention-rate prediction

## What Is In The Repo

- [`data/raw/`](/Users/meb22/clinical-trial-retention-toolkit/data/raw): raw JSON inputs, including the richer ClinicalTrials.gov export at [`ctg-studies.json`](/Users/meb22/clinical-trial-retention-toolkit/data/raw/ctg-studies.json)
- [`data/processed/`](/Users/meb22/clinical-trial-retention-toolkit/data/processed): derived CSV files, including the modelling table at [`retention_modeling_dataset.csv`](/Users/meb22/clinical-trial-retention-toolkit/data/processed/retention_modeling_dataset.csv)
- [`src/trial_retention_toolkit/`](/Users/meb22/clinical-trial-retention-toolkit/src/trial_retention_toolkit): reusable extraction, download, and modelling utilities
- [`scripts/`](/Users/meb22/clinical-trial-retention-toolkit/scripts): dataset-building, modelling, correlation, and figure-generation scripts
- [`notebooks/`](/Users/meb22/clinical-trial-retention-toolkit/notebooks): exploratory and cleaned notebooks
- [`outputs/`](/Users/meb22/clinical-trial-retention-toolkit/outputs): saved figures, metrics, predictions, and model artifacts
- [`app/`](/Users/meb22/clinical-trial-retention-toolkit/app): Gradio app for interactive prediction

## Main Data Sources

There are two distinct ClinicalTrials.gov datasets in this repo:

1. [`data/raw/mental_health_studies.json`](/Users/meb22/clinical-trial-retention-toolkit/data/raw/mental_health_studies.json)
This is a broad v2 API download of mental-health studies. It is useful for landscape analysis but does not directly contain the detailed participant-flow structure needed to compute retention rates.

2. [`data/raw/ctg-studies.json`](/Users/meb22/clinical-trial-retention-toolkit/data/raw/ctg-studies.json)
This is the richer JSON export used for the retention modelling work. It contains `resultsSection`, `participantFlowModule`, and `baselineCharacteristicsModule`, which allow dropout and retention to be derived from posted results.

## Main Outputs

### Modelling Dataset

The main modelling table is:

- [`retention_modeling_dataset.csv`](/Users/meb22/clinical-trial-retention-toolkit/data/processed/retention_modeling_dataset.csv)

This dataset includes:
- study design features such as `studyType`, `phases`, `allocation`, `interventionModel`, `primaryPurpose`, and `masking`
- eligibility and recruitment structure such as age bounds, sex, number of arms, interventions, and locations
- baseline reported participant information such as `mean_age`, `std_age`, `num_female`, and `num_male`
- text features such as condition text, summary text, intervention names, and arm labels
- derived outcome columns such as `dropout_rate` and `completion_rate`

### Saved Model Artifacts

The current saved model bundle is in:

- [`retention_model.joblib`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/retention_model.joblib)
- [`ui_metadata.json`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/ui_metadata.json)
- [`metrics.json`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/metrics.json)
- [`feature_importance.csv`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/feature_importance.csv)
- [`predictions.csv`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/predictions.csv)
- [`reason_model/reason_model_metrics.json`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/reason_model_metrics.json)
- [`reason_model/reason_model_predictions.csv`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/reason_model_predictions.csv)
- [`reason_model/reason_model_feature_importance.csv`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/reason_model_feature_importance.csv)

## Main Results

### Baseline Prediction Performance

Using the cleaned non-leaky model setup:
- rows used: `1881`
- train rows: `1504`
- test rows: `377`
- mean absolute error: `0.1299`
- R²: `0.0532`
- mean dropout rate in the sample: `0.1882`

Interpretation:
- the model can learn some signal, but prediction from clean pre-outcome features remains weak
- much stronger performance appeared only when post-outcome withdrawal-reason features were included, which was treated as leakage and removed from the final predictive setup

### Withdrawal-Reason Model Performance

A second model is also saved to estimate the mix of withdrawal reasons shown in the Gradio app. Its outputs are saved in [`outputs/retention_model/reason_model/`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model).

On the held-out test split:
- rows used: `377`
- overall mean MAE across reason categories: `0.0551`
- overall mean R² across reason categories: `-0.1177`

Per-category performance:
- `lost_to_follow_up`: MAE `0.0552`, R² `-0.1055`
- `adverse_event`: MAE `0.0286`, R² `-0.0027`
- `withdrawal_by_subject`: MAE `0.0457`, R² `-0.4045`
- `lack_of_efficacy`: MAE `0.0140`, R² `-0.0497`
- `protocol_violation`: MAE `0.0074`, R² `0.0373`
- `physician_decision`: MAE `0.0076`, R² `-0.2912`
- `other`: MAE `0.2270`, R² `-0.0073`

Interpretation:
- the reason model is useful as an exploratory UI estimate, but it is clearly weaker and less stable than the already modest retention model
- some categories are sparse, which likely limits predictive performance

### Main Exploratory Findings

From the saved category figures and correlation outputs:

- Interventional studies had higher mean dropout than observational studies.
- Phase 3 and Phase 4 studies showed higher mean dropout than Phase 1 studies.
- Non-randomized studies had higher mean dropout than randomized studies.
- Single-group intervention models had higher mean dropout than crossover designs.
- Treatment-focused trials tended to have higher mean dropout than basic-science studies.
- Trials with no masking tended to have higher dropout than single- or double-masked studies.

At the single-feature level, correlations with dropout were generally small. The strongest clean numeric associations were:
- `locationCount`: Pearson `0.104`
- `armGroupCount`: Pearson `-0.072`
- `maximumAgeYears`: Pearson `0.066`
- `enrollmentCount`: Pearson `-0.049`

This suggests that dropout is influenced by multiple interacting design and population factors rather than one dominant structured variable.

## Saved Figures

### Retention Figures

Saved in [`outputs/retention_figures/`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures):

### Reason-Model Figures

Saved in [`outputs/retention_model/reason_model/`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model):

- [`reason_model_metrics.json`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/reason_model_metrics.json)
- [`reason_model_predictions.csv`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/reason_model_predictions.csv)
- [`reason_model_feature_importance.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/reason_model_feature_importance.png)
- [`reason_model_feature_importance.csv`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/reason_model_feature_importance.csv)
- [`predicted_vs_actual_lost_to_follow_up.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/predicted_vs_actual_lost_to_follow_up.png)
- [`predicted_vs_actual_adverse_event.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/predicted_vs_actual_adverse_event.png)
- [`predicted_vs_actual_withdrawal_by_subject.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/predicted_vs_actual_withdrawal_by_subject.png)
- [`predicted_vs_actual_lack_of_efficacy.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/predicted_vs_actual_lack_of_efficacy.png)
- [`predicted_vs_actual_protocol_violation.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/predicted_vs_actual_protocol_violation.png)
- [`predicted_vs_actual_physician_decision.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/predicted_vs_actual_physician_decision.png)
- [`predicted_vs_actual_other.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/reason_model/predicted_vs_actual_other.png)

### Retention Figures

Saved in [`outputs/retention_figures/`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures):

- [`dropout_distribution.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/dropout_distribution.png)
- [`dropout_by_study_type.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/dropout_by_study_type.png)
- [`dropout_by_phase.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/dropout_by_phase.png)
- [`dropout_by_allocation.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/dropout_by_allocation.png)
- [`dropout_by_intervention_model.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/dropout_by_intervention_model.png)
- [`dropout_by_primary_purpose.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/dropout_by_primary_purpose.png)
- [`dropout_by_masking.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/dropout_by_masking.png)
- [`enrollment_vs_dropout.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/enrollment_vs_dropout.png)
- [`mean_age_vs_dropout.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/mean_age_vs_dropout.png)
- [`withdrawal_reason_totals.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/withdrawal_reason_totals.png)
- [`intervention_type_counts.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/intervention_type_counts.png)
- [`baseline_feature_coverage.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_figures/baseline_feature_coverage.png)

### Correlation Outputs

Saved in [`outputs/correlations/`](/Users/meb22/clinical-trial-retention-toolkit/outputs/correlations):

- [`dropout_correlations_clean.csv`](/Users/meb22/clinical-trial-retention-toolkit/outputs/correlations/dropout_correlations_clean.csv)
- [`dropout_correlations_full.csv`](/Users/meb22/clinical-trial-retention-toolkit/outputs/correlations/dropout_correlations_full.csv)
- [`dropout_by_category.csv`](/Users/meb22/clinical-trial-retention-toolkit/outputs/correlations/dropout_by_category.csv)
- [`top_clean_correlations.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/correlations/top_clean_correlations.png)
- [`top_full_correlations.png`](/Users/meb22/clinical-trial-retention-toolkit/outputs/correlations/top_full_correlations.png)

## Gradio App

An interactive trial-design predictor is available in:

- [`app/gradio_app.py`](/Users/meb22/clinical-trial-retention-toolkit/app/gradio_app.py)
- [`scripts/run_retention_app.py`](/Users/meb22/clinical-trial-retention-toolkit/scripts/run_retention_app.py)

The app includes:
- left-side trial design inputs
- retention estimate card
- simple feature-contribution chart
- predicted withdrawal-reason mix bars

Run locally with:

```bash
PYTHONPATH=src python3 scripts/run_retention_app.py
```

If port `7860` is already in use:

```bash
GRADIO_SERVER_PORT=7861 PYTHONPATH=src python3 scripts/run_retention_app.py
```

## Hugging Face Spaces

This project can be hosted as a Gradio Space. The important app files are:

- [`app/gradio_app.py`](/Users/meb22/clinical-trial-retention-toolkit/app/gradio_app.py)
- [`retention_model.joblib`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/retention_model.joblib)
- [`ui_metadata.json`](/Users/meb22/clinical-trial-retention-toolkit/outputs/retention_model/ui_metadata.json)

Recommended minimal Space files:
- `app.py`
- `requirements.txt`
- `app/`
- `src/`
- `outputs/retention_model/`

## Useful Scripts

Download the broad study dataset:

```bash
PYTHONPATH=src python3 scripts/download_clinical_trials.py
```

Build the retention modelling dataset from the richer JSON export:

```bash
PYTHONPATH=src python3 scripts/build_retention_dataset.py
```

Train and save the retention model:

```bash
PYTHONPATH=src python3 scripts/model_retention.py
```

Generate exploratory retention figures:

```bash
PYTHONPATH=src python3 scripts/generate_retention_figures.py
```

Run correlation analysis against dropout rate:

```bash
PYTHONPATH=src python3 scripts/analyze_feature_correlations.py
```

## Notebooks

- [`01_download_and_prepare.ipynb`](/Users/meb22/clinical-trial-retention-toolkit/notebooks/01_download_and_prepare.ipynb)
- [`02_study_landscape.ipynb`](/Users/meb22/clinical-trial-retention-toolkit/notebooks/02_study_landscape.ipynb)
- [`03_publication_linking.ipynb`](/Users/meb22/clinical-trial-retention-toolkit/notebooks/03_publication_linking.ipynb)
- [`04_retention_model.ipynb`](/Users/meb22/clinical-trial-retention-toolkit/notebooks/04_retention_model.ipynb)
- [`scraping_clinicaltrials_gov.ipynb`](/Users/meb22/clinical-trial-retention-toolkit/notebooks/scraping_clinicaltrials_gov.ipynb)
- [`publications_search.ipynb`](/Users/meb22/clinical-trial-retention-toolkit/notebooks/publications_search.ipynb)
- [`entity_extraction_api.ipynb`](/Users/meb22/clinical-trial-retention-toolkit/notebooks/entity_extraction_api.ipynb)

## Notes

- The clean predictive model excludes withdrawal-reason features because they leak post-outcome information.
- Some baseline features such as age and sex are still results-derived rather than pure protocol-only inputs.
