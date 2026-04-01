#!/usr/bin/env python3
"""Interactive Gradio app for retention-rate prediction."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import pandas as pd

BUNDLE_PATH = ROOT / "outputs/retention_model/retention_model.joblib"
UI_METADATA_PATH = ROOT / "outputs/retention_model/ui_metadata.json"


bundle = joblib.load(BUNDLE_PATH)
ui_metadata = json.loads(UI_METADATA_PATH.read_text())
retention_model = bundle["retention_model"]
reason_model = bundle["reason_model"]
feature_columns = bundle["feature_columns"]
reason_columns = bundle["reason_columns"]


def make_input_frame(values: dict[str, object]) -> pd.DataFrame:
    row = {column: values.get(column, ui_metadata["defaults"].get(column, "")) for column in feature_columns}
    return pd.DataFrame([row])


def predict_retention(*inputs):
    values = {name: value for name, value in zip(INPUT_ORDER, inputs)}
    frame = make_input_frame(values)
    dropout_rate = float(retention_model.predict(frame)[0])
    dropout_rate = min(max(dropout_rate, 0.0), 1.0)
    retention_rate = 1.0 - dropout_rate

    summary_html = f"""
    <div class="result-card green-card">
      <div class="card-label">Retention estimation</div>
      <div class="result-number">{retention_rate:.0%} Retention Rate</div>
      <div class="result-sub">Estimated dropout rate: {dropout_rate:.1%}</div>
    </div>
    """
    explanation_plot = build_explanation_plot(values)
    return summary_html, explanation_plot


def predict_withdrawal_mix(*inputs):
    values = {name: value for name, value in zip(INPUT_ORDER, inputs)}
    frame = make_input_frame(values)
    raw = reason_model.predict(frame)[0]
    raw = [max(float(value), 0.0) for value in raw]
    total = sum(raw)
    if total <= 0:
        shares = [0.0 for _ in raw]
    else:
        shares = [value / total for value in raw]

    labels = [
        "Lost to follow up",
        "Adverse event",
        "Withdrawal by subject",
        "Lack of efficacy",
        "Protocol violation",
        "Physician decision",
        "Other",
    ]
    colors = ["#b64516", "#ff7a45", "#ffd37a", "#c6e7a6", "#7fc8a9", "#7aa6ff", "#d6d6d6"]
    bars = []
    for label, share, color in zip(labels, shares, colors):
        bars.append(
            f"""
            <div class="reason-row">
              <div class="reason-pill">{label}</div>
              <div class="reason-bar-shell">
                <div class="reason-bar-fill" style="width:{share*100:.1f}%; background:{color};"></div>
              </div>
              <div class="reason-pct">{share:.0%}</div>
            </div>
            """
        )
    title = '<div class="result-card soft-card"><div class="card-label">Estimated withdrawal-reason mix</div></div>'
    return title + "".join(bars)


def build_explanation_plot(values: dict[str, object]):
    defaults = ui_metadata["defaults"]
    selected = make_input_frame(values)
    baseline = make_input_frame(defaults)
    selected_pred = float(retention_model.predict(selected)[0])
    base_pred = float(retention_model.predict(baseline)[0])

    candidates = [
        "studyType",
        "phases",
        "allocation",
        "interventionModel",
        "primaryPurpose",
        "masking",
        "enrollmentCount",
        "locationCount",
        "mean_age",
        "num_female",
        "num_male",
        "condition_text",
        "intervention_type_text",
    ]

    contributions = []
    for feature in candidates:
        modified = selected.copy()
        modified[feature] = defaults.get(feature, "")
        pred = float(retention_model.predict(modified)[0])
        contributions.append((feature, pred - selected_pred))

    top = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:6]
    labels = [ui_metadata["labels"].get(name, name) for name, _ in top]
    deltas = [-delta for _, delta in top]
    colors = ["#9ee6b3" if value >= 0 else "#ffb0a8" for value in deltas]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.barh(labels, deltas, color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Approximate effect on retention prediction")
    ax.set_title("What determined this retention rate?")
    fig.tight_layout()
    return fig


CSS = """
.gradio-container {font-family: Georgia, 'Times New Roman', serif; background: #f7f4ef;}
.main-title {font-size: 42px; line-height: 1.1; font-weight: 700; margin-bottom: 18px;}
.panel {border: 1.5px solid #bdb5a8; border-radius: 22px; padding: 18px; background: #fffdf9;}
.result-card {border-radius: 20px; padding: 16px 18px; border: 1.5px solid #909080; margin-bottom: 12px;}
.green-card {background: #dce9ce;}
.soft-card {background: #e4efd4;}
.card-label {font-size: 16px; margin-bottom: 6px;}
.result-number {font-size: 24px; font-weight: 700; text-align: center;}
.result-sub {font-size: 15px; text-align: center;}
.reason-row {display: grid; grid-template-columns: 220px 1fr 64px; gap: 14px; align-items: center; margin: 14px 0;}
.reason-pill {border: 1.5px solid #d5b8aa; border-radius: 999px; padding: 10px 16px; font-size: 18px; background: #fff;}
.reason-bar-shell {height: 40px; border: 1.5px solid #9a9a9a; border-radius: 8px; overflow: hidden; background: #fff;}
.reason-bar-fill {height: 100%;}
.reason-pct {font-size: 18px; font-weight: 700; text-align: right;}
.action-row {display:flex; gap:16px;}
"""


INPUT_ORDER = [
    "studyType",
    "phases",
    "allocation",
    "interventionModel",
    "primaryPurpose",
    "masking",
    "sex",
    "healthyVolunteers",
    "leadSponsorClass",
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
    "condition_text",
    "keyword_text",
    "brief_summary_text",
    "arm_group_labels_text",
    "intervention_names_text",
    "intervention_type_text",
    "group_title_0",
    "group_title_1",
]


def build_app() -> gr.Blocks:
    defaults = ui_metadata["defaults"]
    choices = ui_metadata["choices"]
    labels = ui_metadata["labels"]

    with gr.Blocks(title="Retention Design Tool") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group(elem_classes=["panel"]):
                    study_type = gr.Dropdown(choices=choices["studyType"], value=defaults["studyType"], label=labels["studyType"])
                    phases = gr.Dropdown(choices=choices["phases"], value=defaults["phases"], label=labels["phases"])
                    allocation = gr.Dropdown(choices=choices["allocation"], value=defaults["allocation"], label=labels["allocation"])
                    intervention_model = gr.Dropdown(choices=choices["interventionModel"], value=defaults["interventionModel"], label=labels["interventionModel"])
                    primary_purpose = gr.Dropdown(choices=choices["primaryPurpose"], value=defaults["primaryPurpose"], label=labels["primaryPurpose"])
                    masking = gr.Dropdown(choices=choices["masking"], value=defaults["masking"], label=labels["masking"])
                    sex = gr.Dropdown(choices=choices["sex"], value=defaults["sex"], label=labels["sex"])
                    healthy_volunteers = gr.Dropdown(choices=choices["healthyVolunteers"], value=defaults["healthyVolunteers"], label=labels["healthyVolunteers"])
                    sponsor_class = gr.Dropdown(choices=choices["leadSponsorClass"], value=defaults["leadSponsorClass"], label=labels["leadSponsorClass"])
                    enrollment = gr.Number(value=defaults["enrollmentCount"], label=labels["enrollmentCount"])
                    conditions = gr.Number(value=defaults["conditionCount"], label=labels["conditionCount"])
                    keywords = gr.Number(value=defaults["keywordCount"], label=labels["keywordCount"])
                    arm_groups = gr.Number(value=defaults["armGroupCount"], label=labels["armGroupCount"])
                    interventions = gr.Number(value=defaults["interventionCount"], label=labels["interventionCount"])
                    locations = gr.Number(value=defaults["locationCount"], label=labels["locationCount"])
                    min_age = gr.Number(value=defaults["minimumAgeYears"], label=labels["minimumAgeYears"])
                    max_age = gr.Number(value=defaults["maximumAgeYears"], label=labels["maximumAgeYears"])
                    mean_age = gr.Number(value=defaults["mean_age"], label=labels["mean_age"])
                    std_age = gr.Number(value=defaults["std_age"], label=labels["std_age"])
                    num_female = gr.Number(value=defaults["num_female"], label=labels["num_female"])
                    num_male = gr.Number(value=defaults["num_male"], label=labels["num_male"])
                    baseline_count = gr.Number(value=defaults["baseline_measure_count"], label=labels["baseline_measure_count"])
                    has_bmi = gr.Checkbox(value=bool(defaults["baseline_has_bmi"]), label=labels["baseline_has_bmi"])
                    has_weight = gr.Checkbox(value=bool(defaults["baseline_has_weight"]), label=labels["baseline_has_weight"])
                    has_height = gr.Checkbox(value=bool(defaults["baseline_has_height"]), label=labels["baseline_has_height"])
                    has_dep_scale = gr.Checkbox(value=bool(defaults["baseline_has_depression_scale"]), label=labels["baseline_has_depression_scale"])
                    condition_text = gr.Textbox(value=defaults["condition_text"], label=labels["condition_text"])
                    keyword_text = gr.Textbox(value=defaults["keyword_text"], label=labels["keyword_text"])
                    brief_summary = gr.Textbox(value=defaults["brief_summary_text"], label=labels["brief_summary_text"], lines=3)
                    arm_group_labels = gr.Textbox(value=defaults["arm_group_labels_text"], label=labels["arm_group_labels_text"])
                    intervention_names = gr.Textbox(value=defaults["intervention_names_text"], label=labels["intervention_names_text"])
                    intervention_types = gr.Textbox(value=defaults["intervention_type_text"], label=labels["intervention_type_text"])
                    group_title_0 = gr.Textbox(value=defaults["group_title_0"], label=labels["group_title_0"])
                    group_title_1 = gr.Textbox(value=defaults["group_title_1"], label=labels["group_title_1"])
                    with gr.Row(elem_classes=["action-row"]):
                        predict_btn = gr.Button("Predict retention rate", variant="primary")
                        reasons_btn = gr.Button("Calculate probabilities per withdrawal reason")

            with gr.Column(scale=6):
                retention_html = gr.HTML("<div class='result-card green-card'><div class='card-label'>Retention estimation</div><div class='result-number'>Ready</div></div>")
                explanation_plot = gr.Plot(label="What determined this retention rate?")
                reasons_html = gr.HTML("<div class='result-card soft-card'><div class='card-label'>Estimated withdrawal-reason mix</div></div>")

        all_inputs = [
            study_type, phases, allocation, intervention_model, primary_purpose, masking, sex, healthy_volunteers,
            sponsor_class, enrollment, conditions, keywords, arm_groups, interventions, locations, min_age, max_age,
            mean_age, std_age, num_female, num_male, baseline_count, has_bmi, has_weight, has_height, has_dep_scale,
            condition_text, keyword_text, brief_summary, arm_group_labels, intervention_names, intervention_types,
            group_title_0, group_title_1,
        ]

        predict_btn.click(fn=predict_retention, inputs=all_inputs, outputs=[retention_html, explanation_plot])
        reasons_btn.click(fn=predict_withdrawal_mix, inputs=all_inputs, outputs=reasons_html)

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    build_app().launch(server_name="127.0.0.1", server_port=port)
