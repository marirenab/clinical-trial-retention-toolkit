#!/usr/bin/env python3
"""Interactive Gradio app for retention-rate prediction."""

from __future__ import annotations

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


SECTION_HELP = {
    "study_design": "Core protocol choices like phase, allocation, and masking that shape how the trial is run.",
    "population": "Participant eligibility and demographic characteristics that influence how challenging retention may be.",
    "scale_complexity": "How large and operationally complex the trial is, including sites, arms, interventions, and conditions.",
    "baseline_measures": "Whether key baseline measurements are captured and how much baseline assessment is included.",
    "text_descriptions": "Free-text trial descriptions that provide richer context about condition, intervention, and arm structure.",
}


CSS = """
:root {
  --bg: #f8f9fa;
  --panel: #ffffff;
  --panel-soft: #f3f4f6;
  --border: #e5e7eb;
  --border-strong: #d1d5db;
  --text: #111827;
  --muted: #6b7280;
  --accent: #10b981;
  --accent-dark: #059669;
  --accent-soft: #d1fae5;
  --warn: #f59e0b;
  --danger: #ef4444;
  --shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
}

html, body, .gradio-container {
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto;
  padding: 28px 18px 40px 18px !important;
}

.hero {
  background: linear-gradient(135deg, #ffffff 0%, #effcf7 100%);
  border: 1px solid var(--border);
  border-radius: 28px;
  box-shadow: var(--shadow);
  padding: 28px 30px;
  margin-bottom: 22px;
}

.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent-dark);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.02em;
  margin-bottom: 14px;
}

.hero h1 {
  margin: 0;
  font-size: clamp(2rem, 4vw, 3.2rem);
  line-height: 1.05;
  letter-spacing: -0.03em;
}

.hero p {
  margin: 12px 0 0 0;
  max-width: 820px;
  color: var(--muted);
  font-size: 1.05rem;
  line-height: 1.6;
}

.app-shell {
  gap: 20px;
}

.input-panel,
.result-panel,
.plot-panel,
.reason-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 24px;
  box-shadow: var(--shadow);
}

.input-panel {
  padding: 18px;
  max-height: calc(100vh - 170px);
  overflow-y: auto;
}

.result-column {
  position: sticky;
  top: 20px;
  align-self: start;
}

.panel-title {
  font-size: 1rem;
  font-weight: 700;
  margin: 0 0 6px 0;
}

.panel-subtitle {
  color: var(--muted);
  font-size: 0.95rem;
  margin: 0 0 14px 0;
}

.section-help {
  color: var(--muted);
  font-size: 0.9rem;
  margin: 0 0 12px 0;
}

.accordion-wrap {
  border: 1px solid var(--border);
  border-radius: 18px;
  background: #fcfcfd;
  margin-bottom: 14px;
  overflow: hidden;
}

.accordion-wrap .label-wrap {
  font-weight: 700;
}

.input-panel .wrap,
.input-panel .block {
  border-radius: 16px !important;
}

.input-panel .block {
  border-color: var(--border) !important;
}

.input-panel textarea,
.input-panel input,
.input-panel .wrap textarea,
.input-panel .wrap input {
  font-size: 0.95rem !important;
}

.metric-card {
  padding: 22px;
}

.gauge-shell {
  display: flex;
  align-items: center;
  gap: 20px;
}

.gauge {
  --percent: 72;
  --gauge-color: var(--accent);
  width: 132px;
  height: 132px;
  border-radius: 50%;
  background: conic-gradient(var(--gauge-color) calc(var(--percent) * 1%), #e5e7eb 0);
  display: grid;
  place-items: center;
  flex: 0 0 auto;
}

.gauge::before {
  content: "";
  width: 94px;
  height: 94px;
  border-radius: 50%;
  background: white;
  box-shadow: inset 0 0 0 1px var(--border);
}

.gauge-center {
  position: absolute;
  text-align: center;
}

.gauge-value {
  font-size: 1.8rem;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.gauge-label {
  font-size: 0.8rem;
  color: var(--muted);
}

.metric-copy h3 {
  margin: 0;
  font-size: 1.1rem;
}

.metric-copy p {
  margin: 8px 0 0 0;
  color: var(--muted);
  line-height: 1.55;
}

.metric-copy .metric-stat {
  margin-top: 14px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: var(--panel-soft);
  color: var(--text);
  font-size: 0.92rem;
  font-weight: 600;
}

.plot-panel,
.reason-panel {
  padding: 16px;
  margin-top: 16px;
}

.reason-title {
  font-weight: 700;
  font-size: 1rem;
  margin-bottom: 12px;
}

.reason-row {
  display: grid;
  grid-template-columns: minmax(130px, 180px) 1fr 52px;
  gap: 12px;
  align-items: center;
  margin: 12px 0;
}

.reason-pill {
  padding: 10px 14px;
  border-radius: 999px;
  font-weight: 600;
  border: 1px solid rgba(16, 185, 129, 0.14);
  background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
}

.reason-bar-shell {
  height: 16px;
  border-radius: 999px;
  background: #edf0f2;
  overflow: hidden;
  position: relative;
}

.reason-bar-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.35s ease;
}

.reason-pct {
  text-align: right;
  font-weight: 700;
  color: var(--muted);
}

.button-row {
  display: grid;
  grid-template-columns: 1.25fr 1fr 1fr;
  gap: 12px;
  margin-top: 8px;
}

.button-row button {
  min-height: 52px;
  border-radius: 16px !important;
  font-weight: 700 !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}

.button-row button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(17, 24, 39, 0.12);
}

.button-primary button {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
  color: white !important;
  border: none !important;
}

.button-secondary button {
  background: white !important;
  color: var(--text) !important;
  border: 1px solid var(--border-strong) !important;
}

.button-tertiary button {
  background: #f3f4f6 !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}

.legend-note {
  color: var(--muted);
  font-size: 0.88rem;
  margin-top: 8px;
}

@media (max-width: 1100px) {
  .input-panel {
    max-height: none;
  }

  .result-column {
    position: static;
  }

  .button-row {
    grid-template-columns: 1fr;
  }

  .gauge-shell {
    flex-direction: column;
    align-items: flex-start;
  }
}
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


def make_input_frame(values: dict[str, object]) -> pd.DataFrame:
    row = {column: values.get(column, ui_metadata["defaults"].get(column, "")) for column in feature_columns}
    return pd.DataFrame([row])


def _retention_color(retention_rate: float) -> str:
    if retention_rate >= 0.8:
        return "#10b981"
    if retention_rate >= 0.6:
        return "#f59e0b"
    return "#ef4444"


def _retention_summary_html(retention_rate: float, dropout_rate: float) -> str:
    gauge_color = _retention_color(retention_rate)
    descriptor = "Strong projected retention" if retention_rate >= 0.8 else "Moderate projected retention" if retention_rate >= 0.6 else "Retention risk flagged"
    return f"""
    <div class="metric-card">
      <div class="gauge-shell">
        <div style="position:relative; display:grid; place-items:center;">
          <div class="gauge" style="--percent:{retention_rate*100:.1f}; --gauge-color:{gauge_color};"></div>
          <div class="gauge-center">
            <div class="gauge-value">{retention_rate:.0%}</div>
            <div class="gauge-label">Retention</div>
          </div>
        </div>
        <div class="metric-copy">
          <h3>{descriptor}</h3>
          <p>Estimated from the selected protocol design, population profile, baseline measures, and study description fields.</p>
          <div class="metric-stat">Estimated dropout: {dropout_rate:.1%}</div>
        </div>
      </div>
    </div>
    """


def predict_retention(*inputs):
    values = {name: value for name, value in zip(INPUT_ORDER, inputs)}
    frame = make_input_frame(values)
    dropout_rate = float(retention_model.predict(frame)[0])
    dropout_rate = min(max(dropout_rate, 0.0), 1.0)
    retention_rate = 1.0 - dropout_rate

    summary_html = _retention_summary_html(retention_rate, dropout_rate)
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
    colors = [
        "linear-gradient(90deg, #10b981 0%, #34d399 100%)",
        "linear-gradient(90deg, #22c55e 0%, #86efac 100%)",
        "linear-gradient(90deg, #14b8a6 0%, #5eead4 100%)",
        "linear-gradient(90deg, #f59e0b 0%, #fcd34d 100%)",
        "linear-gradient(90deg, #fb7185 0%, #fda4af 100%)",
        "linear-gradient(90deg, #8b5cf6 0%, #c4b5fd 100%)",
        "linear-gradient(90deg, #94a3b8 0%, #cbd5e1 100%)",
    ]

    rows = sorted(zip(labels, shares, colors), key=lambda item: item[1], reverse=True)
    bars = ["<div class='reason-title'>Estimated withdrawal-reason mix</div>"]
    for label, share, color in rows:
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
    bars.append("<div class='legend-note'>These shares reflect the model's estimated mix across recorded withdrawal-reason categories for a study with similar characteristics.</div>")
    return "".join(bars)


def build_explanation_plot(values: dict[str, object]):
    defaults = ui_metadata["defaults"]
    selected = make_input_frame(values)
    baseline = make_input_frame(defaults)
    selected_pred = float(retention_model.predict(selected)[0])
    _ = float(retention_model.predict(baseline)[0])

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
    colors = ["#10b981" if value >= 0 else "#ef4444" for value in deltas]

    fig, ax = plt.subplots(figsize=(7.2, 4.5), facecolor="white")
    ax.set_facecolor("white")
    ax.barh(labels, deltas, color=colors, edgecolor="none")
    ax.axvline(0, color="#9ca3af", linestyle="--", linewidth=1)
    ax.set_xlabel("Approximate effect on retention prediction", color="#374151")
    ax.set_title("What determined this prediction?", color="#111827", fontsize=13, pad=12)
    ax.tick_params(axis="x", colors="#4b5563")
    ax.tick_params(axis="y", colors="#111827")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def reset_to_defaults():
    defaults = ui_metadata["defaults"]
    summary = _retention_summary_html(1.0 - float(bundle["metrics"]["target_mean"]), float(bundle["metrics"]["target_mean"]))
    return [defaults.get(name, "") for name in INPUT_ORDER] + [summary, None, "<div class='reason-title'>Estimated withdrawal-reason mix</div><div class='legend-note'>Click the withdrawal reason button to generate the breakdown for the current configuration.</div>"]


def build_app() -> gr.Blocks:
    defaults = ui_metadata["defaults"]
    choices = ui_metadata["choices"]
    labels = ui_metadata["labels"]

    initial_summary = _retention_summary_html(1.0 - float(bundle["metrics"]["target_mean"]), float(bundle["metrics"]["target_mean"]))

    with gr.Blocks(title="Clinical Trial Retention Toolkit") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(
            """
            <div class="hero">
              <div class="hero-kicker"><span>🧪</span><span>Clinical trial planning support</span></div>
              <h1>Clinical Trial Retention Toolkit</h1>
              <p>Explore how protocol design, target population, study scale, and descriptive trial features may influence participant retention and withdrawal patterns before a study launches.</p>
            </div>
            """
        )

        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=7):
                with gr.Group(elem_classes=["input-panel"]):
                    gr.HTML("<div class='panel-title'>Trial configuration</div><div class='panel-subtitle'>Adjust the inputs below to simulate a proposed study design. Sections are grouped to keep the workflow easier to navigate.</div>")

                    with gr.Accordion("Study Design", open=True, elem_classes=["accordion-wrap"]):
                        gr.HTML(f"<div class='section-help' title='{SECTION_HELP['study_design']}'>{SECTION_HELP['study_design']}</div>")
                        study_type = gr.Dropdown(choices=choices["studyType"], value=defaults["studyType"], label=labels["studyType"], info="Overall study type and structure.")
                        phases = gr.Dropdown(choices=choices["phases"], value=defaults["phases"], label=labels["phases"], info="Trial phase or development stage.")
                        allocation = gr.Dropdown(choices=choices["allocation"], value=defaults["allocation"], label=labels["allocation"], info="How participants are assigned to interventions.")
                        intervention_model = gr.Dropdown(choices=choices["interventionModel"], value=defaults["interventionModel"], label=labels["interventionModel"], info="Parallel, crossover, sequential, or other intervention structure.")
                        primary_purpose = gr.Dropdown(choices=choices["primaryPurpose"], value=defaults["primaryPurpose"], label=labels["primaryPurpose"], info="Main purpose such as treatment, prevention, or supportive care.")
                        masking = gr.Dropdown(choices=choices["masking"], value=defaults["masking"], label=labels["masking"], info="Degree of blinding used in the trial.")
                        sponsor_class = gr.Dropdown(choices=choices["leadSponsorClass"], value=defaults["leadSponsorClass"], label=labels["leadSponsorClass"], info="Type of lead sponsor managing the study.")

                    with gr.Accordion("Population", open=False, elem_classes=["accordion-wrap"]):
                        gr.HTML(f"<div class='section-help' title='{SECTION_HELP['population']}'>{SECTION_HELP['population']}</div>")
                        sex = gr.Dropdown(choices=choices["sex"], value=defaults["sex"], label=labels["sex"], info="Eligibility by sex.")
                        healthy_volunteers = gr.Dropdown(choices=choices["healthyVolunteers"], value=defaults["healthyVolunteers"], label=labels["healthyVolunteers"], info="Whether healthy volunteers are eligible.")
                        min_age = gr.Number(value=defaults["minimumAgeYears"], label=labels["minimumAgeYears"], info="Minimum eligible age.")
                        max_age = gr.Number(value=defaults["maximumAgeYears"], label=labels["maximumAgeYears"], info="Maximum eligible age.")
                        mean_age = gr.Number(value=defaults["mean_age"], label=labels["mean_age"], info="Average participant age if known or targeted.")
                        std_age = gr.Number(value=defaults["std_age"], label=labels["std_age"], info="Spread in participant age.")
                        num_female = gr.Number(value=defaults["num_female"], label=labels["num_female"], info="Expected or observed female participant count.")
                        num_male = gr.Number(value=defaults["num_male"], label=labels["num_male"], info="Expected or observed male participant count.")

                    with gr.Accordion("Scale & Complexity", open=False, elem_classes=["accordion-wrap"]):
                        gr.HTML(f"<div class='section-help' title='{SECTION_HELP['scale_complexity']}'>{SECTION_HELP['scale_complexity']}</div>")
                        enrollment = gr.Number(value=defaults["enrollmentCount"], label=labels["enrollmentCount"], info="Target sample size.")
                        conditions = gr.Number(value=defaults["conditionCount"], label=labels["conditionCount"], info="Number of conditions linked to the trial.")
                        keywords = gr.Number(value=defaults["keywordCount"], label=labels["keywordCount"], info="Number of keywords used to describe the study.")
                        arm_groups = gr.Number(value=defaults["armGroupCount"], label=labels["armGroupCount"], info="Number of arms or groups in the design.")
                        interventions = gr.Number(value=defaults["interventionCount"], label=labels["interventionCount"], info="Number of interventions included.")
                        locations = gr.Number(value=defaults["locationCount"], label=labels["locationCount"], info="Number of recruiting locations or sites.")

                    with gr.Accordion("Baseline Measures", open=False, elem_classes=["accordion-wrap"]):
                        gr.HTML(f"<div class='section-help' title='{SECTION_HELP['baseline_measures']}'>{SECTION_HELP['baseline_measures']}</div>")
                        baseline_count = gr.Number(value=defaults["baseline_measure_count"], label=labels["baseline_measure_count"], info="Count of baseline measures captured.")
                        has_bmi = gr.Checkbox(value=bool(defaults["baseline_has_bmi"]), label=labels["baseline_has_bmi"], info="Includes BMI among baseline measures.")
                        has_weight = gr.Checkbox(value=bool(defaults["baseline_has_weight"]), label=labels["baseline_has_weight"], info="Includes weight among baseline measures.")
                        has_height = gr.Checkbox(value=bool(defaults["baseline_has_height"]), label=labels["baseline_has_height"], info="Includes height among baseline measures.")
                        has_dep_scale = gr.Checkbox(value=bool(defaults["baseline_has_depression_scale"]), label=labels["baseline_has_depression_scale"], info="Includes a depression-related scale at baseline.")

                    with gr.Accordion("Text Descriptions", open=False, elem_classes=["accordion-wrap"]):
                        gr.HTML(f"<div class='section-help' title='{SECTION_HELP['text_descriptions']}'>{SECTION_HELP['text_descriptions']}</div>")
                        condition_text = gr.Textbox(value=defaults["condition_text"], label=labels["condition_text"], lines=2, placeholder="e.g. major depressive disorder; bipolar disorder", info="List the study condition or diagnosis terms.")
                        keyword_text = gr.Textbox(value=defaults["keyword_text"], label=labels["keyword_text"], lines=2, placeholder="e.g. glutamate, relapse prevention, digital follow-up", info="Keywords or descriptors associated with the study.")
                        brief_summary = gr.Textbox(value=defaults["brief_summary_text"], label=labels["brief_summary_text"], lines=4, placeholder="Briefly describe the trial goal, intervention approach, and expected participant journey.", info="Free-text study summary used as model input.")
                        arm_group_labels = gr.Textbox(value=defaults["arm_group_labels_text"], label=labels["arm_group_labels_text"], lines=2, placeholder="e.g. treatment arm; placebo arm; follow-up arm", info="Names of trial arms or groups.")
                        intervention_names = gr.Textbox(value=defaults["intervention_names_text"], label=labels["intervention_names_text"], lines=2, placeholder="e.g. sertraline; CBT; mobile monitoring", info="Intervention names or components.")
                        intervention_types = gr.Textbox(value=defaults["intervention_type_text"], label=labels["intervention_type_text"], lines=2, placeholder="e.g. drug; behavioral; device; combination product", info="Intervention types or modality labels.")
                        group_title_0 = gr.Textbox(value=defaults["group_title_0"], label=labels["group_title_0"], lines=2, placeholder="Primary group title or arm title", info="First group title field used by the model.")
                        group_title_1 = gr.Textbox(value=defaults["group_title_1"], label=labels["group_title_1"], lines=2, placeholder="Secondary group title if applicable", info="Second group title field used by the model.")

                    with gr.Row(elem_classes=["button-row"]):
                        predict_btn = gr.Button("Predict retention rate", variant="primary", elem_classes=["button-primary"])
                        reasons_btn = gr.Button("Estimate withdrawal reasons", variant="secondary", elem_classes=["button-secondary"])
                        reset_btn = gr.Button("Reset to defaults", variant="secondary", elem_classes=["button-tertiary"])

            with gr.Column(scale=5, elem_classes=["result-column"]):
                with gr.Group(elem_classes=["result-panel"]):
                    retention_html = gr.HTML(initial_summary)
                with gr.Group(elem_classes=["plot-panel"]):
                    gr.HTML("<div class='panel-title'>What determined this prediction?</div><div class='panel-subtitle'>A local contrast against the default configuration to highlight the strongest feature shifts.</div>")
                    explanation_plot = gr.Plot(label=None, value=None, show_label=False)
                with gr.Group(elem_classes=["reason-panel"]):
                    reasons_html = gr.HTML("<div class='reason-title'>Estimated withdrawal-reason mix</div><div class='legend-note'>Click the withdrawal reason button to generate the breakdown for the current configuration.</div>")

        all_inputs = [
            study_type, phases, allocation, intervention_model, primary_purpose, masking, sex, healthy_volunteers,
            sponsor_class, enrollment, conditions, keywords, arm_groups, interventions, locations, min_age, max_age,
            mean_age, std_age, num_female, num_male, baseline_count, has_bmi, has_weight, has_height, has_dep_scale,
            condition_text, keyword_text, brief_summary, arm_group_labels, intervention_names, intervention_types,
            group_title_0, group_title_1,
        ]

        predict_btn.click(fn=predict_retention, inputs=all_inputs, outputs=[retention_html, explanation_plot], show_progress="full")
        reasons_btn.click(fn=predict_withdrawal_mix, inputs=all_inputs, outputs=reasons_html, show_progress="full")
        reset_btn.click(fn=reset_to_defaults, inputs=None, outputs=all_inputs + [retention_html, explanation_plot, reasons_html], show_progress="hidden")

    return demo


if __name__ == "__main__":
    build_app().launch()
