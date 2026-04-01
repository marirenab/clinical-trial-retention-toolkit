"""Helpers for working with ClinicalTrials.gov exports."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_ctg_json(path: str | Path) -> list[dict[str, Any]]:
    """Load a ClinicalTrials.gov JSON export."""
    return json.loads(Path(path).read_text())


def flatten_studies_to_csv(
    json_path: str | Path,
    csv_path: str | Path,
) -> pd.DataFrame:
    """Flatten a list of study dictionaries into a two-column key/value table."""
    studies = load_ctg_json(json_path)
    rows: list[dict[str, Any]] = []
    for study in studies:
        for key, value in study.items():
            rows.append({"Key": key, "Value": value})

    df = pd.DataFrame(rows)
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, quoting=csv.QUOTE_MINIMAL)
    return df


def load_protocol_results(csv_path: str | Path) -> pd.DataFrame:
    """Load only protocol and result sections from the flattened CSV export."""
    df = pd.read_csv(csv_path)
    return df[df["Key"].isin(["protocolSection", "resultsSection"])].copy()


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"Unsupported value type: {type(value)!r}")


def build_results_dataframe(protocol_results: pd.DataFrame) -> pd.DataFrame:
    """Pair each protocol section with its corresponding results section."""
    nct_ids: list[str] = []
    results: list[dict[str, Any]] = []

    for _, row in protocol_results.iterrows():
        value = _ensure_dict(row["Value"])
        if row["Key"] == "protocolSection":
            nct_id = value.get("identificationModule", {}).get("nctId")
            if nct_id:
                nct_ids.append(nct_id)
        elif row["Key"] == "resultsSection":
            results.append(value)

    pairs = min(len(nct_ids), len(results))
    return pd.DataFrame({"nctId": nct_ids[:pairs], "Results": results[:pairs]})


def build_study_summary_frame(studies: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a tidy study-level summary table from API study records."""
    rows = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        conditions = protocol.get("conditionsModule", {})
        design = protocol.get("designModule", {})
        sponsor = protocol.get("sponsorCollaboratorsModule", {})
        contacts = protocol.get("contactsLocationsModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        arms = protocol.get("armsInterventionsModule", {})

        location_list = contacts.get("locations", [])
        countries = sorted(
            {
                location.get("locationCountry")
                for location in location_list
                if location.get("locationCountry")
            }
        )
        interventions = [
            intervention.get("type")
            for intervention in arms.get("interventions", [])
            if intervention.get("type")
        ]

        rows.append(
            {
                "nctId": identification.get("nctId"),
                "briefTitle": identification.get("briefTitle"),
                "officialTitle": identification.get("officialTitle"),
                "overallStatus": status.get("overallStatus"),
                "studyType": design.get("studyType"),
                "phases": _join_values(design.get("phases")),
                "enrollmentCount": _to_float(
                    status.get("enrollmentInfo", {}).get("count")
                    or design.get("enrollmentInfo", {}).get("count")
                ),
                "startDate": status.get("startDateStruct", {}).get("date"),
                "primaryCompletionDate": status.get("primaryCompletionDateStruct", {}).get("date"),
                "completionDate": status.get("completionDateStruct", {}).get("date"),
                "leadSponsor": sponsor.get("leadSponsor", {}).get("name"),
                "hasResults": study.get("hasResults"),
                "conditions": _join_values(conditions.get("conditions")),
                "keywords": _join_values(conditions.get("keywords")),
                "sex": eligibility.get("sex"),
                "minimumAge": eligibility.get("minimumAge"),
                "maximumAge": eligibility.get("maximumAge"),
                "healthyVolunteers": eligibility.get("healthyVolunteers"),
                "locationCount": len(location_list),
                "countries": _join_values(countries),
                "interventionTypes": _join_values(interventions),
            }
        )

    return pd.DataFrame(rows)


def build_retention_modeling_frame(studies: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a tabular dataset for modelling dropout and retention."""
    rows = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        results = study.get("resultsSection", {})
        identification = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        conditions = protocol.get("conditionsModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        sponsor = protocol.get("sponsorCollaboratorsModule", {})
        arms = protocol.get("armsInterventionsModule", {})
        description = protocol.get("descriptionModule", {})

        metrics = extract_trial_metrics(results.get("participantFlowModule", {}))
        baseline = extract_baseline_characteristics(results)
        group_titles = extract_group_titles(results.get("participantFlowModule", {}))
        withdraw_reasons = extract_drop_withdraw_reasons(results)
        reason_category_counts = summarize_withdraw_reason_categories(results)
        baseline_summary = summarize_baseline_measures(results)

        arm_groups = arms.get("armGroups", []) or []
        interventions = arms.get("interventions", []) or []

        total_started = metrics.get("total_started")
        not_completed_total = metrics.get("not_completed_total")
        completion_rate = None
        if total_started not in (None, 0) and not_completed_total is not None:
            completion_rate = (total_started - not_completed_total) / total_started

        rows.append(
            {
                "nctId": identification.get("nctId"),
                "briefTitle": identification.get("briefTitle"),
                "overallStatus": status.get("overallStatus"),
                "studyType": design.get("studyType"),
                "phases": _join_values(design.get("phases")),
                "allocation": design.get("designInfo", {}).get("allocation"),
                "interventionModel": design.get("designInfo", {}).get("interventionModel"),
                "primaryPurpose": design.get("designInfo", {}).get("primaryPurpose"),
                "masking": design.get("designInfo", {}).get("maskingInfo", {}).get("masking"),
                "enrollmentCount": _to_float(
                    status.get("enrollmentInfo", {}).get("count")
                    or design.get("enrollmentInfo", {}).get("count")
                ),
                "conditionCount": len(conditions.get("conditions", []) or []),
                "keywordCount": len(conditions.get("keywords", []) or []),
                "armGroupCount": len(arms.get("armGroups", []) or []),
                "interventionCount": len(arms.get("interventions", []) or []),
                "locationCount": len(protocol.get("contactsLocationsModule", {}).get("locations", []) or []),
                "sex": eligibility.get("sex"),
                "minimumAgeYears": _parse_age_to_years(eligibility.get("minimumAge")),
                "maximumAgeYears": _parse_age_to_years(eligibility.get("maximumAge")),
                "healthyVolunteers": eligibility.get("healthyVolunteers"),
                "leadSponsorClass": sponsor.get("leadSponsor", {}).get("class"),
                "hasResults": study.get("hasResults"),
                "condition_text": _join_values(conditions.get("conditions")),
                "keyword_text": _join_values(conditions.get("keywords")),
                "brief_summary_text": description.get("briefSummary"),
                "detailed_description_text": description.get("detailedDescription"),
                "arm_group_labels_text": _join_values(
                    [arm.get("label") for arm in arm_groups if arm.get("label")]
                ),
                "arm_group_descriptions_text": _join_values(
                    [arm.get("description") for arm in arm_groups if arm.get("description")]
                ),
                "intervention_names_text": _join_values(
                    [intervention.get("name") for intervention in interventions if intervention.get("name")]
                ),
                "intervention_descriptions_text": _join_values(
                    [
                        intervention.get("description")
                        for intervention in interventions
                        if intervention.get("description")
                    ]
                ),
                "intervention_type_text": _join_values(
                    [intervention.get("type") for intervention in interventions if intervention.get("type")]
                ),
                "total_started": total_started,
                "not_completed_total": not_completed_total,
                "dropout_rate": (
                    not_completed_total / total_started
                    if total_started not in (None, 0) and not_completed_total is not None
                    else None
                ),
                "completion_rate": completion_rate,
                **baseline,
                **baseline_summary,
                "reason_count_total": sum(withdraw_reasons.values()) if withdraw_reasons else 0,
                "withdraw_reason_categories": len(withdraw_reasons),
                **reason_category_counts,
                "group_title_0": group_titles.get("FG000"),
                "group_title_1": group_titles.get("FG001"),
            }
        )
    return pd.DataFrame(rows)


def extract_trial_metrics(
    results_entry: dict[str, Any],
    max_groups: int = 14,
) -> dict[str, float | None]:
    """Extract participant counts by group and aggregate retention metrics."""
    metrics: dict[str, float | None] = {}
    for idx in range(max_groups):
        metrics[f"started_fg{idx:03d}"] = None
        metrics[f"not_completed_fg{idx:03d}"] = None

    flow = results_entry.get("participantFlowModule", results_entry)
    periods = flow.get("periods", [])
    for period in periods:
        for milestone in period.get("milestones", []):
            milestone_type = milestone.get("type", "")
            for achievement in milestone.get("achievements", []):
                group_id = achievement.get("groupId", "")
                value = achievement.get("numSubjects")
                if not group_id.startswith("FG"):
                    continue
                key_prefix = None
                if milestone_type == "STARTED":
                    key_prefix = "started"
                elif milestone_type == "NOT COMPLETED":
                    key_prefix = "not_completed"
                if key_prefix is not None:
                    metrics[f"{key_prefix}_{group_id.lower()}"] = value

    started_cols = [f"started_fg{idx:03d}" for idx in range(max_groups)]
    not_completed_cols = [f"not_completed_fg{idx:03d}" for idx in range(max_groups)]
    total_started = _sum_values(metrics[col] for col in started_cols)
    not_completed_total = _sum_values(metrics[col] for col in not_completed_cols)

    metrics["total_started"] = total_started
    metrics["not_completed_total"] = not_completed_total
    metrics["percentage"] = (
        not_completed_total / total_started
        if total_started not in (None, 0)
        else None
    )
    metrics["number_arms"] = _count_present_groups(metrics, started_cols)
    return metrics


def build_trial_metrics_frame(results_df: pd.DataFrame, max_groups: int = 14) -> pd.DataFrame:
    """Expand the results column into one row of numeric metrics per study."""
    extracted = results_df["Results"].apply(
        lambda entry: pd.Series(extract_trial_metrics(entry, max_groups=max_groups))
    )
    return pd.concat([results_df[["nctId"]].reset_index(drop=True), extracted], axis=1)


def extract_group_titles(results_entry: dict[str, Any], max_groups: int = 14) -> dict[str, str | None]:
    """Extract participant-flow group titles such as FG000 and FG001."""
    flow = results_entry.get("participantFlowModule", results_entry)
    groups = flow.get("groups", [])
    extracted = {f"FG{idx:03d}": None for idx in range(max_groups)}
    for group in groups:
        group_id = group.get("id", "")
        if group_id in extracted:
            extracted[group_id] = group.get("title")
    return extracted


def extract_drop_withdraw_reasons(results_entry: dict[str, Any]) -> dict[str, int]:
    """Extract withdrawal counts keyed by reason and group."""
    reasons: dict[str, int] = {}
    periods = results_entry.get("participantFlowModule", {}).get("periods", [])
    for period in periods:
        for withdrawal in period.get("dropWithdraws", []):
            reason_type = withdrawal.get("type", "unknown")
            for reason in withdrawal.get("reasons", []):
                group_id = reason.get("groupId", "unknown")
                key = f"{reason_type}_{group_id}"
                reasons[key] = reasons.get(key, 0) + int(reason.get("numSubjects", 0))
    return reasons


def extract_baseline_characteristics(results_entry: dict[str, Any]) -> dict[str, float | None]:
    """Extract a small set of baseline characteristics used in the notebook."""
    measures = results_entry.get("baselineCharacteristicsModule", {}).get("measures", [])
    summary = {
        "mean_age": None,
        "std_age": None,
        "num_female": None,
        "num_male": None,
    }

    for measure in measures:
        title = measure.get("title", "").lower()
        for class_info in measure.get("classes", []):
            for category in class_info.get("categories", []):
                for measurement in category.get("measurements", []):
                    group_id = measurement.get("groupId", "")
                    value = _to_float(measurement.get("value"))
                    spread = _to_float(measurement.get("spread"))
                    if title == "age, continuous" and group_id == "BG000":
                        summary["mean_age"] = value
                        summary["std_age"] = spread
                    if title == "sex: female, male":
                        title_lower = category.get("title", "").lower()
                        if title_lower == "female" and group_id == "BG000":
                            summary["num_female"] = value
                        if title_lower == "male" and group_id == "BG000":
                            summary["num_male"] = value
    return summary


def summarize_baseline_measures(results_entry: dict[str, Any]) -> dict[str, float | None]:
    """Summarize baseline measure richness and a few frequent numeric measures."""
    measures = results_entry.get("baselineCharacteristicsModule", {}).get("measures", [])
    summary: dict[str, float | None] = {
        "baseline_measure_count": len(measures),
        "baseline_has_bmi": 0,
        "baseline_has_weight": 0,
        "baseline_has_height": 0,
        "baseline_has_depression_scale": 0,
    }

    for measure in measures:
        title = (measure.get("title") or "").lower()
        if "body mass index" in title or "bmi" in title:
            summary["baseline_has_bmi"] = 1
        if "weight" in title:
            summary["baseline_has_weight"] = 1
        if "height" in title:
            summary["baseline_has_height"] = 1
        if "depression" in title or "madrs" in title or "phq" in title or "ham-d" in title:
            summary["baseline_has_depression_scale"] = 1
    return summary


def summarize_withdraw_reason_categories(results_entry: dict[str, Any]) -> dict[str, int]:
    """Collapse detailed withdrawal reasons into a small set of modelling features."""
    categories = {
        "reason_lost_to_follow_up": 0,
        "reason_adverse_event": 0,
        "reason_withdrawal_by_subject": 0,
        "reason_lack_of_efficacy": 0,
        "reason_protocol_violation": 0,
        "reason_physician_decision": 0,
        "reason_other": 0,
    }

    periods = results_entry.get("participantFlowModule", {}).get("periods", [])
    for period in periods:
        for withdrawal in period.get("dropWithdraws", []):
            label = (withdrawal.get("type") or "").lower()
            total = sum(int(reason.get("numSubjects", 0)) for reason in withdrawal.get("reasons", []))
            matched = False
            if "lost to follow" in label:
                categories["reason_lost_to_follow_up"] += total
                matched = True
            if "adverse" in label:
                categories["reason_adverse_event"] += total
                matched = True
            if "withdrawal by subject" in label or "withdrawal of consent" in label:
                categories["reason_withdrawal_by_subject"] += total
                matched = True
            if "lack of efficacy" in label:
                categories["reason_lack_of_efficacy"] += total
                matched = True
            if "protocol violation" in label:
                categories["reason_protocol_violation"] += total
                matched = True
            if "physician decision" in label or "investigator decision" in label:
                categories["reason_physician_decision"] += total
                matched = True
            if not matched:
                categories["reason_other"] += total
    return categories


def write_nct_ids(nct_ids: list[str], path: str | Path) -> None:
    """Write one trial identifier per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(nct_ids) + ("\n" if nct_ids else ""))


def _sum_values(values: Any) -> float:
    total = 0.0
    for value in values:
        number = _to_float(value)
        if number is not None:
            total += number
    return total


def _count_present_groups(metrics: dict[str, Any], started_cols: list[str]) -> int:
    count = 0
    for col in started_cols:
        if _to_float(metrics.get(col)) is not None:
            count += 1
    return count


def _to_float(value: Any) -> float | None:
    if value in (None, "", "NA", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_age_to_years(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split()
    if not parts:
        return None
    number = _to_float(parts[0])
    if number is None:
        return None
    unit = parts[1].lower() if len(parts) > 1 else "years"
    if unit.startswith("year"):
        return number
    if unit.startswith("month"):
        return number / 12.0
    if unit.startswith("week"):
        return number / 52.0
    if unit.startswith("day"):
        return number / 365.0
    return number


def _join_values(values: Any) -> str | None:
    if not values:
        return None
    if isinstance(values, str):
        return values
    return "; ".join(str(value) for value in values if value is not None)
