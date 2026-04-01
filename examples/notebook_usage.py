"""Minimal examples showing how the new helper modules fit the notebooks."""

from pathlib import Path

from trial_retention_toolkit import (
    build_entity_schema,
    build_results_dataframe,
    build_trial_metrics_frame,
    flatten_studies_to_csv,
    load_protocol_results,
    search_pubmed_for_codes,
    write_nct_ids,
)


ROOT = Path(__file__).resolve().parents[1]


def clinical_trials_example() -> None:
    flatten_studies_to_csv(
        ROOT / "data/raw/ctg-studies.json",
        ROOT / "data/interim/ctg-studies.csv",
    )
    protocol_results = load_protocol_results(ROOT / "data/interim/ctg-studies.csv")
    results_df = build_results_dataframe(protocol_results)
    metrics_df = build_trial_metrics_frame(results_df)
    write_nct_ids(results_df["nctId"].dropna().unique().tolist(), ROOT / "data/processed/nctids.txt")
    metrics_df.to_csv(ROOT / "data/processed/trial_metrics.csv", index=False)


def publications_example() -> None:
    nct_ids = (ROOT / "data/processed/nctids.txt").read_text().splitlines()
    pubmed_results = search_pubmed_for_codes(nct_ids)
    print(pubmed_results[:5])


def entity_schema_example() -> None:
    schema = build_entity_schema(["Depression", "Anxiety", "Placebo", "Statin"])
    print(schema)
