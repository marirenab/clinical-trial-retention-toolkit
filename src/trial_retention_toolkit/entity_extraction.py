"""Helpers for building Pydantic schemas for notebook-based entity extraction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model


def build_entity_model(
    terms: list[str],
    *,
    include_age: bool = True,
    include_gender: bool = True,
) -> type[BaseModel]:
    """Create a lightweight entity schema from a list of binary terms."""
    fields: dict[str, tuple[object, object]] = {}

    if include_age:
        fields["age"] = (
            int | None,
            Field(default=None, description="Age or mean age when present."),
        )
    if include_gender:
        fields["gender"] = (
            Literal["Male", "Female"] | None,
            Field(default=None, description="Gender if explicitly stated."),
        )

    for term in terms:
        normalized = _normalize_field_name(term)
        fields[normalized] = (
            bool | None,
            Field(default=None, description=f"Whether '{term}' is mentioned."),
        )

    return create_model("NotebookEntities", **fields)


def build_entity_schema(terms: list[str]) -> dict[str, object]:
    """Return the JSON schema for the generated entity model."""
    model = build_entity_model(terms)
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


def _normalize_field_name(value: str) -> str:
    cleaned = value.strip().replace(" ", "_").replace("-", "_").replace("/", "_")
    return "".join(char for char in cleaned if char.isalnum() or char == "_")
