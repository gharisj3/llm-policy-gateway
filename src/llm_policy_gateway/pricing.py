"""Validated Decimal prices for every public model route."""

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from llm_policy_gateway.policy import UniqueKeyLoader


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_per_1k_tokens: Decimal
    output_per_1k_tokens: Decimal

    @field_validator("input_per_1k_tokens", "output_per_1k_tokens", mode="before")
    @classmethod
    def decimal_string(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("price must be a quoted Decimal string")
        return value

    @field_validator("input_per_1k_tokens", "output_per_1k_tokens")
    @classmethod
    def nonnegative_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value < 0:
            raise ValueError("price must be nonnegative and finite")
        return value


class PricingDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: dict[str, ModelPrice]


def load_pricing(path: str | Path, routed_models: set[str]) -> PricingDocument:
    source = Path(path)
    try:
        raw = yaml.load(source.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        pricing = PricingDocument.model_validate(raw)
    except (OSError, yaml.YAMLError, ValueError) as error:
        raise ValueError(f"Invalid pricing {source}: {error}") from error
    missing = sorted(routed_models - pricing.models.keys())
    if missing:
        raise ValueError(f"Missing pricing for routed models: {', '.join(missing)}")
    return pricing
