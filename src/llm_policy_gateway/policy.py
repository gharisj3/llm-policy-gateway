"""Validated tenant policy loaded once during application startup."""

from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicySettings(StrictModel):
    allowed_models: list[str] = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    redaction_profile: Literal["off", "standard", "phi"]
    budget_usd_per_day: Decimal = Field(ge=0)
    rpm: int = Field(gt=0)
    tpm: int = Field(gt=0)
    injection_mode: Literal["off", "flag", "block"]
    reidentify_response: bool = False
    redaction_allow_terms: list[str] = Field(default_factory=list)


class PolicyOverride(StrictModel):
    allowed_models: list[str] | None = Field(default=None, min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    redaction_profile: Literal["off", "standard", "phi"] | None = None
    budget_usd_per_day: Decimal | None = Field(default=None, ge=0)
    rpm: int | None = Field(default=None, gt=0)
    tpm: int | None = Field(default=None, gt=0)
    injection_mode: Literal["off", "flag", "block"] | None = None
    reidentify_response: bool | None = None
    redaction_allow_terms: list[str] | None = None


class PolicyDenied(Exception):
    pass


class PolicyDocument(StrictModel):
    defaults: PolicySettings
    tenants: dict[str, PolicyOverride]

    def for_tenant(self, name: str) -> PolicySettings:
        override = self.tenants.get(name)
        if override is None:
            raise PolicyDenied(f"No policy configured for tenant {name!r}")
        values = self.defaults.model_dump()
        values.update(override.model_dump(exclude_unset=True))
        return PolicySettings.model_validate(values)


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: UniqueKeyLoader, node: MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _line_for_location(node: Node | None, location: tuple) -> int:
    current = node
    for part in location:
        if not isinstance(current, MappingNode):
            break
        match = next(
            (
                value
                for key, value in current.value
                if getattr(key, "value", None) == str(part)
            ),
            None,
        )
        if match is None:
            break
        current = match
    return current.start_mark.line + 1 if current else 1


def load_policy(path: str | Path) -> PolicyDocument:
    source = Path(path)
    try:
        contents = source.read_text(encoding="utf-8")
        raw = yaml.load(contents, Loader=UniqueKeyLoader)
        node = yaml.compose(contents)
    except OSError as error:
        raise ValueError(f"Cannot read policy {source}: {error}") from error
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        line = mark.line + 1 if mark else 1
        raise ValueError(f"Invalid policy {source}, line {line}: {error}") from error
    try:
        document = PolicyDocument.model_validate(raw)
    except ValidationError as error:
        first = error.errors()[0]
        line = _line_for_location(node, first["loc"])
        raise ValueError(
            f"Invalid policy {source}, line {line}: {first['msg']} at {first['loc']}"
        ) from error
    for name in document.tenants:
        try:
            document.for_tenant(name)
        except ValidationError as error:
            line = _line_for_location(node, ("tenants", name))
            raise ValueError(
                f"Invalid policy {source}, line {line}: "
                f"tenant {name!r} has an invalid override: {error.errors()[0]['msg']}"
            ) from error
    return document
