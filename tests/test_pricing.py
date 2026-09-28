"""Pricing validation uses quoted Decimal amounts."""

from decimal import Decimal

import pytest

from llm_policy_gateway.pricing import load_pricing


def test_pricing_loads_decimal_strings_for_all_models() -> None:
    pricing = load_pricing(
        "pricing.example.yaml", {"approved-local", "clinical-local", "billing-local"}
    )
    assert pricing.models["clinical-local"].input_per_1k_tokens == Decimal("0.02")
    assert isinstance(pricing.models["clinical-local"].output_per_1k_tokens, Decimal)
    assert pricing.models["approved-local"].input_per_1k_tokens == Decimal("0")


def test_missing_or_unquoted_price_is_rejected(tmp_path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        'models:\n  local:\n    input_per_1k_tokens: "0"\n'
        '    output_per_1k_tokens: "0"\n'
    )
    with pytest.raises(ValueError, match="Missing pricing"):
        load_pricing(path, {"local", "other"})
    path.write_text(
        "models:\n  local:\n    input_per_1k_tokens: 0.2\n"
        '    output_per_1k_tokens: "0"\n'
    )
    with pytest.raises(ValueError, match="quoted Decimal string"):
        load_pricing(path, {"local"})
