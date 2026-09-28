from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_policy_gateway.app import create_app
from llm_policy_gateway.policy import PolicyDenied, load_policy


def test_policy_inherits_defaults_and_preserves_decimal(tmp_path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        "defaults:\n"
        "  allowed_models: [local]\n"
        "  max_tokens: 100\n"
        "  redaction_profile: standard\n"
        '  budget_usd_per_day: "1.25"\n'
        "  rpm: 10\n"
        "  tpm: 1000\n"
        "  injection_mode: flag\n"
        "  reidentify_response: true\n"
        "  redaction_allow_terms: [Example Clinic]\n"
        "tenants:\n"
        "  alpha:\n"
        "    max_tokens: 50\n",
        encoding="utf-8",
    )
    policy = load_policy(path).for_tenant("alpha")

    assert policy.max_tokens == 50
    assert policy.rpm == 10
    assert policy.budget_usd_per_day == Decimal("1.25")
    assert policy.reidentify_response is True
    assert policy.redaction_allow_terms == ["Example Clinic"]


def test_policy_denies_tenant_without_entry() -> None:
    policy = load_policy("policy.example.yaml")
    with pytest.raises(PolicyDenied):
        policy.for_tenant("unlisted")


def test_unknown_field_reports_line_and_stops_startup(tmp_path) -> None:
    source = Path("policy.example.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(
        source.replace(
            "  max_tokens: 1024\n", "  max_tokens: 1024\n  surprise: true\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"line \d+: Extra inputs are not permitted"):
        with TestClient(create_app(path)):
            pass


def test_duplicate_yaml_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("defaults: {}\ndefaults: {}\ntenants: {}\n")
    with pytest.raises(ValueError, match="duplicate key"):
        load_policy(path)


def test_null_override_stops_startup(tmp_path) -> None:
    source = Path("policy.example.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(
        source.replace("    max_tokens: 512\n", "    max_tokens: null\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"line \d+: tenant 'clinical-team'"):
        with TestClient(create_app(path)):
            pass


def test_missing_default_policy_stops_startup(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEFAULT_POLICY_PATH")
    with pytest.raises(ValueError, match=r"Cannot read policy policy.yaml"):
        with TestClient(create_app()):
            pass
