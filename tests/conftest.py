from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def key_pepper(monkeypatch) -> None:
    monkeypatch.setenv("KEY_PEPPER", "test-pepper-with-at-least-thirty-two-bytes")
    monkeypatch.setenv(
        "DEFAULT_POLICY_PATH",
        str(Path(__file__).resolve().parents[1] / "policy.example.yaml"),
    )
    monkeypatch.setenv(
        "MODELS_PATH",
        str(Path(__file__).resolve().parents[1] / "models.example.yaml"),
    )
    monkeypatch.setenv("PRESIDIO_NLP_MODEL", "en_core_web_sm")
    monkeypatch.setenv(
        "PRICING_PATH",
        str(Path(__file__).resolve().parents[1] / "pricing.example.yaml"),
    )
