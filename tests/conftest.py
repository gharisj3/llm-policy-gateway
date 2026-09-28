import pytest


@pytest.fixture(autouse=True)
def key_pepper(monkeypatch) -> None:
    monkeypatch.setenv("KEY_PEPPER", "test-pepper-with-at-least-thirty-two-bytes")
