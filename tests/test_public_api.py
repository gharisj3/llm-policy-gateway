import json
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from openai import OpenAI

from llm_policy_gateway.app import create_app
from llm_policy_gateway.db import make_engine, make_session_factory
from llm_policy_gateway.keys import create_tenant, issue_key
from llm_policy_gateway.providers import MockProvider


@pytest.fixture
def gateway(tmp_path, monkeypatch) -> Iterator[tuple[TestClient, str, MockProvider]]:
    url = f"sqlite:///{tmp_path / 'public.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config("alembic.ini"), "head")
    engine = make_engine(url)
    with make_session_factory(engine)() as session:
        tenant = create_tenant(session, "clinical-team")
        issued = issue_key(session, tenant.id, "integration")
    engine.dispose()
    provider = MockProvider()
    with TestClient(
        create_app(database_url=url, providers={"mock": provider})
    ) as client:
        yield client, issued.key, provider


def test_official_sdk_chat_embeddings_and_models(gateway) -> None:
    client, key, provider = gateway
    sdk = OpenAI(
        base_url="http://testserver/v1",
        api_key=key,
        http_client=client,
        max_retries=0,
    )
    reply = sdk.chat.completions.create(
        model="clinical-local",
        messages=[{"role": "user", "content": "hello gateway"}],
    )
    assert reply.choices[0].message.content == "Echo: hello gateway"
    assert reply.usage.total_tokens > 0
    embedding = sdk.embeddings.create(model="clinical-local", input="hello")
    assert len(embedding.data[0].embedding) == 16
    models = sdk.models.list()
    assert [item.id for item in models.data] == ["clinical-local"]
    assert [item["kind"] for item in provider.received] == ["chat", "embed"]


def test_policy_rejections_and_invalid_key_never_call_provider(gateway) -> None:
    client, key, provider = gateway
    auth = {"Authorization": f"Bearer {key}"}
    body = {"model": "clinical-local", "messages": [{"role": "user", "content": "hi"}]}
    disallowed = client.post(
        "/v1/chat/completions", headers=auth, json={**body, "model": "not-allowed"}
    )
    over_limit = client.post(
        "/v1/chat/completions", headers=auth, json={**body, "max_tokens": 513}
    )
    invalid = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer lpg_deadbeef_" + "a" * 43},
        json=body,
    )
    assert (disallowed.status_code, disallowed.json()["error"]["code"]) == (
        403,
        "model_not_allowed",
    )
    assert (over_limit.status_code, over_limit.json()["error"]["code"]) == (
        400,
        "max_tokens_exceeded",
    )
    assert invalid.status_code == 401
    assert provider.received == []


def test_payload_whitelist_and_public_errors(gateway) -> None:
    client, key, provider = gateway
    auth = {"Authorization": f"Bearer {key}", "x-request-id": "request-123"}
    body = {
        "model": "clinical-local",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello", "leak": "no"}],
            }
        ],
        "unknown_top_level": "never-forward",
    }
    response = client.post("/v1/chat/completions", headers=auth, json=body)
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-123"
    payload = provider.received[0]["payload"]
    assert payload["model"] == "clinical-local-v1"
    assert payload["max_tokens"] == 512
    assert "unknown_top_level" not in str(payload)
    assert "leak" not in str(payload)

    streamed = client.post(
        "/v1/chat/completions", headers=auth, json={**body, "stream": True}
    )
    invalid = client.post(
        "/v1/chat/completions", headers=auth, json={"model": "clinical-local"}
    )
    oversized = client.post(
        "/v1/chat/completions", headers=auth, content=b"x" * (1048576 + 1)
    )
    assert (streamed.status_code, streamed.json()["error"]["code"]) == (
        400,
        "streaming_not_supported",
    )
    assert invalid.status_code == 400
    assert set(invalid.json()) == {"error"}
    assert oversized.status_code == 413
    assert "x-request-id" in oversized.headers
    assert len(provider.received) == 1

    missing = client.get("/v1/does-not-exist", headers={"x-request-id": "bad value"})
    assert missing.status_code == 404
    assert set(missing.json()) == {"error"}
    assert missing.headers["x-request-id"] != "bad value"
    assert len(missing.headers["x-request-id"]) <= 64


def test_retry_outcomes_and_no_fallback(gateway) -> None:
    client, key, provider = gateway
    auth = {"Authorization": f"Bearer {key}"}
    body = {"model": "clinical-local", "messages": [{"role": "user", "content": "hi"}]}
    provider.failures = [500, 500]
    success = client.post("/v1/chat/completions", headers=auth, json=body)
    assert success.status_code == 200
    assert len(provider.received) == 3
    assert {item["payload"]["model"] for item in provider.received} == {
        "clinical-local-v1"
    }
    provider.received.clear()
    provider.failures = [429, 429, 429]
    failed = client.post("/v1/chat/completions", headers=auth, json=body)
    assert failed.status_code == 502
    assert failed.json()["error"]["code"] == "upstream_error"
    assert "429" in failed.json()["error"]["message"]
    assert len(provider.received) == 3


def test_phi_request_is_redacted_before_provider(gateway, caplog) -> None:
    client, key, provider = gateway
    raw = (
        "Jane Roe, DOB 01/02/1900, SSN 123-45-6789, "
        "jane.roe@example.com, 202-555-0143, "
        "NPI 0000000006, MRN: ZZZ12345"
    )
    with caplog.at_level(logging.INFO, logger="llm_policy_gateway.app"):
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "clinical-local",
                "messages": [{"role": "user", "content": raw}],
            },
        )
    assert response.status_code == 200
    payload = json.dumps(provider.received[0]["payload"])
    header = response.headers["x-gateway-redactions"]
    for value in (
        "Jane Roe",
        "01/02/1900",
        "123-45-6789",
        "jane.roe@example.com",
        "202-555-0143",
        "0000000006",
        "ZZZ12345",
    ):
        assert value not in payload + header + caplog.text
    for entity in (
        "PERSON",
        "DATE_TIME",
        "US_SSN",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_NPI",
        "MEDICAL_RECORD_NUMBER",
    ):
        assert f"<{entity}_1>" in payload
        assert f"{entity}=1" in header
    log = json.loads(
        next(
            record.message
            for record in caplog.records
            if '"event": "redaction"' in record.message
        )
    )
    assert log["counts"]["PERSON"] == 1
    assert isinstance(log["latency_ms"], float)
    assert "Jane Roe" not in json.dumps(log)


def test_invalid_npi_does_not_hide_other_entities(gateway) -> None:
    client, key, provider = gateway
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "clinical-local",
            "messages": [{"role": "user", "content": "provider line 2125550147"}],
        },
    )
    assert response.status_code == 200
    payload = json.dumps(provider.received[0]["payload"])
    assert "2125550147" not in payload
    invalid = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "clinical-local",
            "messages": [{"role": "user", "content": "provider line 2125550146"}],
        },
    )
    assert invalid.status_code == 200
    assert "2125550146" not in json.dumps(provider.received[1]["payload"])
    assert "PHONE_NUMBER=1" in invalid.headers["x-gateway-redactions"]
    assert "US_NPI" not in invalid.headers["x-gateway-redactions"]


def test_identity_fields_are_removed_or_pseudonymized_by_tenant(gateway) -> None:
    client, key, provider = gateway
    raw_user = "jane.roe@example.com"
    raw_name = "Jane Roe"
    with client.app.state.session_factory() as session:
        other = create_tenant(session, "billing-team")
        other_key = issue_key(session, other.id, "integration").key

    def send(api_key: str, model: str):
        return client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "user": raw_user,
                "messages": [
                    {"role": "user", "name": raw_name, "content": "hello gateway"}
                ],
            },
        )

    assert send(key, "clinical-local").status_code == 200
    assert send(key, "clinical-local").status_code == 200
    assert send(other_key, "billing-local").status_code == 200
    first, second, third = [item["payload"] for item in provider.received]
    assert first["user"].startswith("u_")
    assert len(first["user"]) == 18
    assert first["user"] == second["user"]
    assert first["user"] != third["user"]
    for payload in (first, second, third):
        assert raw_user not in json.dumps(payload)
        assert raw_name not in json.dumps(payload)
        assert "name" not in payload["messages"][0]

    embedded = client.post(
        "/v1/embeddings",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "clinical-local", "input": "hello gateway", "user": raw_user},
    )
    assert embedded.status_code == 200
    assert provider.received[3]["payload"]["user"] == first["user"]

    client.app.state.policy.tenants["clinical-team"].redaction_profile = "off"
    assert send(key, "clinical-local").status_code == 200
    off_payload = provider.received[4]["payload"]
    assert off_payload["user"] == raw_user
    assert off_payload["messages"][0]["name"] == raw_name


def test_response_reidentification_toggle_and_collision(gateway) -> None:
    client, key, provider = gateway
    auth = {"Authorization": f"Bearer {key}"}
    body = {
        "model": "clinical-local",
        "messages": [
            {"role": "user", "content": "<PERSON_1> Patient Jane Roe visited today."}
        ],
    }
    off = client.post("/v1/chat/completions", headers=auth, json=body)
    assert (
        "<PERSON_1> Patient <PERSON_2>"
        in off.json()["choices"][0]["message"]["content"]
    )
    assert "Jane Roe" not in off.text
    client.app.state.policy.tenants["clinical-team"].reidentify_response = True
    on = client.post("/v1/chat/completions", headers=auth, json=body)
    assert (
        "<PERSON_1> Patient Jane Roe" in on.json()["choices"][0]["message"]["content"]
    )
    assert len(provider.received) == 2


def test_off_profile_and_tool_embedding_paths(gateway) -> None:
    client, key, provider = gateway
    auth = {"Authorization": f"Bearer {key}"}
    body = {
        "model": "clinical-local",
        "messages": [
            {"role": "tool", "content": "jane.roe@example.com"},
            {"role": "user", "content": [{"type": "text", "text": "Jane Roe"}]},
        ],
    }
    response = client.post("/v1/chat/completions", headers=auth, json=body)
    assert response.status_code == 200
    assert "jane.roe@example.com" not in json.dumps(provider.received[0])
    assert "Jane Roe" not in json.dumps(provider.received[0])
    embedded = client.post(
        "/v1/embeddings",
        headers=auth,
        json={
            "model": "clinical-local",
            "input": ["jane.roe@example.com", "202-555-0143"],
        },
    )
    assert embedded.status_code == 200
    assert provider.received[1]["payload"]["input"][0] == "<EMAIL_ADDRESS_1>"
    assert "PHONE_NUMBER=1" in embedded.headers["x-gateway-redactions"]
    client.app.state.policy.tenants["clinical-team"].redaction_profile = "off"
    off = client.post("/v1/chat/completions", headers=auth, json=body)
    assert off.status_code == 200
    assert "x-gateway-redactions" not in off.headers
    assert "jane.roe@example.com" in json.dumps(provider.received[2])


def test_analyzer_failure_returns_503_without_provider_call(
    gateway, monkeypatch
) -> None:
    client, key, provider = gateway

    def fail(**_kwargs):
        raise RuntimeError("synthetic sensitive value")

    monkeypatch.setattr(client.app.state.redactor.analyzer, "analyze", fail)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "clinical-local",
            "messages": [
                {"role": "user", "content": "Patient Jane Roe visited today."}
            ],
        },
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "redaction_unavailable"
    assert "synthetic sensitive value" not in response.text
    assert provider.received == []


def test_anonymizer_failure_returns_503_without_provider_call(
    gateway, monkeypatch
) -> None:
    client, key, provider = gateway

    def fail(**_kwargs):
        raise RuntimeError("synthetic sensitive value")

    monkeypatch.setattr(client.app.state.redactor.anonymizer, "anonymize", fail)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "clinical-local",
            "messages": [
                {"role": "user", "content": "Patient Jane Roe visited today."}
            ],
        },
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "redaction_unavailable"
    assert provider.received == []


def test_slow_analysis_does_not_block_healthz(gateway, monkeypatch) -> None:
    client, key, _provider = gateway
    started = threading.Event()
    original = client.app.state.redactor.analyzer.analyze

    def slow(**kwargs):
        started.set()
        time.sleep(0.3)
        return original(**kwargs)

    monkeypatch.setattr(client.app.state.redactor.analyzer, "analyze", slow)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.post,
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "clinical-local",
                "messages": [{"role": "user", "content": "hello gateway"}],
            },
        )
        assert started.wait(2)
        before = time.perf_counter()
        health = client.get("/healthz")
        elapsed = time.perf_counter() - before
        assert health.status_code == 200
        assert elapsed < 0.1
        assert future.result().status_code == 200
