from collections.abc import Iterator

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
