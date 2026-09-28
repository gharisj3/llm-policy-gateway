import asyncio

import pytest

from llm_policy_gateway.providers import MockProvider, ProviderError
from llm_policy_gateway.schemas import ChatRequest, EmbedRequest


def test_mock_is_deterministic_and_records_whitelisted_payload() -> None:
    provider = MockProvider()
    request = ChatRequest.model_validate(
        {
            "model": "upstream-local",
            "messages": [
                {"role": "user", "content": "hello there", "unknown": "hidden"}
            ],
            "unknown": "hidden",
        }
    )
    first = asyncio.run(provider.chat(request))
    second = asyncio.run(provider.chat(request))
    assert first == second
    assert first.content == "Echo: hello there"
    assert provider.received[0]["payload"] == provider.received[1]["payload"]
    assert "unknown" not in str(provider.received)

    embedding = asyncio.run(
        provider.embed(EmbedRequest(model="upstream-local", input=["same", "same"]))
    )
    assert len(embedding.embeddings[0]) == 16
    assert embedding.embeddings[0] == embedding.embeddings[1]


def test_mock_failure_is_recorded() -> None:
    provider = MockProvider(failures=[500, "timeout"])
    request = ChatRequest(
        model="upstream-local", messages=[{"role": "user", "content": "hi"}]
    )
    with pytest.raises(ProviderError) as error:
        asyncio.run(provider.chat(request))
    assert error.value.status_code == 500
    with pytest.raises(TimeoutError):
        asyncio.run(provider.chat(request))
    assert len(provider.received) == 2
