import asyncio
from pathlib import Path

import pytest

from llm_policy_gateway.policy import load_policy
from llm_policy_gateway.providers import MockProvider
from llm_policy_gateway.routing import Router, UpstreamFailure, load_model_routes
from llm_policy_gateway.schemas import ChatRequest


def test_routes_require_every_policy_model_and_reject_extra_fields(tmp_path) -> None:
    policy = load_policy("policy.example.yaml")
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  approved-local:\n    provider: mock\n    upstream_model: one\n"
    )
    with pytest.raises(ValueError, match="Policy models have no route"):
        load_model_routes(path, policy)
    source = Path("models.example.yaml").read_text(encoding="utf-8")
    path.write_text(
        source.replace("    provider: mock", "    extra: true\n    provider: mock", 1)
    )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_model_routes(path, policy)


def test_retries_use_one_route_and_stop_at_bound() -> None:
    policy = load_policy("policy.example.yaml")
    routes = load_model_routes("models.example.yaml", policy)
    sleeps = []

    async def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    provider = MockProvider(failures=[500, 500])
    router = Router(
        routes,
        {"mock": provider},
        sleep=fake_sleep,
        jitter=lambda _low, _high: 0.0,
    )
    request = ChatRequest(
        model="clinical-local", messages=[{"role": "user", "content": "hello"}]
    )
    response = asyncio.run(router.chat(request))
    assert response.model == "clinical-local-v1"
    assert len(provider.received) == 3
    assert all(
        item["payload"]["model"] == "clinical-local-v1" for item in provider.received
    )
    assert sleeps == [0.0, 0.0]

    provider = MockProvider(failures=[429, 429, 429, 429])
    router = Router(routes, {"mock": provider}, sleep=fake_sleep)
    with pytest.raises(UpstreamFailure) as error:
        asyncio.run(router.chat(request))
    assert error.value.status_code == 429
    assert len(provider.received) == 3
