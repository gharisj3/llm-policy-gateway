"""Validated model routing with bounded upstream retries."""

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_policy_gateway.policy import PolicyDocument, UniqueKeyLoader
from llm_policy_gateway.providers import Provider, ProviderError
from llm_policy_gateway.schemas import (
    ChatRequest,
    ChatResult,
    EmbedRequest,
    EmbedResult,
)


class ModelRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["mock", "ollama", "azure_openai", "bedrock", "openai"]
    upstream_model: str = Field(min_length=1)


class ModelRoutes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: dict[str, ModelRoute] = Field(min_length=1)


def load_model_routes(path: str | Path, policy: PolicyDocument) -> ModelRoutes:
    source = Path(path)
    try:
        raw = yaml.load(source.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        routes = ModelRoutes.model_validate(raw)
    except OSError as error:
        raise ValueError(f"Cannot read model routes {source}: {error}") from error
    except (yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"Invalid model routes {source}: {error}") from error
    required = set(policy.defaults.allowed_models)
    for tenant in policy.tenants:
        required.update(policy.for_tenant(tenant).allowed_models)
    missing = sorted(required - routes.models.keys())
    if missing:
        raise ValueError(f"Policy models have no route: {', '.join(missing)}")
    return routes


class UpstreamFailure(Exception):
    def __init__(self, status_code: int | None):
        self.status_code = status_code
        super().__init__(
            f"Upstream request failed with status {status_code}"
            if status_code is not None
            else "Upstream request timed out"
        )


class Router:
    def __init__(
        self,
        routes: ModelRoutes,
        providers: Mapping[str, Provider],
        max_retries: int = 2,
        timeout_seconds: float = 60,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if max_retries < 0 or timeout_seconds <= 0:
            raise ValueError(
                "Upstream retries must be nonnegative and timeout positive"
            )
        missing = {
            route.provider for route in routes.models.values()
        } - providers.keys()
        if missing:
            raise ValueError(
                f"Providers are not configured: {', '.join(sorted(missing))}"
            )
        self.routes = routes
        self.providers = providers
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.sleep = sleep
        self.jitter = jitter

    async def _call(self, operation: Callable[[], Awaitable]):
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(operation(), timeout=self.timeout_seconds)
            except ProviderError as error:
                if error.status_code != 429 and error.status_code < 500:
                    raise UpstreamFailure(error.status_code) from error
                status = error.status_code
            except TimeoutError:
                status = None
            if attempt == self.max_retries:
                raise UpstreamFailure(status)
            await self.sleep(self.jitter(0, min(8.0, 0.25 * 2**attempt)))
        raise AssertionError("Retry loop ended unexpectedly")

    async def chat(self, request: ChatRequest) -> ChatResult:
        route = self.routes.models[request.model]
        upstream = request.model_copy(update={"model": route.upstream_model})
        return await self._call(lambda: self.providers[route.provider].chat(upstream))

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        route = self.routes.models[request.model]
        upstream = request.model_copy(update={"model": route.upstream_model})
        return await self._call(lambda: self.providers[route.provider].embed(upstream))
