"""HTTP application entry point."""

import base64
import hashlib
import hmac
import json
import logging
import os
import struct
import time
from contextlib import asynccontextmanager
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from llm_policy_gateway.auth import (
    AuthContext,
    authenticate_request,
    error_response,
    invalid_key_handler,
    policy_denied_handler,
)
from llm_policy_gateway.boundary import PublicBoundary
from llm_policy_gateway.budget import BudgetExceeded, BudgetManager, estimate_tokens
from llm_policy_gateway.db import (
    make_engine,
    make_session_factory,
    require_current_schema,
)
from llm_policy_gateway.keys import InvalidKey, require_key_pepper
from llm_policy_gateway.policy import PolicyDenied, load_policy
from llm_policy_gateway.pricing import load_pricing
from llm_policy_gateway.providers import MockProvider, Provider
from llm_policy_gateway.rate_limit import InMemoryRateLimiter, RateLimitExceeded
from llm_policy_gateway.redaction.redactor import (
    RedactionReport,
    RedactionSession,
    RedactionUnavailable,
    Redactor,
)
from llm_policy_gateway.routing import Router, UpstreamFailure, load_model_routes
from llm_policy_gateway.schemas import ChatRequest, EmbedRequest

LOGGER = logging.getLogger(__name__)


def _payload_texts(body: ChatRequest | EmbedRequest) -> list[str]:
    if isinstance(body, EmbedRequest):
        return [body.input] if isinstance(body.input, str) else body.input
    return [
        message.content
        if isinstance(message.content, str)
        else " ".join(part.text for part in message.content)
        for message in body.messages
    ]


def _rate_admission(
    request: Request, context: AuthContext, body: ChatRequest | EmbedRequest
) -> None:
    estimate = estimate_tokens(_payload_texts(body))
    state = request.app.state.rate_limiter.admit(
        context.tenant_id, context.policy.rpm, context.policy.tpm, estimate
    )
    request.state.rate_headers = {
        "x-ratelimit-remaining-requests": str(state.remaining_requests),
        "x-ratelimit-remaining-tokens": str(state.remaining_tokens),
    }


async def _reserve_budget(
    request: Request, context: AuthContext, body: ChatRequest | EmbedRequest
) -> str:
    prompt = estimate_tokens(_payload_texts(body))
    completion = (
        (body.max_completion_tokens or body.max_tokens or context.policy.max_tokens)
        if isinstance(body, ChatRequest)
        else 0
    )
    return await run_in_threadpool(
        request.app.state.budget.reserve,
        context.tenant_id,
        body.model,
        prompt,
        completion,
        context.policy.budget_usd_per_day,
    )


async def _complete_budget(
    request: Request, context: AuthContext, reservation: str, result
) -> None:
    remaining = await run_in_threadpool(
        request.app.state.budget.settle,
        reservation,
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
        context.policy.budget_usd_per_day,
    )
    request.state.budget_header = str(
        remaining.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


async def _release_budget(request: Request, reservation: str) -> None:
    await run_in_threadpool(request.app.state.budget.release, reservation)


async def _redact_request(
    request: Request,
    body: ChatRequest | EmbedRequest,
    context: AuthContext,
) -> tuple[ChatRequest | EmbedRequest, RedactionSession | None, RedactionReport]:
    profile = context.policy.redaction_profile
    if profile == "off":
        report = RedactionReport({})
        LOGGER.info(json.dumps({"event": "redaction", "counts": {}, "latency_ms": 0.0}))
        return body, None, report
    started = time.perf_counter()
    try:
        session = request.app.state.redactor.session(
            profile, context.policy.redaction_allow_terms
        )
        if isinstance(body, ChatRequest):
            redacted, report = await run_in_threadpool(session.redact_chat, body)
            for message in redacted.messages:
                message.name = None
        else:
            redacted, report = await run_in_threadpool(session.redact_embed, body)
        if redacted.user is not None:
            identity = f"{context.tenant_id}:{redacted.user}".encode()
            digest = hmac.new(
                require_key_pepper(), identity, hashlib.sha256
            ).hexdigest()
            redacted.user = f"u_{digest[:16]}"
    except Exception:
        latency = (time.perf_counter() - started) * 1000
        LOGGER.error(
            json.dumps(
                {
                    "event": "redaction",
                    "counts": {},
                    "latency_ms": round(latency, 3),
                    "outcome": "failure",
                }
            )
        )
        raise RedactionUnavailable from None
    latency = (time.perf_counter() - started) * 1000
    LOGGER.info(
        json.dumps(
            {
                "event": "redaction",
                "counts": report.counts,
                "latency_ms": round(latency, 3),
            }
        )
    )
    if report.counts:
        request.state.redaction_header = ",".join(
            f"{entity}={report.counts[entity]}" for entity in sorted(report.counts)
        )
    return redacted, session, report


def create_app(
    policy_path: str | Path | None = None,
    database_url: str | None = None,
    models_path: str | Path | None = None,
    providers: dict[str, Provider] | None = None,
    max_request_bytes: int | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        require_key_pepper()
        path = policy_path or os.getenv("DEFAULT_POLICY_PATH", "policy.yaml")
        application.state.policy = load_policy(path)
        policies = [
            application.state.policy.for_tenant(name)
            for name in application.state.policy.tenants
        ]
        if any(policy.redaction_profile != "off" for policy in policies):
            application.state.redactor = Redactor.create(
                os.getenv("PRESIDIO_NLP_MODEL", "en_core_web_lg"),
                language=os.getenv("PRESIDIO_LANGUAGE", "en"),
                threshold=float(os.getenv("REDACTION_SCORE_THRESHOLD", "0.5")),
            )
        else:
            application.state.redactor = None
        route_path = models_path or os.getenv("MODELS_PATH", "models.yaml")
        routes = load_model_routes(route_path, application.state.policy)
        pricing = load_pricing(
            os.getenv("PRICING_PATH", "pricing.yaml"), set(routes.models)
        )
        application.state.router = Router(
            routes,
            providers if providers is not None else {"mock": MockProvider()},
            max_retries=int(os.getenv("UPSTREAM_MAX_RETRIES", "2")),
            timeout_seconds=float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "60")),
        )
        url = database_url or os.getenv("DATABASE_URL", "sqlite:///./gateway.db")
        engine = make_engine(url)
        try:
            require_current_schema(engine)
            application.state.session_factory = make_session_factory(engine)
            application.state.budget = BudgetManager(
                application.state.session_factory,
                pricing,
                sqlite=engine.dialect.name == "sqlite",
            )
            application.state.rate_limiter = InMemoryRateLimiter()
            yield
        finally:
            engine.dispose()

    application = FastAPI(title="LLM Policy Gateway", lifespan=lifespan)
    limit = (
        max_request_bytes
        if max_request_bytes is not None
        else int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
    )
    application.add_middleware(PublicBoundary, max_request_bytes=limit)
    application.add_exception_handler(InvalidKey, invalid_key_handler)
    application.add_exception_handler(PolicyDenied, policy_denied_handler)

    @application.exception_handler(RateLimitExceeded)
    def rate_limited(_request: Request, error: RateLimitExceeded) -> JSONResponse:
        response = error_response(429, "Rate limit exceeded.", "rate_limited")
        response.headers["Retry-After"] = str(error.retry_after)
        return response

    @application.exception_handler(BudgetExceeded)
    def budget_exceeded(_request: Request, _error: BudgetExceeded) -> JSONResponse:
        return error_response(429, "Daily budget exceeded.", "budget_exceeded")

    @application.exception_handler(UpstreamFailure)
    def upstream_error(request: Request, error: UpstreamFailure) -> JSONResponse:
        response = error_response(502, str(error), "upstream_error")
        if hasattr(request.state, "redaction_header"):
            response.headers["x-gateway-redactions"] = request.state.redaction_header
        return response

    @application.exception_handler(RedactionUnavailable)
    def redaction_unavailable(
        _request: Request, _error: RedactionUnavailable
    ) -> JSONResponse:
        return error_response(503, "Redaction is unavailable.", "redaction_unavailable")

    @application.exception_handler(RequestValidationError)
    def validation_error(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return error_response(
            400, f"Invalid {location}: {first['msg']}", "validation_error"
        )

    @application.exception_handler(HTTPException)
    def http_error(_request: Request, error: HTTPException) -> JSONResponse:
        return error_response(error.status_code, str(error.detail), "http_error")

    @application.exception_handler(Exception)
    def internal_error(_request: Request, _error: Exception) -> JSONResponse:
        return error_response(500, "Internal gateway error.", "internal_error")

    @application.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/models")
    def models(context: Annotated[AuthContext, Depends(authenticate_request)]) -> dict:
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "gateway"}
                for name in sorted(context.policy.allowed_models)
            ],
        }

    @application.post("/v1/chat/completions", response_model=None)
    async def chat(
        body: ChatRequest,
        context: Annotated[AuthContext, Depends(authenticate_request)],
        request: Request,
        response: Response,
    ) -> dict | JSONResponse:
        if body.stream:
            return error_response(
                400, "Streaming is not supported.", "streaming_not_supported"
            )
        if body.model not in context.policy.allowed_models:
            return error_response(403, "Model is not allowed.", "model_not_allowed")
        if any(
            value is not None and value > context.policy.max_tokens
            for value in (body.max_tokens, body.max_completion_tokens)
        ):
            return error_response(
                400, "Requested max tokens exceed policy.", "max_tokens_exceeded"
            )
        if body.max_tokens is None and body.max_completion_tokens is None:
            body = body.model_copy(update={"max_tokens": context.policy.max_tokens})
        _rate_admission(request, context, body)
        body, redaction_session, _report = await _redact_request(request, body, context)
        if hasattr(request.state, "redaction_header"):
            response.headers["x-gateway-redactions"] = request.state.redaction_header
        reservation = await _reserve_budget(request, context, body)
        try:
            result = await request.app.state.router.chat(body)
        except Exception:
            await _release_budget(request, reservation)
            raise
        await _complete_budget(request, context, reservation, result)
        response.headers.update(request.state.rate_headers)
        response.headers["x-gateway-budget-remaining-usd"] = request.state.budget_header
        content = (
            redaction_session.reidentify(result.content)
            if redaction_session is not None and context.policy.reidentify_response
            else result.content
        )
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": result.finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        }

    @application.post("/v1/embeddings", response_model=None)
    async def embeddings(
        body: EmbedRequest,
        context: Annotated[AuthContext, Depends(authenticate_request)],
        request: Request,
        response: Response,
    ) -> dict | JSONResponse:
        if body.model not in context.policy.allowed_models:
            return error_response(403, "Model is not allowed.", "model_not_allowed")
        _rate_admission(request, context, body)
        body, _redaction_session, _report = await _redact_request(
            request, body, context
        )
        if hasattr(request.state, "redaction_header"):
            response.headers["x-gateway-redactions"] = request.state.redaction_header
        reservation = await _reserve_budget(request, context, body)
        try:
            result = await request.app.state.router.embed(body)
        except Exception:
            await _release_budget(request, reservation)
            raise
        await _complete_budget(request, context, reservation, result)
        response.headers.update(request.state.rate_headers)
        response.headers["x-gateway-budget-remaining-usd"] = request.state.budget_header
        data = []
        for index, vector in enumerate(result.embeddings):
            embedding = (
                base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode(
                    "ascii"
                )
                if body.encoding_format == "base64"
                else vector
            )
            data.append({"object": "embedding", "index": index, "embedding": embedding})
        return {
            "object": "list",
            "data": data,
            "model": result.model,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        }

    return application


app = create_app()
