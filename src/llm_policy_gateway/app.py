"""HTTP application entry point."""

import base64
import os
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from llm_policy_gateway.auth import (
    AuthContext,
    authenticate_request,
    error_response,
    invalid_key_handler,
    policy_denied_handler,
)
from llm_policy_gateway.boundary import PublicBoundary
from llm_policy_gateway.db import (
    make_engine,
    make_session_factory,
    require_current_schema,
)
from llm_policy_gateway.keys import InvalidKey, require_key_pepper
from llm_policy_gateway.policy import PolicyDenied, load_policy
from llm_policy_gateway.providers import MockProvider, Provider
from llm_policy_gateway.routing import Router, UpstreamFailure, load_model_routes
from llm_policy_gateway.schemas import ChatRequest, EmbedRequest


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
        route_path = models_path or os.getenv("MODELS_PATH", "models.yaml")
        routes = load_model_routes(route_path, application.state.policy)
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

    @application.exception_handler(UpstreamFailure)
    def upstream_error(_request: Request, error: UpstreamFailure) -> JSONResponse:
        return error_response(502, str(error), "upstream_error")

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
        result = await request.app.state.router.chat(body)
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
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
    ) -> dict | JSONResponse:
        if body.model not in context.policy.allowed_models:
            return error_response(403, "Model is not allowed.", "model_not_allowed")
        result = await request.app.state.router.embed(body)
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
