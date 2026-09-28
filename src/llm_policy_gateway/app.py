"""HTTP application entry point."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from llm_policy_gateway.auth import (
    invalid_key_handler,
    policy_denied_handler,
)
from llm_policy_gateway.db import make_engine, make_session_factory
from llm_policy_gateway.keys import InvalidKey, require_key_pepper
from llm_policy_gateway.policy import PolicyDenied, load_policy


def create_app(
    policy_path: str | Path | None = None, database_url: str | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        require_key_pepper()
        path = policy_path or os.getenv("DEFAULT_POLICY_PATH", "policy.example.yaml")
        application.state.policy = load_policy(path)
        url = database_url or os.getenv("DATABASE_URL", "sqlite:///./gateway.db")
        engine = make_engine(url)
        application.state.session_factory = make_session_factory(engine)
        try:
            yield
        finally:
            engine.dispose()

    application = FastAPI(title="LLM Policy Gateway", lifespan=lifespan)
    application.add_exception_handler(InvalidKey, invalid_key_handler)
    application.add_exception_handler(PolicyDenied, policy_denied_handler)

    @application.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
