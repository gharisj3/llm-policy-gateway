"""HTTP application entry point."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from llm_policy_gateway.policy import load_policy


def create_app(policy_path: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        path = policy_path or os.getenv("DEFAULT_POLICY_PATH", "policy.example.yaml")
        application.state.policy = load_policy(path)
        yield

    application = FastAPI(title="LLM Policy Gateway", lifespan=lifespan)

    @application.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
