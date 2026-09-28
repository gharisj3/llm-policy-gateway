"""Administrative API served independently from the public application."""

import hmac
import ipaddress
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from llm_policy_gateway.auth import error_response
from llm_policy_gateway.db import ApiKey, make_engine, make_session_factory
from llm_policy_gateway.keys import (
    create_tenant,
    issue_key,
    list_keys,
    require_key_pepper,
    revoke_key,
)

PLACEHOLDER_TOKEN = "replace-with-a-long-random-secret"
LOGGER = logging.getLogger(__name__)


class AdminUnauthorized(Exception):
    pass


class TenantInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=255)


class KeyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=255)


class KeyOutput(BaseModel):
    id: str
    prefix: str
    label: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


def _key_output(row: ApiKey) -> KeyOutput:
    return KeyOutput(
        id=row.id,
        prefix=row.prefix,
        label=row.label,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
    )


def require_admin(request: Request) -> None:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        value.encode("utf-8"), request.app.state.admin_token.encode("utf-8")
    ):
        raise AdminUnauthorized


def get_session(request: Request):
    with request.app.state.session_factory() as session:
        yield session


def parse_bind(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or not host or not port_text.isdecimal():
        raise ValueError(f"Invalid bind address: {value}")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid bind address: {value}")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host, port


def validate_admin_bind(value: str, allow_remote: bool) -> tuple[str, int]:
    host, port = parse_bind(value)
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host.lower() == "localhost"
    if not local and not allow_remote:
        raise ValueError("ADMIN_BIND must be loopback unless ADMIN_ALLOW_REMOTE=true")
    if not local:
        LOGGER.warning("Remote admin bind enabled: %s", value)
    return host, port


def create_admin_app(
    database_url: str | None = None, admin_token: str | None = None
) -> FastAPI:
    token = admin_token if admin_token is not None else os.getenv("ADMIN_TOKEN", "")
    if not token.strip() or token == PLACEHOLDER_TOKEN:
        raise ValueError("ADMIN_TOKEN must be set to a non-placeholder secret")
    bind = os.getenv("ADMIN_BIND", "127.0.0.1:8081")
    allow_remote = os.getenv("ADMIN_ALLOW_REMOTE", "false").lower() == "true"
    validate_admin_bind(bind, allow_remote)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        require_key_pepper()
        url = database_url or os.getenv("DATABASE_URL", "sqlite:///./gateway.db")
        engine = make_engine(url)
        application.state.session_factory = make_session_factory(engine)
        application.state.admin_token = token
        try:
            yield
        finally:
            engine.dispose()

    application = FastAPI(
        title="LLM Policy Gateway Admin",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @application.exception_handler(AdminUnauthorized)
    def unauthorized(_request: Request, _error: AdminUnauthorized) -> JSONResponse:
        return error_response(401, "Invalid admin token.", "invalid_admin_token")

    router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

    @router.post("/tenants", status_code=201)
    def add_tenant(
        body: TenantInput, session: Annotated[Session, Depends(get_session)]
    ) -> dict:
        try:
            row = create_tenant(session, body.name)
        except IntegrityError:
            session.rollback()
            return error_response(409, "Tenant name already exists.", "tenant_exists")
        return {"id": row.id, "name": row.name, "created_at": row.created_at}

    @router.post("/tenants/{tenant_id}/keys", status_code=201, response_model=None)
    def add_key(
        tenant_id: str,
        body: KeyInput,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict | JSONResponse:
        try:
            issued = issue_key(session, tenant_id, body.label)
        except ValueError:
            return error_response(404, "Tenant not found.", "tenant_not_found")
        row = session.get(ApiKey, issued.id)
        return {**_key_output(row).model_dump(mode="json"), "key": issued.key}

    @router.post("/keys/{key_id}/revoke", response_model=None)
    def revoke(
        key_id: str, session: Annotated[Session, Depends(get_session)]
    ) -> KeyOutput | JSONResponse:
        try:
            return _key_output(revoke_key(session, key_id))
        except ValueError:
            return error_response(404, "Key not found.", "key_not_found")

    @router.get("/tenants/{tenant_id}/keys", response_model=None)
    def keys(
        tenant_id: str, session: Annotated[Session, Depends(get_session)]
    ) -> list[KeyOutput] | JSONResponse:
        try:
            return [_key_output(row) for row in list_keys(session, tenant_id)]
        except ValueError:
            return error_response(404, "Tenant not found.", "tenant_not_found")

    application.include_router(router)
    return application
