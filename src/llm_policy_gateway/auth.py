"""Bearer authentication for public routes."""

import re
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_policy_gateway.keys import InvalidKey, verify_key
from llm_policy_gateway.policy import PolicyDenied, PolicySettings

BEARER_PATTERN = re.compile(r"^Bearer ([^\s]+)$", re.IGNORECASE)


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    key_id: str
    policy: PolicySettings


def error_response(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
    )


def authenticate_request(request: Request) -> AuthContext:
    header = request.headers.get("authorization", "")
    match = BEARER_PATTERN.fullmatch(header)
    if match is None:
        raise InvalidKey
    with request.app.state.session_factory() as session:
        verified = verify_key(session, match.group(1))
        policy = request.app.state.policy.for_tenant(verified.tenant.name)
        return AuthContext(
            tenant_id=verified.tenant.id,
            key_id=verified.api_key.id,
            policy=policy,
        )


def invalid_key_handler(_request: Request, _error: InvalidKey) -> JSONResponse:
    return error_response(401, "Invalid API key.", "invalid_api_key")


def policy_denied_handler(_request: Request, _error: PolicyDenied) -> JSONResponse:
    return error_response(403, "Tenant policy is not configured.", "policy_denied")
