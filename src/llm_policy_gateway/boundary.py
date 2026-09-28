"""Public request size and request identifier boundary."""

import re
from collections.abc import Callable
from uuid import uuid4

from fastapi.responses import JSONResponse

from llm_policy_gateway.auth import error_response

SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


def _header(headers: list[tuple[bytes, bytes]], name: bytes) -> bytes | None:
    return next((value for key, value in headers if key.lower() == name), None)


class PublicBoundary:
    def __init__(self, app: Callable, max_request_bytes: int) -> None:
        if max_request_bytes <= 0:
            raise ValueError("MAX_REQUEST_BYTES must be positive")
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        incoming = _header(headers, b"x-request-id")
        try:
            candidate = incoming.decode("ascii") if incoming is not None else ""
        except UnicodeDecodeError:
            candidate = ""
        request_id = candidate if SAFE_REQUEST_ID.fullmatch(candidate) else uuid4().hex
        response_started = False

        async def send_with_id(message: dict) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode("ascii"))
                ]
            await send(message)

        async def run_app(next_receive: Callable) -> None:
            try:
                await self.app(scope, next_receive, send_with_id)
            except Exception:
                if response_started:
                    raise
                response = error_response(
                    500, "Internal gateway error.", "internal_error"
                )
                await response(scope, next_receive, send_with_id)

        if scope["path"].startswith("/v1/") and scope["method"] == "POST":
            length = _header(headers, b"content-length")
            if (
                length is not None
                and length.isdigit()
                and int(length) > self.max_request_bytes
            ):
                await self._too_large(scope, receive, send_with_id)
                return
            body = bytearray()
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    return
                if event["type"] != "http.request":
                    continue
                body.extend(event.get("body", b""))
                if len(body) > self.max_request_bytes:
                    await self._too_large(scope, receive, send_with_id)
                    return
                if not event.get("more_body", False):
                    break
            replayed = False

            async def replay_receive() -> dict:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {
                        "type": "http.request",
                        "body": bytes(body),
                        "more_body": False,
                    }
                return await receive()

            await run_app(replay_receive)
            return
        await run_app(receive)

    async def _too_large(self, scope: dict, receive: Callable, send: Callable) -> None:
        response: JSONResponse = error_response(
            413, "Request body exceeds MAX_REQUEST_BYTES.", "request_too_large"
        )
        await response(scope, receive, send)
