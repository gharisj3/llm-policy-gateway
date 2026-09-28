import asyncio
import json

from llm_policy_gateway.boundary import PublicBoundary


def test_unexpected_error_has_json_shape_and_request_id() -> None:
    async def broken(_scope, _receive, _send) -> None:
        raise RuntimeError("internal detail")

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"type": "http", "path": "/healthz", "method": "GET", "headers": []}
    asyncio.run(PublicBoundary(broken, 1024)(scope, receive, send))

    assert sent[0]["status"] == 500
    assert any(key == b"x-request-id" for key, _ in sent[0]["headers"])
    body = json.loads(sent[1]["body"])
    assert body["error"]["code"] == "internal_error"
    assert "internal detail" not in str(body)
