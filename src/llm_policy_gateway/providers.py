"""Asynchronous provider interface and deterministic recording provider."""

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

from llm_policy_gateway.schemas import (
    ChatRequest,
    ChatResult,
    EmbedRequest,
    EmbedResult,
    Usage,
)


class ProviderError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"Upstream returned {status_code}")


class Provider(Protocol):
    async def chat(self, request: ChatRequest) -> ChatResult: ...

    async def embed(self, request: EmbedRequest) -> EmbedResult: ...


def _tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


class MockProvider:
    def __init__(self, failures: Sequence[int | str] = ()) -> None:
        self.received: list[dict] = []
        self.failures = list(failures)

    def _record(self, kind: str, payload: dict) -> None:
        self.received.append({"kind": kind, "payload": payload})
        if self.failures:
            failure = self.failures.pop(0)
            if failure == "timeout":
                raise TimeoutError("Upstream timed out")
            raise ProviderError(int(failure))

    async def chat(self, request: ChatRequest) -> ChatResult:
        self._record("chat", request.provider_payload())
        last_user = next(
            (
                message
                for message in reversed(request.messages)
                if message.role == "user"
            ),
            None,
        )
        if last_user is None:
            text = ""
        elif isinstance(last_user.content, str):
            text = last_user.content
        else:
            text = " ".join(part.text for part in last_user.content)
        reply = f"Echo: {text}"
        prompt = sum(
            _tokens(message.content)
            if isinstance(message.content, str)
            else sum(_tokens(part.text) for part in message.content)
            for message in request.messages
        )
        completion = _tokens(reply)
        return ChatResult(
            model=request.model,
            content=reply,
            usage=Usage(prompt, completion, prompt + completion),
        )

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self._record("embed", request.provider_payload())
        inputs = [request.input] if isinstance(request.input, str) else request.input
        vectors = []
        for value in inputs:
            digest = hashlib.sha256(value.encode("utf-8")).digest()
            vectors.append(
                [
                    (int.from_bytes(digest[index : index + 2], "little") / 32767.5)
                    - 1.0
                    for index in range(0, 32, 2)
                ]
            )
        prompt = sum(_tokens(value) for value in inputs)
        return EmbedResult(
            model=request.model,
            embeddings=vectors,
            usage=Usage(prompt, 0, prompt),
        )
