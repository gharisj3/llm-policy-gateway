"""Whitelisted public request and provider result shapes."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TextPart(RequestModel):
    type: Literal["text"]
    text: str


class ChatMessage(RequestModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[TextPart]
    name: str | None = None
    tool_call_id: str | None = None


class ToolFunction(RequestModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDefinition(RequestModel):
    type: Literal["function"]
    function: ToolFunction


class ToolChoiceFunction(RequestModel):
    name: str


class ToolChoice(RequestModel):
    type: Literal["function"]
    function: ToolChoiceFunction


class ResponseFormat(RequestModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: dict[str, Any] | None = None


class ChatRequest(RequestModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    response_format: ResponseFormat | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoice | None = None
    user: str | None = None
    stream: bool = False

    def provider_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude={"stream"})


class EmbedRequest(RequestModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    user: str | None = None

    def provider_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ChatResult:
    model: str
    content: str
    usage: Usage
    finish_reason: str = "stop"


@dataclass(frozen=True)
class EmbedResult:
    model: str
    embeddings: list[list[float]]
    usage: Usage
