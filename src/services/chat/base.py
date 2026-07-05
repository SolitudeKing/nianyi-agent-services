from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ChatContent = str | list[dict[str, Any]] | None


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: ChatContent = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def toPayload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        self._addIfPresent(payload, "content", self.content)
        self._addIfPresent(payload, "name", self.name)
        self._addIfPresent(payload, "tool_call_id", self.tool_call_id)
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        self._addIfPresent(payload, "reasoning_content", self.reasoning_content)
        if self.reasoning_details:
            payload["reasoning_details"] = self.reasoning_details
        payload.update(self.extra)
        return payload

    @staticmethod
    def _addIfPresent(payload: dict[str, Any], key: str, value: Any) -> None:
        if value is not None:
            payload[key] = value


@dataclass(slots=True)
class ChatRequest:
    model: str
    messages: list[ChatMessage | dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    reasoning_split: bool | None = None
    thinking: dict[str, Any] | None = None
    service_tier: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatChoice:
    index: int | None
    message: ChatMessage | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResponse:
    choices: list[ChatChoice]
    response_id: str | None = None
    model: str | None = None
    usage: ChatUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str | None:
        if not self.choices:
            return None
        message = self.choices[0].message
        if not message or not isinstance(message.content, str):
            return None
        return message.content


class ChatServiceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}


class GenerateChat(ABC):
    DEFAULT_ENDPOINT: str = ""
    CHAT_COMPLETIONS_PATH: str = "/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str | None = None,
        timeout: float = 120,
    ) -> None:
        self.api_key = api_key
        self.endpoint = (endpoint or self.DEFAULT_ENDPOINT).rstrip("/")
        self.timeout = timeout

    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse:
        """Create a chat completion and return a normalized response."""

    def _postJson(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self.endpoint:
            raise ChatServiceError("Chat endpoint is not configured.")

        request_headers = {
            "Content-Type": "application/json",
            **self._authHeaders(),
            **(headers or {}),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.endpoint}{path}",
            data=body,
            headers=request_headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            parsed = self._tryLoadsJson(response_body)
            error_response = (
                parsed if isinstance(parsed, dict) else {"body": response_body}
            )
            raise ChatServiceError(
                f"Chat completion HTTP error: {exc.code}",
                status_code=exc.code,
                response=error_response,
            ) from exc
        except URLError as exc:
            raise ChatServiceError(
                f"Chat completion request failed: {exc.reason}"
            ) from exc

        parsed = self._loadsJson(response_body)
        if not isinstance(parsed, dict):
            raise ChatServiceError("Chat completion response is not a JSON object.")
        return parsed

    def _authHeaders(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _loadsJson(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ChatServiceError(
                "Chat completion response is not valid JSON."
            ) from exc

    @staticmethod
    def _tryLoadsJson(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
