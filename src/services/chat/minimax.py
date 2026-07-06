from __future__ import annotations

from typing import Any, overload

from .base import ChatMessage, ChatRequest, ChatResponse
from .openai import OpenaiChat

ChatMessages = list[ChatMessage | dict[str, Any]]


class MinimaxChat(OpenaiChat):
    DEFAULT_ENDPOINT: str = "https://api.minimaxi.com/v1"
    DEFAULT_MODEL: str = "MiniMax-M3"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str | None = None,
        timeout: float = 120,
        model: str = DEFAULT_MODEL,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
        reasoning_split: bool | None = None,
        thinking: dict[str, Any] | None = None,
        service_tier: str | None = None,
        extra: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(api_key=api_key, endpoint=endpoint, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.tools = list(tools or [])
        self.tool_choice = tool_choice
        self.response_format = dict(response_format or {})
        self.stream_options = dict(stream_options or {})
        self.reasoning_split = reasoning_split
        self.thinking = dict(thinking or {})
        self.service_tier = service_tier
        self.extra = dict(extra or {})
        self.extra_body = dict(extra_body or {})

    @overload
    def complete(self, request: ChatRequest) -> ChatResponse:
        ...

    @overload
    def complete(
        self,
        request: ChatMessages,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
        reasoning_split: bool | None = None,
        thinking: dict[str, Any] | None = None,
        service_tier: str | None = None,
        extra: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResponse:
        ...

    def complete(
        self,
        request: ChatRequest | ChatMessages,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
        reasoning_split: bool | None = None,
        thinking: dict[str, Any] | None = None,
        service_tier: str | None = None,
        extra: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResponse:
        if isinstance(request, ChatRequest):
            return super().complete(request)

        return super().complete(
            self.buildRequest(
                request,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                stream_options=stream_options,
                reasoning_split=reasoning_split,
                thinking=thinking,
                service_tier=service_tier,
                extra=extra,
                extra_body=extra_body,
            )
        )

    def buildRequest(
        self,
        messages: ChatMessages,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
        reasoning_split: bool | None = None,
        thinking: dict[str, Any] | None = None,
        service_tier: str | None = None,
        extra: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatRequest:
        return ChatRequest(
            model=model or self.model,
            messages=messages,
            temperature=self._resolve(temperature, self.temperature),
            top_p=self._resolve(top_p, self.top_p),
            max_tokens=self._resolve(max_tokens, self.max_tokens),
            max_completion_tokens=self._resolve(
                max_completion_tokens,
                self.max_completion_tokens,
            ),
            tools=self._resolveList(tools, self.tools),
            tool_choice=self._resolve(tool_choice, self.tool_choice),
            response_format=self._resolveDict(response_format, self.response_format),
            stream_options=self._resolveDict(stream_options, self.stream_options),
            reasoning_split=self._resolve(reasoning_split, self.reasoning_split),
            thinking=self._resolveDict(thinking, self.thinking),
            service_tier=self._resolve(service_tier, self.service_tier),
            extra={**self.extra, **(extra or {})},
            extra_body={**self.extra_body, **(extra_body or {})},
        )

    @staticmethod
    def _resolve(value: Any | None, default: Any | None) -> Any | None:
        return default if value is None else value

    @staticmethod
    def _resolveDict(
        value: dict[str, Any] | None,
        default: dict[str, Any],
    ) -> dict[str, Any] | None:
        resolved = default if value is None else value
        return dict(resolved) if resolved else None

    @staticmethod
    def _resolveList(
        value: list[dict[str, Any]] | None,
        default: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resolved = default if value is None else value
        return list(resolved)
