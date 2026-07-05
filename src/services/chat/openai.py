from __future__ import annotations

from typing import Any

from .base import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatUsage,
    GenerateChat,
)


class OpenaiChat(GenerateChat):
    DEFAULT_ENDPOINT: str = "https://api.openai.com/v1"

    def complete(self, request: ChatRequest) -> ChatResponse:
        payload = self._buildPayload(request)
        raw = self._postJson(self.CHAT_COMPLETIONS_PATH, payload)
        return self._parseResponse(raw)

    def _buildPayload(self, request: ChatRequest) -> dict[str, Any]:
        self._validateRequest(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [self._messagePayload(message) for message in request.messages],
        }

        self._addIfPresent(payload, "temperature", request.temperature)
        self._addIfPresent(payload, "top_p", request.top_p)
        self._addIfPresent(payload, "max_tokens", request.max_tokens)
        self._addIfPresent(
            payload,
            "max_completion_tokens",
            request.max_completion_tokens,
        )
        if request.tools:
            payload["tools"] = request.tools
        self._addIfPresent(payload, "tool_choice", request.tool_choice)
        self._addIfPresent(payload, "response_format", request.response_format)
        self._addIfPresent(payload, "stream_options", request.stream_options)
        self._addIfPresent(payload, "reasoning_split", request.reasoning_split)
        self._addIfPresent(payload, "thinking", request.thinking)
        self._addIfPresent(payload, "service_tier", request.service_tier)

        payload.update(request.extra)
        payload.update(request.extra_body)
        return payload

    @staticmethod
    def _validateRequest(request: ChatRequest) -> None:
        if not request.model:
            raise ValueError("model is required.")
        if not request.messages:
            raise ValueError("messages is required.")
        if request.stream:
            raise ValueError("streaming chat is not supported by this HTTP wrapper.")

    @staticmethod
    def _messagePayload(message: ChatMessage | dict[str, Any]) -> dict[str, Any]:
        if isinstance(message, ChatMessage):
            return message.toPayload()
        return dict(message)

    @classmethod
    def _parseResponse(cls, raw: dict[str, Any]) -> ChatResponse:
        usage = raw.get("usage")
        return ChatResponse(
            response_id=raw.get("id"),
            model=raw.get("model"),
            choices=cls._parseChoices(raw.get("choices")),
            usage=cls._parseUsage(usage) if isinstance(usage, dict) else None,
            metadata={
                key: value
                for key, value in raw.items()
                if key not in {"id", "model", "choices", "usage"}
            },
            raw=raw,
        )

    @classmethod
    def _parseChoices(cls, value: Any) -> list[ChatChoice]:
        if not isinstance(value, list):
            return []
        return [cls._parseChoice(item) for item in value if isinstance(item, dict)]

    @classmethod
    def _parseChoice(cls, value: dict[str, Any]) -> ChatChoice:
        message = value.get("message")
        return ChatChoice(
            index=value.get("index"),
            message=cls._parseMessage(message) if isinstance(message, dict) else None,
            finish_reason=value.get("finish_reason"),
            raw=value,
        )

    @staticmethod
    def _parseMessage(value: dict[str, Any]) -> ChatMessage:
        known_keys = {
            "role",
            "content",
            "name",
            "tool_call_id",
            "tool_calls",
            "reasoning_content",
            "reasoning_details",
        }
        tool_calls = value.get("tool_calls")
        reasoning_details = value.get("reasoning_details")
        return ChatMessage(
            role=value.get("role") or "assistant",
            content=value.get("content"),
            name=value.get("name"),
            tool_call_id=value.get("tool_call_id"),
            tool_calls=tool_calls if isinstance(tool_calls, list) else [],
            reasoning_content=value.get("reasoning_content"),
            reasoning_details=(
                reasoning_details if isinstance(reasoning_details, list) else []
            ),
            extra={key: item for key, item in value.items() if key not in known_keys},
        )

    @staticmethod
    def _parseUsage(value: dict[str, Any]) -> ChatUsage:
        return ChatUsage(
            prompt_tokens=value.get("prompt_tokens"),
            completion_tokens=value.get("completion_tokens"),
            total_tokens=value.get("total_tokens"),
            raw=value,
        )

    @staticmethod
    def _addIfPresent(payload: dict[str, Any], key: str, value: Any) -> None:
        if value is not None:
            payload[key] = value
