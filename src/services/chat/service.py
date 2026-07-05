from __future__ import annotations

from typing import Literal

from .base import ChatRequest, ChatResponse, GenerateChat
from .minimax import MinimaxChat
from .openai import OpenaiChat

ChatProvider = Literal["minimax", "openai"]


class ChatService:
    def __init__(self, generator: GenerateChat) -> None:
        self.generator = generator

    @classmethod
    def fromProvider(
        cls,
        provider: ChatProvider,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 120,
    ) -> ChatService:
        generators: dict[str, type[GenerateChat]] = {
            "minimax": MinimaxChat,
            "openai": OpenaiChat,
        }

        try:
            generator_cls = generators[provider]
        except KeyError as exc:
            raise ValueError(f"unsupported chat provider: {provider}") from exc

        return cls(generator_cls(api_key=api_key, endpoint=endpoint, timeout=timeout))

    def complete(self, request: ChatRequest) -> ChatResponse:
        return self.generator.complete(request)
