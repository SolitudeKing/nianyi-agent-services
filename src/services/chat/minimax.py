from __future__ import annotations

from .openai import OpenaiChat


class MinimaxChat(OpenaiChat):
    DEFAULT_ENDPOINT: str = "https://api.minimaxi.com/v1"
