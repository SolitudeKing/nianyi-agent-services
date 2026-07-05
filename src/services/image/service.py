from __future__ import annotations

from typing import Literal

from .base import GenerateImage, ImageRequest, ImageResponse
from .minimax import MinimaxImage
from .openai import OpenaiImage

ImageProvider = Literal["minimax", "openai"]


class ImageService:
    def __init__(self, generator: GenerateImage) -> None:
        self.generator = generator

    @classmethod
    def fromProvider(
        cls,
        provider: ImageProvider,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 120,
    ) -> ImageService:
        generators: dict[str, type[GenerateImage]] = {
            "minimax": MinimaxImage,
            "openai": OpenaiImage,
        }

        try:
            generator_cls = generators[provider]
        except KeyError as exc:
            raise ValueError(f"unsupported image provider: {provider}") from exc

        return cls(generator_cls(api_key=api_key, endpoint=endpoint, timeout=timeout))

    def generate(self, request: ImageRequest) -> ImageResponse:
        return self.generator.generate(request)
