from __future__ import annotations

import base64
from typing import Any

from .base import GeneratedImage, GenerateImage, ImageRequest, ImageResponse


class OpenaiImage(GenerateImage):
    DEFAULT_ENDPOINT: str = "https://api.openai.com/v1"
    GENERATIONS_PATH: str = "/images/generations"
    EDITS_PATH: str = "/images/edits"

    SIZE_BY_ASPECT_RATIO = {
        "1:1": "1024x1024",
        "3:2": "1536x1024",
        "2:3": "1024x1536",
    }

    def generate(self, request: ImageRequest) -> ImageResponse:
        self._validateRequest(request)

        reference = request.reference_image_url or request.referenceDataUrl()
        path = self.EDITS_PATH if reference else self.GENERATIONS_PATH
        payload = self._buildPayload(request, reference)
        raw = self._postJson(path, payload)

        return ImageResponse(
            images=self._parseImages(raw),
            metadata={key: value for key, value in raw.items() if key != "data"},
            raw=raw,
        )

    def _buildPayload(
        self, request: ImageRequest, reference: str | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "n": request.n,
        }

        size = self._resolveSize(request)
        if size:
            payload["size"] = size

        response_format = self._resolveResponseFormat(request)
        if response_format:
            payload["response_format"] = response_format

        if reference:
            payload["images"] = [{"image_url": reference}]

        payload.update(request.extra)
        return payload

    @staticmethod
    def _validateRequest(request: ImageRequest) -> None:
        if not request.prompt:
            raise ValueError("prompt is required.")
        if not request.model:
            raise ValueError("model is required.")
        if request.n < 1 or request.n > 10:
            raise ValueError("n must be between 1 and 10.")

    def _resolveSize(self, request: ImageRequest) -> str | None:
        if request.width is not None or request.height is not None:
            if request.width is None or request.height is None:
                raise ValueError("width and height must be set together.")
            return f"{request.width}x{request.height}"

        if request.image_size:
            return request.image_size

        return self.SIZE_BY_ASPECT_RATIO.get(request.effectiveAspectRatio)

    @staticmethod
    def _resolveResponseFormat(request: ImageRequest) -> str | None:
        if OpenaiImage._isGptImageModel(request.model):
            return None
        if request.response_format == "base64":
            return "b64_json"
        if request.response_format in {"url", "b64_json"}:
            return request.response_format
        return None

    @staticmethod
    def _isGptImageModel(model: str) -> bool:
        return model.startswith("gpt-image") or model == "chatgpt-image-latest"

    @classmethod
    def _parseImages(cls, raw: dict[str, Any]) -> list[GeneratedImage]:
        images: list[GeneratedImage] = []

        for item in raw.get("data") or []:
            b64_json = item.get("b64_json")
            images.append(
                GeneratedImage(
                    url=item.get("url"),
                    b64_json=b64_json,
                    bytes_data=base64.b64decode(b64_json) if b64_json else None,
                    mime_type=cls._mimeType(item, raw),
                    revised_prompt=item.get("revised_prompt"),
                )
            )

        return images

    @staticmethod
    def _mimeType(item: dict[str, Any], raw: dict[str, Any]) -> str | None:
        output_format = item.get("output_format") or raw.get("output_format")
        if not output_format:
            return None
        return f"image/{output_format}"
