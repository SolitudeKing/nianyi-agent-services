
from __future__ import annotations

import base64
from typing import Any

from .base import (
    GeneratedImage,
    GenerateImage,
    ImageGenerationError,
    ImageRequest,
    ImageResponse,
)


class MinimaxImage(GenerateImage):
    DEFAULT_ENDPOINT: str = "https://api.minimaxi.com"
    API_PATH: str = "/v1/image_generation"

    SUPPORTED_ASPECT_RATIOS = {
        "1:1",
        "16:9",
        "4:3",
        "3:2",
        "2:3",
        "3:4",
        "9:16",
        "21:9",
    }
    SUPPORTED_RESPONSE_FORMATS = {"url", "base64"}

    def generate(self, request: ImageRequest) -> ImageResponse:
        """生成图片。

        Returns:
            ImageResponse: 统一图片生成响应。
        """
        payload = self._buildPayload(request)
        raw = self._postJson(self.API_PATH, payload)
        self._raiseForMinimaxError(raw)

        data = raw.get("data") or {}
        return ImageResponse(
            images=self._parseImages(data),
            response_id=raw.get("id"),
            metadata=raw.get("metadata") or {},
            raw=raw,
        )

    def _buildPayload(self, request: ImageRequest) -> dict[str, Any]:
        self._validateRequest(request)

        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "response_format": request.response_format,
            "n": request.n,
        }

        if request.width is not None or request.height is not None:
            payload["width"] = request.width
            payload["height"] = request.height
        else:
            payload["aspect_ratio"] = request.effectiveAspectRatio

        if request.seed is not None:
            payload["seed"] = request.seed
        if request.prompt_optimizer is not None:
            payload["prompt_optimizer"] = request.prompt_optimizer
        if request.aigc_watermark is not None:
            payload["aigc_watermark"] = request.aigc_watermark
        if request.style_type or request.style_weight is not None:
            payload["style"] = self._buildStyle(request)

        reference = self._buildSubjectReference(request)
        if reference:
            payload["subject_reference"] = [reference]

        payload.update(request.extra)
        return payload

    def _validateRequest(self, request: ImageRequest) -> None:
        if not request.prompt:
            raise ValueError("prompt is required.")
        if not request.model:
            raise ValueError("model is required.")
        if request.width is not None or request.height is not None:
            self._validateSize(request.width, request.height)
        elif request.effectiveAspectRatio not in self.SUPPORTED_ASPECT_RATIOS:
            raise ValueError(
                f"unsupported aspect_ratio: {request.effectiveAspectRatio}"
            )
        if request.response_format not in self.SUPPORTED_RESPONSE_FORMATS:
            raise ValueError(f"unsupported response_format: {request.response_format}")
        if request.n < 1 or request.n > 9:
            raise ValueError("n must be between 1 and 9.")

    @staticmethod
    def _validateSize(width: int | None, height: int | None) -> None:
        if width is None or height is None:
            raise ValueError("width and height must be set together.")

        for name, value in {"width": width, "height": height}.items():
            if value < 512 or value > 2048:
                raise ValueError(f"{name} must be between 512 and 2048.")
            if value % 8 != 0:
                raise ValueError(f"{name} must be a multiple of 8.")

    @staticmethod
    def _buildStyle(request: ImageRequest) -> dict[str, Any]:
        style: dict[str, Any] = {}
        if request.style_type:
            style["style_type"] = request.style_type
        if request.style_weight is not None:
            style["style_weight"] = request.style_weight
        return style

    @staticmethod
    def _buildSubjectReference(request: ImageRequest) -> dict[str, str] | None:
        image_file = request.reference_image_url or request.referenceDataUrl()
        if not image_file:
            return None
        return {
            "type": request.reference_type,
            "image_file": image_file,
        }

    @staticmethod
    def _raiseForMinimaxError(raw: dict[str, Any]) -> None:
        base_resp = raw.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code in (None, 0):
            return

        message = base_resp.get("status_msg") or "MiniMax image generation failed."
        raise ImageGenerationError(
            message,
            status_code=status_code,
            response=raw,
        )

    @classmethod
    def _parseImages(cls, data: dict[str, Any]) -> list[GeneratedImage]:
        images: list[GeneratedImage] = []

        for image_url in data.get("image_urls") or []:
            images.append(GeneratedImage(url=image_url))

        for image_base64 in data.get("image_base64") or []:
            mime_type, payload = cls._splitBase64Image(image_base64)
            images.append(
                GeneratedImage(
                    b64_json=image_base64,
                    bytes_data=base64.b64decode(payload),
                    mime_type=mime_type,
                )
            )

        return images

    @staticmethod
    def _splitBase64Image(value: str) -> tuple[str | None, str]:
        if not value.startswith("data:"):
            return None, value

        if "," not in value:
            return None, value

        header, payload = value.split(",", 1)
        mime_type = header.removeprefix("data:").split(";", 1)[0] or None
        return mime_type, payload
