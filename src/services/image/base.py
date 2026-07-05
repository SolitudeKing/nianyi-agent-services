from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(slots=True)
class ImageRequest:
    """一次图片生成请求的统一入参。"""

    prompt: str
    model: str
    aspect_ratio: str = "1:1"
    image_size: str | None = None
    width: int | None = None
    height: int | None = None
    n: int = 1
    response_format: str = "base64"
    seed: int | None = None
    prompt_optimizer: bool | None = None
    aigc_watermark: bool | None = None
    style_type: str | None = None
    style_weight: float | None = None
    resolution: str | None = None
    reference_image_bytes: bytes | None = None
    reference_mime: str | None = None
    reference_image_url: str | None = None
    reference_type: str = "character"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def effectiveAspectRatio(self) -> str:
        return self.aspect_ratio or self.image_size or "1:1"

    def referenceDataUrl(self) -> str | None:
        if not self.reference_image_bytes:
            return None

        mime = self.reference_mime or "image/png"
        encoded = base64.b64encode(self.reference_image_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}"


@dataclass(slots=True)
class GeneratedImage:
    url: str | None = None
    b64_json: str | None = None
    bytes_data: bytes | None = None
    mime_type: str | None = None
    revised_prompt: str | None = None


@dataclass(slots=True)
class ImageResponse:
    images: list[GeneratedImage]
    response_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class ImageGenerationError(RuntimeError):
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


class GenerateImage(ABC):
    DEFAULT_ENDPOINT: str = ""

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
    def generate(self, request: ImageRequest) -> ImageResponse:
        """生成图片并返回统一响应。"""

    def _postJson(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self.endpoint:
            raise ImageGenerationError("Image generation endpoint is not configured.")

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
            raise ImageGenerationError(
                f"Image generation HTTP error: {exc.code}",
                status_code=exc.code,
                response=error_response,
            ) from exc
        except URLError as exc:
            raise ImageGenerationError(
                f"Image generation request failed: {exc.reason}"
            ) from exc

        parsed = self._loadsJson(response_body)
        if not isinstance(parsed, dict):
            raise ImageGenerationError(
                "Image generation response is not a JSON object."
            )
        return parsed

    def _authHeaders(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _loadsJson(value: str) -> Any:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ImageGenerationError(
                "Image generation response is not valid JSON."
            ) from exc
        return parsed

    @staticmethod
    def _tryLoadsJson(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
