from __future__ import annotations

import json
import mimetypes
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(slots=True)
class VoiceSetting:
    voice_id: str
    speed: float | None = None
    vol: float | None = None
    pitch: int | None = None
    emotion: str | None = None
    text_normalization: bool | None = None
    latex_read: bool | None = None


@dataclass(slots=True)
class AudioSetting:
    sample_rate: int | None = None
    bitrate: int | None = None
    format: str | None = None
    channel: int | None = None
    force_cbr: bool | None = None


@dataclass(slots=True)
class TimbreWeight:
    voice_id: str
    weight: int


@dataclass(slots=True)
class SpeechRequest:
    text: str
    model: str
    voice_id: str | None = None
    voice_setting: VoiceSetting | None = None
    audio_setting: AudioSetting | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    pronunciation_tones: list[str] = field(default_factory=list)
    timbre_weights: list[TimbreWeight] = field(default_factory=list)
    language_boost: str | None = None
    voice_modify: dict[str, Any] | None = None
    subtitle_enable: bool | None = None
    subtitle_type: str | None = None
    output_format: str = "hex"
    aigc_watermark: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SynthesizedAudio:
    audio: str | None = None
    audio_hex: str | None = None
    audio_url: str | None = None
    bytes_data: bytes | None = None
    subtitle_file: str | None = None
    status: int | None = None


@dataclass(slots=True)
class SpeechResponse:
    audio: SynthesizedAudio | None
    trace_id: str | None = None
    extra_info: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UploadCloneAudioRequest:
    file_path: str | Path
    purpose: str = "voice_clone"
    filename: str | None = None


@dataclass(slots=True)
class UploadedFile:
    file_id: int | None = None
    bytes: int | None = None
    created_at: int | None = None
    filename: str | None = None
    purpose: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UploadCloneAudioResponse:
    file: UploadedFile | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClonePrompt:
    prompt_audio: int
    prompt_text: str


@dataclass(slots=True)
class VoiceCloneRequest:
    file_id: int
    voice_id: str
    clone_prompt: ClonePrompt | None = None
    text: str | None = None
    model: str | None = None
    language_boost: str | None = None
    text_validation: str | None = None
    accuracy: float | None = None
    need_noise_reduction: bool | None = None
    need_volume_normalization: bool | None = None
    aigc_watermark: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VoiceCloneResponse:
    voice_id: str
    demo_audio: str | None = None
    input_sensitive: Any | None = None
    input_sensitive_type: int | None = None
    extra_info: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VoiceInfo:
    voice_id: str | None = None
    voice_name: str | None = None
    description: list[str] = field(default_factory=list)
    created_time: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VoiceListResponse:
    system_voice: list[VoiceInfo] = field(default_factory=list)
    voice_cloning: list[VoiceInfo] = field(default_factory=list)
    voice_generation: list[VoiceInfo] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DeleteVoiceResponse:
    voice_id: str | None = None
    created_time: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class TtsServiceError(RuntimeError):
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


class GenerateSpeech(ABC):
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
    def synthesize(self, request: SpeechRequest) -> SpeechResponse:
        """Synthesize text into speech audio."""

    @abstractmethod
    def uploadCloneAudio(
        self, request: UploadCloneAudioRequest
    ) -> UploadCloneAudioResponse:
        """Upload an audio file for voice cloning."""

    @abstractmethod
    def cloneVoice(self, request: VoiceCloneRequest) -> VoiceCloneResponse:
        """Clone a voice from an uploaded audio file."""

    @abstractmethod
    def listVoices(self, voice_type: str = "all") -> VoiceListResponse:
        """List available voices for the account."""

    @abstractmethod
    def deleteVoice(self, voice_id: str, voice_type: str) -> DeleteVoiceResponse:
        """Delete a generated or cloned voice."""

    def _postJson(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._send(
            path,
            body,
            {
                "Content-Type": "application/json",
                **(headers or {}),
            },
        )

    def _postMultipart(
        self,
        path: str,
        *,
        fields: dict[str, str],
        file_field: str,
        file_path: str | Path,
        filename: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        boundary = f"----NianYiAgent{uuid.uuid4().hex}"
        body = self._buildMultipartBody(
            boundary=boundary,
            fields=fields,
            file_field=file_field,
            file_path=Path(file_path),
            filename=filename,
        )
        return self._send(
            path,
            body,
            {
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                **(headers or {}),
            },
        )

    def _send(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        if not self.endpoint:
            raise TtsServiceError("TTS endpoint is not configured.")

        request = Request(
            f"{self.endpoint}{path}",
            data=body,
            headers={
                **headers,
                **self._authHeaders(),
            },
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
            raise TtsServiceError(
                f"TTS HTTP error: {exc.code}",
                status_code=exc.code,
                response=error_response,
            ) from exc
        except URLError as exc:
            raise TtsServiceError(f"TTS request failed: {exc.reason}") from exc

        parsed = self._loadsJson(response_body)
        if not isinstance(parsed, dict):
            raise TtsServiceError("TTS response is not a JSON object.")
        return parsed

    def _authHeaders(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _buildMultipartBody(
        *,
        boundary: str,
        fields: dict[str, str],
        file_field: str,
        file_path: Path,
        filename: str | None = None,
    ) -> bytes:
        if not file_path.is_file():
            raise FileNotFoundError(f"clone audio file not found: {file_path}")

        body = bytearray()
        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            )
            body.extend(str(value).encode())
            body.extend(b"\r\n")

        upload_name = filename or file_path.name
        content_type = (
            mimetypes.guess_type(upload_name)[0] or "application/octet-stream"
        )
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{upload_name}"\r\n'
            ).encode()
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        return bytes(body)

    @staticmethod
    def _loadsJson(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise TtsServiceError("TTS response is not valid JSON.") from exc

    @staticmethod
    def _tryLoadsJson(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
