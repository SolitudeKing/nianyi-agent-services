from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .base import (
    AudioSetting,
    ClonePrompt,
    DeleteVoiceResponse,
    GenerateSpeech,
    SpeechRequest,
    SpeechResponse,
    SynthesizedAudio,
    TimbreWeight,
    TtsServiceError,
    UploadCloneAudioRequest,
    UploadCloneAudioResponse,
    UploadedFile,
    VoiceCloneRequest,
    VoiceCloneResponse,
    VoiceInfo,
    VoiceListResponse,
    VoiceSetting,
)


class MinimaxSpeech(GenerateSpeech):
    DEFAULT_ENDPOINT: str = "https://api.minimaxi.com"
    TTS_PATH: str = "/v1/t2a_v2"
    UPLOAD_FILE_PATH: str = "/v1/files/upload"
    VOICE_CLONE_PATH: str = "/v1/voice_clone"
    GET_VOICE_PATH: str = "/v1/get_voice"
    DELETE_VOICE_PATH: str = "/v1/delete_voice"

    VOICE_TYPES = {"system", "voice_cloning", "voice_generation", "all"}
    DELETABLE_VOICE_TYPES = {"voice_cloning", "voice_generation"}
    OUTPUT_FORMATS = {"hex", "url"}

    def synthesize(self, request: SpeechRequest) -> SpeechResponse:
        self._validateSpeechRequest(request)
        raw = self._postJson(self.TTS_PATH, self._buildSpeechPayload(request))
        self._raiseForMinimaxError(raw)
        return self._parseSpeechResponse(raw, request.output_format)

    def uploadCloneAudio(
        self, request: UploadCloneAudioRequest
    ) -> UploadCloneAudioResponse:
        raw = self._postMultipart(
            self.UPLOAD_FILE_PATH,
            fields={"purpose": request.purpose},
            file_field="file",
            file_path=request.file_path,
            filename=request.filename,
        )
        self._raiseForMinimaxError(raw)
        return UploadCloneAudioResponse(
            file=self._parseUploadedFile(raw.get("file")),
            raw=raw,
        )

    def cloneVoice(self, request: VoiceCloneRequest) -> VoiceCloneResponse:
        self._validateCloneRequest(request)
        raw = self._postJson(self.VOICE_CLONE_PATH, self._buildClonePayload(request))
        self._raiseForMinimaxError(raw)
        return VoiceCloneResponse(
            voice_id=request.voice_id,
            demo_audio=raw.get("demo_audio") or None,
            input_sensitive=raw.get("input_sensitive"),
            input_sensitive_type=raw.get("input_sensitive_type"),
            extra_info=raw.get("extra_info") or {},
            raw=raw,
        )

    def listVoices(self, voice_type: str = "all") -> VoiceListResponse:
        if voice_type not in self.VOICE_TYPES:
            raise ValueError(f"unsupported voice_type: {voice_type}")

        raw = self._postJson(self.GET_VOICE_PATH, {"voice_type": voice_type})
        self._raiseForMinimaxError(raw)
        return VoiceListResponse(
            system_voice=self._parseVoiceList(raw.get("system_voice")),
            voice_cloning=self._parseVoiceList(raw.get("voice_cloning")),
            voice_generation=self._parseVoiceList(raw.get("voice_generation")),
            raw=raw,
        )

    def deleteVoice(self, voice_id: str, voice_type: str) -> DeleteVoiceResponse:
        if not voice_id:
            raise ValueError("voice_id is required.")
        if voice_type not in self.DELETABLE_VOICE_TYPES:
            raise ValueError(f"unsupported deletable voice_type: {voice_type}")

        raw = self._postJson(
            self.DELETE_VOICE_PATH,
            {
                "voice_id": voice_id,
                "voice_type": voice_type,
            },
        )
        self._raiseForMinimaxError(raw)
        return DeleteVoiceResponse(
            voice_id=raw.get("voice_id"),
            created_time=raw.get("created_time"),
            raw=raw,
        )

    def _buildSpeechPayload(self, request: SpeechRequest) -> dict[str, Any]:
        voice_setting = request.voice_setting
        if voice_setting is None and request.voice_id:
            voice_setting = VoiceSetting(voice_id=request.voice_id)

        payload: dict[str, Any] = {
            "model": request.model,
            "text": request.text,
            "stream": request.stream,
            "output_format": request.output_format,
        }

        self._addIfPresent(payload, "stream_options", request.stream_options)
        self._addIfPresent(payload, "voice_setting", self._toPayload(voice_setting))
        self._addIfPresent(
            payload,
            "audio_setting",
            self._toPayload(request.audio_setting or AudioSetting()),
        )
        if request.pronunciation_tones:
            payload["pronunciation_dict"] = {"tone": request.pronunciation_tones}
        if request.timbre_weights:
            payload["timbre_weights"] = [
                self._toPayload(item) for item in request.timbre_weights
            ]
        self._addIfPresent(payload, "language_boost", request.language_boost)
        self._addIfPresent(payload, "voice_modify", request.voice_modify)
        self._addIfPresent(payload, "subtitle_enable", request.subtitle_enable)
        self._addIfPresent(payload, "subtitle_type", request.subtitle_type)
        self._addIfPresent(payload, "aigc_watermark", request.aigc_watermark)
        payload.update(request.extra)
        return payload

    def _buildClonePayload(self, request: VoiceCloneRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "file_id": request.file_id,
            "voice_id": request.voice_id,
        }
        self._addIfPresent(
            payload,
            "clone_prompt",
            self._toPayload(request.clone_prompt),
        )
        self._addIfPresent(payload, "text", request.text)
        self._addIfPresent(payload, "model", request.model)
        self._addIfPresent(payload, "language_boost", request.language_boost)
        self._addIfPresent(payload, "text_validation", request.text_validation)
        self._addIfPresent(payload, "accuracy", request.accuracy)
        self._addIfPresent(
            payload,
            "need_noise_reduction",
            request.need_noise_reduction,
        )
        self._addIfPresent(
            payload,
            "need_volume_normalization",
            request.need_volume_normalization,
        )
        self._addIfPresent(payload, "aigc_watermark", request.aigc_watermark)
        payload.update(request.extra)
        return payload

    def _validateSpeechRequest(self, request: SpeechRequest) -> None:
        if not request.text:
            raise ValueError("text is required.")
        if not request.model:
            raise ValueError("model is required.")
        if (
            not request.voice_id
            and not request.voice_setting
            and not request.timbre_weights
        ):
            raise ValueError("voice_id, voice_setting, or timbre_weights is required.")
        if request.stream:
            raise ValueError("streaming TTS is not supported by this HTTP wrapper.")
        if request.output_format not in self.OUTPUT_FORMATS:
            raise ValueError(f"unsupported output_format: {request.output_format}")

    @staticmethod
    def _validateCloneRequest(request: VoiceCloneRequest) -> None:
        if not request.file_id:
            raise ValueError("file_id is required.")
        if not request.voice_id:
            raise ValueError("voice_id is required.")
        if bool(request.text) != bool(request.model):
            raise ValueError("text and model must be set together for clone preview.")
        if request.clone_prompt:
            if not request.clone_prompt.prompt_audio:
                raise ValueError("clone_prompt.prompt_audio is required.")
            if not request.clone_prompt.prompt_text:
                raise ValueError("clone_prompt.prompt_text is required.")

    @classmethod
    def _parseSpeechResponse(
        cls,
        raw: dict[str, Any],
        output_format: str,
    ) -> SpeechResponse:
        data = raw.get("data") or {}
        extra_info = raw.get("extra_info") or {}
        audio_value = data.get("audio")
        return SpeechResponse(
            audio=cls._parseSynthesizedAudio(data, audio_value, output_format),
            trace_id=raw.get("trace_id"),
            extra_info=extra_info,
            raw=raw,
        )

    @classmethod
    def _parseSynthesizedAudio(
        cls,
        data: dict[str, Any],
        audio_value: str | None,
        output_format: str,
    ) -> SynthesizedAudio | None:
        if not data and not audio_value:
            return None

        audio_hex = audio_value if output_format == "hex" else None
        audio_url = audio_value if output_format == "url" else None
        return SynthesizedAudio(
            audio=audio_value,
            audio_hex=audio_hex,
            audio_url=audio_url,
            bytes_data=cls._decodeHexAudio(audio_hex),
            subtitle_file=data.get("subtitle_file"),
            status=data.get("status"),
        )

    @staticmethod
    def _parseUploadedFile(value: Any) -> UploadedFile | None:
        if not isinstance(value, dict):
            return None
        return UploadedFile(
            file_id=value.get("file_id"),
            bytes=value.get("bytes"),
            created_at=value.get("created_at"),
            filename=value.get("filename"),
            purpose=value.get("purpose"),
            raw=value,
        )

    @classmethod
    def _parseVoiceList(cls, value: Any) -> list[VoiceInfo]:
        if not isinstance(value, list):
            return []
        return [cls._parseVoiceInfo(item) for item in value if isinstance(item, dict)]

    @staticmethod
    def _parseVoiceInfo(value: dict[str, Any]) -> VoiceInfo:
        description = value.get("description")
        return VoiceInfo(
            voice_id=value.get("voice_id"),
            voice_name=value.get("voice_name"),
            description=description if isinstance(description, list) else [],
            created_time=value.get("created_time"),
            raw=value,
        )

    @staticmethod
    def _decodeHexAudio(value: str | None) -> bytes | None:
        if not value:
            return None
        try:
            return bytes.fromhex(value)
        except ValueError as exc:
            raise TtsServiceError("TTS audio payload is not valid hex.") from exc

    @staticmethod
    def _raiseForMinimaxError(raw: dict[str, Any]) -> None:
        base_resp = raw.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code in (None, 0):
            return

        message = base_resp.get("status_msg") or "MiniMax TTS request failed."
        raise TtsServiceError(
            message,
            status_code=status_code,
            response=raw,
        )

    @classmethod
    def _toPayload(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return [cls._toPayload(item) for item in value]
        if isinstance(value, dict):
            return {
                key: cls._toPayload(item)
                for key, item in value.items()
                if item is not None
            }
        if is_dataclass(value):
            return cls._toPayload(asdict(value))
        return value

    @staticmethod
    def _addIfPresent(payload: dict[str, Any], key: str, value: Any) -> None:
        if value is not None and value != {}:
            payload[key] = value


__all__ = [
    "AudioSetting",
    "ClonePrompt",
    "MinimaxSpeech",
    "SpeechRequest",
    "TimbreWeight",
    "UploadCloneAudioRequest",
    "VoiceCloneRequest",
    "VoiceSetting",
]
