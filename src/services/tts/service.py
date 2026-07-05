from __future__ import annotations

from typing import Literal

from .base import (
    DeleteVoiceResponse,
    GenerateSpeech,
    SpeechRequest,
    SpeechResponse,
    UploadCloneAudioRequest,
    UploadCloneAudioResponse,
    VoiceCloneRequest,
    VoiceCloneResponse,
    VoiceListResponse,
)
from .minimax import MinimaxSpeech

SpeechProvider = Literal["minimax"]


class SpeechService:
    def __init__(self, generator: GenerateSpeech) -> None:
        self.generator = generator

    @classmethod
    def fromProvider(
        cls,
        provider: SpeechProvider,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 120,
    ) -> SpeechService:
        generators: dict[str, type[GenerateSpeech]] = {
            "minimax": MinimaxSpeech,
        }

        try:
            generator_cls = generators[provider]
        except KeyError as exc:
            raise ValueError(f"unsupported speech provider: {provider}") from exc

        return cls(generator_cls(api_key=api_key, endpoint=endpoint, timeout=timeout))

    def synthesize(self, request: SpeechRequest) -> SpeechResponse:
        return self.generator.synthesize(request)

    def uploadCloneAudio(
        self, request: UploadCloneAudioRequest
    ) -> UploadCloneAudioResponse:
        return self.generator.uploadCloneAudio(request)

    def cloneVoice(self, request: VoiceCloneRequest) -> VoiceCloneResponse:
        return self.generator.cloneVoice(request)

    def listVoices(self, voice_type: str = "all") -> VoiceListResponse:
        return self.generator.listVoices(voice_type)

    def deleteVoice(self, voice_id: str, voice_type: str) -> DeleteVoiceResponse:
        return self.generator.deleteVoice(voice_id, voice_type)
