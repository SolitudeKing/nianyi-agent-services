"""Compatibility entrypoint for direct Git submodule imports."""

from __future__ import annotations

from pathlib import Path

_SOURCE_PACKAGE = Path(__file__).resolve().parent / "src" / "services"
if _SOURCE_PACKAGE.is_dir():
    __path__.append(str(_SOURCE_PACKAGE))
else:
    raise ImportError("services source package is missing.")

from .image import (  # noqa: E402
    GeneratedImage,
    GenerateImage,
    ImageGenerationError,
    ImageProvider,
    ImageRequest,
    ImageResponse,
    ImageService,
    MinimaxImage,
    OpenaiImage,
)
from .tts import (  # noqa: E402
    AudioSetting,
    ClonePrompt,
    DeleteVoiceResponse,
    GenerateSpeech,
    MinimaxSpeech,
    SpeechProvider,
    SpeechRequest,
    SpeechResponse,
    SpeechService,
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

__version__ = "0.1.0"

__all__ = [
    "AudioSetting",
    "ClonePrompt",
    "DeleteVoiceResponse",
    "GeneratedImage",
    "GenerateImage",
    "GenerateSpeech",
    "ImageGenerationError",
    "ImageProvider",
    "ImageRequest",
    "ImageResponse",
    "ImageService",
    "MinimaxImage",
    "MinimaxSpeech",
    "OpenaiImage",
    "SpeechProvider",
    "SpeechRequest",
    "SpeechResponse",
    "SpeechService",
    "SynthesizedAudio",
    "TimbreWeight",
    "TtsServiceError",
    "UploadCloneAudioRequest",
    "UploadCloneAudioResponse",
    "UploadedFile",
    "VoiceCloneRequest",
    "VoiceCloneResponse",
    "VoiceInfo",
    "VoiceListResponse",
    "VoiceSetting",
    "__version__",
]
