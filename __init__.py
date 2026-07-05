"""Compatibility entrypoint for direct Git submodule imports."""

from __future__ import annotations

from pathlib import Path

_SOURCE_PACKAGE = Path(__file__).resolve().parent / "src" / "services"
if _SOURCE_PACKAGE.is_dir():
    __path__.append(str(_SOURCE_PACKAGE))
else:
    raise ImportError("services source package is missing.")

from .chat import (  # noqa: E402
    ChatChoice,
    ChatMessage,
    ChatProvider,
    ChatRequest,
    ChatResponse,
    ChatService,
    ChatServiceError,
    ChatUsage,
    GenerateChat,
    MinimaxChat,
    OpenaiChat,
)
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
    "ChatChoice",
    "ChatMessage",
    "ChatProvider",
    "ChatRequest",
    "ChatResponse",
    "ChatService",
    "ChatServiceError",
    "ChatUsage",
    "ClonePrompt",
    "DeleteVoiceResponse",
    "GeneratedImage",
    "GenerateChat",
    "GenerateImage",
    "GenerateSpeech",
    "ImageGenerationError",
    "ImageProvider",
    "ImageRequest",
    "ImageResponse",
    "ImageService",
    "MinimaxChat",
    "MinimaxImage",
    "MinimaxSpeech",
    "OpenaiChat",
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
