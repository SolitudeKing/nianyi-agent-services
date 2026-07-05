from __future__ import annotations

from services import ImageRequest, ImageService, SpeechRequest, SpeechService
from services.image import MinimaxImage, OpenaiImage
from services.tts import MinimaxSpeech


def test_public_image_provider_factory() -> None:
    service = ImageService.fromProvider("openai", api_key="test-key")

    assert isinstance(service.generator, OpenaiImage)


def test_public_minimax_image_provider_factory() -> None:
    service = ImageService.fromProvider("minimax", api_key="test-key")

    assert isinstance(service.generator, MinimaxImage)


def test_public_speech_provider_factory() -> None:
    service = SpeechService.fromProvider("minimax", api_key="test-key")

    assert isinstance(service.generator, MinimaxSpeech)


def test_image_reference_data_url() -> None:
    request = ImageRequest(
        model="test-model",
        prompt="test prompt",
        reference_image_bytes=b"abc",
        reference_mime="image/png",
    )

    assert request.referenceDataUrl() == "data:image/png;base64,YWJj"


def test_speech_request_keeps_business_payload_separate() -> None:
    request = SpeechRequest(
        model="test-model",
        text="hello",
        voice_id="voice-1",
        extra={"custom": "value"},
    )

    assert request.extra == {"custom": "value"}
