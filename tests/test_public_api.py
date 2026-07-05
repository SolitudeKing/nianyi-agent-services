from __future__ import annotations

from services import (
    ChatMessage,
    ChatRequest,
    ChatService,
    ImageRequest,
    ImageService,
    SpeechRequest,
    SpeechService,
)
from services.chat import MinimaxChat, OpenaiChat
from services.image import MinimaxImage, OpenaiImage
from services.tts import MinimaxSpeech


def test_public_chat_provider_factory() -> None:
    service = ChatService.fromProvider("openai", api_key="test-key")

    assert isinstance(service.generator, OpenaiChat)


def test_public_minimax_chat_provider_factory() -> None:
    service = ChatService.fromProvider("minimax", api_key="test-key")

    assert isinstance(service.generator, MinimaxChat)


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


def test_chat_message_builds_payload_with_minimax_reasoning_fields() -> None:
    message = ChatMessage(
        role="assistant",
        content="done",
        reasoning_details=[{"text": "thinking"}],
        extra={"custom": "value"},
    )

    assert message.toPayload() == {
        "role": "assistant",
        "content": "done",
        "reasoning_details": [{"text": "thinking"}],
        "custom": "value",
    }


def test_openai_chat_builds_minimax_compatible_payload() -> None:
    chat = OpenaiChat(api_key="test-key")
    payload = chat._buildPayload(
        ChatRequest(
            model="MiniMax-M3",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                {"role": "user", "content": "Hi"},
            ],
            reasoning_split=True,
            thinking={"type": "disabled"},
            max_completion_tokens=128,
            extra_body={"service_tier": "priority"},
        )
    )

    assert payload == {
        "model": "MiniMax-M3",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        "max_completion_tokens": 128,
        "reasoning_split": True,
        "thinking": {"type": "disabled"},
        "service_tier": "priority",
    }


def test_openai_chat_parses_content_and_reasoning_details() -> None:
    response = OpenaiChat._parseResponse(
        {
            "id": "chatcmpl-1",
            "model": "MiniMax-M3",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                        "reasoning_details": [{"text": "thinking"}],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        }
    )

    assert response.response_id == "chatcmpl-1"
    assert response.content == "hello"
    assert response.choices[0].message is not None
    assert response.choices[0].message.reasoning_details == [{"text": "thinking"}]
    assert response.usage is not None
    assert response.usage.total_tokens == 3
