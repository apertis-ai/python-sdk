"""Tests for multimodal chat completions (vision, audio, video)."""

from __future__ import annotations

import httpx
import pytest
import respx

from apertis import Apertis


class TestImageContentPart:
    """Tests for image content in messages."""

    @respx.mock
    def test_image_url_in_message(self, client: Apertis) -> None:
        """Test sending an image URL in message content."""
        respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "This is an image of a cat.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            )
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/cat.jpg"},
                        },
                    ],
                }
            ],
        )

        assert response.choices[0].message.content == "This is an image of a cat."


class TestCreateWithImage:
    """Tests for create_with_image convenience method."""

    @respx.mock
    def test_create_with_image_url(self, client: Apertis) -> None:
        """Test create_with_image with URL."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "I see a dog.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create_with_image(
            model="gpt-4o",
            prompt="What is this?",
            image="https://example.com/dog.jpg",
        )

        assert response.choices[0].message.content == "I see a dog."

        # Verify request body
        import json
        request = route.calls.last.request
        data = json.loads(request.content)
        assert data["model"] == "gpt-4o"
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"

    @respx.mock
    def test_create_with_multiple_images(self, client: Apertis) -> None:
        """Test create_with_image with multiple images."""
        respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Both images show animals.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create_with_image(
            model="gpt-4o",
            prompt="What are these?",
            image=["https://example.com/cat.jpg", "https://example.com/dog.jpg"],
        )

        assert "animals" in response.choices[0].message.content.lower()

    @respx.mock
    def test_create_with_image_and_system(self, client: Apertis) -> None:
        """Test create_with_image with system message."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "A cat.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create_with_image(
            model="gpt-4o",
            prompt="What is this?",
            image="https://example.com/cat.jpg",
            system="Be brief.",
        )

        assert response.choices[0].message.content == "A cat."


class TestVideoContentPart:
    """Tests for video content in messages."""

    @respx.mock
    def test_video_url_in_message(self, client: Apertis) -> None:
        """Test sending a video URL in message content."""
        respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "glm-4.6v",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The video shows a person walking.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create(
            model="glm-4.6v",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is happening?"},
                        {
                            "type": "video_url",
                            "video_url": {"url": "https://example.com/video.mp4"},
                        },
                    ],
                }
            ],
        )

        assert "video" in response.choices[0].message.content.lower()


class TestAudioContentPart:
    """Tests for audio content in messages."""

    @respx.mock
    def test_audio_input_in_message(self, client: Apertis) -> None:
        """Test sending audio input in message content."""
        respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o-audio-preview",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The audio says hello.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What does this say?"},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": "base64audiodatahere",
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
        )

        assert "audio" in response.choices[0].message.content.lower()


class TestAudioOutput:
    """Tests for audio output in responses."""

    @respx.mock
    def test_audio_output_config(self, client: Apertis) -> None:
        """Test audio output configuration."""
        route = respx.post("https://api.apertis.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o-audio-preview",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello there!",
                                "audio": {
                                    "id": "audio-123",
                                    "data": "base64audiodata",
                                    "expires_at": 1234567899,
                                    "transcript": "Hello there!",
                                },
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )

        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            messages=[{"role": "user", "content": "Say hello"}],
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
        )

        assert response.choices[0].message.audio is not None
        assert response.choices[0].message.audio.transcript == "Hello there!"
