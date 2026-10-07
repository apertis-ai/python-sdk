"""Tests for the audio resource (speech, transcriptions, translations)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from apertis import Apertis, AsyncApertis
from apertis.types.audio import Transcription, Translation

BASE = "https://api.apertis.ai/v1"
AUDIO = b"ID3\x00fake-mp3-bytes"


def _multipart(request: httpx.Request) -> bytes:
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    return request.content


class TestSpeech:
    @respx.mock
    def test_speech_returns_bytes_and_writes_file(self, client: Apertis, tmp_path: Path) -> None:
        route = respx.post(f"{BASE}/audio/speech").mock(
            return_value=httpx.Response(200, content=AUDIO, headers={"content-type": "audio/mpeg"})
        )

        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts", input="Hello", voice="alloy", response_format="mp3", speed=1.25
        )

        body = json.loads(route.calls[0].request.content)
        assert body == {
            "model": "gpt-4o-mini-tts",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "mp3",
            "speed": 1.25,
        }
        assert speech.content == AUDIO
        out = tmp_path / "hello.mp3"
        speech.write_to_file(out)
        assert out.read_bytes() == AUDIO

    @respx.mock
    async def test_async_speech(self, async_client: AsyncApertis) -> None:
        respx.post(f"{BASE}/audio/speech").mock(return_value=httpx.Response(200, content=AUDIO))

        speech = await async_client.audio.speech.create(
            model="gpt-4o-mini-tts", input="Hello", voice="alloy"
        )

        assert speech.content == AUDIO


class TestTranscriptions:
    @respx.mock
    def test_transcription_from_path(self, client: Apertis, tmp_path: Path) -> None:
        audio_file = tmp_path / "meeting.mp3"
        audio_file.write_bytes(AUDIO)
        route = respx.post(f"{BASE}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json={"text": "Hello there.", "language": "en"})
        )

        result = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            language="en",
            temperature=0,
            timestamp_granularities=["word", "segment"],
        )

        content = _multipart(route.calls[0].request)
        assert b'name="file"; filename="meeting.mp3"' in content
        assert AUDIO in content
        assert b'name="model"\r\n\r\nwhisper-1' in content
        assert b'name="language"\r\n\r\nen' in content
        assert b'name="temperature"\r\n\r\n0' in content
        assert content.count(b'name="timestamp_granularities[]"') == 2
        assert isinstance(result, Transcription)
        assert result.text == "Hello there."
        # verbose_json fields the SDK does not model are kept.
        assert result.model_dump()["language"] == "en"

    @respx.mock
    def test_retry_resends_file_bytes(self, client: Apertis) -> None:
        route = respx.post(f"{BASE}/audio/transcriptions").mock(
            side_effect=[
                httpx.Response(503, json={"error": {"message": "busy"}}),
                httpx.Response(200, json={"text": "ok"}),
            ]
        )

        with open(__file__, "rb") as f:
            source = f.read()
            f.seek(0)
            result = client.audio.transcriptions.create(file=f, model="whisper-1")

        assert result.text == "ok"
        assert len(route.calls) == 2
        for call in route.calls:
            assert source in _multipart(call.request)

    @respx.mock
    def test_srt_returns_text(self, client: Apertis) -> None:
        srt = "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
        route = respx.post(f"{BASE}/audio/transcriptions").mock(
            return_value=httpx.Response(200, text=srt, headers={"content-type": "text/plain"})
        )

        result = client.audio.transcriptions.create(
            file=("clip.wav", AUDIO), model="whisper-1", response_format="srt"
        )

        assert result == srt
        content = _multipart(route.calls[0].request)
        assert b'filename="clip.wav"' in content
        assert b'name="response_format"\r\n\r\nsrt' in content

    @respx.mock
    async def test_async_transcription_from_bytes(self, async_client: AsyncApertis) -> None:
        route = respx.post(f"{BASE}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json={"text": "async ok"})
        )

        result = await async_client.audio.transcriptions.create(file=AUDIO, model="whisper-1")

        assert result.text == "async ok"
        assert AUDIO in _multipart(route.calls[0].request)


class TestTranslations:
    @respx.mock
    def test_translation(self, client: Apertis) -> None:
        route = respx.post(f"{BASE}/audio/translations").mock(
            return_value=httpx.Response(200, json={"text": "Good morning."})
        )

        result = client.audio.translations.create(
            file=("bonjour.m4a", AUDIO), model="whisper-1", prompt="greeting"
        )

        assert isinstance(result, Translation)
        assert result.text == "Good morning."
        content = _multipart(route.calls[0].request)
        assert b'name="prompt"\r\n\r\ngreeting' in content

    @respx.mock
    async def test_async_translation_text(self, async_client: AsyncApertis) -> None:
        respx.post(f"{BASE}/audio/translations").mock(
            return_value=httpx.Response(200, text="Good morning.")
        )

        result = await async_client.audio.translations.create(
            file=("bonjour.m4a", AUDIO), model="whisper-1", response_format="text"
        )

        assert result == "Good morning."
