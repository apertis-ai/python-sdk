"""Audio resource: speech, transcriptions and translations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from apertis.types.audio import Transcription, Translation

if TYPE_CHECKING:
    import httpx

    from apertis._base_client import AsyncClient, SyncClient

# A path, an open binary file, raw bytes, or a (filename, bytes) tuple.
FileTypes = Union[str, "os.PathLike[str]", bytes, IO[bytes], Tuple[str, Union[bytes, IO[bytes]]]]
AudioResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]
_TEXT_FORMATS = {"text", "srt", "vtt"}


class BinaryResponseContent:
    """Binary body of an API response, such as generated speech."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response

    @property
    def content(self) -> bytes:
        return self.response.content

    def write_to_file(self, file: Union[str, os.PathLike[str]]) -> None:
        """Write the bytes to `file`."""
        Path(file).write_bytes(self.content)


def _read_file(file: FileTypes) -> Tuple[str, bytes]:
    """Return (filename, bytes), reading the file once so retries resend the same bytes."""
    if isinstance(file, tuple):
        name, content = file
        return name, content if isinstance(content, bytes) else content.read()
    if isinstance(file, bytes):
        return "upload", file
    if isinstance(file, (str, os.PathLike)):
        path = Path(file)
        return path.name, path.read_bytes()
    return os.path.basename(getattr(file, "name", "upload")), file.read()


def _form(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Multipart form fields: drop None, stringify scalars, keep lists as repeated fields."""
    form: Dict[str, Any] = {}
    for key, value in fields.items():
        if value is None:
            continue
        form[key] = [str(v) for v in value] if isinstance(value, list) else str(value)
    return form


def _speech_body(
    model: str,
    input: str,
    voice: str,
    instructions: Optional[str],
    response_format: Optional[str],
    speed: Optional[float],
    extra_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"model": model, "input": input, "voice": voice}
    if instructions is not None:
        body["instructions"] = instructions
    if response_format is not None:
        body["response_format"] = response_format
    if speed is not None:
        body["speed"] = speed
    if extra_body:
        body.update(extra_body)
    return body


def _parse_text_result(
    response: httpx.Response, response_format: Optional[str], model: Any
) -> Any:
    if response_format in _TEXT_FORMATS:
        return response.text
    return model.model_validate(response.json())


class Speech:
    """Text to speech (POST /audio/speech)."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: str,
        voice: str,
        instructions: Optional[str] = None,
        response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = None,
        speed: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> BinaryResponseContent:
        """Generate audio from text.

        Args:
            model: TTS model ID (e.g., "gpt-4o-mini-tts").
            input: Text to speak.
            voice: Voice name (e.g., "alloy").
            instructions: Style instructions, for models that support them.
            response_format: Audio format; the API default is mp3.
            speed: Playback speed, 0.25 to 4.0.
            extra_body: Extra fields merged into the request body.

        Returns:
            BinaryResponseContent; use `.content` or `.write_to_file(path)`.
        """
        body = _speech_body(model, input, voice, instructions, response_format, speed, extra_body)
        return BinaryResponseContent(self._client.request("POST", "/audio/speech", json=body))


class Transcriptions:
    """Speech to text (POST /audio/transcriptions)."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    def create(
        self,
        *,
        file: FileTypes,
        model: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[AudioResponseFormat] = None,
        temperature: Optional[float] = None,
        timestamp_granularities: Optional[List[Literal["word", "segment"]]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[Transcription, str]:
        """Transcribe audio.

        Args:
            file: Path, open binary file, bytes, or (filename, bytes). The filename's
                extension tells the API the audio format.
            model: Transcription model ID (e.g., "whisper-1").
            language: ISO-639-1 language of the audio.
            prompt: Text to guide the style or continue a previous segment.
            response_format: json (default), verbose_json, text, srt or vtt.
            temperature: Sampling temperature, 0 to 1.
            timestamp_granularities: "word" and/or "segment"; needs verbose_json.
            extra_body: Extra form fields.

        Returns:
            Transcription for json / verbose_json, otherwise the response text.
        """
        data = _form(
            {
                "model": model,
                "language": language,
                "prompt": prompt,
                "response_format": response_format,
                "temperature": temperature,
                "timestamp_granularities[]": timestamp_granularities,
                **(extra_body or {}),
            }
        )
        response = self._client.request(
            "POST", "/audio/transcriptions", data=data, files={"file": _read_file(file)}
        )
        result: Union[Transcription, str] = _parse_text_result(
            response, response_format, Transcription
        )
        return result


class Translations:
    """Speech to English text (POST /audio/translations)."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    def create(
        self,
        *,
        file: FileTypes,
        model: str,
        prompt: Optional[str] = None,
        response_format: Optional[AudioResponseFormat] = None,
        temperature: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[Translation, str]:
        """Translate audio into English text. See Transcriptions.create() for the arguments."""
        data = _form(
            {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "temperature": temperature,
                **(extra_body or {}),
            }
        )
        response = self._client.request(
            "POST", "/audio/translations", data=data, files={"file": _read_file(file)}
        )
        result: Union[Translation, str] = _parse_text_result(
            response, response_format, Translation
        )
        return result


class Audio:
    """Audio resource: `speech`, `transcriptions`, `translations`."""

    def __init__(self, client: SyncClient) -> None:
        self.speech = Speech(client)
        self.transcriptions = Transcriptions(client)
        self.translations = Translations(client)


class AsyncSpeech:
    """Asynchronous text to speech."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        input: str,
        voice: str,
        instructions: Optional[str] = None,
        response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = None,
        speed: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> BinaryResponseContent:
        """Generate audio from text. See Speech.create()."""
        body = _speech_body(model, input, voice, instructions, response_format, speed, extra_body)
        response = await self._client.request("POST", "/audio/speech", json=body)
        return BinaryResponseContent(response)


class AsyncTranscriptions:
    """Asynchronous speech to text."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        file: FileTypes,
        model: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[AudioResponseFormat] = None,
        temperature: Optional[float] = None,
        timestamp_granularities: Optional[List[Literal["word", "segment"]]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[Transcription, str]:
        """Transcribe audio. See Transcriptions.create()."""
        data = _form(
            {
                "model": model,
                "language": language,
                "prompt": prompt,
                "response_format": response_format,
                "temperature": temperature,
                "timestamp_granularities[]": timestamp_granularities,
                **(extra_body or {}),
            }
        )
        response = await self._client.request(
            "POST", "/audio/transcriptions", data=data, files={"file": _read_file(file)}
        )
        result: Union[Transcription, str] = _parse_text_result(
            response, response_format, Transcription
        )
        return result


class AsyncTranslations:
    """Asynchronous speech to English text."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        file: FileTypes,
        model: str,
        prompt: Optional[str] = None,
        response_format: Optional[AudioResponseFormat] = None,
        temperature: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Union[Translation, str]:
        """Translate audio into English text. See Transcriptions.create()."""
        data = _form(
            {
                "model": model,
                "prompt": prompt,
                "response_format": response_format,
                "temperature": temperature,
                **(extra_body or {}),
            }
        )
        response = await self._client.request(
            "POST", "/audio/translations", data=data, files={"file": _read_file(file)}
        )
        result: Union[Translation, str] = _parse_text_result(
            response, response_format, Translation
        )
        return result


class AsyncAudio:
    """Asynchronous audio resource."""

    def __init__(self, client: AsyncClient) -> None:
        self.speech = AsyncSpeech(client)
        self.transcriptions = AsyncTranscriptions(client)
        self.translations = AsyncTranslations(client)
