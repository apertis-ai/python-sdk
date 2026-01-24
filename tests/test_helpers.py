"""Tests for helper utilities."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import pytest

from apertis._helpers import (
    detect_audio_format,
    detect_media_type,
    encode_audio,
    encode_file_to_base64,
    encode_image,
    is_base64_data_url,
    is_url,
    normalize_image_input,
)


class TestDetectMediaType:
    """Tests for detect_media_type function."""

    def test_jpeg_extension(self) -> None:
        """Test JPEG detection."""
        assert detect_media_type("image.jpg") == "image/jpeg"
        assert detect_media_type("image.jpeg") == "image/jpeg"

    def test_png_extension(self) -> None:
        """Test PNG detection."""
        assert detect_media_type("image.png") == "image/png"

    def test_gif_extension(self) -> None:
        """Test GIF detection."""
        assert detect_media_type("image.gif") == "image/gif"

    def test_webp_extension(self) -> None:
        """Test WebP detection."""
        assert detect_media_type("image.webp") == "image/webp"

    def test_audio_extensions(self) -> None:
        """Test audio format detection."""
        assert detect_media_type("audio.wav") == "audio/wav"
        assert detect_media_type("audio.mp3") == "audio/mp3"
        assert detect_media_type("audio.flac") == "audio/flac"

    def test_unknown_extension(self) -> None:
        """Test unknown extension returns None or falls back."""
        result = detect_media_type("file.xyz")
        assert result is None or isinstance(result, str)


class TestDetectAudioFormat:
    """Tests for detect_audio_format function."""

    def test_wav_format(self) -> None:
        """Test WAV format detection."""
        assert detect_audio_format("audio.wav") == "wav"

    def test_mp3_format(self) -> None:
        """Test MP3 format detection."""
        assert detect_audio_format("audio.mp3") == "mp3"

    def test_flac_format(self) -> None:
        """Test FLAC format detection."""
        assert detect_audio_format("audio.flac") == "flac"

    def test_opus_format(self) -> None:
        """Test Opus format detection."""
        assert detect_audio_format("audio.opus") == "opus"

    def test_unsupported_format_raises(self) -> None:
        """Test unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported audio format"):
            detect_audio_format("audio.aac")


class TestEncodeFileToBase64:
    """Tests for encode_file_to_base64 function."""

    def test_encode_file(self) -> None:
        """Test encoding a file to base64."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            f.flush()
            result = encode_file_to_base64(f.name)
            assert result == base64.standard_b64encode(b"hello world").decode("utf-8")
            Path(f.name).unlink()

    def test_file_not_found(self) -> None:
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            encode_file_to_base64("/nonexistent/file.txt")


class TestEncodeImage:
    """Tests for encode_image function."""

    def test_encode_png_image(self) -> None:
        """Test encoding a PNG image."""
        # Create a minimal PNG file
        png_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00"
            b"\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe5'\xde\xfc"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            f.write(png_data)
            f.flush()
            result = encode_image(f.name)
            assert result.startswith("data:image/png;base64,")
            Path(f.name).unlink()

    def test_unsupported_format_raises(self) -> None:
        """Test unsupported image format raises ValueError."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as f:
            f.write(b"fake bmp data")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported image format"):
                encode_image(f.name)
            Path(f.name).unlink()


class TestEncodeAudio:
    """Tests for encode_audio function."""

    def test_encode_wav_audio(self) -> None:
        """Test encoding a WAV audio file."""
        wav_data = b"RIFF\x00\x00\x00\x00WAVEfmt "
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(wav_data)
            f.flush()
            result = encode_audio(f.name)
            assert result == base64.standard_b64encode(wav_data).decode("utf-8")
            Path(f.name).unlink()


class TestIsUrl:
    """Tests for is_url function."""

    def test_http_url(self) -> None:
        """Test HTTP URL detection."""
        assert is_url("http://example.com/image.png") is True

    def test_https_url(self) -> None:
        """Test HTTPS URL detection."""
        assert is_url("https://example.com/image.png") is True

    def test_data_url(self) -> None:
        """Test data URL detection."""
        assert is_url("data:image/png;base64,abc123") is True

    def test_local_path(self) -> None:
        """Test local path is not URL."""
        assert is_url("/path/to/file.png") is False
        assert is_url("./file.png") is False
        assert is_url("file.png") is False


class TestIsBase64DataUrl:
    """Tests for is_base64_data_url function."""

    def test_data_url(self) -> None:
        """Test data URL detection."""
        assert is_base64_data_url("data:image/png;base64,abc") is True

    def test_http_url(self) -> None:
        """Test HTTP URL is not data URL."""
        assert is_base64_data_url("https://example.com/image.png") is False

    def test_local_path(self) -> None:
        """Test local path is not data URL."""
        assert is_base64_data_url("/path/to/file.png") is False


class TestNormalizeImageInput:
    """Tests for normalize_image_input function."""

    def test_http_url_passthrough(self) -> None:
        """Test HTTP URL is passed through unchanged."""
        url = "https://example.com/image.png"
        assert normalize_image_input(url) == url

    def test_data_url_passthrough(self) -> None:
        """Test data URL is passed through unchanged."""
        data_url = "data:image/png;base64,abc123"
        assert normalize_image_input(data_url) == data_url

    def test_local_file_encoded(self) -> None:
        """Test local file is encoded to data URL."""
        png_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00"
            b"\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe5'\xde\xfc"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            f.write(png_data)
            f.flush()
            result = normalize_image_input(f.name)
            assert result.startswith("data:image/png;base64,")
            Path(f.name).unlink()
