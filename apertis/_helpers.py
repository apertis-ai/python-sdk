"""Helper utilities for the Apertis SDK."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional, Union


# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Supported audio formats
AUDIO_EXTENSIONS = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".flac": "flac",
    ".opus": "opus",
    ".pcm": "pcm16",
}


def detect_media_type(file_path: Union[str, Path]) -> Optional[str]:
    """
    Detect the MIME type from file extension.

    Args:
        file_path: Path to the file.

    Returns:
        MIME type string or None if not recognized.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Check known image types
    if ext in IMAGE_EXTENSIONS:
        return IMAGE_EXTENSIONS[ext]

    # Check known audio types
    if ext in AUDIO_EXTENSIONS:
        return f"audio/{AUDIO_EXTENSIONS[ext]}"

    # Fall back to mimetypes library
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type


def detect_audio_format(file_path: Union[str, Path]) -> str:
    """
    Detect the audio format from file extension.

    Args:
        file_path: Path to the audio file.

    Returns:
        Audio format string (wav, mp3, flac, opus, pcm16).

    Raises:
        ValueError: If audio format is not supported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in AUDIO_EXTENSIONS:
        supported = ", ".join(AUDIO_EXTENSIONS.keys())
        raise ValueError(f"Unsupported audio format: {ext}. Supported: {supported}")

    return AUDIO_EXTENSIONS[ext]


def encode_file_to_base64(file_path: Union[str, Path]) -> str:
    """
    Read a file and encode it to base64.

    Args:
        file_path: Path to the file.

    Returns:
        Base64 encoded string.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def encode_image(image_path: Union[str, Path]) -> str:
    """
    Encode a local image file to a base64 data URL.

    Args:
        image_path: Path to the image file.

    Returns:
        Data URL string (e.g., "data:image/png;base64,...")

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image format is not supported.
    """
    path = Path(image_path)
    ext = path.suffix.lower()

    if ext not in IMAGE_EXTENSIONS:
        supported = ", ".join(IMAGE_EXTENSIONS.keys())
        raise ValueError(f"Unsupported image format: {ext}. Supported: {supported}")

    mime_type = IMAGE_EXTENSIONS[ext]
    base64_data = encode_file_to_base64(path)

    return f"data:{mime_type};base64,{base64_data}"


def encode_audio(audio_path: Union[str, Path]) -> str:
    """
    Encode a local audio file to base64.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Base64 encoded string.

    Raises:
        FileNotFoundError: If audio file does not exist.
        ValueError: If audio format is not supported.
    """
    path = Path(audio_path)
    ext = path.suffix.lower()

    if ext not in AUDIO_EXTENSIONS:
        supported = ", ".join(AUDIO_EXTENSIONS.keys())
        raise ValueError(f"Unsupported audio format: {ext}. Supported: {supported}")

    return encode_file_to_base64(path)


def is_url(path: str) -> bool:
    """
    Check if a string is a URL.

    Args:
        path: String to check.

    Returns:
        True if the string looks like a URL.
    """
    return path.startswith(("http://", "https://", "data:"))


def is_base64_data_url(path: str) -> bool:
    """
    Check if a string is a base64 data URL.

    Args:
        path: String to check.

    Returns:
        True if the string is a data URL.
    """
    return path.startswith("data:")


def normalize_image_input(image: str) -> str:
    """
    Normalize an image input to a URL or data URL.

    If the input is a URL, return it as-is.
    If the input is a local file path, encode it to a data URL.

    Args:
        image: URL or local file path.

    Returns:
        URL or data URL string.
    """
    if is_url(image):
        return image

    # Assume it's a local file path
    return encode_image(image)
