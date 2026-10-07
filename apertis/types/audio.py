"""Audio API type definitions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Transcription(BaseModel):
    """Transcription result. verbose_json fields (segments, words, ...) are kept as extras."""

    model_config = ConfigDict(extra="allow")

    text: str


class Translation(BaseModel):
    """Translation result. verbose_json fields are kept as extras."""

    model_config = ConfigDict(extra="allow")

    text: str
