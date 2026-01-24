"""Shared type definitions."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
