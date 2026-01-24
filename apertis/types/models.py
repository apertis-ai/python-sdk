"""Model type definitions."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class Model(BaseModel):
    """A model available for use."""

    id: str
    object: Literal["model"]
    created: Optional[int] = None
    owned_by: Optional[str] = None


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"]
    data: List[Model]
