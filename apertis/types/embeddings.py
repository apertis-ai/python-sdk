"""Embedding type definitions."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel


class EmbeddingUsage(BaseModel):
    """Token usage for embeddings."""

    prompt_tokens: int
    total_tokens: int


class Embedding(BaseModel):
    """A single embedding."""

    object: Literal["embedding"]
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    object: Literal["list"]
    data: List[Embedding]
    model: str
    usage: EmbeddingUsage
