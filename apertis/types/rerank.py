"""Rerank API type definitions."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class RerankResult(BaseModel):
    """A single rerank result."""

    index: int
    relevance_score: float
    document: Optional[str] = None


class RerankUsage(BaseModel):
    """Token usage for rerank."""

    total_tokens: int


class RerankResponse(BaseModel):
    """Response from the Rerank API."""

    object: Literal["list"]
    model: str
    results: List[RerankResult]
    usage: Optional[RerankUsage] = None
