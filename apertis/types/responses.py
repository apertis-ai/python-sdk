"""Responses API type definitions."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict, Required, NotRequired


# =============================================================================
# Response Types
# =============================================================================


class ResponseTextContent(BaseModel):
    """Text content in a response."""

    type: Literal["text"]
    text: str


class ResponseReasoningContent(BaseModel):
    """Reasoning content in a response (for thinking models)."""

    type: Literal["reasoning"]
    summary: Optional[List[Dict[str, Any]]] = None


ResponseContent = Union[ResponseTextContent, ResponseReasoningContent]


class ResponseOutput(BaseModel):
    """Output item in a response."""

    type: Literal["message"]
    id: str
    status: Literal["completed", "incomplete", "cancelled"]
    role: Literal["assistant"]
    content: List[ResponseContent]


class ResponseUsage(BaseModel):
    """Token usage for responses."""

    input_tokens: int
    output_tokens: int
    total_tokens: Optional[int] = None


class Response(BaseModel):
    """Response from the Responses API."""

    id: str
    object: Literal["response"]
    created_at: int
    status: Literal["completed", "incomplete", "cancelled", "failed"]
    model: str
    output: List[ResponseOutput]
    usage: Optional[ResponseUsage] = None
    error: Optional[Dict[str, Any]] = None


# =============================================================================
# Request Parameter Types
# =============================================================================


class ResponseInputTextContent(TypedDict):
    """Text content for response input."""

    type: Required[Literal["text"]]
    text: Required[str]


class ResponseInputImageContent(TypedDict, total=False):
    """Image content for response input."""

    type: Required[Literal["image"]]
    source: Required[Dict[str, Any]]  # {"type": "base64", "media_type": "...", "data": "..."}


ResponseInputContent = Union[ResponseInputTextContent, ResponseInputImageContent]


class ResponseInputItem(TypedDict, total=False):
    """Input item for responses."""

    type: Required[Literal["message"]]
    role: Required[Literal["user", "assistant"]]
    content: Required[Union[str, List[ResponseInputContent]]]
