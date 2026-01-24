"""Apertis Python SDK.

A Python client for the Apertis AI API.

Example:
    >>> from apertis import Apertis
    >>> client = Apertis()
    >>> response = client.chat.completions.create(
    ...     model="gpt-5.2",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>> print(response.choices[0].message.content)
"""

from __future__ import annotations

from apertis._client import Apertis, AsyncApertis
from apertis._exceptions import (
    ApertisError,
    APIError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    APITimeoutError,
)
from apertis.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionChunkChoice,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionDelta,
    ChatCompletionToolParam,
    FunctionDefinition,
    ToolCall,
    Function,
    Embedding,
    EmbeddingResponse,
    EmbeddingUsage,
    Usage,
)

__version__ = "0.1.1"

__all__ = [
    # Clients
    "Apertis",
    "AsyncApertis",
    # Exceptions
    "ApertisError",
    "APIError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    # Chat types
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionChoice",
    "ChatCompletionChunkChoice",
    "ChatCompletionMessage",
    "ChatCompletionMessageParam",
    "ChatCompletionDelta",
    "ChatCompletionToolParam",
    "FunctionDefinition",
    "ToolCall",
    "Function",
    # Embedding types
    "Embedding",
    "EmbeddingResponse",
    "EmbeddingUsage",
    # Shared types
    "Usage",
]
