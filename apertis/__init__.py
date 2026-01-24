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
    # Chat types
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
    URLCitation,
    AudioData,
    # Content part types
    ContentPart,
    TextContentPart,
    ImageURLContentPart,
    ImageURLDetail,
    VideoURLContentPart,
    VideoURLDetail,
    InputAudioContentPart,
    InputAudioDetail,
    # Config types
    AudioConfig,
    WebSearchOptions,
    UserLocation,
    ReasoningConfig,
    ThinkingConfig,
    StreamOptions,
    # Embedding types
    Embedding,
    EmbeddingResponse,
    EmbeddingUsage,
    # Model types
    Model,
    ModelList,
    # Response types
    Response,
    ResponseOutput,
    ResponseUsage,
    ResponseInputItem,
    # Message types
    Message,
    MessageParam,
    MessageUsage,
    # Rerank types
    RerankResponse,
    RerankResult,
    # Shared types
    Usage,
)

__version__ = "0.2.0"

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
    "URLCitation",
    "AudioData",
    # Content part types
    "ContentPart",
    "TextContentPart",
    "ImageURLContentPart",
    "ImageURLDetail",
    "VideoURLContentPart",
    "VideoURLDetail",
    "InputAudioContentPart",
    "InputAudioDetail",
    # Config types
    "AudioConfig",
    "WebSearchOptions",
    "UserLocation",
    "ReasoningConfig",
    "ThinkingConfig",
    "StreamOptions",
    # Embedding types
    "Embedding",
    "EmbeddingResponse",
    "EmbeddingUsage",
    # Model types
    "Model",
    "ModelList",
    # Response types
    "Response",
    "ResponseOutput",
    "ResponseUsage",
    "ResponseInputItem",
    # Message types
    "Message",
    "MessageParam",
    "MessageUsage",
    # Rerank types
    "RerankResponse",
    "RerankResult",
    # Shared types
    "Usage",
]
