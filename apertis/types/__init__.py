"""Type definitions for the Apertis SDK."""

from __future__ import annotations

from apertis.types.chat import (
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
)
from apertis.types.embeddings import (
    Embedding,
    EmbeddingResponse,
    EmbeddingUsage,
)
from apertis.types.shared import (
    Usage,
)

__all__ = [
    # Chat
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
    # Embeddings
    "Embedding",
    "EmbeddingResponse",
    "EmbeddingUsage",
    # Shared
    "Usage",
]
