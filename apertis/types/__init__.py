"""Type definitions for the Apertis SDK."""

from __future__ import annotations

from apertis.types.chat import (
    # Response types
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionChunkChoice,
    ChatCompletionMessage,
    ChatCompletionDelta,
    ToolCall,
    Function,
    URLCitation,
    AudioData,
    # Request parameter types
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    FunctionDefinition,
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
    GeminiThinkingConfig,
    StreamOptions,
)
from apertis.types.embeddings import (
    Embedding,
    EmbeddingResponse,
    EmbeddingUsage,
)
from apertis.types.models import (
    Model,
    ModelList,
)
from apertis.types.responses import (
    Response,
    ResponseOutput,
    ResponseContent,
    ResponseTextContent,
    ResponseReasoningContent,
    ResponseUsage,
    ResponseInputItem,
)
from apertis.types.messages import (
    Message,
    MessageParam,
    MessageUsage,
    ContentBlock,
    TextBlock,
    ToolUseBlock,
    ToolDefinition,
)
from apertis.types.rerank import (
    RerankResponse,
    RerankResult,
    RerankUsage,
)
from apertis.types.shared import (
    Usage,
)

__all__ = [
    # Chat - Response types
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionChoice",
    "ChatCompletionChunkChoice",
    "ChatCompletionMessage",
    "ChatCompletionDelta",
    "ToolCall",
    "Function",
    "URLCitation",
    "AudioData",
    # Chat - Request parameter types
    "ChatCompletionMessageParam",
    "ChatCompletionToolParam",
    "FunctionDefinition",
    # Chat - Content part types
    "ContentPart",
    "TextContentPart",
    "ImageURLContentPart",
    "ImageURLDetail",
    "VideoURLContentPart",
    "VideoURLDetail",
    "InputAudioContentPart",
    "InputAudioDetail",
    # Chat - Config types
    "AudioConfig",
    "WebSearchOptions",
    "UserLocation",
    "ReasoningConfig",
    "ThinkingConfig",
    "GeminiThinkingConfig",
    "StreamOptions",
    # Embeddings
    "Embedding",
    "EmbeddingResponse",
    "EmbeddingUsage",
    # Models
    "Model",
    "ModelList",
    # Responses API
    "Response",
    "ResponseOutput",
    "ResponseContent",
    "ResponseTextContent",
    "ResponseReasoningContent",
    "ResponseUsage",
    "ResponseInputItem",
    # Messages API
    "Message",
    "MessageParam",
    "MessageUsage",
    "ContentBlock",
    "TextBlock",
    "ToolUseBlock",
    "ToolDefinition",
    # Rerank API
    "RerankResponse",
    "RerankResult",
    "RerankUsage",
    # Shared
    "Usage",
]
