"""Messages API type definitions (Anthropic native format)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict, Required, NotRequired


# =============================================================================
# Response Types
# =============================================================================


class TextBlock(BaseModel):
    """Text content block in a message."""

    type: Literal["text"]
    text: str


class ToolUseBlock(BaseModel):
    """Tool use content block in a message."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ThinkingBlock(BaseModel):
    """Extended thinking content block."""

    type: Literal["thinking"]
    thinking: str
    # Empty in a stream's content_block_start; the signature arrives as a signature_delta.
    signature: str = ""


class RedactedThinkingBlock(BaseModel):
    """Thinking content the provider returns encrypted."""

    type: Literal["redacted_thinking"]
    data: str


ContentBlock = Union[TextBlock, ToolUseBlock, ThinkingBlock, RedactedThinkingBlock]


StopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
    "model_context_window_exceeded",
]


class MessageUsage(BaseModel):
    """Token usage for messages."""

    input_tokens: int
    output_tokens: int


class Message(BaseModel):
    """Message response from the Messages API."""

    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    content: List[ContentBlock]
    model: str
    stop_reason: Optional[StopReason] = None
    stop_sequence: Optional[str] = None
    usage: MessageUsage


class MessageStreamDelta(BaseModel):
    """Delta in a stream event.

    content_block_delta deltas carry `type` (text_delta, input_json_delta,
    thinking_delta, signature_delta) and the matching field; message_delta
    deltas carry `stop_reason` and `stop_sequence` and have no `type`.
    """

    type: Optional[str] = None
    text: Optional[str] = None
    partial_json: Optional[str] = None
    thinking: Optional[str] = None
    signature: Optional[str] = None
    stop_reason: Optional[StopReason] = None
    stop_sequence: Optional[str] = None


class MessageDeltaUsage(BaseModel):
    """Cumulative usage sent with a message_delta event."""

    output_tokens: int
    input_tokens: Optional[int] = None


class MessageStreamEvent(BaseModel):
    """Streaming event from the Messages API."""

    type: Literal[
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    message: Optional[Message] = None
    index: Optional[int] = None
    content_block: Optional[ContentBlock] = None
    delta: Optional[MessageStreamDelta] = None
    usage: Optional[MessageDeltaUsage] = None


# =============================================================================
# Request Parameter Types
# =============================================================================


class TextContent(TypedDict):
    """Text content for message input."""

    type: Required[Literal["text"]]
    text: Required[str]


class ImageSource(TypedDict):
    """Image source for image content."""

    type: Required[Literal["base64", "url"]]
    media_type: NotRequired[str]
    data: NotRequired[str]
    url: NotRequired[str]


class ImageContent(TypedDict):
    """Image content for message input."""

    type: Required[Literal["image"]]
    source: Required[ImageSource]


class ToolResultContent(TypedDict):
    """Tool result content for message input."""

    type: Required[Literal["tool_result"]]
    tool_use_id: Required[str]
    content: Required[str]


MessageContent = Union[TextContent, ImageContent, ToolResultContent]


class MessageParam(TypedDict, total=False):
    """Message parameter for the Messages API."""

    role: Required[Literal["user", "assistant"]]
    content: Required[Union[str, List[MessageContent]]]


class ToolDefinition(TypedDict, total=False):
    """Tool definition for the Messages API."""

    name: Required[str]
    description: str
    input_schema: Dict[str, Any]
