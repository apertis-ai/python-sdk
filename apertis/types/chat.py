"""Chat completion type definitions."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict, Required, NotRequired


# =============================================================================
# Content Part Types (for multimodal messages)
# =============================================================================


class TextContentPart(TypedDict):
    """Text content part for multimodal messages."""

    type: Required[Literal["text"]]
    text: Required[str]


class ImageURLDetail(TypedDict, total=False):
    """Image URL details."""

    url: Required[str]
    detail: Literal["auto", "low", "high"]


class ImageURLContentPart(TypedDict):
    """Image content part for vision-enabled models."""

    type: Required[Literal["image_url"]]
    image_url: Required[ImageURLDetail]


class VideoURLDetail(TypedDict):
    """Video URL details."""

    url: Required[str]


class VideoURLContentPart(TypedDict):
    """Video content part for video-enabled models."""

    type: Required[Literal["video_url"]]
    video_url: Required[VideoURLDetail]


class InputAudioDetail(TypedDict):
    """Input audio details."""

    data: Required[str]  # Base64 encoded audio
    format: Required[Literal["wav", "mp3", "flac", "opus", "pcm16"]]


class InputAudioContentPart(TypedDict):
    """Audio content part for audio-enabled models."""

    type: Required[Literal["input_audio"]]
    input_audio: Required[InputAudioDetail]


# Union of all content part types
ContentPart = Union[
    TextContentPart,
    ImageURLContentPart,
    VideoURLContentPart,
    InputAudioContentPart,
]


# =============================================================================
# Configuration Types (for advanced parameters)
# =============================================================================


class AudioConfig(TypedDict, total=False):
    """Audio output configuration."""

    voice: Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
    format: Literal["wav", "mp3", "flac", "opus", "pcm16"]


class UserLocation(TypedDict, total=False):
    """User location for web search."""

    type: Literal["approximate"]
    approximate: Dict[str, str]  # {"country": "US", "city": "...", "region": "..."}


class WebSearchOptions(TypedDict, total=False):
    """Web search configuration for search-enabled models."""

    search_context_size: Literal["low", "medium", "high"]
    user_location: UserLocation
    filters: List[str]  # Domain allow-list


class ReasoningConfig(TypedDict, total=False):
    """Reasoning mode configuration for thinking models."""

    enabled: bool
    effort: Literal["low", "medium", "high"]
    summary: Literal["auto", "concise", "detailed"]


class ThinkingConfig(TypedDict, total=False):
    """Extended thinking configuration for Gemini models."""

    type: Literal["enabled", "disabled", "auto"]


class GeminiThinkingConfig(TypedDict, total=False):
    """Gemini-specific thinking configuration (via extra_body)."""

    thinking_budget: int


class StreamOptions(TypedDict, total=False):
    """Stream options for streaming responses."""

    include_usage: bool


# =============================================================================
# Response Types
# =============================================================================


class Function(BaseModel):
    """Function call details."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call in a message."""

    id: str
    type: Literal["function"]
    function: Function


class URLCitation(BaseModel):
    """URL citation annotation in web search responses."""

    type: Literal["url_citation"]
    start_index: int
    end_index: int
    url: str
    title: Optional[str] = None


class AudioData(BaseModel):
    """Audio data in response."""

    id: str
    data: str  # Base64 encoded audio
    expires_at: int
    transcript: Optional[str] = None


class ChatCompletionMessage(BaseModel):
    """Assistant message in a chat completion."""

    role: Literal["assistant"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    # Web search annotations
    annotations: Optional[List[URLCitation]] = None
    # Audio output
    audio: Optional[AudioData] = None
    # Reasoning details (for thinking models)
    reasoning_details: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChoice(BaseModel):
    """A choice in a chat completion response."""

    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    logprobs: Optional[Any] = None


class Usage(BaseModel):
    """Token usage for chat completion."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatCompletion(BaseModel):
    """Chat completion response."""

    id: Optional[str] = None
    object: Optional[Literal["chat.completion"]] = None
    created: Optional[int] = None
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


# =============================================================================
# Streaming Types
# =============================================================================


class ChatCompletionDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunkChoice(BaseModel):
    """A choice in a streaming chunk."""

    index: int
    delta: ChatCompletionDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk response."""

    id: Optional[str] = None
    object: Optional[Literal["chat.completion.chunk"]] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


# =============================================================================
# Request Parameter Types
# =============================================================================


class SystemMessageParam(TypedDict):
    """System message parameter."""

    role: Required[Literal["system"]]
    content: Required[str]
    name: NotRequired[str]


class UserMessageParam(TypedDict):
    """User message parameter. Content can be string or multimodal content parts."""

    role: Required[Literal["user"]]
    content: Required[Union[str, List[ContentPart]]]
    name: NotRequired[str]


class AssistantMessageParam(TypedDict, total=False):
    """Assistant message parameter."""

    role: Required[Literal["assistant"]]
    content: Optional[str]
    name: str
    tool_calls: List[Dict[str, Any]]
    # For multi-turn reasoning
    reasoning_details: List[Dict[str, Any]]


class ToolMessageParam(TypedDict):
    """Tool message parameter."""

    role: Required[Literal["tool"]]
    content: Required[str]
    tool_call_id: Required[str]


ChatCompletionMessageParam = Union[
    SystemMessageParam,
    UserMessageParam,
    AssistantMessageParam,
    ToolMessageParam,
]


class FunctionDefinition(TypedDict, total=False):
    """Function definition for tools."""

    name: Required[str]
    description: str
    parameters: Dict[str, Any]
    strict: bool


class ChatCompletionToolParam(TypedDict):
    """Tool parameter for chat completion."""

    type: Required[Literal["function"]]
    function: Required[FunctionDefinition]
