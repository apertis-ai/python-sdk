"""Chat completion type definitions."""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict, Required, NotRequired


class Function(BaseModel):
    """Function call details."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call in a message."""

    id: str
    type: Literal["function"]
    function: Function


class ChatCompletionMessage(BaseModel):
    """Assistant message in a chat completion."""

    role: Literal["assistant"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


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


# Streaming types


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


# Request parameter types


class SystemMessageParam(TypedDict):
    """System message parameter."""

    role: Required[Literal["system"]]
    content: Required[str]
    name: NotRequired[str]


class UserMessageParam(TypedDict):
    """User message parameter."""

    role: Required[Literal["user"]]
    content: Required[Union[str, List[dict]]]
    name: NotRequired[str]


class AssistantMessageParam(TypedDict):
    """Assistant message parameter."""

    role: Required[Literal["assistant"]]
    content: NotRequired[Optional[str]]
    name: NotRequired[str]
    tool_calls: NotRequired[List[dict]]


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


class FunctionDefinition(TypedDict):
    """Function definition for tools."""

    name: Required[str]
    description: NotRequired[str]
    parameters: NotRequired[dict]


class ChatCompletionToolParam(TypedDict):
    """Tool parameter for chat completion."""

    type: Required[Literal["function"]]
    function: Required[FunctionDefinition]
