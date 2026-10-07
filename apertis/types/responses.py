"""Responses API type definitions."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, field_validator
from typing_extensions import TypedDict, Required, NotRequired


# =============================================================================
# Response Types
# =============================================================================


class ResponseOutputText(BaseModel):
    """Text part of an output message."""

    type: Literal["output_text"]
    text: str
    annotations: List[Dict[str, Any]] = []


class ResponseOutputRefusal(BaseModel):
    """Refusal part of an output message."""

    type: Literal["refusal"]
    refusal: str


class ResponseTextContent(BaseModel):
    """Legacy text part; current responses use ResponseOutputText."""

    type: Literal["text"]
    text: str


class ResponseReasoningContent(BaseModel):
    """Legacy reasoning part; current responses use ResponseReasoningItem."""

    type: Literal["reasoning"]
    summary: Optional[List[Dict[str, Any]]] = None


ResponseContent = Union[
    ResponseOutputText, ResponseOutputRefusal, ResponseTextContent, ResponseReasoningContent
]


class ResponseOutput(BaseModel):
    """Assistant message item in a response's output."""

    type: Literal["message"]
    id: str
    status: Literal["in_progress", "completed", "incomplete", "cancelled"]
    role: Literal["assistant"]
    content: List[ResponseContent]


ResponseOutputMessage = ResponseOutput


class ResponseReasoningItem(BaseModel):
    """Reasoning item emitted by reasoning models."""

    type: Literal["reasoning"]
    id: str
    summary: List[Dict[str, Any]] = []
    content: Optional[List[Dict[str, Any]]] = None
    encrypted_content: Optional[str] = None
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class ResponseFunctionToolCall(BaseModel):
    """Function call the model asks the caller to run."""

    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str
    id: Optional[str] = None
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class ResponseUnknownOutputItem(BaseModel):
    """Output item of a type this SDK does not model; all fields are kept."""

    model_config = ConfigDict(extra="allow")

    type: str


ResponseOutputItem = Union[
    ResponseOutput, ResponseReasoningItem, ResponseFunctionToolCall, ResponseUnknownOutputItem
]

_OUTPUT_ITEM_TYPES: Dict[str, Type[BaseModel]] = {
    "message": ResponseOutput,
    "reasoning": ResponseReasoningItem,
    "function_call": ResponseFunctionToolCall,
}


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
    status: Literal["queued", "in_progress", "completed", "incomplete", "cancelled", "failed"]
    model: str
    output: List[ResponseOutputItem]
    usage: Optional[ResponseUsage] = None
    error: Optional[Dict[str, Any]] = None

    @field_validator("output", mode="before")
    @classmethod
    def _parse_output_items(cls, value: Any) -> Any:
        # Dispatch on "type" so a malformed known item raises instead of
        # degrading into an unknown one, and new item types are kept.
        if not isinstance(value, list):
            return value
        return [
            _OUTPUT_ITEM_TYPES.get(str(item.get("type")), ResponseUnknownOutputItem).model_validate(item)
            if isinstance(item, dict)
            else item
            for item in value
        ]

    @property
    def output_text(self) -> str:
        """Concatenated text of every output_text part in the message items."""
        return "".join(
            part.text
            for item in self.output
            if isinstance(item, ResponseOutput)
            for part in item.content
            if isinstance(part, (ResponseOutputText, ResponseTextContent))
        )


class ResponseStreamEvent(BaseModel):
    """One event of a streamed Responses API call.

    ``type`` names the event (``response.output_text.delta``, ``response.completed``, ...).
    Fields not declared here are kept and readable as attributes. Events that the gateway
    builds itself carry no ``sequence_number`` or ``item_id``.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    sequence_number: Optional[int] = None
    item_id: Optional[str] = None
    output_index: Optional[int] = None
    content_index: Optional[int] = None
    # A string for text, argument and audio deltas; Any so an unforeseen shape cannot end the stream.
    delta: Any = None
    text: Optional[str] = None
    item: Optional[Dict[str, Any]] = None
    part: Optional[Dict[str, Any]] = None
    response: Optional[Dict[str, Any]] = None


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


class ResponseInputText(TypedDict):
    """Text part of a Responses API input message."""

    type: Required[Literal["input_text"]]
    text: Required[str]


class ResponseInputImage(TypedDict, total=False):
    """Image part of a Responses API input message: an ``image_url`` or a ``file_id``."""

    type: Required[Literal["input_image"]]
    image_url: str  # URL or data URL
    file_id: str
    detail: Literal["low", "high", "auto"]


class ResponseInputFile(TypedDict, total=False):
    """File part of a Responses API input message."""

    type: Required[Literal["input_file"]]
    file_id: str
    file_data: str  # data URL, e.g. "data:application/pdf;base64,..."
    file_url: str
    filename: str


# The first three are the Responses API part types; the last two are kept for
# code written against SDK 0.4.0 and earlier.
ResponseInputContent = Union[
    ResponseInputText,
    ResponseInputImage,
    ResponseInputFile,
    ResponseInputTextContent,
    ResponseInputImageContent,
]


class ResponseInputItem(TypedDict, total=False):
    """Input item for responses."""

    type: Required[Literal["message"]]
    role: Required[Literal["user", "assistant", "system", "developer"]]
    content: Required[Union[str, List[ResponseInputContent]]]
