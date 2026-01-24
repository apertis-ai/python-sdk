"""Chat completions resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Sequence, Union, overload

from apertis._helpers import (
    detect_audio_format,
    encode_audio,
    is_url,
    normalize_image_input,
)
from apertis._streaming import AsyncStream, Stream
from apertis.types.chat import (
    AudioConfig,
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ContentPart,
    ReasoningConfig,
    StreamOptions,
    ThinkingConfig,
    WebSearchOptions,
)

if TYPE_CHECKING:
    from apertis._base_client import AsyncClient, SyncClient


class Completions:
    """Synchronous chat completions resource."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: Literal[True],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        n: int | None = None,
        logit_bias: dict[str, int] | None = None,
        # Audio output
        modalities: Sequence[Literal["text", "audio"]] | None = None,
        audio: AudioConfig | None = None,
        # Web search
        web_search_options: WebSearchOptions | None = None,
        # Reasoning mode
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        # Extended thinking
        thinking: ThinkingConfig | None = None,
        # Stream options
        stream_options: StreamOptions | None = None,
        # Provider-specific params
        extra_body: dict[str, Any] | None = None,
    ) -> Stream: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: Literal[False] = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        n: int | None = None,
        logit_bias: dict[str, int] | None = None,
        # Audio output
        modalities: Sequence[Literal["text", "audio"]] | None = None,
        audio: AudioConfig | None = None,
        # Web search
        web_search_options: WebSearchOptions | None = None,
        # Reasoning mode
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        # Extended thinking
        thinking: ThinkingConfig | None = None,
        # Stream options
        stream_options: StreamOptions | None = None,
        # Provider-specific params
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        n: int | None = None,
        logit_bias: dict[str, int] | None = None,
        # Audio output
        modalities: Sequence[Literal["text", "audio"]] | None = None,
        audio: AudioConfig | None = None,
        # Web search
        web_search_options: WebSearchOptions | None = None,
        # Reasoning mode
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        # Extended thinking
        thinking: ThinkingConfig | None = None,
        # Stream options
        stream_options: StreamOptions | None = None,
        # Provider-specific params
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletion | Stream: ...

    def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        n: int | None = None,
        logit_bias: dict[str, int] | None = None,
        # Audio output
        modalities: Sequence[Literal["text", "audio"]] | None = None,
        audio: AudioConfig | None = None,
        # Web search
        web_search_options: WebSearchOptions | None = None,
        # Reasoning mode
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        # Extended thinking
        thinking: ThinkingConfig | None = None,
        # Stream options
        stream_options: StreamOptions | None = None,
        # Provider-specific params
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletion | Stream:
        """Create a chat completion.

        Args:
            model: ID of the model to use.
            messages: A list of messages comprising the conversation.
            stream: If True, returns a streaming response.
            temperature: Sampling temperature between 0 and 2.
            max_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling parameter.
            frequency_penalty: Penalty for token frequency.
            presence_penalty: Penalty for token presence.
            stop: Stop sequences.
            seed: Random seed for deterministic results.
            tools: A list of tools the model may call.
            tool_choice: Controls which tool is called.
            response_format: Format specification for the response.
            user: A unique identifier for the end-user.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of most likely tokens to return.
            n: Number of completions to generate.
            logit_bias: Map of token IDs to bias values.
            modalities: Output modalities (["text"] or ["text", "audio"]).
            audio: Audio output configuration (voice, format).
            web_search_options: Web search configuration for search-enabled models.
            reasoning: Reasoning mode configuration for thinking models.
            reasoning_effort: Reasoning effort level (low, medium, high).
            thinking: Extended thinking configuration for Gemini models.
            stream_options: Stream options (e.g., include_usage).
            extra_body: Additional provider-specific parameters.

        Returns:
            ChatCompletion or Stream depending on the stream parameter.
        """
        body = _build_request_body(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            user=user,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            n=n,
            logit_bias=logit_bias,
            modalities=modalities,
            audio=audio,
            web_search_options=web_search_options,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
            stream_options=stream_options,
            extra_body=extra_body,
        )

        if stream:
            response = self._client.stream("POST", "/chat/completions", json=body)
            return Stream(response)

        response = self._client.request("POST", "/chat/completions", json=body)
        return ChatCompletion.model_validate(response.json())

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def create_with_image(
        self,
        model: str,
        prompt: str,
        image: Union[str, List[str]],
        *,
        detail: Literal["auto", "low", "high"] = "auto",
        system: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion with image(s).

        Handles base64 encoding automatically for local files.

        Args:
            model: ID of the model to use (e.g., "gpt-4o", "gpt-4o-mini").
            prompt: Text prompt to accompany the image(s).
            image: Image URL(s) or local file path(s). Can be a single string
                   or a list of strings.
            detail: Image detail level ("auto", "low", "high").
            system: Optional system message.
            **kwargs: Additional parameters passed to create().

        Returns:
            ChatCompletion response.
        """
        # Normalize to list
        images = [image] if isinstance(image, str) else image

        # Build content parts
        content: List[ContentPart] = [{"type": "text", "text": prompt}]
        for img in images:
            image_url = normalize_image_input(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url, "detail": detail},
            })

        # Build messages
        messages: List[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        return self.create(model=model, messages=messages, **kwargs)

    def create_with_audio(
        self,
        model: str,
        prompt: str,
        audio: str,
        *,
        audio_format: str | None = None,
        output_voice: str | None = None,
        output_format: Literal["wav", "mp3", "flac", "opus", "pcm16"] = "wav",
        system: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion with audio input.

        Handles base64 encoding automatically for local files.

        Args:
            model: ID of the model to use (e.g., "gpt-4o-audio-preview").
            prompt: Text prompt to accompany the audio.
            audio: Local path to audio file.
            audio_format: Audio format (auto-detected from extension if not provided).
            output_voice: If set, enables audio output with this voice.
            output_format: Audio output format.
            system: Optional system message.
            **kwargs: Additional parameters passed to create().

        Returns:
            ChatCompletion response.
        """
        # Detect format and encode audio
        if audio_format is None:
            audio_format = detect_audio_format(audio)
        audio_data = encode_audio(audio)

        # Build content parts
        content: List[ContentPart] = [
            {"type": "text", "text": prompt},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_data, "format": audio_format},
            },
        ]

        # Build messages
        messages: List[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        # Add audio output config if voice is specified
        if output_voice:
            kwargs["modalities"] = ["text", "audio"]
            kwargs["audio"] = {"voice": output_voice, "format": output_format}

        return self.create(model=model, messages=messages, **kwargs)

    def create_with_web_search(
        self,
        prompt: str,
        *,
        model: str = "gpt-5-search-api",
        context_size: Literal["low", "medium", "high"] = "medium",
        allowed_domains: List[str] | None = None,
        country: str | None = None,
        city: str | None = None,
        region: str | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion with web search enabled.

        Args:
            prompt: The search/question prompt.
            model: ID of search-enabled model (default: "gpt-5-search-api").
            context_size: Amount of search context ("low", "medium", "high").
            allowed_domains: Domain allow-list for search results.
            country: Country for localized search results.
            city: City for localized search results.
            region: Region for localized search results.
            system: Optional system message.
            **kwargs: Additional parameters passed to create().

        Returns:
            ChatCompletion response with url_citation annotations.
        """
        # Build messages
        messages: List[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build web search options
        web_search_options: WebSearchOptions = {
            "search_context_size": context_size,
        }

        if allowed_domains:
            web_search_options["filters"] = allowed_domains

        if country or city or region:
            location: Dict[str, Any] = {"type": "approximate", "approximate": {}}
            if country:
                location["approximate"]["country"] = country
            if city:
                location["approximate"]["city"] = city
            if region:
                location["approximate"]["region"] = region
            web_search_options["user_location"] = location

        return self.create(
            model=model,
            messages=messages,
            web_search_options=web_search_options,
            **kwargs,
        )


class AsyncCompletions:
    """Asynchronous chat completions resource."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: Literal[True],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        n: int | None = None,
        logit_bias: dict[str, int] | None = None,
        modalities: Sequence[Literal["text", "audio"]] | None = None,
        audio: AudioConfig | None = None,
        web_search_options: WebSearchOptions | None = None,
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        thinking: ThinkingConfig | None = None,
        stream_options: StreamOptions | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncStream: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: Literal[False] = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        n: int | None = None,
        logit_bias: dict[str, int] | None = None,
        modalities: Sequence[Literal["text", "audio"]] | None = None,
        audio: AudioConfig | None = None,
        web_search_options: WebSearchOptions | None = None,
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        thinking: ThinkingConfig | None = None,
        stream_options: StreamOptions | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletion: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        n: int | None = None,
        logit_bias: dict[str, int] | None = None,
        modalities: Sequence[Literal["text", "audio"]] | None = None,
        audio: AudioConfig | None = None,
        web_search_options: WebSearchOptions | None = None,
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        thinking: ThinkingConfig | None = None,
        stream_options: StreamOptions | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletion | AsyncStream: ...

    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        user: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        n: int | None = None,
        logit_bias: dict[str, int] | None = None,
        modalities: Sequence[Literal["text", "audio"]] | None = None,
        audio: AudioConfig | None = None,
        web_search_options: WebSearchOptions | None = None,
        reasoning: ReasoningConfig | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        thinking: ThinkingConfig | None = None,
        stream_options: StreamOptions | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletion | AsyncStream:
        """Create a chat completion asynchronously.

        See Completions.create() for parameter documentation.
        """
        body = _build_request_body(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            user=user,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            n=n,
            logit_bias=logit_bias,
            modalities=modalities,
            audio=audio,
            web_search_options=web_search_options,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
            stream_options=stream_options,
            extra_body=extra_body,
        )

        if stream:
            response = await self._client.stream("POST", "/chat/completions", json=body)
            return AsyncStream(response)

        response = await self._client.request("POST", "/chat/completions", json=body)
        return ChatCompletion.model_validate(response.json())

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def create_with_image(
        self,
        model: str,
        prompt: str,
        image: Union[str, List[str]],
        *,
        detail: Literal["auto", "low", "high"] = "auto",
        system: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion with image(s) asynchronously.

        See Completions.create_with_image() for parameter documentation.
        """
        images = [image] if isinstance(image, str) else image
        content: List[ContentPart] = [{"type": "text", "text": prompt}]
        for img in images:
            image_url = normalize_image_input(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url, "detail": detail},
            })

        messages: List[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        return await self.create(model=model, messages=messages, **kwargs)

    async def create_with_audio(
        self,
        model: str,
        prompt: str,
        audio: str,
        *,
        audio_format: str | None = None,
        output_voice: str | None = None,
        output_format: Literal["wav", "mp3", "flac", "opus", "pcm16"] = "wav",
        system: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion with audio input asynchronously.

        See Completions.create_with_audio() for parameter documentation.
        """
        if audio_format is None:
            audio_format = detect_audio_format(audio)
        audio_data = encode_audio(audio)

        content: List[ContentPart] = [
            {"type": "text", "text": prompt},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_data, "format": audio_format},
            },
        ]

        messages: List[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        if output_voice:
            kwargs["modalities"] = ["text", "audio"]
            kwargs["audio"] = {"voice": output_voice, "format": output_format}

        return await self.create(model=model, messages=messages, **kwargs)

    async def create_with_web_search(
        self,
        prompt: str,
        *,
        model: str = "gpt-5-search-api",
        context_size: Literal["low", "medium", "high"] = "medium",
        allowed_domains: List[str] | None = None,
        country: str | None = None,
        city: str | None = None,
        region: str | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion with web search asynchronously.

        See Completions.create_with_web_search() for parameter documentation.
        """
        messages: List[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        web_search_options: WebSearchOptions = {
            "search_context_size": context_size,
        }

        if allowed_domains:
            web_search_options["filters"] = allowed_domains

        if country or city or region:
            location: Dict[str, Any] = {"type": "approximate", "approximate": {}}
            if country:
                location["approximate"]["country"] = country
            if city:
                location["approximate"]["city"] = city
            if region:
                location["approximate"]["region"] = region
            web_search_options["user_location"] = location

        return await self.create(
            model=model,
            messages=messages,
            web_search_options=web_search_options,
            **kwargs,
        )


# =============================================================================
# Shared helper function
# =============================================================================


def _build_request_body(
    *,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    stream: bool,
    temperature: float | None,
    max_tokens: int | None,
    top_p: float | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    stop: str | Sequence[str] | None,
    seed: int | None,
    tools: Sequence[ChatCompletionToolParam] | None,
    tool_choice: str | dict[str, Any] | None,
    response_format: dict[str, Any] | None,
    user: str | None,
    logprobs: bool | None,
    top_logprobs: int | None,
    n: int | None,
    logit_bias: dict[str, int] | None,
    modalities: Sequence[Literal["text", "audio"]] | None,
    audio: AudioConfig | None,
    web_search_options: WebSearchOptions | None,
    reasoning: ReasoningConfig | None,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    thinking: ThinkingConfig | None,
    stream_options: StreamOptions | None,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the request body for chat completions."""
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": stream,
    }

    # Standard parameters
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if top_p is not None:
        body["top_p"] = top_p
    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    if stop is not None:
        body["stop"] = stop
    if seed is not None:
        body["seed"] = seed
    if tools is not None:
        body["tools"] = list(tools)
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if response_format is not None:
        body["response_format"] = response_format
    if user is not None:
        body["user"] = user
    if logprobs is not None:
        body["logprobs"] = logprobs
    if top_logprobs is not None:
        body["top_logprobs"] = top_logprobs
    if n is not None:
        body["n"] = n
    if logit_bias is not None:
        body["logit_bias"] = logit_bias

    # Audio output
    if modalities is not None:
        body["modalities"] = list(modalities)
    if audio is not None:
        body["audio"] = audio

    # Web search
    if web_search_options is not None:
        body["web_search_options"] = web_search_options

    # Reasoning mode
    if reasoning is not None:
        body["reasoning"] = reasoning
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort

    # Extended thinking
    if thinking is not None:
        body["thinking"] = thinking

    # Stream options
    if stream_options is not None:
        body["stream_options"] = stream_options

    # Provider-specific extra params
    if extra_body is not None:
        body["extra_body"] = extra_body

    return body
