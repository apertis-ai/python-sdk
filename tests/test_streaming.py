"""Tests for streamed Apertis chat completions."""

from __future__ import annotations

import httpx
import respx

from apertis import Apertis, AsyncApertis


WEB_SEARCH_STREAM = b"\n".join(
    [
        b'data: {"id":"web-search","object":"chat.completion.chunk","created":1,"model":"","choices":[{"index":0,"delta":{"role":"assistant","content":"Web searching...\\n\\n"}}]}',
        b"",
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":2,"model":"gpt-5.5","choices":[{"index":0,"delta":{"content":"Result"}}]}',
        b"",
        b"data: [DONE]",
        b"",
    ]
)


@respx.mock
def test_sync_web_search_stream_status_is_not_dropped(client: Apertis) -> None:
    """Yield the preliminary Web Search event before model chunks."""
    respx.post("https://api.apertis.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=WEB_SEARCH_STREAM)
    )

    stream = client.chat.completions.create(
        model="gpt-5.5:web",
        messages=[{"role": "user", "content": "What changed?"}],
        stream=True,
    )
    chunks = list(stream)

    assert [chunk.choices[0].delta.content for chunk in chunks] == [
        "Web searching...\n\n",
        "Result",
    ]


@respx.mock
async def test_async_web_search_stream_status_is_not_dropped(
    async_client: AsyncApertis,
) -> None:
    """Async streams preserve the preliminary Web Search event too."""
    respx.post("https://api.apertis.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=WEB_SEARCH_STREAM)
    )

    stream = await async_client.chat.completions.create(
        model="gpt-5.5:web",
        messages=[{"role": "user", "content": "What changed?"}],
        stream=True,
    )
    chunks = [chunk async for chunk in stream]

    assert [chunk.choices[0].delta.content for chunk in chunks] == [
        "Web searching...\n\n",
        "Result",
    ]
