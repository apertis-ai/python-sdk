# Apertis Python SDK

[![PyPI version](https://badge.fury.io/py/apertis.svg)](https://badge.fury.io/py/apertis)
[![Python Versions](https://img.shields.io/pypi/pyversions/apertis.svg)](https://pypi.org/project/apertis/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official Python SDK for the [Apertis AI API](https://apertis.ai).

## Installation

```bash
pip install apertis
```

## Quick Start

```python
from apertis import Apertis

client = Apertis(api_key="your-api-key")
# Or set APERTIS_API_KEY environment variable

response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## Features

- **Sync and Async Support**: Both synchronous and asynchronous clients
- **Streaming**: Real-time streaming for chat completions
- **Tool Calling**: Function/tool calling support
- **Embeddings**: Text embedding generation
- **Type Hints**: Full type annotations for IDE support
- **Automatic Retries**: Built-in retry logic for transient errors

## Usage

### Chat Completions

```python
from apertis import Apertis

client = Apertis()

response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Streaming

```python
stream = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Async Usage

```python
import asyncio
from apertis import AsyncApertis

async def main():
    client = AsyncApertis()

    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

    await client.close()

asyncio.run(main())
```

### Async Streaming

```python
async def stream_example():
    client = AsyncApertis()

    stream = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": "Tell me a joke"}],
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    await client.close()
```

### Tool Calling

```python
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto",
)

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Hello, world!",
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
```

### Batch Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Hello", "World", "How are you?"],
)

for item in response.data:
    print(f"Index {item.index}: {len(item.embedding)} dimensions")
```

## Error Handling

```python
from apertis import (
    Apertis,
    ApertisError,
    APIError,
    AuthenticationError,
    RateLimitError,
)

client = Apertis()

try:
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except AuthenticationError as e:
    print(f"Invalid API key: {e.message}")
except RateLimitError as e:
    print(f"Rate limited. Status: {e.status_code}")
except APIError as e:
    print(f"API error {e.status_code}: {e.message}")
except ApertisError as e:
    print(f"Error: {e.message}")
```

## Configuration

```python
client = Apertis(
    api_key="your-api-key",           # Or use APERTIS_API_KEY env var
    base_url="https://api.apertis.ai/v1",  # Custom base URL
    timeout=60.0,                      # Request timeout in seconds
    max_retries=2,                     # Number of retries for failed requests
    default_headers={"X-Custom": "value"},  # Additional headers
)
```

## Context Manager

```python
with Apertis() as client:
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": "Hello!"}]
    )
# Client is automatically closed

# Async version
async with AsyncApertis() as client:
    response = await client.chat.completions.create(...)
```

## Available Models

### Chat Models
- `gpt-5.2`
- `gpt-5.2-codex`
- `gpt-5.1`
- `claude-opus-4-5-20251101`
- `claude-sonnet-4.5`
- `claude-haiku-4.5`
- `gemini-3-pro-preview`
- `gemini-3-flash-preview`
- `gemini-2.5-flash-preview`

### Embedding Models
- `text-embedding-3-small`
- `text-embedding-3-large`
- `text-embedding-ada-002`

## Requirements

- Python 3.9+
- httpx
- pydantic

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
