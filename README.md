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
- **Vision/Image**: Analyze images with multimodal models
- **Audio I/O**: Audio input and output support
- **Video**: Video content analysis
- **Web Search**: Real-time web search integration
- **Reasoning Mode**: Chain-of-thought reasoning support
- **Extended Thinking**: Deep thinking for complex problems
- **Messages API**: Anthropic-native message format
- **Responses API**: OpenAI Responses API format
- **Rerank API**: Document reranking for RAG
- **Models API**: List and retrieve available models
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

### Vision / Image Analysis

Analyze images using multimodal models with the convenience method:

```python
# Using image URL
response = client.chat.completions.create_with_image(
    model="gpt-4o",
    prompt="What is in this image?",
    image="https://example.com/photo.jpg",
)
print(response.choices[0].message.content)

# Using local file (automatically base64 encoded)
response = client.chat.completions.create_with_image(
    model="gpt-4o",
    prompt="Describe this image",
    image="/path/to/local/image.png",
)

# Multiple images
response = client.chat.completions.create_with_image(
    model="gpt-4o",
    prompt="Compare these images",
    image=["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
    system="Be detailed in your comparison.",
)
```

Or use the standard API format:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
        ]
    }]
)
```

### Audio Input/Output

```python
# Audio input
response = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What does this audio say?"},
            {"type": "input_audio", "input_audio": {"data": "base64_audio_data", "format": "wav"}}
        ]
    }]
)

# Audio output
response = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    messages=[{"role": "user", "content": "Say hello in a friendly voice"}],
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
)
print(response.choices[0].message.audio.transcript)
```

### Web Search

Enable real-time web search for up-to-date information:

```python
# Using convenience method
response = client.chat.completions.create_with_web_search(
    prompt="What are the latest news about AI?",
    context_size="high",
    country="US",
)

# Check citations
for annotation in response.choices[0].message.annotations or []:
    print(f"Source: {annotation.url}")

# Using standard API
response = client.chat.completions.create(
    model="gpt-5-search-api",
    messages=[{"role": "user", "content": "Latest Python releases?"}],
    web_search_options={
        "search_context_size": "medium",
        "filters": ["python.org", "github.com"],
    },
)
```

### Reasoning Mode

Enable chain-of-thought reasoning for complex problems:

```python
response = client.chat.completions.create(
    model="glm-4.7",
    messages=[{"role": "user", "content": "How many r's are in strawberry?"}],
    reasoning={"enabled": True, "effort": "high"},
)

# Or use shorthand
response = client.chat.completions.create(
    model="glm-4.7",
    messages=[{"role": "user", "content": "Solve this math problem..."}],
    reasoning_effort="high",
)
```

### Extended Thinking (Gemini)

Use extended thinking for deep analysis:

```python
response = client.chat.completions.create(
    model="gemini-3-pro-preview",
    messages=[{"role": "user", "content": "Analyze this complex problem..."}],
    thinking={"type": "enabled"},
)

# With custom thinking budget
response = client.chat.completions.create(
    model="gemini-3-pro-preview",
    messages=[{"role": "user", "content": "Deep analysis needed..."}],
    extra_body={
        "google": {"thinking_config": {"thinking_budget": 10240}}
    },
)
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

### Models API

List and retrieve available models:

```python
# List all models
models = client.models.list()
for model in models.data:
    print(f"{model.id} - owned by {model.owned_by}")

# Retrieve specific model
model = client.models.retrieve("gpt-5.2")
print(f"Model: {model.id}, Created: {model.created}")
```

### Messages API (Anthropic Format)

Use Anthropic-native message format:

```python
message = client.messages.create(
    model="claude-sonnet-4.5",
    messages=[{"role": "user", "content": "Hello, Claude!"}],
    max_tokens=1024,
    system="You are a helpful assistant.",
)

print(message.content[0].text)
print(f"Stop reason: {message.stop_reason}")
```

With tool use:

```python
message = client.messages.create(
    model="claude-sonnet-4.5",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }],
)

if message.stop_reason == "tool_use":
    for block in message.content:
        if block.type == "tool_use":
            print(f"Tool: {block.name}, Input: {block.input}")
```

### Responses API

Use OpenAI Responses API format for advanced use cases:

```python
response = client.responses.create(
    model="gpt-5-pro",
    input="Write a haiku about programming",
)

print(response.output[0].content[0].text)

# With reasoning
response = client.responses.create(
    model="o1-pro",
    input="Solve this complex problem...",
    reasoning={"effort": "high"},
)
```

### Rerank API

Rerank documents for RAG applications:

```python
results = client.rerank.create(
    model="BAAI/bge-reranker-v2-m3",
    query="What is machine learning?",
    documents=[
        "Machine learning is a subset of AI.",
        "The weather is nice today.",
        "Deep learning uses neural networks.",
    ],
    top_n=2,
)

for result in results.results:
    print(f"Index {result.index}: Score {result.relevance_score}")

# With document text in results
results = client.rerank.create(
    model="BAAI/bge-reranker-v2-m3",
    query="Python programming",
    documents=["Python is great", "Java is popular"],
    return_documents=True,
)

for result in results.results:
    print(f"{result.document}: {result.relevance_score}")
```

## Error Handling

```python
from apertis import (
    Apertis,
    ApertisError,
    APIError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
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
except NotFoundError as e:
    print(f"Resource not found: {e.message}")
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
- `gpt-5.2`, `gpt-5.2-codex`, `gpt-5.1`, `gpt-5-pro`
- `gpt-4o`, `gpt-4o-audio-preview`
- `claude-opus-4-5-20251101`, `claude-sonnet-4.5`, `claude-haiku-4.5`
- `gemini-3-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-flash-preview`
- `glm-4.7`, `glm-4.6v` (vision)

### Reasoning Models
- `o1-pro`, `glm-4.7` (with reasoning enabled)

### Search Models
- `gpt-5-search-api`

### Embedding Models
- `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`

### Rerank Models
- `BAAI/bge-reranker-v2-m3`
- `Qwen/Qwen3-Reranker-0.6B`, `Qwen/Qwen3-Reranker-4B`, `Qwen/Qwen3-Reranker-8B`

## Requirements

- Python 3.9+
- httpx
- pydantic

## Changelog

### v0.2.0
- Added Vision/Image support with `create_with_image()` convenience method
- Added Audio input/output support
- Added Video content support
- Added Web Search with `create_with_web_search()` convenience method
- Added Reasoning Mode support
- Added Extended Thinking (Gemini) support
- Added Models API (`client.models.list()`, `client.models.retrieve()`)
- Added Responses API (`client.responses.create()`)
- Added Messages API (`client.messages.create()`) for Anthropic format
- Added Rerank API (`client.rerank.create()`)
- Added helper utilities for base64 encoding

### v0.1.1
- Fixed API response compatibility issues

### v0.1.0
- Initial release with Chat Completions, Streaming, Tool Calling, and Embeddings

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
