"""Constants for the Apertis SDK."""

from __future__ import annotations

DEFAULT_BASE_URL = "https://api.apertis.ai/v1"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2

# Model IDs
CHAT_MODELS = [
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.1",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4.5",
    "claude-haiku-4.5",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-preview",
]

EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
