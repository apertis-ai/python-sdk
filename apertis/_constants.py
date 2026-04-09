"""Constants for the Apertis SDK."""

from __future__ import annotations

DEFAULT_BASE_URL = "https://api.apertis.ai/v1"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2

# Model IDs
CHAT_MODELS = [
    "gpt-5.4",
    "gpt-5.1",
    "gpt-5.3-codex",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4.5",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-preview",
    "glm-5.1",
    "minimax-m2.7",
]

EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
