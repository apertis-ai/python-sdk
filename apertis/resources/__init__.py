"""Resource modules for the Apertis SDK."""

from __future__ import annotations

from apertis.resources.chat import Chat, AsyncChat
from apertis.resources.embeddings import Embeddings, AsyncEmbeddings

__all__ = [
    "Chat",
    "AsyncChat",
    "Embeddings",
    "AsyncEmbeddings",
]
