"""Resource modules for the Apertis SDK."""

from __future__ import annotations

from apertis.resources.chat import Chat, AsyncChat
from apertis.resources.embeddings import Embeddings, AsyncEmbeddings
from apertis.resources.models import Models, AsyncModels
from apertis.resources.responses import Responses, AsyncResponses
from apertis.resources.messages import Messages, AsyncMessages
from apertis.resources.rerank import Rerank, AsyncRerank

__all__ = [
    "Chat",
    "AsyncChat",
    "Embeddings",
    "AsyncEmbeddings",
    "Models",
    "AsyncModels",
    "Responses",
    "AsyncResponses",
    "Messages",
    "AsyncMessages",
    "Rerank",
    "AsyncRerank",
]
