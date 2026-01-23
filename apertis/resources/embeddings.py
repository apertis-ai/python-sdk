"""Embeddings resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from apertis.types.embeddings import EmbeddingResponse

if TYPE_CHECKING:
    from apertis._base_client import AsyncClient, SyncClient


class Embeddings:
    """Synchronous embeddings resource."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: str | Sequence[str],
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResponse:
        """Create embeddings for the given input.

        Args:
            model: ID of the model to use.
            input: Input text to embed. Can be a string or list of strings.
            dimensions: The number of dimensions for the output embeddings.
            user: A unique identifier for the end-user.

        Returns:
            EmbeddingResponse containing the embeddings.
        """
        body = self._build_request_body(
            model=model,
            input=input,
            dimensions=dimensions,
            user=user,
        )

        response = self._client.request("POST", "/embeddings", json=body)
        return EmbeddingResponse.model_validate(response.json())

    def _build_request_body(
        self,
        *,
        model: str,
        input: str | Sequence[str],
        dimensions: int | None,
        user: str | None,
    ) -> dict[str, Any]:
        """Build the request body for embeddings."""
        body: dict[str, Any] = {
            "model": model,
            "input": input if isinstance(input, str) else list(input),
            "encoding_format": "float",
        }

        if dimensions is not None:
            body["dimensions"] = dimensions
        if user is not None:
            body["user"] = user

        return body


class AsyncEmbeddings:
    """Asynchronous embeddings resource."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        input: str | Sequence[str],
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResponse:
        """Create embeddings for the given input.

        Args:
            model: ID of the model to use.
            input: Input text to embed. Can be a string or list of strings.
            dimensions: The number of dimensions for the output embeddings.
            user: A unique identifier for the end-user.

        Returns:
            EmbeddingResponse containing the embeddings.
        """
        body = self._build_request_body(
            model=model,
            input=input,
            dimensions=dimensions,
            user=user,
        )

        response = await self._client.request("POST", "/embeddings", json=body)
        return EmbeddingResponse.model_validate(response.json())

    def _build_request_body(
        self,
        *,
        model: str,
        input: str | Sequence[str],
        dimensions: int | None,
        user: str | None,
    ) -> dict[str, Any]:
        """Build the request body for embeddings."""
        body: dict[str, Any] = {
            "model": model,
            "input": input if isinstance(input, str) else list(input),
            "encoding_format": "float",
        }

        if dimensions is not None:
            body["dimensions"] = dimensions
        if user is not None:
            body["user"] = user

        return body
