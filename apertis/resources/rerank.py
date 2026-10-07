"""Rerank resource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from typing_extensions import deprecated

from apertis.types.rerank import RerankResponse

# ponytail: drop the decorator once the gateway serves POST /v1/rerank.
_UNAVAILABLE = (
    "client.rerank is not currently available: the Apertis API does not serve "
    "POST /v1/rerank, so calls return 404."
)

if TYPE_CHECKING:
    from apertis._base_client import AsyncClient, SyncClient


class Rerank:
    """Synchronous rerank resource."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    @deprecated(_UNAVAILABLE)
    def create(
        self,
        *,
        model: str,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
        return_documents: bool = False,
    ) -> RerankResponse:
        """Rerank documents by relevance to a query.

        Not currently available: the Apertis API does not serve POST /v1/rerank,
        so this call returns a 404 NotFoundError.

        Args:
            model: ID of the rerank model to use.
            query: The query to rank documents against.
            documents: List of documents to rerank.
            top_n: Number of top results to return. If not specified,
                   returns all documents.
            return_documents: Whether to include document text in results.

        Returns:
            RerankResponse with ranked results.
        """
        body = _build_request_body(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
        )

        response = self._client.request("POST", "/rerank", json=body)
        return RerankResponse.model_validate(response.json())


class AsyncRerank:
    """Asynchronous rerank resource."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    @deprecated(_UNAVAILABLE)
    async def create(
        self,
        *,
        model: str,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
        return_documents: bool = False,
    ) -> RerankResponse:
        """Rerank documents asynchronously.

        See Rerank.create() for parameter documentation.
        """
        body = _build_request_body(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
        )

        response = await self._client.request("POST", "/rerank", json=body)
        return RerankResponse.model_validate(response.json())


def _build_request_body(
    *,
    model: str,
    query: str,
    documents: Sequence[str],
    top_n: Optional[int],
    return_documents: bool,
) -> Dict[str, Any]:
    """Build request body for rerank."""
    body: Dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": list(documents),
        "return_documents": return_documents,
    }

    if top_n is not None:
        body["top_n"] = top_n

    return body
