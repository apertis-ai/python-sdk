"""Main client classes for the Apertis SDK."""

from __future__ import annotations

from typing import Mapping

from apertis._base_client import AsyncClient, SyncClient
from apertis.resources.chat import AsyncChat, Chat
from apertis.resources.embeddings import AsyncEmbeddings, Embeddings


class Apertis:
    """Synchronous client for the Apertis API.

    Example:
        >>> from apertis import Apertis
        >>> client = Apertis(api_key="...")
        >>> response = client.chat.completions.create(
        ...     model="gpt-5.2",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.choices[0].message.content)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the Apertis client.

        Args:
            api_key: API key for authentication. If not provided, reads from
                APERTIS_API_KEY environment variable.
            base_url: Base URL for the API. Defaults to https://api.apertis.ai/v1
            timeout: Request timeout in seconds. Defaults to 60.
            max_retries: Maximum number of retries for failed requests. Defaults to 2.
            default_headers: Additional headers to include in all requests.
        """
        self._client = SyncClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self.chat = Chat(self._client)
        self.embeddings = Embeddings(self._client)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "Apertis":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AsyncApertis:
    """Asynchronous client for the Apertis API.

    Example:
        >>> from apertis import AsyncApertis
        >>> client = AsyncApertis(api_key="...")
        >>> response = await client.chat.completions.create(
        ...     model="gpt-5.2",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.choices[0].message.content)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the async Apertis client.

        Args:
            api_key: API key for authentication. If not provided, reads from
                APERTIS_API_KEY environment variable.
            base_url: Base URL for the API. Defaults to https://api.apertis.ai/v1
            timeout: Request timeout in seconds. Defaults to 60.
            max_retries: Maximum number of retries for failed requests. Defaults to 2.
            default_headers: Additional headers to include in all requests.
        """
        self._client = AsyncClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self.chat = AsyncChat(self._client)
        self.embeddings = AsyncEmbeddings(self._client)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def __aenter__(self) -> "AsyncApertis":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
