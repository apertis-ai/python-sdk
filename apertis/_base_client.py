"""Base HTTP client for the Apertis SDK."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

import httpx

from apertis._constants import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from apertis._exceptions import (
    APIConnectionError,
    APITimeoutError,
    _make_api_error,
)


class BaseClient:
    """Base class for Apertis clients with shared HTTP logic."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("APERTIS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Pass api_key or set APERTIS_API_KEY environment variable."
            )

        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
        self._default_headers = dict(default_headers) if default_headers else {}

    def _build_headers(
        self, extra_headers: Optional[Mapping[str, str]] = None
    ) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self._default_headers,
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        return f"{self.base_url}{path}"


class SyncClient(BaseClient):
    """Synchronous HTTP client."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = httpx.Client(timeout=self.timeout)

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        stream: bool = False,
    ) -> httpx.Response:
        """Make an HTTP request."""
        url = self._build_url(path)
        request_headers = self._build_headers(headers)

        retries = 0
        last_exception: Optional[Exception] = None

        while retries <= self.max_retries:
            try:
                response = self._client.request(
                    method,
                    url,
                    json=json,
                    headers=request_headers,
                    extensions={"stream": stream} if stream else None,
                )

                if response.status_code >= 400:
                    # Don't retry client errors (4xx) except 429
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        self._raise_for_status(response)

                    # Retry server errors (5xx) and rate limits (429)
                    if retries < self.max_retries:
                        retries += 1
                        continue

                    self._raise_for_status(response)

                return response

            except httpx.TimeoutException as e:
                last_exception = APITimeoutError(f"Request timed out: {e}", cause=e)
                if retries < self.max_retries:
                    retries += 1
                    continue
                raise last_exception from e

            except httpx.ConnectError as e:
                last_exception = APIConnectionError(f"Connection error: {e}", cause=e)
                if retries < self.max_retries:
                    retries += 1
                    continue
                raise last_exception from e

        if last_exception:
            raise last_exception
        raise APIConnectionError("Unknown error occurred")

    def stream(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> httpx.Response:
        """Make a streaming HTTP request."""
        import json as json_lib
        url = self._build_url(path)
        request_headers = self._build_headers(headers)

        try:
            # Build request manually for streaming
            request = self._client.build_request(
                method,
                url,
                json=json,
                headers=request_headers,
            )
            response = self._client.send(request, stream=True)

            if response.status_code >= 400:
                self._raise_for_status(response)

            return response

        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}", cause=e) from e
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Connection error: {e}", cause=e) from e

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise an appropriate exception for error responses."""
        try:
            body = response.json()
            message = body.get("error", {}).get("message", response.text)
        except Exception:
            body = None
            message = response.text or f"HTTP {response.status_code}"

        raise _make_api_error(
            message,
            status_code=response.status_code,
            response=response,
            body=body,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "SyncClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AsyncClient(BaseClient):
    """Asynchronous HTTP client."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        stream: bool = False,
    ) -> httpx.Response:
        """Make an HTTP request."""
        url = self._build_url(path)
        request_headers = self._build_headers(headers)

        retries = 0
        last_exception: Optional[Exception] = None

        while retries <= self.max_retries:
            try:
                response = await self._client.request(
                    method,
                    url,
                    json=json,
                    headers=request_headers,
                    extensions={"stream": stream} if stream else None,
                )

                if response.status_code >= 400:
                    # Don't retry client errors (4xx) except 429
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        await self._raise_for_status(response)

                    # Retry server errors (5xx) and rate limits (429)
                    if retries < self.max_retries:
                        retries += 1
                        continue

                    await self._raise_for_status(response)

                return response

            except httpx.TimeoutException as e:
                last_exception = APITimeoutError(f"Request timed out: {e}", cause=e)
                if retries < self.max_retries:
                    retries += 1
                    continue
                raise last_exception from e

            except httpx.ConnectError as e:
                last_exception = APIConnectionError(f"Connection error: {e}", cause=e)
                if retries < self.max_retries:
                    retries += 1
                    continue
                raise last_exception from e

        if last_exception:
            raise last_exception
        raise APIConnectionError("Unknown error occurred")

    async def stream(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> httpx.Response:
        """Make a streaming HTTP request."""
        url = self._build_url(path)
        request_headers = self._build_headers(headers)

        try:
            # Build request manually for streaming
            request = self._client.build_request(
                method,
                url,
                json=json,
                headers=request_headers,
            )
            response = await self._client.send(request, stream=True)

            if response.status_code >= 400:
                await self._raise_for_status(response)

            return response

        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}", cause=e) from e
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Connection error: {e}", cause=e) from e

    async def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise an appropriate exception for error responses."""
        try:
            body = response.json()
            message = body.get("error", {}).get("message", response.text)
        except Exception:
            body = None
            message = response.text or f"HTTP {response.status_code}"

        raise _make_api_error(
            message,
            status_code=response.status_code,
            response=response,
            body=body,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
