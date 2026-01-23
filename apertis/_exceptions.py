"""Exception classes for the Apertis SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Type

if TYPE_CHECKING:
    import httpx


class ApertisError(Exception):
    """Base exception for all Apertis SDK errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class APIError(ApertisError):
    """Error returned by the Apertis API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response: "httpx.Response",
        body: Optional[Dict] = None,
    ) -> None:
        self.status_code = status_code
        self.response = response
        self.body = body
        super().__init__(message)


class AuthenticationError(APIError):
    """Authentication failed (401)."""

    pass


class PermissionDeniedError(APIError):
    """Permission denied (403)."""

    pass


class NotFoundError(APIError):
    """Resource not found (404)."""

    pass


class UnprocessableEntityError(APIError):
    """Unprocessable entity (422)."""

    pass


class RateLimitError(APIError):
    """Rate limit exceeded (429)."""

    pass


class InternalServerError(APIError):
    """Internal server error (500+)."""

    pass


class APIConnectionError(ApertisError):
    """Failed to connect to the API."""

    def __init__(self, message: str, *, cause: Optional[Exception] = None) -> None:
        self.cause = cause
        super().__init__(message)


class APITimeoutError(APIConnectionError):
    """Request timed out."""

    pass


def _make_api_error(
    message: str,
    *,
    status_code: int,
    response: "httpx.Response",
    body: Optional[Dict] = None,
) -> APIError:
    """Create the appropriate APIError subclass based on status code."""
    error_class: Type[APIError]
    if status_code == 401:
        error_class = AuthenticationError
    elif status_code == 403:
        error_class = PermissionDeniedError
    elif status_code == 404:
        error_class = NotFoundError
    elif status_code == 422:
        error_class = UnprocessableEntityError
    elif status_code == 429:
        error_class = RateLimitError
    elif status_code >= 500:
        error_class = InternalServerError
    else:
        error_class = APIError

    return error_class(
        message,
        status_code=status_code,
        response=response,
        body=body,
    )
