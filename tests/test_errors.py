"""Tests for error handling."""

from __future__ import annotations

import httpx
import pytest
import respx

from apertis import (
    Apertis,
    APIError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    InternalServerError,
)


class TestErrorHandling:
    """Tests for error handling."""

    def test_authentication_error(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test handling of 401 authentication error."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                401,
                json={"error": {"message": "Invalid API key", "type": "auth_error"}},
            )
        )

        with pytest.raises(AuthenticationError) as exc_info:
            client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.message

    def test_rate_limit_error(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test handling of 429 rate limit error."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                429,
                json={"error": {"message": "Rate limit exceeded"}},
                headers={"retry-after": "60"},
            )
        )

        # Create a client with no retries to test the error directly
        client_no_retry = Apertis(api_key="test-key", max_retries=0)

        with pytest.raises(RateLimitError) as exc_info:
            client_no_retry.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert exc_info.value.status_code == 429
        assert exc_info.value.response.headers.get("retry-after") == "60"

    def test_not_found_error(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test handling of 404 not found error."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                404,
                json={"error": {"message": "Model not found"}},
            )
        )

        with pytest.raises(NotFoundError) as exc_info:
            client.chat.completions.create(
                model="nonexistent-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert exc_info.value.status_code == 404

    def test_internal_server_error(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test handling of 500 server error."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                500,
                json={"error": {"message": "Internal server error"}},
            )
        )

        # Create a client with no retries
        client_no_retry = Apertis(api_key="test-key", max_retries=0)

        with pytest.raises(InternalServerError) as exc_info:
            client_no_retry.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert exc_info.value.status_code == 500

    def test_generic_api_error(
        self, client: Apertis, mock_api: respx.MockRouter
    ) -> None:
        """Test handling of generic API error."""
        mock_api.post("/chat/completions").mock(
            return_value=httpx.Response(
                400,
                json={"error": {"message": "Bad request"}},
            )
        )

        with pytest.raises(APIError) as exc_info:
            client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert exc_info.value.status_code == 400
