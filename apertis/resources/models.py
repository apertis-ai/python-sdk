"""Models resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apertis.types.models import Model, ModelList

if TYPE_CHECKING:
    from apertis._base_client import AsyncClient, SyncClient


class Models:
    """Synchronous models resource."""

    def __init__(self, client: SyncClient) -> None:
        self._client = client

    def list(self) -> ModelList:
        """List all available models.

        Returns:
            ModelList containing all available models.
        """
        response = self._client.request("GET", "/models")
        return ModelList.model_validate(response.json())

    def retrieve(self, model_id: str) -> Model:
        """Retrieve details of a specific model.

        Args:
            model_id: The ID of the model to retrieve.

        Returns:
            Model object with model details.
        """
        response = self._client.request("GET", f"/models/{model_id}")
        return Model.model_validate(response.json())


class AsyncModels:
    """Asynchronous models resource."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def list(self) -> ModelList:
        """List all available models.

        Returns:
            ModelList containing all available models.
        """
        response = await self._client.request("GET", "/models")
        return ModelList.model_validate(response.json())

    async def retrieve(self, model_id: str) -> Model:
        """Retrieve details of a specific model.

        Args:
            model_id: The ID of the model to retrieve.

        Returns:
            Model object with model details.
        """
        response = await self._client.request("GET", f"/models/{model_id}")
        return Model.model_validate(response.json())
