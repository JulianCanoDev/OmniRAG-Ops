from __future__ import annotations

import logging
from typing import Any

from langchain.indexes import SQLRecordManager
from qdrant_client import QdrantClient, models
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import get_settings
from app.services.vector_service import get_qdrant_client

logger = logging.getLogger(__name__)

_DISTANCE_MAP: dict[str, models.Distance] = {
    "Cosine": models.Distance.COSINE,
    "Dot": models.Distance.DOT,
    "Euclid": models.Distance.EUCLID,
}


class CollectionService:

    @staticmethod
    def create(name: str, vector_size: int, distance: str) -> dict[str, Any]:
        client = get_qdrant_client()
        existing = client.get_collections().collections
        if any(c.name == name for c in existing):
            logger.warning("Collection '%s' already exists", name)
            return {"conflict": True, "name": name}

        distance_enum = _DISTANCE_MAP.get(distance, models.Distance.COSINE)

        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance_enum,
            ),
        )
        logger.info("Created collection '%s' (size=%d, distance=%s)", name, vector_size, distance)
        return {"conflict": False, "name": name}

    @staticmethod
    async def delete(name: str) -> int:
        client = get_qdrant_client()

        collections = client.get_collections().collections
        if not any(c.name == name for c in collections):
            logger.warning("Collection '%s' not found", name)
            return 0

        client.delete_collection(collection_name=name)
        logger.info("Deleted collection '%s' from Qdrant", name)

        settings = get_settings()
        try:
            from app.services.ingestion import get_engine
            engine: AsyncEngine = get_engine()
            record_manager = SQLRecordManager(
                namespace=settings.RECORD_MANAGER_NAMESPACE,
                engine=engine.sync_engine,
            )
            record_manager.create_schema()
            record_manager.delete_session(settings.RECORD_MANAGER_NAMESPACE)
            logger.info("Cleared SQLRecordManager namespace '%s'", name)
        except Exception:
            logger.exception("Failed to clear RecordManager for '%s'", name)

        return 1

    @staticmethod
    def list_all() -> list[dict[str, Any]]:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        result: list[dict[str, Any]] = []
        for col in collections:
            info = client.get_collection(collection_name=col.name)
            result.append(
                {
                    "name": col.name,
                    "status": info.status,
                    "vectors_count": info.vectors_count,
                }
            )
        return result