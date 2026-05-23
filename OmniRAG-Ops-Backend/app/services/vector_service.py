from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL_ID,
        google_api_key=settings.GOOGLE_API_KEY,
        output_dimensionality=settings.EMBEDDING_OUTPUT_DIMENSIONALITY,
    )


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    logger.debug("Connecting to remote Qdrant at %s", settings.QDRANT_URL)
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )


def _get_vector_store(client: QdrantClient) -> QdrantVectorStore:
    settings = get_settings()
    return QdrantVectorStore(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding=_get_embeddings(),
    )


def _ensure_collection_exists(client: QdrantClient) -> None:
    settings = get_settings()
    collections = client.get_collections().collections
    if not any(c.name == settings.COLLECTION_NAME for c in collections):
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_OUTPUT_DIMENSIONALITY,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'", settings.COLLECTION_NAME)


def get_record_manager() -> SQLRecordManager:
    settings = get_settings()
    from app.services.ingestion import get_engine
    engine: AsyncEngine = get_engine()
    record_manager = SQLRecordManager(
        namespace=settings.RECORD_MANAGER_NAMESPACE,
        engine=engine.sync_engine,
    )
    record_manager.create_schema()
    return record_manager


def index_documents(
    docs: list[Document], client: QdrantClient
) -> dict[str, Any]:
    settings = get_settings()
    _ensure_collection_exists(client)
    vector_store = _get_vector_store(client)
    record_manager = get_record_manager()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    split_docs = text_splitter.split_documents(docs)
    result = index(
        docs=split_docs,
        record_manager=record_manager,
        vector_store=vector_store,
        cleanup="incremental",
        source_id_key="source",
    )
    return {
        "status": result,
        "chunks_processed": len(split_docs),
    }


def delete_by_source(source_id: str, client: QdrantClient) -> int:
    settings = get_settings()
    points, _ = client.scroll(
        collection_name=settings.COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source", match=models.MatchValue(value=source_id)
                )
            ]
        ),
        limit=10000,
        with_payload=False,
        with_vectors=False,
    )
    if not points:
        return 0
    point_ids = [p.id for p in points]
    client.delete(
        collection_name=settings.COLLECTION_NAME,
        points_selector=models.PointIdsList(points=point_ids),
    )
    record_manager = get_record_manager()
    record_manager.delete([source_id])
    return len(point_ids)


def search_similar(
    query: str, top_k: int, client: QdrantClient
) -> list[tuple[Document, float]]:
    vector_store = _get_vector_store(client)
    results = vector_store.similarity_search_with_relevance_scores(
        query, k=top_k
    )
    return results


def scroll_points_paginated(
    client: QdrantClient,
    limit: int,
    offset: int | None = None,
) -> tuple[list[models.Record], int | None]:
    settings = get_settings()
    points, next_offset = client.scroll(
        collection_name=settings.COLLECTION_NAME,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    return points, next_offset


def scroll_all_points(
    client: QdrantClient,
) -> list[models.Record]:
    settings = get_settings()
    all_points: list[models.Record] = []
    next_offset: int | None = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=1000,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(batch)
        if next_offset is None:
            break
    return all_points


def update_payload_by_source(
    source_id: str, payload: dict[str, Any], client: QdrantClient
) -> int:
    settings = get_settings()
    points, _ = client.scroll(
        collection_name=settings.COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source", match=models.MatchValue(value=source_id)
                )
            ]
        ),
        limit=10000,
        with_payload=False,
        with_vectors=False,
    )
    if not points:
        return 0
    point_ids = [p.id for p in points]
    client.set_payload(
        collection_name=settings.COLLECTION_NAME,
        payload=payload,
        points=point_ids,
    )
    return len(point_ids)


async def check_connectivity() -> tuple[bool, str]:
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True, "reachable"
    except Exception as exc:
        return False, str(exc)