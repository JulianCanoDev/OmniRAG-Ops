from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from PIL import Image
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import get_settings
from app.models.schemas import IngestionRequest, IngestionResponse
from app.services.gemini_service import enrich_metadata
from app.services.vector_service import index_documents

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"}


_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        logger.info("Created async SQLAlchemy engine for %s", settings.DATABASE_URL)
    return _engine


class _ExcelLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        import pandas as pd
        excel_file = pd.ExcelFile(self.file_path)
        docs: list[Document] = []
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            text = df.to_string(index=False)
            docs.append(
                Document(
                    page_content=text,
                    metadata={"sheet": sheet_name},
                )
            )
        return docs


class _ImageLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        img = Image.open(self.file_path)
        description = (
            f"Image file: {Path(self.file_path).name} | "
            f"Format: {img.format} | "
            f"Dimensions: {img.width}x{img.height} | "
            f"Mode: {img.mode}"
        )
        return [
            Document(
                page_content=description,
                metadata={
                    "file_name": Path(self.file_path).name,
                    "file_type": f"image/{img.format.lower()}" if img.format else "image/unknown",
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                },
            )
        ]


def _build_qdrant_client() -> QdrantClient:
    settings = get_settings()
    logger.debug("Connecting to remote Qdrant at %s", settings.QDRANT_URL)
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )


def _get_loader(file_path: str, file_extension: str) -> BaseLoader:
    ext = file_extension.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    if ext in (".docx", ".doc"):
        return Docx2txtLoader(file_path)
    if ext in (".xlsx", ".xls"):
        return _ExcelLoader(file_path)
    if ext in (".png", ".jpg", ".jpeg"):
        return _ImageLoader(file_path)
    raise ValueError(f"Unsupported file extension: {ext}")


def _extract_text(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs if d.page_content)


async def process_ingestion(
    payload: IngestionRequest | None = None,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    file_path: str | None = None,
) -> IngestionResponse:
    if payload is not None:
        content = payload.content
        source = payload.source
        raw_docs = [Document(page_content=content, metadata={"source": source})]
    elif file_path is not None and filename is not None:
        ext = Path(filename).suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            return IngestionResponse(
                status="error", message=f"Unsupported file extension: {ext}"
            )
        loader = _get_loader(file_path, ext)
        raw_docs = loader.load()
        content = _extract_text(raw_docs)
        source = filename
    elif file_bytes is not None and filename is not None:
        ext = Path(filename).suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            return IngestionResponse(
                status="error", message=f"Unsupported file extension: {ext}"
            )
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = _get_loader(tmp_path, ext)
            raw_docs = loader.load()
            content = _extract_text(raw_docs)
            source = filename
        finally:
            os.unlink(tmp_path)
    else:
        return IngestionResponse(
            status="error", message="No content or file provided"
        )

    metadata_enrichment = await enrich_metadata(content)

    doc_id = str(uuid.uuid4())
    enriched_metadata: dict[str, Any] = {
        "document_id": doc_id,
        "source": source,
        "summary": metadata_enrichment.summary,
        "category": metadata_enrichment.category,
        "priority": metadata_enrichment.priority,
    }

    for d in raw_docs:
        d.metadata.update(enriched_metadata)

    client = _build_qdrant_client()
    index_result: dict[str, Any] = index_documents(raw_docs, client=client)

    return IngestionResponse(
        status="success",
        document_id=doc_id,
        chunks_processed=index_result.get("chunks_processed", 0),
        metadata={
            "summary": metadata_enrichment.summary,
            "category": metadata_enrichment.category,
            "priority": metadata_enrichment.priority,
        },
        message="Document ingested successfully",
    )
