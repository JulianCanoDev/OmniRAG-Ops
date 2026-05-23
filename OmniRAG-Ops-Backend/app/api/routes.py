from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query, Response

from app.core.security import get_current_user
from app.models.schemas import (
    IngestionRequest,
    IngestionResponse,
    HealthResponse,
    HealthComponent,
    DocumentListResponse,
    DocumentResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
    MetadataUpdateRequest,
    CollectionCreateRequest,
    CollectionListResponse,
    CollectionInfo,
)
from app.services.ingestion import process_ingestion
from app.services.management import (
    list_documents,
    get_document,
    delete_document,
    query_documents,
    get_statistics,
    update_document_metadata,
)
from app.services.gemini_service import check_connectivity as check_gemini
from app.services.vector_service import check_connectivity as check_qdrant
from app.services.collection_service import CollectionService

logger = logging.getLogger(__name__)

router = APIRouter()

_SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"}


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    summary="Ingest text content into the vector store",
)
async def ingest_text(
    payload: IngestionRequest,
    _user=Depends(get_current_user),
) -> IngestionResponse:
    return await process_ingestion(payload=payload)


@router.post(
    "/ingest/file",
    response_model=IngestionResponse,
    summary="Upload and ingest a PDF, Word, or Excel file",
)
async def ingest_file(
    file: UploadFile = File(...),
    _user=Depends(get_current_user),
) -> IngestionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}",
        )

    raw_bytes: bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    return await process_ingestion(
        file_path=tmp_path, filename=file.filename
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents with pagination",
)
async def get_documents(
    skip: int = Query(default=0, ge=0, description="Number of items to skip"),
    limit: int = Query(default=20, ge=1, le=200, description="Max items to return"),
) -> DocumentListResponse:
    return list_documents(skip=skip, limit=limit)


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentResponse,
    summary="Get detailed information about a specific document",
)
async def get_document_by_id(doc_id: str) -> DocumentResponse:
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(
            status_code=404, detail=f"Document '{doc_id}' not found"
        )
    return doc


@router.delete(
    "/documents/{source_id}",
    summary="Deep delete a document by source ID from Qdrant and RecordManager",
)
async def delete_document_by_source(
    source_id: str,
) -> dict[str, Any]:
    deleted = delete_document(source_id)
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No points found for source '{source_id}'",
        )
    return {"status": "deleted", "source_id": source_id, "points_removed": deleted}


@router.patch(
    "/documents/{source_id}/metadata",
    summary="Override Gemini-generated metadata tags for a document",
)
async def patch_document_metadata(
    source_id: str, body: MetadataUpdateRequest
) -> dict[str, Any]:
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(
            status_code=400, detail="No metadata fields provided to update"
        )
    affected = update_document_metadata(source_id, updates)
    if affected == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No points found for source '{source_id}'",
        )
    return {
        "status": "updated",
        "source_id": source_id,
        "points_affected": affected,
        "updated_fields": list(updates.keys()),
    }


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic search over ingested documents",
)
async def search(body: QueryRequest) -> QueryResponse:
    return query_documents(question=body.question, top_k=body.top_k)


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Aggregated statistics about the vector store",
)
async def stats() -> StatsResponse:
    return get_statistics()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check for all upstream services",
)
async def health(response: Response) -> dict[str, Any]:
    gemini_ok, gemini_msg = await check_gemini()
    qdrant_ok, qdrant_msg = await check_qdrant()

    all_ok = gemini_ok and qdrant_ok
    if not all_ok:
        response.status_code = 503

    return HealthResponse(
        status="healthy" if all_ok else "unavailable",
        gemini=HealthComponent(
            status="ok" if gemini_ok else "unreachable", detail=gemini_msg
        ),
        qdrant=HealthComponent(
            status="ok" if qdrant_ok else "unreachable", detail=qdrant_msg
        ),
    ).model_dump()


# ---------------------------------------------------------------------------
# Control-plane — collection lifecycle
# ---------------------------------------------------------------------------


@router.post(
    "/collections",
    status_code=201,
    summary="Create a new Qdrant collection",
)
async def create_collection(
    body: CollectionCreateRequest,
    _user=Depends(get_current_user),
) -> dict[str, Any]:
    result = CollectionService.create(
        name=body.name,
        vector_size=body.vector_size,
        distance=body.distance,
    )
    if result.get("conflict"):
        raise HTTPException(
            status_code=409,
            detail=f"Collection '{body.name}' already exists",
        )
    return {"status": "created", "name": body.name}


@router.delete(
    "/collections/{name}",
    summary="Permanently delete a Qdrant collection and its RecordManager data",
)
async def delete_collection(
    name: str,
    _user=Depends(get_current_user),
) -> dict[str, Any]:
    deleted = await CollectionService.delete(name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{name}' not found",
        )
    return {"status": "deleted", "name": name}


@router.get(
    "/collections",
    response_model=CollectionListResponse,
    summary="List all collections in the remote Qdrant instance",
)
async def list_collections() -> CollectionListResponse:
    items = CollectionService.list_all()
    return CollectionListResponse(
        collections=[CollectionInfo(**i) for i in items]
    )
