"""FastAPI route tanımları."""

from fastapi import APIRouter, HTTPException

from api.schemas import AskRequest, AskResponse, HealthResponse
from src.rag_pipeline import ask
from src.vector_store import get_store
from src.tracing import flush

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest) -> AskResponse:
    try:
        result = ask(
            query=request.query,
            top_k=request.top_k,
            tags=request.tags if request.tags else ["api"],
        )
        return AskResponse(**result)
    except Exception as e:
        flush()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    try:
        store = get_store()
        index_ok = True
        index_size = store.index.ntotal
    except Exception:
        index_ok = False
        index_size = 0

    status = "healthy" if index_ok else "degraded"

    return HealthResponse(
        status=status,
        bedrock=True,
        index_loaded=index_ok,
        index_size=index_size,
    )