"""FastAPI app factory.

Çalıştır: uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import router
from src.vector_store import get_store

from src.tracing import flush


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading FAISS index...")
    try:
        store = get_store()
        print(f"FAISS index ready: {store.index.ntotal} vectors")
    except FileNotFoundError:
        print("FAISS index not found! Run: python -m src.vector_store")
    
    print("Bedrock Nova Micro ready")

    yield

    # Shutdown
    flush()
    print("Traces flushed.")


app = FastAPI(
    title="RAG Observability Lab",
    description="QASPER RAG pipeline with Llama 3 + Langfuse",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)