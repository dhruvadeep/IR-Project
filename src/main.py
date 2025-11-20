import os

from fastapi import FastAPI, Query

from src.data_loader import load_and_clean
from src.indexer import build_indices
from src.models import SearchResponse, SearchResult
from src.search import search_bm25

from src.semantic.semantic_search import semantic_search
from src.semantic.similarity import similar_articles
from src.semantic.timeline import build_story_timeline


app = FastAPI(title="News Search Engine")


# Initialize on startup
@app.on_event("startup")
async def startup():
    """Load/build indices on startup"""
    if not os.path.exists("indices/bm25.pkl"):
        print("Building indices for first time...")
        load_and_clean(limit=10000)
        build_indices()
    else:
        print("Indices found, loading...")


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=200),
    top_k: int = Query(10, ge=1, le=100),
):
    """Search articles by keyword"""
    results = search_bm25(q, top_k=top_k)
    return SearchResponse(
        query=q,
        total_results=len(results),
        results=[SearchResult(**r) for r in results],
    )

@app.get("/semantic_search")
async def semantic_route(q: str, top_k: int = 10):
    return {
        "query": q,
        "results": semantic_search(q, top_k)
    }


@app.get("/similar")
async def similar_route(doc_id: int, top_k: int = 5):
    return {
        "doc_id": doc_id,
        "results": similar_articles(doc_id, top_k)
    }


@app.get("/timeline")
async def timeline_route(doc_id: int, top_k: int = 8):
    return build_story_timeline(doc_id, top_k)


@app.get("/health")
async def health():
    return {"status": "ok"}
