import os
from typing import Dict, List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader import load_and_clean
from src.evaluation import evaluate_system
from src.indexer import build_indices, load_indices
from src.models import SearchResponse, SearchResult
from src.rag.rag_pipeline import RAGPipeline
from src.search import search_bm25
from src.semantic.semantic_search import semantic_search
from src.semantic.similarity import similar_articles
from src.semantic.timeline import build_story_timeline
from src.seo import analyze_seo

app = FastAPI(title="News Search Engine")
rag = RAGPipeline()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    return {"query": q, "results": semantic_search(q, top_k)}


@app.get("/similar")
async def similar_route(doc_id: int, top_k: int = 5):
    return {"doc_id": doc_id, "results": similar_articles(doc_id, top_k)}


@app.get("/timeline")
async def timeline_route(doc_id: int, top_k: int = 8):
    return build_story_timeline(doc_id, top_k)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/rag/search")
def rag_search(query: str, top_k: int = 5):
    """
    RAG-powered search:
    Semantic retrieval + Gemini summarization
    """
    return rag.run(query, top_k)


class EvaluationRequest(BaseModel):
    ground_truth: Dict[str, List[int]]
    top_k: int = 10


@app.post("/evaluate")
async def evaluate_route(req: EvaluationRequest):
    """
    Run evaluation metrics on the provided ground truth.
    """
    results = {}
    # Convert list to set for ground truth
    # print(req.ground_truth)
    gt_sets = {k: set(v) for k, v in req.ground_truth.items()}

    # print(gt_sets)
    for query in req.ground_truth.keys():
        # print(query)
        # print(gt_sets[query])
        # Run search for each query
        # We use BM25 for now as the baseline
        search_res = search_bm25(query, top_k=req.top_k)
        # print(search_res)
        results[query] = [r["doc_id"] for r in search_res]
        print(results)
    metrics = evaluate_system(results, gt_sets, k=req.top_k)
    return metrics


@app.get("/seo/analyze")
async def seo_route(doc_id: int):
    """
    Analyze SEO factors for a specific document.
    """
    _, _, doc_map = load_indices()
    if doc_id not in doc_map:
        return {"error": "Document not found"}

    doc = doc_map[doc_id]
    # doc structure: (doc_id, title, body, site, date)

    title = doc[1]
    body = doc[2]

    return analyze_seo(title, body)
