from typing import List

from pydantic import BaseModel


class Article(BaseModel):
    title: str
    body: str
    date: str
    site: str


class SearchResult(BaseModel):
    doc_id: int
    title: str
    snippet: str
    site: str
    date: str


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResult]
