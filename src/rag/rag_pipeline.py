# src/rag/rag_pipeline.py

from src.semantic.semantic_search import semantic_search
from src.rag.gemini_client import GeminiClient


class RAGPipeline:

    def __init__(self):
        self.gemini = GeminiClient()

    def run(self, query: str, top_k: int = 5):
        """
        Full RAG pipeline:
        1. Retrieve top-k similar articles
        2. Build prompt
        3. Get LLM summary
        4. Return answer + sources
        """

        # 1. Retrieve documents
        hits = semantic_search(query, top_k=top_k)

        # Extract text from each result
        docs = [hit["text"] for hit in hits]

        # 2. Build the LLM prompt
        prompt = self._build_prompt(query, docs)

        # 3. Ask Gemini for summary
        answer = self.gemini.summarize(prompt)

        return {
            "answer": answer,
            "sources": hits
        }

    def _build_prompt(self, query: str, docs: list[str]):
        """
        Create a high-quality RAG prompt for Gemini.
        """

        joined_docs = "\n\n".join([f"Article {i+1}:\n{d}" for i, d in enumerate(docs)])

        return f"""
You are an AI assistant that answers questions using the retrieved news articles.

User question:
{query}

Relevant news articles:
{joined_docs}

Write a concise, factual answer (100–200 words). 
Include only information supported by the articles. 
Do NOT invent facts.
"""