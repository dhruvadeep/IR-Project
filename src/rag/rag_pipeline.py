# src/rag/rag_pipeline.py

from src.rag.openai_agent_client import OpenAIAgentClient
from src.semantic.semantic_search import semantic_search


class RAGPipeline:
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize RAG pipeline with OpenAI agent.

        Args:
            model: OpenAI model to use (default: gpt-4o)
                   Options: gpt-4o, gpt-4, gpt-4-turbo, etc.
        """
        self.agent = OpenAIAgentClient(model=model)

    def run(self, query: str, top_k: int = 5):
        """
        Full RAG pipeline:
        1. Retrieve top-k similar articles using semantic search
        2. Pass query and articles to OpenAI agent
        3. Agent reasons about the documents and generates answer
        4. Return answer + sources
        """

        # 1. Retrieve documents using semantic search
        hits = semantic_search(query, top_k=top_k, include_text=True)

        # Extract text from each result
        docs = [hit["text"] for hit in hits]

        # 2. Use OpenAI agent to generate answer
        # The agent will use its tool to access documents and reason about them
        answer = self.agent.summarize(query, docs)

        return {"answer": answer, "sources": hits}
