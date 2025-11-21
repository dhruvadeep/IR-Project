import os
from typing import List

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


class OpenAIAgentClient:
    """
    LangChain agent-based RAG client using OpenAI GPT models.
    Uses the create_agent pattern from LangChain documentation.
    """

    def __init__(self, model: str = "gpt-5"):
        self.api_key = os.getenv("OPENAI_API_KEY", "ENTER_YOUR_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in environment variables")

        # Initialize the OpenAI model
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,  # Low temperature for factual responses
            api_key=self.api_key,
        )

        # Store documents context (will be set before each query)
        self.documents_context = []

    def _create_document_tool(self):
        """
        Create a tool that provides access to retrieved documents.
        The agent can call this tool to read the documents.
        """
        documents_context = self.documents_context

        @tool
        def get_retrieved_documents() -> str:
            """
            Get the retrieved news articles that are relevant to the user's query.
            Use this to access the context needed to answer the question.
            """
            if not documents_context:
                return "No documents available."

            result = []
            for i, doc in enumerate(documents_context, 1):
                result.append(f"Article {i}:\n{doc}\n")

            return "\n".join(result)

        return get_retrieved_documents

    def summarize(self, query: str, documents: List[str]) -> str:
        """
        Use LangChain agent to generate a summary based on the query and documents.

        Args:
            query: The user's question
            documents: List of retrieved article texts

        Returns:
            The agent's answer
        """
        # Set the documents context for the tool
        self.documents_context = documents

        # Create the document retrieval tool
        doc_tool = self._create_document_tool()

        # Create the agent with a system prompt
        system_prompt = """You are an AI assistant that answers questions using retrieved news articles.

Your task:
1. Use the get_retrieved_documents tool to access the relevant articles
2. Read through the articles carefully
3. Write a concise, factual answer (100-200 words)
4. Include only information supported by the articles
5. Do NOT invent facts or add information not in the articles

Be helpful, accurate, and cite specific details from the articles when possible."""

        agent = create_agent(
            model=self.llm, tools=[doc_tool], system_prompt=system_prompt
        )

        # Invoke the agent with the user's query
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})

        # Extract the final answer from the agent's response
        # The result contains all messages, the last one is the agent's final answer
        final_message = result["messages"][-1]

        return final_message.content
