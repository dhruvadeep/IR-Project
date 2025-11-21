import os

from google.genai import Client


class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv(
            "GEMINI_API_KEY", "AIzaSyCnSNSDA3JoOScCiCh5OWeT4i4s4FmCbeU"
        )
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment variables")

        self.client = Client(api_key=self.api_key)

    def summarize(self, prompt: str, model: str = "gemini-2.5-flash"):
        """
        Send a prompt to Gemini and return the summary text.
        """
        response = self.client.models.generate(model=model, contents=prompt)
        return response.text
