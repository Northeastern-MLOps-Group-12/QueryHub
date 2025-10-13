import os
from google import genai

class GeminiClient:
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        """
        api_key: your Gemini / Google GenAI key
        model: name of the Gemini model you want to use (e.g. "gemini-2.5-flash") 
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Sends a single prompt to the Gemini model and returns the response text.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        # .text contains the model output
        return response.text
