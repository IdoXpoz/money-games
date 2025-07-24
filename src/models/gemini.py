import google.generativeai as genai


class GeminiModel:
    """Handler for Google's Gemini model."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = api_key
        self._configure_api()
        self.model = genai.GenerativeModel(model_name)

    def _configure_api(self):
        """Configure Gemini API with authentication."""
        genai.configure(api_key=self.api_key)

    def run(self, prompt: str) -> str:
        """
        Run prompt through Gemini model.

        Args:
            prompt (str): The input prompt

        Returns:
            str: The model's response
        """
        response = self.model.generate_content(prompt)
        return response.text.strip()
