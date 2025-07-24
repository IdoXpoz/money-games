from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


class OpenSourceModelManager:
    """Manages loading and caching of open source models."""

    def __init__(self, huggingface_token: str):
        self._model_cache = {}
        self._authenticated = False
        self.huggingface_token = huggingface_token

    def authenticate(self):
        """Authenticate with HuggingFace."""
        if not self._authenticated:
            login(token=self.huggingface_token)
            self._authenticated = True

    def load_model(self, model_name: str):
        """
        Load model once and cache by model name.

        Args:
            model_name (str): The model name/path

        Returns:
            tuple: (tokenizer, model, pipeline)
        """
        if model_name in self._model_cache:
            print(f"Using cached model for '{model_name}'.")
            cache_entry = self._model_cache[model_name]
            return cache_entry["tokenizer"], cache_entry["model"], cache_entry["pipeline"]

        print(f"Loading model '{model_name}' from disk/network...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # Save to cache
        self._model_cache[model_name] = {"tokenizer": tokenizer, "model": model, "pipeline": model_pipeline}

        return tokenizer, model, model_pipeline
