# ALL_OPEN_SOURCE_MODELS = ["google/gemma-3-4b-it", "google/gemma-3-12b-it", "meta-llama/Llama-3.2-3B-Instruct"]
OPEN_SOURCE_MODELS = ["google/gemma-3-12b-it"]
# ALL_REASONING_MODELS = ["Qwen/Qwen3-4B"]
REASONING_MODELS = []

# Use deterministic generation for consistent token probability analysis
NORMAL_GENERATION_PARAMS = {"max_new_tokens": 5, "do_sample": False}

# Reasoning model generation parameters
REASONING_GENERATION_PARAMS = {
    "max_new_tokens": 10000,  # Allow thinking space
    "do_sample": True,  # Enable sampling (non-greedy, we'll use the default conf of Temperature=0.6, TopP=0.95, TopK=20, and MinP=0 )
}


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model."""
    return model_name in REASONING_MODELS
