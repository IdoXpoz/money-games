# OPEN_SOURCE_MODELS = ["google/gemma-3-4b-it"]
OPEN_SOURCE_MODELS = []
REASONING_MODELS = ["Qwen/Qwen3-4B"]

# Use deterministic generation for consistent token probability analysis
NORMAL_GENERATION_PARAMS = {"max_new_tokens": 5, "do_sample": False}

# Reasoning model generation parameters
REASONING_GENERATION_PARAMS = {
    "max_new_tokens": 500,  # Allow thinking space
    "do_sample": False,
    "temperature": 0.1,
}


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model."""
    return model_name in REASONING_MODELS
