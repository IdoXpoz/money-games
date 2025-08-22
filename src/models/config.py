OPEN_SOURCE_MODELS = ["google/gemma-3-4b-it"]
# OPEN_SOURCE_MODELS = []
REASONING_MODELS = ["Qwen/Qwen3-4B"]
# REASONING_MODELS = []

# Use deterministic generation for consistent token probability analysis
NORMAL_GENERATION_PARAMS = {"max_new_tokens": 5, "do_sample": False}

# Reasoning model generation parameters
REASONING_GENERATION_PARAMS = {
    "max_new_tokens": 10000,  # Allow thinking space
    "do_sample": True,        # Enable sampling (non-greedy)
    "temperature": 0.7,       # Control randomness (0.1-2.0, higher = more random)
}



def is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model."""
    return model_name in REASONING_MODELS
