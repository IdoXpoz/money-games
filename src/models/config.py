# OPEN_SOURCE_MODELS = ["google/gemma-3-4b-it", "Qwen/Qwen3-4B", "tiiuae/falcon-7b-instruct"]
OPEN_SOURCE_MODELS = ["google/gemma-3-4b-it"]
# Use deterministic generation for consistent token probability analysis
GENERATION_PARAMS = {"max_new_tokens": 200, "do_sample": False}
