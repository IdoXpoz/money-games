import torch
import re
from typing import List, Tuple, Optional

from src.prompts.configs.games import DECISION_KEYWORDS


def _compute_next_token_probs(prompt: str, tokenizer, model):
    """
    Compute the probability distribution over the next token for a given prompt.

    Returns a torch.Tensor of probabilities over the vocabulary.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_logits = outputs.logits[0, -1]
        probs = torch.softmax(next_logits, dim=-1)
    return probs


def get_decision_token_probs(
    prompt: str, tokenizer, model, keywords: List[str] = DECISION_KEYWORDS
) -> List[Tuple[str, float]]:
    """
    Get probabilities for specific decision keywords for the next token.

    Args:
        prompt (str): The input prompt
        tokenizer: The model tokenizer
        model: The model
        keywords: List of decision keywords

    Returns:
        List[Tuple[str, float]]: Decision keywords and their probabilities
    """

    probs = _compute_next_token_probs(prompt, tokenizer, model)

    # Extract just the keyword probabilities
    decision_probs = []
    for kw in keywords:
        kw_id = tokenizer.encode(kw, add_special_tokens=False)[0]
        decision_probs.append((kw, probs[kw_id].item()))
    return decision_probs


def get_top_token_probs(prompt: str, tokenizer, model, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Get the top-k most probable next tokens and their probabilities.

    Args:
        prompt (str): The input prompt
        tokenizer: The model tokenizer
        model: The model
        top_k (int): Number of top tokens to return (default: 5)

    Returns:
        List[Tuple[str, float]]: Top tokens and their probabilities, sorted by probability (highest first)
    """

    probs = _compute_next_token_probs(prompt, tokenizer, model)

    # Get top-k tokens and their probabilities
    top_probs, top_indices = torch.topk(probs, top_k)

    # Convert token IDs back to text and pair with probabilities
    top_tokens = []
    for i in range(top_k):
        token_id = top_indices[i].item()
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        probability = top_probs[i].item()
        top_tokens.append((token_text, probability))

    return top_tokens


def get_decision_token_probs_reasoning(
    prompt: str, tokenizer, model, keywords: List[str] = DECISION_KEYWORDS, max_new_tokens: int = 500
) -> List[Tuple[str, float]]:
    """
    Get probabilities for decision keywords in reasoning models using two-step approach:
    1. Generate full reasoning response
    2. Add thinking part to prompt and analyze next token probabilities

    Args:
        prompt (str): The input prompt
        tokenizer: The model tokenizer
        model: The model
        keywords: List of decision keywords
        max_new_tokens: Maximum tokens to generate for reasoning

    Returns:
        List[Tuple[str, float]]: Decision keywords and their probabilities
    """
    # Step 1: Generate the full reasoning response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode the full response
        generated_ids = outputs[0, len(inputs.input_ids[0]) :]
        full_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Step 2: Extract thinking part and create new prompt
    thinking_part = _extract_thinking_part(full_response)

    # Create new prompt with thinking part included
    new_prompt = prompt + thinking_part + "\n"

    # Step 3: Analyze probabilities of next token (same as regular models)
    return get_decision_token_probs(new_prompt, tokenizer, model, keywords)


def get_top_token_probs_reasoning(
    prompt: str, tokenizer, model, keywords: List[str] = DECISION_KEYWORDS, max_new_tokens: int = 500, top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Get top-k token probabilities for reasoning models using two-step approach.

    Args:
        prompt (str): The input prompt
        tokenizer: The model tokenizer
        model: The model
        keywords: List of decision keywords (not used in this approach)
        max_new_tokens: Maximum tokens to generate
        top_k: Number of top tokens to return

    Returns:
        List[Tuple[str, float]]: Top tokens and probabilities at decision point
    """
    # Step 1: Generate the full reasoning response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode the full response
        generated_ids = outputs[0, len(inputs.input_ids[0]) :]
        full_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Step 2: Extract thinking part and create new prompt
    thinking_part = _extract_thinking_part(full_response)

    # Create new prompt with thinking part included
    new_prompt = prompt + thinking_part + "\n"

    # Step 3: Analyze top token probabilities (same as regular models)
    return get_top_token_probs(new_prompt, tokenizer, model, top_k)


def _extract_thinking_part(response: str) -> str:
    """
    Extract the thinking part from a reasoning model's response.
    Includes both the thinking tags and content: <think>...</think>

    Args:
        response (str): Full response from reasoning model

    Returns:
        str: Thinking part including tags, or empty string if not found
    """
    # Look for <think>...</think> pattern
    think_pattern = r"<think>.*?</think>"
    match = re.search(think_pattern, response, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(0)

    return ""
