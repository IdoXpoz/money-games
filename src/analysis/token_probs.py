import torch
from typing import List, Tuple

from src.prompts.configs.games import DECISION_KEYWORDS


def extract_decision_token_probs(
    token_probs: List[Tuple[str, float]], keywords: List[str] = DECISION_KEYWORDS
) -> List[Tuple[str, float]]:
    """
    Extract decision-related tokens and their probabilities.

    Args:
        token_probs: List of (token, probability) pairs
        keywords: List of decision keywords to look for

    Returns:
        List[Tuple[str, float]]: Decision tokens and their probabilities
    """

    decision_probs = []
    for token, prob in token_probs:
        if any(kw in token.lower() for kw in keywords):
            decision_probs.append((token, prob))
    return decision_probs


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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_logits = outputs.logits[0, -1]
        probs = torch.softmax(next_logits, dim=-1)

    # Extract just the keyword probabilities
    decision_probs = []
    for kw in keywords:
        kw_id = tokenizer.encode(kw, add_special_tokens=False)[0]
        decision_probs.append((kw, probs[kw_id].item()))
    return decision_probs
