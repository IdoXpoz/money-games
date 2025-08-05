import torch
import re
from typing import List, Tuple, Optional

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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_logits = outputs.logits[0, -1]  # Get logits for the next token
        probs = torch.softmax(next_logits, dim=-1)  # Convert to probabilities

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
    Get probabilities for decision keywords in reasoning models.
    This generates the full reasoning response and analyzes probabilities at the decision point.

    Args:
        prompt (str): The input prompt
        tokenizer: The model tokenizer
        model: The model
        keywords: List of decision keywords
        max_new_tokens: Maximum tokens to generate for reasoning

    Returns:
        List[Tuple[str, float]]: Decision keywords and their probabilities at decision point
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Generate the full reasoning response with scores
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Get the generated sequence and text
        generated_ids = outputs.sequences[0, len(inputs.input_ids[0]) :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Find where the decision is made
        decision_position = _find_decision_position_in_sequence(generated_text, generated_ids, keywords, tokenizer)

        if decision_position is not None and decision_position < len(outputs.scores):
            # Get probabilities at the decision position
            logits = outputs.scores[decision_position][0]  # [0] for batch dimension
            probs = torch.softmax(logits, dim=-1)

            # Extract keyword probabilities
            decision_probs = []
            for kw in keywords:
                kw_tokens = tokenizer.encode(kw, add_special_tokens=False)
                if kw_tokens:  # Take first token of keyword
                    kw_id = kw_tokens[0]
                    decision_probs.append((kw, probs[kw_id].item()))

            return decision_probs

    # Fallback: return zero probabilities
    return [(kw, 0.0) for kw in keywords]


def get_top_token_probs_reasoning(
    prompt: str, tokenizer, model, keywords: List[str] = DECISION_KEYWORDS, max_new_tokens: int = 500, top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Get top-k token probabilities at the decision point for reasoning models.

    Args:
        prompt (str): The input prompt
        tokenizer: The model tokenizer
        model: The model
        keywords: List of decision keywords to locate decision point
        max_new_tokens: Maximum tokens to generate
        top_k: Number of top tokens to return

    Returns:
        List[Tuple[str, float]]: Top tokens and probabilities at decision point
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_ids = outputs.sequences[0, len(inputs.input_ids[0]) :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        decision_position = _find_decision_position_in_sequence(generated_text, generated_ids, keywords, tokenizer)

        if decision_position is not None and decision_position < len(outputs.scores):
            logits = outputs.scores[decision_position][0]
            probs = torch.softmax(logits, dim=-1)

            # Get top-k tokens and their probabilities
            top_probs, top_indices = torch.topk(probs, top_k)

            top_tokens = []
            for i in range(top_k):
                token_id = top_indices[i].item()
                token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                probability = top_probs[i].item()
                top_tokens.append((token_text, probability))

            return top_tokens

    return []


def _find_decision_position_in_sequence(
    generated_text: str, generated_ids: torch.Tensor, keywords: List[str], tokenizer
) -> Optional[int]:
    """
    Find the token position where the decision is made in a reasoning model's output.
    Looks for content after </think> tags (Qwen format).

    Args:
        generated_text: The full generated text
        generated_ids: The generated token IDs
        keywords: Decision keywords to look for
        tokenizer: The tokenizer

    Returns:
        Optional[int]: Token position of decision, or None if not found
    """
    # Look for content after </think>
    thinking_end_match = re.search(r"</think>", generated_text, re.IGNORECASE)
    if thinking_end_match:
        thinking_end_pos = thinking_end_match.end()
        # Find approximate token position
        token_pos = _approximate_char_to_token_position(generated_text, thinking_end_pos, generated_ids, tokenizer)
        if token_pos is not None:
            # Look for decision keywords in the next few tokens
            for offset in range(min(10, len(generated_ids) - token_pos)):
                pos = token_pos + offset
                if pos < len(generated_ids):
                    token_id = generated_ids[pos].item()
                    token_text = tokenizer.decode([token_id], skip_special_tokens=True).lower()
                    if any(kw.lower() in token_text for kw in keywords):
                        return pos

    return None


def _approximate_char_to_token_position(text: str, char_pos: int, token_ids: torch.Tensor, tokenizer) -> Optional[int]:
    """
    Approximate mapping from character position to token position.
    This is not exact due to tokenizer complexities but should be close enough.
    """
    try:
        # Get text up to the character position
        prefix_text = text[:char_pos]

        # Tokenize the prefix
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

        # Return the approximate token position
        return min(len(prefix_tokens), len(token_ids) - 1)
    except:
        return None
