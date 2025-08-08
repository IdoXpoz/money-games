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


def get_decision_token_probs(probs: torch.Tensor, keywords: List[str] = DECISION_KEYWORDS) -> List[Tuple[str, float]]:
    """
    Get probabilities for specific decision keywords for the next token.
    """
    decision_probs = []
    for kw in keywords:
        kw_id = tokenizer.encode(kw, add_special_tokens=False)[0]
        decision_probs.append((kw, probs[kw_id].item()))
    return decision_probs


def get_top_token_probs(probs: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Get the top-k most probable tokens and their probabilities.
    """
    top_probs, top_indices = torch.topk(probs, top_k)
    top_tokens = []
    for i in range(top_k):
        token_id = top_indices[i].item()
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        probability = top_probs[i].item()
        top_tokens.append((token_text, probability))
    return top_tokens


def run_probs_analysis(
    prompt: str, tokenizer, model, keywords: List[str] = DECISION_KEYWORDS, top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    1. Get the probability distribution of the keywords we care about
    2. Get the top-k most probable tokens
    """
    probs = _compute_next_token_probs(prompt, tokenizer, model)
    decision_probs = get_decision_token_probs(probs, keywords)
    top_tokens = get_top_token_probs(probs, top_k)

    return decision_probs, top_tokens


def run_inference_and_add_thinking_part_to_prompt(prompt: str, tokenizer, model, max_new_tokens: int = 500) -> str:
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

    return new_prompt


def run_probs_analysis_reasoning(
    prompt: str, tokenizer, model, keywords: List[str] = DECISION_KEYWORDS, max_new_tokens: int = 500
) -> List[Tuple[str, float]]:
    """
    1. Run inference and add thinking part to prompt
    2. Run probability analysis on the first token after the thinking part
    """
    new_prompt = run_inference_and_add_thinking_part_to_prompt(prompt, tokenizer, model, max_new_tokens)
    return run_probs_analysis(new_prompt, tokenizer, model, keywords)


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
        print(f"Found thinking part: {match.group(0)}")
        return match.group(0)

    print("did not find thinking part")
    return ""
