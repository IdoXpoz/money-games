import torch
import re
from typing import List, Tuple, Optional

from src.prompts.configs.games import DECISION_KEYWORDS
from src.models.config import REASONING_GENERATION_PARAMS


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
    tokenizer, probs: torch.Tensor, keywords: List[str] = DECISION_KEYWORDS
) -> List[Tuple[str, float]]:
    """
    Get probabilities for specific decision keywords for the next token.
    """
    decision_probs = []
    for kw in keywords:
        kw_id = tokenizer.encode(kw, add_special_tokens=False)[0]
        decision_probs.append((kw, probs[kw_id].item()))
    return decision_probs


def get_top_token_probs(tokenizer, probs: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
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
    decision_probs = get_decision_token_probs(tokenizer, probs, keywords)
    top_tokens = get_top_token_probs(tokenizer, probs, top_k)

    return decision_probs, top_tokens


def get_next_token_distribution_after_thinking_tag(prompt: str, tokenizer, model) -> torch.Tensor:
    """
    Run the reasoning model to produce thinking content, locate the </think> end tag,
    and compute the probability distribution of the FIRST token AFTER that tag.

    Returns a torch.Tensor of probabilities over the vocabulary.
    """
    # Build the chat-formatted input with thinking enabled
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate to get thinking and the closing tag; request per-step scores to avoid a second forward pass
    with torch.no_grad():
        generate_output = model.generate(
            **model_inputs,
            max_new_tokens=REASONING_GENERATION_PARAMS["max_new_tokens"],
            do_sample=REASONING_GENERATION_PARAMS["do_sample"],
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = generate_output.sequences  # the chosen outputs per position
    scores = generate_output.scores  # the distribution per position

    # Only the newly generated ids
    output_ids = sequences[0][len(model_inputs.input_ids[0]) :].tolist()

    end_pos = model.find_end_of_thinking_tag(output_ids)

    next_logits = scores[end_pos][0]  # [vocab]
    probs = torch.softmax(next_logits, dim=-1)
    return probs


def run_probs_analysis_reasoning(
    prompt: str, tokenizer, model, keywords: List[str] = DECISION_KEYWORDS, top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Compute decision keyword probabilities and top-k tokens for the FIRST token AFTER </think>.
    """
    probs = get_next_token_distribution_after_thinking_tag(prompt, tokenizer, model)
    decision_probs = get_decision_token_probs(tokenizer, probs, keywords)
    top_tokens = get_top_token_probs(tokenizer, probs, top_k)
    return decision_probs, top_tokens
