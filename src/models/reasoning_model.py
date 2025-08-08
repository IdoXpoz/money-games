import re
import torch
from src.models.config import REASONING_GENERATION_PARAMS
from src.models.model_manager import OpenSourceModelManager
from src.analysis.token_probs import run_probs_analysis_reasoning


class ReasoningModel:
    """Handler for reasoning models that use thinking tags."""

    def __init__(self, model_name: str, model_manager: OpenSourceModelManager):
        self.model_name = model_name
        self.model_manager = model_manager
        self.tokenizer, self.model, self.pipeline = model_manager.load_model(model_name)

    def run(self, prompt: str) -> str:
        """
        Run prompt through the reasoning model and extract the final answer.

        Args:
            prompt (str): The input prompt

        Returns:
            str: The extracted final answer (without thinking content)
        """
        full_response = self.get_full_response(prompt)
        final_answer = self._extract_final_answer(full_response)

        decision_probs, top_tokens = run_probs_analysis_reasoning(prompt, self.tokenizer, self.model)

        return final_answer, full_response, decision_probs, top_tokens

    def get_full_response(self, prompt: str) -> str:
        """Get the full response including thinking for debugging."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            gen = self.model.generate(
                inputs.input_ids,
                max_new_tokens=REASONING_GENERATION_PARAMS["max_new_tokens"],
                do_sample=REASONING_GENERATION_PARAMS["do_sample"],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = gen[0, len(inputs.input_ids[0]) :].tolist()
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _extract_final_answer(self, response: str) -> str:
        """
        Extract the final answer from a reasoning model response.
        Handles Qwen-style thinking tags: <think>...</think>

        Args:
            response (str): Full response including thinking content

        Returns:
            str: Extracted final answer
        """
        # Pattern 1: Look for content after </think>
        thinking_match = re.search(r"</think>\s*(.+?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
        if thinking_match:
            answer = thinking_match.group(1).strip()
            if answer and len(answer) < 50:  # Reasonable answer length
                return answer

        # Pattern 2: Look for decision keywords at the end of response
        decision_keywords = ["betray", "silent"]
        lines = response.split("\n")

        # Check last few lines for decision keywords
        for line in reversed(lines[-5:]):
            line = line.strip().lower()
            for keyword in decision_keywords:
                if keyword in line:
                    # Try to extract just the keyword
                    keyword_match = re.search(rf"\b({keyword})\b", line)
                    if keyword_match:
                        return keyword_match.group(1)

        # Fallback: return the last short line
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) < 20:
                return line.lower()

        return "unknown"
