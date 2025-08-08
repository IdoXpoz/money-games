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
        response, thinking_content = self.run_reasoning_inference_and_split_thinking_content(prompt)

        decision_probs, top_tokens = run_probs_analysis_reasoning(prompt, self.tokenizer, self.model)

        return response, thinking_content, decision_probs, top_tokens

    def run_reasoning_inference_and_split_thinking_content(self, prompt: str):
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
        # Try to find the last occurrence of the </think> token id (151668 for Qwen3 tokenizer)
        try:
            print(f"searching for </think> in {generated_ids}")
            index = len(generated_ids) - generated_ids[::-1].index(151668)
        except ValueError:
            print("did not find </think>")
            index = 0

        thinking_content = self.tokenizer.decode(generated_ids[:index], skip_special_tokens=True).strip("\n")
        print(f"thinking_content: {thinking_content}")
        content = self.tokenizer.decode(generated_ids[index:], skip_special_tokens=True).strip("\n")
        print(f"content: {content}")

        return content, thinking_content
