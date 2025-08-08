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
        # Prepare the model input using Qwen chat template with thinking enabled
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate tokens
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=REASONING_GENERATION_PARAMS["max_new_tokens"],
                do_sample=REASONING_GENERATION_PARAMS["do_sample"],
            )

        # Extract only the newly generated token IDs
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        index = self.find_end_of_thinking_tag(output_ids)

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        print(f"thinking_content: {thinking_content}")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        print(f"content: {content}")

        return content, thinking_content

    def find_end_of_thinking_tag(self, sequence: list) -> int:
        """Return index of last occurrence of value in sequence or None if not found."""
        try:
            thinking_tag_id = 151668
            print(f"searching for </think> in {sequence}")
            return len(sequence) - 1 - sequence[::-1].index(thinking_tag_id)
        except ValueError:
            return 0
