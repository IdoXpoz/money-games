import torch

from src.models.config import GENERATION_PARAMS
from src.models.model_manager import OpenSourceModelManager


class OpenSourceModel:
    """Handler for open source models."""

    def __init__(self, model_name: str, model_manager: OpenSourceModelManager):
        self.model_name = model_name
        self.model_manager = model_manager
        self.tokenizer, self.model, self.pipeline = model_manager.load_model(model_name)

    def run(self, prompt: str) -> str:
        """
        Run prompt through the open source model.

        Args:
            prompt (str): The input prompt

        Returns:
            str: The model's response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            # Generate exactly a few tokens, forbidding EOS
            gen = self.model.generate(
                inputs.input_ids,
                max_new_tokens=GENERATION_PARAMS["max_new_tokens"],
                do_sample=GENERATION_PARAMS["do_sample"],
                eos_token_id=None,  # don't stop on EOS
                pad_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=[[self.tokenizer.eos_token_id]],  # forbid EOS
            )
        generated_ids = gen[0, -GENERATION_PARAMS["max_new_tokens"] :].tolist()  # last n token IDs
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
