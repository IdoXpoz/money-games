import pandas as pd
from datetime import datetime

from src.prompts.configs.money import PREFIXES
from src.prompts.configs.games import TASK_INSTRUCTIONS
from src.models.config import OPEN_SOURCE_MODELS, REASONING_MODELS, is_reasoning_model
from src.models.open_source_model import OpenSourceModel
from src.models.reasoning_model import ReasoningModel
from src.models.model_manager import OpenSourceModelManager
from src.models.gemini import GeminiModel
from src.prompts.prompt_builder import construct_prompt
from src.analysis.token_probs import (
    get_decision_token_probs,
    get_top_token_probs,
    get_decision_token_probs_reasoning,
    get_top_token_probs_reasoning,
)


class ExperimentRunner:
    """Main class for running the LLM monetary priming experiment."""

    def __init__(self, huggingface_token: str, gemini_api_key: str):
        self.model_manager = OpenSourceModelManager(huggingface_token)
        self.results = []
        self.gemini_api_key = gemini_api_key

    def setup(self):
        """Setup authentication and any other required initialization."""
        self.model_manager.authenticate()

    def run_open_source_experiment(self, include_gemini: bool = False) -> pd.DataFrame:
        """
        Run the full experiment on open source models.

        Args:
            include_gemini: Whether to include Gemini model in the experiment

        Returns:
            pd.DataFrame: Results dataframe
        """
        print("Starting experiment...")

        # Iterate through all task instruction paraphrases first
        for paraphrase_idx, task_instruction in enumerate(TASK_INSTRUCTIONS):
            print(f"Testing paraphrase {paraphrase_idx + 1}/{len(TASK_INSTRUCTIONS)}...")

            # Iterate through all prefix conditions
            for prefix_name, prefix_text in PREFIXES.items():
                full_prompt = construct_prompt(prefix_text, task_instruction)

                print(f"  - Testing prefix '{prefix_name}'...")

                # Test regular open source models
                all_models = OPEN_SOURCE_MODELS + REASONING_MODELS

                for model_name in all_models:
                    print(f"    - Testing {model_name}...")

                    # Choose appropriate model class
                    if is_reasoning_model(model_name):
                        model = ReasoningModel(model_name, self.model_manager)
                        is_reasoning = True
                    else:
                        model = OpenSourceModel(model_name, self.model_manager)
                        is_reasoning = False

                    # Get model response
                    response = model.run(full_prompt)

                    # Get token-level probabilities
                    if is_reasoning:
                        decision_probs = get_decision_token_probs_reasoning(full_prompt, model.tokenizer, model.model)
                        top_tokens = get_top_token_probs_reasoning(full_prompt, model.tokenizer, model.model, top_k=10)
                    else:
                        decision_probs = get_decision_token_probs(full_prompt, model.tokenizer, model.model)
                        top_tokens = get_top_token_probs(full_prompt, model.tokenizer, model.model, top_k=10)

                    # Store results
                    self.results.append(
                        {
                            "model": model_name,
                            "prefix_type": prefix_name,
                            "paraphrase_index": paraphrase_idx,
                            "prompt": full_prompt,
                            "response": response,
                            "decision_tokens": decision_probs,
                            "top_tokens": top_tokens,
                            "is_reasoning_model": is_reasoning,
                            "timestamp": datetime.now(),
                        }
                    )

                # Test Gemini model if requested
                if include_gemini:
                    print(f"    - Testing Gemini...")
                    try:
                        gemini_model = GeminiModel(self.gemini_api_key)
                        gemini_response = gemini_model.run(full_prompt)

                        self.results.append(
                            {
                                "model": "Gemini",
                                "prefix_type": prefix_name,
                                "paraphrase_index": paraphrase_idx,
                                "prompt": full_prompt,
                                "response": gemini_response,
                                "decision_tokens": None,  # Token probs not available for Gemini
                                "top_tokens": None,  # Token probs not available for Gemini
                                "timestamp": datetime.now(),
                            }
                        )
                    except Exception as e:
                        print(f"Error with Gemini: {e}")

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.

        Returns:
            pd.DataFrame: Results in DataFrame format
        """
        pd.set_option("display.max_colwidth", None)
        df = pd.DataFrame(self.results)
        return df

    def save_results(self, filename: str = "prisoner_dilemma_all_paraphrases_results.csv"):
        """
        Save results to CSV file.

        Args:
            filename: Name of the CSV file to save
        """
        df = self.get_results_dataframe()
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        print(f"Total experiments run: {len(df)} (4 prefixes × {len(TASK_INSTRUCTIONS)} paraphrases × models)")

    def clear_results(self):
        """Clear stored results for a fresh experiment."""
        self.results = []
