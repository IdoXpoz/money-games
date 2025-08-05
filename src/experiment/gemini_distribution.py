from typing import Dict
from src.models.gemini import GeminiModel
from src.prompts.configs import games, money
from src.prompts.prompt_builder import construct_prompt
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import time


def get_distribution_dict_per_prompt(model: GeminiModel, prompt: str, num_of_iterations: int = 20) -> Dict[str, int]:
    """
    Get Gemini output token distribution to ensure deterministic results.

    Args:
        model: model to prompt
        prompt: prompt to run
        num_of_iterations: number of iterations to run

    Returns:
        Dict[str, int]: token -> number of time it appeared as output
    """

    print("Initializing distribution dictionary")
    distribution_dict = {keyword: 0 for keyword in games.DECISION_KEYWORDS}
    distribution_dict["other"] = 0

    print(f"Running {num_of_iterations} iterations")
    for i in range(num_of_iterations):
        print(f"Iteration {i + 1}/{num_of_iterations}")
        try:
            response = model.run(prompt)
        except Exception:
            print("Reached quota limit. Sleeping for 1 minute")
            time.sleep(62)
            print("Trying again")
            response = model.run(prompt)
        response = response.lower()
        if response not in distribution_dict:
            response = "other"
        distribution_dict[response] += 1

    return distribution_dict


def get_all_distribution_dicts(model: GeminiModel) -> Dict[str, Dict[str, int]]:
    """
    Get Gemini output token distribution to ensure deterministic results for all prompts.

    Args:
        model: model to prompt

    Returns:
        Dict[str, Dict[str, int]]: dict of prefix_paraphrase -> dict of token -> number of time it appeared as output
    """
    distribution_dicts = {}
    for prefix in money.PREFIXES:
        for paraphrase_idx, task_instruction in enumerate(games.TASK_INSTRUCTIONS):
            key = f"{prefix}_paraphrase_{paraphrase_idx}"
            print("\n#####################################")
            print(f"Testing prefix '{prefix}' with paraphrase {paraphrase_idx + 1}/{len(games.TASK_INSTRUCTIONS)}")
            print(f"Constructing prompt")
            prompt = construct_prompt(money.PREFIXES[prefix], task_instruction)
            print(f"Getting output token distributions")
            distribution_dict = get_distribution_dict_per_prompt(model, prompt)
            distribution_dicts[key] = distribution_dict
    return distribution_dicts


def plot_all_distribution_dicts(
    distribution_dicts: Dict[str, Dict[str, int]], output_dir: str = "./src/experiment/plots"
) -> None:
    """
    Plot all Gemini output token distribution.

    Args:
        distibution_dicts: all of the prefix and paraphrase output token distribution dicts
        output_dir: directory to save the plots
    """
    print(f"Verifying existance of plot output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for prefix_paraphrase_key in distribution_dicts:
        print(f"Plotting {prefix_paraphrase_key}")
        distribution_dict = distribution_dicts[prefix_paraphrase_key]
        fig, ax = plt.subplots()
        keys = list(distribution_dict.keys())
        values = list(distribution_dict.values())
        ax.bar(keys, values)
        ax.set_xlabel("Output Tokens")
        ax.set_ylabel("Number of Appearances")

        # Parse the key to create a more readable title
        if "_paraphrase_" in prefix_paraphrase_key:
            prefix_part, paraphrase_part = prefix_paraphrase_key.split("_paraphrase_")
            ax.set_title(f"Prefix: {prefix_part} | Paraphrase: {paraphrase_part}")
        else:
            ax.set_title(f"Condition: {prefix_paraphrase_key}")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        print(f"Saving plot")
        filename = os.path.join(output_dir, f"plot_{prefix_paraphrase_key}.png")
        fig.savefig(filename)
        plt.close(fig)

    print(f"All plots saved and can be found in: {output_dir}")


def save_distribution_results(
    distribution_dicts: Dict[str, Dict[str, int]], output_file: str = "./gemini_distribution_results.csv"
) -> None:
    """
    Save distribution results to a CSV file.

    Args:
        distribution_dicts: all of the prefix and paraphrase output token distribution dicts
        output_file: path to save the CSV file
    """
    import pandas as pd

    print(f"Saving distribution results to {output_file}")

    # Convert nested dict to flat list of records
    records = []
    for prefix_paraphrase_key, distribution_dict in distribution_dicts.items():
        if "_paraphrase_" in prefix_paraphrase_key:
            prefix_part, paraphrase_part = prefix_paraphrase_key.split("_paraphrase_")
        else:
            prefix_part = prefix_paraphrase_key
            paraphrase_part = "0"

        for token, count in distribution_dict.items():
            records.append(
                {
                    "prefix": prefix_part,
                    "paraphrase_index": paraphrase_part,
                    "token": token,
                    "count": count,
                    "full_key": prefix_paraphrase_key,
                }
            )

    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Distribution results saved to {output_file}")


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Please set API_KEY")
    print("API key found. Running tests...")
    distribution_dicts = get_all_distribution_dicts(GeminiModel(api_key))
    plot_all_distribution_dicts(distribution_dicts)
    save_distribution_results(distribution_dicts)
