from typing import Dict
from src.models.gemini import GeminiModel
from src.prompts.configs import games, money
from src.prompts.prompt_builder import construct_prompt
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

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
    distribution_dict = {}
    try:
        for i in range(num_of_iterations):
            print(f"Iteration {str(i)/str(num_of_iterations)}")
            response = model.run(prompt)
            if response not in distribution_dict:
                distribution_dict[response] = 0
            distribution_dict[response] += 1
    except Exception as e:
        print(f"Error with Gemini: {e}")
    return distribution_dict

def get_all_distribution_dicts(model: GeminiModel) -> Dict[str, Dict[str, int]]:
    """
    Get Gemini output token distribution to ensure deterministic results for all prompts.

    Args:
        model: model to prompt

    Returns:
        Dict[str, Dict[str, int]]: dict of prefix -> dict of token -> number of time it appeared as output
    """
    distribution_dicts = {}
    for prefix in money.PREFIXES:
        print(f"Testing prefix {prefix}")
        print(f"Constructing prompt")
        prompt = construct_prompt(money.PREFIXES[prefix], games.DECISION_TASK)
        print(f"Getting output token distributions")
        distribution_dict = get_distribution_dict_per_prompt(model, prompt)
        distribution_dicts[prefix] = distribution_dict
    return distribution_dicts

def plot_all_distribution_dicts(distribution_dicts: Dict[str, Dict[str, int]], output_dir: str = "./src/experiment/plots") -> None:
    """
    Plot all Gemini output token distribution.

    Args:
        distibution_dicts: all of the prefix output token distribution dicts
        output_dir: directory to save the plots
    """
    print(f"Verifying existance of plot output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for prefix in distribution_dicts:
        print(f"Plotting prefix: {prefix}")
        distribution_dict = distribution_dicts[prefix]
        fig, ax = plt.subplots()
        keys = list(distribution_dict.keys())
        values = list(distribution_dict.values())
        ax.bar(keys, values)
        ax.set_xlabel('Output Tokens')
        ax.set_ylabel('Number of Appearances')
        ax.set_title(f'Prefix: {prefix}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        print(f"Saving plot")
        filename = os.path.join(output_dir, f'plot_{prefix}.png')
        fig.savefig(filename)
        plt.close(fig)

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Please set API_KEY")
    print("API key found. Running tests...")
    distribution_dicts = get_all_distribution_dicts(GeminiModel(api_key))
    plot_all_distribution_dicts(distribution_dicts)