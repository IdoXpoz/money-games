import pandas as pd
import matplotlib.pyplot as plt
import ast
import sys
import os

# Add the project root to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.prompts.configs.money import PREFIXES
from src.prompts.configs.games import DECISION_KEYWORDS

MODEL_TO_CSV_PATH_MAP_GAME1 = {
    # "gemma-3-4b-it": "src/analysis/game1/gemma-3-4b-it_results.csv",
    # "gemma-3-12b-it": "src/analysis/game1/gemma-3-12b-it_results.csv",
    # "chat-gemma-3-4b-it": "src/analysis/game1/chat-gemma-3-4b-it_results.csv",
    # "chat-gemma-3-12b-it": "src/analysis/game1/chat-gemma-3-12b-it_results.csv",
    # "Qwen-without-temperature": "src/analysis/game1/qwen_results.csv",
    # "Qwen": "src/analysis/game1/qwen_results_with_temperature.csv",
    "chat-llama-3.2-3B-Instruct": "src/analysis/game1/chat-llama-3.2-3b-instruct_results.csv",
}

MODEL_TO_CSV_PATH_MAP_GAME2 = {
    # "chat-gemma-3-1b-pt": "src/analysis/game2/chat-gemma-3-1b-pt_results.csv",
    # "chat-gemma-3-1b-it": "src/analysis/game2/chat-gemma-3-1b-it_results.csv",
    # "chat-gemma-3-4b-pt": "src/analysis/game2/chat-gemma-3-4b-pt_results.csv",
    # "chat-gemma-3-4b-it": "src/analysis/game2/chat-gemma-3-4b-it_results.csv",
    # "chat-gemma-3-12b-pt": "src/analysis/game2/chat-gemma-3-12b-pt_results.csv",
    # "chat-gemma-3-12b-it": "src/analysis/game2/chat-gemma-3-12b-it_results.csv",
    "chat-llama-3.2-3B-Instruct": "src/analysis/game2/chat-llama-3.2-3b-instruct_results.csv",
}

PXS = PREFIXES.keys()
OUTPUT_DIR_GAME1 = "src/analysis/game1/trends_analysis_results"
OUTPUT_DIR_GAME2 = "src/analysis/game2/trends_analysis_results"


def get_model_map_and_output_dir(game: str):
    """
    Get the appropriate model map and output directory based on the game.
    """
    if game == "game1":
        return MODEL_TO_CSV_PATH_MAP_GAME1, OUTPUT_DIR_GAME1
    elif game == "game2":
        return MODEL_TO_CSV_PATH_MAP_GAME2, OUTPUT_DIR_GAME2
    else:
        raise ValueError(f"Unknown game: {game}. Must be 'game1' or 'game2'")


def convert_decision_tokens_to_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    convert decision_tokens to list of dicts (they are
    saved as strings)
    """
    df["decision_tokens"] = df["decision_tokens"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    df["decision_tokens"] = df["decision_tokens"].apply(
        lambda lst: next(
            (
                v
                for item in (lst or [])
                if isinstance(item, (list, tuple)) and len(item) == 2
                for k, v in [item]
                if k == DECISION_KEYWORDS[0]  # "betray" or "wait"
            ),
            None,
        )
    )
    df.dropna(subset=["decision_tokens"], inplace=True)
    return df


def drop_non_conclusive_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where 'response' is not one word (i.e., non-conclusive responses).
    In the 'prisoner's dilemma' game, for example, we expect responses to be either 'betray' or 'silent'.
    In reasoning models, sometimes the response is repetitive, therefore we drop those rows.
    """
    df = df[df["response"].apply(lambda x: isinstance(x, str) and is_conclusive_response(x))]
    return df


def is_conclusive_response(response: str) -> bool:
    """
    Check if the response is conclusive (i.e., either 'betray' or 'silent').
    """
    return (
        len(response) > 0
        and len(response.split()) == 1
        and len(response.split(",")) == 1
        and len(response.split(".")) == 1
    )


def analyze_all():
    """
    Analyze results for both game1 and game2.
    """
    for game in ["game1"]:
        print(f"Analyzing {game}...")
        model_map, output_dir = get_model_map_and_output_dir(game)
        models_to_analyze = model_map.keys()
        for model_name in models_to_analyze:
            print(f"  Processing model: {model_name}")
            # analyze(model_map[model_name], model_name, game)
            compare_mean_by_prefix_type(model_name, game)
            compare_all_results_by_prefix_type(model_name, game)


def analyze(csv_path: str, model_name: str, game: str):
    """
    Analyze the results from a CSV file:
    """
    df = pd.read_csv(csv_path)
    df = df[["prefix_type", "decision_tokens"]]
    df = convert_decision_tokens_to_dict(df)

    for prefix in PXS:
        df_for_prefix = df[df["prefix_type"] == prefix]
        analyze_mean(df_for_prefix, model_name, prefix, game)
        analyze_distribution(df_for_prefix, model_name, prefix, game)


def analyze_mean(df: pd.DataFrame, model_name: str, prefix: str, game: str):
    """
    Analyze the mean of decision tokens for a given model.
    """
    mean_value = df["decision_tokens"].mean()
    plot_mean(mean_value, model_name, prefix, game)


def plot_mean(mean_value: float, model_name: str, prefix: str, game: str):
    """
    Plot the mean of decision tokens.
    """
    _, output_dir = get_model_map_and_output_dir(game)
    plt.figure(figsize=(8, 6))
    plt.bar(model_name, mean_value, color="blue")
    plt.title(f"Prefix {prefix} Mean Decision Tokens for {model_name}")
    plt.ylabel("Mean Decision Tokens")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_{prefix}_mean_decision_tokens.png")
    plt.close()


def analyze_distribution(df: pd.DataFrame, model_name: str, prefix: str, game: str):
    """
    Analyze the distribution of decision tokens.
    """
    distribution_dict = df["decision_tokens"].value_counts().to_dict()
    df_distribution = pd.DataFrame(list(distribution_dict.items()), columns=["Decision Tokens", "Count"])
    # df_distribution.to_csv(f"{output_dir}/{model_name}_{prefix}_decision_tokens_distribution.csv", index=False)
    plot_distribution(df_distribution, model_name, prefix, game)


def plot_distribution(df_distribution: pd.DataFrame, model_name: str, prefix: str, game: str):
    """
    Plot the distribution of decision tokens.
    """
    _, output_dir = get_model_map_and_output_dir(game)
    ax = df_distribution["Decision Tokens"].plot.hist(bins=20)
    ax.set_title(f"Prefix {prefix} Distribution of '{DECISION_KEYWORDS[0]}' probability for {model_name}")
    ax.set_xlabel(f"'{DECISION_KEYWORDS[0]}' probability")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_{prefix}_{DECISION_KEYWORDS[0]}_prob_hist.png")
    plt.close()


def compare_mean_by_prefix_type(model_name: str, game: str):
    """
    Create a bar chart showing mean {DECISION_KEYWORDS[0]} (betray/wait) probability for each prefix type.
    """
    # Determine the CSV path based on model name and game
    model_map, output_dir = get_model_map_and_output_dir(game)
    csv_path = model_map[model_name]

    # Read and process data
    df = pd.read_csv(csv_path)
    # df = drop_non_conclusive_rows(df)
    df = df[["prefix_type", "decision_tokens"]]
    df = convert_decision_tokens_to_dict(df)

    # Calculate mean {DECISION_KEYWORDS[0]} (betray/wait) probability for each prefix type
    means_by_prefix = {}
    for prefix in PXS:
        df_for_prefix = df[df["prefix_type"] == prefix]
        if not df_for_prefix.empty:
            means_by_prefix[prefix] = df_for_prefix["decision_tokens"].mean()

    # Create bar plot
    plt.figure(figsize=(10, 6))
    prefixes = list(means_by_prefix.keys())
    means = list(means_by_prefix.values())

    bars = plt.bar(prefixes, means, color=["green", "blue", "red", "gray"])
    plt.title(f"Mean {DECISION_KEYWORDS[0]} Probability by Prefix Type - {model_name}")
    plt.xlabel("Prefix Type")
    plt.ylabel(f"Mean {DECISION_KEYWORDS[0]} Probability")
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{mean:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_mean_by_prefix_type.png")
    plt.close()


def compare_all_results_by_prefix_type(model_name: str, game: str):
    """
    Create a line plot with paraphrase_index on x-axis, {DECISION_KEYWORDS[0]} (betray/wait) probability on y-axis,
    different colors for each prefix_type, and connected dots for each prefix_type.
    """
    # Determine the CSV path based on model name and game
    model_map, output_dir = get_model_map_and_output_dir(game)
    csv_path = model_map[model_name]

    # Read and process data
    df = pd.read_csv(csv_path)
    # df = drop_non_conclusive_rows(df)
    df = df[["prefix_type", "paraphrase_index", "decision_tokens"]]
    df = convert_decision_tokens_to_dict(df)

    # Create the plot
    plt.figure(figsize=(12, 8))

    colors = {"positive_money": "green", "neutral_money": "blue", "negative_money": "red", "none": "gray"}

    for prefix in PXS:
        df_for_prefix = df[df["prefix_type"] == prefix]
        if not df_for_prefix.empty:
            # Group by paraphrase_index and calculate mean {DECISION_KEYWORDS[0]} (betray/wait) probability
            grouped = df_for_prefix.groupby("paraphrase_index")["decision_tokens"].mean().reset_index()

            # Sort by paraphrase_index for proper line connection
            grouped = grouped.sort_values("paraphrase_index")

            plt.plot(
                grouped["paraphrase_index"],
                grouped["decision_tokens"],
                color=colors.get(prefix, "black"),
                marker="o",
                linewidth=2,
                markersize=6,
                label=prefix.replace("_", " ").title(),
            )

    plt.title(
        f"{DECISION_KEYWORDS[0]} Probability Trends by Paraphrase Index - {model_name}", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Paraphrase Index", fontsize=16)
    plt.ylabel(f"{DECISION_KEYWORDS[0]} Probability", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_paraphrase_trends.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    analyze_all()
