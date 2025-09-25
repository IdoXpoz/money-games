import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import sys
import os

# Add the project root to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.prompts.configs.money import PREFIXES
from src.prompts.configs.games2 import DECISION_KEYWORDS

MODEL_TO_CSV_PATH_MAP_GAME1 = {
    # "gemma-3-4b-it": "src/analysis/game1/gemma-3-4b-it_results.csv",
    # "gemma-3-12b-it": "src/analysis/game1/gemma-3-12b-it_results.csv",
    # "chat-gemma-3-4b-it": "src/analysis/game1/chat-gemma-3-4b-it_results.csv",
    # "chat-gemma-3-12b-it": "src/analysis/game1/chat-gemma-3-12b-it_results.csv",
    # "Qwen-without-temperature": "src/analysis/game1/qwen_results.csv",
    "Qwen": "src/analysis/game1/qwen_results_with_temperature.csv",
    "chat-llama-3.2-3B-Instruct": "src/analysis/game1/chat-llama-3.2-3b-instruct_results.csv",
}

MODEL_TO_CSV_PATH_MAP_GAME2 = {
    # "chat-gemma-3-1b-pt": "src/analysis/game2/chat-gemma-3-1b-pt_results.csv",
    # "chat-gemma-3-1b-it": "src/analysis/game2/chat-gemma-3-1b-it_results.csv",
    # "chat-gemma-3-4b-pt": "src/analysis/game2/chat-gemma-3-4b-pt_results.csv",
    # "chat-gemma-3-4b-it": "src/analysis/game2/chat-gemma-3-4b-it_results.csv",
    # "chat-gemma-3-12b-pt": "src/analysis/game2/chat-gemma-3-12b-pt_results.csv",
    # "chat-gemma-3-12b-it": "src/analysis/game2/chat-gemma-3-12b-it_results.csv",
    # "chat-llama-3.2-3B-Instruct": "src/analysis/game2/chat-llama-3.2-3b-instruct_results.csv",
    # "Qwen-4b": "src/analysis/game2/qwen-4b_results_with_temperature.csv",
    # "Qwen-8b": "src/analysis/game2/qwen-8b_results_with_temperature.csv",
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
    # Set up the plotting style for publication quality
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

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

    # Define custom order and colors for categories to match other figures
    category_order = ["none", "positive_money", "neutral_money", "negative_money"]
    colors = ["#808080", "#228B22", "#FFA500", "#DC143C"]  # Grey, ForestGreen, Orange, Crimson

    # Reorder data to match the desired order
    ordered_prefixes = [prefix for prefix in category_order if prefix in means_by_prefix]
    ordered_means = [means_by_prefix[prefix] for prefix in ordered_prefixes]
    ordered_colors = [colors[category_order.index(prefix)] for prefix in ordered_prefixes]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    x_pos = np.arange(len(ordered_prefixes))
    bars = ax.bar(x_pos, ordered_means, color=ordered_colors, alpha=0.8, edgecolor="black", linewidth=1.2)

    # Customize the plot
    ax.set_xlabel("Money Context", fontsize=20, fontweight="bold")
    ax.set_ylabel(f"Probability of {DECISION_KEYWORDS[0].title()} Response", fontsize=20, fontweight="bold")
    ax.set_title(
        f"Model {DECISION_KEYWORDS[0].title()} Probability Across Money Conditions - {model_name}",
        fontsize=24,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis labels
    category_labels = [
        "No Money\nContext",
        "Positive Money\nContext",
        "Neutral Money\nContext",
        "Negative Money\nContext",
    ]
    ax.set_xticks(x_pos)
    ax.set_xticklabels([category_labels[category_order.index(prefix)] for prefix in ordered_prefixes], fontsize=18)

    # Customize y-axis
    ax.set_ylim(0, max(ordered_means) + 0.1)
    ax.tick_params(axis="y", labelsize=18)

    # Add value labels on top of bars
    for i, (bar, mean) in enumerate(zip(bars, ordered_means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=16,
        )

    # Add grid and styling
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_mean_by_prefix_type.png", dpi=300, bbox_inches="tight", facecolor="white")
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

    colors = {"none": "#808080", "positive_money": "#228B22", "neutral_money": "#FFA500", "negative_money": "#DC143C"}

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
        f"{DECISION_KEYWORDS[0]} Probability Trends by Paraphrase Index - {model_name}", fontsize=24, fontweight="bold"
    )
    plt.xlabel("Paraphrase Index", fontsize=20)
    plt.ylabel(f"{DECISION_KEYWORDS[0]} Probability", fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_paraphrase_trends.png", bbox_inches="tight")
    plt.close()


def compare_combined_qwen_models_game2():
    """
    Create a bar chart showing mean wait probability for each prefix type,
    combining data from both qwen-4b and qwen-8b models for game2.
    """
    # Set up the plotting style for publication quality
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # Define the paths to both CSV files
    qwen_4b_path = "src/analysis/game2/qwen-4b_results_with_temperature.csv"
    qwen_8b_path = "src/analysis/game2/qwen-8b_results_with_temperature.csv"
    output_dir = OUTPUT_DIR_GAME2

    # Read and process data from both models
    df_4b = pd.read_csv(qwen_4b_path)
    df_8b = pd.read_csv(qwen_8b_path)

    # Combine the data
    df_combined = pd.concat([df_4b, df_8b], ignore_index=True)

    # Process the combined data
    df_combined = df_combined[["prefix_type", "decision_tokens"]]
    df_combined = convert_decision_tokens_to_dict(df_combined)

    # Calculate mean wait probability for each prefix type
    means_by_prefix = {}
    for prefix in PXS:
        df_for_prefix = df_combined[df_combined["prefix_type"] == prefix]
        if not df_for_prefix.empty:
            means_by_prefix[prefix] = df_for_prefix["decision_tokens"].mean()

    # Define custom order and colors for categories to match other figures
    category_order = ["none", "positive_money", "neutral_money", "negative_money"]
    colors = ["#808080", "#228B22", "#FFA500", "#DC143C"]  # Grey, ForestGreen, Orange, Crimson

    # Reorder data to match the desired order
    ordered_prefixes = [prefix for prefix in category_order if prefix in means_by_prefix]
    ordered_means = [means_by_prefix[prefix] for prefix in ordered_prefixes]
    ordered_colors = [colors[category_order.index(prefix)] for prefix in ordered_prefixes]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    x_pos = np.arange(len(ordered_prefixes))
    bars = ax.bar(x_pos, ordered_means, color=ordered_colors, alpha=0.8, edgecolor="black", linewidth=1.2)

    # Customize the plot
    ax.set_xlabel("Money Context", fontsize=20, fontweight="bold")
    ax.set_ylabel("Probability of Wait Response", fontsize=20, fontweight="bold")
    ax.set_title(
        "Model Wait Probability Across Money Conditions - Qwen",
        fontsize=22,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis labels
    category_labels = [
        "No Money\nContext",
        "Positive Money\nContext",
        "Neutral Money\nContext",
        "Negative Money\nContext",
    ]
    ax.set_xticks(x_pos)
    ax.set_xticklabels([category_labels[category_order.index(prefix)] for prefix in ordered_prefixes], fontsize=18)

    # Customize y-axis
    ax.set_ylim(0, max(ordered_means) + 0.1)
    ax.tick_params(axis="y", labelsize=18)

    # Add value labels on top of bars
    for i, (bar, mean) in enumerate(zip(bars, ordered_means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=16,
        )

    # Add grid and styling
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/Combined_Qwen_4B_8B_wait_probability_by_money_context.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    # Print the results for verification
    print("Combined Qwen Models (4B + 8B) Wait Probability by Money Context:")
    for prefix in ordered_prefixes:
        print(f"{prefix}: {means_by_prefix[prefix]:.3f}")

    return f"{output_dir}/Combined_Qwen_4B_8B_wait_probability_by_money_context.png"


if __name__ == "__main__":
    analyze_all()
