import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from src.prompts.configs.money import PREFIXES

GEMMA_RESULTS_CSV_PATH = "src/analysis/gemma_results.csv"
QWEN_RESULTS_CSV_PATH = "src/analysis/qwen_results.csv"
PXS = PREFIXES.keys()
OUTPUT_DIR = "src/analysis/trends_analysis_results"


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
                if k == "betray"
            ),
            None,
        )
    )
    df.dropna(subset=["decision_tokens"], inplace=True)
    return df


def analyze_all():
    # analyze(GEMMA_RESULTS_CSV_PATH, "Gemma")
    # analyze(QWEN_RESULTS_CSV_PATH, "Qwen")
    # New analyses
    analyze_mean_by_prefix_type("Gemma")
    analyze_mean_by_prefix_type("Qwen")
    analyze_paraphrase_trends("Gemma")
    analyze_paraphrase_trends("Qwen")


def analyze(csv_path: str, model_name: str):
    """
    Analyze the results from a CSV file:
    """
    df = pd.read_csv(csv_path)
    df = df[["prefix_type", "decision_tokens"]]
    df = convert_decision_tokens_to_dict(df)

    for prefix in PXS:
        df_for_prefix = df[df["prefix_type"] == prefix]
        analyze_mean(df_for_prefix, model_name, prefix)
        analyze_distribution(df_for_prefix, model_name, prefix)


def analyze_mean(df: pd.DataFrame, model_name: str, prefix: str):
    """
    Analyze the mean of decision tokens for a given model.
    """
    mean_value = df["decision_tokens"].mean()
    plot_mean(mean_value, model_name, prefix)


def plot_mean(mean_value: float, model_name: str, prefix: str):
    """
    Plot the mean of decision tokens.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(model_name, mean_value, color="blue")
    plt.title(f"Prefix {prefix} Mean Decision Tokens for {model_name}")
    plt.ylabel("Mean Decision Tokens")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_{prefix}_mean_decision_tokens.png")
    plt.close()


def analyze_distribution(df: pd.DataFrame, model_name: str, prefix: str):
    """
    Analyze the distribution of decision tokens.
    """
    distribution_dict = df["decision_tokens"].value_counts().to_dict()
    df_distribution = pd.DataFrame(list(distribution_dict.items()), columns=["Decision Tokens", "Count"])
    # df_distribution.to_csv(f"{OUTPUT_DIR}/{model_name}_{prefix}_decision_tokens_distribution.csv", index=False)
    plot_distribution(df_distribution, model_name, prefix)


def plot_distribution(df_distribution: pd.DataFrame, model_name: str, prefix: str):
    """
    Plot the distribution of decision tokens.
    """
    ax = df_distribution["Decision Tokens"].plot.hist(bins=20)
    ax.set_title(f"Prefix {prefix} Distribution of 'betray' probability for {model_name}")
    ax.set_xlabel("'betray' probability")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_{prefix}_betray_prob_hist.png")
    plt.close()


def analyze_mean_by_prefix_type(model_name: str):
    """
    Create a bar chart showing mean betray probability for each prefix type.
    """
    # Determine the CSV path based on model name
    csv_path = GEMMA_RESULTS_CSV_PATH if model_name == "Gemma" else QWEN_RESULTS_CSV_PATH

    # Read and process data
    df = pd.read_csv(csv_path)
    df = df[["prefix_type", "decision_tokens"]]
    df = convert_decision_tokens_to_dict(df)

    # Calculate mean betray probability for each prefix type
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
    plt.title(f"Mean Betray Probability by Prefix Type - {model_name}")
    plt.xlabel("Prefix Type")
    plt.ylabel("Mean Betray Probability")
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{mean:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_mean_by_prefix_type.png")
    plt.close()


def analyze_paraphrase_trends(model_name: str):
    """
    Create a line plot with paraphrase_index on x-axis, betray probability on y-axis,
    different colors for each prefix_type, and connected dots for each prefix_type.
    """
    # Determine the CSV path based on model name
    csv_path = GEMMA_RESULTS_CSV_PATH if model_name == "Gemma" else QWEN_RESULTS_CSV_PATH

    # Read and process data
    df = pd.read_csv(csv_path)
    df = df[["prefix_type", "paraphrase_index", "decision_tokens"]]
    df = convert_decision_tokens_to_dict(df)

    # Create the plot
    plt.figure(figsize=(12, 8))

    colors = {"positive_money": "green", "neutral_money": "blue", "negative_money": "red", "none": "gray"}

    for prefix in PXS:
        df_for_prefix = df[df["prefix_type"] == prefix]
        if not df_for_prefix.empty:
            # Group by paraphrase_index and calculate mean betray probability
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

    plt.title(f"Betray Probability Trends by Paraphrase Index - {model_name}")
    plt.xlabel("Paraphrase Index")
    plt.ylabel("Betray Probability")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_paraphrase_trends.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    analyze_all()
