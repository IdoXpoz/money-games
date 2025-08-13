import pandas as pd
import matplotlib.pyplot as plt
import ast
from src.prompts.configs.money import PREFIXES

GEMMA_RESULTS_CSV_PATH = "src/analysis/gemma_results.csv"
QWEN_RESULTS_CSV_PATH = "src/analysis/qwen_results.csv"
PXS = PREFIXES.keys()
OUTPUT_DIR = "src/analysis/trends_analysis_results"

def analyze_all():
    analyze(GEMMA_RESULTS_CSV_PATH, "Gemma")
    analyze(QWEN_RESULTS_CSV_PATH, "Qwen")

def analyze(csv_path: str, model_name: str):
    """
    Analyze the results from a CSV file:
    """
    df = pd.read_csv(csv_path)
    df = df[["prefix_type", "decision_tokens"]]
    df["decision_tokens"] = df["decision_tokens"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else s
    )
    df["decision_tokens"] = df["decision_tokens"].apply(
        lambda lst: next(
            (v for item in (lst or [])
            if isinstance(item, (list, tuple)) and len(item) == 2
            for k, v in [item] if k == "betray"),
            None
        )
    )
    df.dropna(subset=["decision_tokens"], inplace=True)
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
    plt.bar(model_name, mean_value, color='blue')
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
    #df_distribution.to_csv(f"{OUTPUT_DIR}/{model_name}_{prefix}_decision_tokens_distribution.csv", index=False)
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

if __name__ == "__main__":
    analyze_all()