import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from data import non_reasoning_models_data

# Set up the plotting style for publication quality
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def prepare_data():
    """Prepare data for visualization."""
    # Convert the dictionary to a more workable format
    data_rows = []
    for model, categories in non_reasoning_models_data.items():
        for category, probability in categories.items():
            data_rows.append({"model": model, "category": category, "probability": probability})

    df = pd.DataFrame(data_rows)
    return df


def create_money_category_graph(df):
    """Create graph showing aggregated chance to betray per money category."""
    # Calculate mean probability for each category across all models
    category_means = df.groupby("category")["probability"].agg(["mean", "std"]).reset_index()

    # Define custom order and colors for categories
    category_order = ["none", "positive_money", "neutral_money", "negative_money"]
    colors = ["#808080", "#228B22", "#FFA500", "#DC143C"]  # Grey, ForestGreen, Orange, Crimson

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars without error bars
    x_pos = np.arange(len(category_order))
    means = [category_means[category_means["category"] == cat]["mean"].iloc[0] for cat in category_order]
    stds = [category_means[category_means["category"] == cat]["std"].iloc[0] for cat in category_order]

    bars = ax.bar(x_pos, means, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)

    # Customize the plot
    ax.set_xlabel("Money Context", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability of Selfish Response", fontsize=14, fontweight="bold")
    ax.set_title(
        "Model Selfishness Across Money Conditions",
        fontsize=16,
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
    ax.set_xticklabels(category_labels, fontsize=12)

    # Customize y-axis
    ax.set_ylim(0, max(means) + 0.1)
    ax.tick_params(axis="y", labelsize=12)

    # Add value labels on top of bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Add grid and styling
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def create_money_vs_none_graph(df):
    """Create graph showing aggregated chance to betray for money contexts vs none."""
    # Create binary categorization: money contexts vs none
    df_binary = df.copy()
    df_binary["condition"] = df_binary["category"].apply(
        lambda x: "Money Context" if x in ["positive_money", "neutral_money", "negative_money"] else "No Money Context"
    )

    # Calculate aggregated probabilities
    condition_stats = df_binary.groupby("condition")["probability"].agg(["mean", "std", "count"]).reset_index()

    # Calculate standard error
    condition_stats["se"] = condition_stats["std"] / np.sqrt(condition_stats["count"])

    # Define the desired order and reorder the dataframe
    desired_order = ["No Money Context", "Money Context"]
    condition_stats = condition_stats.set_index("condition").reindex(desired_order).reset_index()

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colors in the order they will appear
    colors = ["#808080", "#1E90FF"]  # No Money Context = grey, Money Context = blue

    # Create bars
    x_pos = np.arange(len(condition_stats))
    bars = ax.bar(x_pos, condition_stats["mean"], color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)

    # Customize the plot
    ax.set_xlabel("Condition", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability of Selfish Response", fontsize=14, fontweight="bold")
    ax.set_title(
        "Effect of Money Context on Model Selfishness",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(condition_stats["condition"], fontsize=12)

    # Customize y-axis
    max_val = condition_stats["mean"].max()
    ax.set_ylim(0, max_val + 0.1)
    ax.tick_params(axis="y", labelsize=12)

    # Add value labels on top of bars
    for i, (bar, mean) in enumerate(zip(bars, condition_stats["mean"])):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Calculate and display the difference
    money_mean = condition_stats[condition_stats["condition"] == "Money Context"]["mean"].iloc[0]
    no_money_mean = condition_stats[condition_stats["condition"] == "No Money Context"]["mean"].iloc[0]
    difference = money_mean - no_money_mean
    percent_increase = (difference / no_money_mean) * 100

    # Add text box with statistics
    textstr = f"Difference: +{difference:.3f}\nIncrease: +{percent_increase:.1f}%"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11, verticalalignment="top", bbox=props)

    # Add grid and styling
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def main():
    """Generate both figures and save them."""
    # Prepare data
    df = prepare_data()

    # Create and save the first figure (money categories)
    fig1 = create_money_category_graph(df)
    fig1.savefig(
        "/Users/idoavnir/Desktop/uni/semF/nlp/project/money_games/src/visuals/non_reasoning_results/money_categories_aggregated.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()

    # Create and save the second figure (money vs none)
    fig2 = create_money_vs_none_graph(df)
    fig2.savefig(
        "/Users/idoavnir/Desktop/uni/semF/nlp/project/money_games/src/visuals/non_reasoning_results/money_vs_none_aggregated.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()

    print("Figures saved successfully!")
    print("1. money_categories_aggregated.png - Shows aggregated selfishness by money category")
    print("2. money_vs_none_aggregated.png - Shows comparison between money context vs no money context")


if __name__ == "__main__":
    main()
