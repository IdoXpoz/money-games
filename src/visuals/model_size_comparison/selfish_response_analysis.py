import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import os

# Set up the plotting style for consistency with other graphs
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def parse_decision_tokens(tokens_str):
    """Parse the decision_tokens string and return as dictionary"""
    try:
        # Convert string representation of list to actual list
        tokens_list = ast.literal_eval(tokens_str)
        return dict(tokens_list)
    except:
        return {}


def extract_selfish_probability(tokens_dict, game_type):
    """Extract the probability of selfish response based on game type"""
    if game_type == "game1":  # Prisoner's dilemma - 'betray' is selfish
        return tokens_dict.get("betray", 0.0)
    elif game_type == "game2":  # Volunteer dilemma - 'wait' is selfish
        return tokens_dict.get("wait", 0.0)
    return 0.0


def load_and_process_data(file_path, game_type, model_name):
    """Load CSV and extract selfish response probabilities"""
    print(f"Processing {model_name} from {game_type}...")

    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")

        # Parse decision tokens and extract selfish probabilities
        selfish_probs = []
        for _, row in df.iterrows():
            tokens_dict = parse_decision_tokens(row["decision_tokens"])
            selfish_prob = extract_selfish_probability(tokens_dict, game_type)
            selfish_probs.append(selfish_prob)

        avg_selfish_prob = np.mean(selfish_probs)
        print(f"{model_name} average selfish probability: {avg_selfish_prob:.4f}")

        return avg_selfish_prob, selfish_probs

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0.0, []


def main():
    # Define the data files and their corresponding information
    data_files = {
        "Gemma-4B": [
            (
                "/Users/idoavnir/Desktop/uni/semF/nlp/project/money_games/src/analysis/game1/gemma-3-4b-it_results.csv",
                "game1",
            )
        ],
        "Gemma-12B": [
            (
                "/Users/idoavnir/Desktop/uni/semF/nlp/project/money_games/src/analysis/game1/gemma-3-12b-it_results.csv",
                "game1",
            ),
            (
                "/Users/idoavnir/Desktop/uni/semF/nlp/project/money_games/src/analysis/game2/chat-gemma-3-12b-it_results.csv",
                "game2",
            ),
        ],
        "Qwen-4B": [
            (
                "/Users/idoavnir/Desktop/uni/semF/nlp/project/money_games/src/analysis/game1/qwen_results_with_temperature.csv",
                "game1",
            ),
            (
                "/Users/idoavnir/Desktop/uni/semF/nlp/project/money_games/src/analysis/game2/qwen-4b_results_with_temperature.csv",
                "game2",
            ),
        ],
        "Qwen-8B": [
            (
                "/Users/idoavnir/Desktop/uni/semF/nlp/project/money_games/src/analysis/game2/qwen-8b_results_with_temperature.csv",
                "game2",
            )
        ],
    }

    # Process each model and collect average selfish probabilities
    model_averages = {}
    all_data = {}

    for model_name, file_info_list in data_files.items():
        model_selfish_probs = []

        for file_path, game_type in file_info_list:
            if os.path.exists(file_path):
                avg_prob, individual_probs = load_and_process_data(file_path, game_type, f"{model_name}-{game_type}")
                model_selfish_probs.extend(individual_probs)
            else:
                print(f"Warning: File not found: {file_path}")

        if model_selfish_probs:
            model_averages[model_name] = np.mean(model_selfish_probs)
            all_data[model_name] = model_selfish_probs
            print(f"\n{model_name} overall average: {model_averages[model_name]:.4f}\n")

    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(model_averages.keys())
    averages = list(model_averages.values())

    # Create bar chart with colors matching the project style
    x_pos = np.arange(len(models))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]  # Keep your original colors but match style
    bars = ax.bar(
        x_pos,
        averages,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    # Customize the plot
    ax.set_title("Average Probability of Selfish Response by Model", fontsize=24, fontweight="bold", pad=20)
    ax.set_xlabel("Model", fontsize=20, fontweight="bold")
    ax.set_ylabel("Average Probability of Selfish Response", fontsize=20, fontweight="bold")

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=18)

    # Customize y-axis
    ax.set_ylim(0, max(averages) * 1.15)
    ax.tick_params(axis="y", labelsize=18)

    # Add value labels on top of bars
    for i, (bar, avg) in enumerate(zip(bars, averages)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{avg:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=16,
        )

    # Add grid and styling to match other graphs
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save the plot with consistent parameters
    output_path = (
        "/Users/idoavnir/Desktop/uni/semF/nlp/project/money_games/src/visuals/average_selfish_response_by_model.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    print(f"\nVisualization saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    for model, avg in model_averages.items():
        count = len(all_data[model])
        print(f"{model}: {avg:.4f} (n={count} responses)")


if __name__ == "__main__":
    main()
