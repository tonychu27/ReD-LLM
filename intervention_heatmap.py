import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str, default="advllm")
args = parser.parse_args()

# Load dataset
df = pd.read_csv(f"csv_results/attack_result_{args.attack}.csv")

# Create output folder
output_dir = "detection_refusal_heatmap"
os.makedirs(output_dir, exist_ok=True)

# Loop through models and save each heatmap
for model in df["model"].unique():
    model_df = df[df["model"] == model]
    heatmap_data = model_df.pivot_table(
        index="refusal_factor",
        columns="detection_factor",
        values="safe_guard"
    )

    plt.figure(figsize=(20, 12))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="viridis",
        cbar=True
    )
    plt.title(f"Model: {model} | Attack: {args.attack}")
    plt.xlabel("Detection Factor")
    plt.ylabel("Refusal Factor")

    # Save each heatmap
    filepath = os.path.join(output_dir, f"heatmap_{model}_{args.attack}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

print(f"Heatmaps saved in: {output_dir}")