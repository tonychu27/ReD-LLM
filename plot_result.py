import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Load data
parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str, default="advllm", help="benign, none, gcg, advllm, advllm_gbs")
parser.add_argument("--detection", action=argparse.BooleanOptionalAction)
parser.add_argument("--refusal", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

df = pd.read_csv(f"csv_results/attack_result_{args.attack}.csv")

factor_title = None
if args.detection and args.refusal or (not args.detection and not args.refusal):
    raise("Please specific either detection heads or refusal heads")
elif args.detection:
    factor_title = "Detection Heads Intervention Factor"
    df = df[df['detection_factor'] >= 1.0]
    df = df[df['detection_factor'] <= 3.0]
    df = df[df['refusal_factor'] == 1.0]
    factor = "detection_factor"
elif args.refusal:
    factor_title = "Refusal Heads Intervention Factor"
    df = df[df['detection_factor'] == 1.0]
    df = df[df['refusal_factor'] >= 1.0]
    df = df[df['refusal_factor'] <= 3.0]
    factor = "refusal_factor"

df = df[df['model'] != 'vicuna']

# Fixed color map
color_map = {
    "llama3": "purple",
    "llama2": "red",
    "mistral": "blue",
    "guanaco": "orange",
    "vicuna": "green"
}

# Create the plot
plt.figure(figsize=(15, 8))
sns.set(style="whitegrid", font_scale=2.5)

for model, sub_df in df.groupby("model"):
    sub_df = sub_df.sort_values(factor)
    plt.plot(
        sub_df[factor],
        sub_df["safe_guard"],
        label=model,
        color=color_map.get(model, "black"),
        marker="o",
        linewidth=2,
        markersize=6
    )

plt.xlabel(factor_title, fontsize=26)
plt.ylabel("Safe Rate", fontsize=26)
plt.title(f"Llama-Guard vs {factor_title}", fontsize=30)
plt.legend(title="Model", fontsize=24)

# Legend outside
plt.legend(
    fontsize=25,
    bbox_to_anchor=(1.05, 1),  # Position legend outside right
    loc='upper left',
    borderaxespad=0.
)

save_path = f"plotting/{args.attack}/detection_heads.png" if args.detection else f"plotting/{args.attack}/refusal_heads.png"
plt.tight_layout()
os.makedirs(f"plotting/{args.attack}", exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.show()
