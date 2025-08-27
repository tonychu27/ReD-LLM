import json
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mistral")
parser.add_argument("--mode", type=str, default="refusal")
args = parser.parse_args()

data_dir = f"data_results/{args.mode}/{args.model}"

all_data = []
for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        with open(os.path.join(data_dir, filename), "r") as f:
            file_data = json.load(f)
            all_data.extend(file_data)

num_layers = max(entry["layer"] for entry in all_data) + 1
num_heads = max(entry["head"] for entry in all_data) + 1
global_max = max(entry["contribution"] for entry in all_data)

grouped_data = defaultdict(list)
for entry in all_data:
    grouped_data[(entry["factor"], entry["percent"])].append(entry)

os.makedirs(f"heatmaps/{args.mode}/{args.model}", exist_ok=True)

for (factor, percent), entries in tqdm(grouped_data.items()):
    heatmap_matrix = np.zeros((num_layers, num_heads))

    for e in entries:
        heatmap_matrix[e["layer"], e["head"]] = e["contribution"]

    plt.figure(figsize=(20, 16))
    if args.mode == "detection":
        img = plt.imshow(heatmap_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=global_max)
    else:    
        img = plt.imshow(heatmap_matrix, cmap='coolwarm', aspect='auto', vmin=-global_max, vmax=global_max)
    
    cbar = plt.colorbar(img)
    cbar.ax.tick_params(labelsize=30)
    # cbar.set_label(f'Average {args.mode.capitalize()} Contribution', fontsize=20)  # increase legend font size

    # plt.colorbar(label=f'Average {args.mode.capitalize()} Contribution',)

    input_path = f"{args.mode}_heads/{args.model}_{percent}.json"
    with open(input_path, "r") as f:
        input_list = json.load(f)
    heads = [(int(h["layer"]), int(h["head"])) for h in input_list]

    # plt.scatter(
    #     [h[1] for h in heads],
    #     [h[0] for h in heads],
    #     marker='*',
    #     edgecolors='black',
    #     linewidths=1.5,
    #     s=100,
    #     label='Refusal Heads' if args.mode == "refusal" else 'Detection Heads'
    # )

    if factor == 1.0:
        plt.title(f'{args.model.capitalize()} Average {args.mode.capitalize()} Contribution', fontsize=45)
    else:
        plt.title(f'{args.model.capitalize()} Average {args.mode.capitalize()} Contribution, factor: {factor}', fontsize=45)
    
    plt.xlabel('Heads', fontsize=30)
    plt.ylabel('Layers', fontsize=30)

    plt.xticks(ticks=np.arange(0, num_heads, 5), labels=[f'H{i}' for i in range(0, num_heads, 5)], fontsize=30, rotation=45)
    plt.yticks(ticks=np.arange(0, num_layers, 5), labels=[f'L{i}' for i in range(0, num_layers, 5)], fontsize=30)

    plt.tight_layout()
    plt.savefig(f'heatmaps/{args.mode}/{args.model}/{args.model}_{args.mode}_attn_{factor}_{percent}.png', dpi=300)
    plt.close()