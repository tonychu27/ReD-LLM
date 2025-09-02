import json
import argparse
import torch
import os

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import MODEL_DICT, load_conversation_template, filter_response

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3")
parser.add_argument("--percent", type=float, default=3.0)
parser.add_argument("--factor", type=float, default=3.0)
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_path = MODEL_DICT[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model.generation_config.do_sample = True
tokenizer.pad_token = tokenizer.eos_token

refusal_direction = torch.load(f"find_refusal_heads/directions/{args.model}_refusal_direction_attn.pt").to(device)
print(refusal_direction.size())
refusal_direction = refusal_direction / torch.norm(refusal_direction, dim=-1, keepdim=True)

with open(f"find_refusal_heads/responses/{args.model}.json", "r") as f:
    response_data = json.load(f)

harmful_prompts = [entry["Harmful Prompt"] for entry in response_data]
neutral_prompts = [entry["Neutral Prompt"] for entry in response_data]
harmful = [entry["Harmful Response"] for entry in response_data]
neutral = [entry["Neutral Response"] for entry in response_data]

filtered_harmful_prompt, _, filtered_harmful, _ = filter_response(harmful_prompts, neutral_prompts, harmful, neutral)


num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
num_heads = model.config.num_attention_heads
head_dim = hidden_size // num_heads

detection_path = f"find_detection_heads/detection_heads/{args.model}_{args.percent}.json"
with open(detection_path, "r") as f:
    det_list = json.load(f)
detection_heads = [(int(d["layer"]), int(d["head"])) for d in det_list]

for layer_idx, head_idx in detection_heads:
    start = head_idx * head_dim
    end = start + head_dim

    W = model.model.layers[layer_idx].self_attn.o_proj.weight.detach().clone()
    W[:, start:end] *= args.factor

    model.model.layers[layer_idx].self_attn.o_proj.weight = nn.Parameter(W)

print(f"Modified {len(detection_heads)} heads in {args.model} with factor {args.factor}")

attn_contribution = []

def capture_attn_contribution_hook():
    def hook_fn(module, input, output):
        attn_out = input[0].detach()[0, :, :]
        attn_out = attn_out.reshape(attn_out.size(0), num_heads, head_dim)
        o_proj = module.weight.detach().clone()
        o_proj = o_proj.reshape(hidden_size, num_heads, head_dim).permute(1, 2, 0).contiguous()
        attn_contribution.append(torch.einsum("snk,nkh->snh", attn_out, o_proj))
    return hook_fn

for layer in model.model.layers:
    layer.self_attn.o_proj.register_forward_hook(capture_attn_contribution_hook())

conv_template = load_conversation_template(args.model)
avg_contribution = torch.zeros((num_layers, num_heads))

for harmful_prompt, harmful_response in tqdm(zip(filtered_harmful_prompt, filtered_harmful), total=len(filtered_harmful)):

    conv_template.messages = []
    conv_template.append_message(conv_template.roles[0], f"{harmful_prompt}")
    conv_template.append_message(conv_template.roles[1], None)
    
    toks = tokenizer(conv_template.get_prompt()).input_ids
    start = len(toks)

    conv_template.update_last_message(f"{harmful_response}")
    toks = tokenizer(conv_template.get_prompt()).input_ids

    if args.model == "llama3":
        end = len(toks) - 1
    else:
        end = len(toks)

    toks = tokenizer(conv_template.get_prompt(), return_tensors="pt")
    _ = model(input_ids=toks['input_ids'].to(device), attention_mask=toks['attention_mask'].to(device))
    all_head_contributions = torch.stack(attn_contribution, dim=0)[:, start-1: end-1, :, :].mean(dim=1)

    refusal_dir_exp = refusal_direction.unsqueeze(1)
    dot_products = torch.einsum('lhd,ld->lh', all_head_contributions, refusal_direction)
    avg_contribution = np.add(avg_contribution, dot_products.cpu().numpy())

    attn_contribution = []
    
avg_contribution = np.asarray(avg_contribution) / len(filtered_harmful)

result_list = []
for layer in range(num_layers):
    for head in range(num_heads):
        result_list.append({
            "layer": layer,
            "head": head,
            "factor": args.factor,
            "percent": args.percent,
            "contribution": float(avg_contribution[layer, head])
        })

os.makedirs(f"data_results/refusal/{args.model}", exist_ok=True)
result_path = f"data_results/refusal/{args.model}/{args.model}_{args.factor}_{args.percent}.json"
with open(result_path, "w") as f:
    json.dump(result_list, f, indent=4)

print(f"Saved results to {result_path}")

max_abs_value = np.abs(avg_contribution[:, :]).max()

plt.figure(figsize=(12, 10))
plt.imshow(avg_contribution[:, :], cmap='coolwarm', aspect='auto', vmin=-max_abs_value, vmax=max_abs_value)
plt.colorbar(label='Average refusal contribution')

refusal_path = f"find_refusal_heads/refusal_heads/{args.model}_{args.percent}.json"
with open(refusal_path, "r") as f:
    refusal_list = json.load(f)
refusal_heads = [(int(h["layer"]), int(h["head"])) for h in refusal_list]

plt.scatter(
    [h[1] for h in refusal_heads],
    [h[0] for h in refusal_heads],
    marker='*',
    edgecolors='black',
    linewidths=1.5,
    s=100,
    label='Refusal Heads'
)

plt.title(f'{args.model.capitalize()} Heatmap of Average Refusal Contribution with Factor {args.factor} and Percent {args.percent}', fontsize=12)
plt.xlabel('Heads')
plt.ylabel('Layers')

plt.xticks(ticks=np.arange(num_heads), labels=[f'H{i}' for i in range(num_heads, 2)], fontsize=6, rotation=45)
plt.yticks(ticks=np.arange(num_layers), labels=[f'L{i}' for i in range(num_layers, 2)], fontsize=8)

plt.tight_layout()
plt.savefig(f'heatmaps/refusal/{args.model}_refusal_attn_{args.factor}_{args.percent}.png', dpi=300)
print(f"Saved heatmap to heatmaps/refusal/{args.model}_refusal_attn_{args.factor}_{args.percent}.png")