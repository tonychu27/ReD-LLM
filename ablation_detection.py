import json
import argparse
import torch
import os

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
hidden_size = model.config.hidden_size
head_dim = hidden_size // num_heads

# -------------------- Load refusal heads --------------------
detect_path = f"refusal_heads/{args.model}_{args.percent}.json"
with open(detect_path, "r") as f:
    det_list = json.load(f)
heads = [(int(d["layer"]), int(d["head"])) for d in det_list]

# -------------------- Apply intervention --------------------
for layer_idx, head_idx in heads:
    start = head_idx * head_dim
    end = start + head_dim

    W = model.model.layers[layer_idx].self_attn.o_proj.weight.detach().clone()

    W[:, start:end] *= args.factor

    model.model.layers[layer_idx].self_attn.o_proj.weight = nn.Parameter(W)

print(f"Modified {len(heads)} heads in {args.model} with factor {args.factor}")


probing_data = pd.read_csv("../filtered_harmful_neutral_prompts.csv")
harmful_prompts = probing_data['Harmful Prompt'].tolist()
neutral_prompts = probing_data['Neutral Prompt'].tolist()

conv_template = load_conversation_template(args.model)

all_diff = torch.zeros(num_layers, num_heads, device=device)

for h, n in zip(harmful_prompts, neutral_prompts):
    conv_template.messages = []
    conv_template.append_message(conv_template.roles[0], f"{h}")
    conv_template.append_message(conv_template.roles[1], None)
    with torch.no_grad():
        toks_h = tokenizer(conv_template.get_prompt(), return_tensors="pt")
        h_attn_weights = model(input_ids=toks_h['input_ids'].to(device), attention_mask=toks_h['attention_mask'].to(device), output_attentions=True).attentions                

    conv_template.messages = []
    conv_template.append_message(conv_template.roles[0], f"{n}")
    conv_template.append_message(conv_template.roles[1], None)
    with torch.no_grad():
        toks_n = tokenizer(conv_template.get_prompt(), return_tensors="pt")
        n_attn_weights = model(input_ids=toks_n['input_ids'].to(device), attention_mask=toks_n['attention_mask'].to(device), output_attentions=True).attentions                

    # stack into tensors of shape [num_layers, batch=1, heads, seq_len, seq_len]
    h_stack = torch.stack(h_attn_weights, dim=0)
    n_stack = torch.stack(n_attn_weights, dim=0)

    # pick out the last query position: result shape [num_layers, heads, seq_len]
    h_last = h_stack[:, 0, :, -1, :]
    n_last = n_stack[:, 0, :, -1, :]

    # find positions where the two token sequences differ
    diff_pos = (toks_h['input_ids'][0] != toks_n['input_ids'][0]).nonzero(as_tuple=True)[0].to(device)

    # gather only at those positions and average over them
    # h_sel/n_sel: [num_layers, heads, num_diff_positions]
    h_sel = h_last.index_select(-1, diff_pos)
    n_sel = n_last.index_select(-1, diff_pos)

    # compute per-layer/head mean difference: shape [num_layers, heads]
    avg_diff = h_sel.mean(dim=-1) - n_sel.mean(dim=-1)
    
    all_diff += avg_diff


all_diff /= len(harmful_prompts)
all_diff = all_diff.cpu().numpy()

result_list = []
for layer in range(num_layers):
    for head in range(num_heads):
        result_list.append({
            "layer": layer,
            "head": head,
            "factor": args.factor,
            "percent": args.percent,
            "contribution": float(all_diff[layer, head])
        })

os.makedirs(f"data_results/detection/{args.model}", exist_ok=True)
result_path = f"data_results/detection/{args.model}/{args.model}_{args.factor}_{args.percent}.json"
with open(result_path, "w") as f:
    json.dump(result_list, f, indent=4)
print(f"Saved results to {result_path}")