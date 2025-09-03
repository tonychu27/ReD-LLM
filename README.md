# ReD-LLM: Editing Detection and Refusal Heads to Reduce Harmful Content in LLMs

This is the official repo for the paper: ReD-LLM: Editing Detection and Refusal Heads to Reduce Harmful Content in LLMs

* In this work, we investigate the internal attention mechanisms that detect harmful content and refusal behaviors in LLMs. We introduce systematic methods to identify detection heads, which are highly sensitive to harmful prompts, and refusal heads, which contribute to the model’s tendency to reject unsafe requests. By analyzing the relationship between these heads and applying targeted interventions, we demonstrate improved model robustness and safety. Our findings provide valuable insights into the structural components of LLM safety and offer practical approaches to mitigate harmful outputs, contributing to the development of more trustworthy AI systems.

* This repository contains two main parts:
    * **Detection Heads Identification**: Identifies attention heads in LLMs that are highly sensitive to harmful prompts. These *detection heads* help the model recognize potentially unsafe content.
    * **Refusal Heads Identification**: Identifies attention heads that contribute most to refusing to generate harmful responses. These *refusal heads* are responsible for producing safe refusals when harmful content is detected.

## Setup
Recommend using cuda12.1, python3.10 and pytorch2.3. To install the necessary packages:

`pip install -r requirements.txt`

## Part I: Identification of Detection Heads
![Overview of Detection Head Identification](figs/overview_detection.png)

Go into the folder for identifying detection heads:

`cd find_detection_heads`

### Finding Detection Heads
To identify the **detection heads** in the model, run:

```
python find_detection_heads.py
```

* Default Settings:
    * --model llama3
    * --percent 3.0
* Adjust Arguments:
    * --model: To sepcify which model to anaylze (e.g. llama2, mistral, guanaco)
    * --percent: To specify the top percent of heads to analyze (e.g. 2.0 to pick the top 2.0% heads)

This script analyzes the model’s attention patterns on harmful versus neutral prompts and outputs the **top attention heads that are most sensitive to harmful content.**

Results are saved in `detection_heads/<model>_<percent>.json`.

A heatmap of average attention differences is saved in `heat_map/<model>_<percent>_avg_attn_diff_heatmap.png`.

### Detection Head Intervention
To perform **intervention on detection heads** in the model, run:

```
python detection_head_intervention.py
```

* Default Settings:
    * --model llama3
    * --attack advllm
    * --dataset advbench
    * --percent 3.0
    * --factor 3.0
    * --save_path detection_head_intervention_result.csv
    * --n_train_data 100
    * --remove_sys_prompt is False
* Adjust Arguments:
    * --model: choose the backbone model (e.g., llama3, llama2).
	* --attack: type of adversarial setup (benign, none, gcg, advllm).
	* --percent: proportion of top detection heads to intervene on (default 3.0).
	* --factor: scale factor for “scale” mode.
	* --dataset: evaluation dataset (advbench or mlcinst).
	* --n_train_data: number of training samples used.
	* --offset: offset for dataset indexing.
	* --save_path: path to save the evaluation results.
	* --remove_sys_prompt: toggle to include/exclude the system prompt.


This script modifies the model’s previously identified detection heads by scaling or adjusting their output weights. It then evaluates the model’s behavior on adversarial prompts.

### Key Results
Test safety rates of intervened models. Amplifying the top 3.0% detection heads significantly improves safety under different attack types: Pure Harmful Prompt / GCG / ADV-LLM.

| Safety Rate (%) ↑ | LLaMA3 | LLaMA2 | Mistral | Guanaco |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | 100/77/15 | 100/53/18 | 100/36/5 | 62/10/7 |
| Detection (3.0) | **100/97/77** | **100/91/94** | **100/85/42** | **77/24/46** |

## Part II: Identification of Refusal Heads
![Overview of Refusal Head Identification](figs/overview_refusal.png)

Go into the folder for identifying refusal heads:

`cd find_refusal_heads`

### Response Generation
To get the model responses for harmful and neutral prompts:

```
python response_generation.py
```

* Default Settings:
    * --model llama3
    * --tensor_parallel_size 1
* Adjust Arguments:
    * --model: specify which model to generate responses with (e.g., llama3, llama2).
    * --tensor_parallel_size: set tensor parallel size when running with vLLM for distributed inference.

Outputs are saved in `responses/<model>.json`

### Extrace Refusal Direction
To get the refusal direction, run:

```
python extract_direction_attn.py
```
* Default Settings:
    * --model llama3
* Adjust Arguments:
    * --model: To specifiy which model to anaylze (e.g. llama2, mistral, guanaco)

The script computes the refusal direction by averaging post-attention residuals over generated tokens for harmful and neutral responses and taking their difference to identify attention heads driving refusal behavior.

### Finding Refusal Heads
To identify the **refusal heads** in the model, run:

```
python find_refusal_heads.py
```

* Default Settings:
    * --model llama3
    * --percent 3.0
* Adjust Arguments:
    * --model: To specifiy which model to analyze (e.g. llama2, mistral, guanaco)
    * --percent: Set the percentage of top heads to select as refusal heads.

This script scores each attention head by the similarity between its output contribution and the layer-level refusal direction, selecting the heads with highest scores as **refusal heads**.

Results are saved in `refusal_heads/<model>_<percent>.json`

### Refusal Head Intervention

To perform **intervention on refusal heads** in the model, run:

```
python refusal_head_intervention.py
```

* Default Settings:
    * --model llama3
    * --attack advllm
    * --dataset advbench
    * --percent 3.0
    * --factor 3.0
    * --save_path refusal_head_intervention_result.csv
    * --n_train_data 100
    * --offset 0
    * --remove_sys_prompt is False
* Adjust Arguments:
    * --model: choose the backbone model (e.g., llama3, llama2).
    * --attack: type of adversarial setup (benign, none, gcg, advllm).
    * --percent: proportion of top refusal heads to intervene on (default 3.0).
    * --factor: scale factor controlling the strength of the intervention.
    * --dataset: evaluation dataset (advbench or mlcinst).
    * --n_train_data: number of training samples used.
    * --offset: offset for dataset indexing.
    * --save_path: path to save the evaluation results.
    * --remove_sys_prompt: toggle to include/exclude the system prompt.

This script modifies the model’s previously identified refusal heads by scaling or adjusting their output weights. It then evaluates the model’s behavior on adversarial prompts.

### Key Results
Test safety rates of intervened models. Amplifying the top 3.0% refusal heads improves safety under different attack types: Pure Harmful Prompt / GCG / ADV-LLM.

| Safety Rate (%) ↑ | LLaMA3 | LLaMA2 | Mistral | Guanaco |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | 100/77/15 | 100/53/18 | 100/36/5 | 62/10/7 |
| Refusal (2.0) | **100/95/48** | **100/74/55** | **100/71/17** | **76/15/22** |

## Intervention on Detection Heads and Refusal Heads
After identifying detection and refusal heads, you can intervene on the model by scaling or modifying these heads to improve safety.

Run the evaluation script to apply the interventions and measure safety rates:

```
python evaluate.py
```

* Default Settings:
    * --model llama3
	* --attack advllm
	* --dataset advbench
	* --percent 3.0
	* --detection_factor 3.0
	* --refusal_factor 2.0
	* --save_path eva_result.csv
	* --n_train_data 100
	* --offset 0
	* --intervened must be specified to evaluate on regenerated GCG attack.
* Adjust Arguments:
    * --model: backbone model to evaluate (e.g., llama3, llama2).
	* --attack: type of adversarial setup (benign, none, gcg, advllm).
	* --percent: percentage of top heads used for intervention.
	* --detection_factor: scaling factor applied to detection heads.
	* --refusal_factor: scaling factor applied to refusal heads.
	* --dataset: evaluation dataset (advbench or mlcinst).
	* --n_train_data: number of training samples used.
	* --offset: offset for dataset indexing.
	* --save_path: file path to save evaluation results.

The script reports safety rates for interventions on both **detection heads** and **refusal heads**.

### Key Results
| Safety Rate (%) ↑ | LLaMA3 | LLaMA2 | Mistral | Guanaco |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | 100/77/15 | 100/53/18 | 100/36/5 | 62/10/7 |
| Detection (3.0) & Refusal (2.0) | **100/99/99** | **100/92/97** | **100/94/61** | **77/52/56** |