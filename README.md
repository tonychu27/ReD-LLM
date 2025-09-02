# ReD-LLM: Editing Detection and Refusal Heads to Reduce Harmful Content in LLMs

This is the official repo for the paper: ReD-LLM: Editing Detection and Refusal Heads to Reduce Harmful Content in LLMs

* In this work, we investigate the internal attention mechanisms that detect harmful content and refusal behaviors in LLMs. We introduce systematic methods to identify detection heads, which are highly sensitive to harmful prompts, and refusal heads, which contribute to the model’s tendency to reject unsafe requests. By analyzing the relationship between these heads and applying targeted interventions, we demonstrate improved model robustness and safety. Our findings provide valuable insights into the structural components of LLM safety and offer practical approaches to mitigate harmful outputs, contributing to the development of more trustworthy AI systems.

* This repository contains two main parts:
    * Detection Heads Identification: Identifies attention heads in LLMs that are highly sensitive to harmful prompts. These *detection heads* help the model recognize potentially unsafe content.
    * Refusal Heads Identification: Identifies attention heads that contribute most to refusing to generate harmful responses. These *refusal heads* are responsible for producing safe refusals when harmful content is detected.