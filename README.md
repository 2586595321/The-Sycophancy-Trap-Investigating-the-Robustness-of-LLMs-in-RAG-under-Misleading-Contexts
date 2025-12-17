# The Sycophancy Trap: Investigating LLM Robustness in RAG

This repository contains the code and experimental data for the final project: **"The Sycophancy Trap: Investigating the Robustness of LLMs in RAG under Misleading Contexts"**.

## 📌 Abstract
Retrieval-Augmented Generation (RAG) systems often assume retrieved contexts are accurate. This project investigates a critical failure mode: **misleading context injection**. We benchmark three state-of-the-art open-weights models (**Mistral-7B**, **Qwen-2.5-7B**, **Gemma-2-9B**) across two datasets (General Knowledge and Natural Questions).

Our findings reveal a significant "sycophancy trap": stronger models like Gemma-2-9B are paradoxically more susceptible to misleading contexts, dropping to **0.1% accuracy** on the NQ dataset when presented with false information.

## 🚀 Models & Setup
We utilized **NVIDIA A100 GPUs** on Google Colab. All models were loaded in `bfloat16` precision to ensure maximum accuracy without quantization.

* **Models:**
    * `mistralai/Mistral-7B-Instruct-v0.3`
    * `Qwen/Qwen2.5-7B-Instruct`
    * `google/gemma-2-9b-it`
* **Datasets:**
    * General Knowledge (Simulated/TruthfulQA style)
    * Natural Questions (Open Domain)

## 📂 Repository Structure
* `RAG_Robustness_Experiment.ipynb`: The main notebook containing the full pipeline (Data Generation -> Inference -> Evaluation).
* `final_rag_benchmark_comparison.png`: Visualization of results on the General Knowledge dataset.
* `final_rag_benchmark_comparison_nq.png`: Visualization of results on the Natural Questions dataset.

## 🛠️ How to Run

### 1. Installation
The code requires the following libraries. Run the installation cell in the notebook:
```bash
pip install -q -U torch transformers accelerate datasets pandas
