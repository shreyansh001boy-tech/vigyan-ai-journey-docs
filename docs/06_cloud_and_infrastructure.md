# 06. Multi-Cloud Topography & Credit Control

## 1. Credit-Preserving Architecture

A central tenet of the Vigyan AI journey was **sovereign financial discipline**: building enterprise-grade foundation models while preserving non-dilutive grant capital.

### AWS Cloud Budget Status:
* **Initial Grant**: $1,000.00 USD.
* **Capital Invested in Training**: ~$199.93 USD (Across distributed FSDP, SFT, and eval runs).
* **Current Preserved Balance**: **$800.07 USD**.
* **Active Instances Running**: **0** (Zero idle burn rate).

---

## 2. Headless GPU Execution (Zero Spend)

To minimize AWS credit drawdowns, all evaluation and synthetic curation jobs were offloaded to headless free GPU fleets:
* **Kaggle Dual NVIDIA T4 (32 GB VRAM)**: 30 hours/week utilized for 7B GSM8k evaluation, LanceDB vector ingestion, and 50k dataset curation.
* **Google Colab Tesla T4**: Headless probing and inference verification.

---

## 3. Hugging Face Storage Vault (81+ GB)

All intellectual property, model weights, adapters, and scorecards are backed up across 8 repositories under `shreyansh12183`:

| Repository Name | Type | Size | Contents |
| :--- | :--- | :--- | :--- |
| `Vigyan-AI-32B` | Model | ~60 GB | 14 Safetensors shards (32B base). |
| `Vigyan-AI-32B-Titan-v1` | Model | ~1.5 GB | Fine-tuned 32B LoRA adapter. |
| `Shreyansh-STEM-AI-7B-Final` | Model | ~14 GB | 7B Scholar base weights. |
| `Shreyansh-STEM-AI-7B-Distilled` | Model | ~800 MB | 7B distilled LoRA adapter (Step-1500). |
| `Shreyansh-STEM-AI-2B-v3` | Model | ~3.5 GB | 2B Edge base weights. |
| `Shreyansh-STEM-AI-2B-v3-Distilled` | Model | ~400 MB | 2B distilled LoRA adapter. |
| `Vigyan-Fleet-Judge-Scorecards` | Dataset | ~50 MB | Benchmark diagnostic files, predictions (2B, 7B v1–v8), and Gemini Judge scorecards. |
| `Vigyan-Defence-STEM-CoT-Master` | Dataset | ~12 MB | 2,205 curated training problem-solution pairs. |
