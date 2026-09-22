# Vigyan AI: The Sovereign Foundation Model & Fleet Engineering Journey

> **An exhaustive, unvarnished technical chronicle of building, training, distilling, benchmarking, and commercially operationalizing an independent sovereign AI model fleet from India.**

[![Fleet Architecture](https://img.shields.io/badge/Architecture-2B%20%7C%207B%20%7C%2032B-blue.svg)](docs/01_architecture_overview.md)
[![Verified Score](https://img.shields.io/badge/7B%20Benchmark-68.2%25%20Calibrated-success.svg)](docs/04_benchmarking_and_evaluation.md)
[![Storage](https://img.shields.io/badge/HuggingFace%20Vault-8%20Repos%20%7C%2081%2B%20GB-orange.svg)](docs/06_cloud_and_infrastructure.md)
[![Cloud Cost](https://img.shields.io/badge/AWS%20Runway-%24800%20Preserved-brightgreen.svg)](docs/06_cloud_and_infrastructure.md)
[![License](https://img.shields.io/badge/License-Proprietary%20%2F%20Confidential-red.svg)](#confidentiality)

---

## 🏛️ Executive Summary

The **Vigyan AI** initiative was established to engineer a sovereign, data-private, domain-specialized artificial intelligence fleet capable of PhD-level STEM, aerospace, defence, and high-velocity edge reasoning without reliance on foreign proprietary APIs. 

Over a multi-week sprint spanning AWS SageMaker FSDP distributed clusters, Kaggle Dual NVIDIA T4 environments, Google Colab runtimes, and local workstation hardware, this project executed:
1. **Foundation Model Surgery**: SOLAR Depth Up-Scaling (DUS) expanding OLMo-2 transformer checkpoints.
2. **Top-Down Distillation**: Transferring reasoning capacity from a 32B Titan teacher into specialized 7B Scholar and 2B Edge student models.
3. **Dual-RAG Grounding**: Coupling dense LanceDB vector embeddings with AST-verified numerical tool-calling.
4. **LLM-as-a-Judge Evaluation**: Calibrating true model performance against rigorous multi-domain rubrics using Gemini Flash models.
5. **Zero-Cost Production Gateway**: Architecting a commercial multi-cloud API gateway capable of 200,000+ monthly queries at ₹0 infrastructure expenditure.

---

## 🗺️ Documentation Blueprint

| Document | Title & Focus Area | Key Concepts |
| :--- | :--- | :--- |
| [**`01_architecture_overview.md`**](docs/01_architecture_overview.md) | **Fleet Architecture & Specifications** | Parameter tiers (2B Edge, 7B Scholar, 32B Titan), SOLAR-DUS surgery, model profiles. |
| [**`02_training_and_distillation.md`**](docs/02_training_and_distillation.md) | **Distributed Training & Distillation** | PyTorch FSDP on AWS SageMaker (`ml.g5.2xlarge`), QLoRA adapters, top-down knowledge cascade. |
| [**`03_curation_and_datasets.md`**](docs/03_curation_and_datasets.md) | **Synthetic Data & Vector Curation** | `Vigyan-Defence-STEM-CoT-Master` (2,205 samples), LanceDB vector embeddings, 50k curator daemon. |
| [**`04_benchmarking_and_evaluation.md`**](docs/04_benchmarking_and_evaluation.md) | **Rigorous Evaluation & Benchmarks** | GSM8k (1,319 problems), Sovereign STEM-10 suite, Gemini Judge rubrics, v1–v8 evolution. |
| [**`05_rag_and_tool_calling.md`**](docs/05_rag_and_tool_calling.md) | **Dual-RAG & Symbolic Tool Execution** | Embedded LanceDB retrieval, two-pass tool execution (`[TOOL: calculate]`), stop-string invariants. |
| [**`06_cloud_and_infrastructure.md`**](docs/06_cloud_and_infrastructure.md) | **Multi-Cloud Topography & Credit Control** | AWS credit preservation ($800 remaining), Kaggle headless GPU pipelines, Hugging Face 81+ GB vault. |
| [**`07_commercial_playbook.md`**](docs/07_commercial_playbook.md) | **Freelancer & Enterprise Monetization** | ₹0 cost stack, ₹5,000/mo Starter plan (10,000 queries), 100% gross margin business model. |
| [**`08_post_mortems_and_lessons.md`**](docs/08_post_mortems_and_lessons.md) | **Engineering Post-Mortems & Radical Truth** | 2B few-shot collapse law, prompt formatting fragility, teacher quality ceiling analysis. |

---

## 🧭 High-Level Fleet Topology

```mermaid
flowchart TD
    subgraph Data Layer
        D1[Synthetic Defence STEM CoT 2,205 samples]
        D2[50k Autonomous Curator Daemon]
        D3[LanceDB 2,205 Vector Store]
    end

    subgraph Model Fleet
        M32[Vigyan-32B Titan\n14 Shards ~60 GB]
        M7[Vigyan-7B Scholar\n68.2% Calibrated Score]
        M2[Vigyan-2B Edge\n1.4 GB Q4_K_M GGUF]
    end

    subgraph Infrastructure
        AWS[AWS SageMaker G5\n$800 Preserved Credits]
        KAG[Kaggle Dual T4\nZero-Cost Batch GPUs]
        HF[Hugging Face Vault\n8 Repositories | 81+ GB]
    end

    subgraph Production Gateway
        GW[FastAPI Unified Gateway]
        GEM[3x Gemini Pro Round-Robin\n135k req/mo @ ₹0]
        MOD[Modal Serverless T4\nScale-to-Zero 7B]
        ORA[Oracle Cloud ARM\n24GB Always-Free 2B Host]
    end

    D1 --> M32
    M32 -- SFT Distillation --> M7
    M32 -- SFT Distillation --> M2
    D3 --> M7
    M7 --> MOD
    M2 --> ORA
    MOD --> GW
    ORA --> GW
    GEM --> GW
```

---

## 🔒 Confidentiality

This repository contains proprietary engineering artifacts, fine-tuning methodologies, private evaluation scorecards, and architectural blueprints created for the Vigyan AI ecosystem. All rights reserved.
