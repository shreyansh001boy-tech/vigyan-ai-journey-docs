# Vigyan AI: The Sovereign Foundation Model & Fleet Engineering Journey

> **An exhaustive, unvarnished technical chronicle of building, training, distilling, benchmarking, and serving an independent sovereign AI model fleet from India.**

[![Fleet Architecture](https://img.shields.io/badge/Architecture-1.5B%20MoE%20%7C%203B%20MoE%20%7C%207B%20%7C%2032B-blue.svg)](docs/01_architecture_overview.md)
[![Verified Score](https://img.shields.io/badge/7B%20Benchmark-68.2%25%20Calibrated-success.svg)](docs/04_benchmarking_and_evaluation.md)
[![Storage](https://img.shields.io/badge/HuggingFace%20Vault-10%20Repos%20%7C%20117%2B%20GB-orange.svg)](docs/06_cloud_and_infrastructure.md)
[![License](https://img.shields.io/badge/License-Proprietary%20%2F%20Confidential-red.svg)](#confidentiality)

---

## 🏛️ Executive Summary

The **Vigyan AI** initiative was established to engineer a sovereign, data-private, domain-specialized artificial intelligence fleet capable of PhD-level STEM, aerospace, defence, and high-velocity edge reasoning without reliance on foreign proprietary APIs. 

Over a multi-week sprint spanning AWS SageMaker FSDP distributed clusters, Kaggle Dual NVIDIA T4 environments, Google Colab runtimes, and local workstation hardware, this project executed:
1. **Foundation Model Surgery**: SOLAR Depth Up-Scaling (DUS) expanding OLMo-2 transformer checkpoints.
2. **Top-Down Distillation**: Transferring reasoning capacity from a 32B Titan teacher into specialized 7B Scholar and 2B Edge student models.
3. **Dual-RAG Grounding**: Coupling dense LanceDB vector embeddings with AST-verified numerical tool-calling.
4. **LLM-as-a-Judge Evaluation**: Calibrating true model performance against rigorous multi-domain rubrics using Gemini Flash models.
5. **Unified Serving Gateway**: Architecting a high-throughput API gateway combining local caching, 2B CPU inference, and 7B vLLM serving.

---

## 🗺️ Documentation Blueprint

| Document | Title & Focus Area | Key Concepts |
| :--- | :--- | :--- |
| [**`01_architecture_overview.md`**](docs/01_architecture_overview.md) | **Fleet Architecture & Specifications** | Parameter tiers (2B Edge, 7B Scholar, 32B Titan), SOLAR-DUS surgery, model profiles. |
| [**`02_training_and_distillation.md`**](docs/02_training_and_distillation.md) | **Distributed Training & Distillation** | PyTorch FSDP on AWS SageMaker (`ml.g5.2xlarge`), QLoRA adapters, top-down knowledge cascade. |
| [**`03_curation_and_datasets.md`**](docs/03_curation_and_datasets.md) | **Synthetic Data & Vector Curation** | `Vigyan-Defence-STEM-CoT-Master` (2,205 samples), LanceDB vector embeddings, 50k curator daemon. |
| [**`04_benchmarking_and_evaluation.md`**](docs/04_benchmarking_and_evaluation.md) | **Rigorous Evaluation & Benchmarks** | GSM8k (1,319 problems), Sovereign STEM-10 suite, Gemini Judge rubrics, v1–v8 evolution. |
| [**`05_rag_and_tool_calling.md`**](docs/05_rag_and_tool_calling.md) | **Dual-RAG & Symbolic Tool Execution** | Embedded LanceDB retrieval, two-pass tool execution (`[TOOL: calculate]`), stop-string invariants. |
| [**`06_cloud_and_infrastructure.md`**](docs/06_cloud_and_infrastructure.md) | **Cloud Topography & Vaults** | Multi-GPU training orchestration, Kaggle headless GPU pipelines, Hugging Face 81+ GB vault. |
| [**`07_post_mortems_and_lessons.md`**](docs/07_post_mortems_and_lessons.md) | **Engineering Post-Mortems & Radical Truth** | 2B few-shot collapse law, prompt formatting fragility, teacher quality ceiling analysis. |
| [**`08_titan_32b_post_training_and_quantization.md`**](docs/08_titan_32b_post_training_and_quantization.md) | **32B Titan Surgery, Alignment & Universal Quantization** | DARE-TIES surgery, Tri-Blend dataset (37k), LanceDB v3 integration, AWQ & GGUF universal quantization. |
| [**`09_sovereign_gguf_quantization_and_lancedb_v4.md`**](docs/09_sovereign_gguf_quantization_and_lancedb_v4.md) | **Sovereign GGUF Quantization & LanceDB v4 Grounding** | 100% Cloud-to-Cloud ($0.00 spend), Interleaved Disk-Purge Protocol, Vigyan-2B/7B GGUFs, LanceDB v4 hybrid indexing, GBNF agentic tools. |
| [**`10_colab_cloud_factory_and_studio_fork.md`**](docs/10_colab_cloud_factory_and_studio_fork.md) | **Colab Cloud Factory & Sovereign Vigyan AI Studio** | Zero local RAM builds via `google-colab-cli`, OpenCode white-labeling, telemetry purge, 100% offline air-gapped Zorin OS Lite & Windows GUI. |
| [**`11_vigyan_stem_alignment_and_pc_agentic_pipeline.md`**](docs/11_vigyan_stem_alignment_and_pc_agentic_pipeline.md) | **Sovereign STEM Alignment, 2B Leak Cure & PC Edge Pipeline** | Response-masked SFT on Dual Tesla T4s, 100% thought-leak cure in 2B, 100% circuit physics in 7B, local PC edge agentic pipeline via `llama.cpp` + LanceDB v4. |
| [**`12_vigyan_7b_production_tool_dpo_and_distribution.md`**](docs/12_vigyan_7b_production_tool_dpo_and_distribution.md) | **Vigyan-7B Production Tool-DPO & Multi-Channel Distribution** | 2B reasoning deprecation, 2,000-pair verified Tool-DPO alignment, lm-eval-harness academic rigor, Hugging Face Space & Ollama distribution. |
| [**`13_gen2_cloud_to_cloud_masterpiece_and_7b_agentic_eval.md`**](docs/13_gen2_cloud_to_cloud_masterpiece_and_7b_agentic_eval.md) | **Gen-2 Cloud-to-Cloud Masterpiece & 7B Agentic Evaluation** | Zero AWS spend ($0.00 compute), 10k balanced master dataset, prompt loss masking (-100), 7B dual-agent eval (84.8% sovereign vs 28.7% baseline). |
| [**`14_vigyan_2b_edge_masterpiece_resurrection_and_honest_benchmarks.md`**](docs/14_vigyan_2b_edge_masterpiece_resurrection_and_honest_benchmarks.md) | **Vigyan-2B Edge Masterpiece Resurrection & Honest Benchmarks** | Curing the 0.048 overfit collapse, all-module LoRA targeting (66 MB), dynamic early stopping (step 60), 50-problem benchmark (+162.5% gain), private showcase repo. |
| [**`15_sovereign_dual_tier_moe_upcycling_and_chinese_lab_mastery.md`**](docs/15_sovereign_dual_tier_moe_upcycling_and_chinese_lab_mastery.md) | **Sovereign Dual-Tier MoE Upcycling & Chinese Lab Mastery** | 1.5B & 3B 4× MoE models, Chinese AI lab architecture, Top-1 routing, contrasting negative prompts, Kaggle `/tmp` partition separation, 5-vector battle test, and OLMo-2 7B 24× roadmap. |
| [**`16_post_hoc_moe_failure_modes_public_releases_and_native_olmoe_master_plan.md`**](docs/16_post_hoc_moe_failure_modes_public_releases_and_native_olmoe_master_plan.md) | **Post-Hoc MoE Failure Modes, Public Adapter Releases & Native OLMoE Master Plan** | Attention-FFN desynchronization, pseudo-router attractor loops, public release of 8 adapters on HF, 70/20/10 curriculum law, native Allen AI OLMoE-1B-7B fine-tuning blueprint. |


---

## 🧭 High-Level Fleet Topology

```mermaid
flowchart TD
    subgraph Data Layer
        D1[Synthetic Defence STEM CoT 2,205 samples]
        D2[400k STEM 4-Domain Curriculum]
        D3[LanceDB 2,205 Vector Store]
    end

    subgraph Sparse MoE Fleet
        MOE1[Vigyan-1.5B 4× MoE\nTop-1 Routing | 11.9 GB]
        MOE3[Vigyan-3B 4× MoE\nTop-1 Routing | 23.9 GB]
        FLAG[Flagship: OLMo-2 7B 24× MoE\nTop-2 Active Workstation]
    end

    subgraph Dense Model Fleet
        M32[Vigyan-32B Titan\n14 Shards ~60 GB]
        M7[Vigyan-7B Scholar\n68.2% Calibrated Score]
        M2[Vigyan-2B Edge\n1.4 GB Q4_K_M GGUF]
    end

    subgraph Infrastructure
        AWS[AWS SageMaker G5 Clusters\nDistributed FSDP]
        KAG[Kaggle Dual T4 Fleet\nPartition-Separated MoE Upcyclers]
        HF[Hugging Face Vault\n10 Repositories | 117+ GB]
    end

    subgraph Serving Layer
        GW[FastAPI Unified Gateway]
        MOD[vLLM Inference Container\n7B Engine]
        ORA[llama.cpp Engine\nEdge MoE & SLMs]
    end

    D1 --> M32
    M32 -- SFT Distillation --> M7
    M32 -- SFT Distillation --> M2
    D2 --> MOE1
    D2 --> MOE3
    MOE1 --> ORA
    MOE3 --> ORA
    M7 --> MOD
    M2 --> ORA
    MOD --> GW
    ORA --> GW
```

---

## 📜 License & Sovereign Attribution

This architectural documentation and technical chronicle is published under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

- **Permitted Use:** Free for academic researchers, university coursework, engineering students, and personal non-commercial exploration.
- **Commercial Restrictions:** Any commercial redistribution, enterprise implementation, or paid SaaS integration requires explicit written authorization.
- **Founder & Chief Architect:** [Shreyansh Singh](https://github.com/shreyansh001boy-tech)
- **Organization:** Vigyan AI / [ExperimentLab.in](https://experimentlab.in)
