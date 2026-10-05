# Changelog & Journey Timeline

All notable milestones and technical evolutions of the Vigyan AI journey are recorded here.

---

## [Phase 1: Foundation Surgery & CPT] - Late August 2026
* **Milestone**: Initiated continual pre-training on OLMo-2 base checkpoints.
* **Architecture**: Explored SOLAR Depth Up-Scaling (DUS) to expand parameter layers.
* **Infrastructure**: Provisioned AWS SageMaker execution roles and S3 bucket staging.

---

## [Phase 2: Distillation Waterfall & CoT Master] - Early September 2026
* **Milestone**: Curated `Vigyan-Defence-STEM-CoT-Master` with 2,205 verified physics, aerospace, and VLSI problems.
* **Training**: Launched SageMaker BYOS PyTorch FSDP jobs (`ml.g5.2xlarge`) distilling 32B Titan into 7B Scholar and 2B Edge.
* **Evaluation**: Evaluated 7B on full 1,319 GSM8k dataset on Kaggle Dual T4 GPUs (~61–65% accuracy).

---

## [Phase 3: The Sovereign STEM Benchmark & Tool Calling] - Mid September 2026
* **Milestone**: Authored the unreleased 10-problem Sovereign STEM Diagnostic Suite.
* **Breakthrough**: Solved the floating-point arithmetic bottleneck by implementing multi-pass tool calling (`[TOOL: calculate]`).
* **Achievement**: Vigyan-7B achieved **68.2% calibrated score** graded by Gemini frontier judges (100% tool invocation rate).
* **Storage**: Uploaded and secured all 81+ GB of models, adapters, and scorecards to Hugging Face Vault.

---

## [Phase 4: Dual-RAG & Vector Grounding] - September 21–22, 2026
* **Milestone**: Indexed 2,205 STEM vectors into an embedded LanceDB database (`all-MiniLM-L6-v2`).
* **Post-Mortem**: Diagnosed the stop-string `tokenizer=tokenizer` bug that caused the v8 evaluation regression.
* **Edge Redirection**: Discovered the 2B attention collapse law and redirected 2B Edge exclusively to high-velocity edge structured routing.

---

## [Phase 5: Unified Serving Architecture & Workstation Stabilization] - September 22, 2026
* **Architecture**: Built the Unified FastAPI Serving Gateway combining local caching, 2B CPU inference, and 7B vLLM serving.
* **Workstation Stabilization**: Resolved the 2.15 GB Chrome GPU shared memory leak and language server indexing explosion on 8GB developer hardware.
* **Open Source / Agent Skills**: Packaged 16 production-ready technical agent skills into a private GitHub repository.

---

## [Phase 6: Titan 32B Surgery, Tri-Blend SFT & Universal Quantization] - September 25, 2026
* **Root Cause Rectification**: Diagnosed the 3.4% MATH-500 collapse as catastrophic forgetting of the mathematical manifold during specialized fine-tuning.
* **DARE-TIES Surgery**: Launched automated DARE-TIES merge on AWS SageMaker (`ml.g5.2xlarge`) splicing `allenai/OLMo-2-0325-32B-Instruct` into `Vigyan-AI-32B` to restore MATH-500 >78%.
* **Tri-Blend SFT Compilation**: Built and published `shreyansh12183/Vigyan-32B-TriBlend-SFT` (37,044 samples) standardizing delimiter compliance (`<thought>...</thought><answer>\boxed{...} #### ...</answer>`) and anchoring 100% unified sovereign identity.
* **LanceDB v3 Integration**: Upgraded the RAG backbone with `shreyansh12183/Vigyan-STEM-LanceDB-v3` (1.49M indexed STEM vectors, IVF-PQ indexed).
* **Universal Quantization Strategy**: Architected AWQ (4-bit single GPU vLLM cloud serving) and GGUF Q4_K_M (universal edge/Apple Silicon/workstation offline deployment).

---

## [Phase 7: Cloud-to-Cloud GGUF Quantization & LanceDB v4 Curation] - September 27, 2026
* **100% Cloud-to-Cloud Execution**: Quantized both `Vigyan-2B` (`1.173 GB`) and `Vigyan-7B` (`4.472 GB`) to GGUF `Q4_K_M` with zero local machine disk use and \$0.00 cloud spend.
* **Interleaved Disk-Purge Protocol**: Implemented real-time purging of raw Safetensors and intermediate FP16 GGUFs, maintaining peak container disk under 18.7 GB on Kaggle.
* **Tokenizer & XET Patching**: Injected `trust_remote_code=True` and `PreTrainedTokenizerFast` for OLMo-2 architectures; neutralized the `XetProgressReporter` crash via `HF_HUB_DISABLE_XET="1"`.
* **Private Storage Quota Recovery**: Cleared 5.22 GB of duplicate models/datasets to restore free private LFS quota; verified private batch upload.
* **LanceDB v4 Sovereign Grounding**: Streamed 62 Parquet shards from `sovereign-stem-cot-gold-v1` into in-process LanceDB v4 with IVF-PQ cosine and BM25 full-text indexing.
* **Agentic Tool Calling & Studio Roadmap**: Codified GBNF grammar constraints (99.9% syntax validity) and a two-tier anti-bad-prompt expansion engine for the OpenCode Vigyan AI Studio fork.

---

## [Phase 8: Vigyan-7B Production Tool-DPO & Multi-Channel Distribution] - October 1, 2026
* **2B Diagnostic Pivot**: Audited unquantized 2B models under zero-quantization conditions; exposed structural capacity ceiling for multi-domain physics laws (e.g. inventing $\Delta M = m \cdot n$ for Newton's 3rd Law). Permanently deprecated 2B for open-ended reasoning, concentrating 100% of reasoning compute on Vigyan-7B.
* **Deterministic Tool-DPO Pipeline**: Formulated and curated 2,000 rigorous STEM DPO pairs (100% AST-verified Python tool execution) contrasting deterministic tool derivations ($y_{win}$) against hallucinated formulas and calculation drift ($y_{lose}$).
* **Academic Benchmark Integrity**: Authored standard, unhacked `lm-evaluation-harness` runner for MMLU-STEM and GSM8K with zero regex manipulation.
* **Multi-Channel Distribution**: Released production Hugging Face Model Card, interactive Gradio Hugging Face Space with tool trace accordions, and one-click Ollama Modelfile (`shreyansh001boy-tech/vigyan-7b-production-suite`).

---

## [Phase 9: Gen-2 Cloud-to-Cloud Masterpiece Paradigm & 7B Dual-Agentic Breakthrough] - October 2–3, 2026
* **Zero AWS Spend Guarantee**: Maintained strict $0.00 compute spend constraint, conducting all heavy post-training and multi-agent evaluations on Kaggle Dual Tesla T4 GPUs.
* **10,000-Pair Master Alignment**: Curated multi-domain balanced dataset (70.5% STEM, 10.1% Conversational, 9.8% Pedagogical, 9.6% General) with strict prompt loss masking (`label = -100`) to completely eliminate thought leaks and conversational forgetting.
* **Head-to-Head 7B Agentic Evaluation**: Sovereign Agent (SymPy AST Engine + C++ KùzuDB GraphRAG) achieved **84.8%** empirical accuracy with 0.21s latency, outperforming HF smolagents CodeAgent (**74.0%** / 3.85s latency) and raw autoregressive 7B generation (**28.7%**).

---

## [Phase 10: Vigyan-2B Edge Masterpiece Resurrection & Unbiased Empirical Showcase] - October 3, 2026
* **Curing the 0.048 Overfit Collapse**: Resurrected the 22-layer DUS edge model via all-module LoRA targeting (all 7 linear projections, 16.58M trainable params, 66.4 MB adapter) and sequence clamping (`max_seq_length = 512`).
* **Dynamic Loss Intercept**: Dynamic early stopping halted post-training at safe convergence (step 60, loss 0.1335 in 193.4 seconds), preventing memorization loops and preserving linguistic fluidity.
* **50-Problem Empirical Benchmark**: Native Sovereign 2B Agent achieved **42.0%** overall accuracy—a **+162.5% relative gain** over the raw unassisted 2B baseline (**16.0%**).
* **The GSM8K Tool Imperative**: Demonstrated empirical necessity of symbolic tool grounding: raw 2B scored 0.0% on multi-step arithmetic due to attention calculation drift, whereas SymPy-augmented 2B jumped to 60.0%.
* **Private Showcase Repository**: Published complete unvarnished evaluation suite and telemetry to private showcase repo (`shreyansh001boy-tech/vigyan-stem-eval-showcase`) upholding radical honesty and zero synthetic 100% claims.

---

## [Phase 11: Post-Hoc MoE Upcycling Failure Modes, Public Adapter Releases & Native OLMoE Master Plan] - October 4–5, 2026
* **Post-Hoc MoE Upcycling Forensic Analysis**: Diagnosed the mathematical mechanism of attention-FFN desynchronization and pseudo-router attractor loops on stitched $\le 3\text{B}$ MoEs.
* **Public Experimental Model Release**: Released all 8 domain-specialized LoRA adapters (1.5B & 3B) and 2 assembled MoE models to Hugging Face as public research artifacts under Apache 2.0 with professional model cards and reproducibility code.
* **The 70/20/10 Dataset Curriculum Law**: Formulated the strict partition standard (70% core domain, 20% conversational alignment, 10% base distribution anchors) to permanently eliminate delimiter blindness.
* **Native MoE Standard (Allen AI OLMoE-1B-7B)**: Successfully deployed native OLMoE-1B-7B on AMD Ryzen 5 CPU locally, achieving 17–22 tokens/second with zero attractor loops.
* **Master Plan Codified**: Created definitive 5-phase master blueprint for fine-tuning Allen AI native OLMoE on Kaggle Dual Tesla T4 GPUs with frozen router invariants.
