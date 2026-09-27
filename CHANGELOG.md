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


