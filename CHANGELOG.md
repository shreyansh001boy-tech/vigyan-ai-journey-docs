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
* **Edge Redirection**: Discovered the 2B attention collapse law and redirected 2B Edge exclusively to high-velocity edge support.

---

## [Phase 5: Commercial Fleet Architecture & Freelancer Stack] - September 22, 2026
* **Product**: Built the Unified FastAPI API Gateway with FAQ caching, intent routing, and quota enforcement.
* **Zero-Cost Scaling**: Implemented the 3-account Gemini Pro key rotator (135,000 queries/month free) and Modal serverless integration.
* **Workstation Stabilization**: Resolved the 2.15 GB Chrome GPU shared memory leak and language server indexing explosion on 8GB developer hardware.
* **Open Source / Agent Skills**: Package 19 production-ready agent skills into a private GitHub repository.
