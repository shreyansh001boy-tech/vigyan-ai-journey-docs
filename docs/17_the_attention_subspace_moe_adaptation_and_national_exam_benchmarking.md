# Chapter 17: Phase 12 — The Attention-Subspace Adaptation Theorem & Sovereign National Exam Benchmarks

## 1. The Low-Level Grouped-GEMM Barrier
During the fine-tuning of `allenai/OLMoE-1B-7B-0924-Instruct` on Kaggle Dual Tesla T4 GPUs, attempts to adapt the 64 expert feed-forward MLPs directly exposed fundamental hardware and framework limitations:

1. **3D Grouped-GEMM Representation:** The 64 experts per layer are packed into unified 3D grouped tensors (`[64, d_in, d_out]`). Standard PEFT cannot slice into grouped 3D GEMM without falling back to `ParamWrapper`, which fails during backward passes through fused CUDA kernels.
2. **BitsAndBytes Quantization Blindness:** BitsAndBytes NF4 only quantizes 2D `nn.Linear` layers, silently leaving 3D expert tensors in unquantized 16-bit float (~14 GB), triggering activation Out-Of-Memory (OOM) during forward execution.
3. **Cross-GPU Address Desynchronization:** Manual tensor partitioning without Accelerate's native dispatch hooks triggers `cudaErrorIllegalAddress` over PCIe bus transfers.

---

## 2. The Attention-Subspace Adaptation Theorem
Instead of modifying the 64 expert MLPs:
- **100% Freeze the MoE Backbone:** Both the router gate ($W_g$) and all 64 expert feed-forward projections remain strictly frozen (`requires_grad = False`).
- **Adapt the Multi-Head Self-Attention Subspace:** Target exclusively the standard 2D attention projections:
  `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]` with $r=32, \alpha=64$.

### Mathematical Principle:
In a transformer, the self-attention subspace dictates *how tokens attend to one another and how prompt representations are contextualized*. By steering self-attention, the model's sovereign persona (**Shreyansh Singh**), conversational turn-taking, and problem-solving instructions are injected into the hidden states *before* they pass through the pre-trained, frozen 64-expert scientific knowledge bank.

---

## 3. Standalone Sovereign Master Curriculum v1
We decoupled data curation from model training, publishing a standalone, verified 50,000-example master dataset to Hugging Face Hub:
- **Repository:** [`shreyansh12183/vigyan-olmoe-master-curriculum-v1`](https://huggingface.co/datasets/shreyansh12183/vigyan-olmoe-master-curriculum-v1)
- **Composition:**
  - 20,000 English STEM Deep Reasoning (`vigyan-stem-experts-400k`)
  - 15,000 High-Quality Bilingual Hinglish STEM (`shreyansh-hinglish-english-stem-500k`)
  - 7,000 Conversational Alignment (`vigyan-gen2-7b-alignment-dataset`)
  - 5,000 Foundation Pretraining Replay (`fineweb-edu`)
  - 3,000 Sovereign Identity Grounding (Attributed to Shreyansh Singh)
  - 1,000 Holdout Benchmark Validation Examples

---

## 4. Real Indian National Exam Benchmark Suite
Constructed standardized evaluation harness covering actual Indian competitive examinations:
- **JEE Mains:** Physics, Chemistry, and Mathematics actual questions.
- **CBSE Class 12 Boards:** Physics, Chemistry, and Mathematics derivations.
- **CBSE Class 10 Boards:** Science optics and Quadratic algebra.
- **Industry Standards:** GSM8K and Olympiad MATH.
- **SymPy AST ChiefJudge:** Deterministic evaluation supporting **0-Shot Direct** and **5-Shot Chain-of-Thought (CoT)** modes.
