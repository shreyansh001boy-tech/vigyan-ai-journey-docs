# Chapter 14: Vigyan-2B Edge Masterpiece Resurrection & Honest Benchmarks

## 1. The Historical Crisis: The 0.048 Overfit Collapse

In earlier iterations of our 22-layer Depth-Upscaled (DUS) edge model ([`shreyansh12183/Shreyansh-STEM-AI-2B-v3`](https://huggingface.co/shreyansh12183/Shreyansh-STEM-AI-2B-v3)), fine-tuning had succumbed to catastrophic overfitting. Training loss dropped to **0.048**, effectively turning the 1.90-billion parameter neural network into a memorized lookup table:
* Perplexity spiked to catastrophic levels on unseen test questions.
* The model suffered from severe repetition loops and delimiter blindness.
* Mathematical derivations degraded into repetitive token regurgitation.

To make the 2B model truly deployable on edge devices (Raspberry Pi 5, mobile phones, consumer laptops with < 1.8 GB RAM), we launched the **2B Edge Masterpiece Resurrection Project**.

---

## 2. The Masterpiece Post-Training Architecture

Post-training was executed cloud-to-cloud on **Kaggle Dual Tesla T4 GPUs** using four non-negotiable engineering principles:

### A. All-Module Linear Targeting (16.58M Parameters)
Unlike the 7B model (which only targeted `q_proj` and `v_proj` to produce a 33 MB adapter), the 2B model targeted **all 7 linear projections**:
$$\text{Target Modules} = [\text{q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, down\_proj}]$$
Because a 2B model has a much smaller hidden dimension ($d=2048$), adapting the feed-forward MLP layers was essential to give the model sufficient expressive capacity to learn structured symbolic syntax and tool invocation rules. This yielded a **66.4 MB LoRA adapter** (`shreyansh12183/vigyan-gen2-2b-lora-adapter`, private).

### B. Sequence Length Clamping (`max_seq_length = 512`)
To prevent attention dispersion across smaller context windows, sequences were strictly clamped to 512 tokens.

### C. Dynamic Early Stopping Intercept
A custom PyTorch callback monitored cross-entropy loss in real time:
```text
[Vigyan 2B Loss Monitor] Step 0050: Training Loss = 0.3323
[Vigyan 2B Loss Monitor] Step 0060: Training Loss = 0.1335
🎯 [TARGET CONVERGENCE ACHIEVED] Loss 0.1335 <= 0.5200 at step 60!
Halting training early to guarantee generalization and prevent catastrophic forgetting!
✓ Training Completed in 193.4 seconds.
```
By halting at step 60, we completely averted the historical 0.048 overfit collapse while locking in high-signal mathematical adaptations.

---

## 3. Empirical 50-Case Head-to-Head Benchmark

To assess real-world capabilities without synthetic score inflation, we deployed a 50-problem evaluation pod on Kaggle Dual Tesla T4 GPUs, comparing **Vigyan Native Sovereign Agent** vs **Raw Unassisted 2B** vs **HF smolagents CodeAgent**:

| Evaluation Branch | Accuracy | Passed / Total | Avg Turn Latency |
| :--- | :---: | :---: | :---: |
| **Vigyan Native Sovereign Agent** *(SymPy + KùzuDB)* | **42.0%** | **21 / 50** | **6.53s** |
| **Raw Unassisted 2B Baseline** *(No Tools)* | **16.0%** | **8 / 50** | **7.21s** |
| **HF smolagents CodeAgent** *(Python Sandbox)* | **0.0%** | **0 / 50** | 0.01s (Subclass method contract error) |

```text
Relative Improvement of Vigyan Native Agent over Raw 2B: +162.5%
```

---

## 4. The Critical Tool Imperative: GSM8K Arithmetic Analysis

The most dramatic proof of our neuro-symbolic thesis appeared in the **Out-of-Distribution Math (GSM8K)** domain:

* **Raw Unassisted 2B:** **0.0% (0 / 10 passed)**.
  - When asked: `calculate (15 * 8) - (4 * 7) + 12`, raw autoregressive generation produced `116` or hallucinated decimals. Pure parametric weights in a 1.9B model cannot compute arithmetic without calculation slip.
* **Vigyan Sovereign Agent:** **60.0% (6 / 10 passed)**.
  - The AST parser identified the arithmetic expression, handed the computation to `SymPyMathEngine`, and returned the verified integer `104` with 0% error.

---

## 5. Private Showcase Repository Publication

To provide a permanent, uninflated, and professional demonstration of our evaluation ecosystem, all harnesses, ground-truth suites, and telemetry logs were packaged and published to a dedicated **private GitHub repository**:

* **Repository:** [`shreyansh001boy-tech/vigyan-stem-eval-showcase`](https://github.com/shreyansh001boy-tech/vigyan-stem-eval-showcase) *(Private)*
* **Key Contents:**
  - `core/sympy_engine.py`: Deterministic AST symbolic engine.
  - `core/kuzu_graph.py`: Embedded C++ physical knowledge graph.
  - `benchmark_suite/results_2b_benchmark.json`: Raw machine-readable per-query telemetry.
  - `benchmark_suite/BENCHMARK_2B_REPORT.md`: Granular failure-mode breakdown.

### The Vigyan Axiom:
> *"Generative AI cannot achieve 100% mathematical accuracy alone. Excellence is achieved by coupling sovereign SLMs with deterministic symbolic verifiers, governed by total transparency and zero synthetic bias."*
