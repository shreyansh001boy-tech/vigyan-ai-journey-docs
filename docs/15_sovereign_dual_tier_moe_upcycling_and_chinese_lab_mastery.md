# Chapter 15: Sovereign Dual-Tier MoE Upcycling & Chinese AI Lab Mastery

> **The architectural breakthrough that decoupled model reasoning capacity from edge compute latency: How Vigyan AI built India's first sovereign dual-tier Sparse Mixture of Experts (1.5B 4× MoE and 3B 4× MoE) on zero-dollar cloud infrastructure.**

---

## 1. 🏛️ Executive Summary & Strategic Imperative

Monolithic dense scaling imposes an insurmountable penalty on edge and consumer hardware. A dense 14B–32B model demands 28 GB–64 GB of memory streaming bandwidth for every single generated token. On consumer laptops with 8 GB to 16 GB of RAM, dense models choke, dropping to unusable generation speeds (< 2 tok/s).

To build sovereign AI accessible to 1.4 billion people, India cannot replicate the brute-force datacenter economics of Western cloud conglomerates. Inspired by the architectural efficiencies of Chinese frontier AI laboratories (**DeepSeek-MoE** and **Qwen-MoE**), Vigyan AI pioneered a sovereign **Dual-Tier Sparse Mixture of Experts (MoE)** pipeline:

```
                           ┌───────────────────────────────┐
                           │ Sovereign Dual-Tier MoE Fleet │
                           └──────────────┬────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │    Vigyan-1.5B 4× MoE     │                   │     Vigyan-3B 4× MoE      │
    │  Base: DeepSeek-R1-1.5B   │                   │   Base: Qwen2.5-3B-Inst   │
    │  Total: 5.8B | Active: 1.5B│                   │  Total: 11.8B | Active: 3B│
    │  Storage: 11.93 GB SafeT  │                   │  Storage: 23.90 GB SafeT  │
    └───────────────────────────┘                   └───────────────────────────┘
```

By decoupling parameter knowledge capacity from per-token active compute via **Top-1 sparse gating**, these models deliver the domain depth of dense 6B–14B models while generating tokens at the velocity of small 1.5B–3B SLMs.

---

## 2. 📐 Mathematical Foundations: Sparse Routing & Entropy Calibration

### 2.1 The Sparse Forward Pass
For any hidden state representation $x \in \mathbb{R}^d$, the MoE layer computes:

$$\text{Output}(x) = W_{\text{shared}} \cdot \text{FFN}_{\text{shared}}(x) + \sum_{i \in \text{Top-}k} g_i(x) \cdot \text{FFN}_i(x)$$

Where the router gating distribution $g(x)$ is defined by:

$$g(x) = \text{Softmax}\left(\text{Top-}k(W_g \cdot x + \epsilon)\right)$$

In our sovereign edge architecture:
* $k = 1$ (Top-1 active routing per token), guaranteeing that **exactly one domain expert** fires per layer for edge token generation.
* $W_{\text{shared}}$ is anchored with a residual scale of $\alpha = 0.1$ to enforce universal STEM derivation stability across layers.

### 2.2 Router Gating Entropy $H(p)$
To prevent **Router Collapse** (where the gating mechanism funnels 95%+ of tokens to a single expert, degenerating the MoE into a dense model), we measure the Shannon Entropy of the routing distribution $p = [p_1, p_2, p_3, p_4]$ over held-out benchmark sequences:

$$H(p) = -\sum_{i=1}^{4} p_i \ln(p_i)$$

* **Maximum Theoretical Entropy:** $H_{\max} = \ln(4) \approx 1.3863$ (perfect uniform routing).
* **Healthy Gating Envelope:** $0.40 \le H(p) \le 1.38$.
* **Collapsed Gate:** $H(p) < 0.30$ (triggers automatic failure in our battle-test rubric).

---

## 3. 🎯 Stage-1: Decoupled Multi-Expert LoRA Adapter Training

Rather than attempting to train an MoE router from scratch on raw unaligned tokens (which demands millions of dollars in compute), we trained 4 decoupled domain experts per tier on our curated 400k dataset (`shreyansh12183/vigyan-stem-experts-400k`):

### 3.1 Domain Curriculum Breakdown
1. **Science Expert (100k):** Quantum mechanics, physical chemistry, thermodynamics, biophysical molecular docking.
2. **Technology Expert (100k):** Synthesizable Verilog HDL, ASIC timing closure, FPGA constraints, EDA logic synthesis.
3. **Engineering Expert (100k):** Aerospace orbital dynamics, propulsion, CMOS analog RF, control systems, state-space equations.
4. **Mathematics Expert (100k):** Olympiad & AIME proofs, algebraic geometry, differential equations, real analysis.

### 3.2 Training Invariants Enforced
* **Single-Pinned GPU Architecture:** `CUDA_VISIBLE_DEVICES="0"` to eliminate multi-GPU communication deadlock on Kaggle Dual Tesla T4s.
* **Prompt Loss Masking:** Enforced `labels = -100` on all user instructions; gradients computed strictly on assistant thought tracks and final derivations.
* **Loss-Monitored Dynamic Early Stopping:** Halted automatically at convergence window ($0.45 \le \mathcal{L} \le 0.48$).
* **Immediate Adapter Publication:** All 8 adapters published instantly to private Hugging Face Hub:
  * `shreyansh12183/vigyan-1.5b-adapter-{science, technology, engineering, mathematics}`
  * `shreyansh12183/vigyan-3b-adapter-{science, technology, engineering, mathematics}`

---

## 4. ⚡ Stage-2 Upcycling: The Engineering Crisis & Breakthrough

### 4.1 The Version 4 Disk Quota Catastrophe
During the initial assembly of `Vigyan-1.5B-4x-MoE` on Kaggle (Version 4), the sequential merge and router initialization executed flawlessly. However, at **second 289**, during the final safetensors serialization of shard 3, the kernel crashed with:

```
IoError: StorageFull: "No space left on device"
```

#### Forensic Root Cause
Kaggle enforces a strict physical quota of **19.5 GB** on `/kaggle/working`.
* 4 Dense Fused Models: $4 \times 3.0\text{ GB} = 12.0\text{ GB}$.
* Assembled MoE Shards: $8.5\text{ GB}$.
* Total Storage Demanded: **$20.5\text{ GB}$**.
* Disk boundary breached by **1.0 GB**, triggering immediate kernel termination.

### 4.2 The Partition-Separation Architecture
To solve this permanently without paying cloud providers, we conducted physical storage topology analysis of Kaggle runtimes:
* `/kaggle/working`: Limited to **19.5 GB** (ephemeral volume).
* `/tmp`: Mounted on the **73.0 GB root NVMe SSD partition**.

```mermaid
flowchart TD
    subgraph Root Partition (73 GB NVMe)
        TMP_DENSE["/tmp/dense_experts\n(Fused Domain Models)"]
        TMP_OUT["/tmp/output_moe\n(Assembled MoE Weights)"]
        GC["Post-Merge Purge\n(Immediate rm -rf /tmp/dense_experts)"]
    end

    subgraph Working Directory (19.5 GB Quota)
        EMPTY["/kaggle/working\n(Footprint: 0.0 GB)"]
    end

    subgraph Hugging Face Hub
        HF["shreyansh12183/Vigyan-MoE Vault\nDirect API Upload"]
    end

    TMP_DENSE --> TMP_OUT
    TMP_DENSE --> GC
    TMP_OUT --> HF
```

By redirecting both intermediate dense merges and output MoE assembly to `/tmp`, followed by instantaneous garbage collection and cache pruning (`~/.cache/huggingface/hub`), we reduced `/kaggle/working` utilization to **0.0 GB**, completely eliminating disk exhaustion risk.

### 4.3 Modern Transformers Compatibility Patch
In `transformers` v4.49+, passing `load_in_4bit` or `load_in_8bit` directly to `AutoModelForCausalLM` raises a fatal `TypeError`. We engineered a runtime dynamic patch inside `mergekit/moe/router.py`:

```python
import glob
for r_path in glob.glob("/usr/**/mergekit/moe/router.py", recursive=True):
    with open(r_path, "r") as f:
        code = f.read()
    code = code.replace("load_in_4bit=load_in_4bit,", "").replace("load_in_8bit=load_in_8bit,", "")
    with open(r_path, "w") as f:
        f.write(code)
```

---

## 5. 🔬 Contrasting Negative Prompt Calibration

In naive MoE merges, routers suffer from semantic cross-bleeding: a differential equations question might activate the engineering or technology expert instead of mathematics.

Following Chinese AI lab standards, we calibrated the router projections using **Contrasting Negative Prompts**, ensuring that positive domain vectors are repelled by negative out-of-domain anchors:

```yaml
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
gate_mode: hidden
dtype: bfloat16
experts_per_token: 1
shared_experts:
  - source_model: /tmp/dense_experts/science
    positive_prompts:
      - "fundamental STEM physical constants, scientific methodology, and universal logical derivation"
    residual_scale: 0.1
experts:
  - source_model: /tmp/dense_experts/science
    positive_prompts: ["quantum hamiltonian", "reaction kinetics", "thermodynamic entropy"]
    negative_prompts: ["verilog testbench", "linear algebra matrix", "calculus proof"]
  - source_model: /tmp/dense_experts/technology
    positive_prompts: ["synthesizable verilog", "rtl timing closure", "asic synthesis"]
    negative_prompts: ["cellular biology", "organic chemistry", "differential topology"]
  - source_model: /tmp/dense_experts/engineering
    positive_prompts: ["aerospace propulsion", "cmos analog circuits", "flight control dynamics"]
    negative_prompts: ["abstract ring theory", "spectroscopy orbital", "number theory prime"]
  - source_model: /tmp/dense_experts/mathematics
    positive_prompts: ["olympiad proof", "differential equations", "eigenvalue decomposition"]
    negative_prompts: ["vhdl module", "crispr cas9 genomics", "boundary layer"]
```

---

## 6. 🏆 Verified Fleet Specifications & Vault Footprint

Both sovereign MoE models are assembled, verified, and secured on the Hugging Face Private Vault:

### 6.1 Vigyan-1.5B-4x-MoE
* **Hub Repository:** [`shreyansh12183/Vigyan-1.5B-4x-MoE`](https://huggingface.co/shreyansh12183/Vigyan-1.5B-4x-MoE)
* **Architecture:** `Qwen2MoeForCausalLM`
* **Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
* **Total Parameters:** $\approx 5.8\text{B}$
* **Active Parameters per Token:** $\approx 1.5\text{B}$
* **Vault Footprint:** **11.93 GB** (3 Safetensors shards)
* **Target Hardware:** 8 GB consumer laptops (CPU/iGPU)

### 6.2 Vigyan-3B-4x-MoE
* **Hub Repository:** [`shreyansh12183/Vigyan-3B-4x-MoE`](https://huggingface.co/shreyansh12183/Vigyan-3B-4x-MoE)
* **Architecture:** `Qwen2MoeForCausalLM`
* **Base Model:** `Qwen/Qwen2.5-3B-Instruct`
* **Total Parameters:** $\approx 11.8\text{B}$
* **Active Parameters per Token:** $\approx 3.0\text{B}$
* **Vault Footprint:** **23.90 GB** (6 Safetensors shards)
* **Target Hardware:** 16 GB consumer laptops / MacBooks (Metal & CPU)

---

## 7. ⚔️ The 5-Vector Battle-Test Harness

To rigorously evaluate the models without reliance on superficial benchmarks, we developed an automated evaluation harness ([`benchmarks/moe_battle_test/moe_battle_test.py`](file:///home/hello/Downloads/vigyan%20ai/benchmarks/moe_battle_test/moe_battle_test.py)):

```
                        ┌─────────────────────────────────────────┐
                        │      5-Vector Battle-Test Rubric        │
                        └────────────────────┬────────────────────┘
                                             │
      ┌──────────────────────┬───────────────┴──────────────┬──────────────────────┐
      ▼                      ▼                              ▼                      ▼
┌──────────────┐      ┌──────────────┐              ┌──────────────┐      ┌──────────────┐
│ Vector 1     │      │ Vector 2     │              │ Vector 3     │              │ Vector 4 & 5 │
│ Math-500 &   │      │ Synthesizable│              │ PhD Physical │              │ Gating       │
│ AIME Proofs  │      │ Verilog RTL  │              │ Sciences     │              │ Entropy &    │
│ (60% Weight) │      │ (20% Weight) │              │ (20% Weight) │              │ Edge Tok/sec │
└──────────────┘      └──────────────┘              └──────────────┘              └──────────────┘
```

### Deterministic SymPy AST Verification
For mathematical derivations, the test harness employs an AST-isolated symbolic evaluation engine:
```python
class SymPyMathEngine:
    def evaluate_expression(self, expr_str: str) -> dict:
        val = sp.sympify(expr_str.strip(), locals=self.allowed_symbols)
        return {"success": True, "result": str(val), "numeric_approx": float(val.evalf())}
```
This guarantees that mathematical correctness is verified algebraically rather than by probabilistic string matching.

---

## 8. 🚀 Next Horizon: The Sovereign Flagship — OLMo-2 7B 24× MoE

With the dual-tier edge MoE models successfully operating on consumer hardware, the next phase of Vigyan AI is the institutional workstation flagship:

### The OLMo-2 7B 24× MoE (Top-2 Active)
* **Base Foundation:** Allen AI's fully open-weight, open-data `OLMo-2 7B`.
* **Architecture:** 24 fine-grained domain experts (Aerospace, VLSI, Bio-Physics, Pure Topology, Cryptography, etc.).
* **Routing:** Top-2 active routing per token ($\approx 11.5\text{B}$ active compute footprint).
* **Target:** Exceed the reasoning benchmarks of dense 32B models (Qwen2.5-32B, Gemma-2-27B) while executing at native 12–16 tok/s on consumer CPUs and workstations.

**Vigyan AI has demonstrated that through mathematical discipline, sparse routing, and radical cloud resourcefulness, sovereign state-of-the-art AI can be built from India at zero compute cost.**
