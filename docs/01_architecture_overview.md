# 01. Fleet Architecture & Foundation Specifications

## 1. Architectural Philosophy

The Vigyan AI fleet was designed under a strict **asymmetric parameter deployment principle**:
* Large frontier models (30B+) are too expensive and latency-heavy for routine real-time edge interactions.
* Tiny models (1B–2B) lack the parameter volume to execute 10-step calculus or aerospace thermal equations.
* Therefore, the fleet is split into three tightly targeted operational tiers:
  1. **Vigyan-32B Titan**: The Sovereign Teacher & Knowledge Oracle.
  2. **Vigyan-7B Scholar**: The Applied Engineering Workhorse (Tool-Calling specialist).
  3. **Vigyan-2B Edge**: The Low-Latency Edge Dialogue & Intent Routing Agent.

---

## 2. Fleet Parameter Specifications

| Model Name | Base Architecture | Shards / Size | Precision | Quantized Size | Primary Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vigyan-32B Titan** | OLMo-2 / Qwen-2.5 Hybrid | 14 shards (~60 GB) | BF16 / 8-bit | ~18 GB (4-bit) | Teacher distillation, foundational CoT reasoning, high-fidelity synthesis. |
| **Vigyan-7B Scholar** | `allenai/OLMo-2-1124-7B-Instruct` | 4 shards (~14 GB) | BF16 / 4-bit | ~4.2 GB (Q4_K_M) | Applied physics, chemistry, engineering calculations, numerical tool invocation. |
| **Vigyan-2B Edge** | `OLMo-2-1124-1.5B/2B` | 2 shards (~3.5 GB) | BF16 / 4-bit | **1.4 GB (Q4_K_M)** | High-speed edge customer support, intent classification, local offline inference. |

---

## 3. SOLAR Depth Up-Scaling (DUS) Surgery

To bridge capacity gaps without incurring millions of dollars in pre-training compute, the fleet utilized the **SOLAR Depth Up-Scaling (DUS)** methodology:
* **Mechanism**: Repeating and interleaving intermediate transformer blocks from base checkpoints while pruning redundant residual streams.
* **Seam Healing via Continual Pre-Training (CPT)**: When layers are surgically duplicated, the attention transitions between the original layer $N$ and duplicated layer $N+1$ introduce cross-entropy loss spikes. A short continual pre-training phase on high-density STEM tokens heals the seam transitions.
* **Merge Configuration**: Executed via custom passthrough recipes ensuring parameter alignment across query, key, value, and gate projection matrices.
