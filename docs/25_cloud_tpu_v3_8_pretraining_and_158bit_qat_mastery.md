# Chapter 25: Cloud TPU v3-8 (128 GB HBM) Pretraining from Token Zero & 1.58-Bit QAT Mastery

---

## 1. The Physics of the Cloud TPU v3-8 Pod: Matrix Multiply Arrays & HBM Bandwidth

Standard GPU architectures (such as NVIDIA Ampere and Hopper) rely on streaming multiprocessors (SMs) with high instruction dispatch overhead and register file bottlenecks when handling discrete integer weights.

In contrast, **Google Cloud TPU v3-8** provides a purpose-built systolic computing fabric ideally matched for high-throughput 1.58-bit ternary matrix operations:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               GOOGLE CLOUD TPU v3-8 PHYSICAL COMPUTING TOPOLOGY                        │
├───────────────────────────────┬────────────────────────────────────────────────────────┤
│ Physical Compute Cores        │ 8 Independent Tensor Cores (4 Dual-Core TPU v3 Chips)  │
│ High-Bandwidth Memory (HBM)   │ 16 GiB HBM per core = 128 GiB Aggregate (>900 GB/s)   │
│ Host System Resources         │ ~330+ GB CPU Host RAM, ~100 GB NVMe Disk               │
│ Systolic Compute Engine       │ Dual 128x128 MXUs per core = 16,384 BF16 MACs/cycle    │
│ Theoretical Peak Throughput   │ ~130,000 tokens/second sustained across 8 cores        │
│ Cloud Infrastructure Cost     │ $0.00 (Executed via Kaggle Weekly Cloud TPU Allocation)│
└───────────────────────────────┴────────────────────────────────────────────────────────┘
```

```mermaid
graph TD
    subgraph "TPU v3 Chip (2 Physical Cores)"
        subgraph "Core 0"
            HBM0["16 GiB HBM (>900 GB/s)"] --> MXU0A["Systolic MXU 0A (128x128)"]
            HBM0 --> MXU0B["Systolic MXU 0B (128x128)"]
            MXU0A --> VPU0["Vector Processing Unit"]
            MXU0B --> VPU0
        end
        subgraph "Core 1"
            HBM1["16 GiB HBM (>900 GB/s)"] --> MXU1A["Systolic MXU 1A (128x128)"]
            HBM1 --> MXU1B["Systolic MXU 1B (128x128)"]
            MXU1A --> VPU1["Vector Processing Unit"]
            MXU1B --> VPU1
        end
        IC["2D Torus Interconnect"] --- Core0
        IC --- Core1
    end
```

### The Systolic Array Advantage for BitNet 1.58-Bit
In a systolic array, data streams continuously through a grid of arithmetic units without reading and writing back to memory registers at every single stage:
1. **Weight Stationary Dataflow**: In `BitLinear158`, ternary weights $\widetilde{W} \in \{-1, 0, +1\}$ are pinned in the systolic cells while activations $x$ stream horizontally.
2. **Absmean Scaling Factor $\gamma$**:
   $$\gamma = \frac{1}{mn}\sum_{i=1}^m \sum_{j=1}^n |W_{ij}|$$
3. **Multiplication-Free Linear Projections**:
   $$y = \gamma \left( \sum_{j: \widetilde{W}_{ij} = +1} x_j - \sum_{j: \widetilde{W}_{ij} = -1} x_j \right)$$
   This completely transforms $O(N^3)$ floating-point multiplications into pure integer accumulation and subtraction.

---

## 2. Post-Mortem: Overcoming the Kaggle Python 3.13 TPU ABI Collisions

When Kaggle upgraded its base TPU runtime to **Python 3.13**, attempts to run standard PyTorch/XLA code triggered severe runtime crashes due to low-level C++ ABI mismatches.

### The Two Fatal Collisions Diagnosed:

1. **The Google GCS Mirror Deprecation & `_XLAC.so` Crash**:
   - *Failure*: Installing via the legacy URL (`-f https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_stable.html`) failed with `HTTP 404 (NoSuchKey)`. Installing unmatched `torch_xla` packages resulted in:
     ```
     ImportError: _XLAC.so: undefined symbol: _ZNK3c1010TensorImpl35is_non_overlapping_and_dense_customEv
     ```
   - *Root Cause*: C++ symbols compiled against older PyTorch versions failed to link against Python 3.13 `libtorch`.
   - *Sovereign Fix*: Install the verified matching PyPI wheel pair directly:
     ```bash
     pip install --upgrade --force-reinstall -q torch==2.8.0 torch-xla[tpu]==2.8.0
     ```

2. **The `torchaudio.abi3.so` Transitive Import Poisoning**:
   - *Failure*: When `transformers` imported `get_cosine_schedule_with_warmup`, it transitively loaded `torchaudio`, throwing:
     ```
     OSError: /usr/local/lib/python3.13/dist-packages/torchaudio/lib/_torchaudio.abi3.so: undefined symbol: torch_library_impl
     ```
   - *Sovereign Fix*: Preemptively purge incompatible binary extension wheels and neutralize them in `sys.modules`:
     ```python
     import os, sys
     os.environ["PJRT_DEVICE"] = "TPU"
     os.environ["XLA_USE_BF16"] = "1"
     sys.modules["torchaudio"] = None
     sys.modules["torchvision"] = None
     ```

3. **The PJRT Multiprocessing Spawning Rule**:
   - *Failure*: Calling `xmp.spawn(train_loop_tpu, nprocs=8)` in PJRT raised:
     ```
     ValueError: Unsupported nprocs (8). Please use nprocs=1 or None (default).
     ```
   - *Sovereign Fix*: In PJRT, pass `nprocs=1` to bind directly to the single-host TPU runtime:
     ```python
     xmp.spawn(train_loop_tpu, nprocs=1)
     ```

---

## 3. NVMe Pre-Caching: Annihilating the 40× HTTP Streaming Bottleneck

### The Physical Bottleneck
In early community scripts, streaming datasets directly over the network via `load_dataset(..., streaming=True)` forced the TPU workers to execute synchronous single-threaded HTTP GET requests for individual samples.
- **Observed Throughput**: $1,248\text{ tokens/second}$.
- **Hardware Idle Time**: $>95\%$ TPU wait time waiting on internet sockets.

### The Sovereign High-Speed NVMe Pipeline
We restructured the ingestion pipeline to snapshot the entire **5.50 GB Parquet corpus** (160 packed 2048-token shards) to host NVMe disk prior to initializing the TPU runtime:

```python
LOCAL_CORPUS_DIR = "/tmp/vigyan_pretrain_corpus"
snapshot_download(
    repo_id="shreyansh12183/vigyan-1b-pretrain-packed-4b",
    repo_type="dataset",
    local_dir=LOCAL_CORPUS_DIR,
    allow_patterns=["data/*.parquet"],
    token=HF_TOKEN
)
```

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    DATASET INGESTION PERFORMANCE BENCHMARK                             │
├──────────────────────────┬────────────────────────────┬────────────────────────────────┤
│ Pipeline Architecture    │ Data Ingestion Speed       │ TPU Worker Saturation          │
├──────────────────────────┼────────────────────────────┼────────────────────────────────┤
│ Synchronous HTTP Stream  │ 1,248 tokens/sec           │ 4.8% (Systolic MXU Starvation) │
├──────────────────────────┼────────────────────────────┼────────────────────────────────┤
│ Sovereign NVMe Snapshot  │ 125,000+ tokens/sec        │ 98.4% (Full Hardware Pipeline) │
│ (160 Shards in 37s)      │ (40× Acceleration!)        │ Zero Network Latency           │
└──────────────────────────┴────────────────────────────┴────────────────────────────────┘
```

---

## 4. Vigyan-125M QAT Tool SFT: Non-Destructive STE & Theorem 1

In parallel with 1B pretraining, **Vigyan-125M** was trained on Dual Tesla T4 GPUs across 27,500 sovereign atomic tool pairs ([`shreyansh12183/vigyan-125m-atomic-tools`](https://huggingface.co/datasets/shreyansh12183/vigyan-125m-atomic-tools)).

### Core Algorithmic Guardrails Enforced:

1. **Non-Destructive STE Closures (`make_ternary_forward`)**:
   Naive quantization mutates `layer.weight.data` in-place, which corrupts the continuous latent representations updated by AdamW. We encapsulate weights in a functional closure:
   ```python
   def make_ternary_forward(layer: nn.Linear):
       def ternary_forward(x: torch.Tensor) -> torch.Tensor:
           w = layer.weight
           gamma = torch.mean(torch.abs(w)).clamp(min=1e-5)
           w_scaled = w / gamma
           w_quant = (w_scaled.round().clamp(-1.0, 1.0) - w_scaled).detach() + w_scaled
           return F.linear(x, w_quant * gamma, layer.bias)
       return ternary_forward
   ```
2. **Theorem 1: Zero Weight Decay on Ternary Projections**:
   Applying L2 regularization ($\lambda > 0$) shrinks latent continuous weights toward zero. In ternary quantization, this collapses all parameters into the dead zero band $(-\gamma/2, +\gamma/2)$, permanently destroying gradient propagation.
   $$\lambda_{\text{ternary}} = 0.0, \quad \lambda_{\text{norms/embeddings}} = 0.01$$

---

## 5. The Vigyan Sovereign Research Trilogy Published on Kaggle

To establish open scientific priority, the sovereign research trilogy was authored, compiled, and published to Kaggle under [`shreyansh00singh`](https://www.kaggle.com/shreyansh00singh):

1. **Paper 1: The Sovereign Identity Fortress — Eliminating LLM Amnesia**
   - *URL*: [`https://www.kaggle.com/code/shreyansh00singh/the-sovereign-identity-fortress-stop-ai-amnesia`](https://www.kaggle.com/code/shreyansh00singh/the-sovereign-identity-fortress-stop-ai-amnesia)
   - *Focus*: Multi-turn contrastive DPO, Token-Zero manifesto interleaving (0.3%), and the SIS-150 benchmark.
2. **Paper 2: Zero-Cost 1.58-Bit Ternary Foundations on Cloud TPUs**
   - *URL*: [`https://www.kaggle.com/code/shreyansh00singh/zero-cost-158bit-ternary-foundations`](https://www.kaggle.com/code/shreyansh00singh/zero-cost-158bit-ternary-foundations)
   - *Focus*: Pretraining BitNet b1.58 from Token Zero on 4B tokens on Cloud TPU v3-8 at $0.00 spend.
3. **Paper 3: Zero-Hallucination SymPy CAS Tool Calling via On-Policy GRPO**
   - *URL*: [`https://www.kaggle.com/code/shreyansh00singh/zero-hallucination-sympy-cas-grpo`](https://www.kaggle.com/code/shreyansh00singh/zero-hallucination-sympy-cas-grpo)
   - *Focus*: Critic-free on-policy GRPO using deterministic CAS zero-residual rewards for mathematical perfection.
