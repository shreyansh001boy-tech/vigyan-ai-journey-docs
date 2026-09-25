# 08. Vigyan 32B Titan: Post-Training Surgery, Alignment & Universal Quantization

This chapter documents the end-to-end engineering lifecycle for recovering, aligning, and quantizing **Vigyan-AI-32B**, transforming it into a deployable State-of-the-Art sovereign reasoning model.

---

## 1. The Catastrophic Math Forgetting Post-Mortem

### Root Cause Analysis:
During domain fine-tuning of `Vigyan-AI-32B` on specialized defence and aerospace literature, the model suffered catastrophic forgetting in pure mathematical manifolds:
* **GSM8k Score**: 63.99%
* **MATH-500 Score**: **3.4%** (Severe reasoning collapse and parser format rejection).

### Resolution: MergeKit DARE-TIES Weight Surgery
Rather than expending thousands of dollars retraining mathematical foundations from scratch, we employ **Drop-And-Rescale (DARE-TIES)** to isolate and excise the parameter deltas responsible for math degradation while preserving specialized STEM weights:

```yaml
merge_method: dare_ties
base_model: allenai/OLMo-2-0325-32B-Instruct
models:
  - model: allenai/OLMo-2-0325-32B-Instruct
    parameters:
      weight: 1.0
  - model: shreyansh12183/Vigyan-AI-32B
    parameters:
      weight: 0.6
      density: 0.75   # Excise 25% of corrupted weight shifts
dtype: bfloat16
tokenizer_source: allenai/OLMo-2-0325-32B-Instruct
```

### Low-Level Infrastructure Invariant:
In AWS SageMaker Deep Learning Containers, native `hf_transfer` triggers runtime crashes during large multi-shard transfers. Setting:
```python
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
```
enforces robust chunked HTTP streaming over AWS network transits.

---

## 2. Dataset Fleet & Sovereign Identity Alignment

### The Tri-Blend Composition (`shreyansh12183/Vigyan-32B-TriBlend-SFT`):
To prevent both formatting collapse and conversational rigidity, the 32B model is aligned using a curated 37,044-sample tri-blend:
1. **Core Math Delimiter Base (81%)**: ~30,000 samples from `Vigyan-32B-Delimiter-Curated-Master` enforcing `<thought>...</thought>` and `<answer>\boxed{...} #### ...</answer>`.
2. **Standardized Sovereign Defence (9.5%)**: 3,544 samples from `Vigyan-Defence-STEM-CoT-Master` (Radar, SAR, Nuclear) with regex-normalized answer closures.
3. **Conversational Regularizer (9.5%)**: 3,500 samples from `shreyansh-hinglish-english-stem-500k` to maintain conversational fluidity.

### Identity Superposition Resolution:
DARE-TIES merging introduces latent identity superposition between AllenAI tokens and Vigyan AI tokens. Cross-entropy loss across all 37k rows anchors the unified sovereign identity:
> *"You are Vigyan AI, a sovereign frontier scientific and mathematical reasoning foundation model developed by Shreyansh Singh."*

---

## 3. Hybrid RAG via Vigyan STEM LanceDB v3

To eliminate factual hallucinations during complex calculations, the model interfaces with **`shreyansh12183/Vigyan-STEM-LanceDB-v3`**:
* **Scale**: 1,489,259 indexed STEM theorems, Olympiad lemmas, and physics proofs.
* **Embeddings**: `BAAI/bge-small-en-v1.5` (384 dimensions).
* **Index**: IVF-PQ (256 partitions, 48 sub-vectors) delivering sub-5ms retrieval latency.
* **Mechanism**: When calculations require complex empirical constants or Olympiad lemmas, the inference engine retrieves verified context directly into the prompt buffer.

---

## 4. Universal Edge-to-Cloud Quantization Roadmap

To enable instant deployment anywhere without multi-million-dollar clusters, the 32B model is quantized into two complementary formats:

```
[32B BF16 Base (65 GB)] ──► [DoRA SFT Alignment] ──► [Model Soup (10% Base Blend)]
                                                              │
                                ┌─────────────────────────────┴─────────────────────────────┐
                                ▼                                                           ▼
                [AWQ 4-Bit Cloud Package (~18 GB)]                         [GGUF Q4_K_M Universal Package (~19 GB)]
                • Engine: vLLM with PagedAttention                         • Engine: llama.cpp / Ollama
                • Target: 1x NVIDIA A10G (24GB) or L40S (48GB)             • Target: Apple Silicon (M1/M2/M3), 32GB RAM PC
                • Throughput: 45–65 tokens/sec                             • Throughput: 15–25 tokens/sec
                • Cold Start: <1.5 seconds on AWS Scale-to-Zero            • Offline: Runs 100% locally with zero cloud connection
```

### Benchmark Targets:
* **MATH-500**: >78.0%
* **GSM8k**: >90.0%
* **MMLU STEM**: >80.0%
