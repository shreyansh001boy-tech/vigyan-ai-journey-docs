# 03. Synthetic Data Curation & Vector Indexing

## 1. The Core Curated Dataset

### `Vigyan-Defence-STEM-CoT-Master`
* **Volume**: 2,205 high-density Chain-of-Thought (CoT) problem-solution pairs.
* **Domain Distribution**:
  * Hypersonic Aerodynamics & Aerothermodynamics
  * Quantum Mechanics & Condensed Matter Physics
  * Semiconductor Physics, CDC FIFO Verification & VLSI
  * Aerospace Propulsion & Rocket Dynamics
  * Chemical Kinetics & Bio-Molecular Energetics
* **Format**:
  ```json
  {
    "instruction": "Derive the stagnation temperature behind a normal shock at Mach 6...",
    "input": "",
    "output": "<thought>\nStep 1: Identify freestream parameters...\n[TOOL: calculate(T0 = T1 * (1 + (gamma-1)/2 * M1^2))]\n[TOOL_RESULT: 1680.0 K]\n</thought>\n<answer>\nThe stagnation temperature is 1680 K.</answer>"
  }
  ```

---

## 2. Autonomous 50k Dataset Curator Daemon

To expand fleet capacity toward enterprise-grade robustness, an autonomous background curator was deployed on Kaggle:
* **Workflow**: Iteratively samples hard STEM textbook repositories, generates synthetic variations, evaluates mathematical correctness, and deduplicates embeddings.
* **Storage Target**: Staged for upload to Hugging Face Hub as `Vigyan-Sovereign-STEM-50k`.

---

## 3. LanceDB Vector Database Indexing

* **Storage Path**: Embedded locally at `vigyan_products/lancedb_data/cot_chains.lance`.
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
* **Index Scale**: 2,205 canonical verified formulas and derivations.
* **Retrieval Speed**: <5ms sub-millisecond similarity search using cosine distance.
