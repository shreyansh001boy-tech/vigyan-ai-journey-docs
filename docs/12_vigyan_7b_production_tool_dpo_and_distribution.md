# Chapter 12: Vigyan-7B Production Tool-DPO Alignment & Multi-Channel Distribution

## 1. Executive Summary & Strategic Architectural Pivot

Following empirical diagnostic audits of unquantized SLMs on Newton's laws and planetary physics, an essential architectural reality emerged:
- **2B Parameter Limit:** 2B models do not possess sufficient parametric capacity to store dense multi-domain physical constants and theorems without hallucinating false laws (e.g., inventing $\Delta M = m \cdot n$ for Newton's 3rd Law). Attempting to force open-ended reasoning onto 2B models produces compounding hallucinations. The 2B model is permanently relegated to fast parameter extraction, routing, and tool-invocation syntax formatting.
- **7B Production Focus:** **Vigyan-7B** (`shreyansh12183/Shreyansh-STEM-AI-7B-Final` + SFT) possesses the necessary representational capacity for complex multi-step reasoning. To turn it into an industry-grade, deployable asset, 100% of compute and alignment efforts are dedicated to 7B via **Deterministic Tool-DPO**.

---

## 2. Deterministic Tool-DPO Formulation

Standard DPO minimizes conversational sycophancy or toxicity. In STEM engineering, Direct Preference Optimization must target **hallucination elimination** and **grounded computation**:

$$\mathcal{L}_{DPO}(\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]$$

### The Contrastive Schema:
- **Winner ($y_w$):**
  - States the foundational physical theorem precisely (e.g. $\vec{F}_{net} = m \vec{a}$, $g = \frac{GM}{R^2}$, $\frac{1}{R_{eq}} = \sum \frac{1}{R_i}$).
  - Emits a clean, parseable `<tool_call>python_interpreter\n{"code": ...}\n</tool_call>` block for all calculations.
  - Concludes with the tool-verified exact numerical value and SI units.
- **Loser ($y_l$):**
  - Confuses physical principles (e.g. treating parallel resistors as series, inventing fake kinetic energy relationships for action-reaction).
  - Performs mental floating-point arithmetic with significant rounding drift.
  - Generates non-terminating repetitive problem loops.

---

## 3. Dataset Curation & Validation (`alignment/`)

- **Dataset:** `vigyan_7b_stem_tool_dpo.jsonl` (2,000 rigorous STEM pairs across Mechanics, Circuit Analysis, Planetary Gravitation, and Calculus).
- **Automated Soundness Check (`verify_dataset.py`):**
  - 2,000 / 2,000 pairs verified.
  - 1,983 pairs (99.2%) feature structured `<tool_call>` execution blocks.
  - 100% of Python code blocks AST-parsed with zero syntax errors.

---

## 4. Academic Benchmarking Protocol (`evaluators/`)

To prevent synthetic evaluation bias and regex score inflation:
- Unmodified **`lm-evaluation-harness`** runner (`run_lmeval_standard.py`).
- Standard academic tasks:
  - `mmlu_stem` (College Physics, College Mathematics, High School Physics, Electrical Engineering).
  - `gsm8k` (Standard 8-shot exact-match evaluation).
- Pure log-likelihood multi-choice and exact-match extraction.

---

## 5. Multi-Channel Distribution Infrastructure (`distribution/`)

1. **Hugging Face Model Hub:**
   - Production Model Card (`MODEL_CARD.md`) with complete architectural lineage, QLoRA recipe, and inference quickstart.
2. **Interactive Hugging Face Space (`distribution/hf_space/`):**
   - Standalone Gradio application with KaTeX math rendering and real-time Python tool execution accordions.
3. **Ollama Modelfile (`Modelfile`):**
   - Instant local developer deployment via `ollama create vigyan-7b -f Modelfile` and `ollama run vigyan-7b`.

---

## 6. Repository Vault

All source code, datasets, training recipes, and deployment artifacts are tracked in the private GitHub repository:
- **Repository:** `https://github.com/shreyansh001boy-tech/vigyan-7b-production-suite` (Verified `PRIVATE`).
