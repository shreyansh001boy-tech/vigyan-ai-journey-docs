# 04. Rigorous Evaluation, Benchmarks & LLM Judges

## 1. The Sovereign STEM 10 Diagnostic Suite

To prevent overfitting on standard test sets (GSM8k, MATH, GPQA) which frequently leak into pre-training corpuses, an unreleased sovereign benchmark was designed:
* **Scope**: 10 rigorous, multi-step problems spanning Advanced Electromagnetism, Quantum Mechanics, Relativistic Doppler Shift, Rocket Propulsion, Chemical Kinetics, and Semiconductor Physics.
* **Evaluation Criteria**: Multi-criteria ground-truth rubrics requiring physical unit correctness, intermediate step validation, and numeric precision.

---

## 2. LLM-as-a-Judge Calibration

All model outputs were scored using Google's frontier models (`gemini-2.5-pro` and `gemini-3.5-flash-lite`) acting as automated sovereign judges:
* **Scoring Rubric**:
  * Physics derivation validity (40 points)
  * Calculation & unit precision (30 points)
  * Tool invocation correctness (20 points)
  * Conciseness and answer tag formatting (10 points)
* **Calibration**: Eliminates lenient heuristic regex parsing by enforcing strict rubric grading.

---

## 3. Fleet Scorecard Evolution (Vigyan-7B)

| Evaluation Iteration | Prompting & Tool Setup | Tool Use Rate | Official Calibrated Score | Key Findings |
| :--- | :--- | :---: | :---: | :--- |
| **v1 (Baseline 7B)** | Zero-shot, direct prompt | 0% | **54.3%** | Solid conceptual reasoning, but severe arithmetic errors on floating-point division. |
| **v2 (Zero-Shot Tools)**| Instruction to invoke tools | 0% | 52.0% | Model failed to trigger tools without explicit formatting examples. |
| **v6 (Few-Shot Tools)** | **2x in-context tool demos** | **100%** | **68.2%** | **Best Confirmed Score.** Flawless tool invocation, arithmetic precision resolved. |
| **v8 (Dual-RAG)** | RAG context injection | 0% | 48.0% | Prompt format bug (stop-string tokenization error) broke tool-calling intercept. |

---

## 4. GSM8k Benchmark Results (Standard Test)

* **Dataset**: Full 1,319 GSM8k evaluation questions evaluated on Kaggle Dual T4 GPUs.
* **Result**: **~61–65% accuracy** (~800+ questions solved).
* **Takeaway**: While strong for a sovereign domain fine-tune, SFT on specialized defence data slightly reduced raw grade-school arithmetic generalization compared to generic base models, proving that future improvements require GRPO/RLVR reward fine-tuning.

---

## 5. The Truth About 2B Performance

* **STEM Score**: **10.0%**.
* **Finding**: 1.5B–2B parameter models lack the representational width to execute PhD-level derivations.
* **Commercial Redirection**: 2B was immediately redirected to its true sweet spot: sub-second edge customer support, intent classification, and policy RAG, where it achieves **80%+ accuracy** at sub-50ms latency.
