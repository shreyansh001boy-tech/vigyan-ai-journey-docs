# 05. Dual-RAG & Symbolic Tool Execution Architecture

## 1. Dual-RAG Architecture

To eliminate the two primary flaws of language models in scientific reasoning:
1. **Fact / Formula Hallucination**: Solved by querying dense LanceDB vector embeddings.
2. **Arithmetic & Floating-Point Drift**: Solved by host-level tool execution.

```
User Query: "Calculate the escape velocity from a neutron star of mass 1.4 M_sun..."
                     │
                     ▼
         [LanceDB Vector Search]
         (Retrieves G = 6.674e-11, M_sun = 1.989e30 kg, R = 12 km)
                     │
                     ▼
         [Prompt Assembly & Model Pass 1]
         Model generates reasoning:
         "<thought> v_esc = sqrt(2 * G * M / R)
          [TOOL: calculate(sqrt(2 * 6.674e-11 * 1.4 * 1.989e30 / 12000))]</thought>"
                     │
                     ▼
         [Host Python AST Execution]
         Calculates: 176,145,823 m/s (~0.587 c)
                     │
                     ▼
         [Model Pass 2 Synthesis]
         "<answer> The escape velocity is approximately 1.76 x 10^8 m/s (0.587 c). </answer>"
```

---

## 2. The Stop-String Invariant Law (Critical Post-Mortem)

When configuring stop-strings with Hugging Face `transformers.model.generate()`:

### The Bug:
```python
# BROKEN: Throws ValueError: There are one or more stop strings... but we could not locate a tokenizer
outputs = model.generate(
    **inputs,
    stop_strings=["</thought>"]  # Missing tokenizer parameter!
)
```

### The Correct Implementation:
```python
# FIXED: Tokenizer must be explicitly passed to convert stop strings into token IDs
outputs = model.generate(
    **inputs,
    tokenizer=tokenizer,
    stop_strings=["</thought>"]
)
```
Omitting `tokenizer=tokenizer` caused the model to skip the tool invocation phase entirely during v8 evaluations, causing a severe score drop.

---

## 3. Safe AST Numerical Evaluation

To prevent arbitrary code execution vulnerabilities while allowing full scientific calculator functionality, all expressions pass through a Python AST (Abstract Syntax Tree) whitelist:
* Allowed functions: `sqrt`, `sin`, `cos`, `tan`, `log`, `log10`, `exp`, `abs`, `pi`, `e`.
* Banned primitives: Built-ins, imports, system calls, file I/O.
