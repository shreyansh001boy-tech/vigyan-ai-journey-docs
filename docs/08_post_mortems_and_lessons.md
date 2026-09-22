# 08. Engineering Post-Mortems & Radical Truth

## 1. The 2B In-Context Attention Collapse Law

### Failure Mode:
During early 2B evaluation rounds, providing 2 in-context few-shot demonstration problems caused the 2B model to memorize the numbers inside the demonstrations (e.g. `300,000 J`, `600 Hz`). The model outputted these memorized values regardless of the physics question asked.

### Law:
> **Small Language Models (<2B parameters) must NEVER receive few-shot mathematical demonstrations. They require zero-shot prompt framing coupled with host-level tool offloading (SymPy).**

---

## 2. The Teacher Quality Ceiling in Knowledge Distillation

### Insight:
A student model can never exceed the reasoning fidelity of its teacher's demonstrations.
* When 32B Titan generated synthetic CoT traces with incorrect algebraic steps, 7B learned to replicate those exact derivation flaws.
* Fine-tuning students on synthetic teacher outputs without rigorous reward verification causes **"reasoning mimicry"**—the model adopts the formatting of thinking (`<thought>...`), but arrives at incorrect conclusions.

---

## 3. Workstation RAM & Shared GPU Leak Post-Mortem

### Problem:
During local testing, an 8GB RAM Windows workstation experienced severe UI freezing, 63% active SSD paging, and 94% memory consumption.

### Root Cause:
* **Chrome GPU Process (`--type=gpu-process`, PID 3320)**: Allocated **2.15 GB of shared system RAM** as video memory for the integrated AMD Radeon GPU and failed to release it when tabs were closed.
* **IDE Language Server (`language_server.exe`)**: Ballooned to **3.74 GB** attempting to recursively index 10 GB of workspace data, binary LanceDB archives, and Electron `node_modules`.

### Solution:
* Terminated the stuck Chrome GPU process (reclaimed 2.1 GB immediately).
* Configured `files.exclude` in `.vscode/settings.json` to block indexing of heavy build and vector directories.
* Purged dormant working sets using the Windows API `EmptyWorkingSet`.
