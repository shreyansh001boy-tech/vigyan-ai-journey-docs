# Chapter 10: Colab Cloud Factory & Sovereign Vigyan AI Studio

> **Date:** September 28, 2026  
> **Status:** Production Standard  
> **Authors:** Vigyan Autonomous Agentic Engineering Team

---

## 1. The Local Resource Dilemma & Architectural Pivot

As Vigyan progressed toward deploying a sovereign, air-gapped engineering studio for researchers, a critical hardware bottleneck emerged:
- Forking modern full-featured code agent harnesses like `anomalyco/opencode` requires deep dependency trees: Bun, Go toolchains, Rust/Tauri compilers, and massive node module graphs.
- Compiling these frameworks on local developer workstations (typically constrained to 8 GB or 16 GB RAM) chokes local system resources, introduces architecture-specific dependency hell, and disrupts local productivity.
- **The Core Mandate:** Zero local RAM used for development or compilation. The software must run 100% offline and air-gapped on **Zorin OS Lite (Linux)** and **Windows 10/11**, while **Google Colab serves as the headless cloud build factory**.

---

## 2. Headless Cloud Orchestration via `google-colab-cli`

Rather than manually clicking through browser notebook cells, the engineering pipeline operates via `google-colab-cli` (`colab-cli 2.2.5+`):

1. **Automated VM Allocation:** Remote execution of `colab new -s vigyan-builder` provisions an isolated cloud environment with 12 GB RAM, 100 GB SSD, and gigabit networking.
2. **Deterministic Source Patching:** Telemetry neutralization scripts recursively eliminate third-party analytics (PostHog, Sentry, Segment, Google Analytics) from the OpenCode source.
3. **In-Process LanceDB v4 Adapter Injection:** Embeds the pre-compiled LanceDB hybrid vector engine directly into the application loop, connecting to `~/.vigyan/lancedb_v4/`.
4. **Cross-Platform Compilation:** Colab compiles static release binaries for:
   - **Linux x64 (Zorin OS Lite / Ubuntu):** Static binary package with `.desktop` launcher and AVX2 `llama-server`.
   - **Windows 10/11 x64:** Portable executable bundle with automated background daemon lifecycle management.
5. **Session Teardown:** Immediate execution of `colab stop -s vigyan-builder` prevents orphan compute unit burn.

---

## 3. Local Runtime Specifications & Memory Budget

Vigyan AI Studio operates completely disconnected from the internet once downloaded:

| Subsystem | Low-RAM Mode (Vigyan-2B) | High-Reasoning Mode (Vigyan-7B) | Invariant |
| :--- | :--- | :--- | :--- |
| **Model Weights (Q4_K_M)** | 1.173 GB | 4.472 GB | AVX2 SIMD CPU vectorization |
| **KV Cache (4k Context)** | ~180 MB | ~520 MB | Quantized block memory management |
| **LanceDB v4 Grounding** | ~120 MB | ~150 MB | Zero-copy `mmap` Arrow IPC tables |
| **Studio GUI Process** | ~70 MB | ~85 MB | Standalone native binary |
| **Total Active RAM** | **~1.54 GB** | **~5.22 GB** | **Zero swap thrashing on 8 GB / 16 GB RAM** |

---

## 4. In-App First-Run Wizard & Auto-Managed Daemon

1. **Auto-Managed Inference Daemon:**
   - Launching `Vigyan AI Studio` automatically spins up a local `llama-server` process on `localhost:8080` bound to the selected model.
   - Closing the GUI triggers a graceful termination hook, cleanly reclaiming all memory.
2. **First-Run Onboarding Wizard:**
   - On clean installs without local models, the GUI detects missing weights and presents a 1-click hardware selector:
     - **Profile 1:** Low-RAM / Fast Mode (Vigyan-2B + LanceDB v4).
     - **Profile 2:** High-Reasoning Mode (Vigyan-7B + LanceDB v4).
   - Authenticated stream download directly from private Hugging Face repositories with chunked resume support and SHA-256 integrity validation.
