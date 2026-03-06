# 📚 Lesson 02: Streaming Responses & Quantitative Analysis

This module upgrades the AI's core capabilities by implementing real-time streaming responses and introduces a robust, platform-agnostic benchmarking suite. 

The primary architectural goal of this module is to establish a standardized Model Wrapper to evaluate multiple Large Language Models (LLMs) dynamically, testing their performance against a deterministic ("CLEAN") persona versus an instruction-following ("JARVIS") persona.

**Standard Testing Prompt:** *What is the primary benefit of a microservices architecture?*

## 🛠️ File Breakdown & System Architecture

### 1. `01-StreamResponse.py` (Core Streaming Implementation)
Establishes the foundational logic for requesting and handling data chunks in real-time, significantly reducing perceived latency for the end user. 
* Dynamically loads the `mistralai/ministral-3-3b` model.
* Evaluates verbosity and output structure between CLEAN and JARVIS personas.

![StreamResponse-A](../../assets/images/Lesson-02/StreamResponse-A.png)

### 2. `02-StreamTest.py` (Multi-Model Benchmarking)
Scales the streaming implementation to test against six distinct models:
`google/gemma-2-9b` | `llama-3.2-3b-instruct` | `mistralai/ministral-3-3b` | `openai/gpt-oss-20b` | `phi-3-mini-4k-instruct` | `qwen2.5-7b-instruct`.

**Key Telemetry Tracked:**
* **Total Time:** Absolute latency to complete the response.
* **Time to First Token (TTFT):** Measures system responsiveness and "cold start" latency.
* **Tokens & Tokens per Second (TPS):** Evaluates model verbosity and processing engine speed (Targeting >15 TPS for real-time voice applications).
* **Mode:** Variance between persona states.

![StreamTest-A](../../assets/images/Lesson-02/StreamTest-A.png)

> **💡 Architectural Challenges Solved:**
> * **Separation of Concerns:** Abstracted telemetry tracking out of the core model wrapper and into a dedicated `TestResult` class.
> * **Defeating Prompt Caching:** Implemented nonce injection (UUIDs) on prompts to bypass KV-Cache, ensuring accurate, un-cached benchmark times for subsequent queries.
> * **Mitigating Cold Starts:** Engineered a `warm_up` method to handle initial model-load latency before executing official benchmark prompts.

### 3. `03-AnalyzeStreamTest.py` (Quantitative Performance Analysis)
Performs deep quantitative analysis by mapping telemetry data against the Total VRAM footprint and Active weights per token of each model.

**Calculated Metrics:**
* **Compute Efficiency (TPS / Active):** Identifies optimized architectures and kernel speeds.
* **Resource ROI (TPS / Total):** Measures speed gained versus VRAM sacrificed.
* **Density Score (Tokens / Total Time):** Penalizes models with high TTFT or mid-generation latency pauses.
* **Persona Disparity:** `((Density Score:Jarvis - Density Score:Clean) / Density Score:Clean) * 100` — Measures performance degradation when enforcing strict system personas.

![AnalyzeStreamTest-A](../../assets/images/Lesson-02/AnalyzeStreamTest-A.png)
![AnalyzeStreamTest-B](../../assets/images/Lesson-02/AnalyzeStreamTest-B.png)

### 4. `04-RecommendationTest.py` (Heuristic Recommendation Engine)
Applies a strict heuristic threshold to the aggregated data to programmatically recommend the optimal model for a local Jarvis implementation based on hardware constraints.

**Rejection Criteria:**
* **Instability:** Persona Disparity > 20.
* **High Latency:** Jarvis TTFT > 0.8 seconds.
* **Laggy Output:** Jarvis TPS < 20.
* **Poor VRAM ROI:** Model Total > 15 AND Jarvis TPS < 30.

![RecomendationTest-A](../../assets/images/Lesson-02/RecomendationTest-A.png)