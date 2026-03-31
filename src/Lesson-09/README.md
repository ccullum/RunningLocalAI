# Lesson 09: Algorithmic Semantic Routing & Telemetry

## 🎯 Overview
In this lesson, we graduated from standard "LLM wrappers" to building a true deterministic AI pipeline. We replaced the brittle, stochastic LLM intent router with a lightning-fast **Algorithmic Semantic Router** powered by vector embeddings and Cosine Similarity. 

To prove the ROI of this architectural shift, we also introduced a global **Metrics Logger** to benchmark execution times and track Time To First Token (TTFT).

## 🐛 Real-World Integration Lessons (Traps & Fixes)

During the integration of the Semantic Router and Web UI, we encountered and resolved several classic architectural traps:

* **The Global Hook Trap (Barge-In):** Using `keyboard.add_hotkey('space')` for audio interruption creates a global OS-level hook. When testing in a Web UI, typing a space in the chat box silently triggered the backend audio kill-switch. **Lesson:** Always map hardware interrupts to non-typing keys (like `Esc`).
* **The Token Exhaustion Trap:** When switching from conversational chatting to document summarization, the local LLM appeared to "freeze" mid-sentence. It hadn't frozen; it hit the `LLM_CHAT_MAX_TOKENS = 300` limit. **Lesson:** Document ingestion requires a significantly larger lung capacity (1024+ tokens) configured in the Control Room.
* **The "Ghost in the Machine" (Legacy Overrides):** Our new telemetry timer initially failed to trigger because a legacy "Heuristic Override" (hardcoded keyword checks) was intercepting the query first. **Lesson:** When migrating to a dynamic, math-based system, you must ruthlessly prune the old "training wheels" (hardcoded `if/else` statements), or your new engine will never actually receive the traffic.

## ✨ Key Features & Upgrades

### 1. Deterministic Semantic Router (`semantic_router.py`)
* **Vector Math vs. LLM Guesswork:** Instead of asking an 8B parameter model to "guess" the user's intent, we now embed the user's query into a 768-dimensional vector and calculate the mathematical distance against pre-computed "Anchor Sentences."
* **100% Accuracy:** Eliminated the need for fragile, hardcoded "Heuristic Overrides" (keyword checks). The math naturally understands that *"What did the document say?"* is a `RECALL` command.
* **Faster Execution:** Bypassed the generative LLM step for routing, reducing local routing time by ~25% and saving valuable context tokens.

### 2. Global Telemetry (`utils/metrics.py`)
* **High-Resolution Benchmarking:** Implemented a Python decorator using `time.perf_counter()` to measure the exact execution time of our functions.
* **A/B Testing Switch:** Built a non-destructive feature flag (`USE_SEMANTIC_ROUTER`) to seamlessly toggle between the old LLM router and the new Math router for baseline comparisons.

### 3. The "Control Room" Perfection (`config.py`)
* **Magic Number Extraction:** Moved all VAD tuning parameters, LLM temperatures, and max tokens into the centralized config.
* **Meta-Prompt Templates:** Extracted all structural and system prompts, utilizing Python's `.format()` method to keep `memory.py` purely focused on logic and flow control.

## 📂 Updated Directory Structure

```text
src/
├── core/
│   ├── brain.py             # LLM connections (stream/background)
│   ├── colors.py            # Terminal output styling
│   ├── config.py            # The Control Room (Anchors, Tuning, Flags)
│   ├── ear.py               # VAD and local transcription (Faster-Whisper)
│   ├── memory.py            # Qdrant VDB, RAG, and Routing Dispatcher
│   ├── mouth.py             # Local TTS (Piper)
│   ├── parser.py            # Multimodal Document ingestion
│   └── semantic_router.py   # Vector Math Intent Routing
├── Lesson-09/
│   ├── 01-WebUI.py          # The Streamlit Orchestrator
│   └── README.md            # This file
└── utils/
    └── metrics.py           # Global Telemetry and Benchmarking
```

## 🚀 How to Run

1. Ensure your local LLM server (e.g., LM Studio) is running and hosting both your generation model and your embedding model (Nomic).
2. Open your terminal and navigate to the `Lesson-09` directory.
3. Execute the Streamlit run command:

```bash
cd src/Lesson-09
streamlit run 01-WebUI.py
```

## 📊 A/B Test Results (Local Hardware Baseline)

During our baseline testing on local hardware, the Semantic Router proved its superiority:
* **Old LLM Router:** ~2.79 seconds (Failed to route document query, defaulted to CHAT).
* **New Math Router:** ~2.08 seconds (100% accurate classification, successfully triggered Vector RAG). 

By trusting the math, JARVIS is now significantly more reliable, context-aware, and computationally efficient.