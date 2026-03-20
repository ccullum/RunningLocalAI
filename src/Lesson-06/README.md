# Lesson 06: The Asynchronous Multimodal Engine (JARVIS V1.1)

This lesson represents a major architectural milestone. What began as a monolithic, synchronous voice assistant script has been successfully refactored into a completely decoupled, asynchronous, and scalable Python package. 

JARVIS can now listen, think, and speak simultaneously, while maintaining a strict 100% offline, local-only execution model.

## 🏗️ Key Architectural Decisions & The "Why"

### 1. TTS Engine Selection: Piper Binary > Kokoro
During benchmarking, we tested Kokoro-ONNX against Piper. While Kokoro produces excellent audio, its CPU execution latency (~2.7 to 4.8 seconds Time-To-First-Audio) created an unacceptable delay for conversational AI. Furthermore, it required a strict `numpy==1.26.4` dependency that fundamentally broke our LangChain modules.
* **The Solution:** We pivoted to the standalone **Piper C++ Binary**. By writing a custom, OS-aware Python wrapper via `subprocess`, we bypassed broken PyPI packages and achieved a conversational TTFA of **~0.75 seconds**.

### 2. Asynchronous Pipelining & Hardware Barge-in
A voice assistant that forces you to wait for a full paragraph to be generated before speaking feels unnatural. Furthermore, a system that cannot be interrupted requires hard terminal crashes (`Ctrl+C`) to stop.
* **Sentence-Level Pipelining:** The LLM runs on the main thread. The exact millisecond it generates punctuation, the sentence is pushed to a thread-safe Queue. Piper synthesizes and speaks Sentence 1 while the LLM simultaneously calculates Sentence 2.
* **Hardware Kill Switch (Barge-in):** We implemented a background worker thread using the `keyboard` library. Pressing the `Spacebar` while JARVIS is speaking instantly flips a global state flag, flushing the TTS Queue, halting the PyAudio byte-stream mid-word, and severing the LLM generation loop so the user can speak again.

### 3. Modular Encapsulation (The `core` Package)
To prepare for a web-based UI in future lessons, the 250+ line monolithic script was torn down and separated by domain logic into `src/core/`:
* `ear.py`: Handles Voice Activity Detection (VAD) and Faster-Whisper STT.
* `brain.py`: Manages the OpenAI-compatible streaming connection to LM Studio.
* `memory.py`: Houses the Heuristic Recommendation Engine (HRE) and persistent Qdrant Vector DB.
* `mouth.py`: Encapsulates the multi-threaded Queue, Piper TTS wrapper, and PyAudio streams.
* **The Benefit:** The master script (`03-ModularAsyncJarvis.py`) now contains zero threading, database, or audio logic. It serves purely as a high-level orchestrator.

### 4. Centralized Assets & Dynamic Path Anchoring
Copying 300MB of TTS binaries and Vector Databases into every new lesson folder would rapidly bloat the repository.
* **The Solution:** All persistent assets (Piper `.exe`, `.onnx` models, and the `qdrant_storage` database) were moved to a sibling `data/` directory.
* **Dynamic Anchoring:** The `core` modules use `os.path.abspath(__file__)` to dynamically locate the `data/` directory. The codebase is now immune to working-directory path bugs and can be executed from anywhere on the host machine.

### 5. 100% Local Embeddings
To strictly enforce our "No Internet Required" rule, we removed the `langchain_huggingface` dependency. The Qdrant memory module now routes all text embedding requests directly to the local LM Studio API endpoint.

---

## 📂 System Architecture

```text
RunningLocalAI/
├── data/                       <-- Centralized Assets
│   ├── piper/                  <-- Piper binary
│   ├── qdrant_storage/         <-- Persistent Vector DB
│   └── voices/                 <-- Centralized Voices
│       └── piper-lessac.onnx   <-- TTS Voice Model
│
└── src/
    ├── core/                   <-- Reusable Encapsulated Engine
    │   ├── brain.py
    │   ├── colors.py
    │   ├── ear.py
    │   ├── memory.py
    │   ├── mouth.py
    │   └── piper_wrapper.py
    │
    └── Lesson-06/              <-- Execution Scripts
        ├── 01-BenchmarkTTS.py
        ├── 02-JarvisCore.py
        └── 03-ModularAsyncJarvis.py