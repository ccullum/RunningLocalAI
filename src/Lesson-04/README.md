# 📚 Lesson 04: Local Memory & Conversation Buffers

This module transitions the JARVIS architecture from stateless, single-turn interactions to a fully stateful conversational agent. The objective was to build and benchmark custom memory management systems from scratch to understand the exact hardware constraints (VRAM and compute) of local SLMs before engineering a unified, dynamic solution.

## 🏗️ Architectural Overview
Instead of relying on black-box libraries like LangChain's `ConversationBufferMemory`, this lesson built and tested four distinct memory architectures:
1. **Sliding Window (N-Turn)**
2. **Token-Bounded Buffer**
3. **Rolling Summary (LLM Compression)**
4. **Vectorized RAG (In-Memory Qdrant)**

Ultimately, these were combined into a **Heuristic Recommendation Engine (HRE)**—an autonomous routing agent that dynamically selects the optimal memory strategy per turn.

---

## 📊 Phase 1: Short-Term Memory & The "Goldfish Effect"

The first phase tested passive, in-memory Python lists to hold conversation history. While incredibly fast (relying entirely on the SLM's internal KV cache), they exposed critical limitations.

### 1. The Jinja "Orphaned Assistant" Crash
* **The Bug:** Naively popping the oldest message from a list often leaves an Assistant message at the top of the context payload. Modern instruction-tuned SLMs (like Llama 3 and Mistral) require strict `User -> Assistant -> User` alternation and will crash if the Jinja template detects two Assistant blocks out of order.
* **The Fix:** Implemented a **Pair-Pruning** algorithm to ensure the buffer always deletes the oldest User and Assistant messages simultaneously.

### 2. Empirical Telemetry: Context Penalty & Amnesia
During a 5-turn benchmark, the **Sliding Window (2-Turn)** provided rapid responses (TTFT: ~0.40s) but suffered from the **"Goldfish Effect"**. By Turn 5, when asked to recall facts from Turn 1, the SLM confidently hallucinated incorrect answers because the context had fallen out of the active window.

---

## 🔍 Phase 2: Active Compute Memory & "Vector Dilution"

To solve long-term amnesia, the architecture moved to Active Compute Memory, utilizing background LLM tasks to process history. This revealed a strict trade-off between Compute Overhead and Semantic Accuracy.

### 1. Rolling Summary Strategy
* **Mechanism:** An LLM background thread compresses old messages into a running system prompt.
* **Result:** Semantically perfect recall, but introduced a massive **~3.0s to 10.0s compute penalty** per compression cycle, forcing the user to wait.

### 2. Vectorized RAG Strategy & The Compound Query Problem
* **Mechanism:** Every turn is embedded via Nomic and stored in an ephemeral Qdrant `:memory:` database.
* **The Bug (Vector Dilution):** When asked a compound question in Turn 5 (*"What is my name, and what database am I using?"*), the RAG pipeline failed to retrieve the database.
* **Architectural Insight:** The Nomic embedding mathematically averaged the two unrelated concepts into a single coordinate that landed exactly halfway between the target vectors, missing the `0.5` similarity threshold for both.

---

## 🎯 Phase 3: The Capstone: Dynamic Memory Manager (HRE)

To bypass the limitations of individual buffers, I engineered a **Dynamic Memory Manager**. This router intercepts the user's prompt, classifies the intent, and autonomously routes the data through the most efficient pipeline.

### The HRE Routing Logic
1. **The Fast Path (Turns 1-3):** Uses a basic Sliding Window. Bypasses all databases for **near 0.0s overhead**, giving a lightning-fast initial user experience.
2. **The Narrative Path (Standard Chat > 3 Turns):** Triggers a background Rolling Summary compression to preserve VRAM while maintaining conversational flow.
3. **The Deep Search Path (Recall Intent):** Solves the Vector Dilution bug via **Query Deconstruction**.
   * When the HRE detects a `RECALL` intent, it uses a tiny, zero-temperature LLM call to split compound questions into separate arrays (e.g., `["What is the user's name?", "What database are they using?"]`).
   * It queries Qdrant individually for each sub-question, merging the deduplicated results into the system prompt.
   * **Result:** Flawless retrieval of complex historical data with minimal combined routing and search overhead.

---

## 🧪 Phase 4: Multi-Model Benchmarking & Agentic Reliability

To prove the HRE's viability in production, I ran a fully automated hardware gauntlet testing 6 different local models (ranging from 3B to 20B parameters). 

### 1. The "Negative Constraint Trap"
During testing, an architectural flaw in Prompt Engineering was discovered. When the HRE router was given strict negative constraints (e.g., *"Do not use this for..."*, *"Use ONLY if..."*), **100% of the tested models failed the routing task.** * **The Cause:** Smaller models suffer from cognitive overload when processing strict negative boundaries. They panic and default to the safest, broadest category (in this case, `CHAT`), completely bypassing the Vector RAG database.
* **The Fix:** Reverting to a positive-reinforcement, explicitly defined prompt restored agentic reliability.

### 2. AI-Assisted Hardware Evaluation
I developed an automated scoring script using Min-Max Normalization to grade models from 0-100 based on Fast Path Speed (TTFT), Throughput (TPS), Background Compute Overhead, and Agentic Reliability.

**The Final Results:**
* **The Speed King:** `mistralai/ministral-3-3b` dominated the Fast Path with a blazing **0.435s TTFT**, making it indistinguishable from cloud-hosted latency.
* **The Unreliable Giants:** Models like `openai/gpt-oss-20b` failed basic intent classification, proving that parameter size does not equate to agentic logic. Popular models like `llama-3.2-3b-instruct` and `phi-3-mini-4k-instruct` also struggled with the strict zero-shot routing required to switch memory states.
* **The Architect's Choice:** **`mistralai/ministral-3-3b`** achieved the highest overall score, successfully executing the HRE's complex background summarization and deconstruction tasks while maintaining exceptional throughput. 

## 🛠️ Conclusion
This module proves that robust AI memory is not just about storing text; it is an active orchestration problem. By combining KV-cache windows, asynchronous LLM summarization, and deconstructed vector retrieval, the JARVIS system can maintain indefinite, accurate conversations without exhausting local hardware limits.