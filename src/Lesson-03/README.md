# 📚 Lesson 03: Local RAG Implementation & Performance Analysis

This module transitions the project from a general-purpose assistant to a **Retrieval-Augmented Generation (RAG)** system. The goal was to build a local pipeline that allows JARVIS to answer questions based on a private, local knowledge base without sending data to a cloud provider.

## 🏗️ Architectural Overview
* **Embedding Model:** `nomic-embed-text-v1.5@q8_0` (Local via LM Studio)
* **Vector Database:** Qdrant Local (File-based storage)
* **Orchestration:** Python-based ETL and Query pipeline with similarity thresholding.

## 📊 Performance Metrics: The "RAG Tax"
To understand the cost of adding knowledge to an SLM, I benchmarked the end-to-end RAG pipeline against the zero-shot persona developed in Lesson 02.

### 1. The Retrieval Bottleneck
The data reveals that the actual "search" is negligible, while the "translation" of text to math (embedding) is the primary bottleneck.
* **Average Search Latency (Qdrant):** **0.0025s**
* **Average Embedding Latency (Nomic):** **2.0925s**

### 2. Lesson 02 vs. Lesson 03 (Persona vs. RAG)
The following table illustrates the impact on Time to First Token (TTFT) when moving from a standard system prompt (`JARVIS`) to one injected with local context (`JARVIS-RAG`).

| Model | Persona TTFT (L02) | RAG TTFT (L03) | Total Hit |
| :--- | :---: | :---: | :---: |
| **google/gemma-2-9b** | 1.67s | 3.04s | **+1.37s** |
| **qwen2.5-7b-instruct** | 1.07s | 2.34s | **+1.27s** |
| **mistralai/ministral-3-3b** | 0.49s | 1.00s | **+0.51s** |
| **phi-3-mini-4k-instruct** | 0.23s | 0.39s | **+0.16s** |

**Average TTFT Increase:** **1.0887s**

### 3. Throughput (TPS) Paradox
Interestingly, while RAG increased initial latency, it significantly improved **Tokens Per Second (TPS)** for smaller models.
* **Ministral-3b:** Improved from **31.97 TPS** to **47.74 TPS** (**+49%**).
* **Llama-3.2-3b:** Improved from **27.73 TPS** to **36.09 TPS** (**+30%**).

**Architectural Insight:** Providing grounded context simplifies the "probability space" the model must navigate. By narrowing the scope of the answer via RAG, the SLM can generate tokens more decisively and at higher speeds once the initial processing is complete.

## 🛠️ Lessons Learned & Edge Cases

* **API Volatility:** Qdrant's recent move to the `.query_points()` API required a refactor of the standard retrieval logic to handle Pydantic response models rather than simple lists.
* **Concurrency & Locks:** Local vector databases place a strict file-lock on their data directory. I implemented a **Dependency Injection** pattern in the benchmark script to share a single database connection across multiple test instances, preventing `RuntimeError` crashes.
* **VRAM Management:** Running an embedding model and an SLM simultaneously requires careful VRAM budgeting. I introduced a **Manual Purge Gate** in the testing pipeline to ensure that large models (like Gemma-9B) were fully ejected before loading smaller ones, preventing CPU offloading from contaminating the benchmarks.

## 🎯 Conclusion
The Qdrant-based retrieval system is exceptionally efficient, adding only **2.5ms** to the pipeline. The current user-experience bottleneck is the **~2-second delay** during query embedding. Future optimizations will focus on asynchronous embedding generation or switching to a lighter embedding architecture to bring the total RAG TTFT under the 1.0s "instant-response" threshold.
