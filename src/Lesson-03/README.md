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


***Old***
📚 Lesson 03: Local RAG Implementation & Context Analysis
This module upgrades the AI's capabilities from relying solely on pre-trained, static weights to acting as a reasoning engine over dynamic, local data. This is achieved through Retrieval-Augmented Generation (RAG).

Here are my goals for this lesson:

Build an end-to-end ETL (Extract, Transform, Load) pipeline to process raw text into mathematical vectors.

Implement a deliberate, algorithm-based chunking strategy to preserve semantic context.

Evaluate and benchmark local Vector Databases (LanceDB vs. Qdrant Local) to determine the most efficient retrieval engine for a Windows 11 desktop environment.

Orchestrate the retrieval pipeline with the existing StreamingModel from Lesson 02.

Quantify the architectural overhead (latency, compute, and hardware constraints) introduced by context injection.

The local embedding model chosen for this architecture is Nomic (text-embedding-nomic-embed-text-v1.5@q8_0), running via LM Studio.

File Breakdown:
01-DocumentIngest.py: The ETL implementation. This script introduces a custom SmartChunker that respects sentence boundaries and semantic overlap, avoiding the "orphan context" problem common in naive character-splitters. It generates embeddings via Nomic and races two local databases (LanceDB and Qdrant) to test disk I/O and indexing speed.

Architectural Decision: Chunk size was set to 400 with a 50-character overlap to balance precision for the embedding model against the finite context windows of the downstream SLMs.

02-VectorRetrieval.py: The Search Phase. Before attaching an LLM, the database retrieval must be validated. This script embeds a user query and measures retrieval latency and accuracy.

Lesson Learned 1 (API Volatility): During implementation, Qdrant updated their Python client, deprecating the .search() method in favor of .query_points(), which fundamentally changed the return type from a list to a Pydantic model. The wrapper was refactored to handle this modern API standard.

Database Verdict: On my local hardware, Qdrant Local significantly outperformed LanceDB in retrieval latency (~0.001s vs ~0.022s). Furthermore, Qdrant's Cosine Similarity Score provides a highly intuitive threshold metric (0.0 to 1.0) compared to LanceDB's standard L2 Distance. Qdrant was selected for the final architecture.

(Placeholder for your Qdrant vs LanceDB latency screenshot)

03-RAGStream.py: The Orchestration Layer. This script marries the Qdrant retrieval engine with the streaming generation models.

Architectural Decision (The Threshold Gate): A strict similarity threshold (>= 0.6) was implemented. If a user asks a question unrelated to the database (e.g., "How many continents are there?"), the system rejects the weak vector matches and gracefully falls back to a standard prompt, completely eliminating hallucination risks from irrelevant context.

Architectural Decision (Prompting): The model's temperature was dropped from 0.7 to 0.3 to prioritize strict factual adherence over conversational creativity.

(Placeholder for your Continents rejection screenshot)

04-RAGPerformanceTest.py: The quantitative capstone for this lesson. RAG is not free; it introduces network hops, embedding generation, distance calculations, and massive prompt payloads. This script benchmarks the CLEAN (zero-shot) pipeline against the JARVIS-RAG pipeline to track exact latency additions (Embed Time, Search Time, and modified TTFT).

Lesson Learned 2 (Hardware Contention): When looping through multiple SLMs automatically, LM Studio attempts to retain previous models in memory. If total VRAM limits are exceeded, the system offloads to CPU RAM, destroying TTFT metrics. I engineered a "Manual Purge Gate" into the testing loop, pausing execution between models to allow for manual VRAM flushing. This guaranteed pure, isolated hardware metrics for every test.

Lesson Learned 3 (Database File Locks): Instantiating the embedding engine repeatedly inside a loop caused file-lock collisions, as Qdrant explicitly protects its local data directory. I refactored the benchmark to use Dependency Injection, opening a single, shared connection to Qdrant at runtime and passing it into the benchmark classes.

(Placeholder for your final CSV results or the VRAM purge warning screenshot)
