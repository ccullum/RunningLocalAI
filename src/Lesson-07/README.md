# Lesson 07: The Web UI & Cognitive Memory Engine

## Overview
In this lesson, JARVIS transitions from a simple Command Line Interface to a fully interactive, browser-based Command Center using Streamlit. More importantly, the underlying memory architecture was completely rewritten from a basic Retrieval-Augmented Generation (RAG) system into a Stanford-inspired Frequency-Weighted Architecture (FWA). 

JARVIS now possesses a biological-style memory: he filters out noise, remembers facts, and strengthens the neural pathways of the memories he accesses most frequently, while allowing unused memories to slowly decay over time.

## Key Architectural Upgrades

### 1. The Streamlit Web Interface (`01-WebUI.py`)
* **Session State:** Maintains the active chat history on the screen without resetting on UI refreshes.
* **Streaming Output:** Connects to the local LM Studio server to stream tokens directly to the UI with a blinking cursor effect.
* **Control Sidebar:** Features a voice output toggle and a quick-clear button for the active session context.
* **Threaded Audio:** (WIP) Decoupled the TTS engine into a background thread to prevent UI freezing during audio playback.

### 2. The Cognitive Memory Engine (`src/core/memory.py`)
* **The Bouncer (Memory Filter):** Heuristically analyzes user input to ensure only declarative facts (e.g., "My name is Christopher") are embedded into the long-term Qdrant database, completely ignoring questions (e.g., "What is my name?") to prevent vector pollution.
* **The Heuristic Router Override:** Bypasses the LLM intent classifier for obvious recall requests (e.g., "Do you remember..."), forcing a database search and saving compute time.
* **Stanford FWA Scoring:** Re-ranks Qdrant vector search results using a multi-dimensional priority queue formula: `Final Score = Vector Similarity * (1 + (0.1 * Retrieval Count)) * Time Decay`.
* **Ghost Thread Reinforcement:** Uses non-blocking background threading (`_reinforce_memories_background`) to silently increment a memory's `retrieval_count` and update its `last_accessed` timestamp every time it is successfully recalled, organically strengthening important pathways without lagging the UI.

### 3. Surgical Database Management (`src/Utils/manage_memory.py`)
* A standalone administrative terminal tool that connects directly to the persistent `qdrant_storage`.
* Allows the developer to scroll through all saved payloads, view their UUIDs, and surgically delete specific polluted memories or collisions without wiping the entire AI's brain.

## How to Run

**1. To surgically manage JARVIS's memories:**
Ensure the Streamlit server is stopped, then run:
```bash
python src/Utils/manage_memory.py