# 🧠 Running Local AI (Jarvis-Like Assistant)

My goal is to build a locally hosted, platform agnostic, and customized AI/Automation assistant.

This project demonstrates the ability to deploy and interact with Language Models, both Large (LLM) and Small (SLM), entirely offline, ensuring complete data privacy and zero API costs. It serves as a foundation for a personalized "Jarvis-like" digital assistant.

I intend to break out the development of this project based on Lesson's that anyone can follow.  In each SRC directory, I will document the purpose of each file, the lessons learned, and results of testing throughout this features implementation.  Each Lesson builds on knowledge gained from the previous Lesson though each can be run individually.  Please make sure to check out the Lesson folders, there is a lot to see that will help others understand why certain methods or processes were implemented.
* [Lesson-01: Establishing Local Inference](https://github.com/ccullum/RunningLocalAI/tree/main/src/Lesson-01)
* [Lesson-02: Streaming Responses & Analysis](https://github.com/ccullum/RunningLocalAI/tree/main/src/Lesson-02)
* [Lesson-03: Local RAG Implementation & Performance Analysis](https://github.com/ccullum/RunningLocalAI/tree/main/src/Lesson-03)
* [Lesson-04: Local Memory & Conversation Buffers](https://github.com/ccullum/RunningLocalAI/tree/main/src/Lesson-04)
* [Lesson-05: Adding hearing to the AI and testing/comparing STT engines](https://github.com/ccullum/RunningLocalAI/tree/main/src/Lesson-05)
* [Lesson-06: Adding speech and refactored into a completely decoupled, asynchronous, and scalable Python package](https://github.com/ccullum/RunningLocalAI/tree/main/src/Lesson-06)
* [Lesson-07: The Web UI & Cognitive Memory Engine](https://github.com/ccullum/RunningLocalAI/tree/main/src/Lesson-07)
* [Lesson-08: Multimodal Document Parsing & OCR Pipeline](https://github.com/ccullum/RunningLocalAI/tree/main/src/Lesson-08)
* [Lesson 09: Algorithmic Semantic Routing & Telemetry](https://github.com/ccullum/RunningLocalAI/tree/main/src/Lesson-09)
* [Lesson 10: Headless CI/CD Benchmarking & Automated Hardware Telemetry](https://github.com/ccullum/RunningLocalAI/tree/main/src/Lesson-10)

## 🚀 Features
* **100% Local Inference:** All processing is done locally via LM Studio, ensuring no data leaves the machine.
* **Custom Python Backend:** A scalable Python architecture designed to communicate with local model APIs.
* **Privacy-First:** Secure handling of prompts and responses without relying on external cloud providers.
* **Extensible Framework:** Built to easily accommodate future integrations like Voice-to-Text (STT), Text-to-Speech (TTS), and system automation.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **AI Model Hosting:** LM Studio (compatible with GGUF models like Llama 3, Mistral, etc.)
* **Environment:** Windows 11 / VS Code
* **Version Control:** Git & GitHub

## 📋 Prerequisites
To run this project locally, you will need:
1. [Python](https://www.python.org/downloads/) installed on your system.
2. [LM Studio](https://lmstudio.ai/) installed and running.
3. A downloaded LLM of your choice (in `.gguf` format) loaded into LM Studio.

## ⚙️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ccullum/RunningLocalAI.git](https://github.com/ccullum/RunningLocalAI.git)
   cd RunningLocalAI
    ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Local Server:**
   * Open LM Studio.
   * Load your preferred model.
   * Start the local inference server (typically running on `http://localhost:1234/v1`).

5. **Run the AI:**
   *(Note: Each feature is built out into different lessons. The lessons will build upon each other and the current highest Lesson is the AI at this point).*

## 🔮 Future Roadmap

**Completed Milestones**
- [x] Add RAG components (Vector Databases & Embeddings).
- [x] Add conversation history, memory buffers, and cognitive filtering.
- [x] Integrate Speech-to-Text (STT) for fast local voice transcription.
- [x] Implement Text-to-Speech (TTS) for a true "Jarvis" voice experience.
- [x] Build Multimodal Document Parsing and OCR pipelines.
- [x] Implement deterministic Algorithmic Semantic Routing and Telemetry.

**Upcoming Features**
- [ ] Integrate local system commands (e.g., opening apps, checking system metrics, managing files).
- [ ] Add Agentic Automation / Tool Use (allowing the LLM to trigger local Python scripts dynamically).
- [ ] Introduce Computer Vision / Screen analysis for contextual desktop awareness.
- [ ] Connect to local Smart Home ecosystems (e.g., Home Assistant API) for offline environmental control.
- [ ] Implement proactive background tasks and asynchronous agent behaviors.

## 📬 Contact
Created by **Christopher Cullum** 
* LinkedIn: https://www.linkedin.com/in/christophercullum/
* Substack: https://stableintelligence.substack.com/
* Portfolio/Website: https://sites.google.com/yemtech.com/christophercullum-portfolio/home

*Currently open to new opportunities in AI and Software Engineering!*
