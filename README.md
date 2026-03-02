# 🧠 RunningLocalAI (Jarvis-Like Assistant)

My goal is to build a locally hosted, customized AI assistant that is as platform agnostic as possible. 

This project demonstrates the ability to deploy and interact with Language Models, both Large (LLM) and Small (SLM), entirely offline, ensuring complete data privacy and zero API costs. It serves as a foundation for a personalized "Jarvis-like" digital assistant.

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
   python -m venv venv
   venv\Scripts\activate
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
- [ ] Add RAG components.
- [ ] Implement Text-to-Speech (TTS) for a true "Jarvis" voice experience.
- [ ] Add conversation history and memory buffers.
- [ ] Integrate local system commands (e.g., opening apps, checking weather).

## 📬 Contact
Created by **Christopher Cullum** 
* LinkedIn: https://www.linkedin.com/in/christophercullum/
* Portfolio/Website: https://sites.google.com/yemtech.com/christophercullum-portfolio/home

*Currently open to new opportunities in AI and Software Engineering!*
