# 📚 Lesson 05: The Ear of JARVIS (Speech-to-Text Architecture)

This module transitions the JARVIS architecture from a text-based terminal application into a multimodal assistant capable of hearing. The primary objective was to engineer a robust, locally hosted Speech-to-Text (STT) pipeline that feeds seamlessly into the Heuristic Recommendation Engine (HRE) developed in Lesson 04.

## 🏗️ The Architectural Challenge: VRAM Scarcity & Latency
Running a Large Language Model (e.g., Ministral 3B) and an embedding model (Nomic) simultaneously already places heavy constraints on consumer VRAM. Simply dropping in a standard Voice AI model on top of this stack introduces critical risks:
1. **Out of Memory (OOM) Crashes:** Exceeding GPU VRAM allocations.
2. **Audio Transcription Latency (ATL):** If the STT engine takes too long to transcribe, the conversational UX breaks down, regardless of how fast the LLM's Time-to-First-Token (TTFT) is.

To solve this, we isolated the STT testing from the main application to benchmark models strictly on latency and accuracy before integration.

---

## 🔬 Phase 1: The STT Benchmark Suite
Rather than testing with live, variable human speech, a controlled benchmark (`02-BenchmarkVoice.py`) was constructed using a single, static audio file. This ensured mathematical fairness across three distinct STT paradigms:

1. **PocketSphinx (Legacy Baseline):** A traditional, rule-based offline recognizer requiring zero VRAM. 
2. **OpenAI Whisper (Standard AI Baseline):** The official PyTorch implementation from OpenAI (FP16/FP32).
3. **Faster-Whisper (Optimized AI):** A highly optimized reimplementation using the `CTranslate2` inference engine and INT8 quantization to drastically reduce VRAM and compute time.

### Empirical Findings: Audio Transcription Latency (ATL)
* **PocketSphinx:** Failed gracefully. While fast (~2.6s), the legacy phoneme-matching lacked contextual awareness, resulting in unusable transcriptions (e.g., transcribed "vector database" as "resemblance but").
* **OpenAI Whisper (Base):** Achieved perfect accuracy but suffered from a **1.23s - 1.70s ATL**, introducing an unnatural pause into the conversational loop.
* **Faster-Whisper (Tiny & Base):** The undisputed champion. By leveraging C++ inference and quantization, it achieved practically instant transcription (**~0.45s ATL**) with identical or near-identical accuracy to the heavier OpenAI model.

**Architectural Decision:** `faster-whisper` was selected for the production integration. The CTranslate2 backend prevents VRAM exhaustion while keeping STT latency under the 500ms threshold required for natural conversation.

---

## ⚙️ Phase 2: Voice Activity Detection (VAD) Tuning
During live integration, a critical UX issue emerged: **The "Pause Threshold" Cutoff.**
By default, standard Voice Activity Detection libraries cut the microphone stream after 0.8 seconds of silence. For deliberate speakers, or users pausing to formulate complex architectural questions, this resulted in truncated prompts.

* **The Fix:** The `pause_threshold` was explicitly tuned to `2.0` seconds in the `LocalAudioTranscriber` class, and the maximum recording limits were extended. This UX adjustment tells the system to wait for a definitive end to the user's sentence rather than aggressively cutting off the audio stream.

---

## 🧠 Phase 3: Full System Integration (HRE + STT)
The final implementation (`01-VoiceEngine.py`) successfully merged the new auditory inputs with the Dynamic Memory Manager from Lesson 04. 

The pipeline now functions autonomously:
1. **Listen:** Microphone captures audio dynamically based on ambient noise calibration.
2. **Transcribe:** `faster-whisper` transcribes the audio to text in <0.5s.
3. **Route:** The HRE intercepts the transcribed text, classifying the intent (CHAT, SUMMARY, RECALL).
4. **Retrieve:** If necessary, query deconstruction triggers a search against the Qdrant vector database.
5. **Generate:** The LLM streams the contextually aware response back to the terminal.

## 🤖 Phase 4: AI-Assisted Telemetry Evaluation
To maintain rigorous testing standards, an AI Judge script (`03-AudioRecommender.py`) was built to parse the benchmark CSV data. This automated the analysis of ATL vs. Accuracy trade-offs, providing dynamic hardware recommendations for future deployments. *Note: As architects, we caught a minor AI hallucination regarding underlying model architectures, reinforcing the principle that AI output must always be verified by human oversight.*

---

## 🚀 Next Steps: Text-to-Speech (TTS)
With JARVIS now capable of hearing, remembering, and reasoning locally, the final piece of the conversational loop is vocalization. Now that JARVIS has ears, Lesson 06 will focus on benchmarking lightweight Neural Text-to-Speech (TTS) engines to finally give him a voice without breaking our local hardware constraints.