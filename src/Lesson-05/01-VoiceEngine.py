import time
import os
import speech_recognition as sr
from faster_whisper import WhisperModel
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from core.colors import Colors

# --- 1. The Local STT Engine ---
class LocalAudioTranscriber:
    def __init__(self, model_size="base.en"):
        print(f"{Colors.SYSTEM}[⚙️ Initializing STT Engine: faster-whisper ({model_size})]{Colors.RESET}")
        self.whisper = WhisperModel(model_size, device="auto", compute_type="int8")
        self.recognizer = sr.Recognizer()
        
        # Since I am a slower speaker I need more time between words
        # --- THE FIX: Extend the pause threshold for slower speakers ---
        # Default is 0.8s. Let's extend it to 2.0s so you can pause mid-sentence.
        self.recognizer.pause_threshold = 2.0 
        
        self.microphone = sr.Microphone()
        
        with self.microphone as source:
            print(f"{Colors.SYSTEM}[🎙️ Calibrating microphone for ambient noise...]{Colors.RESET}")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"{Colors.SYSTEM}[✅ Calibration complete. Ready to listen.]{Colors.RESET}")

    def listen_and_transcribe(self) -> str:
        """Listens to the mic until silence is detected, then transcribes locally."""
        with self.microphone as source:
            print(f"\n{Colors.SYSTEM}[Listening... Speak now]{Colors.RESET}")
            try:
                # Listen until the user stops speaking
                # Since I am a slower speaker the total phrase time needs to increase
                # --- THE FIX: Extend the max recording time ---
                # Changed phrase_time_limit from 15 to 30 seconds
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=30)
                print(f"{Colors.SYSTEM}[Processing Audio...]{Colors.RESET}")
                
                # Save temporarily for faster-whisper to read
                temp_filename = "temp_capture.wav"
                with open(temp_filename, "wb") as f:
                    f.write(audio.get_wav_data())

                # Transcribe
                start_time = time.perf_counter()
                segments, info = self.whisper.transcribe(temp_filename, beam_size=5)
                
                transcription = "".join([segment.text for segment in segments]).strip()
                stt_latency = time.perf_counter() - start_time
                
                os.remove(temp_filename) # Cleanup
                
                if transcription:
                    print(f"{Colors.SYSTEM}[Audio Transcription Latency (ATL): {stt_latency:.2f}s]{Colors.RESET}")
                
                return transcription

            except sr.WaitTimeoutError:
                return "" # No speech detected
            except Exception as e:
                print(f"{Colors.ERROR}Audio Error: {e}{Colors.RESET}")
                return ""


# --- 2. The HRE Memory Manager (From Lesson 04) ---
# (Keeping it streamlined here for the implementation file)
class DynamicMemoryManager:
    def __init__(self, model_id: str, client: OpenAI):
        self.client = client
        self.model_id = model_id
        self.embed_model = "text-embedding-nomic-embed-text-v1.5@q8_0"
        self.raw_history = []
        self.qdrant = QdrantClient(location=":memory:")
        self.collection = "voice_memory"
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        self.active_strategy = "Sliding Window" # Defaulting for simplicity in the basic implementation

    def add_message(self, role: str, content: str):
        self.raw_history.append({"role": role, "content": content})
        try:
            vector = self.client.embeddings.create(input=content, model=self.embed_model).data[0].embedding
            self.qdrant.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=uuid.uuid4().hex, vector=vector, payload={"role": role, "text": content})]
            )
        except Exception:
            pass

    def get_messages(self, system_prompt: str):
        # We will use the Fast Path (Sliding Window) for the basic STT test 
        # to isolate the audio latency from the HRE routing overhead.
        return [{"role": "system", "content": system_prompt}] + self.raw_history[-5:]


# --- 3. The Voice Engine ---
class VoiceChatEngine:
    def __init__(self, model_id: str):
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.model_id = model_id
        self.memory = DynamicMemoryManager(model_id, self.client)
        self.transcriber = LocalAudioTranscriber(model_size="base.en") # "base.en" is incredibly fast
        self.system_prompt = "You are Jarvis. You are communicating via a voice interface. Keep answers concise, conversational, and easy to hear."

    def start_loop(self):
        print("="*70)
        print("🎙️ JARVIS CONTINUOUS VOICE INTERFACE 🎙️")
        print("="*70)
        print("Say 'Goodbye Jarvis' or 'Exit' to stop the program.\n")

        while True:
            # 1. Listen and Transcribe
            user_input = self.transcriber.listen_and_transcribe()
            
            if not user_input:
                continue # Skip if nothing was heard

            print(f"{Colors.USER}User (Spoken): {user_input}{Colors.RESET}")

            if "goodbye" in user_input.lower() or "exit" in user_input.lower():
                print(f"{Colors.JARVIS}Jarvis: Powering down. Goodbye, sir.{Colors.RESET}")
                break

            # 2. Process through LLM
            self.memory.add_message("user", user_input)
            messages_to_send = self.memory.get_messages(self.system_prompt)

            print(f"{Colors.JARVIS}Jarvis: ", end="", flush=True)

            start_time = time.perf_counter()
            ttft = 0.0
            full_response = ""

            try:
                stream = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages_to_send,
                    temperature=0.7,
                    stream=True
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        if ttft == 0.0:
                            ttft = time.perf_counter() - start_time
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content

                self.memory.add_message("assistant", full_response)
                print(f"\n{Colors.SYSTEM}   -> [LLM TTFT: {ttft:.2f}s]{Colors.RESET}\n")

            except Exception as e:
                print(f"\n{Colors.ERROR}[Error]: {e}{Colors.RESET}")

if __name__ == "__main__":
    # Ensure Ministral or your preferred model is loaded in LM Studio!
    TARGET_MODEL = "mistralai/ministral-3-3b" 
    engine = VoiceChatEngine(model_id=TARGET_MODEL)
    engine.start_loop()