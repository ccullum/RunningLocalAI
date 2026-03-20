import sys
import os
import time
import wave
import uuid
import warnings
import re
import pyaudio
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings
from core.colors import Colors

# ==========================================
# THE ARCHITECT'S PATH INJECTION
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import the wrapper from the new core package!
from core.piper_wrapper import PiperTTSWrapper

# Suppress harmless warnings for clean terminal output
warnings.filterwarnings("ignore", category=UserWarning)

print(f"{Colors.SYSTEM}[System] Initializing JARVIS Core V1.0 Architecture...{Colors.RESET}")

# ==========================================
# 🧠 MODULE 1: THE BRAIN & MEMORY (LLM + QDRANT)
# ==========================================
# Connect to LM Studio
llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Initialize Local Embeddings (For routing and memory)
print(f"{Colors.SYSTEM}[System] Loading Nomic Embedding Engine...{Colors.RESET}")
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5", model_kwargs={'trust_remote_code': True})

# Initialize Persistent Qdrant Vector DB (Saves to disk!)
qdrant = QdrantClient(path="./qdrant_storage")

# Only create the collection if this is the very first time running
if not qdrant.collection_exists("jarvis_memory"):
    print(f"{Colors.SYSTEM}[System] Creating new persistent memory database...{Colors.RESET}")
    qdrant.create_collection(
        collection_name="jarvis_memory",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE) # Nomic uses 768 dimensions
    )
else:
    print(f"{Colors.SYSTEM}[System] Persistent memory loaded successfully.{Colors.RESET}")

def save_to_memory(text, role):
    """Embeds and saves conversation turns into the vector database."""
    vector = embeddings.embed_query(text)
    qdrant.upsert(
        collection_name="jarvis_memory",
        points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": text, "role": role})]
    )

def recall_memory(query, top_k=2):
    """Searches the vector database for relevant past context using Qdrant v1.10+ API."""
    vector = embeddings.embed_query(query)
    
    # Use the new query_points method instead of the deprecated search method
    response = qdrant.query_points(
        collection_name="jarvis_memory", 
        query=vector, 
        limit=top_k
    )
    
    # Extract the actual list of matches from the response object
    results = response.points
    
    if not results or results[0].score < 0.5: # Threshold filter
        return None
    
    context = "\n".join([f"{hit.payload['role'].upper()}: {hit.payload['text']}" for hit in results])
    return context

# ==========================================
# 👂 MODULE 2: THE EAR (FASTER-WHISPER)
# ==========================================
print(f"{Colors.SYSTEM}[System] Loading Faster-Whisper (tiny.en) into VRAM/CPU...{Colors.RESET}")
stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
recognizer = sr.Recognizer()

def listen_and_transcribe():
    """Captures audio from the microphone and transcribes it instantly."""
    with sr.Microphone() as source:
        print(f"\n{Colors.SYSTEM}[Listening... Speak now (or say 'Shutdown' to exit)]{Colors.RESET}")
        # Adjust for ambient noise and set a healthy pause threshold
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        recognizer.pause_threshold = 1.5 
        
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
            print(f"{Colors.SYSTEM}[Processing Audio...]{Colors.RESET}")
            
            # Save temporarily for Faster-Whisper
            temp_wav = "temp_capture.wav"
            with open(temp_wav, "wb") as f:
                f.write(audio.get_wav_data())
            
            # Transcribe
            segments, _ = stt_model.transcribe(temp_wav, beam_size=5)
            transcription = "".join([segment.text for segment in segments]).strip()
            
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return transcription

        except sr.WaitTimeoutError:
            return ""
        except Exception as e:
            print(f"{Colors.SYSTEM}[Audio Error: {e}]{Colors.RESET}")
            return ""

# ==========================================
# 🗣️ MODULE 3: THE MOUTH (PIPER TTS)
# ==========================================
print(f"{Colors.SYSTEM}[System] Loading Piper TTS Neural Engine (Binary Wrapper)...{Colors.RESET}")

# Initialize the platform-agnostic TTS Engine pointing to the data folder
piper_tts = PiperTTSWrapper(model_path="../data/piper-lessac.onnx", piper_dir="../data/piper")

def speak(text):
    """Generates and plays local TTS audio (Cross-Platform via PyAudio & Binary Wrapper)."""
    # 1. Aggressively strip markdown and symbols
    clean_text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    if not clean_text:
        return

    temp_file = "temp_piper.wav"
    try:
        # 2. Synthesize using our new OS-aware wrapper
        success, result = piper_tts.synthesize(clean_text, temp_file)
        
        if not success:
            print(f"{Colors.ERROR}[TTS Binary Error: {result}]{Colors.RESET}")
            return
            
        # 3. Play audio cross-platform using PyAudio
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 100:
            wf = wave.open(temp_file, 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
                
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
        else:
            print(f"{Colors.ERROR}[TTS Warning: Generated file is empty.]{Colors.RESET}")
            
    except Exception as e:
        print(f"{Colors.SYSTEM}[TTS Playback Error: {e}]{Colors.RESET}")
    finally:
        if os.path.exists(temp_file): 
            os.remove(temp_file)

# ==========================================
# 🔀 MODULE 4: THE MASTER ROUTER (HRE)
# ==========================================
def generate_response(user_input):
    """Acts as the HRE, pulling memory if needed, and generating the LLM response."""
    
    # Check Memory Vector DB
    memory_context = recall_memory(user_input)
    
    system_prompt = "You are JARVIS, a highly intelligent and concise AI software architecture assistant."
    if memory_context:
        print(f"{Colors.ROUTER}[HRE: Relevant past conversation recalled from Vector DB]{Colors.RESET}")
        system_prompt += f"\n\nRecall this past context if relevant to the user's prompt:\n{memory_context}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    print(f"{Colors.JARVIS}JARVIS: {Colors.RESET}", end="", flush=True)
    
    # We buffer the stream slightly to create clean sentences for the TTS engine
    response_text = ""
    try:
        stream = llm_client.chat.completions.create(
            model="local-model",
            messages=messages,
            temperature=0.3,
            max_tokens=200, # Keep it concise for voice
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                response_text += token
                
        print() # Newline after stream finishes
        return response_text

    except Exception as e:
        error_msg = "I am having trouble connecting to my language model."
        print(error_msg)
        return error_msg

# ==========================================
# 🚀 MAIN APPLICATION LOOP
# ==========================================
def main():
    startup_msg = "All core architecture modules are loaded. I am ready."
    print(f"\n{Colors.JARVIS}JARVIS: {startup_msg}{Colors.RESET}")
    speak(startup_msg)
    
    while True:
        # 1. Listen
        user_input = listen_and_transcribe()
        
        if not user_input:
            continue
            
        print(f"\n{Colors.USER}User: {user_input}{Colors.RESET}")
        
        if "shutdown" in user_input.lower() or "shut down" in user_input.lower() or "exit" in user_input.lower():
            goodbye = "Shutting down core systems. Goodbye."
            print(f"{Colors.JARVIS}JARVIS: {goodbye}{Colors.RESET}")
            speak(goodbye)
            # Gracefully close the vector database so it doesn't crash on exit
            qdrant.close()
            break

        # 2. Save User Input to Memory
        save_to_memory(user_input, "User")
        
        # 3. Generate Answer (Brain + Memory)
        response = generate_response(user_input)
        
        # 4. Save Jarvis Output to Memory
        save_to_memory(response, "Jarvis")
        
        # 5. Speak
        speak(response)

if __name__ == "__main__":
    main()