import time
import os
import csv
from datetime import datetime
import speech_recognition as sr
import whisper
from faster_whisper import WhisperModel
from core.colors import Colors

def record_benchmark_audio(filename="benchmark_audio.wav"):
    """Records a single static audio file to ensure fair testing across all models."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    print(f"{Colors.SYSTEM}🎙️ We need a static audio sample for the benchmark.{Colors.RESET}")
    print("When ready, speak a complex 5-10 second sentence.")
    print("Example: 'Jarvis, please analyze the vector database and tell me the primary benefits of using microservices.'\n")
    
    input("Press ENTER to start recording...")
    
    with microphone as source:
        print(f"\n{Colors.METRICS}[Recording... Speak Now!]{Colors.RESET}")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
            print(f"{Colors.SYSTEM}[✅ Audio saved to {filename}]{Colors.RESET}\n")
            return filename
        except Exception as e:
            print(f"Recording failed: {e}")
            return None

def test_pocketsphinx(filename):
    """Traditional, ultra-lightweight offline STT (Non-Transformer)."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    
    start_time = time.perf_counter()
    try:
        transcription = recognizer.recognize_sphinx(audio)
    except Exception:
        transcription = "[Failed to transcribe]"
    atl = time.perf_counter() - start_time
    return atl, transcription

def test_openai_whisper(filename, model_size="base"):
    """Original OpenAI PyTorch implementation."""
    model = whisper.load_model(model_size)
    
    start_time = time.perf_counter()
    result = model.transcribe(filename, fp16=False)    
    atl = time.perf_counter() - start_time
    return atl, result["text"].strip()

def test_faster_whisper(filename, model_size="base"):
    """Optimized CTranslate2 implementation."""
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    
    start_time = time.perf_counter()
    segments, _ = model.transcribe(filename, beam_size=5)
    transcription = "".join([segment.text for segment in segments]).strip()
    atl = time.perf_counter() - start_time
    return atl, transcription

def run_stt_benchmark():
    print("="*70)
    print("🎙️ JARVIS STT BENCHMARK SUITE 🎙️")
    print("="*70)

    # 1. Capture the Control Variable
    audio_file = record_benchmark_audio()
    if not audio_file: return

    # 2. Define the Gauntlet
    tests = [
        {"Tool": "PocketSphinx", "Size": "N/A", "Func": lambda: test_pocketsphinx(audio_file)},
        {"Tool": "OpenAI Whisper", "Size": "base", "Func": lambda: test_openai_whisper(audio_file, "base")},
        {"Tool": "Faster-Whisper", "Size": "tiny.en", "Func": lambda: test_faster_whisper(audio_file, "tiny.en")},
        {"Tool": "Faster-Whisper", "Size": "base.en", "Func": lambda: test_faster_whisper(audio_file, "base.en")}
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"STT_benchmark_{timestamp}.csv"
    
    results = []

    print("Running benchmarks... (This will download the models on the first run)\n")
    
    for test in tests:
        print(f"Testing {test['Tool']} ({test['Size']})...")
        atl, text = test["Func"]()
        
        print(f"   -> ATL: {atl:.2f}s | Output: {text}\n")
        
        results.append({
            "Timestamp": datetime.now().strftime("%m/%d/%Y %H:%M"),
            "Tool": test["Tool"],
            "Model_Size": test["Size"],
            "ATL_Sec": round(atl, 3),
            "Transcription": text
        })

    # 3. Export to CSV
    headers = ["Timestamp", "Tool", "Model_Size", "ATL_Sec", "Transcription"]
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Benchmark complete! Results saved to {csv_filename}")
    
    # Clean up the audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)

if __name__ == "__main__":
    run_stt_benchmark()