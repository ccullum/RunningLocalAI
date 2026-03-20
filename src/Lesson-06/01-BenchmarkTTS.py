import sys
import os
import time
import csv
import urllib.request
from datetime import datetime
import numpy as np
from core.colors import Colors

# ==========================================
# THE ARCHITECT'S PATH INJECTION
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Bootstrapping script into UTF-8 mode for Windows compatibility
if sys.platform == "win32" and os.environ.get("PYTHONUTF8") != "1":
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Import the wrapper from the new core package!
from core.piper_wrapper import PiperTTSWrapper

# --- 1. Smart Downloader for Neural Models ---
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"{Colors.WARNING}[Downloading] {filename} (This may take a minute depending on your connection...){Colors.RESET}")
        urllib.request.urlretrieve(url, filename)
        print(f"{Colors.SYSTEM}[✅ Downloaded] {filename}{Colors.RESET}")

def ensure_models():
    print(f"{Colors.SYSTEM}Checking required neural TTS models...{Colors.RESET}")
    # Kokoro Models (~300MB)
    download_file("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx", "kokoro-v1.0.onnx")
    download_file("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", "voices.bin")
    
    # Piper Models (~50MB)
    download_file("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx", "piper-lessac.onnx")
    download_file("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json", "piper-lessac.onnx.json")
    print("\n")

# --- 2. The STT Implementations ---
def test_pyttsx3(text):
    """Legacy offline TTS."""
    import pyttsx3
    engine = pyttsx3.init()
    
    start_time = time.perf_counter()
    engine.save_to_file(text, 'temp_pyttsx3.wav')
    engine.runAndWait()
    ttfa = time.perf_counter() - start_time
    
    if os.path.exists('temp_pyttsx3.wav'): os.remove('temp_pyttsx3.wav')
    return ttfa, "pyttsx3"

def test_piper_binary(text):
    """Optimized neural TTS via the Platform-Agnostic Binary Wrapper."""
    # Point it to the centralized data folder!
    piper = PiperTTSWrapper(model_path="../data/piper-lessac.onnx", piper_dir="../data/piper")
    
    start_time = time.perf_counter()
    
    # Run the binary synthesis
    success, result = piper.synthesize(text, "temp_benchmark_piper.wav")
    ttfa = time.perf_counter() - start_time
    
    if success and os.path.exists(result):
        os.remove(result)
        return ttfa, "Piper TTS (Binary)"
    else:
        raise RuntimeError(f"Binary failed: {result}")

def test_kokoro(text):
    """SOTA neural TTS via ONNX. Supports streaming!"""
    import asyncio
    from kokoro_onnx import Kokoro
    
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices.bin")
    
    async def get_kokoro_ttfa():
        start_time = time.perf_counter()
        ttfa = 0.0
        stream = kokoro.create_stream(text, voice="af_sarah", speed=1.0, lang="en-us")
        
        async for _ in stream:
            if ttfa == 0.0:
                ttfa = time.perf_counter() - start_time
                break
        return ttfa

    ttfa = asyncio.run(get_kokoro_ttfa())
    return ttfa, "Kokoro-ONNX"

# --- 3. The Benchmark Runner ---
def run_tts_benchmark():
    ensure_models()
    
    test_sentence = "Hello Christopher. All memory systems are online, and I am fully operational. How can I assist you with your Python architecture today?"
    print(f"Test Payload: \"{test_sentence}\"\n")
    
    tests = [
        {"Tool": "pyttsx3 (Legacy)", "Func": test_pyttsx3},
        {"Tool": "Piper TTS (Binary)", "Func": test_piper_binary}
    ]
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"TTS_benchmark_{timestamp}.csv"
    results = []
    
    for test in tests:
        print(f"Testing {test['Tool']}...")
        try:
            ttfa, engine_name = test["Func"](test_sentence)
            print(f" -> {Colors.METRICS}Time-To-First-Audio (TTFA): {ttfa:.3f}s{Colors.RESET}\n")
        except Exception as e:
            print(f" -> {Colors.WARNING}Failed: {e}{Colors.RESET}\n")
            ttfa = float('inf')
            
        results.append({
            "Timestamp": datetime.now().strftime("%m/%d/%Y %H:%M"),
            "Tool": test["Tool"],
            "Payload_Length": len(test_sentence),
            "TTFA_Sec": round(ttfa, 3)
        })

    # Export to CSV
    headers = ["Timestamp", "Tool", "Payload_Length", "TTFA_Sec"]
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Benchmark complete! Results saved to {csv_filename}")

if __name__ == "__main__":
    run_tts_benchmark()