import os
import re
import wave
import uuid
import queue
import threading
import keyboard
import pyaudio
from .piper_wrapper import PiperTTSWrapper
from .colors import Colors

class AsyncMouth:
    """Handles threaded TTS generation and hardware barge-in (interrupts)."""
    
    def __init__(self):
        print(f"{Colors.SYSTEM}[System] Initializing Async Mouth (Piper TTS)...{Colors.RESET}")
        
        # 1. Dynamically anchor to this file's location (.../src/core/mouth.py)
        core_dir = os.path.dirname(os.path.abspath(__file__))
        # 2. Go up two levels to the root, then into the data folder
        data_dir = os.path.abspath(os.path.join(core_dir, "..", "..", "data"))
        
        model_path = os.path.join(data_dir, "piper-lessac.onnx")
        piper_dir = os.path.join(data_dir, "piper")
        
        self.piper_tts = PiperTTSWrapper(model_path=model_path, piper_dir=piper_dir)
        
        self.tts_queue = queue.Queue()
        self.is_interrupted = False
        
        # Bind the hardware kill switch to the Spacebar
        keyboard.add_hotkey('space', self.trigger_interrupt)
        
        # Start the background worker thread
        self.worker_thread = threading.Thread(target=self._tts_worker_loop, daemon=True)
        self.worker_thread.start()

    def trigger_interrupt(self):
        """Flips the kill switch. Thread-safe."""
        if not self.is_interrupted:
            self.is_interrupted = True
            print(f"\n{Colors.WARNING}[Barge-in Detected: Halting Audio...]{Colors.RESET}")

    def reset_state(self):
        """Clears the interrupt flag and flushes the queue for a new conversational turn."""
        self.is_interrupted = False
        with self.tts_queue.mutex:
            self.tts_queue.queue.clear()

    def enqueue_sentence(self, text):
        """Pushes a sentence to the background thread to be spoken."""
        self.tts_queue.put(text)

    def wait_until_done(self):
        """Blocks the main thread until the mouth finishes speaking all queued sentences."""
        self.tts_queue.join()

    def _tts_worker_loop(self):
        """The background thread that constantly watches the queue."""
        while True:
            text = self.tts_queue.get()
            
            if text is None: # Poison pill to kill thread safely on shutdown
                break
                
            # If interrupted before starting this sentence, skip it entirely
            if self.is_interrupted:
                self.tts_queue.task_done()
                continue

            clean_text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if not clean_text:
                self.tts_queue.task_done()
                continue

            # --- THE ARCHITECT'S PATH ANCHORING ---
            core_dir = os.path.dirname(os.path.abspath(__file__))
            temp_file = os.path.join(core_dir, f"temp_piper_{uuid.uuid4().hex[:6]}.wav")
            
            try:
                success, result = self.piper_tts.synthesize(clean_text, temp_file)
                
                if success and os.path.exists(temp_file) and os.path.getsize(temp_file) > 100:
                    wf = wave.open(temp_file, 'rb')
                    p = pyaudio.PyAudio()
                    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                    channels=wf.getnchannels(),
                                    rate=wf.getframerate(),
                                    output=True)
                    
                    data = wf.readframes(1024)
                    
                    # TRUE BARGE-IN: Check the kill switch on every single audio chunk!
                    while data and not self.is_interrupted:
                        stream.write(data)
                        data = wf.readframes(1024)
                        
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    wf.close()
            except Exception as e:
                print(f"{Colors.ERROR}[Mouth Error: {e}]{Colors.RESET}")
            finally:
                if os.path.exists(temp_file): 
                    os.remove(temp_file)
                self.tts_queue.task_done()