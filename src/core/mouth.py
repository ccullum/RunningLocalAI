import os
import re
import wave
import uuid
import queue
import threading
import keyboard
import pyaudio
from .piper_wrapper import PiperTTSWrapper
from core.colors import Colors

class AsyncMouth:
    def __init__(self):
        print("[System] Initializing Async Mouth (Piper TTS)...")
        
        # Dynamically locate the centralized ../data/ folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "..", "data")
        
        model_path = os.path.join(data_dir, "piper-lessac.onnx")
        piper_dir = os.path.join(data_dir, "piper")
        
        self.piper_tts = PiperTTSWrapper(model_path=model_path, piper_dir=piper_dir)
        
        self.tts_queue = queue.Queue()
        self.is_interrupted = False
        
        # Bind the hardware kill switch
        keyboard.add_hotkey('space', self.trigger_interrupt)
        
        # Start the background worker thread
        self.worker_thread = threading.Thread(target=self._tts_worker_loop, daemon=True)
        self.worker_thread.start()

    def trigger_interrupt(self):
        """Flips the kill switch. Thread-safe."""
        if not self.is_interrupted:
            self.is_interrupted = True
            print("\n{Colors.WARNING}[Barge-in Detected: Halting Audio...]{Colors.RESET}")

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
            
            if text is None: # Poison pill to kill thread
                break
                
            # If interrupted before starting this sentence, skip it
            if self.is_interrupted:
                self.tts_queue.task_done()
                continue

            clean_text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if not clean_text:
                self.tts_queue.task_done()
                continue

            temp_file = f"temp_piper_{uuid.uuid4().hex[:6]}.wav"
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
                print(f"[Mouth Error: {e}]")
            finally:
                if os.path.exists(temp_file): 
                    os.remove(temp_file)
                self.tts_queue.task_done()