import os
import speech_recognition as sr
from faster_whisper import WhisperModel
from core.colors import Colors

class JarvisEar:
    """Handles Voice Activity Detection (VAD) and local transcription."""
    
    def __init__(self, model_size="tiny.en", device="cpu", compute_type="int8"):
        print("[System] Loading Faster-Whisper Ear into VRAM/CPU...")
        self.stt_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.recognizer = sr.Recognizer()

    def listen(self) -> str:
        """Captures audio from the microphone and transcribes it instantly."""
        with sr.Microphone() as source:
            print("\n{Colors.SYSTEM}[Listening... Speak now (Press 'Space' during playback to interrupt)]{Colors.RESET}")
            # Calibrate for ambient noise and set the slow-speaker pause threshold
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self.recognizer.pause_threshold = 2.0 
            
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)
                print("{Colors.SYSTEM}[Processing Audio...]{Colors.RESET}")
                
                temp_wav = "temp_capture.wav"
                with open(temp_wav, "wb") as f:
                    f.write(audio.get_wav_data())
                
                segments, _ = self.stt_model.transcribe(temp_wav, beam_size=5)
                transcription = "".join([segment.text for segment in segments]).strip()
                
                if os.path.exists(temp_wav): os.remove(temp_wav)
                return transcription
                
            except sr.WaitTimeoutError:
                return ""
            except Exception as e:
                print(f"{Colors.ERROR}[Audio Error: {e}]{Colors.RESET}")
                return ""