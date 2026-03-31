import os
import speech_recognition as sr
from faster_whisper import WhisperModel
from .colors import Colors
from .config import Config

class VADTranscriptEar: # (Or JarvisEar, depending on your rename!)
    """Handles Voice Activity Detection (VAD) and local transcription."""
    
    def __init__(self):
        print(f"{Colors.SYSTEM}[System] Loading Faster-Whisper Ear into VRAM/CPU...{Colors.RESET}")
        self.stt_model = WhisperModel(
            Config.STT_MODEL, 
            device=Config.EAR_DEVICE, 
            compute_type=Config.EAR_COMPUTE_TYPE
        )
        self.recognizer = sr.Recognizer()

    def listen(self) -> str:
        """Captures audio from the microphone and transcribes it instantly."""
        with sr.Microphone() as source:
            print(f"\n{Colors.SYSTEM}[Listening... Speak now (Press 'Space' during playback to interrupt)]{Colors.RESET}")
            
            # Use Config for all VAD tuning!
            self.recognizer.adjust_for_ambient_noise(source, duration=Config.EAR_AMBIENT_DURATION)
            self.recognizer.pause_threshold = Config.EAR_PAUSE_THRESHOLD 
            
            try:
                audio = self.recognizer.listen(
                    source, 
                    timeout=Config.EAR_TIMEOUT, 
                    phrase_time_limit=Config.EAR_PHRASE_LIMIT
                )
                print(f"{Colors.SYSTEM}[Processing Audio...]{Colors.RESET}")
                
                # Drop the microphone capture directly into the data/temp folder
                temp_wav = os.path.join(Config.AUDIO_TEMP_DIR, "temp_capture.wav")
                
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