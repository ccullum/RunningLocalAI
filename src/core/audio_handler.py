import os
import threading
import time
import wave
import pyaudio
import keyboard
import re
import numpy as np
from faster_whisper import WhisperModel
from piper.voice import PiperVoice
from utils.metrics import perf_tracker
from .colors import Colors
from .config import Config

class AudioEngine:
    def __init__(self, memory_manager=None):
        self.memory = memory_manager
        self.fs = Config.AUDIO_SAMPLE_RATE
        self.channels = Config.AUDIO_CHANNELS
        self.chunk_size = Config.AUDIO_CHUNK_SIZE
        self.recording = False
        self.speaker_enabled = True  
        self.frames = []
        self.output_dir = Config.AUDIO_TEMP_DIR
        self.last_toggle_time = 0.0
        self.processing = False 
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.p = pyaudio.PyAudio()
        
        # Open hardware microphone stream hook
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.fs,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        threading.Thread(target=self._constant_mic_capture_loop, daemon=True).start()
        
        print(f"{Colors.SYSTEM}[Audio] Activating Global Hotkey Listener ({Config.GLOBAL_PTT_HOTKEY})...{Colors.RESET}")
        threading.Thread(target=self._start_hotkey_loop, daemon=True).start()

    def _constant_mic_capture_loop(self):
        """Continuously unloads the hardware mic stream, exiting cleanly if the stream dies."""
        consecutive_errors = 0
        while True:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                if self.recording:
                    self.frames.append(data)
                consecutive_errors = 0  
            except Exception:
                consecutive_errors += 1
                if consecutive_errors > Config.AUDIO_MAX_CONSECUTIVE_ERRORS:
                    print("[Audio] Hardware stream closed. Exiting mic thread cleanly.")
                    break
                time.sleep(0.01)

    def _start_hotkey_loop(self):
        keyboard.add_hotkey(Config.GLOBAL_PTT_HOTKEY, self.toggle_recording)
        keyboard.wait()

    def toggle_recording(self):
        current_time = time.time()
        if current_time - self.last_toggle_time < Config.AUDIO_DEBOUNCE_TIME:
            return
            
        if self.processing:
            print(f"{Colors.WARNING}[Audio Engine] JARVIS is currently processing or speaking. Please wait...{Colors.RESET}")
            return
            
        self.last_toggle_time = current_time
        
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.frames = []  
        self.recording = True
        print(f"\n{Colors.WARNING}🎙️ [Mics Hot] Recording... Press '{Config.GLOBAL_PTT_HOTKEY}' again to finalize audio payload.{Colors.RESET}")

    def stop_recording(self):
        self.recording = False
        print(f"{Colors.SYSTEM}🛑 [Mics Cold] Sound capture completed. Running AI Audio Pipeline...{Colors.RESET}")
        
        time.sleep(0.05)
        if not self.frames:
            print(f"{Colors.ERROR}[Audio Error] No audio array frame blocks captured.{Colors.RESET}")
            return

        raw_bytes = b''.join(self.frames)
        capture_path = os.path.join(self.output_dir, "temp_capture.wav")
        try:
            wf = wave.open(capture_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.fs)
            wf.writeframes(raw_bytes)
            wf.close()
        except Exception as e:
            print(f"{Colors.ERROR}[Audio File Writing Error]: {e}{Colors.RESET}")
            return
        
        threading.Thread(target=self._process_voice_pipeline, args=(capture_path,), daemon=True).start()

    def listen(self, audio_file_path: str) -> str:
        """Transcribe user audio using faster-whisper with anti-hallucination guardrails."""
        perf_tracker.start("Speech-to-Text (faster-whisper)")
        try:
            if not hasattr(self, "whisper_model"):
                print(f"{Colors.SYSTEM}[Audio] Loading Local faster-whisper engine ({Config.STT_MODEL})...{Colors.RESET}")
                self.whisper_model = WhisperModel(
                    Config.STT_MODEL, 
                    device=Config.EAR_DEVICE, 
                    compute_type=Config.EAR_COMPUTE_TYPE
                )
                
            segments, info = self.whisper_model.transcribe(
                audio_file_path, 
                beam_size=Config.STT_BEAM_SIZE,
                language=Config.STT_LANGUAGE,
                condition_on_previous_text=False,
                vad_filter=True
            )
            
            text = "".join(segment.text for segment in segments).strip()
            perf_tracker.stop("Speech-to-Text (faster-whisper)")
            return text
        except Exception as e:
            print(f"{Colors.ERROR}[faster-whisper Processing Exception]: {e}{Colors.RESET}")
            perf_tracker.stop("Speech-to-Text (faster-whisper)")
            return ""

    def _clean_text_for_speech(self, text: str) -> str:
        """Strips out Markdown syntax and ensures structural line breaks get natural punctuation pauses."""
        # 1. Strip heavy code fences and structural formatting anchors first
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'[\*_]{1,3}(.*?)[\*_]{1,3}', r'\1', text)
        text = re.sub(r'\[(.*?)\]\([^\)]+\)', r'\1', text)
        
        # 2. Break text down by major double-linebreak paragraph boundaries
        paragraphs = text.split('\n\n')
        processed_paragraphs = []
        
        for para in paragraphs:
            lines = para.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Clean header formatting hashes and list bullet nodes per line
                line = re.sub(r'^#+\s+', '', line)
                line = re.sub(r'^\s*[\*\+-]\s+', '', line)
                line = re.sub(r'^\s*\d+\.\s+', '', line)
                line = re.sub(r'[\U00010000-\U0010ffff]', '', line)  # Strip Emojis
                
                trimmed = line.strip()
                if not trimmed:
                    continue
                    
                # 🌟 THE NATURAL PACING FIX:
                # If a list item or structural line finishes without a standard punctuation ending,
                # inject a period. This forces the audio engine to pause naturally between lines.
                if not trimmed[-1] in ('.', '!', '?', ':', ',', ';'):
                    trimmed += '.'
                    
                cleaned_lines.append(trimmed)
            
            if cleaned_lines:
                # Combine intra-paragraph items using a single space to yield flowing prose sentences
                processed_paragraphs.append(" ".join(cleaned_lines))
                
        # Return paragraph layouts separated by dual breaks so the synthesis engine applies deep breaks
        return "\n\n".join(processed_paragraphs)

    def speak(self, text: str, output_path: str) -> str:
        """Synthesize response text to speech using Piper with markdown cleaning and paragraph pauses."""
        perf_tracker.start("Text-to-Speech (Piper)")
        try:
            if not hasattr(self, "piper_voice"):
                model_path = Config.PIPER_MODEL_PATH
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Piper ONNX model file missing: {model_path}")
                print(f"{Colors.SYSTEM}[Audio] Initializing Piper Speech Synthesis Engine...{Colors.RESET}")
                self.piper_voice = PiperVoice.load(model_path)
                
            cleaned_text = self._clean_text_for_speech(text)
            paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
            
            if not paragraphs:
                perf_tracker.stop("Text-to-Speech (Piper)")
                return ""
                
            if len(paragraphs) == 1:
                with wave.open(output_path, 'wb') as wav_file:
                    self.piper_voice.synthesize_wav(paragraphs[0], wav_file)
                perf_tracker.stop("Text-to-Speech (Piper)")
                return output_path
                
            temp_files = []
            for idx, para in enumerate(paragraphs):
                temp_p = os.path.join(self.output_dir, f"temp_para_{idx}.wav")
                with wave.open(temp_p, 'wb') as wav_file:
                    self.piper_voice.synthesize_wav(para, wav_file)
                temp_files.append(temp_p)
                
            with wave.open(temp_files[0], 'rb') as first_wf:
                params = first_wf.getparams()
                sample_rate = first_wf.getframerate()
                sample_width = first_wf.getsampwidth()
                channels = first_wf.getnchannels()
            
            with wave.open(output_path, 'wb') as master_wf:
                master_wf.setparams(params)
                
                for idx, temp_p in enumerate(temp_files):
                    with wave.open(temp_p, 'rb') as wf:
                        master_wf.writeframes(wf.readframes(wf.getnframes()))
                    
                    if idx < len(temp_files) - 1:
                        silence_duration = Config.AUDIO_PARAGRAPH_SILENCE_DURATION  
                        silence_bytes = b'\x00' * int(sample_rate * sample_width * channels * silence_duration)
                        master_wf.writeframes(silence_bytes)
                        
            for temp_p in temp_files:
                try: os.remove(temp_p)
                except: pass
                
            perf_tracker.stop("Text-to-Speech (Piper)")
            return output_path
        except Exception as e:
            print(f"{Colors.ERROR}[Piper Synthesis Exception]: {e}{Colors.RESET}")
            perf_tracker.stop("Text-to-Speech (Piper)")
            return ""

    def play_audio_async(self, file_path: str):
        threading.Thread(target=self._play_audio, args=(file_path,), daemon=True).start()

    def _play_audio(self, file_path: str):
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return
            
        with wave.open(file_path, 'rb') as wf:
            p = pyaudio.PyAudio()
            try:
                playback_chunk = Config.AUDIO_PLAYBACK_CHUNK 
                stream = p.open(
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    frames_per_buffer=playback_chunk
                )
                
                data = wf.readframes(playback_chunk)
                while len(data) > 0:
                    if not self.speaker_enabled:
                        print(f"{Colors.WARNING}[Audio] Playback instantly interrupted via UI command.{Colors.RESET}")
                        break
                    stream.write(data)
                    data = wf.readframes(playback_chunk)
                    
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"{Colors.ERROR}[Audio Playback Exception]: {e}{Colors.RESET}")
            finally:
                p.terminate()

    def _process_voice_pipeline(self, wav_path: str):
        """Asynchronous voice execution pipeline sequence."""
        self.processing = True 
        try:
            user_query = self.listen(wav_path)
            if not user_query.strip():
                print(f"{Colors.WARNING}[Audio] Pure ambient silence filtered out by VAD. Aborting pipeline.{Colors.RESET}")
                self.processing = False
                return
            
            print(f"\n{Colors.ROUTER}🗣️ [Spoken Input]: {user_query}{Colors.RESET}")
            
            if self.memory:
                self.memory.add_user_message(user_query)
                payload = self.memory.get_context_payload(user_query)
                
                enriched_prompt = user_query
                if payload:
                    context_text = Config.CONTEXT_INJECTION_TEMPLATE.format(context=payload)
                    enriched_prompt = f"{user_query}{context_text}"
                
                response_text = self.memory.brain.process_background_task(enriched_prompt)
                print(f"{Colors.ROUTER}🤖 [JARVIS Voice]: {response_text}{Colors.RESET}")
                self.memory.add_assistant_message(response_text)
                
                response_wav = os.path.join(self.output_dir, "response.wav")
                self.speak(response_text, response_wav)
                
                if self.speaker_enabled:
                    self._play_audio(response_wav)
                
        except Exception as e:
            print(f"{Colors.ERROR}[Voice Loop Exception Error]: {e}{Colors.RESET}")
        finally:
            self.processing = False