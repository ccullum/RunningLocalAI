import os
import sys
import subprocess
from core.colors import Colors

class PiperTTSWrapper:
    """A platform-agnostic wrapper for the standalone Piper TTS binary."""
    
    def __init__(self, model_path="piper-lessac.onnx", piper_dir="piper"):
        self.model_path = model_path
        
        # 1. OS-Aware Binary Selection
        if sys.platform == "win32":
            self.binary_path = os.path.join(piper_dir, "piper.exe")
        else:
            # Linux and macOS use extensionless binaries
            self.binary_path = os.path.join(piper_dir, "piper")
            
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"Piper binary not found at {self.binary_path}. Please download and extract it.")

    def synthesize(self, text, output_file="temp_piper.wav"):
        """Pipes text to the binary and generates the WAV file."""
        command = [
            self.binary_path,
            "--model", self.model_path,
            "--output_file", output_file
        ]
        
        # 2. Execute securely via subprocess
        try:
            process = subprocess.run(
                command, 
                input=text.encode('utf-8'), 
                capture_output=True,
                check=True # Raises CalledProcessError if the binary fails
            )
            return True, output_file
        except subprocess.CalledProcessError as e:
            error_log = e.stderr.decode('utf-8') if e.stderr else "Unknown Binary Error"
            return False, error_log