import time
from functools import wraps
from core.colors import Colors
from core.config import Config

class MetricsLogger:
    """Centralized telemetry for tracking Execution Time, TTFT, TPS, and system values."""
    
    def __init__(self):
        self.is_enabled = getattr(Config, 'ENABLE_METRICS', True)
        self._active_timers = {}
        self.session_data = {} # Stores all completed metrics for the current interaction

    def set_enabled(self, state: bool):
        """Programmatically toggle metrics on or off at runtime."""
        self.is_enabled = state
        state_str = "ENABLED" if state else "DISABLED"
        print(f"{Colors.METRICS}[Telemetry] Performance tracking is now {state_str}.{Colors.RESET}")

    def start(self, label: str):
        """Starts a manual stopwatch for a specific code block."""
        if self.is_enabled:
            self._active_timers[label] = time.perf_counter()

    def stop(self, label: str) -> float:
        """Stops the manual stopwatch and logs the duration to the session."""
        if self.is_enabled and label in self._active_timers:
            elapsed = time.perf_counter() - self._active_timers.pop(label)
            self.session_data[label] = elapsed
            return elapsed
        return 0.0
        
    def record_value(self, label: str, value):
        """Records a static value (like token counts) to the session."""
        if self.is_enabled:
            self.session_data[label] = value

    def measure(self, label: str):
        """A decorator to automatically wrap, time, and record entire functions."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                
                self.session_data[label] = elapsed
                return result
            return wrapper
        return decorator

    def generate_report(self):
        """Calculates complex metrics (like TPS) and prints a clean summary table."""
        if not self.is_enabled or not self.session_data:
            return

        # 1. TPS Math
        if "LLM Output Tokens" in self.session_data and "LLM Generation Time" in self.session_data:
            tokens = self.session_data["LLM Output Tokens"]
            gen_time = self.session_data["LLM Generation Time"]
            if gen_time > 0:
                self.session_data["Tokens Per Second (TPS)"] = tokens / gen_time

        # 2. Advanced Ingestion Math
        if self.session_data.get("Total Images Processed", 0) > 0:
            if "PDF Extraction & OCR Time" in self.session_data:
                self.session_data["Avg Time per Image"] = self.session_data["PDF Extraction & OCR Time"] / self.session_data["Total Images Processed"]
                
        if self.session_data.get("Total Chunks Embedded", 0) > 0:
            if "Vector Embedding Time" in self.session_data:
                self.session_data["Avg Time per Embedding"] = self.session_data["Vector Embedding Time"] / self.session_data["Total Chunks Embedded"]

        print(f"\n{Colors.METRICS}=== 📊 INTERACTION METRICS REPORT ==={Colors.RESET}")
        for key, value in self.session_data.items():
            if isinstance(value, float):
                print(f"{Colors.METRICS}- {key}: {value:.4f}{Colors.RESET}")
            else:
                print(f"{Colors.METRICS}- {key}: {value}{Colors.RESET}")
        print(f"{Colors.METRICS}====================================={Colors.RESET}\n")

    def reset_session(self):
        """Clears the slate for the next user interaction."""
        self._active_timers.clear()
        self.session_data.clear()

# Global singleton
telemetry = MetricsLogger()