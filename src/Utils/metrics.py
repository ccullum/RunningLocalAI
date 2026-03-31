import time
from functools import wraps
from core.colors import Colors
from core.config import Config

class MetricsLogger:
    """Centralized telemetry for tracking Execution Time and TTFT."""
    
    def __init__(self):
        # Pull the default state from the Control Room
        self.is_enabled = getattr(Config, 'ENABLE_METRICS', True)
        self._timers = {}

    def set_enabled(self, state: bool):
        """Programmatically toggle metrics on or off at runtime."""
        self.is_enabled = state
        state_str = "ENABLED" if state else "DISABLED"
        print(f"{Colors.METRICS}[Telemetry] Performance tracking is now {state_str}.{Colors.RESET}")

    def start(self, label: str):
        """Starts a manual stopwatch for a specific code block."""
        if self.is_enabled:
            self._timers[label] = time.perf_counter()

    def stop(self, label: str) -> float:
        """Stops the manual stopwatch and logs the duration."""
        if self.is_enabled and label in self._timers:
            elapsed = time.perf_counter() - self._timers.pop(label)
            print(f"{Colors.METRICS}[Telemetry] {label}: {elapsed:.4f} seconds{Colors.RESET}")
            return elapsed
        return 0.0

    def measure(self, label: str):
        """A decorator to automatically wrap and time entire functions."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                
                print(f"{Colors.METRICS}[Telemetry] {label} ({func.__name__}): {elapsed:.4f} seconds{Colors.RESET}")
                return result
            return wrapper
        return decorator

# Instantiate a global singleton so the whole app shares the exact same logger
telemetry = MetricsLogger()