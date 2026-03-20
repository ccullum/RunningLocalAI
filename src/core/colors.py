from enum import StrEnum

class Colors(StrEnum):
    """Centralized ANSI color codes for terminal UI formatting."""
    SYSTEM = '\033[96m'   # Cyan
    USER = '\033[92m'     # Green
    JARVIS = '\033[93m'   # Yellow
    MEMORY = '\033[95m'   # Magenta
    ERROR = '\033[91m'    # Red
    WARNING = '\033[93m'  # Yellow (same as JARVIS/METRICS, but semantically distinct)
    ROUTER = '\033[95m'   # Magenta
    METRICS = '\033[93m'  # Yellow
    RESET = '\033[0m'     # Reset to default terminal color