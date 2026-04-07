# Type stub for the C++ llama_engine module

class EmbeddingEngine:
    def __init__(self, model_path: str) -> None:
        """Initializes the bare-metal llama.cpp engine and loads the GGUF model into memory."""
        ...

    def generate_embedding(self, text: str) -> list[float]:
        """Performs a single forward pass to generate a mathematical vector representation of the text."""
        ...