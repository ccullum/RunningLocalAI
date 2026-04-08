import time
import sys
import os

# ==========================================
# PATH INJECTION (To find the core package)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.colors import Colors
from core.config import Config

# We will compile our C++ engine to be imported as 'llama_engine'
import llama_engine 

def main():
    print(f"{Colors.SYSTEM}--- Bare-Metal Embedding POC ---{Colors.RESET}")
    
    # You will need the absolute path to your Nomic GGUF file
    model_path = Config.NOMIC_MODEL_PATH
    
    print(f"{Colors.SYSTEM}[1] Loading Model straight into RAM...{Colors.RESET}")
    start_load = time.perf_counter()
    
    # Initialize our C++ class!
    embedder = llama_engine.EmbeddingEngine(model_path)
    
    print(f"{Colors.USER}Model loaded in {time.perf_counter() - start_load:.2f}s{Colors.RESET}\n")

    # The test prompt
    prompt = Config.EM_WRAP_TEST_PROMPT
    print(f"{Colors.USER}Prompt:{Colors.RESET} {prompt}")
    
    print(f"{Colors.SYSTEM}[2] Running bare-metal forward pass...{Colors.RESET}")
    start_embed = time.perf_counter()
    
    # Call the C++ forward pass
    vector = embedder.generate_embedding(prompt)
    
    embed_time = time.perf_counter() - start_embed
    
    print(f"{Colors.USER}Embedding Generated in {embed_time:.4f}s!{Colors.RESET}")
    print(f"Vector Dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}\n")

if __name__ == "__main__":
    main()