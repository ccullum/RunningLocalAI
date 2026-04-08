import llama_engine
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

def run_semantic_test():
    print(f"{Colors.SYSTEM}--- Bare-Metal Semantic Similarity POC ---{Colors.RESET}")
    
    # 1. Initialize engine using Config path
    if not Config.NOMIC_MODEL_PATH:
        print(f"{Colors.ERROR}⚠️ ERROR: NOMIC_MODEL_PATH not found in .env or Config.{Colors.RESET}")
        return

    print(f"{Colors.SYSTEM}[1] Loading Engine with model: {os.path.basename(Config.NOMIC_MODEL_PATH)}{Colors.RESET}")
    engine = llama_engine.EmbeddingEngine(Config.NOMIC_MODEL_PATH)

    # 2. Get test data from Config
    query = Config.SEMANTIC_TEST_QUERY
    docs = Config.SEMANTIC_TEST_DOCS

    print(f"{Colors.SYSTEM}\n[2] Generating Query Embedding...{Colors.RESET}")
    print(f"{Colors.SYSTEM}Query: '{query}'{Colors.RESET}")
    v_query = engine.generate_embedding(query)

    print(f"\n{Colors.METRICS}{'='*50}")
    print(f"{'SIMILARITY':<12} | {'DOCUMENT'}")
    print(f"{'='*50}{Colors.RESET}")

    # 3. Perform C++ similarity math for each document
    for doc in docs:
        v_doc = engine.generate_embedding(doc)
        score = engine.similarity(v_query, v_doc)
        print(f"{score:>10.4f}   | {doc}")

if __name__ == "__main__":
    run_semantic_test()