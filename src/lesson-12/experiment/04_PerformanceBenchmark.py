import sys
import os
import time
from openai import OpenAI

# ==========================================
# PATH INJECTION (To find the core package)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.config import Config
from core.semantic_router import LocalSemanticRouter, LegacySemanticRouter
from core.colors import Colors

def run_benchmark():
    print(f"{Colors.SYSTEM}--- Bare-Metal vs. API Performance Benchmark ---{Colors.RESET}")
    
    # 1. Setup Legacy (API-based)
    # We assume LM Studio is running on the port defined in your Config
    client = OpenAI(base_url=Config.LLM_BASE_URL, api_key=Config.LLM_API_KEY)
    
    print(f"\n[1/4] Initializing Legacy Router (Network + API)...")
    start_init_legacy = time.perf_counter()
    legacy_router = LegacySemanticRouter(client, Config.EMBED_MODEL)
    legacy_init_time = time.perf_counter() - start_init_legacy

    # 2. Setup Local (C++ Engine)
    print(f"\n[2/4] Initializing Local Router (RAM + C++ Engine)...")
    start_init_local = time.perf_counter()
    local_router = LocalSemanticRouter()
    local_init_time = time.perf_counter() - start_init_local

    # 3. Define Test Queries
    queries = [
        "What is my name?",
        "Can you summarize the document?",
        "How is the weather today?",
        "Tell me a joke.",
        "Recap our conversation so far."
    ]

    print(f"\n[3/4] Running {len(queries)} routing tests...")

    # Benchmark Legacy
    legacy_latencies = []
    for q in queries:
        start = time.perf_counter()
        legacy_router.route(q)
        legacy_latencies.append(time.perf_counter() - start)

    # Benchmark Local
    local_latencies = []
    for q in queries:
        start = time.perf_counter()
        local_router.route(q)
        local_latencies.append(time.perf_counter() - start)

    # 4. Final Comparison Report
    avg_legacy = sum(legacy_latencies) / len(legacy_latencies)
    avg_local = sum(local_latencies) / len(local_latencies)
    speedup = avg_legacy / avg_local

    print("\n" + "="*50)
    print(f"{'METRIC':<25} | {'LEGACY (API)':<15} | {'LOCAL (C++)'}")
    print("-" * 50)
    print(f"{'Startup (Pre-embed)':<25} | {legacy_init_time:>13.4f}s | {local_init_time:>10.4f}s")
    print(f"{'Avg Routing Latency':<25} | {avg_legacy:>13.4f}s | {avg_local:>10.4f}s")
    print("-" * 50)
    print(f"{Colors.SYSTEM}REAL-TIME SPEEDUP: {speedup:.2f}x faster{Colors.RESET}")
    print("="*50)

if __name__ == "__main__":
    # Ensure LM Studio is running if testing Legacy
    run_benchmark()