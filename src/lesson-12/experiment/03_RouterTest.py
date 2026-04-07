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
from core.semantic_router import SemanticRouter

def test_routing():
    router = SemanticRouter()
    
    test_queries = [
        "What is the capital of France?",             # Should be CHAT
        "What did I tell you about my name earlier?", # Should be RECALL
        "Can you give me a summary of our chat?",     # Should be SUMMARY
        "I like eating apples."                       # Should be CHAT
    ]

    print("\n--- Routing Decisions ---")
    for q in test_queries:
        intent, score = router.route(q)
        print(f"{Colors.SYSTEM}[{intent}] (Score: {score:.4f}) -> Query: {q}{Colors.RESET}")

if __name__ == "__main__":
    test_routing()