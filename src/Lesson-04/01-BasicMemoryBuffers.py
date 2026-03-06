import time
import tiktoken
from openai import OpenAI

# --- 1. Token Estimation Tool ---
def estimate_tokens(text: str) -> int:
    """Uses cl100k_base as a fast, platform-agnostic proxy for local SLM token counts."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4  # Rough fallback

# --- 2. Memory Strategies ---
class SlidingWindowMemory:
    """Option 1: Retains exactly N conversational turns (User + Assistant)."""
    def __init__(self, max_turns=2):
        self.max_turns = max_turns
        self.history = []
        self.strategy_name = f"Sliding Window ({max_turns}-Turn)"

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        
        # max_turns * 2 gives us the past pairs. We add +1 to allow the current pending user message.
        max_allowed = (self.max_turns * 2) + 1
        
        # ALWAYS pop in pairs (User then Assistant) to maintain the strict alternating sequence
        while len(self.history) > max_allowed:
            self.history.pop(0) # Remove oldest User
            self.history.pop(0) # Remove oldest Assistant

    def get_messages(self, system_prompt: str):
        return [{"role": "system", "content": system_prompt}] + self.history

class TokenBoundedMemory:
    """Option 2: Retains as many messages as possible under a hard token limit."""
    def __init__(self, max_tokens=300):
        self.max_tokens = max_tokens
        self.history = []
        self.strategy_name = f"Token-Bounded ({max_tokens} Tokens)"

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self._prune()

    def _prune(self):
        # ALWAYS pop in pairs (User then Assistant) to maintain the strict alternating sequence
        while self._get_current_token_count() > self.max_tokens and len(self.history) >= 2:
            self.history.pop(0) # Remove oldest User
            self.history.pop(0) # Remove oldest Assistant

    def _get_current_token_count(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.history)

    def get_messages(self, system_prompt: str):
        return [{"role": "system", "content": system_prompt}] + self.history

# --- 3. The Chat Engine & Benchmarker ---

class StatefulChatEngine:
    def __init__(self, model_id: str, memory_strategy):
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.model_id = model_id
        self.memory = memory_strategy
        self.system_prompt = "You are Jarvis. Provide concise, helpful answers."

    def chat_turn(self, user_input: str, turn_number: int):
        self.memory.add_message("user", user_input)
        messages_to_send = self.memory.get_messages(self.system_prompt)
        
        # Calculate context payload size for telemetry
        payload_tokens = sum(estimate_tokens(m["content"]) for m in messages_to_send)

        print(f"\n[Turn {turn_number}] Payload: ~{payload_tokens} tokens")
        print(f"User: {user_input}")
        print("Jarvis: ", end="", flush=True)

        start_time = time.perf_counter()
        ttft = 0.0
        token_count = 0
        full_response = ""

        try:
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages_to_send,
                temperature=0.7,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    if ttft == 0.0:
                        ttft = time.perf_counter() - start_time
                    
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
                    token_count += 1

            total_time = time.perf_counter() - start_time
            tps = token_count / (total_time - ttft) if (total_time - ttft) > 0 else 0
            
            # Save assistant response to memory for the next turn
            self.memory.add_message("assistant", full_response)

            print(f"\n   -> [TTFT: {ttft:.3f}s | TPS: {tps:.1f}]")
            return ttft, tps, payload_tokens

        except Exception as e:
            print(f"\n[Error]: {e}")
            return 0, 0, 0

def run_memory_benchmark():
    # Test queries designed to build on each other to prove memory works
    test_conversation = [
        "What are the three core laws of robotics?",
        "Who originally wrote them?",
        "Can you summarize the first law in exactly 5 words?",
        "What was the name of the author we just discussed?",
        "What were the three laws again?"
    ]

    target_model = "mistralai/ministral-3-3b" # Change this to your preferred SLM

    strategies = [
        SlidingWindowMemory(max_turns=2), # Only remembers the last 2 interactions
        TokenBoundedMemory(max_tokens=200) # Highly constrained to force pruning
    ]

    print("="*60)
    print("🧠 JARVIS MEMORY BUFFER BENCHMARK 🧠")
    print("="*60)

    for strategy in strategies:
        print(f"\n\n⚙️  TESTING STRATEGY: {strategy.strategy_name}")
        print("-" * 50)
        
        engine = StatefulChatEngine(model_id=target_model, memory_strategy=strategy)
        
        for i, query in enumerate(test_conversation, 1):
            engine.chat_turn(query, turn_number=i)

if __name__ == "__main__":
    run_memory_benchmark()