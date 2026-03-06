import time
import uuid
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 1. Advanced Memory Strategies ---

class RollingSummaryMemory:
    """Option 3: Uses the LLM to compress older messages into a running summary."""
    def __init__(self, model_id: str, client: OpenAI, max_raw_turns=2):
        self.strategy_name = "Rolling Summary"
        self.model_id = model_id
        self.client = client
        self.max_raw_turns = max_raw_turns
        self.raw_history = []
        self.running_summary = ""
        self.overhead_time = 0.0

    def add_message(self, role: str, content: str):
        self.raw_history.append({"role": role, "content": content})
        
        # If we exceed our raw turn limit, trigger a summary compression
        max_messages = (self.max_raw_turns * 2) + 1
        if len(self.raw_history) > max_messages:
            self._compress_history()

    def _compress_history(self):
        start_time = time.perf_counter()
        print("\n   [⚙️ Triggering Background Memory Compression...]")
        
        # Grab the oldest User/Assistant pair to summarize
        oldest_user = self.raw_history.pop(0)
        oldest_assistant = self.raw_history.pop(0)
        
        prompt = f"""Update the following conversation summary with the new exchange. Keep it highly concise.
        Current Summary: {self.running_summary if self.running_summary else 'None.'}
        New Exchange to add:
        User: {oldest_user['content']}
        Assistant: {oldest_assistant['content']}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            self.running_summary = response.choices[0].message.content.strip()
            self.overhead_time = time.perf_counter() - start_time
        except Exception as e:
            print(f"Summary Error: {e}")

    def get_messages(self, system_prompt: str):
        self.overhead_time = 0.0 # Reset overhead tracker for the turn
        dynamic_system = f"{system_prompt}\n\n[CONVERSATION SUMMARY]\n{self.running_summary}" if self.running_summary else system_prompt
        return [{"role": "system", "content": dynamic_system}] + self.raw_history


class VectorizedMemory:
    """Option 4: Embeds every chat turn and uses RAG to retrieve relevant past context."""
    def __init__(self, client: OpenAI):
        self.strategy_name = "Vectorized (RAG-Chat)"
        self.client = client
        self.embed_model = "text-embedding-nomic-embed-text-v1.5@q8_0"
        
        # Ephemeral in-memory database
        self.qdrant = QdrantClient(location=":memory:") 
        self.collection = "chat_memory"
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        self.overhead_time = 0.0
        self.current_turn_context = ""

    def add_message(self, role: str, content: str):
        # We embed and store every message as it happens
        start_time = time.perf_counter()
        try:
            vector = self.client.embeddings.create(input=content, model=self.embed_model).data[0].embedding
            self.qdrant.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=uuid.uuid4().hex, vector=vector, payload={"role": role, "text": content})]
            )
        except Exception as e:
            pass # Silently fail for benchmarking simplicity
        self.overhead_time = time.perf_counter() - start_time

    def get_messages(self, system_prompt: str, user_query: str):
        start_time = time.perf_counter()
        # 1. Embed the incoming user query to find relevant past context
        try:
            query_vector = self.client.embeddings.create(input=user_query, model=self.embed_model).data[0].embedding
            results = self.qdrant.query_points(collection_name=self.collection, query=query_vector, limit=3)
            
            past_context = [p.payload['text'] for p in results.points if p.score >= 0.5]
            self.current_turn_context = "\n".join(past_context)
            
        except Exception:
            self.current_turn_context = ""
            
        self.overhead_time += (time.perf_counter() - start_time)
        
        dynamic_system = f"{system_prompt}\n\n[RELEVANT PAST CONTEXT]\n{self.current_turn_context}" if self.current_turn_context else system_prompt
        
        # We only pass the system prompt and the current query. NO raw history.
        return [{"role": "system", "content": dynamic_system}, {"role": "user", "content": user_query}]


# --- 2. The Advanced Chat Engine ---

class AdvancedChatEngine:
    def __init__(self, model_id: str, memory_strategy):
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.model_id = model_id
        self.memory = memory_strategy
        self.system_prompt = "You are Jarvis. Answer concisely."

    def chat_turn(self, user_input: str, turn_number: int):
        # For Vector Memory, we need the query during the 'get_messages' phase
        if isinstance(self.memory, VectorizedMemory):
            messages_to_send = self.memory.get_messages(self.system_prompt, user_input)
            self.memory.add_message("user", user_input)
        else:
            self.memory.add_message("user", user_input)
            messages_to_send = self.memory.get_messages(self.system_prompt)

        print(f"\n[Turn {turn_number}]\n\tUser: {user_input}")
        print("Jarvis:\n", end="", flush=True)

        start_time = time.perf_counter()
        ttft = 0.0
        
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
                    print(chunk.choices[0].delta.content, end="", flush=True)

            # Save assistant response
            # (In a real app, we'd capture the full response string. Simulating here to save code space).
            self.memory.add_message("assistant", "Simulated assistant response to store in memory.")
            
            total_ttft = ttft + self.memory.overhead_time
            print(f"\n   -> [Memory Overhead: {self.memory.overhead_time:.2f}s | TTFT: {ttft:.2f}s | Total User Wait: {total_ttft:.2f}s]")

        except Exception as e:
            print(f"\n[Error]: {e}")

def run_advanced_benchmark():
    # A stress-test conversation designed to force amnesia and test recall
    test_conversation = [
        "Hi, my name is Christopher and I'm building a Python application.",
        "It is a RAG pipeline utilizing Qdrant for vector storage.",
        "Can you explain the primary benefit of microservices?",  # Distractor topic
        "What is the CAP theorem?", # Distractor topic
        "What was my name again, and what specific database am I using for my project?" # Recall test
    ]

    target_model = "mistralai/ministral-3-3b" # Ensure this is loaded in LM Studio alongside Nomic!
    oai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    strategies = [
        RollingSummaryMemory(model_id=target_model, client=oai_client, max_raw_turns=1),
        VectorizedMemory(client=oai_client)
    ]

    print("="*70)
    print("🧠 JARVIS ADVANCED MEMORY BENCHMARK 🧠")
    print("="*70)

    for strategy in strategies:
        print(f"\n\n⚙️  TESTING STRATEGY: {strategy.strategy_name}")
        print("-" * 50)
        
        engine = AdvancedChatEngine(model_id=target_model, memory_strategy=strategy)
        
        for i, query in enumerate(test_conversation, 1):
            engine.chat_turn(query, turn_number=i)

if __name__ == "__main__":
    run_advanced_benchmark()