import time
import uuid
import csv
from datetime import datetime
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 1. Dynamic Memory Manager (The HRE) ---
class DynamicMemoryManager:
    def __init__(self, model_id: str, client: OpenAI):
        self.client = client
        self.model_id = model_id
        self.embed_model = "text-embedding-nomic-embed-text-v1.5@q8_0"
        
        self.raw_history = []
        self.running_summary = ""
        
        self.qdrant = QdrantClient(location=":memory:")
        self.collection = "jarvis_dynamic_memory"
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

        self.overhead_time = 0.0
        self.active_strategy = ""

    def add_message(self, role: str, content: str):
        start_time = time.perf_counter()
        self.raw_history.append({"role": role, "content": content})
        try:
            vector = self.client.embeddings.create(input=content, model=self.embed_model).data[0].embedding
            self.qdrant.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=uuid.uuid4().hex, vector=vector, payload={"role": role, "text": content})]
            )
        except Exception:
            pass
        self.overhead_time += (time.perf_counter() - start_time)

    def _route_intent(self, user_query: str) -> str:
        prompt = f"""Analyze this user query: "{user_query}"
        Reply with EXACTLY ONE WORD from this list:
        RECALL (If asking for a specific past detail, name, or fact)
        SUMMARY (If asking to summarize or review the conversation)
        CHAT (If standard conversational reply)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            intent = response.choices[0].message.content.strip().upper()
            if "RECALL" in intent: return "RECALL"
            if "SUMMARY" in intent: return "SUMMARY"
            return "CHAT"
        except Exception:
            return "CHAT"

    def _deconstruct_query(self, user_query: str) -> list:
        prompt = f"""Split this query into distinct, simple search questions. Separate them with a pipe character '|'.
        Query: "{user_query}"
        Provide ONLY the separated questions with no introductory text or markdown.
        Example Output: What is the user's name? | What database are they using?"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            raw = response.choices[0].message.content.strip()
            return [q.strip() for q in raw.split('|') if q.strip()]
        except Exception:
            return [user_query]

    def _update_summary(self):
        if len(self.raw_history) <= 4: return 
        oldest_user = self.raw_history.pop(0)
        oldest_assistant = self.raw_history.pop(0)
        prompt = f"""Update this summary with the new facts. Be incredibly brief.
        Current: {self.running_summary}
        New: User said "{oldest_user['content']}", Assistant replied "{oldest_assistant['content']}" """
        try:
            res = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            self.running_summary = res.choices[0].message.content.strip()
        except Exception:
            pass

    def get_messages(self, system_prompt: str, user_query: str, turn_number: int):
        self.overhead_time = 0.0
        start_time = time.perf_counter()

        # Rule 1: The Fast Path
        if turn_number <= 3:
            self.active_strategy = "Sliding Window"
            self.overhead_time += (time.perf_counter() - start_time)
            return [{"role": "system", "content": system_prompt}] + self.raw_history

        intent = self._route_intent(user_query)

        # Rule 2: The Deep Search Path
        if intent == "RECALL":
            self.active_strategy = "Vector RAG (Deconstructed)"
            sub_queries = self._deconstruct_query(user_query)
            context_pieces = []
            for sq in sub_queries:
                try:
                    query_vector = self.client.embeddings.create(input=sq, model=self.embed_model).data[0].embedding
                    results = self.qdrant.query_points(collection_name=self.collection, query=query_vector, limit=2)
                    context_pieces.extend([p.payload['text'] for p in results.points if p.score >= 0.5])
                except Exception:
                    pass
            unique_context = "\n".join(list(set(context_pieces)))
            dynamic_system = f"{system_prompt}\n\n[RELEVANT PAST CONTEXT]\n{unique_context}"
            self.overhead_time += (time.perf_counter() - start_time)
            return [{"role": "system", "content": dynamic_system}] + self.raw_history[-1:]

        # Rule 3: The Narrative Path
        self.active_strategy = "Rolling Summary"
        self._update_summary()
        dynamic_system = f"{system_prompt}\n\n[CONVERSATION SUMMARY]\n{self.running_summary}"
        self.overhead_time += (time.perf_counter() - start_time)
        return [{"role": "system", "content": dynamic_system}] + self.raw_history


# --- 2. The Benchmark Engine ---
class BenchmarkChatEngine:
    def __init__(self, model_id: str):
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.model_id = model_id
        self.memory = DynamicMemoryManager(model_id, self.client)
        self.system_prompt = "You are Jarvis. Answer concisely."

    def warmup(self):
        print(f"\n[🔥 LOADING & WARMING UP MODEL: {self.model_id} ...]")
        print("   (This may take a moment as LM Studio loads the model from disk to VRAM)")
        start_time = time.perf_counter()
        try:
            self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "Hello."}],
                max_tokens=5
            )
            load_time = time.perf_counter() - start_time
            print(f"[✅ WARM-UP COMPLETE in {load_time:.2f}s. MODEL LOADED & KV CACHE INITIALIZED.]\n")
        except Exception as e:
            print(f"[❌ WARM-UP FAILED. Make sure Just-In-Time loading is enabled in LM Studio]: {e}")

    def chat_turn(self, user_input: str, turn_number: int):
        self.memory.add_message("user", user_input)
        messages_to_send = self.memory.get_messages(self.system_prompt, user_input, turn_number)

        start_time = time.perf_counter()
        ttft = 0.0
        full_response = ""
        tokens = 0

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
                    full_response += chunk.choices[0].delta.content
                    tokens += 1

            total_gen_time = time.perf_counter() - start_time
            tps = tokens / (total_gen_time - ttft) if (total_gen_time - ttft) > 0 else 0
            
            self.memory.add_message("assistant", full_response)
            
            return {
                "Turn": turn_number,
                "Strategy": self.memory.active_strategy,
                "Overhead_Sec": round(self.memory.overhead_time, 3),
                "TTFT_Sec": round(ttft, 3),
                "Total_Gen_Time_Sec": round(total_gen_time, 3),
                "Total_User_Wait_Sec": round(total_gen_time + self.memory.overhead_time, 3),
                "Tokens": tokens,
                "TPS": round(tps, 2),
                "Raw_Output": full_response.strip().replace("\n", " ")
            }

        except Exception as e:
            print(f"Error during generation: {e}")
            return None


def run_hre_benchmark():
    models_to_test = [
        "google/gemma-2-9b",
        "llama-3.2-3b-instruct",
        "mistralai/ministral-3-3b",
        "openai/gpt-oss-20b",
        "phi-3-mini-4k-instruct",
        "qwen2.5-7b-instruct"
    ]

    test_conversation = [
        "Hi, my name is Christopher and I'm building a Python application.",
        "It is a RAG pipeline utilizing Qdrant for vector storage.",
        "Can you explain the primary benefit of microservices?", 
        "What is the CAP theorem?", 
        "What was my name again, and what specific database am I using for my project?"
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"HRE_benchmark_{timestamp}.csv"

    headers = [
        "Timestamp", "Model", "Mode", "Turn", "Strategy", 
        "Overhead_Sec", "TTFT_Sec", "Total_User_Wait_Sec", "Tokens", "TPS", "Raw_Output"
    ]

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        print("="*70)
        print("🧠 JARVIS HRE BENCHMARK SUITE 🧠")
        print("="*70)

        for model in models_to_test:
            print(f"\n\n{'='*50}")
            print(f"🛑 MANUAL PURGE GATE: {model}")
            print(f"{'='*50}")
            print(f"1. Open LM Studio.")
            print(f"2. EJECT the currently loaded LLM to free up VRAM.")
            print(f"3. Do NOT load the next model. (The script will automatically load it via API).")
            print(f"4. Ensure Nomic embedding model remains loaded.")
            input("\nPress ENTER when you are ready for the script to load and test the next model...")

            engine = BenchmarkChatEngine(model_id=model)
            engine.warmup()

            for turn_num, query in enumerate(test_conversation, 1):
                print(f"Testing Turn {turn_num}...")
                metrics = engine.chat_turn(query, turn_num)
                
                if metrics:
                    writer.writerow({
                        "Timestamp": datetime.now().strftime("%m/%d/%Y %H:%M"),
                        "Model": model,
                        "Mode": "JARVIS-HRE",
                        "Turn": metrics["Turn"],
                        "Strategy": metrics["Strategy"],
                        "Overhead_Sec": metrics["Overhead_Sec"],
                        "TTFT_Sec": metrics["TTFT_Sec"],
                        "Total_User_Wait_Sec": metrics["Total_User_Wait_Sec"],
                        "Tokens": metrics["Tokens"],
                        "TPS": metrics["TPS"],
                        "Raw_Output": metrics["Raw_Output"]
                    })
                    print(f"  -> {metrics['Strategy']} | Overhead: {metrics['Overhead_Sec']}s | TTFT: {metrics['TTFT_Sec']}s | TPS: {metrics['TPS']}")
            
            print(f"\n✅ {model} benchmark complete and logged.")

    print(f"\n🎉 All testing complete! Results saved to {csv_filename}")

if __name__ == "__main__":
    run_hre_benchmark()