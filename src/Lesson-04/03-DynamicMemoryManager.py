import time
import uuid
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from core.colors import Colors

class DynamicMemoryManager:
    def __init__(self, model_id: str, client: OpenAI):
        self.client = client
        self.model_id = model_id
        self.embed_model = "text-embedding-nomic-embed-text-v1.5@q8_0"
        
        # State Storage
        self.raw_history = []
        self.running_summary = ""
        
        # Vector Storage (In-Memory)
        self.qdrant = QdrantClient(location=":memory:")
        self.collection = "jarvis_dynamic_memory"
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

        self.overhead_time = 0.0
        self.active_strategy = ""

    def add_message(self, role: str, content: str):
        """Saves to both short-term history and long-term vector storage."""
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
        """The HRE Router: Decides WHICH memory to use."""
        prompt = f"""Analyze this user query: "{user_query}"
        Reply with EXACTLY ONE WORD from this list:
        RECALL (If they are asking for a specific past detail, name, or fact)
        SUMMARY (If they ask to summarize or review the conversation)
        CHAT (If it is a standard conversational reply)"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            intent = response.choices[0].message.content.strip().upper()
            print(f"{Colors.SYSTEM}   [🧠 Internal HRE Thought]: Classified intent as '{intent}'{Colors.RESET}")
            if "RECALL" in intent: return "RECALL"
            if "SUMMARY" in intent: return "SUMMARY"
            return "CHAT"
        except Exception:
            return "CHAT"

    def _deconstruct_query(self, user_query: str) -> list:
        """Solves Vector Dilution by splitting compound questions."""
        # FIX 2: Stricter prompting to prevent conversational filler
        prompt = f"""Split this query into distinct, simple search questions. Separate them with a pipe character '|'.
        Query: "{user_query}"
        Provide ONLY the separated questions with no introductory text, markdown, or conversational filler.
        Example Output: What is the user's name? | What database are they using?"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            raw = response.choices[0].message.content.strip()
            queries = [q.strip() for q in raw.split('|') if q.strip()]
            print(f"{Colors.SYSTEM}   [✂️ Internal HRE Thought]: Deconstructed query into {len(queries)} parts: {queries}{Colors.RESET}")
            return queries
        except Exception:
            return [user_query]

    def _update_summary(self):
        """Compresses the oldest messages into the running summary."""
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
            self.active_strategy = "Sliding Window (Fast Path)"
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
            # FIX: Use [-1:] so we ONLY pass the User's current message, avoiding 'Orphaned Assistant' Jinja errors!
            return [{"role": "system", "content": dynamic_system}] + self.raw_history[-1:]

        # Rule 3: The Narrative Path
        self.active_strategy = "Rolling Summary + Window"
        self._update_summary()
        dynamic_system = f"{system_prompt}\n\n[CONVERSATION SUMMARY]\n{self.running_summary}"
        self.overhead_time += (time.perf_counter() - start_time)
        return [{"role": "system", "content": dynamic_system}] + self.raw_history

class HeuristicChatEngine:
    def __init__(self, model_id: str):
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.model_id = model_id
        self.memory = DynamicMemoryManager(model_id, self.client)
        self.system_prompt = "You are Jarvis. Answer concisely."

    def chat_turn(self, user_input: str, turn_number: int):
        print(f"\n{Colors.USER}[Turn {turn_number}] User: {user_input}{Colors.RESET}")
        
        # Save the user input to memory FIRST so it is available for get_messages
        self.memory.add_message("user", user_input)
        
        # Get the formatted payload for the LLM
        messages_to_send = self.memory.get_messages(self.system_prompt, user_input, turn_number)

        print(f"{Colors.ROUTER}[HRE ROUTER]: Selected Strategy ➔ {self.memory.active_strategy}{Colors.RESET}")
        print(f"{Colors.JARVIS}Jarvis: ", end="", flush=True)

        start_time = time.perf_counter()
        ttft = 0.0
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

            self.memory.add_message("assistant", full_response)
            
            total_ttft = ttft + self.memory.overhead_time
            print(f"{Colors.RESET}\n   {Colors.METRICS}-> [HRE Compute Overhead: {self.memory.overhead_time:.2f}s | TTFT: {ttft:.2f}s | Total User Wait: {total_ttft:.2f}s]{Colors.RESET}")

        except Exception as e:
            print(f"\n{Colors.ERROR}[Error]: {e}{Colors.RESET}")

if __name__ == "__main__":
    test_conversation = [
        "Hi, my name is Christopher and I'm building a Python application.",
        "It is a RAG pipeline utilizing Qdrant for vector storage.",
        "Can you explain the primary benefit of microservices?", 
        "What is the CAP theorem?", 
        "What was my name again, and what specific database am I using for my project?" 
    ]

    target_model = "mistralai/ministral-3-3b" 

    print(f"{Colors.SYSTEM}{'='*70}")
    print("🧠 JARVIS DYNAMIC MEMORY MANAGER (HRE) 🧠")
    print(f"{'='*70}{Colors.RESET}")
    try:
        engine = HeuristicChatEngine(model_id=target_model)
        
        for i, query in enumerate(test_conversation, 1):
            engine.chat_turn(query, turn_number=i)
    finally:
            print(f"{'='*70}{Colors.RESET}")
    