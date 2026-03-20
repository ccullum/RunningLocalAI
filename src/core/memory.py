import os
import uuid
import warnings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from .brain import JarvisBrain
from core.colors import Colors

class JarvisMemory:
    def __init__(self, model_id="local-model"):
        print("[System] Initializing HRE Memory Manager...")
        self.brain = JarvisBrain(model_id=model_id)
        
        # We use the Brain's LM Studio client to keep embeddings 100% local!
        self.embed_model = "text-embedding-nomic-embed-text-v1.5@q8_0"
        
        self.raw_history = []
        self.running_summary = ""
        self.turn_count = 0
        
        # Dynamically locate the centralized ../data/ folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "..", "data")
        qdrant_path = os.path.join(data_dir, "qdrant_storage")
        
        # Ensure the data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        self.qdrant = QdrantClient(path=qdrant_path)
        self.collection = "jarvis_memory"
        
        if not self.qdrant.collection_exists(self.collection):
            print(f"[System] Creating persistent memory database at {qdrant_path}...")
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        else:
            print("[System] Persistent memory loaded successfully.")

    def add_user_message(self, content: str):
        self.turn_count += 1
        self._embed_and_save("User", content)
        self.raw_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self._embed_and_save("Jarvis", content)
        self.raw_history.append({"role": "assistant", "content": content})

    def _embed_and_save(self, role: str, content: str):
        if not content.strip(): return
        try:
            # Route to LM Studio instead of Hugging Face!
            vector = self.brain.client.embeddings.create(input=content, model=self.embed_model).data[0].embedding
            self.qdrant.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": content, "role": role})]
            )
        except Exception as e:
            print(f"{Colors.ERROR}[Memory DB Error: {e}]{Colors.RESET}")

    def _route_intent(self, user_query: str) -> str:
        prompt = f"""You are a strict internal routing agent. Analyze the user's query: "{user_query}"
        1. RECALL: Use ONLY if the user is asking to remember a specific detail mentioned EARLIER IN THIS CONVERSATION.
        2. SUMMARY: Use ONLY if the user is asking for a broad recap of the chat.
        3. CHAT: Use for EVERYTHING ELSE.
        Respond with EXACTLY ONE WORD: [RECALL, SUMMARY, or CHAT]."""
        
        intent = self.brain.process_background_task(prompt, max_tokens=5).upper()
        if "RECALL" in intent: return "RECALL"
        if "SUMMARY" in intent: return "SUMMARY"
        return "CHAT"

    def _deconstruct_query(self, user_query: str) -> list:
        prompt = f"""Split this query into distinct, simple search questions separated by a pipe '|'.
        Query: "{user_query}"
        Example Output: What is the user's name? | What database are they using?"""
        raw = self.brain.process_background_task(prompt, max_tokens=50)
        return [q.strip() for q in raw.split('|') if q.strip()] if raw else [user_query]

    def _update_summary(self):
        if len(self.raw_history) <= 4: return 
        oldest_user = self.raw_history.pop(0)
        oldest_assistant = self.raw_history.pop(0)
        
        prompt = f"""Update this summary with the new facts. Be incredibly brief.
        Current: {self.running_summary}
        New: User said "{oldest_user['content']}", Assistant replied "{oldest_assistant['content']}" """
        
        self.running_summary = self.brain.process_background_task(prompt, max_tokens=100)

    def get_context_payload(self, user_query: str):
        """The HRE Router: Returns the optimized prompt payload."""
        system_prompt = "You are JARVIS, a highly intelligent and concise AI software architecture assistant."
        
        # Rule 1: The Fast Path
        if self.turn_count <= 3:
            print("{Colors.ROUTER}[HRE ROUTER]: Sliding Window (Fast Path){Colors.RESET}")
            return [{"role": "system", "content": system_prompt}] + self.raw_history

        intent = self._route_intent(user_query)

        # Rule 2: The Deep Search Path
        if intent == "RECALL":
            print("{Colors.ROUTER}[HRE ROUTER]: Vector RAG (Deconstructed){Colors.RESET}")
            sub_queries = self._deconstruct_query(user_query)
            context_pieces = []
            
            for sq in sub_queries:
                try:
                    vector = self.embeddings.embed_query(sq)
                    results = self.qdrant.query_points(collection_name=self.collection, query=vector, limit=2)
                    context_pieces.extend([p.payload['text'] for p in results.points if p.score >= 0.5])
                except Exception:
                    pass
                    
            if context_pieces:
                unique_context = "\n".join(list(set(context_pieces)))
                system_prompt += f"\n\n[RELEVANT PAST CONTEXT]\n{unique_context}"
            
            return [{"role": "system", "content": system_prompt}] + self.raw_history[-1:]

        # Rule 3: The Narrative Path
        print("{Colors.ROUTER}[HRE ROUTER]: Rolling Summary + Window{Colors.RESET}")
        self._update_summary()
        if self.running_summary:
            system_prompt += f"\n\n[CONVERSATION SUMMARY]\n{self.running_summary}"
            
        return [{"role": "system", "content": system_prompt}] + self.raw_history

    def close_database(self):
        self.qdrant.close()