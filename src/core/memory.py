import os
import uuid
import time
import threading
import warnings

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from .brain import JarvisBrain
from .colors import Colors

# Suppress annoying Pydantic/Qdrant warnings in the terminal
warnings.filterwarnings("ignore", category=UserWarning)

class JarvisMemory:
    def __init__(self, model_id: str = "local-model"):
        print(f"{Colors.SYSTEM}[System] Initializing HRE Memory Manager...{Colors.RESET}")
        self.brain = JarvisBrain(model_id=model_id)
        self.embed_model = "text-embedding-nomic-embed-text-v1.5@q8_0"
        
        # Setup Paths dynamically
        core_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(core_dir, "..", "..", "data"))
        qdrant_path = os.path.join(data_dir, "qdrant_storage")
        
        self.collection = "jarvis_memory"
        self.qdrant = QdrantClient(path=qdrant_path)
        
        # Ensure collection exists (using 768 dimensions for Nomic text embeddings)
        if not self.qdrant.collection_exists(self.collection):
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            
        print(f"{Colors.SYSTEM}[System] Persistent memory loaded successfully.{Colors.RESET}")
        
        self.raw_history = []
        self.turn_count = 0
        self.running_summary = ""

    def add_user_message(self, user_input: str):
        """Adds to short-term history and selectively saves facts to long-term memory."""
        self.raw_history.append({"role": "user", "content": user_input})
        self.turn_count += 1
        
        # THE MEMORY FILTER (Bouncer)
        query_lower = user_input.lower().strip()
        is_question = "?" in user_input or any(query_lower.startswith(q) for q in 
            ["what", "who", "where", "when", "why", "how", "do ", "does ", "did ", "is ", "are ", "can ", "could "])
        
        if is_question:
            print(f"{Colors.MEMORY}[Memory Filter]: Ignored question. Not saving to long-term DB.{Colors.RESET}")
            return
            
        try:
            print(f"{Colors.MEMORY}[Memory Manager]: Embedding fact into long-term storage...{Colors.RESET}")
            vector = self.brain.client.embeddings.create(input=user_input, model=self.embed_model).data[0].embedding
            
            # THE FWA PAYLOAD
            current_time = int(time.time())
            payload = {
                "text": user_input,
                "retrieval_count": 1,
                "created_at": current_time,
                "last_accessed": current_time
            }
            
            self.qdrant.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=uuid.uuid4().hex, vector=vector, payload=payload)]
            )
        except Exception as e:
            print(f"{Colors.ERROR}[Memory Save Error: {e}]{Colors.RESET}")

    def add_assistant_message(self, assistant_response: str):
        self.raw_history.append({"role": "assistant", "content": assistant_response})
        self.turn_count += 1

    def _reinforce_memories_background(self, point_ids: list):
        """Runs in a background thread to strengthen the memory paths without lagging JARVIS."""
        try:
            points = self.qdrant.retrieve(collection_name=self.collection, ids=point_ids)
            for p in points:
                new_count = p.payload.get("retrieval_count", 1) + 1
                p.payload["retrieval_count"] = new_count
                p.payload["last_accessed"] = int(time.time())
                
                self.qdrant.set_payload(
                    collection_name=self.collection,
                    payload=p.payload,
                    points=[p.id]
                )
            print(f"{Colors.SYSTEM}[Background Thread]: Reinforced {len(point_ids)} memory paths.{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.ERROR}[Background Reinforcement Error: {e}]{Colors.RESET}")

    def _route_intent(self, user_query: str) -> str:
        # 1. The Architect's Override (Fast Keyword Heuristics)
        query_lower = user_query.lower()
        recall_triggers = ["what is my", "what's my", "do you remember", "did i say", "did i tell", "what was my"]
        
        if any(trigger in query_lower for trigger in recall_triggers):
            print(f"{Colors.WARNING}[ROUTER OVERRIDE]: Keyword triggered RECALL{Colors.RESET}")
            return "RECALL"

        # 2. The LLM Fallback
        prompt = f"""You are a strict internal routing agent. Analyze the user's query: "{user_query}"
        1. RECALL: Use if the user asks to remember a specific detail, fact, or something mentioned in the past.
        2. SUMMARY: Use ONLY if the user is asking for a broad recap of the chat.
        3. CHAT: Use for EVERYTHING ELSE.
        Respond with EXACTLY ONE WORD: [RECALL, SUMMARY, or CHAT]."""
        
        raw_intent = self.brain.process_background_task(prompt, max_tokens=10).upper()
        print(f"{Colors.SYSTEM}[LLM Router Output: {raw_intent.strip()}]{Colors.RESET}") 
        
        if "RECALL" in raw_intent: return "RECALL"
        if "SUMMARY" in raw_intent: return "SUMMARY"
        return "CHAT"

    def _deconstruct_query(self, user_query: str) -> list:
        prompt = f"""You are a database query generator. 
        Convert the following question into a declarative statement to search a database.
        Example: "What is my name?" -> "The user's name is"
        If the input is NOT a question (e.g., "Thank you", "Yes it is"), respond with exactly the word: SKIP.
        Question: "{user_query}"
        RESPOND WITH THE RAW STATEMENT ONLY. NO CONVERSATIONAL FILLER."""
        
        raw_output = self.brain.process_background_task(prompt, max_tokens=15)
        cleaned_output = raw_output.replace('"', '').replace('*', '').strip()
        
        if "SKIP" in cleaned_output.upper() or len(cleaned_output) > 50:
            return []
        
        return [cleaned_output]

    def _update_summary(self):
        if self.turn_count % 5 == 0 and len(self.raw_history) >= 5:
            prompt = "Summarize the key points of the conversation so far in one short paragraph."
            self.running_summary = self.brain.process_background_task(prompt, max_tokens=100)

    def get_context_payload(self, user_query: str):
        # The Pronoun Primer
        system_prompt = (
            "You are JARVIS, a highly intelligent and concise AI. "
            "When the user says 'I', 'me', or 'my', they are referring to themselves. "
            "Use the provided context to answer their questions about themselves. "
            "If the answer is not in the context below, DO NOT guess. Say you don't know."
        )
        
        intent = self._route_intent(user_query)

        if intent == "RECALL":
            print(f"{Colors.ROUTER}[HRE ROUTER]: FWA Vector RAG{Colors.RESET}")
            
            search_queries = [user_query]
            try:
                deconstructed = self._deconstruct_query(user_query)
                if isinstance(deconstructed, list):
                    search_queries.extend(deconstructed)
            except Exception:
                pass 
                
            raw_results = []
            for sq in search_queries:
                try:
                    vector = self.brain.client.embeddings.create(input=sq, model=self.embed_model).data[0].embedding
                    results = self.qdrant.query_points(collection_name=self.collection, query=vector, limit=5)
                    raw_results.extend(results.points)
                except Exception as e:
                    pass
            
            # THE FWA SCORING ENGINE
            scored_memories = {}
            current_time = int(time.time()) # Get the exact time right now
            
            for p in raw_results:
                if p.id not in scored_memories:
                    base_score = p.score
                    freq_weight = p.payload.get("retrieval_count", 1) 
                    
                    # 1. Calculate Age (How long since it was last accessed?)
                    last_accessed = p.payload.get("last_accessed", current_time)
                    age_in_seconds = current_time - last_accessed
                    age_in_days = age_in_seconds / 86400.0 # 86400 seconds in a day
                    
                    # 2. Calculate Decay (Loses 1% power per day ignored, max 50% penalty)
                    decay_multiplier = max(0.5, 1.0 - (age_in_days * 0.01))
                    
                    # 3. The Stanford Cognitive Equation
                    final_score = base_score * (1 + (0.1 * freq_weight)) * decay_multiplier
                    
                    scored_memories[p.id] = {
                        "text": p.payload.get('text', ''),
                        "final_score": final_score,
                        "point_id": p.id
                    }
            
            sorted_memories = sorted(scored_memories.values(), key=lambda x: x["final_score"], reverse=True)
            
            context_pieces = []
            used_point_ids = []
            
            for mem in sorted_memories[:3]:
                print(f"{Colors.MEMORY}[Path Weight {mem['final_score']:.2f}]: {mem['text']}{Colors.RESET}")
                context_pieces.append(mem['text'])
                used_point_ids.append(mem['point_id'])
                
            if context_pieces:
                unique_context = "\n".join(list(set(context_pieces)))
                system_prompt += f"\n\n[FACTS AND CONTEXT ABOUT THE USER]\n{unique_context}"
                
                # FIRE THE GHOST THREAD!
                threading.Thread(target=self._reinforce_memories_background, args=(used_point_ids,), daemon=True).start()
            else:
                print(f"{Colors.WARNING}[No relevant memory paths found]{Colors.RESET}")
            
            return [{"role": "system", "content": system_prompt}] + self.raw_history[-1:]

        print(f"{Colors.ROUTER}[HRE ROUTER]: Rolling Summary + Window{Colors.RESET}")
        self._update_summary()
        if self.running_summary:
            system_prompt += f"\n\n[CONVERSATION SUMMARY]\n{self.running_summary}"
            
        return [{"role": "system", "content": system_prompt}] + self.raw_history