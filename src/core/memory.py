import uuid
import time
import threading
import warnings

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from utils.metrics import telemetry
from .brain import LocalStreamBrain
from .colors import Colors
from .config import Config
from .semantic_router import SemanticRouter

# Suppress annoying Pydantic/Qdrant warnings in the terminal
warnings.filterwarnings("ignore", category=UserWarning)

class AsyncMemory:
    def __init__(self):
        print(f"{Colors.SYSTEM}[System] Initializing HRE Memory Manager...{Colors.RESET}")
        self.brain = LocalStreamBrain() # No need to pass model_id anymore
        
        # Use config variables
        self.embed_model = Config.EMBED_MODEL
        self.collection = Config.COLLECTION_NAME
        self.qdrant = QdrantClient(path=Config.QDRANT_STORAGE_PATH)
        
        if not self.qdrant.collection_exists(self.collection):
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            
        print(f"{Colors.SYSTEM}[System] Persistent memory loaded successfully.{Colors.RESET}")
        
        self.raw_history = []
        self.turn_count = 0
        self.running_summary = ""
        self.semantic_router = SemanticRouter(self.brain.client, self.embed_model)

    def add_user_message(self, user_input: str):
        """Adds to short-term history and selectively saves facts to long-term memory."""
        self.raw_history.append({"role": "user", "content": user_input})
        self.turn_count += 1
        
        # THE MEMORY FILTER (Bouncer)
        query_lower = user_input.lower().strip()
        is_question = "?" in user_input or any(query_lower.startswith(q) for q in 
            ["what", "who", "where", "when", "why", "how", "do ", "does ", "did ", "is ", "are ", "can ", "could "])
        
        is_command = any(query_lower.startswith(c) for c in 
            ["please", "summarize", "tell me", "give me", "can you", "could you"])
        
        if is_question or is_command:
            print(f"{Colors.MEMORY}[Memory Filter]: Ignored question/command. Not saving to long-term DB.{Colors.RESET}")
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
    
    def save_document_chunks(self, filename: str, chunks: list) -> bool:
        """Takes pre-sliced chunks, embeds them, and saves them to Qdrant."""
        print(f"{Colors.SYSTEM}[Memory Manager]: Embedding {len(chunks)} chunks from '{filename}'...{Colors.RESET}")
        
        telemetry.start("Vector Embedding Time")

        points = []
        current_time = int(time.time())
        
        for i, chunk in enumerate(chunks):
            clean_chunk = chunk.replace('\n', ' ').strip()
            if len(clean_chunk) < Config.MIN_CHUNK_CHARACTER_COUNT: 
                continue
                
            try:
                # Ask Nomic for the math coordinates
                vector = self.brain.client.embeddings.create(input=clean_chunk, model=self.embed_model).data[0].embedding
                
                payload = {
                    "text": clean_chunk,
                    "source_file": filename,
                    "chunk_id": i,
                    "retrieval_count": 1,
                    "created_at": current_time,
                    "last_accessed": current_time
                }
                
                points.append(PointStruct(id=uuid.uuid4().hex, vector=vector, payload=payload))
            except Exception as e:
                print(f"{Colors.ERROR}[Embedding Error on chunk {i}: {e}]{Colors.RESET}")
        
        telemetry.record_value("Total Chunks Embedded", len(points))
        telemetry.stop("Vector Embedding Time")

        if points:
            self.qdrant.upsert(collection_name=self.collection, points=points)
            print(f"{Colors.MEMORY}[Memory Manager]: Successfully saved {len(points)} vectors to Qdrant.{Colors.RESET}")
            return True
        return False

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

    @telemetry.measure("Router [LLM 8B]")
    def _route_intent_llm(self, user_query: str) -> str:        # 1. The Architect's Override (Fast Keyword Heuristics)
        query_lower = user_query.lower()
        recall_triggers = ["what is my", "what's my", "do you remember", "did i say", "did i tell", "what was my"]
        
        if any(trigger in query_lower for trigger in recall_triggers):
            print(f"{Colors.WARNING}[ROUTER OVERRIDE]: Keyword triggered RECALL{Colors.RESET}")
            return "RECALL"

        # 2. The LLM Fallback
        prompt = Config.LLM_FALLBACK_QUERY_TEMPLATE.format(user_query=user_query)
        
        raw_intent = self.brain.process_background_task(prompt, max_tokens=Config.LLM_ROUTING_MAX_TOKENS).upper()
        print(f"{Colors.SYSTEM}[LLM Router Output: {raw_intent.strip()}]{Colors.RESET}") 
        
        if "RECALL" in raw_intent: return "RECALL"
        if "SUMMARY" in raw_intent: return "SUMMARY"
        return "CHAT"

    @telemetry.measure("Router [Semantic Math]")
    def _route_intent_semantic(self, user_query: str) -> str:
        # We don't need keyword heuristics anymore, the math handles it all!
        return self.semantic_router.route(user_query)
    
    def _route_intent(self, user_query: str) -> str:
        """Dispatcher: Checks the config flag and fires the appropriate routing engine."""
        if Config.USE_SEMANTIC_ROUTER:
            return self._route_intent_semantic(user_query)
        else:
            return self._route_intent_llm(user_query)
    
    def _deconstruct_query(self, user_query: str) -> list:
        # Pull the template from Config and inject the user_query
        prompt = Config.DECONSTRUCT_QUERY_TEMPLATE.format(user_query=user_query)
        
        raw_output = self.brain.process_background_task(prompt, max_tokens=15)
        cleaned_output = raw_output.replace('"', '').replace('*', '').strip()
        
        if "SKIP" in cleaned_output.upper() or len(cleaned_output) > 50:
            return []
        
        return [cleaned_output]

    def _update_summary(self):
        if self.turn_count % Config.SUMMARY_TRIGGER_TURN_COUNT == 0 and len(self.raw_history) >= Config.SUMMARY_TRIGGER_TURN_COUNT:
            prompt = Config.UPDATE_SUMMARY_PROMPT
            self.running_summary = self.brain.process_background_task(prompt, max_tokens=Config.LLM_SUMMARY_MAX_TOKENS)

    def get_context_payload(self, user_query: str):
        # Pull the base persona and rules from the Control Room
        system_prompt = Config.SYSTEM_PROMPT
        
        # Let the Semantic Router do its job!
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
                    results = self.qdrant.query_points(collection_name=self.collection, 
                                                       query=vector, 
                                                       limit=Config.VECTOR_SEARCH_LIMIT)
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
                    
                    # 2. Calculate Decay 
                    decay_multiplier = max(
                        Config.MEMORY_DECAY_FLOOR, 
                        1.0 - (age_in_days * Config.MEMORY_DECAY_RATE)
                    )
                    
                    # 3. The Stanford Cognitive Equation
                    final_score = base_score * (1 + (Config.MEMORY_REINFORCE_WEIGHT * freq_weight)) * decay_multiplier
                    
                    scored_memories[p.id] = {
                        "text": p.payload.get('text', ''),
                        "final_score": final_score,
                        "point_id": p.id
                    }
            
            sorted_memories = sorted(scored_memories.values(), key=lambda x: x["final_score"], reverse=True)
            
            context_pieces = []
            used_point_ids = []
            
            for mem in sorted_memories[:Config.CONTEXT_CHUNKS_LIMIT]:
                print(f"{Colors.MEMORY}[Path Weight {mem['final_score']:.2f}]: {mem['text']}{Colors.RESET}")
                context_pieces.append(mem['text'])
                used_point_ids.append(mem['point_id'])
                
            if context_pieces:
                unique_context = "\n".join(list(set(context_pieces)))
                system_prompt += Config.CONTEXT_INJECTION_TEMPLATE.format(context=unique_context)
                
                # FIRE THE GHOST THREAD!
                threading.Thread(target=self._reinforce_memories_background, args=(used_point_ids,), daemon=True).start()
            else:
                print(f"{Colors.WARNING}[No relevant memory paths found]{Colors.RESET}")
            
            return [{"role": "system", "content": system_prompt}] + self.raw_history[-1:]

        print(f"{Colors.ROUTER}[HRE ROUTER]: Rolling Summary + Window{Colors.RESET}")
        self._update_summary()
        if self.running_summary:
            system_prompt += Config.SUMMARY_INJECTION_TEMPLATE.format(summary=self.running_summary)
            
        return [{"role": "system", "content": system_prompt}] + self.raw_history