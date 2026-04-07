import llama_engine
import fast_router
import math
import os
from .colors import Colors
from .config import Config
from utils.metrics import perf_tracker

# --- IMPLEMENTATION A: LEGACY ROUTER (API + FAST_ROUTER MATH) ---
class LegacySemanticRouter:
    """Uses external API for embeddings but C++ for the cosine math."""
    def __init__(self, embed_client, embed_model):
        print(f"{Colors.SYSTEM}[Router] Initializing Legacy Semantic Router...{Colors.RESET}")
        self.embed_client = embed_client
        self.embed_model = embed_model
        self.anchor_vectors = {"RECALL": [], "SUMMARY": []}
        self._precompute_anchors()

    @perf_tracker.measure("Anchor Pre-computation Time (Legacy)")
    def _precompute_anchors(self):
        """Embeds the anchor phrases using external API."""
        print(f"{Colors.SYSTEM}[Router] Pre-computing legacy anchor vectors...{Colors.RESET}")
        try:
            for text in Config.ROUTER_RECALL_ANCHORS:
                # API-based embeddings often don't require specific prefixes, 
                # but we keep it consistent with how Lesson 11 worked.
                vec = self.embed_client.embeddings.create(input=text, model=self.embed_model).data[0].embedding
                self.anchor_vectors["RECALL"].append(vec)
            for text in Config.ROUTER_SUMMARY_ANCHORS:
                vec = self.embed_client.embeddings.create(input=text, model=self.embed_model).data[0].embedding
                self.anchor_vectors["SUMMARY"].append(vec)
            print(f"{Colors.SYSTEM}[Router] Successfully mapped legacy semantic space.{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.ERROR}[Legacy Router Init Error: {e}]{Colors.RESET}")

    @perf_tracker.measure("Semantic Routing Math & Embed Time (Legacy)")
    def route(self, user_query: str):
        """Calculates intent using API embeddings and fast_router math."""
        try:
            query_vec = self.embed_client.embeddings.create(input=user_query, model=self.embed_model).data[0].embedding
            best_intent = "CHAT"
            highest_score = 0.0
            for intent, vectors in self.anchor_vectors.items():
                for anchor_vec in vectors:
                    score = fast_router.cosine_similarity(query_vec, anchor_vec)
                    if score > highest_score:
                        highest_score = score
                        best_intent = intent
            
            self._log_decision(best_intent, highest_score)
            
            # Return tuple to satisfy test script
            if highest_score >= Config.ROUTER_CONFIDENCE_THRESHOLD:
                return best_intent, highest_score
            return "CHAT", highest_score
        except Exception:
            return "CHAT", 0.0

    def _log_decision(self, intent, score):
        print(f"{Colors.ROUTER}[Legacy Router]: Best Match = {intent} (Score: {score:.2f}){Colors.RESET}")
        perf_tracker.record_value("Legacy Router Decision", f"{intent} ({score:.2f})")


# --- IMPLEMENTATION B: NEW LOCAL ROUTER (ALL C++) ---
class LocalSemanticRouter:
    """Uses the bare-metal C++ llama_engine for both embeddings and math."""
    def __init__(self):
        print(f"{Colors.SYSTEM}[Router] Initializing Local C++ Embedding Engine...{Colors.RESET}")
        if not Config.NOMIC_MODEL_PATH:
            raise ValueError("NOMIC_MODEL_PATH not set in Config.")
        
        self.engine = llama_engine.EmbeddingEngine(Config.NOMIC_MODEL_PATH)
        self.anchor_vectors = {"RECALL": [], "SUMMARY": []}
        self._precompute_anchors()

    @perf_tracker.measure("Anchor Pre-computation Time (Local C++)")
    def _precompute_anchors(self):
        """Embeds the anchor phrases locally in C++ using 'search_document' prefix."""
        print(f"{Colors.SYSTEM}[Router] Pre-embedding anchors locally in C++...{Colors.RESET}")
        # Nomic v1.5 requires 'search_document:' for the reference data (anchors)
        for text in Config.ROUTER_RECALL_ANCHORS:
            self.anchor_vectors["RECALL"].append(self.engine.generate_embedding(f"search_document: {text}"))
        for text in Config.ROUTER_SUMMARY_ANCHORS:
            self.anchor_vectors["SUMMARY"].append(self.engine.generate_embedding(f"search_document: {text}"))
        print(f"{Colors.SYSTEM}[Router] Local semantic space mapping complete.{Colors.RESET}")

    @perf_tracker.measure("Semantic Routing Math & Embed Time (Local C++)")
    def route(self, user_query: str):
        """Calculates intent entirely within the C++ layer using 'search_query' prefix."""
        # Nomic v1.5 requires 'search_query:' for the user's incoming query
        query_v = self.engine.generate_embedding(f"search_query: {user_query}")
        best_intent = "CHAT"
        highest_score = 0.0

        for intent, vectors in self.anchor_vectors.items():
            for ref_v in vectors:
                score = self.engine.similarity(query_v, ref_v)
                if score > highest_score:
                    highest_score = score
                    best_intent = intent

        self._log_decision(best_intent, highest_score)
        
        # Return tuple to satisfy test script
        if highest_score < Config.ROUTER_CONFIDENCE_THRESHOLD:
            return "CHAT", highest_score
        return best_intent, highest_score

    def _log_decision(self, intent, score):
        print(f"{Colors.ROUTER}[Local Router]: Best Match = {intent} (Score: {score:.2f}){Colors.RESET}")
        perf_tracker.record_value("Local Router Decision", f"{intent} ({score:.2f})")


# --- THE MAIN WRAPPER ---
class SemanticRouter:
    """Decides implementation based on Config.USE_LOCAL_CPP_ROUTER."""
    def __init__(self, embed_client=None, embed_model=None):
        if getattr(Config, "USE_LOCAL_CPP_ROUTER", False):
            self.implementation = LocalSemanticRouter()
        else:
            self.implementation = LegacySemanticRouter(embed_client, embed_model)

    def route(self, user_query: str):
        # Delegate routing and return the (intent, score) tuple
        return self.implementation.route(user_query)