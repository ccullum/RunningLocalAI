import os
import numpy as np
from fastembed import TextEmbedding
from .colors import Colors
from .config import Config
from utils.metrics import perf_tracker

# --- IMPLEMENTATION A: LEGACY ROUTER (API-based) ---
class LegacySemanticRouter:
    """Uses external API for embeddings (LM Studio/OpenAI)."""
    def __init__(self, embed_client, embed_model):
        print(f"{Colors.SYSTEM}[Router] Initializing Legacy Semantic Router (API)...{Colors.RESET}")
        self.embed_client = embed_client
        self.embed_model = embed_model
        self.anchor_vectors = {"RECALL": [], "SUMMARY": []}
        self._precompute_anchors()

    @perf_tracker.measure("Anchor Pre-computation Time (Legacy)")
    def _precompute_anchors(self):
        for text in Config.ROUTER_RECALL_ANCHORS:
            vec = self.embed_client.embeddings.create(input=text, model=self.embed_model).data[0].embedding
            self.anchor_vectors["RECALL"].append(vec)
        for text in Config.ROUTER_SUMMARY_ANCHORS:
            vec = self.embed_client.embeddings.create(input=text, model=self.embed_model).data[0].embedding
            self.anchor_vectors["SUMMARY"].append(vec)

    def route(self, user_query: str):
        query_v = self.embed_client.embeddings.create(input=user_query, model=self.embed_model).data[0].embedding
        return self._find_best_match(query_v)

    def _find_best_match(self, query_v):
        best_intent = "CHAT"
        highest_score = 0.0
        for intent, vectors in self.anchor_vectors.items():
            for ref_v in vectors:
                # Standard Python math is fine for small anchor sets
                score = np.dot(query_v, ref_v) / (np.linalg.norm(query_v) * np.linalg.norm(ref_v))
                if score > highest_score:
                    highest_score = score
                    best_intent = intent
        return best_intent, highest_score

# --- IMPLEMENTATION B: AGNOSTIC FAST ROUTER (Local ONNX) ---
class AgnosticSemanticRouter:
    """Uses FastEmbed (ONNX) for local, platform-agnostic inference."""
    def __init__(self):
        print(f"{Colors.SYSTEM}[Router] Initializing Agnostic FastEmbed Router...{Colors.RESET}")
        
        # Ensure the cache stays inside our project data directory
        cache_path = os.path.join(Config.DATA_DIR, "fastembed_cache")
        
        # Load Nomic v1.5 (Agnostic ONNX version)
        # Note: This will download once (~150MB) to your data directory
        self.model = TextEmbedding(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            cache_dir=cache_path
        )
        
        self.anchor_vectors = {"RECALL": [], "SUMMARY": []}
        self._precompute_anchors()

    @perf_tracker.measure("Anchor Pre-computation Time (Agnostic)")
    def _precompute_anchors(self):
        """Batch-embeds the anchor phrases locally."""
        # Nomic v1.5 requires 'search_query: ' prefix for best results
        recall_vecs = list(self.model.embed([f"search_query: {t}" for t in Config.ROUTER_RECALL_ANCHORS]))
        summary_vecs = list(self.model.embed([f"search_query: {t}" for t in Config.ROUTER_SUMMARY_ANCHORS]))
        
        self.anchor_vectors["RECALL"] = recall_vecs
        self.anchor_vectors["SUMMARY"] = summary_vecs

    @perf_tracker.measure("Semantic Routing Math & Embed Time (Agnostic)")
    def route(self, user_query: str):
        # Generate embedding locally via ONNX
        query_v = list(self.model.embed([f"search_query: {user_query}"]))[0]
        
        best_intent = "CHAT"
        highest_score = 0.0

        for intent, vectors in self.anchor_vectors.items():
            for ref_v in vectors:
                # Optimized similarity via numpy
                score = np.dot(query_v, ref_v) / (np.linalg.norm(query_v) * np.linalg.norm(ref_v))
                if score > highest_score:
                    highest_score = score
                    best_intent = intent

        print(f"{Colors.ROUTER}[Agnostic Router]: Best Match = {best_intent} ({highest_score:.2f}){Colors.RESET}")
        
        if highest_score < Config.ROUTER_CONFIDENCE_THRESHOLD:
            return "CHAT", highest_score
        return best_intent, highest_score

# --- THE UNIFIED FACTORY ---
class SemanticRouter:
    def __init__(self, embed_client=None, embed_model=None):
        # We repurpose the flag to mean "Local Engine" (Now Agnostic)
        if getattr(Config, "USE_LOCAL_CPP_ROUTER", False):
            self.implementation = AgnosticSemanticRouter()
        else:
            self.implementation = LegacySemanticRouter(embed_client, embed_model)

    def route(self, user_query: str):
        return self.implementation.route(user_query)