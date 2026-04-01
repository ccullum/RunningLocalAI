import math
from .colors import Colors
from .config import Config
from utils.metrics import telemetry

class SemanticRouter:
    def __init__(self, embed_client, embed_model):
        print(f"{Colors.SYSTEM}[Router] Initializing Algorithmic Semantic Router...{Colors.RESET}")
        self.embed_client = embed_client
        self.embed_model = embed_model
        
        # Dictionary to hold the pre-computed mathematical coordinates of our anchors
        self.anchor_vectors = {
            "RECALL": [],
            "SUMMARY": []
        }
        
        self._precompute_anchors()

    @telemetry.measure("Anchor Pre-computation Time")
    def _precompute_anchors(self):
        """Embeds the anchor phrases into math upon boot so we don't waste time later."""
        print(f"{Colors.SYSTEM}[Router] Pre-computing anchor vectors...{Colors.RESET}")
        try:
            for text in Config.ROUTER_RECALL_ANCHORS:
                vec = self.embed_client.embeddings.create(input=text, model=self.embed_model).data[0].embedding
                self.anchor_vectors["RECALL"].append(vec)
                
            for text in Config.ROUTER_SUMMARY_ANCHORS:
                vec = self.embed_client.embeddings.create(input=text, model=self.embed_model).data[0].embedding
                self.anchor_vectors["SUMMARY"].append(vec)
                
            print(f"{Colors.SYSTEM}[Router] Successfully mapped semantic space.{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.ERROR}[Router Initialization Error: {e}]{Colors.RESET}")

    def _cosine_similarity(self, vec1, vec2):
        """The core mathematical equation for comparing two ideas."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    @telemetry.measure("Semantic Routing Math & Embed Time")
    def route(self, user_query: str) -> str:
        """Embeds the user's query and mathematically finds the closest intent."""
        try:
            query_vec = self.embed_client.embeddings.create(input=user_query, model=self.embed_model).data[0].embedding
        except Exception as e:
            print(f"{Colors.ERROR}[Router Embedding Error: {e}]{Colors.RESET}")
            return "CHAT" # Safe fallback

        best_intent = "CHAT"
        highest_score = 0.0

        # Calculate distance against all pre-computed anchors
        for intent, vectors in self.anchor_vectors.items():
            for anchor_vec in vectors:
                score = self._cosine_similarity(query_vec, anchor_vec)
                if score > highest_score:
                    highest_score = score
                    best_intent = intent

        print(f"{Colors.ROUTER}[Semantic Router]: Best Match = {best_intent} (Score: {highest_score:.2f}){Colors.RESET}")

        # If it's too ambiguous, default to standard chatting
        if highest_score >= Config.ROUTER_CONFIDENCE_THRESHOLD:
            return best_intent
        else:
            print(f"{Colors.ROUTER}[Semantic Router]: Below threshold ({Config.ROUTER_CONFIDENCE_THRESHOLD}). Falling back to CHAT.{Colors.RESET}")
            return "CHAT"