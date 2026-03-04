import time
import uuid
from openai import OpenAI
from qdrant_client import QdrantClient

# --- 1. The Embedding & Retrieval Layer ---

class EmbeddingEngine:
    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = "text-embedding-nomic-embed-text-v1.5@q8_0"

    def get_embedding(self, text: str):
        try:
            response = self.client.embeddings.create(input=text, model=self.model_id)
            return response.data[0].embedding
        except Exception as e:
            print(f"\n[Error] Embedding failed: {e}")
            return None

class QdrantRetrieval:
    def __init__(self, db_path="./qdrant_data"):
        self.client = QdrantClient(path=db_path)
        self.collection_name = "jarvis_knowledge"
        
    def retrieve_context(self, query_vector, limit=2, threshold=0.5):
        """Retrieves chunks, filtering out anything below the similarity threshold."""
        start_time = time.perf_counter()
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        
        valid_chunks = []
        print("\n  [Retrieval Diagnostics]")
        for i, point in enumerate(response.points):
            # Log the score and whether it passed the gate
            if point.score >= threshold:
                print(f"    - Rank {i+1}: Score {point.score:.4f} [✅ ACCEPTED]")
                valid_chunks.append(point.payload['text'])
            else:
                print(f"    - Rank {i+1}: Score {point.score:.4f} [❌ REJECTED - Below {threshold} Threshold]")
                
        elapsed = time.perf_counter() - start_time
        
        # Join chunks with newlines for the prompt
        context_string = "\n---\n".join(valid_chunks) if valid_chunks else None
        return context_string, elapsed

# --- 2. The Generation Layer ---

class RAGStreamingModel:
    def __init__(self, model_id: str, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        
        # RAG requires lower temperatures to prevent hallucination
        self.temperature = 0.3 

    def stream_rag_query(self, query: str, context: str = None):
        """Builds the RAG prompt and streams the response."""
        
        nonce = f" [RID:{uuid.uuid4().hex[:6]}]"
        
        if context:
            # The Grounded JARVIS Prompt
            system_prompt = (
                "You are Jarvis, a highly efficient, sophisticated software assistant. "
                "Answer the user's question using ONLY the context provided below. "
                "If the answer is not contained in the context, politely state that you "
                "do not have that information in your current database. Do not guess."
                f"\n\nCONTEXT:\n{context}"
                f"{nonce}"
            )
        else:
            # Fallback if no relevant context was found
            system_prompt = (
                "You are Jarvis. The user asked a question, but no relevant documents "
                "were found in your database. Inform the user of this gracefully."
                f"{nonce}"
            )

        try:
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=self.temperature,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"\n[Architect Error]: LM Studio Connection failed. Details: {str(e)}"

# --- 3. The Orchestration Layer ---

def run_rag_pipeline():
    print("\n" + "="*70)
    print("🧠 JARVIS RAG ORCHESTRATOR 🧠")
    print("="*70)

    # We will use Ministral for the generation test
    target_llm = "mistralai/ministral-3-3b"
    query = "How many continents are there?"
    #query = "What is the primary benefit of a microservices architecture?"
    
    print(f"User Query: '{query}'")
    
    # Step 1: Embed
    embedder = EmbeddingEngine()
    query_vector = embedder.get_embedding(query)
    
    if not query_vector:
        return

    # Step 2: Retrieve
    retriever = QdrantRetrieval()
    print("🔍 Searching Local Qdrant Vector DB...")
    context, retrieval_time = retriever.retrieve_context(query_vector, limit=2, threshold=0.6)
    
    if context:
        print(f"✅ Context found and loaded in {retrieval_time:.4f}s.")
    else:
        print(f"⚠️ No context met the similarity threshold. Proceeding with fallback.")

    # Step 3: Generate
    print(f"⚙️ Streaming response from {target_llm}...\n")
    print("-" * 50)
    
    llm = RAGStreamingModel(model_id=target_llm)
    
    ttft_captured = False
    start_time = time.perf_counter()
    
    for chunk in llm.stream_rag_query(query=query, context=context):
        if not ttft_captured:
            ttft = time.perf_counter() - start_time
            print(f"[TTFT: {ttft:.2f}s] ", end="", flush=True)
            ttft_captured = True
            
        print(chunk, end="", flush=True)
        
    print("\n" + "-" * 50)
    print("="*70 + "\n")

if __name__ == "__main__":
    run_rag_pipeline()