import time
from openai import OpenAI
import lancedb
from qdrant_client import QdrantClient

# --- 1. The Query Embedding Engine ---
# (Reused from our ETL pipeline to ensure mathematical consistency)

class EmbeddingEngine:
    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = "text-embedding-nomic-embed-text-v1.5@q8_0"

    def get_embedding(self, text: str):
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_id
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding Error: {e}")
            return None

# --- 2. Database Retrieval Clients ---

class LanceDBClient:
    def __init__(self, db_path="./lancedb_data"):
        self.db = lancedb.connect(db_path)
        self.table_name = "jarvis_knowledge"

    def search(self, query_vector, limit=3):
        start_time = time.perf_counter()
        
        # LanceDB search syntax
        table = self.db.open_table(self.table_name)
        results = table.search(query_vector).limit(limit).to_pandas()
        
        elapsed = time.perf_counter() - start_time
        return results, elapsed

class QdrantLocalClient:
    def __init__(self, db_path="./qdrant_data"):
        self.client = QdrantClient(path=db_path)
        self.collection_name = "jarvis_knowledge"

    def search(self, query_vector, limit=3):
        start_time = time.perf_counter()
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector, 
            limit=limit
        )
        
        elapsed = time.perf_counter() - start_time
        # CRITICAL FIX: Add .points here to return the list, not the Pydantic model
        return response.points, elapsed

# --- Execution ---

def run_retrieval_test():
    print("="*60)
    print("🔍 JARVIS VECTOR RETRIEVAL TEST 🔍")
    print("="*60)

    # 1. Define the test query based on our ingested dummy data
    query = "What is the primary benefit of a microservices architecture?"
    print(f"User Query: '{query}'\n")

    # 2. Embed the query
    print("⚙️ Embedding query via Nomic...")
    embedder = EmbeddingEngine()
    query_vector = embedder.get_embedding(query)

    if not query_vector:
        print("Failed to embed query. Exiting.")
        return

    # 3. Race the Databases
    TOP_K = 2 # How many chunks we want to retrieve

    # --- LANCEDB TEST ---
    print("\n[LANCEDB RESULTS]")
    lance_db = LanceDBClient()
    try:
        lance_results, lance_time = lance_db.search(query_vector, limit=TOP_K)
        print(f"⏱️  Retrieval Latency: {lance_time:.4f}s")
        for index, row in lance_results.iterrows():
            # LanceDB returns an '_distance' column representing vector similarity
            print(f"  - Distance: {row['_distance']:.4f} | Text: {row['text'][:80]}...")
    except Exception as e:
        print(f"❌ LanceDB Error: {e}")

    # --- QDRANT TEST ---
    print("\n[QDRANT RESULTS]")
    qdrant_db = QdrantLocalClient()
    try:
        qdrant_results, qdrant_time = qdrant_db.search(query_vector, limit=TOP_K)
        print(f"⏱️  Retrieval Latency: {qdrant_time:.4f}s")
        for hit in qdrant_results:
            # Qdrant returns a 'score' (closer to 1.0 is better for Cosine)
            print(f"  - Score: {hit.score:.4f} | Text: {hit.payload['text'][:80]}...")
    except Exception as e:
        print(f"❌ Qdrant Error: {e}")

    print("\n" + "="*60)

if __name__ == "__main__":
    run_retrieval_test()