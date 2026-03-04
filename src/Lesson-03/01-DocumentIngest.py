import os
import time
import uuid
import re
import pandas as pd
from openai import OpenAI
import lancedb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 1. The ETL Transform Phase: Smart Chunking ---

class SmartChunker:
    """Splits text into chunks while respecting sentence boundaries and overlap."""
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, text: str):
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Split by sentences (roughly, looking for punctuation followed by a space)
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding the next sentence keeps us under the limit, append it
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                # Chunk is full. Save it.
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Start a new chunk, bringing over a portion of the previous text for overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                # Try to snap the overlap to the first space to avoid cutting words in half
                space_index = overlap_text.find(' ')
                if space_index != -1:
                    overlap_text = overlap_text[space_index:].strip()
                
                current_chunk = overlap_text + " " + sentence + " "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

# --- 2. The Embedding Engine ---

class EmbeddingEngine:
    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # We define a standard default, but LM Studio will use whatever embedding model is loaded
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

# --- 3. Database Abstraction Layer ---

class LanceDBClient:
    def __init__(self, db_path="./lancedb_data"):
        self.db = lancedb.connect(db_path)
        self.table_name = "jarvis_knowledge"

    def ingest(self, records):
        print("\n[LanceDB] Ingesting vectors...")
        start_time = time.perf_counter()
        
        # LanceDB natively ingests Pandas DataFrames
        df = pd.DataFrame(records)
        
        # Overwrite table for testing purposes; in production, you'd append.
        self.table = self.db.create_table(self.table_name, data=df, mode="overwrite")
        
        elapsed = time.perf_counter() - start_time
        print(f"✅ [LanceDB] Ingested {len(records)} chunks in {elapsed:.3f}s")

class QdrantLocalClient:
    def __init__(self, db_path="./qdrant_data"):
        # Running completely locally, saving to disk
        self.client = QdrantClient(path=db_path)
        self.collection_name = "jarvis_knowledge"

    def ingest(self, records):
        print("\n[Qdrant] Ingesting vectors...")
        start_time = time.perf_counter()
        
        # Detect vector dimension dynamically from the first record
        vector_size = len(records[0]['vector'])
        
        # Recreate collection for clean testing
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        # Convert records to Qdrant PointStructs
        points = [
            PointStruct(
                id=record["id"],
                vector=record["vector"],
                payload={"text": record["text"], "source": record["source"]}
            ) for record in records
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        elapsed = time.perf_counter() - start_time
        print(f"✅ [Qdrant] Ingested {len(records)} chunks in {elapsed:.3f}s")

# --- Execution ---

def run_ingestion():
    print("="*60)
    print("🧠 JARVIS KNOWLEDGE INGESTION ENGINE 🧠")
    print("="*60)

    # 1. Setup sample document (Create a dummy text file if it doesn't exist)
    sample_file = "sample_knowledge.txt"
    if not os.path.exists(sample_file):
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("Microservices architecture is a design pattern where applications are composed of small, independent services. " * 50)
            f.write("The primary benefit of a microservices architecture is independent deployment and scaling. " * 50)
            f.write("LanceDB is an embedded database. Qdrant is a vector search engine. " * 50)

    with open(sample_file, "r", encoding="utf-8") as f:
        document_text = f.read()

    # 2. Chunk the text
    chunker = SmartChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.chunk_document(document_text)
    print(f"📄 Document split into {len(chunks)} chunks.")

    # 3. Generate Embeddings
    print("⚙️ Generating embeddings via local Nomic model...")
    embedder = EmbeddingEngine()
    records = []
    
    for i, chunk in enumerate(chunks):
        vector = embedder.get_embedding(chunk)
        if vector:
            records.append({
                "id": str(uuid.uuid4()), # LanceDB/Qdrant both accept string UUIDs well
                "vector": vector,
                "text": chunk,
                "source": sample_file
            })
            
    print(f"🔢 Successfully generated {len(records)} embeddings.")

    # 4. Ingest into databases for side-by-side comparison
    if records:
        lance_db = LanceDBClient()
        lance_db.ingest(records)

        qdrant_db = QdrantLocalClient()
        qdrant_db.ingest(records)
        
    print("\n" + "="*60)

if __name__ == "__main__":
    run_ingestion()