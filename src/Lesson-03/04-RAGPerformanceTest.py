import time
import uuid
import csv
import os
from datetime import datetime
from openai import OpenAI
from qdrant_client import QdrantClient

# --- 1. Data Containers & Engines ---

class RAGTestResult:
    """Holds metrics for both standard generation and RAG pipelines."""
    def __init__(self, model_id, mode):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.model_id = model_id
        self.mode = mode
        
        # Pipeline Metrics
        self.embedding_time = 0.0
        self.retrieval_time = 0.0
        self.ttft = 0.0          # Time to First Token (Generation only)
        self.total_time = 0.0    # End-to-end pipeline time
        
        # Generation Metrics
        self.token_count = 0
        self.tps = 0.0
        self.context_provided = False
        self.full_response = ""

class PipelineEngines:
    """Encapsulates the Embedding and Retrieval tools."""
    def __init__(self):
        self.oai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.embed_model = "text-embedding-nomic-embed-text-v1.5@q8_0"
        self.qdrant = QdrantClient(path="./qdrant_data")
        self.collection = "jarvis_knowledge"

    def warm_up_embedding(self):
        print(f"\n🔥 Warming up Embedding Model ({self.embed_model})...", end="", flush=True)
        try:
            self.oai_client.embeddings.create(input="warmup", model=self.embed_model)
            print(" Ready.")
        except Exception as e:
            print(f" Failed: {e}")
    
    def get_context(self, query: str):
        start_embed = time.perf_counter()
        query_vector = self.oai_client.embeddings.create(input=query, model=self.embed_model).data[0].embedding
        embed_time = time.perf_counter() - start_embed

        start_search = time.perf_counter()
        response = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=2
        )
        
        valid_chunks = [p.payload['text'] for p in response.points if p.score >= 0.6]
        search_time = time.perf_counter() - start_search
        
        context_str = "\n---\n".join(valid_chunks) if valid_chunks else None
        return context_str, embed_time, search_time

# --- 2. The Benchmarker ---

class RAGBenchmarker:
    # Added 'engines' to the parameters here
    def __init__(self, model_id, engines, base_url="http://localhost:1234/v1"):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model_id = model_id
        # Now it just uses the one we pass in
        self.engines = engines 

    def warm_up(self):
        print(f"\n🔥 Warming up {self.model_id}...", end="", flush=True)
        try:
            self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False
            )
            print(" Ready.")
        except Exception as e:
            print(f" Failed: {e}")

    def run_test(self, prompt, mode="CLEAN"):
        res = RAGTestResult(self.model_id, mode)
        nonce = f" [RID:{uuid.uuid4().hex[:6]}]"
        
        print(f"\n[Testing {res.model_id} | Mode: {res.mode}]")
        
        pipeline_start = time.perf_counter()

        # Step 1: Orchestration (Only for JARVIS-RAG)
        context = None
        if mode == "JARVIS-RAG":
            context, res.embedding_time, res.retrieval_time = self.engines.get_context(prompt)
            if context:
                res.context_provided = True
                system_content = f"You are Jarvis. Answer using ONLY this context:\n{context}{nonce}"
                temperature = 0.3
            else:
                system_content = f"You are Jarvis. State you don't have this in your files.{nonce}"
                temperature = 0.3
        else:
            system_content = f"Respond raw.{nonce}"
            temperature = 0.0

        # Step 2: Generation
        gen_start_time = time.perf_counter()
        try:
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    if res.ttft == 0.0:
                        res.ttft = time.perf_counter() - gen_start_time
                        print(f" (TTFT: {res.ttft:.2f}s) ", end="", flush=True)

                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    res.full_response += content
                    res.token_count += 1

            res.total_time = time.perf_counter() - pipeline_start
            
            # TPS is based purely on the generation phase
            generation_period = (time.perf_counter() - gen_start_time) - res.ttft
            res.tps = res.token_count / generation_period if generation_period > 0 else 0
            
            return res

        except Exception as e:
            print(f"\n[Error]: {e}")
            return None

# --- 3. Execution ---
if __name__ == "__main__":
    try:
        slms = ["google/gemma-2-9b",
                "llama-3.2-3b-instruct",
                "mistralai/ministral-3-3b",
                "openai/gpt-oss-20b",
                "phi-3-mini-4k-instruct", 
                "qwen2.5-7b-instruct"
                ] 
    
        test_query = "What is the primary benefit of a microservices architecture?"
        #test_query = "How many continents are there?"

        benchmark_results = []
        
        print("="*60)
        print("🚀 JARVIS RAG PIPELINE BENCHMARK 🚀")
        print("="*60)

        engines = PipelineEngines()
        engines.warm_up_embedding()

        for i, model in enumerate(slms):
            # The Hardware Management Gate
            if i > 0:
                print("\n" + "!"*60)
                print(f"🛑 VRAM PURGE REQUIRED")
                print(f"Please switch to LM Studio and manually eject the previous LLM.")
                print("!"*60)
                input(f"Press [Enter] when ready to load and test '{model}'...")
                print("Resuming pipeline...\n")

            # Pass the engines instance we already created into the benchmarker!
            tester = RAGBenchmarker(model, engines=engines)
            tester.warm_up()

            for mode in ["CLEAN", "JARVIS-RAG"]:
                result_obj = tester.run_test(test_query, mode=mode)
                
                if result_obj:
                    benchmark_results.append({
                        "Timestamp": result_obj.timestamp,
                        "Model": result_obj.model_id,
                        "Mode": result_obj.mode,
                        "Total_Time_Sec": round(result_obj.total_time, 2),
                        "TTFT_Sec": round(result_obj.ttft, 3),
                        "TPS": round(result_obj.tps, 2),
                        "Embed_Time_Sec": round(result_obj.embedding_time, 4),
                        "Search_Time_Sec": round(result_obj.retrieval_time, 4),
                        "Context_Used": result_obj.context_provided,
                        "Raw_Output": result_obj.full_response.strip().replace("\n", " ")[:150] + "..."
                    })
        
        # Export to CSV (Keep your existing CSV logic here)
        if benchmark_results:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"RAG_benchmark_{timestamp_str}.csv"
            csv_columns = ["Timestamp", "Model", "Mode", "Total_Time_Sec", "TTFT_Sec", "TPS", "Embed_Time_Sec", "Search_Time_Sec", "Context_Used", "Raw_Output"]

            try:
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    writer.writerows(benchmark_results)

                print("\n" + "="*60)
                print(f"✅ Success! Data pipeline telemetry saved to: {filename}")
                print("="*60)
            except IOError as e:
                print(f"❌ CSV Write Error: {e}")
    finally:
        print("\nClosing database connections...")
        # Access the client inside your engines instance
        engines.qdrant.close()