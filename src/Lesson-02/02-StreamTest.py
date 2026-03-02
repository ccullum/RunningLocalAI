import time
import uuid
import csv
import os
from datetime import datetime
from openai import OpenAI

class TestResult:
    """A data container with public members to hold benchmark metrics."""
    def __init__(self, model_id, mode):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.model_id = model_id
        self.mode = mode
        self.start_time = 0.0
        self.ttft = 0.0
        self.total_time = 0.0
        self.token_count = 0
        self.tps = 0.0
        self.full_response = ""
        self.quality_score = "" # Manual placeholder

class StreamingBenchmarker:
    def __init__(self, model_id, base_url="http://localhost:1234/v1"):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model_id = model_id

    def warm_up(self):
        """Pre-loads the model and warms up the VRAM cache."""
        print(f"\n\n🔥 Warming up {self.model_id}...", end="", flush=True)
        try:
            # We send a trivial request to force the model into memory
            self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False # No need to stream a warm-up
            )
            print(" Ready.")
        except Exception as e:
            print(f" Failed warm-up: {e}")

    def run_test(self, prompt, mode="CLEAN"):
        # Initialize our public member container
        res = TestResult(self.model_id, mode)
        
        nonce = f" [RID:{uuid.uuid4().hex[:6]}]"
        system_content = ("Respond raw." if mode == "CLEAN" else "You are Jarvis.") + nonce

        print(f"\n[Testing {res.model_id} | Mode: {res.mode}]")

        res.start_time = time.perf_counter()

        try:
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0 if mode == "CLEAN" else 0.7,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    # Capture TTFT as a public member
                    if res.ttft == 0.0:
                        res.ttft = time.perf_counter() - res.start_time
                        print(f" (TTFT: {res.ttft:.2f}s) ", end="", flush=True)

                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    
                    # Update public members directly
                    res.full_response += content
                    res.token_count += 1

            res.total_time = time.perf_counter() - res.start_time
            # Speed logic: tokens / (total time - initial thinking time)
            generation_period = res.total_time - res.ttft
            res.tps = res.token_count / generation_period if generation_period > 0 else 0
            
            return res

        except Exception as e:
            print(f"\n[Error]: {e}")
            return None

# --- Execution ---
if __name__ == "__main__":
    slms = ["google/gemma-2-9b",
            "llama-3.2-3b-instruct",
            "mistralai/ministral-3-3b",
            "openai/gpt-oss-20b",
            "phi-3-mini-4k-instruct", 
            "qwen2.5-7b-instruct"
            ] 
    prompt = "What is the primary benefit of a microservices architecture?"
    benchmark_results = []
    
    for model in slms:
        tester = StreamingBenchmarker(model)
        # 1. THE ARCHITECT'S MOVE: Warm up once per model load
        tester.warm_up()

        for mode in ["CLEAN", "JARVIS"]:
            # metrics is now a 'TestResult' object, not a dict
            result_obj = tester.run_test(prompt, mode=mode)
            
            if result_obj:
                # We can access result_obj.ttft or result_obj.tps directly here
                benchmark_results.append({
                    "Timestamp": result_obj.timestamp,
                    "Model": result_obj.model_id,
                    "Mode": result_obj.mode,
                    "Total_Time_Sec": round(result_obj.total_time, 2),
                    "TTFT_Sec": round(result_obj.ttft, 3),
                    "Tokens": result_obj.token_count,
                    "TPS": round(result_obj.tps, 2),
                    "Quality_Score": result_obj.quality_score,
                    "Raw_Output": result_obj.full_response.strip().replace("\n", " ")[:200] + "..."
                })
    
    # CSV Writing Logic
    if benchmark_results:
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Stream_benchmark_{timestamp_str}.csv"
        csv_columns = ["Timestamp", "Model", "Mode", "Total_Time_Sec", "TTFT_Sec", "Tokens", "TPS", "Quality_Score", "Raw_Output"]

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerows(benchmark_results)

            print("\n" + "="*40)
            print(f"✅ Success! Saved to: {filename}")
            print("="*40)
        except IOError as e:
            print(f"❌ CSV Write Error: {e}")