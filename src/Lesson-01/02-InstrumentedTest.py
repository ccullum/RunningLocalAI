import time
import csv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# List the exact model identifiers from your LM Studio logs
models_to_test = [
    "google/gemma-2-9b",
    "llama-3.2-3b-instruct",
    "microsoft/phi-4-mini-reasoning",
    "mistralai/ministral-3-3b",
    "openai/gpt-oss-20b",
    "qwen2.5-7b-instruct"
]

# The magic single-space prompt to defeat the hidden tokens
prompt = ChatPromptTemplate.from_messages([
    ("system", " "), 
    ("user", "Question: {question} \nAnswer: Let's have a quick answer.")
])

# List to store our results
benchmark_results = []

print("🚀 Starting Automated Benchmark...\n")

for model_name in models_to_test:
    print(f"🔄 Swapping to and loading: {model_name}...")
    
    # Initialize LangChain with the specific model for this loop iteration
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model=model_name,
        temperature=0.1
    )
    
    chain = prompt | llm
    
    try:
        # WARM-UP: Send a tiny prompt to force LM Studio to load the model into memory
        chain.invoke({"question": "Hi"})
        
        # Start the actual timed benchmark
        start_time = time.time()
        response = chain.invoke({"question": "How many continents are there?"})
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        token_usage = response.response_metadata.get("token_usage", {})
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        # End-to-end TPS (Tokens Per Second)
        tps = completion_tokens / total_time if total_time > 0 else 0
        
        # Clean up the output string to fit cleanly in a CSV cell
        clean_output = response.content.strip().replace('\n', ' | ')
        
        # Save results
        benchmark_results.append({
            "Model": model_name.split("/")[-1], 
            "Time (Seconds)": round(total_time, 2),
            "Tokens Generated": completion_tokens,
            "Speed (TPS)": round(tps, 2),
            "Raw Output": clean_output
        })
        print(f"✅ Done! Speed: {tps:.2f} TPS\n")
        
    except Exception as e:
        print(f"❌ Error testing {model_name}. Make sure it is downloaded in LM Studio! Error: {e}\n")

# --- NEW: CSV Export Section ---
if benchmark_results:
    # Generate a filename with a timestamp so we don't overwrite previous runs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"llm_benchmark_{timestamp}.csv"
    
    # Define the column headers based on our dictionary keys
    csv_columns = ["Model", "Time (Seconds)", "Tokens Generated", "Speed (TPS)", "Raw Output"]
    
    # Write the data to the CSV file
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for row in benchmark_results:
                writer.writerow(row)
        
        print("📊 FINAL BENCHMARK RESULTS 📊")
        print("-" * 85)
        print(f"🎉 Success! Data successfully saved to: {filename}")
        print("-" * 85)
        
    except IOError as e:
        print(f"❌ Could not write to CSV file: {e}")