import pandas as pd
from openai import OpenAI

# Initialize the LM Studio client for the AI Judge
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
JUDGE_MODEL = "local-model" # LM studio will automatically use whatever is currently loaded

def generate_use_case(model_name, ttft, tps, overhead, reliability, score):
    """Uses the loaded LLM to dynamically write a hardware use case based on the metrics."""
    
    prompt = f"""You are a Senior AI Architect evaluating local LLM deployments.
    I have benchmarked the model '{model_name}' on consumer hardware with the following metrics:
    - Time-to-First-Token (Speed): {ttft:.2f} seconds
    - Throughput: {tps:.1f} Tokens/Second
    - Background Compute Overhead: {overhead if overhead != float('inf') else 'FAILED'} seconds
    - Agentic Reliability: {reliability}
    - Overall Performance Score: {score:.1f}/100

    Write a highly concise, 2-sentence recommendation on what hardware use cases this model is best suited for. 
    If its reliability is "Failed Routing", you MUST explicitly warn the user not to use it for complex agentic workflows or background routing.
    Do not use introductory filler, just output the recommendation."""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating use case. Make sure a model is loaded in LM Studio. ({e})"


def run_recommendation_engine(csv_filename):
    print(f"Loading benchmark data from: {csv_filename}...\n")
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filename}.")
        return

    print("="*80)
    print("🏆 JARVIS AI-ASSISTED RECOMMENDATION ENGINE 🏆")
    print("="*80)
    print("Generating dynamic use-cases... (This will take a moment as the AI analyzes the data)\n")

    models = df['Model'].unique()
    raw_results = []

    # 1. Gather Raw Data
    for model in models:
        model_df = df[df['Model'] == model]
        
        fast_path = model_df[model_df['Strategy'] == 'Sliding Window']
        avg_ttft = fast_path['TTFT_Sec'].mean()
        avg_tps = fast_path['TPS'].mean()
        
        has_summary = 'Rolling Summary' in model_df['Strategy'].values
        has_vector = 'Vector RAG (Deconstructed)' in model_df['Strategy'].values
        reliability = "Excellent" if (has_summary and has_vector) else "Failed Routing"

        summary_overhead = model_df[model_df['Strategy'] == 'Rolling Summary']['Overhead_Sec'].mean() if has_summary else float('inf')
        vector_overhead = model_df[model_df['Strategy'] == 'Vector RAG (Deconstructed)']['Overhead_Sec'].mean() if has_vector else float('inf')
        avg_overhead = (summary_overhead + vector_overhead) / 2 if (has_summary and has_vector) else float('inf')
        
        raw_results.append({
            "Model": model,
            "Avg_TTFT": avg_ttft,
            "Avg_TPS": avg_tps,
            "Reliability": reliability,
            "Avg_Overhead": avg_overhead
        })

    results_df = pd.DataFrame(raw_results)

    # 2. Calculate 0-100 Scores (Min-Max Normalization)
    best_ttft, worst_ttft = results_df['Avg_TTFT'].min(), results_df['Avg_TTFT'].max()
    best_tps, worst_tps = results_df['Avg_TPS'].max(), results_df['Avg_TPS'].min()
    
    valid_overhead = results_df[results_df['Avg_Overhead'] != float('inf')]['Avg_Overhead']
    best_over = valid_overhead.min() if not valid_overhead.empty else 1
    worst_over = valid_overhead.max() if not valid_overhead.empty else 10

    scored_results = []
    
    for _, row in results_df.iterrows():
        # TTFT Score (Lower is better)
        ttft_score = max(0, 100 * (worst_ttft - row['Avg_TTFT']) / (worst_ttft - best_ttft)) if worst_ttft != best_ttft else 100
        
        # TPS Score (Higher is better)
        tps_score = max(0, 100 * (row['Avg_TPS'] - worst_tps) / (best_tps - worst_tps)) if worst_tps != best_tps else 100
        
        # Overhead Score (Lower is better, 0 if it failed)
        if row['Avg_Overhead'] == float('inf'):
            over_score = 0
            rel_score = 0
        else:
            rel_score = 100
            over_score = max(0, 100 * (worst_over - row['Avg_Overhead']) / (worst_over - best_over)) if worst_over != best_over else 100

        # Weighted Final Score (35% Speed, 35% TPS, 15% Overhead, 15% Reliability)
        final_score = (ttft_score * 0.35) + (tps_score * 0.35) + (over_score * 0.15) + (rel_score * 0.15)
        
        # 3. Ask the AI to write the Use Case
        ai_use_case = generate_use_case(row['Model'], row['Avg_TTFT'], row['Avg_TPS'], row['Avg_Overhead'], row['Reliability'], final_score)
        
        scored_results.append({
            "Model": row['Model'],
            "Score": final_score,
            "TTFT": row['Avg_TTFT'],
            "TPS": row['Avg_TPS'],
            "Reliability": row['Reliability'],
            "AI_Analysis": ai_use_case
        })

    final_df = pd.DataFrame(scored_results).sort_values(by="Score", ascending=False).reset_index(drop=True)

    # --- PRINT OUT THE FINAL REPORT ---
    print(f"{'RANK':<5} | {'MODEL':<25} | {'SCORE':<7} | {'TTFT':<6} | {'TPS':<6} | {'RELIABILITY'}")
    print("-" * 80)
    for idx, row in final_df.iterrows():
        status = "✅ PASS" if row['Reliability'] == "Excellent" else "❌ FAIL"
        print(f"#{idx+1:<3} | {row['Model']:<25} | {row['Score']:>5.1f}/100 | {row['TTFT']:>5.2f}s | {row['TPS']:>5.1f} | {status}")
        print(f"\n   🤖 AI ARCHITECT VERDICT:\n   {row['AI_Analysis']}\n")
        print("-" * 80)

if __name__ == "__main__":
    # Ensure this matches your latest CSV filename!
    run_recommendation_engine("HRE_benchmark_2026-03-06_14-45-21.csv")