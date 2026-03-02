import pandas as pd
import glob
import os

MODEL_METADATA = {
    "google/gemma-2-9b": {"total": 9.24, "active": 9.24},
    "llama-3.2-3b-instruct": {"total": 3.21, "active": 3.21},
    "mistralai/ministral-3-3b": {"total": 3.00, "active": 3.00},
    "openai/gpt-oss-20b": {"total": 21.00, "active": 4.00},
    "phi-3-mini-4k-instruct": {"total": 3.82, "active": 3.82},
    "qwen2.5-7b-instruct": {"total": 7.61, "active": 7.61}
}

def analyze_and_recommend():
    list_of_files = glob.glob('Stream_benchmark_*.csv')
    if not list_of_files: return
    df = pd.read_csv(max(list_of_files, key=os.path.getctime))

    # Calculations (Consolidated)
    df['Active_Params'] = df['Model'].map(lambda x: MODEL_METADATA.get(x, {}).get('active', 0))
    df['Total_Params'] = df['Model'].map(lambda x: MODEL_METADATA.get(x, {}).get('total', 0))
    df['Compute_Efficiency'] = df['TPS'] / df['Active_Params']
    df['Resource_ROI'] = df['TPS'] / df['Total_Params']
    df['Density_Score'] = df['Tokens'] / df['Total_Time_Sec']

    # --- RECOMMENDATION ENGINE LOGIC ---
    print("\n" + "="*60)
    print("🤖 JARVIS MODEL RECOMMENDATION ENGINE 🤖")
    print("="*60)

    # Pivot for Disparity Analysis
    pivot = df.pivot(index='Model', columns='Mode', values=['Density_Score', 'TTFT_Sec', 'TPS'])
    
    for model in pivot.index:
        clean_ds = pivot.loc[model, ('Density_Score', 'CLEAN')]
        jarvis_ds = pivot.loc[model, ('Density_Score', 'JARVIS')]
        jarvis_ttft = pivot.loc[model, ('TTFT_Sec', 'JARVIS')]
        jarvis_tps = pivot.loc[model, ('TPS', 'JARVIS')]
        
        disparity = ((jarvis_ds - clean_ds) / clean_ds) * 100
        
        # Scoring Logic
        recommendation = "✅ RECOMMENDED"
        reasons = []

        if abs(disparity) > 20:
            recommendation = "⚠️ PERSONA UNSTABLE"
            reasons.append(f"High Mode Disparity ({disparity:.1f}%)")
        
        if jarvis_ttft > 0.8:
            recommendation = "🐢 LATENCY WARNING"
            reasons.append(f"High TTFT ({jarvis_ttft:.2f}s)")

        if jarvis_tps < 20:
            recommendation = "📉 LOW FLUIDITY"
            reasons.append(f"Low TPS ({jarvis_tps:.1f})")

        # Edge Case: Mixture of Experts VRAM check
        if MODEL_METADATA[model]['total'] > 15 and jarvis_tps < 30:
            recommendation = "💾 VRAM INEFFICIENT"
            reasons.append("Heavy footprint with modest throughput")

        # Output the verdict
        print(f"\nModel: {model}")
        print(f"Status: {recommendation}")
        if reasons:
            print(f"Notes:  {', '.join(reasons)}")
        else:
            print(f"Notes:  Excellent performance and stability.")

    print("\n" + "="*60)

if __name__ == "__main__":
    analyze_and_recommend()