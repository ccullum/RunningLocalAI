import pandas as pd
import glob
import os

# 1. Advanced Model Metadata
# Total: VRAM footprint. Active: The portion of weights used per token.
MODEL_METADATA = {
    "google/gemma-2-9b": {"total": 9.24, "active": 9.24},
    "llama-3.2-3b-instruct": {"total": 3.21, "active": 3.21},
    "mistralai/ministral-3-3b": {"total": 3.00, "active": 3.00},
    "openai/gpt-oss-20b": {"total": 21.00, "active": 4.00}, # MoE architecture
    "phi-3-mini-4k-instruct": {"total": 3.82, "active": 3.82},
    "qwen2.5-7b-instruct": {"total": 7.61, "active": 7.61}
}

def analyze_benchmarks():
    # Load latest CSV
    list_of_files = glob.glob('Stream_benchmark_*.csv')
    if not list_of_files: return
    df = pd.read_csv(max(list_of_files, key=os.path.getctime))

    # --- ENHANCED ARCHITECTURAL CALCULATIONS ---

    # Map metadata to dataframe
    df['Total_Params'] = df['Model'].map(lambda x: MODEL_METADATA.get(x, {}).get('total', 0))
    df['Active_Params'] = df['Model'].map(lambda x: MODEL_METADATA.get(x, {}).get('active', 0))

    # A. Compute Efficiency: Is the 'active' part of the model fast? 
    # (High score = well-optimized architecture/kernels)
    df['Compute_Efficiency'] = df['TPS'] / df['Active_Params']

    # B. Resource ROI: How much speed do I get for the VRAM I'm sacrificing?
    # (High score = high value for low-resource hardware)
    df['Resource_ROI'] = df['TPS'] / df['Total_Params']

    # C. Density Score: (Tokens / Total_Time)
    # Penalizes models with high TTFT or long pauses.
    df['Density_Score'] = df['Tokens'] / df['Total_Time_Sec']

    # --- REPORTING ---
    
    # 1. Compute vs Resource Comparison
    print("\n[COMPUTE EFFICIENCY vs RESOURCE ROI]")
    summary = df.groupby('Model').agg({
        'Compute_Efficiency': 'mean',
        'Resource_ROI': 'mean',
        'TPS': 'mean'
    }).sort_values(by='Compute_Efficiency', ascending=False)
    print(summary.round(2))

    # 2. Mode Disparity Analysis
    print("\n[PERSONA IMPACT: CLEAN vs JARVIS DENSITY]")
    pivot = df.pivot(index='Model', columns='Mode', values='Density_Score')
    pivot['Disparity_Pct'] = ((pivot['JARVIS'] - pivot['CLEAN']) / pivot['CLEAN']) * 100
    print(pivot[['CLEAN', 'JARVIS', 'Disparity_Pct']].sort_values(by='Disparity_Pct').round(2))

if __name__ == "__main__":
    analyze_benchmarks()