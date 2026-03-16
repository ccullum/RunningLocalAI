import pandas as pd
from openai import OpenAI

# Initialize LM Studio client for the AI Judge
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
JUDGE_MODEL = "local-model" # Uses whatever is currently loaded

def run_audio_recommendation(csv_filename):
    print(f"Loading STT benchmark data from: {csv_filename}...\n")
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filename}. Make sure it is in the same directory.")
        return

    print("="*80)
    print("🏆 JARVIS AI-ASSISTED STT RECOMMENDATION ENGINE 🏆")
    print("="*80)
    print("Analyzing Audio Transcription Latency (ATL) and outputs...\n")

    # Format data for the AI Judge
    benchmark_data = ""
    for _, row in df.iterrows():
        benchmark_data += f"- Tool: {row['Tool']} ({row['Model_Size']})\n"
        benchmark_data += f"  ATL: {row['ATL_Sec']} seconds\n"
        benchmark_data += f"  Transcription: \"{row['Transcription']}\"\n\n"

    print(f"{'RANK':<5} | {'TOOL':<20} | {'SIZE':<10} | {'ATL (sec)':<10} | {'TRANSCRIPTION'}")
    print("-" * 80)
    
    # Sort purely by speed
    sorted_df = df.sort_values(by="ATL_Sec").reset_index(drop=True)
    
    for idx, row in sorted_df.iterrows():
        trans = str(row['Transcription'])
        # Truncate transcription for clean printing
        if len(trans) > 30: trans = trans[:27] + "..."
        print(f"#{idx+1:<3} | {row['Tool']:<20} | {row['Model_Size']:<10} | {row['ATL_Sec']:<10.2f} | {trans}")

    print("\nAsking the Local AI Architect to analyze the results... (Please wait)\n")

    prompt = f"""You are a Senior AI Architect evaluating local Speech-to-Text (STT) models for a voice assistant.
    The user recorded a single audio clip of themselves speaking. 
    Here are the benchmark results of the different STT engines trying to transcribe that exact same audio clip:

    {benchmark_data}

    Write a 3-paragraph architectural analysis:
    1. Explain why PocketSphinx failed so poorly compared to the others.
    2. Compare the official OpenAI Whisper to Faster-Whisper in terms of latency (ATL) and accuracy.
    3. Make a final recommendation on which Faster-Whisper model (tiny.en vs base.en) the user should deploy for their Jarvis application, considering the trade-off between speed and hardware resources.

    Do not use any introductory filler like "Here is the analysis". Output the analysis directly in a professional tone."""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        print("🤖 AI ARCHITECT VERDICT:\n")
        print(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error generating analysis. Is LM Studio running? ({e})")

if __name__ == "__main__":
    # Ensure this matches the exact filename of your generated CSV!
    run_audio_recommendation("STT_benchmark_2026-03-16_14-06-15.csv")