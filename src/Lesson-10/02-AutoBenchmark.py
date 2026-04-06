import sys
import os
import time
import requests

# ==========================================
# PATH INJECTION (To find the core package)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# ==========================================
# MODULAR IMPORTS
# ==========================================
from core.config import Config
from core.colors import Colors
from core.brain import LocalStreamBrain
from core.memory import AsyncMemory
from utils.metrics import perf_tracker
from utils.telemetry import AutoBenchmark

LM_STUDIO_MGMT_URL = Config.LLM_BASE_URL.replace("/v1", "/api/v1")

def swap_lm_studio_model(target_model_id, previous_model_id, brain):
    """Ejects the previous LLM, loads the target, and warms up the KV cache."""
    
    # 1. Eject the PREVIOUS model (if one exists)
    if previous_model_id and previous_model_id != target_model_id:
        print(f"{Colors.SYSTEM}🔄 Ejecting previous LLM: {previous_model_id}...{Colors.RESET}")
        try:
            unload_resp = requests.post(
                f"{LM_STUDIO_MGMT_URL}/models/unload", 
                json={"instance_id": previous_model_id},
                timeout=10
            )
            if unload_resp.status_code == 200:
                print(f"{Colors.USER}✅ Ejection successful.{Colors.RESET}")
                time.sleep(2) # Give VRAM a second to flush
            else:
                print(f"{Colors.WARNING}⚠️ Ejection returned status {unload_resp.status_code}. Moving on.{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.ERROR}⚠️ Ejection error: {e}{Colors.RESET}")

    # 2. Load the NEW target model unconditionally
    print(f"{Colors.SYSTEM}⏳ Loading target LLM: {target_model_id}... (This takes a moment){Colors.RESET}")
    try:
        load_resp = requests.post(
            f"{LM_STUDIO_MGMT_URL}/models/load", 
            json={"model": target_model_id}, 
            timeout=120
        )
        
        if load_resp.status_code == 200:
            print(f"{Colors.USER}✅ Model successfully loaded into VRAM!{Colors.RESET}")
            
            # THE WARM-UP BLOCK
            print(f"{Colors.SYSTEM}🔥 Warming up KV Cache (absorbing initialization penalty)...{Colors.RESET}")
            warmup_payload = [{"role": "user", "content": "Wake up. Reply with the word 'ready'."}]
            for _ in brain.stream_response(warmup_payload):
                pass 
            print(f"{Colors.USER}✅ Warm-up complete. Starting official telemetry.{Colors.RESET}")
            time.sleep(3) 
            
        else:
            print(f"{Colors.ERROR}❌ Failed to load model {target_model_id}. Status Code: {load_resp.status_code}{Colors.RESET}")
            print(f"{Colors.ERROR}Response: {load_resp.text}{Colors.RESET}")
            sys.exit(1)
            
    except Exception as e:
        print(f"{Colors.ERROR}⚠️ Error loading model: {e}{Colors.RESET}")
        sys.exit(1)

def run_benchmarks():
    print(f"{Colors.SYSTEM}Booting JARVIS Core Engine for Headless Benchmarking...{Colors.RESET}")
    brain = LocalStreamBrain()
    memory = AsyncMemory()
    hardware_monitor = AutoBenchmark()

    models_to_test = getattr(Config, "TARGET_TEST_MODELS", [])
    prompts = getattr(Config, "TEST_PROMPTS", [])

    if not models_to_test or not prompts:
        print(f"{Colors.ERROR}❌ No models or prompts found in Config!{Colors.RESET}")
        return

    print(f"{Colors.SYSTEM}🚀 Starting Benchmark Suite: {len(models_to_test)} Models x {len(prompts)} Prompts{Colors.RESET}")

    # --- THE STATE TRACKER ---
    previous_model = None

    # Outer Loop: Hot-Swap the Models
    for model_id in models_to_test:
        print(f"\n{Colors.METRICS}{'='*50}")
        print(f"🧱 PREPARING HARDWARE FOR: {model_id}")
        print(f"{'='*50}{Colors.RESET}")
        
        # Pass the previous model so it knows exactly what to nuke!
        swap_lm_studio_model(model_id, previous_model, brain)
        
        # Update the state tracker for the next loop
        previous_model = model_id

        # Inner Loop: Run the prompts
        for i, prompt in enumerate(prompts):
            print(f"\n{Colors.SYSTEM}🧪 {model_id} | Test {i+1}/{len(prompts)}: '{Colors.USER}{prompt}{Colors.SYSTEM}'{Colors.RESET}")
            
            perf_tracker.reset_session()
            perf_tracker.start("Total Pipeline Time")
            perf_tracker.start("Total User Wait Time")
            hardware_monitor.start_monitoring()

            if getattr(Config, "LOG_USER_PROMPT", False):
                perf_tracker.record_value("User Prompt", prompt)

            memory.add_user_message(prompt)
            context_payload = memory.get_context_payload(prompt)
            
            first_chunk_received = False
            full_response = ""
            
            for chunk in brain.stream_response(context_payload):
                if chunk.choices[0].delta.content:
                    if not first_chunk_received:
                        perf_tracker.stop("Total User Wait Time")
                        first_chunk_received = True
                    full_response += chunk.choices[0].delta.content
            
            memory.add_assistant_message(full_response)
            
            if getattr(Config, "LOG_LLM_RESPONSE", False):
                perf_tracker.record_value("LLM Response", full_response)

            perf_tracker.stop("Total Pipeline Time")
            avg_cpu, avg_ram = hardware_monitor.stop_monitoring()
            
            perf_tracker.generate_report()
            hardware_monitor.log_full_report(perf_tracker.session_data, avg_cpu, avg_ram)

            print(f"{Colors.WARNING}⏳ Sleeping for 5 seconds to let APU cool down...{Colors.RESET}")
            time.sleep(5)
            
    print(f"\n{Colors.USER}✅ Automated Benchmarking Complete! All data logged to CSV.{Colors.RESET}")

if __name__ == "__main__":
    run_benchmarks()