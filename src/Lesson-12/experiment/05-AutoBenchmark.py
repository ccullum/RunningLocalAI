import sys
import os
import time
import requests

# ==========================================
# PATH INJECTION
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.config import Config
from core.colors import Colors
from core.brain import LocalStreamBrain
from core.memory import AsyncMemory
from utils.metrics import perf_tracker
from utils.telemetry import AutoBenchmark

LM_STUDIO_MGMT_URL = Config.LLM_BASE_URL.replace("/v1", "/api/v1")

def swap_lm_studio_model(target_model_id, previous_model_id, brain):
    if previous_model_id and previous_model_id != target_model_id:
        print(f"{Colors.SYSTEM}🔄 Ejecting previous LLM: {previous_model_id}...{Colors.RESET}")
        try:
            requests.post(f"{LM_STUDIO_MGMT_URL}/models/unload", json={"instance_id": previous_model_id}, timeout=10)
            time.sleep(2)
        except Exception as e:
            print(f"{Colors.ERROR}⚠️ Ejection error: {e}{Colors.RESET}")

    print(f"{Colors.SYSTEM}⏳ Loading target LLM: {target_model_id}...{Colors.RESET}")
    try:
        load_resp = requests.post(f"{LM_STUDIO_MGMT_URL}/models/load", json={"model": target_model_id}, timeout=120)
        if load_resp.status_code == 200:
            print(f"{Colors.USER}✅ Model loaded. Warming up KV cache...{Colors.RESET}")
            warmup_payload = [{"role": "user", "content": "Wake up. Reply with 'ready'."}]
            for _ in brain.stream_response(warmup_payload): pass 
            time.sleep(3) 
        else:
            print(f"{Colors.ERROR}❌ Load failed: {load_resp.text}{Colors.RESET}")
            sys.exit(1)
    except Exception as e:
        print(f"{Colors.ERROR}⚠️ Error loading model: {e}{Colors.RESET}")
        sys.exit(1)

def run_benchmarks():
    # Detect and log the Router state for the report
    mode = "LOCAL C++" if Config.USE_LOCAL_CPP_ROUTER else "LEGACY API"
    print(f"{Colors.SYSTEM}--- JARVIS Auto-Benchmark (Router Mode: {mode}) ---{Colors.RESET}")
    
    brain = LocalStreamBrain()
    memory = AsyncMemory()
    hardware_monitor = AutoBenchmark()

    models_to_test = getattr(Config, "TARGET_TEST_MODELS", [])
    prompts = getattr(Config, "TEST_PROMPTS", [])

    previous_model = None

    for model_id in models_to_test:
        print(f"\n{Colors.METRICS}{'='*50}\n🧱 PREPARING HARDWARE FOR: {model_id}\n{'='*50}{Colors.RESET}")
        swap_lm_studio_model(model_id, previous_model, brain)
        previous_model = model_id

        for i, prompt in enumerate(prompts):
            print(f"\n{Colors.SYSTEM}🧪 {model_id} | Test {i+1}/{len(prompts)}: '{Colors.USER}{prompt}{Colors.SYSTEM}'{Colors.RESET}")
            
            perf_tracker.reset_session()
            perf_tracker.start("Total Pipeline Time")
            perf_tracker.start("Total User Wait Time")
            hardware_monitor.start_monitoring()

            # Record which router we are using for this session
            perf_tracker.record_value("Router Implementation", mode)

            memory.add_user_message(prompt)
            # This triggers the semantic routing math
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
            perf_tracker.stop("Total Pipeline Time")
            avg_cpu, avg_ram = hardware_monitor.stop_monitoring()
            
            perf_tracker.generate_report()
            hardware_monitor.log_full_report(perf_tracker.session_data, avg_cpu, avg_ram)

            print(f"{Colors.WARNING}⏳ Cooling down...{Colors.RESET}")
            time.sleep(5)
            
    print(f"\n{Colors.USER}✅ Automated Benchmarking Complete!{Colors.RESET}")

if __name__ == "__main__":
    run_benchmarks()