import time
import threading
import psutil
import requests
import csv
import os
from datetime import datetime
from core.config import Config
from core.colors import Colors

class AutoBenchmark:
    def __init__(self):
        self.csv_filename = Config.TELEMETRY_CSV_PATH
        self.poll_interval = Config.TELEMETRY_POLL_INTERVAL
        self.is_monitoring = False
        self.cpu_usage_log = []
        self.ram_usage_log = []
        self.monitor_thread = None

    def get_lm_studio_model(self):
        """Pings LM Studio's local server to get the currently loaded LLM."""
        try:
            # Dynamically uses the base URL from config.py
            endpoint = f"{Config.LLM_BASE_URL}/models"
            response = requests.get(endpoint, timeout=2)
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and len(data["data"]) > 0:
                    # Loop through all loaded models
                    for model in data["data"]:
                        model_id = model.get("id", "")
                        
                        # Skip if it matches our exact embedding model from config
                        if model_id == Config.EMBED_MODEL:
                            continue
                            
                        # Skip if it has 'embed' in the name (bulletproof fallback)
                        if "embed" in model_id.lower():
                            continue
                            
                        # If it passes the checks, this is our LLM!
                        return model_id
                        
                    # Absolute fallback if somehow only the embedder was found
                    return data["data"][0]["id"]
                    
            return "Unknown_Model"
        except requests.exceptions.RequestException:
            return "LM_Studio_Not_Found"

    def _poll_hardware(self):
        """Background thread function to poll CPU and RAM."""
        while self.is_monitoring:
            self.cpu_usage_log.append(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            self.ram_usage_log.append(mem.used / (1024**3))
            time.sleep(self.poll_interval)

    def start_monitoring(self):
        """Starts the hardware polling background thread."""
        self.cpu_usage_log.clear()
        self.ram_usage_log.clear()
        self.is_monitoring = True
        psutil.cpu_percent(interval=None) # Baseline call
        
        self.monitor_thread = threading.Thread(target=self._poll_hardware, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stops the thread and calculates the averages."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        avg_cpu = sum(self.cpu_usage_log) / len(self.cpu_usage_log) if self.cpu_usage_log else 0
        avg_ram = sum(self.ram_usage_log) / len(self.ram_usage_log) if self.ram_usage_log else 0
        
        return round(avg_cpu, 2), round(avg_ram, 2)

    def log_full_report(self, metrics_dict, avg_cpu, avg_ram):
        """Appends the full interaction metrics report to the CSV file."""
        model_name = self.get_lm_studio_model()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        file_exists = os.path.isfile(self.csv_filename)
        
        # Define the exact columns based on your console report + config settings
        headers = [
            "Timestamp", 
            "Model", 
            "Context Chunks Limit",
            "Avg CPU (%)", 
            "Avg RAM (GB)",
            "Memory Action",           
            "Semantic Router",         
            "Semantic Routing Math & Embed Time",
            "HRE Router",              
            "Background LLM Task",
            "TTFT",
            "Total User Wait Time",
            "LLM Output Tokens",
            "LLM Generation Time",
            "Total Pipeline Time",
            "TPS"
        ]
        
        with open(self.csv_filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(headers)
            
            # Map the dictionary values safely to the CSV row
            writer.writerow([
                timestamp,
                model_name,
                Config.CONTEXT_CHUNKS_LIMIT,  # Added this directly from Config!
                avg_cpu,
                avg_ram,
                metrics_dict.get("Memory Action", "N/A"),
                metrics_dict.get("Semantic Router", "N/A"),
                metrics_dict.get("Semantic Routing Math & Embed Time", 0),
                metrics_dict.get("HRE Router", "N/A"),
                metrics_dict.get("Background LLM Task", 0),
                metrics_dict.get("Time To First Token (TTFT)", 0),
                metrics_dict.get("Total User Wait Time", 0),
                metrics_dict.get("LLM Output Tokens", 0),
                metrics_dict.get("LLM Generation Time", 0),
                metrics_dict.get("Total Pipeline Time", 0),
                metrics_dict.get("Tokens Per Second (TPS)", 0)
            ])
            
        print(f"{Colors.METRICS}[Telemetry] Metrics automatically logged to: {Config.TELEMETRY_CSV_NAME}{Colors.RESET}")