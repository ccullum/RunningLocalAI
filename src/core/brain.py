from openai import OpenAI
from .colors import Colors
from .config import Config
from utils.metrics import perf_tracker 

class LocalStreamBrain:
    """Manages the connection to the local LLM and handles streaming responses."""
    
    def __init__(self):
        print(f"{Colors.SYSTEM}[System] Initializing Local Stream Brain...{Colors.RESET}")
        self.client = OpenAI(base_url=Config.LLM_BASE_URL, api_key=Config.LLM_API_KEY)
        self.model_id = Config.LLM_MODEL

    def stream_response(self, messages, temperature=Config.LLM_CHAT_TEMPERATURE, max_tokens=Config.LLM_CHAT_MAX_TOKENS):
        """Yields chunks of the LLM response for asynchronous processing."""
        
        perf_tracker.start("LLM Generation Time")
        perf_tracker.start("Time To First Token (TTFT)")
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # We intercept the stream here to count the tokens for TPS math
            token_count = 0
            first_token_recorded = False
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if not first_token_recorded:
                        perf_tracker.stop("Time To First Token (TTFT)")
                        first_token_recorded = True
                    
                    token_count += 1
                yield chunk
                
            perf_tracker.record_value("LLM Output Tokens", token_count)
            perf_tracker.stop("LLM Generation Time")
            
        except Exception as e:
            print(f"{Colors.ERROR}[Brain Connection Error: {e}]{Colors.RESET}")
            perf_tracker.stop("LLM Generation Time")
            perf_tracker.stop("Time To First Token (TTFT)")
            return []
            
    @perf_tracker.measure("Background LLM Task")
    def process_background_task(self, prompt, max_tokens=Config.LLM_TASK_MAX_TOKENS):
        """Utility method for non-streaming background tasks (like HRE routing)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.LLM_TASK_TEMPERATURE, # Hard-locked to zero for deterministic tasks
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"{Colors.ERROR}[Brain Task Error: {e}]{Colors.RESET}")
            return ""