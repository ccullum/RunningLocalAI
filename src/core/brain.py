from openai import OpenAI
from core.colors import Colors

class JarvisBrain:
    """Manages the connection to the local LLM and handles streaming responses."""
    
    def __init__(self, model_id="local-model", base_url="http://localhost:1234/v1", api_key="lm-studio"):
        print(f"[System] Initializing Jarvis Brain (LM Studio)...")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id

    def stream_response(self, messages, temperature=0.3, max_tokens=300):
        """Yields chunks of the LLM response for asynchronous processing."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            return stream
        except Exception as e:
            print(f"{Colors.ERROR}[Brain Connection Error: {e}]{Colors.RESET}")
            return []
            
    def process_background_task(self, prompt, max_tokens=50):
        """Utility method for non-streaming background tasks (like HRE routing)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"{Colors.ERROR}[Brain Task Error: {e}]{Colors.RESET}")
            return ""