import os
from openai import OpenAI

class StreamingModel:
    def __init__(self, model_id: str, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        
        # Define our Modes
        self.MODES = {
            "CLEAN": {
                "system": "", # Empty string to suppress defaults
                "temperature": 0.0, 
                "description": "Raw data mode - no personality, no fluff."
            },
            "JARVIS": {
                "system": "You are Jarvis, a highly efficient, sophisticated software assistant. Be concise, professional, and slightly witty.",
                "temperature": 0.7,
                "description": "Persona mode - conversational and helpful."
            }
        }

    def stream_query(self, prompt: str, mode: str = "CLEAN"):
        """
        Streams response based on the selected mode.
        :param prompt: The user's question
        :param mode: 'CLEAN' or 'JARVIS'
        """
        selected_mode = self.MODES.get(mode.upper(), self.MODES["CLEAN"])
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": selected_mode["system"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=selected_mode["temperature"],
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"\n[Architect Error]: Connection failed for {self.model_id}. Details: {str(e)}"

# --- Implementation Example ---

def run_comparison():
    # Example setup: Testing one model in both modes
    my_slm = StreamingModel(model_id="mistralai/ministral-3-3b")
    test_prompt = "What is the primary benefit of a microservices architecture?"

    for mode_name in ["CLEAN", "JARVIS"]:
        print(f"\n\n=== MODE: {mode_name} ===")
        
        for chunk in my_slm.stream_query(test_prompt, mode=mode_name):
            print(chunk, end="", flush=True)

if __name__ == "__main__":
    run_comparison()