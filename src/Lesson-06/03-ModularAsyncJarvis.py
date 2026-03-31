import sys
import os
import subprocess

# ==========================================
# THE ARCHITECT'S PATH INJECTION
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# ==========================================
# SPACE-SAFE BOOTSTRAPPER
# ==========================================
if sys.platform == "win32" and os.environ.get("PYTHONUTF8") != "1":
    os.environ["PYTHONUTF8"] = "1"
    sys.exit(subprocess.run([sys.executable] + sys.argv).returncode)

# ==========================================
# MODULAR IMPORTS
# ==========================================
from core.colors import Colors
from core.ear import VADTranscriptEar
from core.brain import LocalStreamBrain
from core.memory import AsyncMemory
from core.mouth import AsyncMouth

def main():
    print(f"\n{Colors.SYSTEM}[System] Booting Modular JARVIS Core V1.1 (Async/Barge-in)...\n{Colors.RESET}")
    
    # 1. Initialize Modules
    ear = VADTranscriptEar()
    brain = LocalStreamBrain(model_id="local-model")
    memory = AsyncMemory(model_id="local-model")
    mouth = AsyncMouth() # Paths are handled by the default args we set!
    
    mouth.enqueue_sentence("All core modules are loaded. I am ready.")
    mouth.wait_until_done()
    
    while True:
        # Reset the Mouth's interrupt flag and flush the queue for a new turn
        mouth.reset_state()
        
        user_input = ear.listen()
        if not user_input:
            continue
            
        print(f"\n{Colors.USER}User: {user_input}{Colors.RESET}")
        
        if "shut down" in user_input.lower() or "exit" in user_input.lower():
            mouth.enqueue_sentence("Shutting down core systems. Goodbye.")
            mouth.wait_until_done()
            memory.close_database()
            # Send the poison pill to close the background TTS thread safely
            mouth.tts_queue.put(None)
            break
            
        # Route Memory
        memory.add_user_message(user_input)
        messages = memory.get_context_payload(user_input)
        
        print(f"{Colors.JARVIS}JARVIS: {Colors.RESET}", end="", flush=True)
        
        # 2. The Sentence Accumulator (Pipelining)
        sentence_buffer = ""
        full_response = ""
        
        stream = brain.stream_response(messages)
        for chunk in stream:
            # If user hit Spacebar, instantly break the LLM generation!
            if mouth.is_interrupted:
                full_response += " [Interrupted]"
                break
                
            token = chunk.choices[0].delta.content
            if token:
                print(token, end="", flush=True)
                sentence_buffer += token
                full_response += token
                
                # If we hit punctuation, fire the sentence to the AsyncMouth!
                if any(p in token for p in ['.', '!', '?']):
                    mouth.enqueue_sentence(sentence_buffer.strip())
                    sentence_buffer = ""
                    
        # Catch any trailing thoughts that didn't end in punctuation
        if sentence_buffer.strip() and not mouth.is_interrupted:
            mouth.enqueue_sentence(sentence_buffer.strip())
            
        print() # Clean console newline
        
        # Save to memory
        memory.add_assistant_message(full_response)
        
        # Wait for the audio to finish speaking before listening to the mic again
        if not mouth.is_interrupted:
            mouth.wait_until_done()

if __name__ == "__main__":
    main()