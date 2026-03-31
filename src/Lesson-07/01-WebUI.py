import sys
import os
import threading
import streamlit as st

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
from core.brain import LocalStreamBrain
from core.memory import AsyncMemory
#from core.mouth import AsyncMouth

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="JARVIS Core V2.0", page_icon="🧠", layout="centered")

# --- NEW: SIDEBAR SETTINGS ---
#with st.sidebar:
#    st.header("⚙️ JARVIS Control")
#    voice_enabled = st.toggle("🗣️ Enable Voice Output", value=False)
#    
#    # Bonus feature: A button to quickly wipe the UI chat state!
#    if st.button("🗑️ Clear Chat History"):
#        st.session_state.messages = []
#        st.rerun()

st.title("JARVIS Command Center")

# ==========================================
# INITIALIZE BACKEND (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Booting JARVIS Core Engine...")
def load_engine():
    """Loads the Brain and Memory exactly once and keeps them in RAM."""
    brain = LocalStreamBrain(model_id="local-model")
    memory = AsyncMemory(model_id="local-model")
    return brain, memory

brain, memory = load_engine()

#@st.cache_resource
#def get_jarvis_mouth():
#    return AsyncMouth()

#mouth = get_jarvis_mouth()

# ==========================================
# UI SESSION STATE
# ==========================================
# Streamlit clears standard variables on rerun. We use session_state to persist the UI chat log.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render the existing chat history on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# CHAT INTERFACE
# ==========================================
# This creates the text box at the bottom of the screen
if user_input := st.chat_input("Message JARVIS..."):
    
    # 1. Display User Message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Add to Backend Vector Database and retrieve context
    memory.add_user_message(user_input)
    context_payload = memory.get_context_payload(user_input)
    
    # 3. Stream JARVIS's Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the chunks from LM Studio to the Web UI
        for chunk in brain.stream_response(context_payload):
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                # Add a blinking cursor effect while typing
                response_placeholder.markdown(full_response + "▌")
                
        # Finalize the text without the cursor
        response_placeholder.markdown(full_response)
        
    # 4. Save to Backend Vector Database and UI state
    memory.add_assistant_message(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # 5. --- NEW: THE VOICE TRIGGER ---
#    if voice_enabled:
#        print("[WebUI] Sending response to Piper TTS...")
#        # NOTE: If your function inside core/mouth.py is named something else (like 'say' or 'generate_audio'), update 'mouth.speak' below!
#        threading.Thread(target=mouth.enqueue_sentence, args=(full_response,), daemon=True).start()