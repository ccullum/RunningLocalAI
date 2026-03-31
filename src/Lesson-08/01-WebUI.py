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
from core.brain import JarvisBrain
from core.memory import JarvisMemory
from core.mouth import AsyncMouth
from core.ingestor import DocumentIngestor

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="JARVIS Core V2.0", page_icon="🧠", layout="centered")

# ==========================================
# INITIALIZE BACKEND (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Booting JARVIS Core Engine...")
def load_engine():
    """Loads the Brain and Memory exactly once and keeps them in RAM."""
    brain = JarvisBrain(model_id="local-model")
    memory = JarvisMemory(model_id="local-model")
    return brain, memory

brain, memory = load_engine()

@st.cache_resource
def get_jarvis_mouth():
    return AsyncMouth()

@st.cache_resource
def get_jarvis_ingestor():
    return DocumentIngestor()

mouth = get_jarvis_mouth()
ingestor = get_jarvis_ingestor()

# ==========================================
# SIDEBAR SETTINGS & UPLOADER
# ==========================================
with st.sidebar:
    st.header("⚙️ JARVIS Control")
    voice_enabled = st.toggle("🗣️ Enable Voice Output", value=False)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    
    # --- DOCUMENT INGESTION ---
    st.header("📄 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        if st.button("🧠 Ingest Document"):
            with st.spinner("Extracting, parsing, and embedding..."):
                try:
                    # Read bytes and file name
                    file_bytes = uploaded_file.read()
                    filename = uploaded_file.name
                    
                    # Pass to Ingestor for raw text / OCR
                    extracted_text = ingestor.process_file(file_bytes, filename)
                    
                    # Pass raw text to Memory for chunking and Qdrant storage
                    if extracted_text and extracted_text.strip():
                        success = memory.ingest_document(filename, extracted_text)
                        
                        if success:
                            st.success(f"Successfully memorized {filename}!")
                            sys_note = f"[System Note]: The user just uploaded and ingested a file named '{filename}'."
                            st.session_state.messages.append({"role": "system", "content": sys_note})
                            memory.raw_history.append({"role": "system", "content": sys_note})
                        else:
                            st.error("Document processed, but no usable text chunks were generated.")
                    else:
                        st.warning("No text could be found or OCR'd from this file.")
                        
                except Exception as e:
                    st.error(f"Ingestion Pipeline Error: {e}")

st.title("JARVIS Command Center")

# ==========================================
# UI SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    # Hide the invisible system notes from the actual UI rendering
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ==========================================
# CHAT INTERFACE
# ==========================================
if user_input := st.chat_input("Message JARVIS..."):
    
    # 1. Display User Message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Add to Vector Database and retrieve context
    memory.add_user_message(user_input)
    context_payload = memory.get_context_payload(user_input)
    
    # 3. Stream JARVIS's Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in brain.stream_response(context_payload):
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
                
        response_placeholder.markdown(full_response)
        
    # 4. Save to Backend Database and UI state
    memory.add_assistant_message(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # 5. Voice Trigger (Threaded)
    if voice_enabled:
        print("[WebUI] Sending response to Piper TTS...")
        threading.Thread(target=mouth.speak, args=(full_response,), daemon=True).start()