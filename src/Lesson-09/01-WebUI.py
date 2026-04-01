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
from core.mouth import AsyncMouth
from core.parser import DocumentParser
from utils.metrics import telemetry

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
    brain = LocalStreamBrain()
    memory = AsyncMemory()
    return brain, memory

brain, memory = load_engine()

@st.cache_resource
def get_mouth():
    return AsyncMouth()

@st.cache_resource
def get_parser():
    return DocumentParser()

mouth = get_mouth()
parser = get_parser()

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
        if st.sidebar.button("Process Document"):
            with st.spinner("Processing document... This may take a while for large PDFs."):
                
                # --- TELEMETRY: Start of Document Ingestion ---
                telemetry.reset_session()
                telemetry.start("Total Pipeline Time")
                telemetry.start("Document Processing Wait Time")
                
                # 1. Read the file bytes
                file_bytes = uploaded_file.read()
                filename = uploaded_file.name
                
                # 2. Parse and chunk the document
                chunks = parser.extract_and_chunk(file_bytes, filename)
                
                # 3. Save to Vector Database
                success = memory.save_document_chunks(filename, chunks)
                
                if success:
                    st.sidebar.success(f"Successfully processed {filename} into {len(chunks)} chunks!")
                else:
                    st.sidebar.error("Failed to process document.")
                    
                # --- TELEMETRY: End of Document Ingestion ---
                telemetry.stop("Document Processing Wait Time")
                telemetry.stop("Total Pipeline Time")
                telemetry.generate_report()

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

# Make sure to import telemetry at the top of your file!
# from utils.metrics import telemetry

# ==========================================
# CHAT INTERFACE
# ==========================================
if user_input := st.chat_input("Message JARVIS..."):
    
    # --- TELEMETRY: Start of turn ---
    telemetry.reset_session()
    telemetry.start("Total Pipeline Time")
    telemetry.start("Total User Wait Time")
    
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
        
        first_chunk_received = False
        
        for chunk in brain.stream_response(context_payload):
            if chunk.choices[0].delta.content:
                
                # --- TELEMETRY: End User Wait Time on first character ---
                if not first_chunk_received:
                    telemetry.stop("Total User Wait Time")
                    first_chunk_received = True
                    
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

    # --- TELEMETRY: End of turn ---
    telemetry.stop("Total Pipeline Time")
    telemetry.generate_report()