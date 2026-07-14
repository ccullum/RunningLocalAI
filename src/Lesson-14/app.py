import os
import sys
import streamlit as st

# --- AUTOMATIC PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from core.config import Config
from core.memory import AsyncMemory
from core.audio_handler import AudioEngine
from utils.metrics import perf_tracker

# --- STREAMLIT UI SETUP ---
st.set_page_config(
    page_title="JARVIS - Local Voice Portal", 
    page_icon="🤖",
    layout="wide"
)

# --- BACKEND INITIALIZATION ---
# @st.cache_resource tells Streamlit to handle the multi-threading safety for us.
# It guarantees this function executes exactly once, preventing Qdrant file-lock collisions.
@st.cache_resource
def initialize_local_systems():
    print("[System] Performing unified thread-safe systems boot...")
    core_memory = AsyncMemory()
    core_audio = AudioEngine(memory_manager=core_memory)
    return core_memory, core_audio

# Safely extract our synchronized engine references
memory, audio = initialize_local_systems()

# --- SIDEBAR: CONTROLLER WITH MIDWAY INTERRUPT SWITCHES ---
with st.sidebar:
    st.header("🎛️ System Controller")
    st.markdown(f"**Push-To-Talk Hotkey:** `{Config.GLOBAL_PTT_HOTKEY.upper()}`")
    
    st.write("---")
    
    speaker_switch = st.toggle("🔊 Speaker Output Enabled", value=audio.speaker_enabled)
    audio.speaker_enabled = speaker_switch  
    
    if not speaker_switch:
        st.warning("🔇 Speaker output muted. Active voice playbacks killed.")
        
    st.write("---")
    st.header("📊 Performance Monitor")
    
    metrics_data = perf_tracker.get_all_metrics() if hasattr(perf_tracker, "get_all_metrics") else {}
    if metrics_data:
        for metric_name, timing_val in metrics_data.items():
            st.metric(label=metric_name, value=f"{timing_val:.3f}s")
    else:
        st.markdown("* **Router Math Latency:** ~0.020s (FastEmbed)")
        st.markdown("* **Ear Latency (STT):** Pending input...")
        st.markdown("* **Mouth Latency (TTS):** Pending input...")

    st.write("---")
    if st.button("🔄 Refresh Interface State", use_container_width=True):
        st.rerun()

# --- MAIN CONVERSATION INTERFACE ---
st.title("🤖 JARVIS: Unified Voice & Web Portal")
st.caption("Lesson 14: Integrating Amalgamated Agnostic Speed Engine, Audio Senses, and Streamlit Frontend")
st.header("💬 Active Conversation Space")

# 🌟 THE AUTO-REFRESH FIX: Wrap the history renderer inside an isolated fragment loop.
# It automatically repolls our shared RAM state every 0.5 seconds, flashing your spoken text 
# and JARVIS's voice responses directly onto your screen without requiring manual interactions.
@st.fragment(run_every=0.5)
def render_conversation_stream():
    if len(memory.raw_history) == 0:
        st.info("The conversation tree is currently empty. Say something or tap the hotkey combination to activate JARVIS.")
    else:
        for message in memory.raw_history:
            role = message["role"]
            content = message["content"]
            
            with st.chat_message(role):
                if role == "user":
                    st.markdown(f"👤 **You:** {content}")
                else:
                    st.markdown(f"🤖 **JARVIS:** {content}")

# Fire our active stream container
render_conversation_stream()

# --- TEXT ENTRY FALLBACK ROUTINE ---
keyboard_query = st.chat_input("Or type your message here if you prefer keyboard interaction...")

if keyboard_query:
    # 1. Update historical state arrays
    memory.add_user_message(keyboard_query)
    
    with st.chat_message("user"):
        st.markdown(f"👤 **You:** {keyboard_query}")
        
    # 2. Generate response text
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("🤖 *JARVIS is thinking...*")
        
        payload = memory.get_context_payload(keyboard_query)
        response_text = memory.brain.process_background_task(keyboard_query)
        response_placeholder.markdown(f"🤖 **JARVIS:** {response_text}")
        
    # 3. Save response statement cleanly to core memory
    memory.add_assistant_message(response_text)
    
    # 4. Synthesize and handle async hardware playback
    response_wav = os.path.join(Config.AUDIO_TEMP_DIR, "response.wav")
    audio.speak(response_text, response_wav)
    
    if audio.speaker_enabled:
        audio.play_audio_async(response_wav)
        
    st.rerun()