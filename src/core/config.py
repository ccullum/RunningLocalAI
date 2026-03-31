import os
from dotenv import load_dotenv

load_dotenv()

# --- DYNAMIC BASE PATHS ---
# Anchors paths relative to this config file so it works on any OS
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CORE_DIR, "..", "..", "data"))
TEMP_DIR = os.path.join(DATA_DIR, "temp")

class Config:
    # --- LLM (Brain) ---
    LLM_MODEL = "local-model" 
    LLM_BASE_URL = "http://localhost:1234/v1"
    LLM_API_KEY = "lm-studio"
    
    # Brain Tuning
    LLM_CHAT_TEMPERATURE = 0.3
    LLM_CHAT_MAX_TOKENS = 1024
    LLM_TASK_TEMPERATURE = 0.0
    LLM_TASK_MAX_TOKENS = 50
    
    # --- RAG & MEMORY ---
    EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5@q8_0"
    COLLECTION_NAME = "jarvis_memory"
    QDRANT_STORAGE_PATH = os.path.join(DATA_DIR, "qdrant_storage")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # --- AUDIO (Ear & Mouth) ---
    STT_MODEL = "tiny.en"
    EAR_DEVICE = "cpu"
    EAR_COMPUTE_TYPE = "int8"   
    PIPER_DIR = os.path.join(DATA_DIR, "piper")
    PIPER_MODEL_PATH = os.path.join(DATA_DIR, "voices", "piper-lessac.onnx")
    HOTKEY_KILL = 'esc'
    AUDIO_TEMP_DIR = TEMP_DIR

    # --- VAD (Voice Activity Detection) TUNING ---
    EAR_AMBIENT_DURATION = 0.5
    EAR_PAUSE_THRESHOLD = 2.0
    EAR_TIMEOUT = 10
    EAR_PHRASE_LIMIT = 30
    
    # --- EXTERNAL DEPENDENCIES ---
    TESSERACT_CMD_PATH = os.getenv("TESSERACT_CMD_PATH")

    # --- PROMPT ENGINEERING ---
    # Prompts
    SYSTEM_PROMPT = (
        "You are JARVIS, a highly intelligent and concise AI. "
        "When the user says 'I', 'me', or 'my', they are referring to themselves. "
        "The user may also extract document text and provide it to you in the context block below. "
        "Use the provided context to answer questions about the user or the provided documents. "
        "If the answer is not in the context below, DO NOT guess, do not make up an answer, and do not apologize about your capabilities. Say you don't know."
    )
    UPDATE_SUMMARY_PROMPT = "Summarize the key points of the conversation so far in one short paragraph."
    # Templates
    DECONSTRUCT_QUERY_TEMPLATE = (
        "You are a database query generator. "
        "Convert the following question into a declarative statement to search a database. "
        "Example: \"What is my name?\" -> \"The user's name is\" "
        "If the input is NOT a question (e.g., \"Thank you\", \"Yes it is\"), respond with exactly the word: SKIP. "
        "Question: \"{user_query}\" "
        "RESPOND WITH THE RAW STATEMENT ONLY. NO CONVERSATIONAL FILLER."
    )
    LLM_FALLBACK_QUERY_TEMPLATE = (
        "You are a strict internal routing agent. Analyze the user's query: \"{user_query}\" "
        "1. RECALL: Use if the user asks to remember a specific detail, fact, or something mentioned in the past."
        "2. SUMMARY: Use ONLY if the user is asking for a broad recap of the chat."
        "3. CHAT: Use for EVERYTHING ELSE."
        "Respond with EXACTLY ONE WORD: [RECALL, SUMMARY, or CHAT]."
    )
    # Structural Injection Templates
    CONTEXT_INJECTION_TEMPLATE = "\n\n[FACTS AND CONTEXT ABOUT THE USER OR DOCUMENTS]\n{context}"
    SUMMARY_INJECTION_TEMPLATE = "\n\n[CONVERSATION SUMMARY]\n{summary}"

    # --- SEMANTIC ROUTER ANCHORS ---
    # These dictate the "center of gravity" for each intent
    ROUTER_RECALL_ANCHORS = [
        "What did the document say about",
        "Search my files for",
        "Do you remember when I told you",
        "Look up the PDF",
        "What was that fact I asked you to remember",
        "Can you find the reference to",
        "According to the text I uploaded"
    ]
    
    ROUTER_SUMMARY_ANCHORS = [
        "Can you summarize our conversation",
        "Give me a recap of what we just talked about",
        "What is the summary of this chat",
        "Bring me up to speed on our discussion"
    ]
    
    # We define a strict similarity threshold. 
    # If the user's prompt doesn't score at least this high against ANY anchor, it safely defaults to CHAT.
    ROUTER_CONFIDENCE_THRESHOLD = 0.65
    
    # --- ROUTING ENGINE ---
    # True = Lightning-fast math router. False = Old 8B LLM router.
    USE_SEMANTIC_ROUTER = True

    # --- METRICS & TELEMETRY ---
    ENABLE_METRICS = True

# This runs immediately when config.py is imported!
os.makedirs(TEMP_DIR, exist_ok=True)