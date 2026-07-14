import os
from dotenv import load_dotenv

load_dotenv(override=True)

# --- DYNAMIC BASE PATHS ---
# Anchors paths relative to this config file so it works on any OS
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CORE_DIR, "..", "..", "data"))
TEMP_DIR = os.path.join(DATA_DIR, "temp")
LOG_DIR = os.path.join(DATA_DIR, "logs")

# This runs immediately when config.py is imported!
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class Config:
    # Model Paths
    NOMIC_MODEL_PATH = os.getenv("NOMIC_MODEL_PATH")
    CHAT_MODEL_PATH = os.getenv("CHAT_MODEL_PATH")
    
    # Engine Parameters
    EMBEDDING_CTX_LENGTH = 2048

    # --- LLM (Brain) ---
    LLM_MODEL = "local-model" 
    LLM_BASE_URL = "http://localhost:1234/v1"
    LLM_API_KEY = "lm-studio"
    
    # Brain Tuning
    LLM_CHAT_TEMPERATURE = 0.3
    LLM_CHAT_MAX_TOKENS = 1024
    LLM_TASK_TEMPERATURE = 0.0
    LLM_TASK_MAX_TOKENS = 1024
    
    # --- RAG & MEMORY ---
    EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5@q8_0"
    COLLECTION_NAME = "jarvis_memory"
    QDRANT_STORAGE_PATH = os.path.join(DATA_DIR, "qdrant_storage")
    FASTEMBED_CACHE_DIR = os.path.join(DATA_DIR, "fastembed_cache")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # --- AUDIO (Ear & Mouth) ---
    STT_MODEL = "small"             
    EAR_DEVICE = "cpu"
    EAR_COMPUTE_TYPE = "int8"   
    PIPER_DIR = os.path.join(DATA_DIR, "piper")
    PIPER_MODEL_PATH = os.path.join(DATA_DIR, "voices", "piper-lessac.onnx")
    AUDIO_TEMP_DIR = TEMP_DIR
    
    # Unified PyAudio and Keyboard Stream Parameters
    AUDIO_SAMPLE_RATE = 16000      
    AUDIO_CHANNELS = 1             
    AUDIO_CHUNK_SIZE = 1024        
    GLOBAL_PTT_HOTKEY = "ctrl+shift+space"
    HOTKEY_KILL = "esc"
    
    # Centralized Audio Engine Variables
    AUDIO_PLAYBACK_CHUNK = 2048              # Output buffer to prevent choppy audio
    AUDIO_DEBOUNCE_TIME = 0.5                # Minimum delay (seconds) between key taps
    AUDIO_PARAGRAPH_SILENCE_DURATION = 1.0   # Pause length (seconds) between paragraphs
    AUDIO_MAX_CONSECUTIVE_ERRORS = 10        # Safety threshold to shut down dead streams
    STT_BEAM_SIZE = 1                        # 1 = Greedy search (stops hallucinations)
    STT_LANGUAGE = "en"                      # Forces Whisper to listen only for English

    # --- VAD (Voice Activity Detection) TUNING ---
    EAR_AMBIENT_DURATION = 0.5
    EAR_PAUSE_THRESHOLD = 2.0
    EAR_TIMEOUT = 10
    EAR_PHRASE_LIMIT = 30
    
    # --- EXTERNAL DEPENDENCIES ---
    TESSERACT_CMD_PATH = os.getenv("TESSERACT_CMD_PATH")

    # --- PROMPT ENGINEERING ---
    SYSTEM_PROMPT = (
        "You are JARVIS, a highly intelligent and concise AI. "
        "Your default responses should be in English, unless otherwise instructed."
        "When the user says 'I', 'me', or 'my', they are referring to themselves. "
        "The user may also extract document text and provide it to you in the context block below. "
        "Use the provided context to answer questions about the user or the provided documents. "
        "If the answer is not in the context below, DO NOT guess, do not make up an answer, and do not apologize about your capabilities. Say you don't know."
    )
    UPDATE_SUMMARY_PROMPT = "Summarize the key points of the conversation so far in one short paragraph."
    
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
    CONTEXT_INJECTION_TEMPLATE = "\n\n[FACTS AND CONTEXT ABOUT THE USER OR DOCUMENTS]\n{context}"
    SUMMARY_INJECTION_TEMPLATE = "\n\n[CONVERSATION SUMMARY]\n{summary}"

    # --- SEMANTIC ROUTER ANCHORS ---
    ROUTER_RECALL_ANCHORS = [
        "What did I say earlier?",
        "Do you remember my name?",
        "What did I tell you about my name earlier?",
        "What was that fact I told you?",
        "What have I told you",
        "Summarize chapter 1 from the document.",
        "What did the file report.pdf say about this?",
        "Give me a breakdown of the specific pdf file.",
        "Summarize the main points of the attached document.",
        "Summarize the file report.pdf for me.",
        "What are the key points of the document?",
        "Please summarize the attached document.",
        "Could you give me a summary of chapter 1?"
    ]

    ROUTER_SUMMARY_ANCHORS = [
        "Summarize this text.",
        "What is the summary of this chat",
        "Give me the TLDR.",
        "Give me a recap of what we just talked about",
        "I would appreciate it if you could recap our conversation."
    ]   

    ROUTER_CONFIDENCE_THRESHOLD = 0.65
    
    # --- ROUTING ENGINE ---
    USE_SEMANTIC_ROUTER = True       
    USE_LOCAL_CPP_ROUTER = True       

    # --- AUTOMATED TESTING SUITE ---
    TARGET_TEST_MODELS = [
        "google/gemma-3-4b",
        "ibm/granite-4-h-tiny",
        "llama-3.2-3b-instruct",
        "mistralai/ministral-3-3b-instruct-2512@q4_k_m",
        "mistralai/ministral-3-3b-instruct-2512@q5_k_m",
        "mistralai/ministral-3-3b-instruct-2512@q8_0",
        "nvidia/nemotron-3-nano-4b",
        "phi-3-mini-4k-instruct",
        "microsoft/phi-4-mini-reasoning",
        "qwen/qwen3-4b",
        "qwen/qwen3-4b-2507"
    ]

    TEST_PROMPTS = [
        "What is my name?",  
        "What are the key points of the document?",  
        "Can you explain the theory of relativity in one paragraph?",  
        "Please summarize our conversation so far."  
    ]

    EM_WRAP_TEST_PROMPT = "What is the capital of France?"
    SEMANTIC_TEST_QUERY = "search_query: What is the capital of France?"
    SEMANTIC_TEST_DOCS = [
        "search_query: Paris is the capital and most populous city of France.",
        "search_query: The Eiffel Tower is a famous landmark in Paris.",
        "search_query: I really enjoy eating fresh green apples in the morning."
    ]
    
    LOG_USER_PROMPT = True
    LOG_LLM_RESPONSE = True

    # --- METRICS & TELEMETRY ---
    ENABLE_METRICS = True
    TELEMETRY_CSV_NAME = "benchmark_report.csv"
    TELEMETRY_CSV_PATH = os.path.join(LOG_DIR, TELEMETRY_CSV_NAME)
    TELEMETRY_POLL_INTERVAL = 0.5  

    # --- LLM TASK TUNING ---
    LLM_ROUTING_MAX_TOKENS = 10
    LLM_DECONSTRUCT_MAX_TOKENS = 15
    LLM_SUMMARY_MAX_TOKENS = 100
    
    # --- COGNITIVE TUNING (The "Brain" Math) ---
    MEMORY_DECAY_FLOOR = 0.5        
    MEMORY_DECAY_RATE = 0.01        
    MEMORY_REINFORCE_WEIGHT = 0.1   
    
    # --- FILTERS & HEURISTICS ---
    MIN_CHUNK_CHARACTER_COUNT = 50
    SUMMARY_TRIGGER_TURN_COUNT = 5
    CONTEXT_CHUNKS_LIMIT = 5
    VECTOR_SEARCH_LIMIT = 7  

# This runs immediately when config.py is imported!
os.makedirs(TEMP_DIR, exist_ok=True)