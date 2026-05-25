"""
Configuration settings for the AI Interview Preparation Agent.
Centralizes all configurable parameters for easy modification.
"""
import os

# =============================================================================
# Ollama LLM Configuration
# =============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TEMPERATURE = 0.7
OLLAMA_TOP_P = 0.9
OLLAMA_NUM_CTX = 4096  # Context window size

# =============================================================================
# Embedding Model Configuration
# =============================================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model
EMBEDDING_DEVICE = "cpu"  # Use 'cuda' if GPU available

# =============================================================================
# ChromaDB Vector Store Configuration
# =============================================================================
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "resume_embeddings"

# =============================================================================
# Text Processing Configuration
# =============================================================================
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
RETRIEVAL_TOP_K = 4  # Number of chunks to retrieve

# =============================================================================
# Interview Configuration
# =============================================================================
DEFAULT_NUM_QUESTIONS = 5
MIN_QUESTIONS = 3
MAX_QUESTIONS = 15

# Interview modes
MODE_HR = "HR & Behavioral"
MODE_TECHNICAL = "Technical"
MODE_MIXED = "Mixed"
INTERVIEW_MODES = [MODE_HR, MODE_TECHNICAL, MODE_MIXED]

# =============================================================================
# Whisper Speech-to-Text Configuration
# =============================================================================
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
AUDIO_SAMPLE_RATE = 16000  # Hz
AUDIO_CHANNELS = 1
DEFAULT_RECORD_DURATION = 30  # seconds

# =============================================================================
# UI Configuration
# =============================================================================
APP_TITLE = "🎯 AI Interview Prep Agent"
APP_ICON = "🎯"
APP_LAYOUT = "wide"

# Score thresholds for color coding
SCORE_EXCELLENT = 8  # >= 8 is excellent (green)
SCORE_GOOD = 6       # >= 6 is good (blue)
SCORE_FAIR = 4        # >= 4 is fair (orange)
# < 4 is needs improvement (red)
