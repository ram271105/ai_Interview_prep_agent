"""
AI Interview Preparation Agent — Main Application
==================================================
A Streamlit-based AI interview prep tool using:
- LangChain for RAG orchestration
- Ollama (llama3) for local LLM
- ChromaDB for vector storage
- PyPDF2 for resume extraction
- Whisper for optional speech-to-text

Entry point: `streamlit run app.py`
"""

import os
import sys
import logging
import streamlit as st

# ── Ensure project root is in the Python path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import APP_TITLE, APP_ICON, APP_LAYOUT
from components.sidebar import render_sidebar
from components.upload import render_upload_page
from components.interview import render_interview_page
from components.results import render_results_page

# ── Configure logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration — MUST be the first Streamlit command
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Interview Prep Agent",
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI-powered interview preparation using RAG, LangChain, and Ollama."
    }
)


def load_custom_css():
    """Load the custom CSS stylesheet for the application."""
    css_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"Custom CSS not found at {css_path}")


def init_session_state():
    """
    Initialize all session state variables with sensible defaults.
    Called once at the start of each session.
    """
    defaults = {
        # Navigation
        "current_page": "📄 Upload Resume",
        
        # Resume state
        "resume_text": None,
        "resume_name": None,
        "resume_chunks": [],
        "vector_store": None,
        
        # Interview state
        "interview_mode": "Mixed",
        "num_questions": 5,
        "questions": [],
        "current_question_idx": 0,
        "answers": [],
        "evaluations": [],
        "interview_started": False,
        "interview_complete": False,
        "show_feedback": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def route_page():
    """
    Route to the appropriate page component based on the current navigation state.
    """
    current = st.session_state.current_page
    
    if current == "📄 Upload Resume":
        render_upload_page()
    elif current == "🎤 Interview":
        render_interview_page()
    elif current == "📊 Results":
        render_results_page()
    else:
        # Fallback to upload page
        render_upload_page()


def main():
    """Main application entry point."""
    # Load custom styles
    load_custom_css()
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar navigation
    render_sidebar()
    
    # Route to the selected page
    route_page()


if __name__ == "__main__":
    main()
