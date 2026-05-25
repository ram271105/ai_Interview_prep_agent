"""
Sidebar Navigation Component
-----------------------------
Provides the main navigation sidebar with page selection, interview settings,
session info, and Ollama connection status.
"""

import streamlit as st
from config import (
    APP_TITLE,
    INTERVIEW_MODES,
    DEFAULT_NUM_QUESTIONS,
    MIN_QUESTIONS,
    MAX_QUESTIONS,
    MODE_MIXED,
)
from utils.llm_chain import check_ollama_connection


def render_sidebar():
    """
    Render the sidebar with navigation and settings.
    Updates session state based on user selections.
    """
    with st.sidebar:
        # ----- App Branding -----
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 1.8rem; background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;">🎯 AI Interview Prep</h1>
            <p style="color: #888; font-size: 0.85rem;">Powered by Ollama + LangChain</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ----- Ollama Status -----
        ollama_connected = check_ollama_connection()
        if ollama_connected:
            st.success("🟢 Ollama Connected", icon="✅")
        else:
            st.error("🔴 Ollama Disconnected", icon="❌")
            st.caption("Make sure Ollama is running with `llama3` model.")
        
        st.markdown("---")
        
        # ----- Navigation -----
        st.markdown("### 📍 Navigation")
        
        # Determine available pages based on state
        pages = ["📄 Upload Resume", "🎤 Interview", "📊 Results"]
        
        # Set default page
        if "current_page" not in st.session_state:
            st.session_state.current_page = pages[0]
        
        selected_page = st.radio(
            "Go to:",
            pages,
            index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0,
            label_visibility="collapsed",
            key="nav_radio"
        )
        st.session_state.current_page = selected_page
        
        st.markdown("---")
        
        # ----- Interview Settings -----
        st.markdown("### ⚙️ Interview Settings")
        
        # Interview mode selector
        if "interview_mode" not in st.session_state:
            st.session_state.interview_mode = MODE_MIXED
        
        mode = st.selectbox(
            "Interview Mode",
            INTERVIEW_MODES,
            index=INTERVIEW_MODES.index(st.session_state.interview_mode),
            key="mode_select"
        )
        st.session_state.interview_mode = mode
        
        # Number of questions
        if "num_questions" not in st.session_state:
            st.session_state.num_questions = DEFAULT_NUM_QUESTIONS
        
        num_q = st.slider(
            "Number of Questions",
            min_value=MIN_QUESTIONS,
            max_value=MAX_QUESTIONS,
            value=st.session_state.num_questions,
            key="num_q_slider"
        )
        st.session_state.num_questions = num_q
        
        st.markdown("---")
        
        # ----- Session Info -----
        st.markdown("### 📋 Session Info")
        
        if st.session_state.get("resume_name"):
            st.markdown(f"**Resume:** {st.session_state.resume_name}")
        else:
            st.markdown("**Resume:** Not uploaded")
        
        # Show progress
        questions_answered = len(st.session_state.get("evaluations", []))
        total_questions = len(st.session_state.get("questions", []))
        
        if total_questions > 0:
            st.markdown(f"**Progress:** {questions_answered}/{total_questions} answered")
            st.progress(questions_answered / total_questions)
        
        st.markdown("---")
        
        # ----- New Session Button -----
        if st.button("🔄 New Session", use_container_width=True, type="secondary"):
            _reset_session()
            st.rerun()
        
        # ----- Footer -----
        st.markdown("""
        <div style="text-align: center; padding-top: 2rem; opacity: 0.5;">
            <small>Built with Streamlit + LangChain + Ollama</small>
        </div>
        """, unsafe_allow_html=True)


def _reset_session():
    """
    Reset all session state variables for a fresh start.
    """
    keys_to_reset = [
        "resume_text", "resume_name", "vector_store",
        "questions", "current_question_idx", "evaluations",
        "answers", "interview_started", "interview_complete"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.current_page = "📄 Upload Resume"
