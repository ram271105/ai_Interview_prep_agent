"""
Resume Upload Component
-----------------------
Handles PDF resume upload, text extraction, and vector store creation.
Shows extraction progress, resume preview, and readiness status.
"""

import streamlit as st
import time

from utils.pdf_extractor import extract_text_from_pdf, chunk_text, extract_resume_highlights
from utils.vector_store import create_vector_store


def render_upload_page():
    """
    Render the resume upload page with drag-and-drop upload,
    extraction progress, and resume preview.
    """
    # Page header
    st.markdown("""
    <div class="page-header">
        <h1>📄 Upload Your Resume</h1>
        <p class="subtitle">Upload your PDF resume to get started with personalized interview preparation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose your resume (PDF)",
            type=["pdf"],
            help="Upload a PDF resume. We'll extract the text and use it to generate personalized interview questions.",
            key="pdf_uploader"
        )
        
        if uploaded_file is not None:
            _process_upload(uploaded_file)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use a **text-based PDF** (not scanned images)
        - Include your **skills, experience, and education**
        - More detail = better questions
        - Supported format: **PDF only**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show resume preview if already uploaded
    if st.session_state.get("resume_text"):
        _show_resume_preview()


def _process_upload(uploaded_file):
    """
    Process the uploaded PDF file: extract text, chunk it, and create vector store.
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
    """
    # Skip if already processed this file
    if st.session_state.get("resume_name") == uploaded_file.name and st.session_state.get("resume_text"):
        st.success(f"✅ Resume '{uploaded_file.name}' already loaded!")
        return
    
    st.markdown("---")
    st.markdown("### 🔄 Processing Resume...")
    
    # Step 1: Extract text
    progress_bar = st.progress(0, text="Extracting text from PDF...")
    time.sleep(0.3)  # Small delay for visual effect
    
    resume_text = extract_text_from_pdf(uploaded_file)
    
    if not resume_text:
        st.error("❌ Could not extract text from the PDF. Please make sure it's a text-based PDF (not a scanned image).")
        progress_bar.empty()
        return
    
    progress_bar.progress(33, text="✅ Text extracted! Chunking text...")
    time.sleep(0.3)
    
    # Step 2: Chunk the text
    chunks = chunk_text(resume_text)
    
    if not chunks:
        st.error("❌ Could not split the resume text into chunks.")
        progress_bar.empty()
        return
    
    progress_bar.progress(66, text="✅ Text chunked! Creating embeddings...")
    
    # Step 3: Create vector store
    with st.spinner("Generating embeddings and storing in ChromaDB..."):
        vector_store = create_vector_store(chunks)
    
    if not vector_store:
        st.error("❌ Failed to create the vector store. Please try again.")
        progress_bar.empty()
        return
    
    progress_bar.progress(100, text="✅ Resume processed successfully!")
    time.sleep(0.5)
    progress_bar.empty()
    
    # Save to session state
    st.session_state.resume_text = resume_text
    st.session_state.resume_name = uploaded_file.name
    st.session_state.vector_store = vector_store
    st.session_state.resume_chunks = chunks
    
    st.success(f"✅ Resume '{uploaded_file.name}' processed successfully!")
    st.info(f"📊 Extracted {len(resume_text)} characters | {len(chunks)} chunks created | Embeddings stored in ChromaDB")


def _show_resume_preview():
    """
    Display a preview of the extracted resume text with highlights.
    """
    st.markdown("---")
    st.markdown("### 📋 Resume Preview")
    
    resume_text = st.session_state.resume_text
    highlights = extract_resume_highlights(resume_text)
    
    # Metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("📝 Words", f"{highlights['total_words']:,}")
    with m2:
        st.metric("📄 Lines", highlights["total_lines"])
    with m3:
        st.metric("🧩 Chunks", len(st.session_state.get("resume_chunks", [])))
    
    # Expandable text preview
    with st.expander("📖 View Full Resume Text", expanded=False):
        st.text(resume_text)
    
    _show_start_button()


def _show_start_button():
    """
    Show the 'Start Interview' button if resume is ready.
    """
    st.markdown("---")
    if st.button("🚀 Start Interview", type="primary", use_container_width=True, key="start_interview_btn"):
        st.session_state.current_page = "🎤 Interview"
        st.rerun()
