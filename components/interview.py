"""
Interview Session Component
---------------------------
Manages the interactive interview session with question display,
text/voice input, answer submission, and real-time feedback.
"""

import streamlit as st
import time

from utils.llm_chain import generate_questions
from utils.evaluator import evaluate_answer
from utils.speech import is_whisper_available, is_recording_available, transcribe_uploaded_audio
from config import SCORE_EXCELLENT, SCORE_GOOD, SCORE_FAIR


def render_interview_page():
    """
    Render the interview session page.
    Handles question generation, answer input, and evaluation flow.
    """
    # Check prerequisites
    if not st.session_state.get("resume_text"):
        st.warning("⚠️ Please upload your resume first.")
        if st.button("📄 Go to Upload", type="primary"):
            st.session_state.current_page = "📄 Upload Resume"
            st.rerun()
        return
    
    # Page header
    st.markdown("""
    <div class="page-header">
        <h1>🎤 Interview Session</h1>
        <p class="subtitle">Answer questions naturally — you'll get instant AI feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate questions if not already done
    if not st.session_state.get("questions"):
        _generate_interview_questions()
        return
    
    # Initialize tracking variables
    if "current_question_idx" not in st.session_state:
        st.session_state.current_question_idx = 0
    if "evaluations" not in st.session_state:
        st.session_state.evaluations = []
    if "answers" not in st.session_state:
        st.session_state.answers = []
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
    
    questions = st.session_state.questions
    current_idx = st.session_state.current_question_idx
    
    # Check if interview is complete
    if current_idx >= len(questions):
        st.session_state.interview_complete = True
        st.session_state.current_page = "📊 Results"
        st.rerun()
        return
    
    # Show progress
    _show_progress_bar(current_idx, len(questions))
    
    # Display current question
    current_q = questions[current_idx]
    _display_question(current_q, current_idx, len(questions))
    
    # Answer input section
    if not st.session_state.show_feedback:
        _show_answer_input(current_q)
    else:
        # Show feedback for the last answer
        _show_feedback()
        _show_navigation_buttons(current_idx, len(questions))


def _generate_interview_questions():
    """
    Generate interview questions using the RAG pipeline.
    Shows a loading animation during generation.
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    mode = st.session_state.get("interview_mode", "Mixed")
    num_q = st.session_state.get("num_questions", 5)
    
    st.markdown(f"### 🔄 Generating {num_q} {mode} Questions...")
    st.markdown("_Analyzing your resume and crafting personalized questions..._")
    
    # Loading animation
    with st.spinner("🤖 AI is preparing your interview questions..."):
        questions = generate_questions(
            vector_store=st.session_state.vector_store,
            mode=mode,
            num_questions=num_q
        )
    
    if not questions:
        st.error("❌ Failed to generate questions. Please check if Ollama is running and try again.")
        if st.button("🔄 Retry", type="primary"):
            st.rerun()
        return
    
    # Store questions in session
    st.session_state.questions = questions
    st.session_state.current_question_idx = 0
    st.session_state.evaluations = []
    st.session_state.answers = []
    st.session_state.interview_started = True
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.rerun()


def _show_progress_bar(current_idx: int, total: int):
    """
    Display the interview progress bar.
    """
    progress = current_idx / total
    st.progress(progress, text=f"Question {current_idx + 1} of {total}")


def _display_question(question: dict, idx: int, total: int):
    """
    Display the current interview question with metadata.
    """
    # Question card
    q_type = question.get("type", "General")
    difficulty = question.get("difficulty", "Medium")
    focus = question.get("focus_area", "")
    
    # Color based on type
    type_color = "#667eea" if q_type == "Technical" else "#764ba2"
    
    # Difficulty badge color
    diff_colors = {"Easy": "#4CAF50", "Medium": "#FF9800", "Hard": "#f44336"}
    diff_color = diff_colors.get(difficulty, "#FF9800")
    
    st.markdown(f"""
    <div class="question-card">
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.8rem; flex-wrap: wrap;">
            <span class="badge" style="background: {type_color};">{q_type}</span>
            <span class="badge" style="background: {diff_color};">{difficulty}</span>
            <span class="badge" style="background: #333; border: 1px solid #555;">{focus}</span>
        </div>
        <h3 style="margin: 0; font-size: 1.2rem; line-height: 1.5;">Q{idx + 1}: {question['question']}</h3>
    </div>
    """, unsafe_allow_html=True)


def _show_answer_input(current_q: dict):
    """
    Show the answer input area with text and optional voice input.
    """
    st.markdown("### ✍️ Your Answer")
    
    # Text input
    answer = st.text_area(
        "Type your answer here:",
        height=200,
        placeholder="Take your time and provide a detailed answer. Use specific examples from your experience...",
        key=f"answer_input_{st.session_state.current_question_idx}",
        label_visibility="collapsed"
    )
    
    # Voice input option
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if is_whisper_available():
            audio_file = st.file_uploader(
                "🎙️ Upload audio",
                type=["wav", "mp3", "m4a"],
                key=f"audio_upload_{st.session_state.current_question_idx}",
                label_visibility="collapsed"
            )
            if audio_file:
                with st.spinner("Transcribing audio..."):
                    transcribed = transcribe_uploaded_audio(audio_file)
                if transcribed:
                    st.info(f"📝 Transcribed: {transcribed[:200]}...")
                    answer = transcribed
        else:
            st.caption("🎙️ Install Whisper for voice input")
    
    with col1:
        # Submit button
        if st.button("📨 Submit Answer", type="primary", use_container_width=True, key=f"submit_{st.session_state.current_question_idx}"):
            if not answer or len(answer.strip()) < 5:
                st.warning("⚠️ Please provide a more detailed answer before submitting.")
            else:
                _submit_answer(answer, current_q)


def _submit_answer(answer: str, question: dict):
    """
    Submit and evaluate the user's answer.
    """
    with st.spinner("🤖 Evaluating your answer..."):
        evaluation = evaluate_answer(
            question=question["question"],
            answer=answer,
            resume_context=st.session_state.resume_text[:2000],
            question_type=question.get("type", "General"),
            difficulty=question.get("difficulty", "Medium")
        )
    
    # Store the answer and evaluation
    st.session_state.answers.append(answer)
    st.session_state.evaluations.append({
        "question": question,
        "answer": answer,
        "evaluation": evaluation
    })
    
    # Show feedback
    st.session_state.show_feedback = True
    st.rerun()


def _show_feedback():
    """
    Display evaluation feedback for the last submitted answer.
    """
    if not st.session_state.evaluations:
        return
    
    last_eval = st.session_state.evaluations[-1]
    evaluation = last_eval["evaluation"]
    score = evaluation.get("score", 0)
    
    # Score color
    if score >= SCORE_EXCELLENT:
        score_color = "#4CAF50"  # Green
        score_emoji = "🌟"
    elif score >= SCORE_GOOD:
        score_color = "#2196F3"  # Blue
        score_emoji = "✅"
    elif score >= SCORE_FAIR:
        score_color = "#FF9800"  # Orange
        score_emoji = "⚡"
    else:
        score_color = "#f44336"  # Red
        score_emoji = "📚"
    
    # Feedback card
    st.markdown(f"""
    <div class="feedback-card" style="border-left: 4px solid {score_color};">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <div style="font-size: 2.5rem;">{score_emoji}</div>
            <div>
                <div style="font-size: 2rem; font-weight: 700; color: {score_color};">{score}/10</div>
                <div style="color: #aaa;">Score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Overall feedback
    st.markdown(f"**💬 Feedback:** {evaluation.get('overall_feedback', '')}")
    
    # Strengths and improvements columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### ✅ Strengths")
        for s in evaluation.get("strengths", []):
            st.markdown(f"- {s}")
    
    with col2:
        st.markdown("##### 🔧 Areas to Improve")
        for imp in evaluation.get("improvements", []):
            st.markdown(f"- {imp}")
    
    # Sample answer points
    if evaluation.get("sample_answer_points"):
        with st.expander("💡 Key Points for a Strong Answer"):
            for point in evaluation["sample_answer_points"]:
                st.markdown(f"- {point}")


def _show_navigation_buttons(current_idx: int, total: int):
    """
    Show navigation buttons after feedback is displayed.
    """
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if current_idx < total - 1:
            if st.button("➡️ Next Question", type="primary", use_container_width=True):
                st.session_state.current_question_idx += 1
                st.session_state.show_feedback = False
                st.rerun()
    
    with col2:
        if st.button(
            "✅ Finish Interview" if current_idx >= total - 1 else "⏭️ Skip to Results",
            use_container_width=True,
            type="primary" if current_idx >= total - 1 else "secondary"
        ):
            st.session_state.interview_complete = True
            st.session_state.current_page = "📊 Results"
            st.rerun()
