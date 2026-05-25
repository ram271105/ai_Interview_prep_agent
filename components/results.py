"""
Results & Report Component
--------------------------
Displays the comprehensive interview session report with scores,
feedback, performance visualization, and action items.
"""

import streamlit as st

from utils.evaluator import generate_session_report
from config import SCORE_EXCELLENT, SCORE_GOOD, SCORE_FAIR


def render_results_page():
    """
    Render the results page with session report and detailed breakdown.
    """
    # Check if there are results to show
    if not st.session_state.get("evaluations"):
        st.warning("⚠️ No interview results yet. Complete an interview first.")
        if st.button("🎤 Start Interview", type="primary"):
            st.session_state.current_page = "🎤 Interview"
            st.rerun()
        return
    
    # Page header
    st.markdown("""
    <div class="page-header">
        <h1>📊 Interview Results</h1>
        <p class="subtitle">Your performance summary and detailed feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate session report
    all_evals = [e["evaluation"] for e in st.session_state.evaluations]
    report = generate_session_report(all_evals)
    
    # Overall score section
    _show_overall_score(report)
    
    # Score distribution
    _show_score_distribution(report)
    
    # Strengths and improvements
    _show_strengths_improvements(report)
    
    # Detailed breakdown
    _show_detailed_breakdown()
    
    # Action buttons
    _show_action_buttons()


def _show_overall_score(report: dict):
    """
    Display the overall score with a visual gauge.
    """
    avg_score = report["average_score"]
    level = report["performance_level"]
    
    # Score color
    if avg_score >= SCORE_EXCELLENT:
        color = "#4CAF50"
    elif avg_score >= SCORE_GOOD:
        color = "#2196F3"
    elif avg_score >= SCORE_FAIR:
        color = "#FF9800"
    else:
        color = "#f44336"
    
    # Animated score display
    st.markdown(f"""
    <div class="score-hero">
        <div class="score-circle" style="border-color: {color};">
            <span class="score-value" style="color: {color};">{avg_score}</span>
            <span class="score-label">/ 10</span>
        </div>
        <div class="score-level">{level}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Questions Answered", report["questions_answered"])
    with col2:
        st.metric("📊 Average Score", f"{avg_score}/10")
    with col3:
        st.metric("🎯 Total Questions", report["total_questions"])


def _show_score_distribution(report: dict):
    """
    Show score distribution as a horizontal bar chart.
    """
    st.markdown("---")
    st.markdown("### 📈 Score Distribution")
    
    scores = report.get("individual_scores", [])
    if scores:
        # Create a simple bar chart using streamlit
        import pandas as pd
        chart_data = pd.DataFrame({
            "Question": [f"Q{i+1}" for i in range(len(scores))],
            "Score": scores
        })
        st.bar_chart(chart_data.set_index("Question"), color="#667eea")
    
    # Distribution summary
    dist = report.get("score_distribution", {})
    if dist:
        cols = st.columns(len(dist))
        for col, (label, count) in zip(cols, dist.items()):
            with col:
                st.metric(label, count)


def _show_strengths_improvements(report: dict):
    """
    Display aggregated strengths and areas for improvement.
    """
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Key Strengths")
        strengths = report.get("all_strengths", [])
        if strengths:
            for s in strengths:
                st.markdown(f"- ✨ {s}")
        else:
            st.markdown("_Complete more questions for strength analysis._")
    
    with col2:
        st.markdown("### 🔧 Areas to Improve")
        improvements = report.get("all_improvements", [])
        if improvements:
            for imp in improvements:
                st.markdown(f"- 📌 {imp}")
        else:
            st.markdown("_Complete more questions for improvement suggestions._")


def _show_detailed_breakdown():
    """
    Show detailed per-question breakdown with expandable sections.
    """
    st.markdown("---")
    st.markdown("### 📋 Question-by-Question Breakdown")
    
    evaluations = st.session_state.get("evaluations", [])
    
    for i, eval_data in enumerate(evaluations):
        question = eval_data["question"]
        answer = eval_data["answer"]
        evaluation = eval_data["evaluation"]
        score = evaluation.get("score", 0)
        
        # Color based on score
        if score >= SCORE_EXCELLENT:
            color = "#4CAF50"
            icon = "🌟"
        elif score >= SCORE_GOOD:
            color = "#2196F3"
            icon = "✅"
        elif score >= SCORE_FAIR:
            color = "#FF9800"
            icon = "⚡"
        else:
            color = "#f44336"
            icon = "📚"
        
        with st.expander(f"{icon} Q{i+1}: {question['question'][:80]}... — **{score}/10**"):
            st.markdown(f"**Type:** {question.get('type', 'N/A')} | **Difficulty:** {question.get('difficulty', 'N/A')} | **Focus:** {question.get('focus_area', 'N/A')}")
            
            st.markdown("---")
            st.markdown("**Your Answer:**")
            st.markdown(f"> {answer}")
            
            st.markdown("---")
            st.markdown(f"**Score:** <span style='color: {color}; font-size: 1.3rem; font-weight: 700;'>{score}/10</span>", unsafe_allow_html=True)
            st.markdown(f"**Feedback:** {evaluation.get('overall_feedback', '')}")
            
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                st.markdown("**Strengths:**")
                for s in evaluation.get("strengths", []):
                    st.markdown(f"- {s}")
            with fcol2:
                st.markdown("**Improvements:**")
                for imp in evaluation.get("improvements", []):
                    st.markdown(f"- {imp}")


def _show_action_buttons():
    """
    Show action buttons for retry, new resume, etc.
    """
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Retry Interview", use_container_width=True, type="primary"):
            # Keep resume, regenerate questions
            for key in ["questions", "current_question_idx", "evaluations", "answers", "interview_started", "interview_complete", "show_feedback"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_page = "🎤 Interview"
            st.rerun()
    
    with col2:
        if st.button("📄 New Resume", use_container_width=True, type="secondary"):
            # Full reset
            for key in list(st.session_state.keys()):
                if key not in ["nav_radio"]:
                    del st.session_state[key]
            st.session_state.current_page = "📄 Upload Resume"
            st.rerun()
    
    with col3:
        if st.button("📥 Download Report", use_container_width=True, type="secondary"):
            _download_report()


def _download_report():
    """
    Generate and offer a downloadable text report.
    """
    evaluations = st.session_state.get("evaluations", [])
    all_evals = [e["evaluation"] for e in evaluations]
    report = generate_session_report(all_evals)
    
    # Build report text
    lines = [
        "=" * 60,
        "AI INTERVIEW PREPARATION - SESSION REPORT",
        "=" * 60,
        f"\nOverall Score: {report['average_score']}/10",
        f"Performance Level: {report['performance_level']}",
        f"Questions Answered: {report['questions_answered']}/{report['total_questions']}",
        "\n" + "-" * 60,
        "DETAILED BREAKDOWN",
        "-" * 60,
    ]
    
    for i, eval_data in enumerate(evaluations):
        q = eval_data["question"]
        a = eval_data["answer"]
        e = eval_data["evaluation"]
        lines.extend([
            f"\nQ{i+1}: {q['question']}",
            f"Type: {q.get('type', 'N/A')} | Difficulty: {q.get('difficulty', 'N/A')}",
            f"Your Answer: {a}",
            f"Score: {e.get('score', 'N/A')}/10",
            f"Feedback: {e.get('overall_feedback', '')}",
            "---"
        ])
    
    report_text = "\n".join(lines)
    
    st.download_button(
        label="📥 Download Report (.txt)",
        data=report_text,
        file_name="interview_report.txt",
        mime="text/plain"
    )
