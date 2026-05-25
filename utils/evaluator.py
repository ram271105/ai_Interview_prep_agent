"""
Answer Evaluation Module
------------------------
Evaluates user's interview answers using the Ollama LLM.
Provides detailed scoring, feedback, strengths, and improvement suggestions.
"""

import json
import logging
import re
from typing import Dict, List, Optional

from utils.llm_chain import get_ollama_llm

logger = logging.getLogger(__name__)


# Evaluation prompt template
EVALUATION_PROMPT = """You are an expert interview evaluator. Evaluate the candidate's answer to the following interview question.

RESUME CONTEXT (for reference):
{resume_context}

INTERVIEW QUESTION:
{question}

QUESTION TYPE: {question_type}
DIFFICULTY: {difficulty}

CANDIDATE'S ANSWER:
{answer}

Evaluate the answer and return ONLY a valid JSON object with the following fields:
{{
    "score": <integer from 1 to 10>,
    "overall_feedback": "<2-3 sentences of overall assessment>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "improvements": ["<improvement 1>", "<improvement 2>"],
    "sample_answer_points": ["<key point 1 that a strong answer would include>", "<key point 2>"]
}}

Scoring Guide:
- 9-10: Exceptional - Comprehensive, specific, well-structured with great examples
- 7-8: Strong - Good detail and relevance, minor areas for improvement
- 5-6: Adequate - Covers basics but lacks depth or specifics
- 3-4: Needs Work - Vague, off-topic, or missing key elements
- 1-2: Poor - Irrelevant or extremely brief

Return ONLY the JSON object, no other text."""


def calculate_heuristic_score(
    answer: str,
    question: str,
    resume_context: str,
    question_type: str
) -> int:
    """
    Calculate a programmatic score based on response depth, structure,
    STAR indicators, and relevant keywords matching question/resume.
    """
    words = answer.strip().split()
    word_count = len(words)
    
    if word_count < 5:
        return 1
        
    # 1. Base score from word count (max 6 points)
    if word_count < 25:
        base_score = 3
    elif word_count < 55:
        base_score = 4
    elif word_count < 95:
        base_score = 5
    elif word_count < 145:
        base_score = 6
    else:
        base_score = 7
        
    # 2. Structure / Sentence complexity bonus (max 1 point)
    sentences = [s for s in re.split(r'[.!?]+', answer) if s.strip()]
    structure_bonus = 1 if len(sentences) >= 4 else 0
        
    # 3. Domain keyword match bonus (max 1.5 points)
    keywords_to_check = set()
    words_in_q = [w.lower() for w in re.findall(r'\b[a-zA-Z]{4,15}\b', question)]
    q_stops = {'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how', 'this', 'that', 'these', 'those', 'there', 'their', 'about', 'would', 'could', 'should', 'interview', 'question', 'candidate', 'resume', 'experience'}
    for w in words_in_q:
        if w not in q_stops:
            keywords_to_check.add(w)
            
    words_in_resume = [w.lower() for w in re.findall(r'\b[a-zA-Z]{4,15}\b', resume_context[:1000])]
    tech_indicators = {'react', 'node', 'java', 'python', 'javascript', 'html', 'css', 'sql', 'mongodb', 'git', 'api', 'rest', 'docker', 'aws', 'spring', 'express', 'database', 'backend', 'frontend', 'developer', 'engineer', 'project'}
    for w in words_in_resume:
        if w in tech_indicators:
            keywords_to_check.add(w)
            
    answer_lower = answer.lower()
    matches = sum(1 for kw in keywords_to_check if kw in answer_lower)
    keyword_bonus = 1.5 if matches >= 3 else (0.5 if matches >= 1 else 0.0)
        
    # 4. STAR method and impact verbs bonus (max 1.5 points)
    star_keywords = {
        'situation', 'task', 'action', 'result', 'challenge', 'goal', 'resolved', 'solved', 
        'led', 'managed', 'created', 'designed', 'implemented', 'achieved', 'delivered', 
        'improved', 'increased', 'reduced', 'developed', 'coordinated', 'impact', 'metrics', 
        'percent', 'because', 'consequently', 'therefore', 'resulted'
    }
    star_matches = sum(1 for w in star_keywords if w in answer_lower)
    
    if question_type.lower() == 'hr' or 'behavioral' in question_type.lower():
        star_bonus = 1.5 if star_matches >= 4 else (0.8 if star_matches >= 2 else 0.0)
    else:
        tech_explain_keywords = {'explain', 'concept', 'use', 'using', 'example', 'process', 'flow', 'different', 'benefit', 'performance', 'error', 'debug', 'component', 'function', 'class', 'method'}
        tech_matches = sum(1 for w in tech_explain_keywords if w in answer_lower)
        star_bonus = 1.5 if tech_matches >= 4 else (0.8 if tech_matches >= 2 else 0.0)
            
    final_h_score = base_score + structure_bonus + keyword_bonus + star_bonus
    return max(1, min(10, round(final_h_score)))


def compress_score(score: int) -> int:
    """
    Compresses evaluations toward the center of the 1-10 scale (3-8 range)
    as requested by guidelines.
    """
    mapping = {
        1: 3,
        2: 3,
        3: 4,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 7,
        9: 8,
        10: 8
    }
    return mapping.get(score, 5)


def evaluate_answer(
    question: str,
    answer: str,
    resume_context: str,
    question_type: str = "General",
    difficulty: str = "Medium"
) -> Dict:
    """
    Evaluate a candidate's answer using hybrid heuristic scoring and LLM evaluation.
    """
    words = answer.strip().split()
    word_count = len(words)
    
    # Handle empty/short answers
    if not answer or word_count < 5:
        return {
            "score": 3,  # Compressed 1 -> 3
            "overall_feedback": "The answer is too brief to evaluate. Please provide a more detailed response (aim for 30+ words).",
            "strengths": ["Attempted to answer"],
            "improvements": [
                "Provide a more detailed response",
                "Include specific examples from your experience",
                "Structure your answer using the STAR method"
            ],
            "sample_answer_points": ["A complete answer would address the question directly with specific examples"]
        }
        
    # Programmatic heuristic evaluation
    heuristic_score = calculate_heuristic_score(answer, question, resume_context, question_type)
    logger.info(f"Calculated heuristic score: {heuristic_score}/10")
    
    # Standard fallback feedback
    evaluation = {
        "score": compress_score(heuristic_score),
        "overall_feedback": f"Your response is about {word_count} words long and includes some relevant details. To improve, try incorporating more structured achievements and specific action verbs.",
        "strengths": ["Answered the question directly", f"Provided a response of {word_count} words"],
        "improvements": ["Elaborate with specific projects or metrics", "Use a more structured communication approach (e.g. STAR method)"],
        "sample_answer_points": ["Detail specific technical implementations", "Quantify results where possible (e.g., performance improvements, team size)"]
    }
    
    try:
        llm = get_ollama_llm()
        formatted_prompt = EVALUATION_PROMPT.format(
            resume_context=resume_context[:1500],
            question=question,
            question_type=question_type,
            difficulty=difficulty,
            answer=answer
        )
        
        logger.info("Evaluating answer with LLM...")
        response = llm.invoke(formatted_prompt)
        
        parsed_eval = _parse_evaluation_response(response)
        
        if parsed_eval:
            llm_score = parsed_eval.get("score")
            
            if llm_score is not None:
                llm_score = max(1, min(10, int(llm_score)))
                # Blend: 40% heuristic, 60% LLM
                final_score = max(1, min(10, round(0.4 * heuristic_score + 0.6 * llm_score)))
                logger.info(f"Blended score: {final_score}/10 (Heuristic: {heuristic_score}, LLM: {llm_score})")
            else:
                final_score = heuristic_score
                logger.info(f"Using heuristic score: {final_score}/10 (LLM did not return a score)")
                
            evaluation["score"] = compress_score(final_score)
            
            if parsed_eval.get("overall_feedback"):
                evaluation["overall_feedback"] = parsed_eval["overall_feedback"]
            if parsed_eval.get("strengths"):
                evaluation["strengths"] = parsed_eval["strengths"]
            if parsed_eval.get("improvements"):
                evaluation["improvements"] = parsed_eval["improvements"]
            if parsed_eval.get("sample_answer_points"):
                evaluation["sample_answer_points"] = parsed_eval["sample_answer_points"]
        else:
            logger.warning("LLM response parsing failed completely. Using pure heuristic evaluation.")
            
    except Exception as e:
        import traceback
        logger.error(f"Error during LLM evaluation: {str(e)}\n{traceback.format_exc()}")
        
    return evaluation


def _parse_evaluation_response(response: str) -> Optional[Dict]:
    """
    Parse the LLM evaluation response into a structured dictionary.
    Supports regex fallback parsing if standard JSON loading fails.
    """
    response_str = response.strip()
    
    # 1. Try standard JSON parsing
    start_idx = response_str.find("{")
    end_idx = response_str.rfind("}") + 1
    
    if start_idx != -1 and end_idx > start_idx:
        try:
            json_str = response_str[start_idx:end_idx]
            evaluation = json.loads(json_str)
            
            score = evaluation.get("score")
            if score is not None:
                try:
                    score = int(score)
                except ValueError:
                    score = None
                    
            return {
                "score": score,
                "overall_feedback": evaluation.get("overall_feedback"),
                "strengths": evaluation.get("strengths"),
                "improvements": evaluation.get("improvements"),
                "sample_answer_points": evaluation.get("sample_answer_points")
            }
        except Exception as e:
            logger.warning(f"JSON parsing failed inside _parse_evaluation_response: {e}")
            
    # 2. Try regex extraction if JSON parsing failed
    score_match = re.search(r'"score"\s*:\s*(\d+)', response_str)
    if not score_match:
        score_match = re.search(r'\bscore\b.*?(\d+)', response_str, re.IGNORECASE)
        
    score = int(score_match.group(1)) if score_match else None
    
    strengths = []
    improvements = []
    sample_points = []
    
    lines = response_str.split('\n')
    current_section = None
    for line in lines:
        line_lower = line.lower()
        if 'strength' in line_lower:
            current_section = 'strengths'
            continue
        elif 'improvement' in line_lower or 'improve' in line_lower:
            current_section = 'improvements'
            continue
        elif 'sample' in line_lower or 'point' in line_lower or 'strong answer' in line_lower:
            current_section = 'sample'
            continue
            
        clean_line = line.strip().lstrip('-*•123456789.0) ')
        if clean_line and len(clean_line) > 5:
            if current_section == 'strengths':
                strengths.append(clean_line)
            elif current_section == 'improvements':
                improvements.append(clean_line)
            elif current_section == 'sample':
                sample_points.append(clean_line)
                
    if score is not None or strengths or improvements:
        return {
            "score": score,
            "overall_feedback": "Evaluation extracted from unstructured response.",
            "strengths": strengths if strengths else ["Answer provided"],
            "improvements": improvements if improvements else ["Add more detail"],
            "sample_answer_points": sample_points
        }
        
    return None


def generate_session_report(evaluations: List[Dict]) -> Dict:
    """
    Generate a comprehensive session report from all question evaluations.
    
    Aggregates individual scores and feedback into an overall performance summary.
    
    Args:
        evaluations: List of evaluation dictionaries from each question.
        
    Returns:
        Session report dictionary with averages and summaries.
    """
    if not evaluations:
        return {
            "average_score": 0,
            "total_questions": 0,
            "questions_answered": 0,
            "score_distribution": {},
            "all_strengths": [],
            "all_improvements": [],
            "performance_level": "No data"
        }
    
    # Calculate aggregate metrics
    scores = [e.get("score", 0) for e in evaluations]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Score distribution
    distribution = {
        "Excellent (9-10)": sum(1 for s in scores if s >= 9),
        "Strong (7-8)": sum(1 for s in scores if 7 <= s < 9),
        "Adequate (5-6)": sum(1 for s in scores if 5 <= s < 7),
        "Needs Work (3-4)": sum(1 for s in scores if 3 <= s < 5),
        "Poor (1-2)": sum(1 for s in scores if s < 3),
    }
    
    # Collect all strengths and improvements (unique)
    all_strengths = list(set(
        s for e in evaluations
        for s in e.get("strengths", [])
    ))
    all_improvements = list(set(
        imp for e in evaluations
        for imp in e.get("improvements", [])
    ))
    
    # Determine overall performance level
    if avg_score >= 8:
        level = "🌟 Excellent - You're well prepared!"
    elif avg_score >= 6:
        level = "✅ Good - Solid performance with room to grow"
    elif avg_score >= 4:
        level = "⚡ Fair - Some areas need more preparation"
    else:
        level = "📚 Needs Practice - Focus on structured responses"
    
    return {
        "average_score": round(avg_score, 1),
        "total_questions": len(evaluations),
        "questions_answered": len([e for e in evaluations if e.get("score", 0) > 0]),
        "score_distribution": distribution,
        "all_strengths": all_strengths[:8],  # Limit to top 8
        "all_improvements": all_improvements[:8],
        "performance_level": level,
        "individual_scores": scores
    }
