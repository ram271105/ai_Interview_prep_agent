"""
LLM Chain Module
----------------
Orchestrates the LangChain retrieval pipeline with Ollama (llama3).
Creates specialized chains for question generation and answer evaluation.
Implements RAG (Retrieval-Augmented Generation) architecture.
"""

import json
import logging
from typing import List, Dict, Optional

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_CTX,
    MODE_HR,
    MODE_TECHNICAL,
    MODE_MIXED,
)

logger = logging.getLogger(__name__)

# Cache the LLM instance
_llm_instance = None


def get_ollama_llm() -> Ollama:
    """
    Initialize and return the Ollama LLM instance with llama3 model.
    Uses singleton pattern for efficiency.
    
    Returns:
        Configured Ollama LLM instance.
    """
    global _llm_instance
    
    if _llm_instance is None:
        logger.info(f"Initializing Ollama LLM with model: {OLLAMA_MODEL}")
        _llm_instance = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=OLLAMA_TEMPERATURE,
            num_ctx=OLLAMA_NUM_CTX,
        )
        logger.info("Ollama LLM initialized successfully.")
    
    return _llm_instance


def check_ollama_connection() -> bool:
    """
    Check if Ollama is running and the model is available.
    
    Returns:
        True if Ollama is accessible, False otherwise.
    """
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            if OLLAMA_MODEL in model_names:
                return True
            logger.warning(f"Model '{OLLAMA_MODEL}' not found. Available: {model_names}")
            return False
        return False
    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {str(e)}")
        return False


def _get_question_prompt(mode: str) -> PromptTemplate:
    """
    Get the appropriate prompt template based on interview mode.
    
    Args:
        mode: Interview mode (HR, Technical, or Mixed)
    
    Returns:
        PromptTemplate configured for the specified mode.
    """
    # Base context instruction
    base_context = """You are an expert interview coach. Based on the candidate's resume information provided below, generate personalized interview questions.

RESUME CONTEXT:
{context}

"""
    
    if mode == MODE_HR:
        mode_instruction = """Generate {num_questions} HR and behavioral interview questions. Focus on:
- Behavioral questions using the STAR method (Situation, Task, Action, Result)
- Cultural fit and teamwork scenarios
- Leadership and conflict resolution
- Career goals and motivation
- Communication skills and adaptability

Base each question on specific experiences, skills, or roles mentioned in the resume."""
    
    elif mode == MODE_TECHNICAL:
        mode_instruction = """Generate {num_questions} technical interview questions. Focus on:
- Technical skills and technologies mentioned in the resume
- Problem-solving and system design
- Coding concepts and best practices
- Architecture and design patterns
- Domain-specific technical knowledge

Tailor questions to the candidate's specific tech stack and experience level."""
    
    else:  # Mixed mode
        mode_instruction = """Generate {num_questions} interview questions with a mix of HR/behavioral and technical questions. Include:
- 40% behavioral/HR questions (teamwork, leadership, communication)
- 60% technical questions (skills, problem-solving, domain knowledge)

Base all questions on the resume content."""
    
    format_instruction = """

IMPORTANT: Return ONLY a valid JSON array of objects. Each object must have:
- "id": question number (integer)
- "question": the interview question (string)
- "type": either "HR" or "Technical" (string)
- "difficulty": either "Easy", "Medium", or "Hard" (string)
- "focus_area": what skill/experience this targets (string)

Example format:
[{{"id": 1, "question": "Tell me about...", "type": "HR", "difficulty": "Medium", "focus_area": "Leadership"}}]

Return ONLY the JSON array, no other text."""
    
    full_template = base_context + mode_instruction + format_instruction
    
    return PromptTemplate(
        template=full_template,
        input_variables=["context", "num_questions"]
    )


def generate_questions(
    vector_store,
    mode: str,
    num_questions: int = 5
) -> List[Dict]:
    """
    Generate personalized interview questions using RAG pipeline.
    
    Flow:
    1. Retrieve relevant resume chunks from ChromaDB (or fall back to raw session text)
    2. Build prompt with resume context and interview mode
    3. Send to Ollama llama3 for question generation
    4. Parse JSON response into structured questions
    
    Args:
        vector_store: ChromaDB vector store with resume embeddings.
        mode: Interview mode (HR, Technical, or Mixed).
        num_questions: Number of questions to generate.
        
    Returns:
        List of question dictionaries.
    """
    context = ""
    try:
        llm = get_ollama_llm()
        
        # If vector_store is None, try loading it from disk
        if vector_store is None:
            logger.info("vector_store is None in generate_questions. Attempting to load from disk...")
            try:
                from utils.vector_store import load_vector_store
                vector_store = load_vector_store()
            except Exception as load_err:
                logger.warning(f"Could not load vector store from disk: {load_err}")
        
        # 1. Try to retrieve relevant resume chunks from vector store
        if vector_store is not None:
            try:
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6}  # Get more chunks for question generation
                )
                docs = retriever.invoke("skills experience education projects achievements")
                context = "\n\n".join([doc.page_content for doc in docs])
                logger.info(f"Successfully retrieved {len(docs)} chunks from vector store.")
            except Exception as retrieval_err:
                logger.warning(f"Failed to retrieve context from vector store: {retrieval_err}. Trying fallback...")
        
        # 2. Fallback: Use raw resume text from session state if vector store failed or has no data
        if not context.strip():
            import streamlit as st
            if "resume_text" in st.session_state and st.session_state.resume_text:
                logger.info("Using session_state.resume_text for context fallback.")
                context = st.session_state.resume_text[:3000]  # Limit to 3000 chars for prompt safety
            else:
                logger.warning("No resume context available. Proceeding with empty context.")
                context = "Candidate resume details not available. Generate general professional questions."
        
        # Get the appropriate prompt
        prompt = _get_question_prompt(mode)
        formatted_prompt = prompt.format(context=context, num_questions=num_questions)
        
        # Generate questions using the LLM
        logger.info(f"Generating {num_questions} {mode} questions...")
        response = llm.invoke(formatted_prompt)
        
        # Parse the JSON response
        questions = _parse_questions_response(response, num_questions)
        
        logger.info(f"Successfully generated {len(questions)} questions.")
        return questions
        
    except Exception as e:
        import traceback
        logger.error(f"Error generating questions: {str(e)}\n{traceback.format_exc()}")
        return []


def _parse_questions_response(response: str, expected_count: int) -> List[Dict]:
    """
    Parse the LLM response into a list of question dictionaries.
    Handles various response formats and edge cases.
    
    Args:
        response: Raw LLM response string.
        expected_count: Expected number of questions.
        
    Returns:
        List of parsed question dictionaries.
    """
    try:
        # Try to find JSON array in the response
        response = response.strip()
        
        # Find the JSON array boundaries
        start_idx = response.find("[")
        end_idx = response.rfind("]") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            questions = json.loads(json_str)
            
            # Validate and normalize each question
            validated = []
            for i, q in enumerate(questions):
                validated.append({
                    "id": q.get("id", i + 1),
                    "question": q.get("question", "Question not available"),
                    "type": q.get("type", "General"),
                    "difficulty": q.get("difficulty", "Medium"),
                    "focus_area": q.get("focus_area", "General")
                })
            return validated
        
        # Fallback: if no JSON found, create simple questions from text
        logger.warning("Could not parse JSON from LLM response, using fallback.")
        lines = [l.strip() for l in response.split("\n") if l.strip() and "?" in l]
        return [
            {
                "id": i + 1,
                "question": line.lstrip("0123456789.-) "),
                "type": "General",
                "difficulty": "Medium",
                "focus_area": "General"
            }
            for i, line in enumerate(lines[:expected_count])
        ]
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}")
        return []


def get_session_memory() -> ConversationBufferMemory:
    """
    Create a conversation memory buffer for maintaining session context.
    
    Returns:
        ConversationBufferMemory instance.
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="result"
    )
