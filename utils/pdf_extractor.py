"""
PDF Text Extraction Module
--------------------------
Handles extracting and chunking text from uploaded PDF resumes.
Uses PyPDF2 for extraction and LangChain's text splitter for chunking.
"""

import io
import logging
from typing import List, Optional

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def extract_text_from_pdf(uploaded_file) -> Optional[str]:
    """
    Extract all text content from an uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object (PDF)
        
    Returns:
        Extracted text as a single string, or None if extraction fails.
    """
    try:
        # Read the uploaded file into a BytesIO buffer
        pdf_bytes = io.BytesIO(uploaded_file.read())
        reader = PdfReader(pdf_bytes)
        
        # Check if PDF has any pages
        if len(reader.pages) == 0:
            logger.warning("PDF has no pages.")
            return None
        
        # Extract text from all pages
        all_text = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text.strip())
            else:
                logger.warning(f"No text extracted from page {page_num + 1}")
        
        # Combine all page texts
        full_text = "\n\n".join(all_text)
        
        if not full_text.strip():
            logger.warning("No text content found in the PDF.")
            return None
        
        logger.info(f"Successfully extracted {len(full_text)} characters from {len(reader.pages)} pages.")
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Uses LangChain's RecursiveCharacterTextSplitter which tries to split
    on natural boundaries (paragraphs, sentences, words) before falling
    back to character-level splitting.
    
    Args:
        text: The full text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        
    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []
    
    # Create the text splitter with sensible separators for resume content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]  # Priority order
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    
    logger.info(f"Split text into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def extract_resume_highlights(text: str) -> dict:
    """
    Extract key highlights from resume text for display.
    This is a simple heuristic-based extraction for the UI preview.
    
    Args:
        text: The full resume text.
        
    Returns:
        Dictionary with basic resume metrics.
    """
    lines = text.split("\n")
    non_empty_lines = [l.strip() for l in lines if l.strip()]
    
    return {
        "total_characters": len(text),
        "total_lines": len(non_empty_lines),
        "total_words": len(text.split()),
        "preview": text[:500] + ("..." if len(text) > 500 else "")
    }
