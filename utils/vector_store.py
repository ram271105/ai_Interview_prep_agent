"""
Vector Store Module
-------------------
Manages ChromaDB vector storage for resume embeddings.
Uses sentence-transformers for embedding generation and ChromaDB for
persistent vector storage and similarity search.
"""

import logging
import shutil
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)

# Cache the embedding function to avoid reloading the model
_embedding_function = None


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Get or create the HuggingFace embedding function.
    Uses singleton pattern to avoid reloading the model on every call.
    
    Returns:
        HuggingFaceEmbeddings instance using the configured model.
    """
    global _embedding_function
    
    if _embedding_function is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}  # For cosine similarity
        )
        logger.info("Embedding model loaded successfully.")
    
    return _embedding_function


def create_vector_store(text_chunks: List[str]) -> Optional[Chroma]:
    """
    Create a ChromaDB vector store from text chunks.
    
    Each chunk is embedded using the sentence-transformer model and stored
    in a persistent ChromaDB collection for retrieval.
    
    Args:
        text_chunks: List of text chunks to embed and store.
        
    Returns:
        Chroma vector store instance, or None if creation fails.
    """
    if not text_chunks:
        logger.warning("No text chunks provided for vector store creation.")
        return None
    
    try:
        # Get the embedding function
        embeddings = get_embedding_function()
        
        # Clear any existing collection first
        clear_vector_store()
        
        # Create Document objects from chunks
        documents = [
            Document(
                page_content=chunk,
                metadata={"chunk_index": i, "source": "resume"}
            )
            for i, chunk in enumerate(text_chunks)
        ]
        
        # Create the Chroma vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        
        logger.info(f"Created vector store with {len(text_chunks)} chunks.")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None


def load_vector_store() -> Optional[Chroma]:
    """
    Load an existing ChromaDB vector store from the persist directory.
    
    Returns:
        Chroma vector store instance, or None if not found.
    """
    try:
        embeddings = get_embedding_function()
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        return None


def query_vector_store(
    vector_store: Chroma,
    query: str,
    top_k: int = RETRIEVAL_TOP_K
) -> List[Document]:
    """
    Query the vector store for the most relevant document chunks.
    
    Uses similarity search to find the top-k most relevant chunks
    based on the query embedding.
    
    Args:
        vector_store: The Chroma vector store to query.
        query: The search query string.
        top_k: Number of top results to return.
        
    Returns:
        List of the most relevant Document objects.
    """
    try:
        results = vector_store.similarity_search(query, k=top_k)
        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        return results
    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}")
        return []


def clear_vector_store():
    """
    Clear the existing ChromaDB collection and persist directory.
    Used when uploading a new resume to start fresh.
    """
    try:
        # Remove the persist directory if it exists
        import os
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
            logger.info("Cleared existing vector store.")
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
