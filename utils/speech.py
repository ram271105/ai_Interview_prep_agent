"""
Speech-to-Text Module
---------------------
Optional Whisper-based speech-to-text for voice input during interviews.
Gracefully degrades if Whisper or audio dependencies are not available.
"""

import io
import logging
import tempfile
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Check Whisper availability at import time
_whisper_available = False
_whisper_model = None

try:
    import whisper
    _whisper_available = True
    logger.info("Whisper is available for speech-to-text.")
except ImportError:
    logger.info("Whisper not installed. Speech-to-text will be disabled.")

# Check audio recording availability
_audio_available = False
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    _audio_available = True
    logger.info("Audio recording libraries available.")
except ImportError:
    logger.info("Audio recording libraries not installed. Recording will be disabled.")


def is_whisper_available() -> bool:
    """
    Check if Whisper speech-to-text is available.
    
    Returns:
        True if Whisper and required audio libraries are installed.
    """
    return _whisper_available


def is_recording_available() -> bool:
    """
    Check if audio recording capability is available.
    
    Returns:
        True if sounddevice and soundfile are installed.
    """
    return _audio_available


def load_whisper_model(model_name: str = "base") -> bool:
    """
    Load the Whisper model for transcription.
    Uses lazy loading to avoid long startup times.
    
    Args:
        model_name: Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        True if model loaded successfully, False otherwise.
    """
    global _whisper_model
    
    if not _whisper_available:
        return False
    
    if _whisper_model is not None:
        return True
    
    try:
        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model = whisper.load_model(model_name)
        logger.info("Whisper model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        return False


def record_audio(
    duration: int = 30,
    sample_rate: int = 16000
) -> Optional[bytes]:
    """
    Record audio from the microphone.
    
    Args:
        duration: Recording duration in seconds.
        sample_rate: Audio sample rate in Hz.
        
    Returns:
        Audio data as bytes (WAV format), or None if recording fails.
    """
    if not _audio_available:
        logger.warning("Audio recording not available.")
        return None
    
    try:
        logger.info(f"Recording audio for {duration} seconds...")
        
        # Record audio using sounddevice
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()  # Wait for recording to finish
        
        # Save to a temporary WAV file
        temp_buffer = io.BytesIO()
        sf.write(temp_buffer, audio_data, sample_rate, format="WAV")
        temp_buffer.seek(0)
        
        logger.info("Audio recording complete.")
        return temp_buffer.read()
        
    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        return None


def transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe audio bytes to text using Whisper.
    
    Args:
        audio_bytes: Audio data in WAV format.
        
    Returns:
        Transcribed text string, or None if transcription fails.
    """
    if not _whisper_available:
        logger.warning("Whisper not available for transcription.")
        return None
    
    # Ensure model is loaded
    if not load_whisper_model():
        return None
    
    try:
        # Write audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Transcribe using Whisper
        logger.info("Transcribing audio...")
        result = _whisper_model.transcribe(tmp_path, fp16=False)
        text = result.get("text", "").strip()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        logger.info(f"Transcription complete: '{text[:50]}...'")
        return text if text else None
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return None


def transcribe_uploaded_audio(uploaded_file) -> Optional[str]:
    """
    Transcribe an uploaded audio file using Whisper.
    Accepts WAV, MP3, M4A, and other audio formats.
    
    Args:
        uploaded_file: Streamlit UploadedFile object (audio)
        
    Returns:
        Transcribed text string, or None if transcription fails.
    """
    if not _whisper_available:
        return None
    
    if not load_whisper_model():
        return None
    
    try:
        # Save uploaded file to temp location
        suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Transcribe
        result = _whisper_model.transcribe(tmp_path, fp16=False)
        text = result.get("text", "").strip()
        
        # Cleanup
        os.unlink(tmp_path)
        
        return text if text else None
        
    except Exception as e:
        logger.error(f"Error transcribing uploaded audio: {str(e)}")
        return None
