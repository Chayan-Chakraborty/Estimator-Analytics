"""
Speech-to-Text (STT) module using faster-whisper for audio transcription.
"""

import os
import tempfile
from fastapi import UploadFile, HTTPException
from faster_whisper import WhisperModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Whisper model (cached globally)
_model = None

def get_whisper_model():
    """Get or initialize the Whisper model."""
    global _model
    if _model is None:
        try:
            # Use base model for good balance of speed and accuracy
            _model = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("Whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise HTTPException(status_code=500, detail="STT model initialization failed")
    return _model

def transcribe_audio(file: UploadFile) -> str:
    """
    Transcribe audio file to text using faster-whisper.
    
    Args:
        file: UploadFile containing audio data
        
    Returns:
        str: Transcribed text
        
    Raises:
        HTTPException: If transcription fails
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size (limit to 25MB)
        file_content = file.file.read()
        if len(file_content) > 25 * 1024 * 1024:  # 25MB
            raise HTTPException(status_code=400, detail="File too large (max 25MB)")
        
        # Check file format
        allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Get Whisper model
            model = get_whisper_model()
            
            # Transcribe audio
            logger.info(f"Transcribing audio file: {file.filename}")
            segments, info = model.transcribe(
                temp_file_path,
                language=None,  # Auto-detect language
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False
            )
            
            # Combine all segments
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text
            
            # Clean up
            transcribed_text = transcribed_text.strip()
            
            if not transcribed_text:
                raise HTTPException(status_code=400, detail="No speech detected in audio")
            
            logger.info(f"Transcription successful: {len(transcribed_text)} characters")
            logger.info(f"Detected language: {info.language}")
            
            return transcribed_text
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
