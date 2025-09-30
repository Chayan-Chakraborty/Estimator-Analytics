"""
Translation module for converting text to English using multiple translation services.
"""

import logging
from typing import Optional
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Global translation pipeline (cached)
_translation_pipeline = None

def get_translation_pipeline():
    """Get or initialize the translation pipeline."""
    global _translation_pipeline
    if _translation_pipeline is None:
        try:
            # Use a lightweight model for translation
            _translation_pipeline = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-mul-en",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Translation pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize translation pipeline: {e}")
            _translation_pipeline = None
    return _translation_pipeline

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text: Input text to detect language for
        
    Returns:
        str: Detected language code (e.g., 'en', 'hi', 'bn')
    """
    try:
        if not text or not text.strip():
            return 'en'  # Default to English for empty text
        
        # Clean text for better detection
        clean_text = text.strip()
        if len(clean_text) < 3:
            return 'en'
        
        detected_lang = detect(clean_text)
        logger.info(f"Detected language: {detected_lang}")
        return detected_lang
        
    except Exception as e:
        logger.warning(f"Language detection failed: {e}, defaulting to English")
        return 'en'

def translate_to_english(text: str, source_language: Optional[str] = None) -> str:
    """
    Translate text to English using the best available translation method.
    
    Args:
        text: Input text to translate
        source_language: Optional source language code (if None, will auto-detect)
        
    Returns:
        str: Translated English text
    """
    try:
        if not text or not text.strip():
            return text
        
        # Detect language if not provided
        if source_language is None:
            source_language = detect_language(text)
        
        # If already English, return as is
        if source_language == 'en':
            logger.info("Text is already in English")
            return text.strip()
        
        # Try Google Translator first (most reliable)
        try:
            logger.info(f"Translating from {source_language} to English using Google Translator")
            translator = GoogleTranslator(source=source_language, target='en')
            translated = translator.translate(text)
            
            if translated and translated.strip():
                logger.info("Translation successful with Google Translator")
                return translated.strip()
                
        except Exception as e:
            logger.warning(f"Google Translator failed: {e}")
        
        # Fallback to HuggingFace model
        try:
            pipeline = get_translation_pipeline()
            if pipeline:
                logger.info(f"Translating from {source_language} to English using HuggingFace model")
                result = pipeline(text, src_lang=source_language, tgt_lang='en')
                translated = result[0]['translation_text']
                
                if translated and translated.strip():
                    logger.info("Translation successful with HuggingFace model")
                    return translated.strip()
                    
        except Exception as e:
            logger.warning(f"HuggingFace translation failed: {e}")
        
        # If all translation methods fail, return original text
        logger.warning("All translation methods failed, returning original text")
        return text.strip()
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        # Return original text if translation fails
        return text.strip()

def translate_with_confidence(text: str, source_language: Optional[str] = None) -> tuple[str, str, float]:
    """
    Translate text to English with confidence score.
    
    Args:
        text: Input text to translate
        source_language: Optional source language code
        
    Returns:
        tuple: (translated_text, detected_language, confidence_score)
    """
    try:
        if not text or not text.strip():
            return text, 'en', 1.0
        
        # Detect language
        detected_lang = detect_language(text) if source_language is None else source_language
        
        # If already English, return with high confidence
        if detected_lang == 'en':
            return text.strip(), 'en', 1.0
        
        # Translate
        translated = translate_to_english(text, detected_lang)
        
        # Simple confidence estimation based on translation success
        confidence = 0.9 if translated != text else 0.5
        
        return translated, detected_lang, confidence
        
    except Exception as e:
        logger.error(f"Translation with confidence failed: {e}")
        return text.strip(), 'en', 0.3
