import re
from typing import Tuple, Optional
from functools import lru_cache

from deep_translator import GoogleTranslator
from langdetect import detect
from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI, BENGALI, GUJARATI, GURMUKHI, KANNADA, MALAYALAM, ORIYA, TAMIL, TELUGU
from wordfreq import zipf_frequency
# Removed heavy NLLB model import for better performance


def _score_englishness(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r"[A-Za-z]+", text)
    # Use wordfreq to reward common English words
    score = 0.0
    for w in words:
        score += max(0.0, zipf_frequency(w, 'en'))
    # Small bonus for length/coverage
    score += 0.05 * len(words)
    return score


@lru_cache(maxsize=1000)
def _translate_auto(text: str, source_lang: Optional[str] = None) -> Tuple[str, str]:
    tr = GoogleTranslator(source=source_lang or "auto", target="en")
    translated = tr.translate(text)
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    return translated, lang


def intelligent_translate(text: str, source_language: Optional[str] = None) -> Tuple[str, str]:
    """
    Fast, efficient translation layer.
    - Use GoogleTranslator for primary translation (fast)
    - Only try transliteration for common romanized patterns
    - Skip heavy NLLB model for better performance
    """
    print(f"Intelligent translate: Text: {text}, Source language: {source_language}")
    if not text:
        return "", "unknown"

    base = text.strip()
    
    # Fast path: try direct Google translation first
    try:
        primary, detected_src = _translate_auto(base, source_language)
        best_text = (primary or "").strip()
        best_score = _score_englishness(best_text)
    except Exception:
        # Fallback: if translation fails, return original text
        return base, "unknown"

    # Quick check: if already good English, return early
    if best_score > 5.0:  # High confidence threshold
        return best_text, detected_src

    # Only do complex processing for ASCII/romanized text
    try:
        base.encode("ascii")
        is_ascii = True
    except Exception:
        is_ascii = False

    if is_ascii and len(base.split()) <= 10:  # Limit to reasonable length
        # Try only the most common Indic languages (reduce API calls)
        common_candidates = [
            ("hi", DEVANAGARI),  # Hindi
            ("bn", BENGALI),     # Bengali  
            ("ta", TAMIL),       # Tamil
            ("te", TELUGU),      # Telugu
        ]
        
        for lang, script in common_candidates:
            try:
                native = transliterate(base, ITRANS, script)
                tr = GoogleTranslator(source=lang, target="en")
                translated = (tr.translate(native) or "").strip()
                s = _score_englishness(translated)
                if s > best_score:
                    best_score = s
                    best_text = translated
            except Exception:
                continue
    print(f"Best text: {best_text}, Detected src: {detected_src} Best score: {best_score} Source language: {source_language} Base: {base} Primary: {primary} Best text: {best_text} Is ascii: {is_ascii} Common candidates: {common_candidates}")
    return translated, detected_src


