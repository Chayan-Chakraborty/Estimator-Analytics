import io
import json
import logging
import os
from typing import Optional, Tuple
import re
from app.mappings import ROMAN_TO_EN

from langdetect import detect
from deep_translator import GoogleTranslator
from .translate_to_english import detect_and_translate
from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI, BENGALI, GUJARATI, GURMUKHI, KANNADA, MALAYALAM, ORIYA, TAMIL, TELUGU

# Whisper STT (preferred) and Vosk fallback
try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except Exception:
    _WHISPER_AVAILABLE = False

try:
    import whisper as openai_whisper
    _OPENAI_WHISPER_AVAILABLE = True
except Exception:
    _OPENAI_WHISPER_AVAILABLE = False

try:
    from vosk import Model, KaldiRecognizer
    import soundfile as sf
    _VOSK_AVAILABLE = True
except Exception:  # pragma: no cover
    _VOSK_AVAILABLE = False


logger = logging.getLogger(__name__)
def _stt_with_whisper(audio_bytes: bytes) -> str:
    """
    STT using faster-whisper. Works with many audio formats; requires ffmpeg.
    Model name is configured via WHISPER_MODEL (e.g., 'small', 'medium', 'large-v3').
    """
    if not _WHISPER_AVAILABLE:
        raise RuntimeError("Whisper not available. Please install 'faster-whisper'.")

    model_name = os.getenv("WHISPER_MODEL", "small")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float32")  # or int8 for CPU optimizations
    model = WhisperModel(model_name, compute_type=compute_type)

    # Convert bytes to a buffer readable by faster-whisper (it can read file-like objects)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments, _info = model.transcribe(tmp.name, beam_size=1)
        texts = [seg.text.strip() for seg in segments if seg.text]
        return " ".join(texts).strip()


def _stt_with_openai_whisper(audio_bytes: bytes) -> str:
    """
    STT using OpenAI's reference Whisper implementation (as in the screenshot).
    Slower on CPU but very accurate.
    """
    if not _OPENAI_WHISPER_AVAILABLE:
        raise RuntimeError("openai-whisper not available. Please install 'openai-whisper'.")
    model_name = os.getenv("OPENAI_WHISPER_MODEL", "base")
    model = openai_whisper.load_model(model_name)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        # OpenAI whisper API can transcribe directly from file path
        res = model.transcribe(tmp.name)
        return (res.get("text") or "").strip()



def _stt_with_vosk(wav_bytes: bytes) -> str:
    """
    Offline STT using Vosk. Expects PCM WAV bytes (mono, 16kHz ideal).
    Returns recognized text (space separated tokens).
    """
    if not _VOSK_AVAILABLE:
        raise RuntimeError("Vosk not available. Please install 'vosk' and 'soundfile' and provide a model.")

    # Load audio
    data, samplerate = sf.read(io.BytesIO(wav_bytes))
    if len(data.shape) == 2:
        # convert to mono
        data = data.mean(axis=1)

    # Vosk expects 16k ideally
    if samplerate != 16000:
        # simple resample using soundfile's write-read roundtrip to 16k in-memory
        import soundfile as sf2
        buf = io.BytesIO()
        sf2.write(buf, data, 16000, format='WAV')
        buf.seek(0)
        data, samplerate = sf2.read(buf)

    # Initialize model (assume English small model is mounted in /models/vosk)
    # You can swap to multilingual/Indian language acoustic models as needed.
    model = Model(model_path="/models/vosk")
    rec = KaldiRecognizer(model, samplerate)

    # Feed in small chunks
    import numpy as np
    chunk_size = 4000
    text_parts = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        pcm16 = (chunk * 32767).astype(np.int16).tobytes()
        if rec.AcceptWaveform(pcm16):
            j = json.loads(rec.Result())
            if j.get("text"):
                text_parts.append(j["text"])
    final = json.loads(rec.FinalResult())
    if final.get("text"):
        text_parts.append(final["text"])
    return " ".join([p for p in text_parts if p]).strip()


def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


_DEVANAGARI_LANGS = {"hi", "mr", "ne"}

_SCRIPT_MAP = {
    "hi": DEVANAGARI,
    "mr": DEVANAGARI,
    "bn": BENGALI,
    "gu": GUJARATI,
    "pa": GURMUKHI,
    "kn": KANNADA,
    "ml": MALAYALAM,
    "or": ORIYA,
    "ta": TAMIL,
    "te": TELUGU,
}


def _maybe_transliterate_roman(text: str, lang: str) -> str:
    """
    Attempt to transliterate romanized Indian language text to native script
    before translation, using ITRANS schema as a heuristic.
    """
    script = _SCRIPT_MAP.get(lang)
    if not script:
        return text
    try:
        return transliterate(text, ITRANS, script)
    except Exception:
        return text


def _is_ascii_like(text: str) -> bool:
    try:
        text.encode('ascii')
        return True
    except Exception:
        return False


def _try_multi_transliterate_and_translate(text: str) -> str:
    """
    Heuristic for romanized inputs: try multiple likely Indian languages' scripts,
    translate each to English, and pick the first translation that differs from the input.
    Order is a rough priority based on prevalence.
    """
    candidates = [
        ("hi", DEVANAGARI),
        ("bn", BENGALI),
        ("mr", DEVANAGARI),
        ("gu", GUJARATI),
        ("pa", GURMUKHI),
        ("ta", TAMIL),
        ("te", TELUGU),
    ]
    base = text.strip()
    for lang, script in candidates:
        try:
            native = transliterate(base, ITRANS, script)
        except Exception:
            continue
        translated, _ = _translate_to_english(native, source_lang=lang)
        if translated and translated.strip() and translated.strip().lower() != base.lower():
            return translated.strip()
    return text


# Common romanized Indic terms → English direct mapping (heuristic)


def _roman_direct_english(text: str) -> Optional[str]:
    base = (text or "").strip().lower()
    if not base:
        return None
    # exact match
    if base in ROMAN_TO_EN:
        return ROMAN_TO_EN[base]
    # prefix/word boundary heuristic for partial recognitions like "bichan"
    for k, v in ROMAN_TO_EN.items():
        if base == k or base.startswith(k[:-1]):
            return v
    return None


def _roman_tokenwise_map(text: str) -> Optional[str]:
    """
    Token-wise romanized mapping: replace known roman words (e.g., 'bichana'→'bed', 'ar'→'and')
    while preserving unknown words (e.g., 'console', 'table'). Returns None if no change.
    """
    if not text:
        return None
    tokens = re.findall(r"[A-Za-z]+|\d+|\S", text)
    changed = False
    out_tokens = []
    for tok in tokens:
        low = tok.lower()
        repl = ROMAN_TO_EN.get(low)
        if repl:
            out_tokens.append(repl)
            changed = True
        else:
            # Try partial like 'bichan' => 'bed'
            partial = None
            for k, v in ROMAN_TO_EN.items():
                if len(k) > 3 and low.startswith(k[:-1]):
                    partial = v
                    break
            if partial:
                out_tokens.append(partial)
                changed = True
            else:
                out_tokens.append(tok)
    if not changed:
        return None
    # Simple spacing: join with spaces and then fix space before punctuation
    joined = " ".join(out_tokens)
    joined = re.sub(r"\s+([,.!?;:])", r"\\1", joined)
    return joined.strip()


def _translate_to_english(text: str, source_lang: Optional[str]) -> Tuple[str, float]:
    # Prefer transformers-based NLLB model; fallback to GoogleTranslator
    try:
        translated = detect_and_translate(text)
        return translated, 0.0
    except Exception:
        try:
            tr = GoogleTranslator(source=source_lang or "auto", target="en")
            translated = tr.translate(text)
            return translated, 0.0
        except Exception as e:  # pragma: no cover
            logger.warning("Translation failed: %s", e)
            return text, 0.0


def process_speech_to_english_query(audio_bytes: bytes) -> str:
    """
    Captures microphone speech, converts it to English text,
    and returns the text ready for the API query.

    Setup notes:
    - Install dependencies: pip install vosk soundfile langdetect deep-translator indic-transliteration
    - Download a Vosk model and mount it at /models/vosk inside the backend container.
      Example (English small): https://alphacephei.com/vosk/models

    Input:
    - audio_bytes: PCM WAV bytes (mono preferred). If not 16kHz, it's resampled on-the-fly.

    Steps:
    1) Speech-to-text via Vosk
    2) Language detection
    3) If non-English: transliterate romanized Indian text → native script, then translate to English
    4) Return final English string
    """
    # 1) STT (Whisper preferred, fallback to Vosk)
    recognized_text = ""
    prefer_whisper = os.getenv("PREFER_WHISPER", "1") not in {"0", "false", "False"}
    prefer_openai = os.getenv("PREFER_OPENAI_WHISPER", "0") in {"1", "true", "True"}
    stt_errors = []
    if prefer_whisper and prefer_openai:
        try:
            recognized_text = _stt_with_openai_whisper(audio_bytes)
        except Exception as e:
            stt_errors.append(f"openai-whisper: {e}")
            logger.warning("OpenAI Whisper STT failed: %s", e)
    if prefer_whisper and not recognized_text:
        try:
            recognized_text = _stt_with_whisper(audio_bytes)
        except Exception as e:
            stt_errors.append(f"whisper: {e}")
            logger.warning("Whisper STT failed: %s", e)
    if not recognized_text:
        try:
            recognized_text = _stt_with_vosk(audio_bytes)
        except Exception as e:
            stt_errors.append(f"vosk: {e}")
            logger.error("Vosk STT failed: %s", e)
            raise RuntimeError("All STT backends failed: " + "; ".join(stt_errors))

    if not recognized_text:
        return ""

    # 2) Detect language
    lang = _detect_language(recognized_text)
    if lang.startswith("en"):
        return recognized_text

    # 3) If ascii-like, attempt multi-script transliteration fallback
    if _is_ascii_like(recognized_text):
        # token-wise mapping first
        token_mapped = _roman_tokenwise_map(recognized_text)
        if token_mapped:
            return token_mapped
        direct = _roman_direct_english(recognized_text)
        if direct:
            return direct
        return _try_multi_transliterate_and_translate(recognized_text)

    # Otherwise transliterate per detected language and translate
    roman_to_native = _maybe_transliterate_roman(recognized_text, lang)
    english_text, _ = _translate_to_english(roman_to_native, source_lang=lang)
    return english_text


def process_text_to_english_query(text: str) -> str:
    """Utility to process already-recognized text through detection/transliteration/translation."""
    if not text:
        return ""
    lang = _detect_language(text)
    if lang.startswith("en"):
        return text
    if _is_ascii_like(text):
        token_mapped = _roman_tokenwise_map(text)
        if token_mapped:
            return token_mapped
        direct = _roman_direct_english(text)
        if direct:
            return direct
        return _try_multi_transliterate_and_translate(text)
    roman_to_native = _maybe_transliterate_roman(text, lang)
    english_text, _ = _translate_to_english(roman_to_native, source_lang=lang)
    return english_text


