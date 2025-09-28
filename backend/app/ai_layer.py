import re
from typing import Tuple, Optional

from deep_translator import GoogleTranslator
from langdetect import detect
from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI, BENGALI, GUJARATI, GURMUKHI, KANNADA, MALAYALAM, ORIYA, TAMIL, TELUGU
from wordfreq import zipf_frequency
from .translate_to_english import detect_and_translate


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
    AI-first translation layer.
    - Auto-detect language and translate with GoogleTranslator
    - If input is ASCII/romanized and translation seems weak, try
      multiple Indic scripts via transliteration and pick the best
      English output by a simple fluency/Englishness score.
    Returns only the clean, natural English translation.
    """
    if not text:
        return "", "unknown"

    base = text.strip()
    # First try robust model-based translation (NLLB via transformers)
    try:
        primary = detect_and_translate(base)
        detected_src = detect(base)
    except Exception:
        primary, detected_src = _translate_auto(base, source_language)
    best_text = (primary or "").strip()
    best_score = _score_englishness(best_text)

    # Identify English tokens we should preserve (e.g., console, table)
    tokens = re.findall(r"[A-Za-z]+|\d+|\S", base)
    english_tokens = [t for t in tokens if zipf_frequency(t.lower(), 'en') >= 3.5]

    # Heuristic: if ascii-like, try Indic scripts as candidates
    try:
        base.encode("ascii")
        is_ascii = True
    except Exception:
        is_ascii = False

    if is_ascii:
        candidates = [
            ("hi", DEVANAGARI),
            ("bn", BENGALI),
            ("mr", DEVANAGARI),
            ("gu", GUJARATI),
            ("pa", GURMUKHI),
            ("ta", TAMIL),
            ("te", TELUGU),
            ("kn", KANNADA),
            ("ml", MALAYALAM),
            ("or", ORIYA),
        ]
        for lang, script in candidates:
            try:
                native = transliterate(base, ITRANS, script)
            except Exception:
                continue
            tr = GoogleTranslator(source=lang, target="en")
            translated = (tr.translate(native) or "").strip()
            s = _score_englishness(translated)
            if s > best_score:
                best_score = s
                best_text = translated

        # Token-wise: translate only the low-frequency (non-English) tokens, keep English tokens
        rebuilt = []
        for tok in tokens:
            if zipf_frequency(tok.lower(), 'en') >= 3.5:
                rebuilt.append(tok)
                continue
            # translate token via multiple scripts and take best
            best_tok = tok
            best_tok_score = _score_englishness(tok)
            for lang, script in candidates:
                try:
                    native_tok = transliterate(tok, ITRANS, script)
                except Exception:
                    continue
                tr = GoogleTranslator(source=lang, target="en")
                tr_tok = (tr.translate(native_tok) or "").strip()
                sc = _score_englishness(tr_tok)
                if sc > best_tok_score:
                    best_tok_score = sc
                    best_tok = tr_tok
            rebuilt.append(best_tok)
        token_wise = " ".join(rebuilt).strip()
        # Prefer token-wise if it preserves English tokens and is reasonably fluent
        if english_tokens:
            preserved = all(et in token_wise for et in english_tokens)
        else:
            preserved = True
        if preserved:
            s2 = _score_englishness(token_wise)
            if s2 > best_score:
                best_score = s2
                best_text = token_wise

    return best_text, detected_src


