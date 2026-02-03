import unicodedata
from typing import Dict, Any, List, Optional
from keybert import KeyBERT
from .amount_calculator import AmountCalculatorUtils
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import sys
import os
import re
import difflib
import numpy as np
from difflib import SequenceMatcher
from app.mappings import MATERIAL_ALIASES
from collections import defaultdict
from deep_translator import GoogleTranslator
from langdetect import detect
from indic_transliteration.sanscript import transliterate

from app.qdrant_client_helper import client, COLLECTION_NAME

from fastapi.middleware.cors import CORSMiddleware
from app.ingest import ingest_to_qdrant, fetch_data, parse_item_attributes, measurement_to_sqft
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchValue, Range
from app.ai_layer import intelligent_translate
from transformers import pipeline

from app.workers.stt import transcribe_audio
from app.workers.translate import translate_to_english, translate_with_confidence


zero_shot = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add workspace root so we can import multilingual_item_extractor
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_APP_DIR)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

# Multilingual item extractor: synonym index for query -> item_name (lazy-loaded)
_SYNONYM_INDEX = None
# Follow multilingual_item_extractor/demo.py:
# - load item synonyms JSON (items.json)
# - build synonym index
# - extract canonical item_name
#
# In Docker, only `backend/app/` is copied, so we default to `app/items.json`.
# In the dev workspace, we also fall back to `multilingual_item_extractor/item1.json`.
_ITEM_JSON_PATH = os.environ.get("ITEM_SYNONYM_JSON_PATH")
if not _ITEM_JSON_PATH:
    _candidates = [
        os.path.join(_APP_DIR, "items.json"),
        os.path.join(_WORKSPACE_ROOT, "multilingual_item_extractor", "item1.json"),
    ]
    _ITEM_JSON_PATH = next((p for p in _candidates if os.path.isfile(p)), _candidates[0])


def _get_synonym_index():
    """Load item synonym JSON and build index once (demo.py style)."""
    global _SYNONYM_INDEX
    if _SYNONYM_INDEX is not None:
        return _SYNONYM_INDEX
    try:
        import json
        from app.extractor import build_synonym_index

        if not os.path.isfile(_ITEM_JSON_PATH):
            print(f"⚠️ items.json not found at: {_ITEM_JSON_PATH}")
            _SYNONYM_INDEX = {}
            return _SYNONYM_INDEX

        with open(_ITEM_JSON_PATH, "r", encoding="utf-8") as f:
            item_json = json.load(f)

        _SYNONYM_INDEX = build_synonym_index(item_json)
    except Exception as e:
        print("⚠️ Could not load multilingual item extractor synonym index:", e)
        _SYNONYM_INDEX = {}
    return _SYNONYM_INDEX


def extract_item_name_from_query(query: str):
    """Demo.py flow: extract_item(query, synonym_index) -> canonical item_name.

    If the direct extractor result clearly doesn't correspond to the original
    query (e.g., token collisions), we fall back to scanning items.json for
    a substring match in any language's native/roman forms.
    """
    index = _get_synonym_index()
    if not index:
        return None
    original_query = (query or "").strip()
    if not original_query:
        return None

    # 1) Primary: use extractor over the synonym index
    try:
        from app.extractor import extract_item
        candidate = extract_item(original_query, index)
    except Exception:
        candidate = None

    # Helper to check whether a given item name actually has the original
    # query as a substring in any of its native/roman forms.
    def _item_matches_query(item_name: str) -> bool:
        try:
            import json
            if not os.path.isfile(_ITEM_JSON_PATH):
                return False
            with open(_ITEM_JSON_PATH, "r", encoding="utf-8") as f:
                item_json = json.load(f)
            entry = item_json.get(item_name, {})
            for forms in entry.values():
                native = str(forms.get("native", "") or "")
                roman = str(forms.get("roman", "") or "")
                if original_query in native or original_query in roman:
                    return True
        except Exception:
            return False
        return False

    # 2) If extractor gave us a candidate that clearly corresponds to the
    # original query (substring in any synonym), trust it.
    if candidate and _item_matches_query(candidate):
        return candidate

    # 3) Fallback: scan items.json directly for substring matches and pick
    # the first canonical item whose synonyms contain the query.
    try:
        import json
        if os.path.isfile(_ITEM_JSON_PATH):
            with open(_ITEM_JSON_PATH, "r", encoding="utf-8") as f:
                item_json = json.load(f)
            for item_name, langs in item_json.items():
                for forms in langs.values():
                    native = str(forms.get("native", "") or "")
                    roman = str(forms.get("roman", "") or "")
                    if original_query in native or original_query in roman:
                        return item_name
    except Exception:
        pass

    # 4) As a last resort, return whatever the extractor produced (may be None)
    return candidate


app = FastAPI()

# ---------------- NLP-based filtering helpers ----------------


def normalize_text(text):
    if not text:
        return ""
    return re.sub(r'[^\w\s]', '', str(text).lower().strip())


def calculate_area_wise_averages(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate average prices grouped by (area, item_name) combination.
    Returns a dict with key: f"{area}|{item_name}" and value: average price.
    Only includes groups with at least 2 items and valid amounts.
    """
    if not items:
        return {}

    # Group items by (area, item_name) - case-insensitive
    groups = defaultdict(list)
    for item in items:
        area = (item.get("area") or "").strip().lower()
        item_name = (item.get("item_name") or "").strip().lower()
        if area and item_name:
            key = f"{area}|{item_name}"
            groups[key].append(item)

    # Calculate averages for groups with at least 2 items
    averages = {}
    for key, group_items in groups.items():
        if len(group_items) < 2:
            continue

        # Extract valid amounts
        amounts = []
        for item in group_items:
            amount = item.get("amount")
            if amount is None:
                continue
            # Handle string amounts
            if isinstance(amount, str):
                try:
                    amount_val = float(amount.strip())
                except (ValueError, AttributeError):
                    continue
            elif isinstance(amount, (int, float)):
                amount_val = float(amount)
            else:
                continue

            if amount_val > 0:
                amounts.append(amount_val)

        # Only calculate if we have at least one valid amount
        if len(amounts) > 0:
            avg = sum(amounts) / len(amounts)
            averages[key] = avg

    return averages


def find_similar_items(target_text, items, field, threshold=0.6):
    if not target_text:
        return items
    normalized_target = normalize_text(target_text)
    similar_items = []
    for item in items:
        field_value = item.get(field, '')
        normalized_field = normalize_text(field_value)
        similarity = difflib.SequenceMatcher(
            None, normalized_target, normalized_field).ratio()
        if similarity >= threshold:
            similar_items.append(item)
    return similar_items


def extract_numeric_range(text):
    if not text:
        return None, None
    between_pattern = r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)'
    match = re.search(between_pattern, text.lower())
    if match:
        return float(match.group(1)), float(match.group(2))
    range_pattern = r'(\d+(?:\.\d+)?)\s*[-to]\s*(\d+(?:\.\d+)?)'
    match = re.search(range_pattern, text.lower())
    if match:
        return float(match.group(1)), float(match.group(2))
    greater_pattern = r'(?:greater than|more than|above|over)\s+(\d+(?:\.\d+)?)'
    match = re.search(greater_pattern, text.lower())
    if match:
        return float(match.group(1)), None
    less_pattern = r'(?:less than|below|under)\s+(\d+(?:\.\d+)?)'
    match = re.search(less_pattern, text.lower())
    if match:
        return None, float(match.group(1))
    single_pattern = r'(\d+(?:\.\d+)?)'
    match = re.search(single_pattern, text)
    if match:
        return float(match.group(1)), float(match.group(1))
    return None, None


def extract_city_from_text(text):
    if not text:
        return None
    city_patterns = [
        r'\bin\s+([A-Za-z\s]+?)(?:\s|$|,|\.)',
        r'\bfrom\s+([A-Za-z\s]+?)(?:\s|$|,|\.)',
        r'\bat\s+([A-Za-z\s]+?)(?:\s|$|,|\.)',
        r'\blocated\s+in\s+([A-Za-z\s]+?)(?:\s|$|,|\.)',
    ]
    text_lower = text.lower()
    for pattern in city_patterns:
        match = re.search(pattern, text_lower)
        if match:
            city = match.group(1).strip()
            if len(city) > 2:
                return city
    return None


def extract_measurement_from_text(text):
    if not text:
        return None, None
    measurement_pattern = r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:feet|ft|sqft|sq\s*ft)'
    match = re.search(measurement_pattern, text.lower())
    if match:
        width = float(match.group(1))
        length = float(match.group(2))
        return width * length, width * length
    single_measurement_pattern = r'(\d+(?:\.\d+)?)\s*(?:sqft|sq\s*ft|square\s*feet)'
    match = re.search(single_measurement_pattern, text.lower())
    if match:
        measurement = float(match.group(1))
        return measurement, measurement
    return None, None


def extract_amount_from_text(text):
    if not text:
        return None, None
    amount_pattern = r'[₹$]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
    matches = re.findall(amount_pattern, text)
    if len(matches) == 1:
        amount = float(matches[0].replace(',', ''))
        return amount, amount
    elif len(matches) == 2:
        min_amount = float(matches[0].replace(',', ''))
        max_amount = float(matches[1].replace(',', ''))
        return min(min_amount, max_amount), max(min_amount, max_amount)
    return None, None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------- Pydantic Models ----------------


class SearchQuery(BaseModel):
    query: str
    page: Optional[int] = None
    page_size: Optional[int] = None


class FilteredSearchQuery(BaseModel):
    item_name: Optional[str] = None
    city: Optional[str] = None
    measurement: Optional[float] = None
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None

    class Config:
        # This helps with Union type handling in older Pydantic versions
        use_enum_values = True
        validate_assignment = True


class NLPSearchQuery(BaseModel):
    query: str
    use_nlp: bool = True


class InsertItemRequest(BaseModel):
    estimator_id: int
    room_name: str
    item_name: str
    item_id: int
    amount: Optional[float] = None
    area: str
    project_name: str
    attributes: Optional[str] = None
    item_identifier: Optional[str] = None
    item_type_identifier: Optional[str] = None
    user_id: Optional[int] = None
    image: Optional[str] = None

    class Config:
        validate_assignment = True


class InsertItemsRequest(BaseModel):
    items: List[InsertItemRequest]

    class Config:
        validate_assignment = True


class InsertResponse(BaseModel):
    status: str
    inserted_count: int
    errors: List[str] = []
    success_ids: List[int] = []


class MultilingualSearchQuery(BaseModel):
    query: str
    source_language: Optional[str] = None  # Auto-detect if not provided
    target_language: str = "en"  # Default to English
    use_nlp: bool = True
    page: Optional[int] = None
    page_size: Optional[int] = None

    class Config:
        validate_assignment = True


class MultilingualSearchResponse(BaseModel):
    original_query: str
    translated_query: str
    detected_language: str
    search_results: List[Dict[str, Any]]
    total_found: int
    page: Optional[int] = None
    page_size: Optional[int] = None
    translation_confidence: Optional[float] = None

# ---------------- Helper ----------------


def translate_query(query: str,
                    source_lang: Optional[str] = None,
                    target_lang: str = "en") -> Dict[str, Any]:
    """
    Convert text in any language (including romanised Hindi) to English.
    Adds a transliteration step when needed.
    """
    try:
        # 1. Detect language if not provided
        detected = source_lang or detect(query)

        # 2. Handle romanised Hindi as an example
        #    (You can add other languages and rules here)
        if detected == "hi" and query.isascii():
            # 'itrans' is a common romanisation scheme
            query = transliterate(query, "itrans", "devanagari")

        # 3. Translate
        translator = GoogleTranslator(source=detected, target=target_lang)
        translated_text = translator.translate(query)

        print(f"Debug: Translated text: {translated_text}")
        print(f"Debug: Detected language: {detected}")
        print(f"Debug: Target language: {target_lang}")
        print(f"Debug: Confidence: 0.9")
        print(
            f"Debug: Translation needed: {detected.lower() != target_lang.lower()}")

        return {
            "translated_query": translated_text,
            "detected_language": detected,
            "confidence": 0.9,
            "translation_needed": detected.lower() != target_lang.lower()
        }

    except Exception as e:
        return {
            "translated_query": query,
            "detected_language": "unknown",
            "confidence": 0.0,
            "translation_needed": False,
            "error": str(e)
        }


def calculate_amount_from_attributes(item):
    if not item or not item.get('attributes'):
        return 0.0
    try:
        attrs = item['attributes']
        if isinstance(attrs, str):
            import json
            attrs = json.loads(attrs)
        if not attrs or not isinstance(attrs, dict):
            return 0.0
        item_id = item.get('item_type_identifier') or ''
        if 'WD' in item_id:
            return AmountCalculatorUtils.calc_woodwork_amount(type('Item', (), {'attributes': item['attributes']})())
        elif 'FC' in item_id:
            return AmountCalculatorUtils.calc_false_ceiling_amount(type('Item', (), {'attributes': item['attributes']})())
        elif 'ACS' in item_id:
            return AmountCalculatorUtils.calc_accessories_amount(type('Item', (), {'attributes': item['attributes']})())
        elif 'LF' in item_id:
            return AmountCalculatorUtils.calc_loose_furniture_amount(type('Item', (), {'attributes': item['attributes']})())
        elif 'OTH' in item_id:
            return AmountCalculatorUtils.calc_other_service_amount(type('Item', (), {'attributes': item['attributes']})())
        return 0.0
    except Exception as e:
        print(f"Error calculating amount: {e}")
        return 0.0

# ---------------- Routes ----------------


@app.post("/search")
def search_items(data: SearchQuery):
    vector = MODEL.encode(data.query).tolist()
    if data.page is None or data.page_size is None:
        result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=10000
        )
    else:
        page = max(1, data.page)
        page_size = max(1, min(50, data.page_size))
        offset = (page - 1) * page_size
        result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=page_size,
            offset=offset
        )
    items = []
    for r in result:
        try:
            item = r.payload.copy()
            current_amount = item.get('amount')
            if not current_amount or current_amount == 0:
                calculated_amount = calculate_amount_from_attributes(item)
                if calculated_amount > 0:
                    item['amount'] = calculated_amount
            items.append(item)
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    if data.page is None or data.page_size is None:
        return {"results": items}
    else:
        return {"results": items, "page": data.page, "page_size": data.page_size}


# ---------------- Speech → English → Search ----------------


@app.post("/speech/query")
async def speech_query(audio: UploadFile = File(...)):
    """
    Accepts an audio file (PCM WAV recommended), converts to English query string,
    and returns it. The frontend can then call the standard search endpoints with
    the returned English text.

    Returns:
        {"english_query": "..."}
    """
    try:
        # Use transcribe_audio from workers.stt to convert audio to text
        transcribed_text = transcribe_audio(audio)
        if not transcribed_text or not transcribed_text.strip():
            raise HTTPException(
                status_code=400, detail="No speech detected in audio")
        # Translate to English if needed
        english_text, _, _ = translate_with_confidence(transcribed_text)
        print(f"Debug: English text: {english_text}")
        return {"english_query": english_text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Speech processing failed: {e}")


@app.post("/speech/text-to-english")
async def text_to_english(payload: Dict[str, str]):
    """
    Fast text translation: detect language and translate to English.
    Optimized for speed with caching and early returns.
    """
    txt = payload.get("text", "") if payload else ""
    src = payload.get("source_language") if payload else None
    if not txt:
        return {"english_query": ""}

    try:
        # Fast path: use simple Google translation
        if src:
            print(
                f"GoogleTranslator conversion Text to english: Text: {txt}, Source language: {src}")
            # If source language is specified, use translate_to_english
            english_text = translate_to_english(txt, src)
            detected_src = src
        else:
            # Use intelligent_translate for auto-detection
            english_text, detected_src = intelligent_translate(txt, src)
            print(
                f"Intelligent translate Text to english: Text: {txt}, Source language: {src}, English text: {english_text}, Detected language: {detected_src}")

        return {"english_query": english_text, "detected_language": detected_src}
    except Exception as e:
        # Fallback: return original text if translation fails
        return {"english_query": txt, "detected_language": "unknown"}


def extract_intent_and_keywords(query: str) -> Dict[str, Any]:
    """
    Enhanced NLP extraction for natural language queries.
    Handles queries like "I want a bed", "I need a console table", "items for 1 BHK flat"
    """
    query_lower = query.lower().strip()

    # Common furniture and home items mapping
    furniture_keywords = {
        'bed': ['bed', 'beds', 'mattress', 'sleeping'],
        'table': ['table', 'tables', 'dining table', 'coffee table', 'console table', 'study table'],
        'chair': ['chair', 'chairs', 'seating', 'dining chair'],
        'sofa': ['sofa', 'sofas', 'couch', 'settee'],
        'wardrobe': ['wardrobe', 'wardrobes', 'closet', 'almirah'],
        'door': ['door', 'doors', 'entrance door', 'room door'],
        'window': ['window', 'windows', 'ventilation'],
        'kitchen': ['kitchen', 'kitchen cabinet', 'kitchen unit'],
        'bathroom': ['bathroom', 'toilet', 'washroom'],
        'lighting': ['light', 'lights', 'lighting', 'lamp', 'bulb'],
        'flooring': ['floor', 'flooring', 'tiles', 'marble'],
        'ceiling': ['ceiling', 'roof', 'false ceiling']
    }

    # Property type keywords
    property_keywords = {
        '1bhk': ['1 bhk', '1bhk', 'one bhk', '1 bedroom'],
        '2bhk': ['2 bhk', '2bhk', 'two bhk', '2 bedroom'],
        '3bhk': ['3 bhk', '3bhk', 'three bhk', '3 bedroom'],
        'studio': ['studio', 'studio apartment'],
        'flat': ['flat', 'apartment', 'unit'],
        'house': ['house', 'villa', 'home']
    }

    # Extract intent
    intent = "search"  # default
    if any(word in query_lower for word in ['want', 'need', 'looking for', 'require', 'search for']):
        intent = "search"
    elif any(word in query_lower for word in ['show', 'display', 'list', 'find']):
        intent = "list"
    elif any(word in query_lower for word in ['buy', 'purchase', 'order']):
        intent = "purchase"

    # Extract furniture/home items
    found_items = []
    for category, keywords in furniture_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                found_items.append(category)
                break

    # Extract property type
    found_property = None
    for prop_type, keywords in property_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                found_property = prop_type
                break

    # Extract room types
    room_keywords = {
        'bedroom': ['bedroom', 'master bedroom', 'guest room'],
        'living': ['living room', 'hall', 'drawing room'],
        'kitchen': ['kitchen', 'cooking area'],
        'bathroom': ['bathroom', 'toilet', 'washroom'],
        'dining': ['dining room', 'dining area'],
        'study': ['study room', 'office', 'work area']
    }

    found_rooms = []
    for room_type, keywords in room_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                found_rooms.append(room_type)
                break

    return {
        'intent': intent,
        'furniture_items': found_items,
        'property_type': found_property,
        'room_types': found_rooms,
        'original_query': query
    }


@app.post("/search/nlp")
def search_with_nlp(data: NLPSearchQuery):
    try:
        # Enhanced NLP processing
        nlp_analysis = extract_intent_and_keywords(data.query)

        # Get vector search results
        vector = MODEL.encode(data.query).tolist()
        result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=10000
        )

        all_items = []
        for r in result:
            try:
                item = r.payload.copy()
                current_amount = item.get('amount')
                if not current_amount or current_amount == 0:
                    calculated_amount = calculate_amount_from_attributes(item)
                    if calculated_amount > 0:
                        item['amount'] = calculated_amount
                all_items.append(item)
            except Exception as e:
                print(f"Error processing item: {e}")
                continue

        if not data.use_nlp:
            return {"results": all_items, "extracted_filters": {}, "nlp_analysis": nlp_analysis}

        # Enhanced filtering based on NLP analysis
        filtered_items = all_items
        extracted_filters = {}

        # Filter by furniture items if found
        if nlp_analysis['furniture_items']:
            furniture_filtered = []
            for item in all_items:
                item_name_lower = item.get('item_name', '').lower()
                item_keywords = [k.lower() for k in item.get('keywords', [])]

                # Check if any furniture item matches
                for furniture in nlp_analysis['furniture_items']:
                    if (furniture in item_name_lower or
                            any(furniture in kw for kw in item_keywords)):
                        furniture_filtered.append(item)
                        break

            if furniture_filtered:
                filtered_items = furniture_filtered
                extracted_filters['furniture_items'] = nlp_analysis['furniture_items']

        # Filter by room type if found
        if nlp_analysis['room_types']:
            room_filtered = []
            for item in filtered_items:
                room_name = item.get('room_name', '').lower()
                for room_type in nlp_analysis['room_types']:
                    if room_type in room_name:
                        room_filtered.append(item)
                        break

            if room_filtered:
                filtered_items = room_filtered
                extracted_filters['room_types'] = nlp_analysis['room_types']

        # Apply traditional filters
        city = extract_city_from_text(data.query)
        if city:
            extracted_filters['city'] = city
            filtered_items = [
                item for item in filtered_items if city.lower() in item.get('area', '').lower()]

        measurement_min, measurement_max = extract_measurement_from_text(
            data.query)
        if measurement_min is not None:
            extracted_filters['measurement_min'] = measurement_min
        if measurement_max is not None:
            extracted_filters['measurement_max'] = measurement_max

        amount_min, amount_max = extract_amount_from_text(data.query)
        if amount_min is not None:
            extracted_filters['amount_min'] = amount_min
        if amount_max is not None:
            extracted_filters['amount_max'] = amount_max
        if 'city' in extracted_filters:
            filtered_items = find_similar_items(
                extracted_filters['city'],
                filtered_items,
                'area',
                threshold=0.6
            )
        if 'measurement_min' in extracted_filters or 'measurement_max' in extracted_filters:
            measurement_filtered = []
            for item in filtered_items:
                measurement = item.get('measurement_sqft', 0) or 0
                min_val = extracted_filters.get('measurement_min')
                max_val = extracted_filters.get('measurement_max')
                if min_val is not None and max_val is not None:
                    if min_val <= measurement <= max_val:
                        measurement_filtered.append(item)
                elif min_val is not None:
                    if measurement >= min_val:
                        measurement_filtered.append(item)
                elif max_val is not None:
                    if measurement <= max_val:
                        measurement_filtered.append(item)
            filtered_items = measurement_filtered
        if 'amount_min' in extracted_filters or 'amount_max' in extracted_filters:
            amount_filtered = []
            for item in filtered_items:
                amount = item.get('amount', 0) or 0
                min_val = extracted_filters.get('amount_min')
                max_val = extracted_filters.get('amount_max')
                if min_val is not None and max_val is not None:
                    if min_val <= amount <= max_val:
                        amount_filtered.append(item)
                elif min_val is not None:
                    if amount >= min_val:
                        amount_filtered.append(item)
                elif max_val is not None:
                    if amount <= max_val:
                        amount_filtered.append(item)
            filtered_items = amount_filtered
        return {
            "results": filtered_items,
            "extracted_filters": extracted_filters,
            "total_found": len(filtered_items),
            "nlp_analysis": nlp_analysis
        }
    except Exception as e:
        print(f"Error in NLP search: {e}")
        return {"results": [], "extracted_filters": {}, "error": str(e), "nlp_analysis": {}}



class NLPSearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 10000
    use_nlp: Optional[bool] = False
    is_voice: Optional[bool] = False
    page: Optional[int] = 1
    page_size: Optional[int] = 20


# Embedding and keyword extraction models (must match Qdrant collection dim=384)
# Defensive imports: some deployments may run an older image where top-level imports differ.
try:
    SentenceTransformer  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    from sentence_transformers import SentenceTransformer  # type: ignore[no-redef]

try:
    KeyBERT  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    from keybert import KeyBERT  # type: ignore[no-redef]

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
KW_MODEL = KeyBERT(model=MODEL)


def build_retrieval_prompt(user_query: str, top_k: int = 10) -> str:
    """
    Returns a ready-to-send prompt for an external AI planner to steer semantic search.
    """
    template = (
        "You are a semantic retrieval planner for an interior/furniture catalog stored in a vector database (Qdrant).\n\n"
        "Task:\n"
        "- Understand the user’s intent from the query.\n"
        "- Propose high-signal keywords and keyphrases (1–2 grams) for payload filters.\n"
        "- Produce an English embedding text that best captures the intent for vector search.\n"
        "- Return a clear, minimal plan the backend can follow.\n\n"
        "Context:\n"
        "- Data fields: item_name, room_name, project_name, area, attributes_parsed (Material, Finish, Measurement, Rate), keywords (precomputed).\n"
        "- Vector model: all-MiniLM-L6-v2 (384-dim).\n"
        "- Backend execution: filter keywords ANY; embed embedding_text; vector search limit=TOP_K; post-enrich.\n\n"
        "Rules:\n"
        "- Prefer specific, discriminative keywords over generic ones.\n"
        "- Include only domain-relevant terms (e.g., console table, blockboard).\n"
        "- Exclude connectors/fillers (and, with, of, by, for, made, using, diye/toiri, etc.).\n"
        "- Never invent attributes not implied by the query.\n\n"
        "Input:\n"
        f"- user_query: \"{user_query}\"\n"
        f"- top_k: {top_k}\n\n"
        "Output (JSON only):\n"
        "{\n"
        "  \"embedding_text\": \"STRING — final English text to embed for vector search\",\n"
        "  \"keywords\": [\"TERM1\", \"TERM2\"],\n"
        "  \"notes\": \"Optional short note on interpretation\"\n"
        "}"
    )
    return template


class VectorPlan(BaseModel):
    embedding_text: str
    keywords: List[str] = []
    top_k: Optional[int] = 10000
    item_name: Optional[str] = None  # from multilingual_item_extractor; filters Qdrant by item_name


# ----- Script detection -----


def detect_script(text: str) -> str:
    """Detect script/language based on Unicode range"""
    if not text:
        return "unknown"

    for ch in text:
        code_point = ord(ch)
        if 0x0900 <= code_point <= 0x097F:
            return "devanagari"    # Hindi, Marathi
        elif 0x0980 <= code_point <= 0x09FF:
            return "bengali"
        elif 0x0B80 <= code_point <= 0x0BFF:
            return "tamil"
        elif 0x0C00 <= code_point <= 0x0C7F:
            return "telugu"
        elif 0x0C80 <= code_point <= 0x0CFF:
            return "kannada"
        elif 0x0D00 <= code_point <= 0x0D7F:
            return "malayalam"
        elif 0x0000 <= code_point <= 0x007F:
            return "latin"
    return "unknown"

# ----- Normalization according to script -----


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()
    script = detect_script(text)
    # Remove zero-width characters
    text = re.sub(r"[\u200c\u200d]", "", text)

    if script in ("devanagari", "bengali", "tamil", "telugu", "kannada", "malayalam"):
        text = unicodedata.normalize("NFC", text)
    else:
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()

    return text


# ----- Main vector plan execution -----

# Detect script/language based on Unicode range

def detect_script(text: str) -> str:
    if not text:
        return "unknown"
    for ch in text:
        code_point = ord(ch)
        if 0x0900 <= code_point <= 0x097F:
            return "devanagari"    # Hindi, Marathi
        elif 0x0980 <= code_point <= 0x09FF:
            return "bengali"
        elif 0x0B80 <= code_point <= 0x0BFF:
            return "tamil"
        elif 0x0C00 <= code_point <= 0x0C7F:
            return "telugu"
        elif 0x0C80 <= code_point <= 0x0CFF:
            return "kannada"
        elif 0x0D00 <= code_point <= 0x0D7F:
            return "malayalam"
        elif 0x0000 <= code_point <= 0x007F:
            return "latin"
    return "unknown"

# Normalize text according to script


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    script = detect_script(text)
    # Remove zero-width characters
    text = re.sub(r"[\u200c\u200d]", "", text)
    if script in ("devanagari", "bengali", "tamil", "telugu", "kannada", "malayalam"):
        text = unicodedata.normalize("NFC", text)
    else:
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
    return text


# Detect script/language based on Unicode range


def detect_script(text: str) -> str:
    if not text:
        return "unknown"
    for ch in text:
        code_point = ord(ch)
        if 0x0900 <= code_point <= 0x097F:
            return "devanagari"    # Hindi, Marathi
        elif 0x0980 <= code_point <= 0x09FF:
            return "bengali"
        elif 0x0B80 <= code_point <= 0x0BFF:
            return "tamil"
        elif 0x0C00 <= code_point <= 0x0C7F:
            return "telugu"
        elif 0x0C80 <= code_point <= 0x0CFF:
            return "kannada"
        elif 0x0D00 <= code_point <= 0x0D7F:
            return "malayalam"
        elif 0x0000 <= code_point <= 0x007F:
            return "latin"
    return "unknown"

# Normalize text according to script


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()
    script = detect_script(text)
    # Remove zero-width characters
    text = re.sub(r"[\u200c\u200d]", "", text)

    if script in ("devanagari", "bengali", "tamil", "telugu", "kannada", "malayalam"):
        text = unicodedata.normalize("NFC", text)
    else:
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()

    return text


from typing import Dict, Any

from typing import Dict, Any, List, Set

def execute_vector_plan(plan: "VectorPlan") -> Dict[str, Any]:
    import unicodedata, re, json

    try:
        # --- Helper function to normalize text ---
        def normalize_text(text: str) -> str:
            if not isinstance(text, str):
                return ""
            text = text.strip()
            text = re.sub(r"[\u200c\u200d]", "", text)
            return unicodedata.normalize("NFKC", text).lower()

        # --- Helper function to check if item matches plan keywords ---
        def matches_plan_keywords(item: dict, plan_keywords: Set[str]) -> bool:
            # Normalize keywords
            raw_keywords = item.get("keywords", [])
            if isinstance(raw_keywords, str):
                try:
                    raw_keywords = json.loads(raw_keywords)
                    if not isinstance(raw_keywords, list):
                        raw_keywords = [str(raw_keywords)]
                except Exception:
                    raw_keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
            keywords_list = [normalize_text(k) for k in (raw_keywords or []) if k]

            # Normalize area
            area_norm = normalize_text(item.get("area", ""))

            # Flatten and normalize attributes
            attrs_dict = item.get("attributes_parsed") or item.get("parsed_attrs") or {}
            attrs_norm = []
            if isinstance(attrs_dict, dict):
                for v in attrs_dict.values():
                    if v is None:
                        continue
                    if isinstance(v, list):
                        attrs_norm.extend([normalize_text(str(sv)) for sv in v])
                    else:
                        attrs_norm.append(normalize_text(str(v)))

            # Combine searchable fields
            searchable_fields = set(keywords_list + attrs_norm + [area_norm])
            searchable_fields.discard("")  # remove empty strings

            # Check if any plan keyword exists in searchable fields
            return bool(searchable_fields.intersection(plan_keywords))

        # --- Prepare plan keywords ---
        plan_keywords = set(normalize_text(kw) for kw in (plan.keywords or []) if kw.strip())
        has_item_name = bool(getattr(plan, "item_name", None) and str(plan.item_name).strip())
        if not plan_keywords and not has_item_name:
            return {"results": [], "total_found": 0, "error": "No keywords or item_name provided"}

        # --- Optional: query vector for embedding search ---
        query_vector = None
        if getattr(plan, "embedding_text", None):
            query_vector = MODEL.encode(plan.embedding_text).tolist()

        # --- Execute vector DB search ---
        must_conditions = []
        if plan.keywords:
            try:
                from qdrant_client.http.models import Filter, FieldCondition, MatchAny
                must_conditions.append(
                    FieldCondition(key="keywords", match=MatchAny(any=list(plan_keywords)))
                )
            except Exception:
                pass
        # Filter by item_name when provided (from multilingual_item_extractor)
        if getattr(plan, "item_name", None) and str(plan.item_name).strip():
            try:
                from qdrant_client.http.models import MatchValue
                must_conditions.append(
                    FieldCondition(key="item_name", match=MatchValue(value=plan.item_name.strip()))
                )
            except Exception:
                pass
        q_filter = None
        if must_conditions:
            try:
                from qdrant_client.http.models import Filter
                q_filter = Filter(must=must_conditions)
            except Exception:
                q_filter = None

        raw_items = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=getattr(plan, "top_k", 10000) or 10000,
            query_filter=q_filter,
        )

        # --- Filter results by keywords, area, attributes (skip when only item_name filter) ---
        results = []
        for r in raw_items:
            try:
                item = getattr(r, "payload", r) if hasattr(r, "payload") else r.copy()
                item["id"] = getattr(r, "id", None)
                similarity = getattr(r, "score", None)
                if similarity is not None:
                    item["similarity_score"] = similarity
                if plan_keywords:
                    if matches_plan_keywords(item, plan_keywords):
                        item["score"] = 1.0  # keyword match
                        results.append(item)
                else:
                    # Only item_name filter (from multilingual extractor); accept all from Qdrant
                    item["score"] = getattr(r, "score", 1.0)
                    results.append(item)
            except Exception as ex:
                print("Item processing error:", ex)
                continue

        total_found = len(results)
        return {
            "keywords": list(plan_keywords),
            "results": results,
            "total_found": total_found,
            "page": 1,
            "page_size": 20,
            "total_pages": (total_found + 19) // 20,
            "has_next": total_found > 20,
            "has_previous": False,
            "search_type": "keyword_match_fields_only",
        }

    except Exception as e:
        return {"results": [], "total_found": 0, "error": str(e)}


@app.post("/search/voice")
async def search_voice(file: UploadFile = File(...)):
    """
    Voice search endpoint: audio → text → translation → vector search.

    Pipeline:
    1. Transcribe audio to text using faster-whisper
    2. Detect language and translate to English if needed
    3. Perform vector search with translated text
    4. Return results with transcription and translation info
    """
    try:
        # Step 1: Transcribe audio to text
        transcribed_text = transcribe_audio(file)

        if not transcribed_text or not transcribed_text.strip():
            raise HTTPException(
                status_code=400, detail="No speech detected in audio")

        # Step 2: Translate to English if needed
        translated_text, detected_language, confidence = translate_with_confidence(
            transcribed_text)

        # Step 3: Perform vector search with translated text
        search_data = NLPSearchQuery(query=translated_text, use_nlp=True)
        search_results = search_with_vector_similarity(search_data)

        # Step 4: Format results
        formatted_results = []
        for item in search_results.get("results", []):
            formatted_item = {
                "id": item.get("id"),
                "score": round(item.get("similarity_score", 0.0), 3),
                "item_name": item.get("item_name"),
                "attributes_parsed": item.get("attributes_parsed", {}),
                "image": item.get("image"),
                "amount": str(item.get("amount", "0.00"))
            }
            formatted_results.append(formatted_item)

        return {
            "query_transcribed": transcribed_text,
            "query_translated": translated_text,
            "detected_language": detected_language,
            "translation_confidence": confidence,
            "results": formatted_results,
            "total_found": len(formatted_results)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Voice search failed: {str(e)}")


@app.post("/search/vector")
def search_with_vector_similarity(data: NLPSearchQuery) -> Dict[str, Any]:
    """
    Simplified vector search using multilingual_item_extractor to get item_name and search Qdrant.
    - Extracts item_name from query using multilingual_item_extractor
    - Performs vector search in Qdrant filtered by item_name
    - Returns results ranked by vector similarity
    """
    try:
        query_text = (data.query or "").strip()
        if not query_text:
            return {"results": [], "query": data.query, "total_found": 0}

        # Extract item_name from query using multilingual_item_extractor
        extracted_item_name = extract_item_name_from_query(query_text)

        print(f"Query text: ------------------------------- {query_text}")
        print(f"Extracted item name: ------------------------------- {extracted_item_name}")

        # Create vector embedding from query
        query_vector = MODEL.encode(query_text).tolist()

        # For maximum recall, we DON'T filter in Qdrant by item_name.
        # Instead we run vector search over the whole collection and
        # then post-filter results by whether item_name contains the
        # extracted canonical name (or any of its comma-separated parts).
        q_filter = None

        # Search Qdrant (support multiple qdrant-client APIs)
        top_k = getattr(data, "top_k", 10000) or 10000

        raw_items = None
        search_fn = getattr(client, "search", None)
        if callable(search_fn):
            raw_items = search_fn(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                query_filter=q_filter,
            )

        if raw_items is None:
            search_points_fn = getattr(client, "search_points", None)
            if callable(search_points_fn):
                raw_items = search_points_fn(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=q_filter,
                )

        if raw_items is None:
            # Newer clients expose `query_points`
            query_points_fn = getattr(client, "query_points", None)
            if callable(query_points_fn):
                raw_items = query_points_fn(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=q_filter,
                    limit=top_k,
                    with_payload=True,
                )
                # query_points returns QueryResponse with `.points`
                if hasattr(raw_items, "points"):
                    raw_items = getattr(raw_items, "points")

        if raw_items is None:
            raise RuntimeError(
                "Qdrant client does not expose search/search_points/query_points. "
                "Please upgrade 'qdrant-client' in backend/requirements.txt."
            )

        # Format results
        all_formatted_results = []
        for r in raw_items:
            try:
                item = getattr(r, "payload", r) if hasattr(r, "payload") else r.copy()
                item["id"] = getattr(r, "id", None)
                similarity = getattr(r, "score", None)
                
                formatted_item = {
                    "id": item.get("id"),
                    "area": item.get("area"),
                    "project_name": item.get("project_name"),
                    "room_name": item.get("room_name"),
                    "item_identifier": item.get("item_identifier"),
                    "item_type_identifier": item.get("item_type_identifier"),
                    "description": item.get("description"),
                    "score": round(similarity if similarity is not None else 0.0, 3),
                    "item_name": item.get("item_name"),
                    "attributes_parsed": item.get("attributes_parsed", {}),
                    "image": item.get("image"),
                    "amount": str(item.get("amount", "0.00"))
                }
                all_formatted_results.append(formatted_item)
            except Exception as ex:
                print(f"Item processing error: {ex}")
                continue

        # If we have an extracted item name, (1) keep only items whose
        # item_name CONTAINS any of the canonical names (case-insensitive),
        # and (2) re-rank so exact matches for the canonical names come first,
        # then strong word-boundary matches (e.g. "bed side table"), then
        # weaker substring matches.
        if extracted_item_name:
            parts = [
                p.strip().lower()
                for p in str(extracted_item_name).split(",")
                if p.strip()
            ]
            if parts:
                filtered = []
                for item in all_formatted_results:
                    iname = (item.get("item_name") or "").strip().lower()
                    if any(part in iname for part in parts):
                        filtered.append(item)
                # Only replace if we found at least one match; otherwise
                # fall back to the original vector-ranked results.
                if filtered:
                    def _boost_score(it):
                        base = float(it.get("score", 0.0))
                        name = (it.get("item_name") or "").strip().lower()
                        boost = 0.0
                        for p in parts:
                            if not p:
                                continue
                            if name == p:
                                # Exact item ("bed")
                                boost = max(boost, 2.0)
                            else:
                                # Word-boundary / prefix match ("bed side table")
                                if name.startswith(p + " ") or (" " + p + " ") in name or name.endswith(" " + p):
                                    boost = max(boost, 1.0)
                                # Any substring ("storage for bed")
                                elif p in name:
                                    boost = max(boost, 0.5)
                        return base + boost

                    all_formatted_results = sorted(
                        filtered, key=_boost_score, reverse=True
                    )

        # Compute Area-wise Average Price
        averages = calculate_area_wise_averages(all_formatted_results)
        for item in all_formatted_results:
            area = (item.get("area") or "").strip().lower()
            item_name = (item.get("item_name") or "").strip().lower()
            if area and item_name:
                key = f"{area}|{item_name}"
                if key in averages:
                    item["average_price"] = round(averages[key], 2)
                else:
                    item["average_price"] = None
            else:
                item["average_price"] = None

        # Pagination
        page = max(1, getattr(data, 'page', 1) or 1)
        page_size = max(1, min(100, getattr(data, 'page_size', 20) or 20))
        total_count = len(all_formatted_results)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = all_formatted_results[start_idx:end_idx]
        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0

        # Final Response
        return {
            "query": data.query,
            "extracted_item_name": extracted_item_name,
            "results": paginated_results,
            "total_found": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1,
            "search_type": "vector_search_with_item_extractor"
        }

    except Exception as e:
        print("❌ Error in search_with_vector_similarity:", e)
        return {"results": [], "error": str(e), "total_found": 0}


@app.post("/search/vector/prompt")
def get_vector_search_prompt(data: NLPSearchQuery) -> Dict[str, Any]:
    """Builds and returns a prebuilt AI prompt to plan the retrieval for a given query."""
    q = (data.query or "").strip()
    prompt = build_retrieval_prompt(q, top_k=getattr(data, 'top_k', 10) or 10)
    # Also provide baseline keywords we computed locally (optional)
    try:
        kw_pairs = KW_MODEL.extract_keywords(
            q, keyphrase_ngram_range=(1, 2), stop_words='english')
        keywords = [kw for kw, _ in kw_pairs]
    except Exception:
        keywords = []
    return {"prompt": prompt, "baseline_keywords": keywords}


@app.post("/search/vector/by-plan")
def search_with_vector_plan(plan: VectorPlan) -> Dict[str, Any]:
    """
    Executes a search using a plan produced by an external AI:
    - Embeds plan.embedding_text
    - Filters by plan.keywords (keywords ANY) if provided
    - Searches Qdrant and returns results
    """
    try:
        embed_text = (plan.embedding_text or "").strip()
        if not embed_text:
            return {"results": [], "total_found": 0, "error": "empty embedding_text"}

        query_vector = MODEL.encode(embed_text).tolist()

        q_filter = None
        if plan.keywords:
            try:
                q_filter = Filter(must=[FieldCondition(
                    key="keywords", match=MatchAny(any=plan.keywords))])
            except Exception:
                q_filter = None

        raw = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=getattr(plan, 'top_k', 10000) or 10000,
            query_filter=q_filter
        )

        items: List[Dict[str, Any]] = []
        for r in raw:
            try:
                item = r.payload.copy()
                item['id'] = r.id
                item['similarity_score'] = float(getattr(r, 'score', 0.0))
                if not item.get('amount') or item.get('amount') == 0:
                    calc = calculate_amount_from_attributes(item)
                    if calc > 0:
                        item['amount'] = calc
                items.append(item)
            except Exception as ex:
                print("Vector result formatting error:", ex)
                continue

        return {
            "results": items,
            "embedding_text": plan.embedding_text,
            "keywords": plan.keywords,
            "total_found": len(items)
        }
    except Exception as e:
        return {"results": [], "error": str(e), "total_found": 0}


@app.post("/search/filtered-stats")
def get_filtered_stats(data: FilteredSearchQuery) -> Dict[str, Any]:
    """
    Get filtered statistics.

    - Queries Qdrant without filters
    - Automatically clusters similar city names using SequenceMatcher
    - Returns only:
        * city/cluster name
        * average amount across the cluster
        * total count
    """

    print(f"DEBUG: Received data: {data}")

    # 1️⃣ Fetch all items from Qdrant (ignore filters)
    result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=None,
        limit=10000
    )

    items: List[Dict[str, Any]] = []
    for point in result[0]:
        item = point.payload.copy()

        # Calculate amount if missing
        if not item.get("amount") or item.get("amount") == 0:
            calculated_amount = calculate_amount_from_attributes(item)
            if calculated_amount > 0:
                item["amount"] = calculated_amount

        items.append(item)

    # 2️⃣ Helper to normalize and canonicalize raw area names
    def canonicalize_city(name: str) -> str:
        raw = (name or "unknown").strip().lower()
        return re.sub(r"[^a-z]", "", raw) or "unknown"

    # 3️⃣ Build list of unique normalized city names
    unique_areas = sorted({canonicalize_city(i.get("area")) for i in items})

    # 4️⃣ Build clusters of similar city names dynamically
    def build_clusters(areas: List[str], threshold: float = 0.8) -> Dict[str, List[str]]:
        """
        Group similar names together if their SequenceMatcher ratio >= threshold.
        Returns dict: {cluster_representative: [members]}
        """
        clusters: Dict[str, List[str]] = {}
        assigned: set[str] = set()

        for area in areas:
            if area in assigned:
                continue

            # start a new cluster
            clusters[area] = [area]
            assigned.add(area)

            for other in areas:
                if other in assigned:
                    continue
                if SequenceMatcher(None, area, other).ratio() >= threshold:
                    clusters[area].append(other)
                    assigned.add(other)

        return clusters

    clusters = build_clusters(unique_areas, threshold=0.8)

    # Reverse lookup: member → cluster representative
    member_to_cluster = {
        member: representative
        for representative, members in clusters.items()
        for member in members
    }

    # 5️⃣ Aggregate stats by city and measurement (group by both)
    merged_stats = defaultdict(
        lambda: {"area": "", "measurement_sqft": None, "amounts": [], "count": 0})

    for item in items:
        norm_area = canonicalize_city(item.get("area"))
        cluster_rep = member_to_cluster.get(norm_area, norm_area)

        try:
            amount = float(item.get("amount") or 0)
        except (ValueError, TypeError):
            amount = 0

        try:
            measurement = float(item.get("measurement_sqft")) if item.get(
                "measurement_sqft") is not None else None
        except (ValueError, TypeError):
            measurement = None

        # Create unique key for city + measurement combination
        measurement_key = round(
            measurement, 2) if measurement is not None else None
        group_key = f"{cluster_rep}_{measurement_key}"

        ms = merged_stats[group_key]
        if not ms["area"]:
            ms["area"] = cluster_rep
            ms["measurement_sqft"] = measurement_key
        ms["amounts"].append(amount)
        ms["count"] += 1

    # 6️⃣ Final list: grouped by city and measurement
    final_area_stats: List[Dict[str, Any]] = []
    for g in merged_stats.values():
        valid_amounts = [a for a in g["amounts"]
                         if isinstance(a, (int, float))]
        if not valid_amounts:
            continue

        # Calculate 2x2 format for measurement
        measurement_2x2 = None
        if g["measurement_sqft"] is not None:
            # Assume square measurement, calculate 2x2 format
            side_length = (g["measurement_sqft"] ** 0.5)
            measurement_2x2 = f"{round(side_length, 2)}x{round(side_length, 2)}"

        final_area_stats.append({
            "area": g["area"],
            "measurement_sqft": g["measurement_sqft"],
            "measurement_2x2": measurement_2x2,
            "min": round(min(valid_amounts), 2),
            "max": round(max(valid_amounts), 2),
            "avg": round(sum(valid_amounts) / len(valid_amounts), 2),
            "count": g["count"]
        })

    final_area_stats.sort(key=lambda x: x["area"])

    return {"results": items, "area_stats": final_area_stats}


@app.post("/search/filtered-stats/only")
def get_filtered_stats_only(data: FilteredSearchQuery) -> Dict[str, Any]:
    """
    Get filtered statistics (only area_stats) with improved city grouping.
    - Applies filters for item_name, city, and measurement
    - Groups similar city names using alias-based canonicalization + fuzzy clustering
    - Returns area_stats with min/max/avg amounts and measurement ranges
    """
    print(f"DEBUG: Received data: {data}")

    # 1️⃣ Build filters based on request data
    filters: List[FieldCondition] = []
    try:
        data_dict = data.dict() if hasattr(data, "dict") else data.__dict__

        if data_dict.get("item_name"):
            filters.append(FieldCondition(
                key="item_name", match=MatchValue(value=data_dict["item_name"])
            ))
        if data_dict.get("city"):
            filters.append(FieldCondition(
                key="area", match=MatchValue(value=data_dict["city"])
            ))
        measurement = data_dict.get("measurement")
        if measurement is not None:
            # For single measurement value, we'll use a range with some tolerance
            # This allows for slight variations in measurement values
            tolerance = measurement * 0.1  # 10% tolerance
            filters.append(FieldCondition(
                key="measurement_sqft",
                range=Range(
                    gte=measurement - tolerance,
                    lte=measurement + tolerance
                )
            ))
    except Exception as e:
        print(f"DEBUG: Error building filters (only): {e}")

    filter_obj = Filter(must=filters) if filters else None

    # 2️⃣ Query Qdrant with filters
    result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_obj,
        limit=10000
    )

    # 3️⃣ Collect items and calculate amount
    items: List[Dict[str, Any]] = []
    for point in result[0]:
        item = point.payload.copy()
        if not item.get("amount") or item.get("amount") == 0:
            calculated_amount = calculate_amount_from_attributes(item)
            if calculated_amount > 0:
                item["amount"] = calculated_amount
        items.append(item)

    # 4️⃣ Helper to normalize and canonicalize raw area names
    def canonicalize_city(name: str) -> str:
        raw = (name or "unknown").strip().lower()
        return re.sub(r"[^a-z]", "", raw) or "unknown"

    # 5️⃣ Build list of unique normalized city names
    unique_areas = sorted({canonicalize_city(i.get("area")) for i in items})

    # 6️⃣ Build clusters of similar city names dynamically
    def build_clusters(areas: List[str], threshold: float = 0.8) -> Dict[str, List[str]]:
        """
        Group similar names together if their SequenceMatcher ratio >= threshold.
        Returns dict: {cluster_representative: [members]}
        """
        clusters: Dict[str, List[str]] = {}
        assigned: set[str] = set()

        for area in areas:
            if area in assigned:
                continue

            # start a new cluster
            clusters[area] = [area]
            assigned.add(area)

            for other in areas:
                if other in assigned:
                    continue
                if SequenceMatcher(None, area, other).ratio() >= threshold:
                    clusters[area].append(other)
                    assigned.add(other)

        return clusters

    clusters = build_clusters(unique_areas, threshold=0.8)

    # Reverse lookup: member → cluster representative
    member_to_cluster = {
        member: representative
        for representative, members in clusters.items()
        for member in members
    }

    # 7️⃣ Check if measurement filter is provided to determine grouping strategy
    has_measurement_filter = any([
        data_dict.get("measurement_min") is not None,
        data_dict.get("measurement_max") is not None
    ])

    # 8️⃣ Aggregate stats based on grouping strategy
    if has_measurement_filter:
        # Group by city + measurement when measurement filter is provided
        merged_stats = defaultdict(
            lambda: {"area": "", "measurement_sqft": None, "amounts": [], "count": 0})

        for item in items:
            norm_area = canonicalize_city(item.get("area"))
            cluster_rep = member_to_cluster.get(norm_area, norm_area)

            try:
                amount = float(item.get("amount") or 0)
            except (ValueError, TypeError):
                amount = 0

            try:
                measurement = float(item.get("measurement_sqft")) if item.get(
                    "measurement_sqft") is not None else None
            except (ValueError, TypeError):
                measurement = None

            # Create unique key for city + measurement combination
            measurement_key = round(
                measurement, 2) if measurement is not None else None
            group_key = f"{cluster_rep}_{measurement_key}"

            ms = merged_stats[group_key]
            if not ms["area"]:
                ms["area"] = cluster_rep
                ms["measurement_sqft"] = measurement_key
            ms["amounts"].append(amount)
            ms["count"] += 1
    else:
        # Group by city only when no measurement filter
        merged_stats = defaultdict(
            lambda: {"area": "", "amounts": [], "count": 0})

        for item in items:
            norm_area = canonicalize_city(item.get("area"))
            cluster_rep = member_to_cluster.get(norm_area, norm_area)

            try:
                amount = float(item.get("amount") or 0)
            except (ValueError, TypeError):
                amount = 0

            ms = merged_stats[cluster_rep]
            if not ms["area"]:
                ms["area"] = cluster_rep
            ms["amounts"].append(amount)
            ms["count"] += 1

    # 9️⃣ Final list based on grouping strategy
    final_area_stats: List[Dict[str, Any]] = []
    for g in merged_stats.values():
        valid_amounts = [a for a in g["amounts"]
                         if isinstance(a, (int, float))]
        if not valid_amounts:
            continue

        if has_measurement_filter:
            # Include measurement data when grouped by city + measurement
            measurement_2x2 = None
            if g.get("measurement_sqft") is not None:
                side_length = (g["measurement_sqft"] ** 0.5)
                measurement_2x2 = f"{round(side_length, 2)}x{round(side_length, 2)}"

            final_area_stats.append({
                "area": g["area"],
                "measurement_sqft": g.get("measurement_sqft"),
                "measurement_2x2": measurement_2x2,
                "min": round(min(valid_amounts), 2),
                "max": round(max(valid_amounts), 2),
                "avg": round(sum(valid_amounts) / len(valid_amounts), 2),
                "count": g["count"]
            })
        else:
            # City-only grouping
            final_area_stats.append({
                "area": g["area"],
                "min": round(min(valid_amounts), 2),
                "max": round(max(valid_amounts), 2),
                "avg": round(sum(valid_amounts) / len(valid_amounts), 2),
                "count": g["count"]
            })

    final_area_stats.sort(key=lambda x: x["area"])

    return {"area_stats": final_area_stats}


@app.post("/test-filtered-stats")
def test_filtered_stats(data: FilteredSearchQuery):
    """Test endpoint to debug the Union type issue"""
    try:
        print(f"TEST: Received data: {data}")
        print(f"TEST: data type: {type(data)}")

        # Try different ways to access the data
        print(f"TEST: hasattr item_name: {hasattr(data, 'item_name')}")
        print(
            f"TEST: dir(data): {[attr for attr in dir(data) if not attr.startswith('_')]}")

        # Try to convert to dict
        if hasattr(data, 'dict'):
            data_dict = data.dict()
            print(f"TEST: data.dict() = {data_dict}")
        else:
            print("TEST: No dict() method available")

        # Try direct attribute access
        try:
            item_name = data.item_name
            print(f"TEST: Direct access item_name = {item_name}")
        except Exception as e:
            print(f"TEST: Direct access failed: {e}")

        return {
            "status": "success",
            "data_received": str(data),
            "data_type": str(type(data)),
            "has_dict_method": hasattr(data, 'dict'),
            "attributes": [attr for attr in dir(data) if not attr.startswith('_')]
        }

    except Exception as e:
        print(f"TEST: Error in test endpoint: {e}")
        return {"error": str(e), "status": "failed"}


@app.post("/search/measurement")
def search_by_measurement(data: FilteredSearchQuery):
    """Search by measurement range"""
    filters = []

    measurement_min = getattr(data, 'measurement_min', None)
    measurement_max = getattr(data, 'measurement_max', None)
    if measurement_min is not None or measurement_max is not None:
        range_filter = {}
        if measurement_min is not None:
            range_filter["gte"] = measurement_min
        if measurement_max is not None:
            range_filter["lte"] = measurement_max
        filters.append(FieldCondition(
            key="measurement_sqft", range=Range(**range_filter)))

    item_name = getattr(data, 'item_name', None)
    if item_name is not None:
        filters.append(FieldCondition(key="item_name",
                       match=MatchValue(value=item_name)))

    city = getattr(data, 'city', None)
    if city is not None:
        filters.append(FieldCondition(
            key="area", match=MatchValue(value=city)))

    amount_min = getattr(data, 'amount_min', None)
    amount_max = getattr(data, 'amount_max', None)
    if amount_min is not None or amount_max is not None:
        range_filter = {}
        if amount_min is not None:
            range_filter["gte"] = amount_min
        if amount_max is not None:
            range_filter["lte"] = amount_max
        filters.append(FieldCondition(
            key="amount", range=Range(**range_filter)))

    filter_obj = Filter(must=filters) if filters else None

    result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_obj,
        limit=10000
    )

    items = []
    for point in result[0]:
        item = point.payload.copy()
        if not item.get('amount') or item.get('amount') == 0:
            calculated_amount = calculate_amount_from_attributes(item)
            if calculated_amount > 0:
                item['amount'] = calculated_amount
        items.append(item)

    return {"results": items}


@app.post("/search/amount")
def search_by_amount(data: FilteredSearchQuery):
    """Search by amount range"""
    filters = []

    amount_min = getattr(data, 'amount_min', None)
    amount_max = getattr(data, 'amount_max', None)
    if amount_min is not None or amount_max is not None:
        range_filter = {}
        if amount_min is not None:
            range_filter["gte"] = amount_min
        if amount_max is not None:
            range_filter["lte"] = amount_max
        filters.append(FieldCondition(
            key="amount", range=Range(**range_filter)))

    item_name = getattr(data, 'item_name', None)
    if item_name is not None:
        filters.append(FieldCondition(key="item_name",
                       match=MatchValue(value=item_name)))

    city = getattr(data, 'city', None)
    if city is not None:
        filters.append(FieldCondition(
            key="area", match=MatchValue(value=city)))

    measurement_min = getattr(data, 'measurement_min', None)
    measurement_max = getattr(data, 'measurement_max', None)
    if measurement_min is not None or measurement_max is not None:
        range_filter = {}
        if measurement_min is not None:
            range_filter["gte"] = measurement_min
        if measurement_max is not None:
            range_filter["lte"] = measurement_max
        filters.append(FieldCondition(
            key="measurement_sqft", range=Range(**range_filter)))

    filter_obj = Filter(must=filters) if filters else None

    result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_obj,
        limit=10000
    )

    items = []
    for point in result[0]:
        item = point.payload.copy()
        if not item.get('amount') or item.get('amount') == 0:
            calculated_amount = calculate_amount_from_attributes(item)
            if calculated_amount > 0:
                item['amount'] = calculated_amount
        items.append(item)

    return {"results": items}


@app.post("/search/city")
def search_by_city(data: FilteredSearchQuery):
    """Search by city/area"""
    filters = []

    city = getattr(data, 'city', None)
    if city is not None:
        filters.append(FieldCondition(
            key="area", match=MatchValue(value=city)))

    item_name = getattr(data, 'item_name', None)
    if item_name is not None:
        filters.append(FieldCondition(key="item_name",
                       match=MatchValue(value=item_name)))

    measurement_min = getattr(data, 'measurement_min', None)
    measurement_max = getattr(data, 'measurement_max', None)
    if measurement_min is not None or measurement_max is not None:
        range_filter = {}
        if measurement_min is not None:
            range_filter["gte"] = measurement_min
        if measurement_max is not None:
            range_filter["lte"] = measurement_max
        filters.append(FieldCondition(
            key="measurement_sqft", range=Range(**range_filter)))

    amount_min = getattr(data, 'amount_min', None)
    amount_max = getattr(data, 'amount_max', None)
    if amount_min is not None or amount_max is not None:
        range_filter = {}
        if amount_min is not None:
            range_filter["gte"] = amount_min
        if amount_max is not None:
            range_filter["lte"] = amount_max
        filters.append(FieldCondition(
            key="amount", range=Range(**range_filter)))

    filter_obj = Filter(must=filters) if filters else None

    result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_obj,
        limit=10000
    )

    items = []
    for point in result[0]:
        item = point.payload.copy()
        if not item.get('amount') or item.get('amount') == 0:
            calculated_amount = calculate_amount_from_attributes(item)
            if calculated_amount > 0:
                item['amount'] = calculated_amount
        items.append(item)

    return {"results": items}


@app.post("/search/itemname")
def search_by_itemname(data: FilteredSearchQuery):
    """Search by item name"""
    filters = []

    item_name = getattr(data, 'item_name', None)
    if item_name is not None:
        filters.append(FieldCondition(key="item_name",
                       match=MatchValue(value=item_name)))

    city = getattr(data, 'city', None)
    if city is not None:
        filters.append(FieldCondition(
            key="area", match=MatchValue(value=city)))

    measurement_min = getattr(data, 'measurement_min', None)
    measurement_max = getattr(data, 'measurement_max', None)
    if measurement_min is not None or measurement_max is not None:
        range_filter = {}
        if measurement_min is not None:
            range_filter["gte"] = measurement_min
        if measurement_max is not None:
            range_filter["lte"] = measurement_max
        filters.append(FieldCondition(
            key="measurement_sqft", range=Range(**range_filter)))

    amount_min = getattr(data, 'amount_min', None)
    amount_max = getattr(data, 'amount_max', None)
    if amount_min is not None or amount_max is not None:
        range_filter = {}
        if amount_min is not None:
            range_filter["gte"] = amount_min
        if amount_max is not None:
            range_filter["lte"] = amount_max
        filters.append(FieldCondition(
            key="amount", range=Range(**range_filter)))

    filter_obj = Filter(must=filters) if filters else None

    result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_obj,
        limit=10000
    )

    items = []
    for point in result[0]:
        item = point.payload.copy()
        if not item.get('amount') or item.get('amount') == 0:
            calculated_amount = calculate_amount_from_attributes(item)
            if calculated_amount > 0:
                item['amount'] = calculated_amount
        items.append(item)

    return {"results": items}


# ---------------- Multilingual Search Endpoints ----------------


@app.post("/search/multilingual", response_model=MultilingualSearchResponse)
def search_multilingual(data: MultilingualSearchQuery):
    """
    Search with multilingual support. Translates query to English and performs vector search.
    Supports auto-detection of source language or manual specification.
    """
    try:
        # Translate the query to English
        translation_result = translate_query(
            query=data.query,
            source_lang=data.source_language,
            target_lang=data.target_language
        )

        translated_query = translation_result["translated_query"]
        detected_language = translation_result["detected_language"]
        confidence = translation_result.get("confidence", 0.0)

        # Perform vector search with translated query
        vector = MODEL.encode(translated_query).tolist()

        if data.page is None or data.page_size is None:
            result = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=10000
            )
        else:
            page = max(1, data.page)
            page_size = max(1, min(50, data.page_size))
            offset = (page - 1) * page_size
            result = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=page_size,
                offset=offset
            )

        items = []
        for r in result:
            try:
                item = r.payload.copy()
                current_amount = item.get('amount')
                if not current_amount or current_amount == 0:
                    calculated_amount = calculate_amount_from_attributes(item)
                    if calculated_amount > 0:
                        item['amount'] = calculated_amount
                items.append(item)
            except Exception as e:
                print(f"Error processing item: {e}")
                continue

        return MultilingualSearchResponse(
            original_query=data.query,
            translated_query=translated_query,
            detected_language=detected_language,
            search_results=items,
            total_found=len(items),
            page=data.page,
            page_size=data.page_size,
            translation_confidence=confidence
        )

    except Exception as e:
        print(f"Error in multilingual search: {e}")
        return MultilingualSearchResponse(
            original_query=data.query,
            translated_query=data.query,
            detected_language="unknown",
            search_results=[],
            total_found=0,
            page=data.page,
            page_size=data.page_size,
            translation_confidence=0.0
        )


@app.post("/search/multilingual/nlp", response_model=MultilingualSearchResponse)
def search_multilingual_nlp(data: MultilingualSearchQuery):
    """
    Multilingual search with NLP processing. Translates query and applies NLP filters.
    """
    try:
        # Translation is already done before this call. Use the provided query as-is.
        translated_query = data.query or ""
        detected_language = data.source_language or "auto"
        confidence = 1.0

        # Use the existing NLP search logic with the (pre)translated query
        nlp_data = NLPSearchQuery(query=translated_query, use_nlp=data.use_nlp)
        nlp_result = search_with_nlp(nlp_data)

        return MultilingualSearchResponse(
            original_query=data.query,
            translated_query=translated_query,
            detected_language=detected_language,
            search_results=nlp_result.get("results", []),
            total_found=nlp_result.get("total_found", 0),
            translation_confidence=confidence
        )

    except Exception as e:
        print(f"Error in multilingual NLP search: {e}")
        return MultilingualSearchResponse(
            original_query=data.query,
            translated_query=data.query,
            detected_language="unknown",
            search_results=[],
            total_found=0,
            translation_confidence=0.0
        )


@app.post("/search/multilingual/vector", response_model=MultilingualSearchResponse)
def search_multilingual_vector(data: MultilingualSearchQuery):
    """
    Multilingual search with vector similarity. Translates query and performs keyword-first vector search.
    """
    try:
        # Translate the query to English
        translation_result = translate_query(
            query=data.query,
            source_lang=data.source_language,
            target_lang=data.target_language
        )

        translated_query = translation_result["translated_query"]
        detected_language = translation_result["detected_language"]
        confidence = translation_result.get("confidence", 0.0)

        # Use the existing vector search logic with translated query
        vector_data = NLPSearchQuery(
            query=translated_query, use_nlp=data.use_nlp)
        vector_result = search_with_vector_similarity(vector_data)

        return MultilingualSearchResponse(
            original_query=data.query,
            translated_query=translated_query,
            detected_language=detected_language,
            search_results=vector_result.get("results", []),
            total_found=vector_result.get("total_found", 0),
            translation_confidence=confidence
        )

    except Exception as e:
        print(f"Error in multilingual vector search: {e}")
        return MultilingualSearchResponse(
            original_query=data.query,
            translated_query=data.query,
            detected_language="unknown",
            search_results=[],
            total_found=0,
            translation_confidence=0.0
        )


@app.get("/translate")
def translate_text(query: str, source_language: Optional[str] = None, target_language: str = "en"):
    """
    Simple translation endpoint for testing translation functionality.
    """
    try:
        result = translate_query(query, source_language, target_language)
        return {
            "original_text": query,
            "translated_text": result["translated_query"],
            "detected_language": result["detected_language"],
            "confidence": result.get("confidence", 0.0),
            "translation_needed": result.get("translation_needed", False)
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/ingest")
def trigger_ingest():
    ingest_to_qdrant()
    return {"status": "ingested"}


@app.get("/item/{item_id}")
def get_item(item_id: str):
    # fetch by qdrant payload id OR item_identifier
    # use scroll with filter on payload.id or payload.item_identifier
    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        flt = Filter(must=[
            FieldCondition(key="id", match=MatchValue(value=item_id))
        ])
        res = client.scroll(collection_name=COLLECTION_NAME,
                            scroll_filter=flt, limit=1)
        if res and res[0]:
            return res[0][0].payload
    except Exception:
        pass
    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        flt = Filter(must=[
            FieldCondition(key="item_identifier",
                           match=MatchValue(value=item_id))
        ])
        res = client.scroll(collection_name=COLLECTION_NAME,
                            scroll_filter=flt, limit=1)
        if res and res[0]:
            return res[0][0].payload
    except Exception:
        pass
    return {"error": "not found"}


@app.get("/db/items")
def get_db_items():
    rows = fetch_data()
    items = []
    for row in rows:
        estimator_id, room_name, item_name, item_id, amount, area, project_name, attributes, item_type_identifier, item_identifier, user_id, image = row
        parsed_attrs = parse_item_attributes(attributes)
        measurement = parsed_attrs.get("Measurement")
        measurement_sqft = measurement_to_sqft(measurement)
        items.append({
            "estimator_id": estimator_id,
            "room_name": room_name,
            "item_name": item_name,
            "id": item_id,
            "amount": amount,
            "area": area,
            "project_name": project_name,
            "attributes": attributes,
            "attributes_parsed": parsed_attrs,
            "measurement_sqft": measurement_sqft,
            "item_type_identifier": item_type_identifier,
            "item_identifier": item_identifier,
            "user_id": user_id,
            "image": image
        })
    # basic derived stats similar to ai-search-qg
    # compute min/max/avg by item_name+area
    stats = {}
    for it in items:
        key = (it["item_name"], it["area"]
               ) if it["item_name"] and it["area"] else None
        if not key:
            continue
        stats.setdefault(
            key, {"min": it["amount"], "max": it["amount"], "sum": it["amount"], "count": 1})
        s = stats[key]
        s["min"] = min(s["min"], it["amount"])
        s["max"] = max(s["max"], it["amount"])
        s["sum"] += it["amount"]
        s["count"] += 1
    stats_out = [
        {"item_name": k[0], "area": k[1], "Min_Amount": v["min"],
            "Max_Amount": v["max"], "Avg_Amount": (v["sum"] / max(1, v["count"]))}
        for k, v in stats.items()
    ]
    return {"items": items, "stats": stats_out}


# ---------------- Insert API Endpoints ----------------


@app.post("/insert/item", response_model=InsertResponse)
def insert_single_item(data: InsertItemRequest):
    """
    Insert a single item into the Qdrant database.
    Similar to the ingest process but for individual items.
    """
    try:
        # Create text for vector encoding
        text = f"{data.item_name} in {data.room_name} of {data.project_name}, located at {data.area}"
        vector = MODEL.encode(text).tolist()

        # Parse attributes if provided
        parsed_attrs = {}
        if data.attributes:
            parsed_attrs = parse_item_attributes(data.attributes)

        # Calculate measurement
        measurement = parsed_attrs.get("Measurement")
        measurement_sqft = measurement_to_sqft(measurement)

        # Calculate amount if missing/null using AmountCalculatorUtils
        amount_to_store = data.amount
        try:
            if not amount_to_store or amount_to_store == 0:
                type_identifier = None
                if isinstance(data.item_identifier, str):
                    if "WD" in data.item_identifier:
                        type_identifier = "WD"
                    elif "FC" in data.item_identifier:
                        type_identifier = "FC"
                    elif "ACS" in data.item_identifier:
                        type_identifier = "ACS"
                    elif "LF" in data.item_identifier:
                        type_identifier = "LF"
                    elif "OTH" in data.item_identifier:
                        type_identifier = "OTH"
                if type_identifier:
                    dummy_item = type(
                        "Item", (), {"attributes": data.attributes})()
                    calculated_amount = AmountCalculatorUtils.calc_item_amount(
                        type_identifier, dummy_item)
                    if calculated_amount and calculated_amount > 0:
                        amount_to_store = calculated_amount
        except Exception:
            # Swallow calculation errors and fall back to original amount
            pass

        # Prepare image data
        image_data = data.image
        if isinstance(image_data, dict) and "default" in image_data:
            image_data = image_data["default"]

        # Insert into Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": data.item_id,
                    "vector": vector,
                    "payload": {
                        "estimator_id": data.estimator_id,
                        "room_name": data.room_name,
                        "item_name": data.item_name,
                        "amount": amount_to_store,
                        "area": data.area,
                        "project_name": data.project_name,
                        "attributes": data.attributes,
                        "attributes_parsed": parsed_attrs,
                        "measurement_sqft": measurement_sqft,
                        "id": data.item_id,
                        "item_identifier": data.item_identifier,
                        "item_type_identifier": data.item_type_identifier,
                        "user_id": data.user_id,
                        "image": image_data
                    }
                }
            ]
        )

        return InsertResponse(
            status="success",
            inserted_count=1,
            success_ids=[data.item_id]
        )

    except Exception as e:
        return InsertResponse(
            status="error",
            inserted_count=0,
            errors=[str(e)]
        )


@app.post("/insert/items", response_model=InsertResponse)
def insert_multiple_items(data: InsertItemsRequest):
    """
    Insert multiple items into the Qdrant database.
    Processes items in batch for better performance.
    """
    success_ids = []
    errors = []

    for item_data in data.items:
        try:
            # Create text for vector encoding
            text = f"{item_data.item_name} in {item_data.room_name} of {item_data.project_name}, located at {item_data.area}"
            vector = MODEL.encode(text).tolist()

            # Parse attributes if provided
            parsed_attrs = {}
            if item_data.attributes:
                parsed_attrs = parse_item_attributes(item_data.attributes)

            # Calculate measurement
            measurement = parsed_attrs.get("Measurement")
            measurement_sqft = measurement_to_sqft(measurement)

            # Calculate amount if missing/null using AmountCalculatorUtils
            amount_to_store = item_data.amount
            try:
                if not amount_to_store or amount_to_store == 0:
                    item_type_identifier = item_data.item_type_identifier
                    if isinstance(item_data.item_identifier, str) and item_type_identifier:
                        dummy_item = type(
                            "Item", (), {"attributes": item_data.attributes})()
                        calculated_amount = AmountCalculatorUtils.calc_item_amount(
                            item_type_identifier, dummy_item)
                        if calculated_amount and calculated_amount > 0:
                            amount_to_store = calculated_amount
            except Exception:
                # Swallow calculation errors and fall back to original amount
                pass

            # Prepare image data
            image_data = item_data.image
            if isinstance(image_data, dict) and "default" in image_data:
                image_data = image_data["default"]

            # Insert into Qdrant
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    {
                        "id": item_data.item_id,
                        "vector": vector,
                        "payload": {
                            "estimator_id": item_data.estimator_id,
                            "room_name": item_data.room_name,
                            "item_name": item_data.item_name,
                            "amount": amount_to_store,
                            "area": item_data.area,
                            "project_name": item_data.project_name,
                            "attributes": item_data.attributes,
                            "attributes_parsed": parsed_attrs,
                            "measurement_sqft": measurement_sqft,
                            "id": item_data.item_id,
                            "item_identifier": item_data.item_identifier,
                            "item_type_identifier": item_data.item_type_identifier,
                            "user_id": item_data.user_id,
                            "image": image_data
                        }
                    }
                ]
            )

            success_ids.append(item_data.item_id)

        except Exception as e:
            errors.append(f"Item {item_data.item_id}: {str(e)}")

    return InsertResponse(
        status="success" if not errors else "partial_success",
        inserted_count=len(success_ids),
        errors=errors,
        success_ids=success_ids
    )


@app.post("/insert/batch", response_model=InsertResponse)
def insert_batch_items(data: InsertItemsRequest):
    """
    Insert multiple items into the Qdrant database using batch upsert for better performance.
    """
    try:
        points = []
        success_ids = []
        errors = []

        for item_data in data.items:
            try:
                # Create text for vector encoding
                text = f"{item_data.item_name} in {item_data.room_name} of {item_data.project_name}, located at {item_data.area}"
                vector = MODEL.encode(text).tolist()

                # Parse attributes if provided
                parsed_attrs = {}
                if item_data.attributes:
                    parsed_attrs = parse_item_attributes(item_data.attributes)

                # Calculate measurement
                measurement = parsed_attrs.get("Measurement")
                measurement_sqft = measurement_to_sqft(measurement)

                # Calculate amount if not provided
                amount_to_store = item_data.amount
                if not amount_to_store or amount_to_store == 0:
                    try:
                        type_identifier = None
                        if item_data.item_type_identifier:
                            type_identifier = item_data.item_type_identifier

                        if type_identifier and item_data.attributes:
                            dummy_item = type(
                                "Item", (), {"attributes": item_data.attributes})()
                            calculated_amount = AmountCalculatorUtils.calc_item_amount(
                                type_identifier, dummy_item)
                            if calculated_amount and calculated_amount > 0:
                                amount_to_store = calculated_amount
                    except Exception:
                        # Swallow calculation errors and fall back to original amount
                        pass

                # Prepare image data
                image_data = item_data.image
                if isinstance(image_data, dict) and "default" in image_data:
                    image_data = image_data["default"]

                # Prepare point for batch insert
                points.append({
                    "id": item_data.item_id,
                    "vector": vector,
                    "payload": {
                        "estimator_id": item_data.estimator_id,
                        "room_name": item_data.room_name,
                        "item_name": item_data.item_name,
                        "amount": amount_to_store,
                        "area": item_data.area,
                        "project_name": item_data.project_name,
                        "attributes": item_data.attributes,
                        "attributes_parsed": parsed_attrs,
                        "measurement_sqft": measurement_sqft,
                        "id": item_data.item_id,
                        "item_identifier": item_data.item_identifier,
                        "item_type_identifier": item_data.item_type_identifier,
                        "user_id": item_data.user_id,
                        "image": image_data
                    }
                })

                success_ids.append(item_data.item_id)

            except Exception as e:
                errors.append(f"Item {item_data.item_id}: {str(e)}")

        # Batch insert all points at once
        if points:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

        return InsertResponse(
            status="success" if not errors else "partial_success",
            inserted_count=len(success_ids),
            errors=errors,
            success_ids=success_ids
        )

    except Exception as e:
        return InsertResponse(
            status="error",
            inserted_count=0,
            errors=[str(e)]
        )


@app.delete("/delete/item/{item_id}")
def delete_item(item_id: int):
    """
    Delete an item from the Qdrant database by item ID.
    """
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[item_id]
        )
        return {"status": "success", "message": f"Item {item_id} deleted successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.put("/update/item/{item_id}")
def update_item(item_id: int, data: InsertItemRequest):
    """
    Update an existing item in the Qdrant database.
    This will upsert the item with the new data.
    """
    try:
        # Create text for vector encoding
        text = f"{data.item_name} in {data.room_name} of {data.project_name}, located at {data.area}"
        vector = MODEL.encode(text).tolist()

        # Parse attributes if provided
        parsed_attrs = {}
        if data.attributes:
            parsed_attrs = parse_item_attributes(data.attributes)

        # Calculate measurement
        measurement = parsed_attrs.get("Measurement")
        measurement_sqft = measurement_to_sqft(measurement)

        # Calculate amount if missing/null using AmountCalculatorUtils
        amount_to_store = data.amount
        try:
            if not amount_to_store or amount_to_store == 0:
                item_type_identifier = data.item_type_identifier
                if item_type_identifier:
                    dummy_item = type(
                        "Item", (), {"attributes": data.attributes})()
                    calculated_amount = AmountCalculatorUtils.calc_item_amount(
                        item_type_identifier, dummy_item)
                    if calculated_amount and calculated_amount > 0:
                        amount_to_store = calculated_amount
        except Exception:
            # Swallow calculation errors and fall back to original amount
            pass

        # Prepare image data
        image_data = data.image
        if isinstance(image_data, dict) and "default" in image_data:
            image_data = image_data["default"]

        # Update in Qdrant (upsert will update if exists, insert if not)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": item_id,
                    "vector": vector,
                    "payload": {
                        "estimator_id": data.estimator_id,
                        "room_name": data.room_name,
                        "item_name": data.item_name,
                        "amount": amount_to_store,
                        "area": data.area,
                        "project_name": data.project_name,
                        "attributes": data.attributes,
                        "attributes_parsed": parsed_attrs,
                        "measurement_sqft": measurement_sqft,
                        "id": item_id,
                        "item_identifier": data.item_identifier,
                        "item_type_identifier": data.item_type_identifier,
                        "user_id": data.user_id,
                        "image": image_data
                    }
                }
            ]
        )

        return {"status": "success", "message": f"Item {item_id} updated successfully"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
