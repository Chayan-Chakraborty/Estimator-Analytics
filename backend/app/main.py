from .amount_calculator import AmountCalculatorUtils
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
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
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from app.ingest import ingest_to_qdrant, fetch_data, parse_item_attributes, measurement_to_sqft
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchValue, Range
from app.speech_processor import process_speech_to_english_query, process_text_to_english_query
from app.ai_layer import intelligent_translate
from transformers import pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SK_STOP
from qdrant_client import QdrantClient

zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

# ---------------- NLP-based filtering helpers ----------------


def normalize_text(text):
    if not text:
        return ""
    return re.sub(r'[^\w\s]', '', str(text).lower().strip())


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
    measurement_min: Optional[float] = None
    measurement_max: Optional[float] = None
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
        print(f"Debug: Translation needed: {detected.lower() != target_lang.lower()}")

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
        item_id = item.get('item_identifier') or ''
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
        content = await audio.read()
        english_text = process_speech_to_english_query(content)
        print(f"Debug: English text: {english_text}")
        return {"english_query": english_text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Speech processing failed: {e}")


@app.post("/speech/text-to-english")
async def text_to_english(payload: Dict[str, str]):
    """
    Intelligent text translation: detect language, handle romanized Indic by
    trying multiple scripts, translate to fluent English, and keep meaning/tone.
    """
    txt = payload.get("text", "") if payload else ""
    src = payload.get("source_language") if payload else None
    if not txt:
        return {"english_query": ""}
    try:
        english_text, detected_src = intelligent_translate(txt, src)
        print(f"Debug: English text: {english_text}")
        return {"english_query": english_text, "detected_language": detected_src}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text processing failed: {e}")


@app.post("/search/nlp")
def search_with_nlp(data: NLPSearchQuery):
    try:
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
            return {"results": all_items, "extracted_filters": {}}
        extracted_filters = {}
        item_name = data.query.strip()
        if len(item_name) > 2:
            extracted_filters['item_name'] = item_name
        city = extract_city_from_text(data.query)
        if city:
            extracted_filters['city'] = city
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
        filtered_items = all_items
        if 'item_name' in extracted_filters:
            filtered_items = find_similar_items(
                extracted_filters['item_name'],
                filtered_items,
                'item_name',
                threshold=0.5
            )
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
            "total_found": len(filtered_items)
        }
    except Exception as e:
        print(f"Error in NLP search: {e}")
        return {"results": [], "extracted_filters": {}, "error": str(e)}


# @app.post("/search/vector")
# def search_with_vector_similarity(data: NLPSearchQuery):
#     """
#     Keyword-first search with vector similarity ranking.
#     First extracts keywords from query, then searches for items containing those keywords,
#     then ranks by vector similarity.
#     """
#     try:
#         # Extract keywords from query first
#         query_lower = data.query.lower()
#         extracted_keywords = []
        
#         # Smart keyword extraction - filter out stop words
#         import re
#         from difflib import SequenceMatcher
#         from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SK_STOP
        
#         # Common stop words to ignore
#         # Start with sklearn's general English stopwords, then add domain-generic words that
#         # we don't want to influence matching regardless of language
#         stop_words = set(SK_STOP)
#         stop_words.update({'material', 'finish', 'color', 'size', 'type', 'style', 'design', 'model', 'brand',
#                            'room', 'area', 'location', 'city', 'price', 'cost', 'amount', 'value', 'rate'})

#         # Remove short romanized filler words heuristically: words of length <= 3 that are very common
#         words = re.findall(r'\b\w+\b', query_lower)
#         common_short_fillers = {w for w in words if len(w) <= 3}
#         stop_words.update(common_short_fillers)
        
#         # Split by common words and punctuation
#         # Filter out stop words and short words
#         extracted_keywords = [
#             word for word in words 
#             if len(word) > 2 and word not in stop_words
#         ]

#         # Corrections map for common misspellings
#         corrections = {
#             'blockboaard': 'blockboard',
#             'blockbord': 'blockboard',
#             'consol': 'console'
#         }
#         extracted_keywords = [corrections.get(w, w) for w in extracted_keywords]
        
#         # Debug: print extracted keywords
#         print(f"Debug: Query: {data.query}")
#         print(f"Debug: All words: {words}")
#         print(f"Debug: Extracted keywords: {extracted_keywords}")
        
#         # Get all items from the collection for keyword filtering
#         all_items = []
#         scroll_result = client.scroll(
#             collection_name=COLLECTION_NAME,
#             limit=10000,
#             with_payload=True
#         )
        
#         for point in scroll_result[0]:
#             try:
#                 item = point.payload.copy()
#                 # Calculate amount if missing
#                 current_amount = item.get('amount')
#                 if not current_amount or current_amount == 0:
#                     calculated_amount = calculate_amount_from_attributes(item)
#                     if calculated_amount > 0:
#                         item['amount'] = calculated_amount
#                 # Build a consolidated searchable text field once
#                 item_text = ""
#                 item_text += f" {item.get('item_name', '')}"
#                 item_text += f" {item.get('room_name', '')}"
#                 item_text += f" {item.get('area', '')}"
#                 item_text += f" {item.get('project_name', '')}"
#                 attrs_parsed = item.get('attributes_parsed', {})
#                 if isinstance(attrs_parsed, dict):
#                     for _k, _v in attrs_parsed.items():
#                         item_text += f" {_v}"
#                 item['__search_text__'] = item_text.lower()
#                 all_items.append(item)
#             except Exception as e:
#                 print(f"Error processing item: {e}")
#                 continue
        
#         # Helper: normalize ascii-only string
#         def norm(s: str) -> str:
#             s = (s or '').lower()
#             return re.sub(r'[^a-z0-9 ]+', ' ', s)

#         # Fuzzy contains: substring or SequenceMatcher ratio >= 0.82 against any token
#         def fuzzy_contains(haystack: str, needle: str) -> bool:
#             h = norm(haystack)
#             n = norm(needle)
#             if not n:
#                 return False
#             if n in h:
#                 return True
#             tokens = h.split()
#             for t in tokens:
#                 if len(t) < 3:
#                     continue
#                 if SequenceMatcher(None, t, n).ratio() >= 0.82:
#                     return True
#             return False

#         # Narrow keywords to those that appear in at least one item (avoid forcing filler words)
#         def appears_in_corpus(kw: str) -> bool:
#             for it in all_items:
#                 if fuzzy_contains(it.get('__search_text__', ''), kw):
#                     return True
#             return False

#         effective_keywords = [kw for kw in extracted_keywords if appears_in_corpus(kw)]
#         # If nothing left, fall back to original keywords (so we don't empty the query)
#         if not effective_keywords:
#             effective_keywords = extracted_keywords

#         # Material-awareness: if the query mentions a specific material (e.g., blockboard),
#         # only keep items whose parsed Material matches that material and exclude others
#         material_aliases = MATERIAL_ALIASES
#         # detect requested material from keywords
#         requested_material = None
#         for kw in effective_keywords:
#             for mat_key, aliases in material_aliases.items():
#                 if kw in aliases or kw == mat_key:
#                     requested_material = mat_key
#                     break
#             if requested_material:
#                 break

#         # Filter items that contain ALL of the effective keywords (with fuzzy matching)
#         filtered_items = []
#         for item in all_items:
#             # Check if item contains any of the keywords
#             item_text = ""
            
#             # Combine all searchable text from the item
#             item_text += f" {item.get('item_name', '')}"
#             item_text += f" {item.get('room_name', '')}"
#             item_text += f" {item.get('area', '')}"
#             item_text += f" {item.get('project_name', '')}"
            
#             # Add attributes
#             attrs_parsed = item.get('attributes_parsed', {})
#             if isinstance(attrs_parsed, dict):
#                 for key, value in attrs_parsed.items():
#                     item_text += f" {value}"
            
#             item_text = item_text.lower()
            
#             # If material is requested, enforce exact material family match
#             if requested_material:
#                 mat_val = ''
#                 attrs_parsed = item.get('attributes_parsed', {})
#                 if isinstance(attrs_parsed, dict):
#                     mat_val = str(attrs_parsed.get('Material') or '')
#                 mat_norm = norm(mat_val)
#                 allowed_aliases = material_aliases.get(requested_material, {requested_material})
#                 # ensure one of the aliases is present and competing materials are not
#                 if not any(norm(a) in mat_norm for a in allowed_aliases):
#                     continue

#             # Check if ALL keywords match
#             keyword_matches = 0
#             for keyword in effective_keywords:
#                 if fuzzy_contains(item_text, keyword):
#                     keyword_matches += 1
            
#             # Only include items that have ALL keywords matching
#             if keyword_matches == len(effective_keywords) and len(effective_keywords) > 0:
#                 item['keyword_matches'] = keyword_matches
#                 filtered_items.append(item)
        
#         # Now rank by vector similarity
#         if filtered_items:
#             vector = MODEL.encode(data.query).tolist()
            
#             # Calculate similarity scores for filtered items
#             scored_items = []
#             for item in filtered_items:
#                 try:
#                     # Create a text representation for vector encoding
#                     item_text = f"{item.get('item_name', '')} {item.get('room_name', '')} {item.get('area', '')}"
#                     attrs_parsed = item.get('attributes_parsed', {})
#                     if isinstance(attrs_parsed, dict):
#                         for key, value in attrs_parsed.items():
#                             item_text += f" {value}"
                    
#                     # Encode the item text
#                     item_vector = MODEL.encode(item_text).tolist()
                    
#                     # Calculate cosine similarity
#                     import numpy as np
#                     similarity = np.dot(vector, item_vector) / (np.linalg.norm(vector) * np.linalg.norm(item_vector))
                    
#                     item['similarity_score'] = round(float(similarity), 4)
#                     scored_items.append(item)
#                 except Exception as e:
#                     print(f"Error calculating similarity: {e}")
#                     continue
            
#             # Sort by similarity score (highest first)
#             scored_items.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
#             items = scored_items
#         else:
#             items = []
        
#         return {
#             "results": items,
#             "query": data.query,
#             "total_found": len(items),
#             "search_type": "keyword_first_vector_similarity",
#             "extracted_keywords": effective_keywords
#         }
#     except Exception as e:
#         print(f"Error in vector search: {e}")
#         return {"results": [], "error": str(e), "total_found": 0}

from keybert import KeyBERT

class NLPSearchQuery(BaseModel):
    query: str
    top_k: int = 10


# Embedding and keyword extraction models (must match Qdrant collection dim=384)
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
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
    top_k: Optional[int] = 10


def execute_vector_plan(plan: "VectorPlan") -> Dict[str, Any]:
    try:
        embed_text = (plan.embedding_text or "").strip()
        if not embed_text:
            return {"results": [], "total_found": 0, "error": "empty embedding_text"}

        query_vector = MODEL.encode(embed_text).tolist()

        q_filter = None
        if plan.keywords:
            try:
                q_filter = Filter(must=[FieldCondition(key="keywords", match=MatchAny(any=plan.keywords))])
            except Exception:
                q_filter = None

        raw = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=max(1, min(50, getattr(plan, 'top_k', 10) or 10)),
            query_filter=q_filter
        )

        items: List[Dict[str, Any]] = []
        strict_keywords = [k.strip().lower() for k in (plan.keywords or []) if k and k.strip()]
        for r in raw:
            try:
                item = r.payload.copy()
                item['id'] = r.id
                item['similarity_score'] = float(getattr(r, 'score', 0.0))
                if not item.get('amount') or item.get('amount') == 0:
                    calc = calculate_amount_from_attributes(item)
                    if calc > 0:
                        item['amount'] = calc
                # Strict match: all extracted keywords must be present in item keywords
                if strict_keywords:
                    item_kws = set(str(k).lower() for k in (item.get('keywords') or []))
                    if not all(k in item_kws for k in strict_keywords):
                        continue
                items.append(item)
            except Exception as ex:
                print("Vector result formatting error:", ex)
                continue
        
        return {"results": items, "total_found": len(items)}
    except Exception as e:
        return {"results": [], "total_found": 0, "error": str(e)}

@app.post("/search/vector")
def search_with_vector_similarity(data: NLPSearchQuery) -> Dict[str, Any]:
    """
    Optimized semantic search using Qdrant vectors with optional keyword pre-filter.
    - Extract top keywords from the query (KeyBERT over the same embedding model)
    - Build a Qdrant payload filter on `keywords` if any exist
    - Perform vector search using 384‑dim embeddings
    - Post-process payloads to ensure computed fields (e.g., amount) are present
    """
    try:
        query_text = (data.query or "").strip()
        if not query_text:
            return {"results": [], "query": data.query, "total_found": 0}

        # 1) Build a local plan (can be replaced by external AI planner)
        try:
            kw_pairs = KW_MODEL.extract_keywords(query_text, keyphrase_ngram_range=(1, 2), stop_words='english')
            attr_blacklist = {
                "rate", "rate per sqft", "quantity", "price/unit", "price",
                "measurement", "material", "finish", "description", "brand"
            }
            # Build ordered keyword list: include cleansed bigrams (if both tokens allowed)
            # and include individual allowed tokens. Preserve order of appearance.
            ordered: list[str] = []
            seen: set[str] = set()

            def add_term(term: str):
                t = term.strip().lower()
                if not t or t in seen:
                    return
                seen.add(t)
                ordered.append(t)

            for kw, _score in kw_pairs:
                if not isinstance(kw, str) or not kw.strip():
                    continue
                toks = re.findall(r"\b\w+\b", kw.lower())
                toks = [t for t in toks if t not in attr_blacklist]
                if len(toks) >= 2:
                    bigram = " ".join(toks[:2])
                    add_term(bigram)
                for t in toks:
                    add_term(t)

            keywords = ordered
        except Exception:
            keywords = []

        plan = VectorPlan(embedding_text=query_text, keywords=keywords, top_k=getattr(data, 'top_k', 10))

        # 2) Execute plan via shared executor
        exec_res = execute_vector_plan(plan)

        return {
            "results": exec_res.get("results", []),
            "query": data.query,
            "total_found": exec_res.get("total_found", 0),
            "search_type": "vector_similarity_with_keywords",
            "extracted_keywords": keywords
        }
    except Exception as e:
        return {"results": [], "error": str(e), "total_found": 0}


@app.post("/search/vector/prompt")
def get_vector_search_prompt(data: NLPSearchQuery) -> Dict[str, Any]:
    """Builds and returns a prebuilt AI prompt to plan the retrieval for a given query."""
    q = (data.query or "").strip()
    prompt = build_retrieval_prompt(q, top_k=getattr(data, 'top_k', 10) or 10)
    # Also provide baseline keywords we computed locally (optional)
    try:
        kw_pairs = KW_MODEL.extract_keywords(q, keyphrase_ngram_range=(1, 2), stop_words='english')
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
                q_filter = Filter(must=[FieldCondition(key="keywords", match=MatchAny(any=plan.keywords))])
            except Exception:
                q_filter = None

        raw = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=max(1, min(50, getattr(plan, 'top_k', 10) or 10)),
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
    merged_stats = defaultdict(lambda: {"area": "", "measurement_sqft": None, "amounts": [], "count": 0})

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
        measurement_key = round(measurement, 2) if measurement is not None else None
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
        valid_amounts = [a for a in g["amounts"] if isinstance(a, (int, float))]
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
        measurement_min = data_dict.get("measurement_min")
        measurement_max = data_dict.get("measurement_max")
        if measurement_min is not None or measurement_max is not None:
            r = {}
            if measurement_min is not None:
                r["gte"] = measurement_min
            if measurement_max is not None:
                r["lte"] = measurement_max
            filters.append(FieldCondition(
                key="measurement_sqft", range=Range(**r)))
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
        merged_stats = defaultdict(lambda: {"area": "", "measurement_sqft": None, "amounts": [], "count": 0})
        
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
            measurement_key = round(measurement, 2) if measurement is not None else None
            group_key = f"{cluster_rep}_{measurement_key}"

            ms = merged_stats[group_key]
            if not ms["area"]:
                ms["area"] = cluster_rep
                ms["measurement_sqft"] = measurement_key
            ms["amounts"].append(amount)
            ms["count"] += 1
    else:
        # Group by city only when no measurement filter
        merged_stats = defaultdict(lambda: {"area": "", "amounts": [], "count": 0})
        
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
        valid_amounts = [a for a in g["amounts"] if isinstance(a, (int, float))]
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
        # Translate the query to English
        translation_result = translate_query(
            query=data.query,
            source_lang=data.source_language,
            target_lang=data.target_language
        )
        
        translated_query = translation_result["translated_query"]
        detected_language = translation_result["detected_language"]
        confidence = translation_result.get("confidence", 0.0)
        
        # Use the existing NLP search logic with translated query
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
        vector_data = NLPSearchQuery(query=translated_query, use_nlp=data.use_nlp)
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
        estimator_id, room_name, item_name, item_id, amount, area, project_name, attributes, item_identifier, user_id, image = row
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
                    dummy_item = type("Item", (), {"attributes": data.attributes})()
                    calculated_amount = AmountCalculatorUtils.calc_item_amount(type_identifier, dummy_item)
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
                    type_identifier = None
                    if isinstance(item_data.item_identifier, str):
                        if "WD" in item_data.item_identifier:
                            type_identifier = "WD"
                        elif "FC" in item_data.item_identifier:
                            type_identifier = "FC"
                        elif "ACS" in item_data.item_identifier:
                            type_identifier = "ACS"
                        elif "LF" in item_data.item_identifier:
                            type_identifier = "LF"
                        elif "OTH" in item_data.item_identifier:
                            type_identifier = "OTH"
                    if type_identifier:
                        dummy_item = type("Item", (), {"attributes": item_data.attributes})()
                        calculated_amount = AmountCalculatorUtils.calc_item_amount(type_identifier, dummy_item)
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
                        if item_data.item_identifier:
                            if "WD" in item_data.item_identifier:
                                type_identifier = "WD"
                            elif "FC" in item_data.item_identifier:
                                type_identifier = "FC"
                            elif "ACS" in item_data.item_identifier:
                                type_identifier = "ACS"
                            elif "LF" in item_data.item_identifier:
                                type_identifier = "LF"
                            elif "OTH" in item_data.item_identifier:
                                type_identifier = "OTH"
                        
                        if type_identifier and item_data.attributes:
                            dummy_item = type("Item", (), {"attributes": item_data.attributes})()
                            calculated_amount = AmountCalculatorUtils.calc_item_amount(type_identifier, dummy_item)
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
                    dummy_item = type("Item", (), {"attributes": data.attributes})()
                    calculated_amount = AmountCalculatorUtils.calc_item_amount(type_identifier, dummy_item)
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
                        "user_id": data.user_id,
                        "image": image_data
                    }
                }
            ]
        )
        
        return {"status": "success", "message": f"Item {item_id} updated successfully"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
