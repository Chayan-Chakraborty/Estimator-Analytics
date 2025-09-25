from .amount_calculator import AmountCalculatorUtils
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import sys
import os
import re
import difflib
from difflib import SequenceMatcher
from collections import defaultdict

from app.qdrant_client_helper import client, COLLECTION_NAME
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from app.ingest import ingest_to_qdrant, fetch_data, parse_item_attributes, measurement_to_sqft
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

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

# ---------------- Helper ----------------


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


@app.post("/search/vector")
def search_with_vector_similarity(data: NLPSearchQuery):
    """
    Keyword-first search with vector similarity ranking.
    First extracts keywords from query, then searches for items containing those keywords,
    then ranks by vector similarity.
    """
    try:
        # Extract keywords from query first
        query_lower = data.query.lower()
        extracted_keywords = []
        
        # Smart keyword extraction - filter out stop words
        import re
        
        # Common stop words to ignore
        stop_words = {
            'with', 'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'by', 'from',
            'material', 'finish', 'color', 'size', 'type', 'style', 'design', 'model', 'brand',
            'room', 'area', 'location', 'city', 'price', 'cost', 'amount', 'value', 'rate'
        }
        
        # Split by common words and punctuation
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Filter out stop words and short words
        extracted_keywords = [
            word for word in words 
            if len(word) > 2 and word not in stop_words
        ]
        
        # Debug: print extracted keywords
        print(f"Debug: Query: {data.query}")
        print(f"Debug: All words: {words}")
        print(f"Debug: Extracted keywords: {extracted_keywords}")
        
        # Get all items from the collection for keyword filtering
        all_items = []
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True
        )
        
        for point in scroll_result[0]:
            try:
                item = point.payload.copy()
                # Calculate amount if missing
                current_amount = item.get('amount')
                if not current_amount or current_amount == 0:
                    calculated_amount = calculate_amount_from_attributes(item)
                    if calculated_amount > 0:
                        item['amount'] = calculated_amount
                all_items.append(item)
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        # Filter items that contain any of the extracted keywords
        filtered_items = []
        for item in all_items:
            # Check if item contains any of the keywords
            item_text = ""
            
            # Combine all searchable text from the item
            item_text += f" {item.get('item_name', '')}"
            item_text += f" {item.get('room_name', '')}"
            item_text += f" {item.get('area', '')}"
            item_text += f" {item.get('project_name', '')}"
            
            # Add attributes
            attrs_parsed = item.get('attributes_parsed', {})
            if isinstance(attrs_parsed, dict):
                for key, value in attrs_parsed.items():
                    item_text += f" {value}"
            
            item_text = item_text.lower()
            
            # Check if ALL keywords match
            keyword_matches = 0
            for keyword in extracted_keywords:
                if keyword in item_text:
                    keyword_matches += 1
            
            # Only include items that have ALL keywords matching
            if keyword_matches == len(extracted_keywords) and len(extracted_keywords) > 0:
                item['keyword_matches'] = keyword_matches
                filtered_items.append(item)
        
        # Now rank by vector similarity
        if filtered_items:
            vector = MODEL.encode(data.query).tolist()
            
            # Calculate similarity scores for filtered items
            scored_items = []
            for item in filtered_items:
                try:
                    # Create a text representation for vector encoding
                    item_text = f"{item.get('item_name', '')} {item.get('room_name', '')} {item.get('area', '')}"
                    attrs_parsed = item.get('attributes_parsed', {})
                    if isinstance(attrs_parsed, dict):
                        for key, value in attrs_parsed.items():
                            item_text += f" {value}"
                    
                    # Encode the item text
                    item_vector = MODEL.encode(item_text).tolist()
                    
                    # Calculate cosine similarity
                    import numpy as np
                    similarity = np.dot(vector, item_vector) / (np.linalg.norm(vector) * np.linalg.norm(item_vector))
                    
                    item['similarity_score'] = round(float(similarity), 4)
                    scored_items.append(item)
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    continue
            
            # Sort by similarity score (highest first)
            scored_items.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            items = scored_items
        else:
            items = []
        
        return {
            "results": items,
            "query": data.query,
            "total_found": len(items),
            "search_type": "keyword_first_vector_similarity",
            "extracted_keywords": extracted_keywords
        }
    except Exception as e:
        print(f"Error in vector search: {e}")
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
