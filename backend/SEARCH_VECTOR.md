## `/search/vector` API – Multilingual Item Extraction & Vector Search

This document describes how the backend’s `/search/vector` endpoint works, with a focus on **item extraction using `extractor.py`** and how that influences Qdrant vector search.

---

### 1. Endpoint overview

- **Route**: `POST /search/vector`
- **Handler**: `search_with_vector_similarity` in `app/main.py`
- **Request model**: `NLPSearchQuery`

```json
{
  "query": "user free‑text query",
  "top_k": 10000,
  "use_nlp": false,
  "is_voice": false,
  "page": 1,
  "page_size": 20
}
```

- **Response (simplified)**:

```json
{
  "query": "original query",
  "extracted_item_name": "canonical item (if any)",
  "results": [ /* paginated items */ ],
  "total_found": 123,
  "page": 1,
  "page_size": 20,
  "total_pages": 7,
  "search_type": "vector_search_with_item_extractor"
}
```

---

### 2. Item catalogue and synonyms (`items.json`)

- File: `backend/app/items.json`
- Structure: a dict of **canonical item names** → **per‑language forms**:

```json
{
  "storage cabinet": {
    "hindi":   { "native": "स्टोरेज कैबिनेट, भंडारण अलमारी", "roman": "storej kainabinet, bhandaran almari" },
    "bengali": { "native": "স্টোরেজ ক্যাবিনেট",                "roman": "storej kabinet" },
    ...
  },
  "tv wall unit": {
    "hindi":   { "native": "टीवी वॉल यूनिट", "roman": "ti vi wol unit" },
    ...
  },
  ...
}
```

Each entry:
- The **key** (e.g. `"storage cabinet"`, `"tv wall unit"`) is the **canonical `item_name`** that will be used in search.
- For each language:
  - `native` holds native‑script variants (may be comma‑separated).
  - `roman` holds romanized variants (may also be comma‑separated).

---

### 3. Synonym index & extractor (`app/extractor.py`)

File: `backend/app/extractor.py`

#### 3.1 Normalization

`normalize(text: str) -> str`:
- NFKC Unicode normalization
- `lower()` case
- removes zero‑width chars
- keeps only letters/digits/whitespace
- collapses multiple spaces → single space

Example:
- `"  स्टोरेज  यूनिट  "` → `"स्टोरेज यूनिट"`

#### 3.2 Building the synonym index

`build_synonym_index(item_json: dict) -> dict`:

For each canonical item (e.g. `"storage cabinet"`):
- Add **canonical phrase** and its tokens:
  - `"storage cabinet"` → `"storage cabinet"`, `"storage"`, `"cabinet"`.
- For each `native` and `roman` value:
  - Split on commas into variants.
  - For each variant:
    - Normalize full phrase and index it.
    - Split into tokens and index each token.
  - First writer wins: if a key already exists in the index, it is **not overwritten** by later items. This prevents later entries (e.g. city names) from hijacking generic tokens like `"स्टोरेज"` or `"tv"`.

The result:

```python
synonym_index: Dict[str, str]
```

mapping **normalized synonym/token → canonical item name**.

#### 3.3 Item extraction

`extract_item(query: str, synonym_index: dict) -> Optional[str]`:

1. Normalize `query` → `query_norm`.
2. Split into tokens.
3. Consider all n‑grams from length 1 up to `max(10, token_count)`.
4. For each n‑gram:
   - If the n‑gram string exists in `synonym_index`, and its length (n) is greater than the best so far, remember this canonical item.
5. Return the **longest‑matching n‑gram’s canonical item**, or `None` if no match.

Examples (conceptual):
- `"दीवार इकाई"` → `"wall unit"`
- `"mancha"` → `"bed, cot"`
- `"tv"` → `"tv wall unit"`
- `"स्टोरेज"` → some storage‑related canonical (e.g. `"storage unit"` / `"storage cabinet"`) depending on the first mapping in `items.json`.

---

### 4. Query → item extraction logic (`extract_item_name_from_query`)

File: `backend/app/main.py`, function `extract_item_name_from_query(query: str)`.

Steps:

1. **Load synonym index** (cached) from `items.json` using `build_synonym_index`.
2. **Primary extraction**: call `extract_item(query, synonym_index)`.
3. **Fuzzy spelling correction** (if primary returns `None`):
   - Normalize the query as extractor does.
   - Split into tokens.
   - For each token, use `difflib.get_close_matches` against all index keys (tokens/phrases).
   - If a close match (cutoff ~0.8) is found, replace the token with that match.
   - Re‑join tokens and call `extract_item` again with the corrected query.
   - This helps with minor misspellings like:
     - `"bichana"` → `"bichhana"` → `"bed"`
3. **Validation against `items.json`**:
   - For any candidate canonical `item_name`, load its entry from `items.json`.
   - Check whether the **original query substring** appears in any `native` or `roman` form for that item.
   - If yes, trust this candidate.
4. **Fallback scan**:
   - If the candidate doesn’t clearly match, scan all items in `items.json` and pick the **first canonical item whose `native` or `roman` text contains the original query**.
5. **Last resort**:
   - If all else fails, return whatever the extractor produced (which may be `None`).

The final result is what the API returns as `extracted_item_name`.

---

### 5. Vector search in Qdrant and item‑aware ranking

Still in `search_with_vector_similarity`:

1. **Embedding**:
   - Compute `query_vector = MODEL.encode(query_text).tolist()` using `SentenceTransformer('all-MiniLM-L6-v2')`.

2. **Qdrant search (no item_name filter)**:
   - Call whichever client method is available: `.search`, `.search_points`, or `.query_points`.
   - Always search the entire collection (no payload filter), to avoid losing results when item names don’t exactly match.

3. **Format raw results**:
   - For each hit, build a result dict with:
     - `id`, `area`, `project_name`, `room_name`, `item_identifier`, `item_type_identifier`, `description`, `item_name`, `attributes_parsed`, `image`, `amount`, and `score` (vector similarity).

4. **Item‑aware filtering and re‑ranking (using `extracted_item_name`)**:

If `extracted_item_name` is present:

- **Split canonical name on commas**:
  - e.g. `"bed, cot"` → `["bed", "cot"]`.
- Derive **core tokens** from those phrases:
  - `"tv wall unit"` → tokens `["tv", "wall", "unit"]`, then drop generic ones like `"unit"`, `"wall"`, etc., keep core tokens like `"tv"`, `"bed"`, `"cot"`.
- **Filtering**:
  - Keep only items where:
    - `item_name` contains any full phrase (e.g. `"tv wall unit"`), **or**
    - `item_name` contains any core token (e.g. `"tv"`, `"bed"`).
- **Re‑ranking with boosted score**:
  - Start from the original similarity score.
  - For each phrase:
    - Exact match (`item_name == phrase`) → strong boost (e.g. +2.0).
    - Word‑boundary / prefix match (e.g. `"bed side table"`, `"tv area wall panelling"`) → medium boost.
    - Any substring match (`"storage for bed"`) → small boost.
  - Extra small boost if any core token appears (e.g. `"tv"` in `"bedroom tv unit base"`).
  - Sort results by `(similarity + boost)` descending.

Effect:

- Queries like `"mancha"` (→ `"bed, cot"`) will show:
  - `"bed"` and `"cot"` first,
  - then related items like `"bed side table"`, etc.
- Queries like `"tv"` (→ `"tv wall unit"`) will show:
  - `"tv wall unit"` first,
  - then `"tv area wall panelling"`, `"tv base unit"`, `"bedroom tv unit base"`, etc.

5. **Area‑wise averages and pagination**:
   - Compute area‑wise average prices via `calculate_area_wise_averages`.
   - Attach `average_price` per `(area, item_name)` group.
   - Apply pagination (`page`, `page_size`) before returning.

---

### 6. Summary

The `/search/vector` API is **item‑aware semantic search**:

- **Multilingual & fuzzy**:
  - Handles Hindi/Bengali/Tamil/Telugu/Kannada/Malayalam native and romanized forms via `items.json`.
  - Tolerates small spelling errors via fuzzy token correction.
- **Canonical item extraction**:
  - Maps arbitrary text to a canonical `item_name` using `extractor.py` and `items.json`.
- **Vector + payload‑aware ranking**:
  - Uses dense embeddings for semantic similarity (SentenceTransformer + Qdrant).
  - Then prunes and re‑ranks results so items whose `item_name` matches the extracted canonical concept (and its core tokens) appear first.

This design lets you plug in rich, hand‑curated item synonym data (`items.json`) while still leveraging a generic vector search backend for robust retrieval.

