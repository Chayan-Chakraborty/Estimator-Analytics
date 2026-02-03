
import unicodedata
import re

def normalize(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # remove zero-width chars (Bengali / Indic safe)
    text = text.replace("\u200c", "").replace("\u200d", "")

    text = "".join(
        ch if ch.isalnum() or ch.isspace() else " "
        for ch in text
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_variants(text: str):
    if not text:
        return []

    text = text.replace("，", ",").replace("、", ",")
    return [t.strip() for t in text.split(",") if t.strip()]

def build_synonym_index(item_json: dict):
    synonym_index = {}

    for item, langs in item_json.items():
        # Index the canonical item name itself and its tokens
        item_norm = normalize(item)
        if item_norm:
            if item_norm not in synonym_index:
                synonym_index[item_norm] = item
            for tok in item_norm.split():
                if tok:
                    if tok not in synonym_index:
                        synonym_index[tok] = item

        for forms in langs.values():
            for val in split_variants(forms.get("native", "")):
                phrase_norm = normalize(val)
                if phrase_norm:
                    # Full phrase
                    if phrase_norm not in synonym_index:
                        synonym_index[phrase_norm] = item
                    # Also index individual tokens so queries like "स्टोरेज"
                    # can match "स्टोरेज कैबिनेट"
                    for tok in phrase_norm.split():
                        if tok:
                            if tok not in synonym_index:
                                synonym_index[tok] = item
            for val in split_variants(forms.get("roman", "")):
                phrase_norm = normalize(val)
                if phrase_norm:
                    if phrase_norm not in synonym_index:
                        synonym_index[phrase_norm] = item
                    for tok in phrase_norm.split():
                        if tok:
                            if tok not in synonym_index:
                                synonym_index[tok] = item

    return synonym_index

def extract_item(query: str, synonym_index: dict):
    query_norm = normalize(query)
    tokens = query_norm.split()
    token_count = len(tokens)
    print(f"Token count: {token_count}")
    max_n=max(10, token_count)
    best_item = None
    best_len = 0

    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i+n])
            if gram in synonym_index and n > best_len:
                best_item = synonym_index[gram]
                best_len = n

    return best_item

