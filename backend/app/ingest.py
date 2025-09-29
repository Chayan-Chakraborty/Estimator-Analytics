from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from app.qdrant_client_helper import client, create_collection, COLLECTION_NAME
from app.amount_calculator import AmountCalculatorUtils
import mysql.connector
import os
from uuid import uuid4
import json
import re

MODEL = SentenceTransformer('all-MiniLM-L6-v2')
KW_MODEL = KeyBERT(model=MODEL)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_NAME = os.getenv("DB_NAME", "vishanti")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "password")

def fetch_data():
    conn = mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()
    query = """SELECT e.id AS estimator_id, r.name AS room_name, i.name AS item_name, i.id, i.amount,
           a.city AS area, p.name AS project_name, i.attributes, i.type_identifier as item_identifier,
           e.user_id, i.image
    FROM vishanti.item i
    JOIN vishanti.room r ON r.id = i.room_id
    JOIN vishanti.estimator e ON e.id = r.estimator_id
    JOIN zeus.project p ON e.project_id = p.id
    JOIN zeus.address a ON a.project_id = p.id
    LEFT JOIN shield.internal_users iu on e.user_id = iu.user_id
    WHERE iu.user_id IS NULL;"""
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# ---- Attribute parsing similar to ai-search-qg.py ----
ATTRIBUTE_MAP = {
    "RPSF_LBR_WD_ATTR": "Rate per Sqft",
    "RAT_WD_ATTR": "Rate",
    "RAT_OTH_ATTR": "Rate",
    "RAT_FC_ATTR": "Rate",
    "QNT_ACCS_ATTR": "Quantity",
    "QUA_LF_ATTR": "Quantity",
    "PPU_ACCS_ATTR": "Price/Unit",
    "PPU_LF_ATTR": "Price/Unit",
    "PRI_LBR_WD_ATTR": "Price",
    "MES_WD_ATTR": "Measurement",
    "MES_OTH_ATTR": "Measurement",
    "MES_FC_ATTR": "Measurement",
    "MAT_WD_ATTR": "Material",
    "FIN_WD_ATTR": "Finish",
    "DES_ACCS_ATTR": "Description",
    "DES_FC_ATTR": "Description",
    "DES_LF_ATTR": "Description",
    "DES_WD_ATTR": "Description",
    "DES_OTH_ATTR": "Description",
    "BRD_ACCS_ATTR": "Brand",
}

def parse_item_attributes(attr_json: str) -> dict:
    if not attr_json or attr_json in ("null", "NULL"):
        return {}
    try:
        attrs = json.loads(attr_json) if isinstance(attr_json, str) else attr_json
        if not isinstance(attrs, dict):
            return {}
    except Exception:
        return {}
    parsed = {}
    for key, val in attrs.items():
        label = ATTRIBUTE_MAP.get(key, key)
        if isinstance(val, dict):
            if key.startswith("MES_"):
                width = val.get("width")
                length = val.get("length")
                unit = val.get("selectedOption", "")
                if width and length:
                    parsed[label] = f"{width}x{length} {unit}".strip()
                elif unit:
                    parsed[label] = unit
            elif "value" in val and "selectedOption" in val:
                parsed[label] = f"{val['value']} {val['selectedOption']}".strip()
            elif "value" in val:
                parsed[label] = str(val["value"])
            elif "selectedOption" in val:
                parsed[label] = str(val["selectedOption"])
            else:
                parsed[label] = " ".join(
                    f"{k}:{v}" for k, v in val.items() if v not in (None, "", "null")
                )
        else:
            if val not in (None, "", "null"):
                parsed[label] = str(val)
    return parsed

def measurement_to_sqft(measurement) -> float:
    if measurement is None:
        return 0.0
    if isinstance(measurement, (int, float)):
        return float(measurement)
    s = str(measurement).lower()
    is_inches = "inch" in s
    s = s.replace("feet", "").replace("ft", "").replace("inches", "").replace("inch", "")
    m = re.match(r"(\d+(\.\d+)?)x(\d+(\.\d+)?)", s)
    if m:
        w = float(m.group(1))
        l = float(m.group(3))
        if is_inches:
            w /= 12
            l /= 12
        return w * l
    try:
        return float(s)
    except Exception:
        return 0.0

def ingest_to_qdrant():
    create_collection()
    data = fetch_data()
    for row in data:
        estimator_id, room_name, item_name, item_id, amount, area, project_name, attributes, item_identifier, user_id, image = row
        # Qdrant expects point id as unsigned int or UUID
        try:
            point_id = int(item_id)
        except (TypeError, ValueError):
            point_id = str(uuid4())
        if isinstance(image, dict) and "default" in image:
            image = image["default"]
        parsed_attrs = parse_item_attributes(attributes)
        measurement = parsed_attrs.get("Measurement")
        measurement_sqft = measurement_to_sqft(measurement)

        text = (
            f"The material is {parsed_attrs.get('Material')} with a {parsed_attrs.get('Finish')} finish, "
            f"priced at {parsed_attrs.get('Rate')}, and measuring {parsed_attrs.get('Measurement')}. "
            f"{item_name} in {room_name} of {project_name}, located at {area}."
        )
        vector = MODEL.encode(text).tolist()

        # Derive stable keywords for filtering at query time
        try:
            raw_keywords = KW_MODEL.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=12
            )
            # Blacklist attribute labels (exact matches)
            attr_blacklist = {
                "rate", "rate per sqft", "quantity", "price/unit", "price",
                "measurement", "material", "finish", "description", "brand"
            }
            def allowed_phrase(phrase: str) -> bool:
                kl = phrase.strip().lower()
                if not kl:
                    return False
                # reject if exact or any token equals a blacklisted label
                if kl in attr_blacklist:
                    return False
                toks = re.findall(r"\b\w+\b", kl)
                return all(tok not in attr_blacklist for tok in toks)

            # Extract keywords from KeyBERT
            extracted_keywords = sorted({
                kw.strip().lower() for kw, _ in raw_keywords
                if isinstance(kw, str) and allowed_phrase(kw)
            })
            
            # Ensure important words from item_name are always included
            item_name_words = re.findall(r'\b\w+\b', item_name.lower())
            for word in item_name_words:
                if len(word) > 2 and word not in attr_blacklist:
                    extracted_keywords.append(word)
            
            # Also include important words from room_name
            room_name_words = re.findall(r'\b\w+\b', room_name.lower())
            for word in room_name_words:
                if len(word) > 2 and word not in attr_blacklist:
                    extracted_keywords.append(word)
            
            keywords = sorted(set(extracted_keywords))
        except Exception:
            keywords = []

        # Calculate amount if missing/null using AmountCalculatorUtils
        amount_to_store = amount
        try:
            if not amount_to_store or amount_to_store == 0:
                type_identifier = None
                if isinstance(item_identifier, str):
                    if "WD" in item_identifier:
                        type_identifier = "WD"
                    elif "FC" in item_identifier:
                        type_identifier = "FC"
                    elif "ACS" in item_identifier:
                        type_identifier = "ACS"
                    elif "LF" in item_identifier:
                        type_identifier = "LF"
                    elif "OTH" in item_identifier:
                        type_identifier = "OTH"
                if type_identifier:
                    dummy_item = type("Item", (), {"attributes": attributes})()
                    calculated_amount = AmountCalculatorUtils.calc_item_amount(type_identifier, dummy_item)
                    if calculated_amount and calculated_amount > 0:
                        amount_to_store = calculated_amount
        except Exception:
            # Swallow calculation errors and fall back to original amount
            pass
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "estimator_id": estimator_id,
                        "room_name": room_name,
                        "item_name": item_name,
                        "amount": amount_to_store,
                        "area": area,
                        "project_name": project_name,
                        "attributes": attributes,
                        "attributes_parsed": parsed_attrs,
                        "measurement_sqft": measurement_sqft,
                        "id": item_id,
                        "item_identifier": item_identifier,
                        "image": image,
                        "keywords": keywords
                    }
                }
            ]
        )

if __name__ == "__main__":
    ingest_to_qdrant()