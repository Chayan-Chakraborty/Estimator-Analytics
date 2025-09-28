"""
Central place to maintain romanized → English mappings and domain aliases
for house interior/common terms. Extend these dictionaries as needed.
"""

# Romanized Indic (primarily Bengali/Hindi) to English terms
ROMAN_TO_EN = {
    # Furniture - sleeping/storage/seating
    "bichana": "bed", "bichhana": "bed", "bistar": "bed",
    "palang": "bed", "khatiya": "cot", "charpai": "cot",
    "almari": "wardrobe", "almirah": "wardrobe", "almariy": "wardrobe",
    "almirahh": "wardrobe", "almirahh": "wardrobe",
    "almira": "wardrobe", "almirah": "wardrobe", "almirah": "wardrobe",
    "almirah": "wardrobe",
    "almari": "wardrobe", "almirah": "wardrobe",
    "almirah": "wardrobe", "wardrob": "wardrobe",
    "almirah": "wardrobe",
    "almirah": "wardrobe",
    "almirah": "wardrobe",
    "kabard": "cupboard", "kabat": "cupboard", "almari": "wardrobe",
    "table": "table", "console": "console", "sofa": "sofa",
    "diwan": "divan", "dressing": "dressing", "dresser": "dresser",
    "tv": "tv", "unit": "unit", "cabinet": "cabinet",

    # Kitchen
    "khana": "food", "rasoi": "kitchen", "ranna": "cooking", "chula": "stove",
    "chimni": "chimney", "sink": "sink", "slab": "slab",

    # Rooms/areas
    "ghar": "house", "bari": "house", "ghor": "room",
    "bedroom": "bedroom", "drawing": "living", "living": "living",
    "bathroom": "bathroom", "washroom": "bathroom", "toilet": "toilet",
    "balcony": "balcony",

    # Materials
    "blockboard": "blockboard", "blockbord": "blockboard", "blockboaard": "blockboard",
    "ply": "ply", "plywood": "plywood", "mdf": "mdf", "hdf": "hdf",
    "bwp": "bwp", "bwr": "bwr",

    # Connectors/phrases
    "ar": "and", "o": "and", "ebong": "and", "and": "and",
    "diye": "with", "dia": "with", "sathe": "with", "sathe": "with",
    "toiri": "made", "tairi": "made", "banano": "made", "kora": "made",
}


# Material aliases/families used by search filters
MATERIAL_ALIASES = {
    'blockboard': {'blockboard', 'block board', 'block-board', 'bb'},
    'bwp': {'bwp', 'boiling waterproof', 'boiling water proof'},
    'bwr': {'bwr', 'boiling water resistant', 'boiling-water-resistant'},
    'ply': {'ply', 'plywood'},
    'mdf': {'mdf'},
    'hdf': {'hdf'},
}


