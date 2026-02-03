"""
Utilities for normalizing/expanding keyword lists.

This module exists primarily to support ingestion/search flows that expect a
`generate_flat_keywords()` helper. Some deployments/import paths referenced this
module, so we keep it small and dependency-free.
"""

from __future__ import annotations

from typing import Iterable, Set


def generate_flat_keywords(keywords: Iterable[str]) -> Set[str]:
    """
    Given an iterable of keyword phrases, return a flattened set including:
    - the original phrase (trimmed, lowercased)
    - individual tokens from the phrase

    Example:
      ["Console Table", "TV wall unit"] ->
      {"console table", "console", "table", "tv wall unit", "tv", "wall", "unit"}
    """
    out: Set[str] = set()
    if not keywords:
        return out

    for kw in keywords:
        if not isinstance(kw, str):
            continue
        phrase = " ".join(kw.strip().lower().split())
        if not phrase:
            continue
        out.add(phrase)
        for tok in phrase.split():
            if tok:
                out.add(tok)
    return out

