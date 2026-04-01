from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional
import math
import numpy as np
import pandas as pd

from .text_utils import tokenize, get_ngrams, count_ngrams
from .embeddings import cosine_distance_vec

import pickle
from pathlib import Path


def mattr(text: str, window_size: int = 50) -> float:
    tokens = tokenize(text)
    if not tokens:
        return np.nan
    if len(tokens) <= window_size:
        return len(set(tokens)) / max(len(tokens), 1)
    scores = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        scores.append(len(set(window)) / window_size)
    return float(np.mean(scores))


def distinct_n(text: str, n: int = 2) -> float:
    tokens = tokenize(text)
    grams = get_ngrams(tokens, n)
    if not grams:
        return np.nan
    return len(set(grams)) / len(grams)

def load_ngram_reference(path: str):
    with Path(path).open("rb") as f:
        return pickle.load(f)

def build_ngram_reference(texts: List[str], n: int = 2) -> Counter:
    tokenized = [tokenize(t) for t in texts]
    return count_ngrams(tokenized, n=n)


def ngram_rarity_score(text: str, reference_counts: Counter, n: int = 2, smoothing: float = 1.0) -> float:
    tokens = tokenize(text)
    grams = get_ngrams(tokens, n)
    if not grams:
        return np.nan
    total = sum(reference_counts.values()) + smoothing * max(len(reference_counts), 1)
    scores = []
    for gram in grams:
        count = reference_counts.get(gram, 0)
        prob = (count + smoothing) / total
        scores.append(-math.log(prob))
    return float(np.mean(scores))


def response_to_centroid_distance(response_vec: np.ndarray, centroid_vec: np.ndarray) -> float:
    return cosine_distance_vec(response_vec, centroid_vec)


def add_novelty_metrics(
    df: pd.DataFrame,
    response_embeddings: np.ndarray,
    rarity_reference_counts: Counter,
    mattr_window: int = 50,
    distinct_n_value: int = 2,
    rarity_ngram_n: int = 2,
) -> pd.DataFrame:
    out = df.copy()
    out["novelty_mattr"] = out["response_content"].apply(lambda x: mattr(x, window_size=mattr_window))
    out["novelty_distinct_n"] = out["response_content"].apply(lambda x: distinct_n(x, n=distinct_n_value))
    out["novelty_ngram_rarity"] = out["response_content"].apply(
        lambda x: ngram_rarity_score(x, reference_counts=rarity_reference_counts, n=rarity_ngram_n)
    )

    centroid = np.nanmean(response_embeddings, axis=0)
    out["novelty_semantic_distance_centroid"] = [
        response_to_centroid_distance(vec, centroid) for vec in response_embeddings
    ]
    return out
