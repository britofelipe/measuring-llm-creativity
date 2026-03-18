from __future__ import annotations

from typing import Iterable, List, Optional
import numpy as np


class SentenceEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64, normalize_embeddings: bool = True):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers est requis pour les métriques basées sur embeddings. "
                "Installez-le avec `pip install sentence-transformers`."
            ) from e
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return a @ b.T


def cosine_distance_vec(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(1.0 - np.dot(a, b) / denom)
