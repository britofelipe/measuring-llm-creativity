from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

from .text_utils import split_sentences
from .embeddings import cosine_distance_vec, cosine_similarity


def controlled_prompt_response_distance(prompt_vec: np.ndarray, response_vec: np.ndarray) -> float:
    return cosine_distance_vec(prompt_vec, response_vec)


def distance_to_question_neighbors(
    df: pd.DataFrame,
    response_embeddings: np.ndarray,
    question_embeddings: np.ndarray,
    neighbor_k: int = 5,
) -> np.ndarray:
    # Voisins calculés dans l'espace des questions ; on mesure ensuite la distance entre réponses.
    nn = NearestNeighbors(n_neighbors=min(neighbor_k + 1, len(df)), metric="cosine")
    nn.fit(question_embeddings)
    distances, indices = nn.kneighbors(question_embeddings)

    scores = []
    for i in range(len(df)):
        neigh_idx = [j for j in indices[i].tolist() if j != i]
        if not neigh_idx:
            scores.append(np.nan)
            continue
        dists = [cosine_distance_vec(response_embeddings[i], response_embeddings[j]) for j in neigh_idx]
        scores.append(float(np.mean(dists)))
    return np.array(scores)


def divergent_thinking_score(response: str, sentence_embedder, question: Optional[str] = None) -> Dict[str, float]:
    sentences = split_sentences(response)
    if len(sentences) < 2:
        return {
            "surprise_divergent_idea_count": np.nan,
            "surprise_unexpected_variance": np.nan,
            "surprise_divergent_score": np.nan,
        }

    sent_vecs = sentence_embedder.encode(sentences)

    # Variance de l'inattendu : variabilité des similarités entre phrases consécutives
    consecutive_sims = []
    for i in range(len(sent_vecs) - 1):
        sim = float(cosine_similarity(sent_vecs[i], sent_vecs[i + 1])[0, 0])
        consecutive_sims.append(sim)
    unexpected_variance = float(np.std(consecutive_sims)) if consecutive_sims else np.nan

    # Idées distinctes : clustering agglomératif naïf sur les phrases
    # distance_threshold adapté à des embeddings normalisés
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=0.35,
        )
    except TypeError:
        # compatibilité anciennes versions sklearn
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=0.35,
        )

    labels = clustering.fit_predict(sent_vecs)
    idea_count = int(len(set(labels.tolist())))

    # Score simple : nombre d'idées distinctes pondéré par la variance de l'inattendu
    # à raffiner ensuite empiriquement
    score = float(idea_count * (1.0 + (unexpected_variance if not np.isnan(unexpected_variance) else 0.0)))

    return {
        "surprise_divergent_idea_count": float(idea_count),
        "surprise_unexpected_variance": unexpected_variance,
        "surprise_divergent_score": score,
    }


def add_surprise_metrics(
    df: pd.DataFrame,
    question_embeddings: np.ndarray,
    response_embeddings: np.ndarray,
    sentence_embedder,
    neighbor_k: int = 5,
) -> pd.DataFrame:
    out = df.copy()

    out["surprise_prompt_response_distance"] = [
        controlled_prompt_response_distance(qv, rv)
        for qv, rv in zip(question_embeddings, response_embeddings)
    ]

    out["surprise_distance_to_neighbors"] = distance_to_question_neighbors(
        df=out,
        response_embeddings=response_embeddings,
        question_embeddings=question_embeddings,
        neighbor_k=neighbor_k,
    )

    divergent_scores = [
        divergent_thinking_score(resp, sentence_embedder, question=q)
        for q, resp in zip(out["question_content"].tolist(), out["response_content"].tolist())
    ]
    out["surprise_divergent_idea_count"] = [d["surprise_divergent_idea_count"] for d in divergent_scores]
    out["surprise_unexpected_variance"] = [d["surprise_unexpected_variance"] for d in divergent_scores]
    out["surprise_divergent_score"] = [d["surprise_divergent_score"] for d in divergent_scores]

    return out
