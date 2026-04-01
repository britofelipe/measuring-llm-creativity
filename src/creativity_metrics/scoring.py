from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


NOVELTY_METRICS = [
    "novelty_mattr",
    "novelty_distinct_n",
    "novelty_ngram_rarity",
    "novelty_semantic_distance_centroid",
]

VALUE_METRICS = [
    "value_bertscore_f1",
    "value_local_coherence",
    "value_rouge_l_prompt_response",
]

SURPRISE_METRICS = [
    "surprise_prompt_response_distance",
    "surprise_distance_to_neighbors",
    "surprise_divergent_idea_count",
    "surprise_unexpected_variance",
    "surprise_divergent_score",
]


def robust_zscore(series: pd.Series) -> pd.Series:
    """
    Standardisation robuste:
    z = (x - median) / IQR
    """
    s = pd.to_numeric(series, errors="coerce")
    median = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        # fallback sur écart-type si IQR nul
        std = s.std()
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    return (s - median) / iqr


def sigmoid(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalize_metric(series: pd.Series) -> pd.Series:
    """
    Transforme une métrique brute en score [0,1].
    """
    z = robust_zscore(series)
    return pd.Series(sigmoid(z), index=series.index)


def weighted_row_mean(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Moyenne pondérée ligne par ligne, en ignorant les NaN
    et en renormalisant les poids disponibles.
    """
    weight_series = pd.Series(weights, dtype=float)
    available_cols = [c for c in weights if c in df.columns]
    if not available_cols:
        return pd.Series(np.nan, index=df.index)

    sub = df[available_cols].copy()
    row_scores = []

    for _, row in sub.iterrows():
        valid = row.notna()
        if valid.sum() == 0:
            row_scores.append(np.nan)
            continue

        valid_cols = row.index[valid].tolist()
        valid_weights = weight_series[valid_cols]
        valid_weights = valid_weights / valid_weights.sum()

        score = float((row[valid_cols] * valid_weights).sum())
        row_scores.append(score)

    return pd.Series(row_scores, index=df.index)

def build_default_weights() -> Dict[str, Dict[str, float]]:
    
    #Définit un poids par métrique.
    #Version initiale : tous les poids sont égaux.
    all_metrics = NOVELTY_METRICS + VALUE_METRICS + SURPRISE_METRICS
    equal_weight = 1.0 / len(all_metrics)

    metric_weights = {metric: equal_weight for metric in all_metrics}

    return {
        "all_metrics": metric_weights,
    }

def add_provisional_creativity_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    weights = build_default_weights()
    all_metrics = NOVELTY_METRICS + VALUE_METRICS + SURPRISE_METRICS

    # normalisation
    for metric in all_metrics:
        if metric in out.columns:
            out[f"{metric}_norm"] = normalize_metric(out[metric])

    # scores descriptifs par dimension
    novelty_norm_cols = [f"{m}_norm" for m in NOVELTY_METRICS if f"{m}_norm" in out.columns]
    value_norm_cols = [f"{m}_norm" for m in VALUE_METRICS if f"{m}_norm" in out.columns]
    surprise_norm_cols = [f"{m}_norm" for m in SURPRISE_METRICS if f"{m}_norm" in out.columns]

    if novelty_norm_cols:
        out["novelty_score"] = out[novelty_norm_cols].mean(axis=1)
    if value_norm_cols:
        out["value_score"] = out[value_norm_cols].mean(axis=1)
    if surprise_norm_cols:
        out["surprise_score"] = out[surprise_norm_cols].mean(axis=1)

    # score final = somme pondérée de toutes les métriques
    metric_norm_weights = {
        f"{metric}_norm": weight
        for metric, weight in weights["all_metrics"].items()
        if f"{metric}_norm" in out.columns
    }

    out["creativity_index_provisional"] = weighted_row_mean(out, metric_norm_weights)
    out["creative_pred_provisional"] = out["creativity_index_provisional"] >= 0.5

    return out
"""
def build_default_weights() -> Dict[str, Dict[str, float]]:
    novelty_weights = {m: 1 / len(NOVELTY_METRICS) for m in NOVELTY_METRICS}
    value_weights = {m: 1 / len(VALUE_METRICS) for m in VALUE_METRICS}
    surprise_weights = {m: 1 / len(SURPRISE_METRICS) for m in SURPRISE_METRICS}

    dimension_weights = {
        "novelty_score": 1 / 3,
        "value_score": 1 / 3,
        "surprise_score": 1 / 3,
    }

    return {
        "novelty": novelty_weights,
        "value": value_weights,
        "surprise": surprise_weights,
        "dimensions": dimension_weights,
    }


def add_provisional_creativity_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    weights = build_default_weights()

    metric_groups = {
        "novelty": NOVELTY_METRICS,
        "value": VALUE_METRICS,
        "surprise": SURPRISE_METRICS,
    }

    # 1. normaliser chaque métrique individuelle
    for group_name, metrics in metric_groups.items():
        for metric in metrics:
            if metric in out.columns:
                out[f"{metric}_norm"] = normalize_metric(out[metric])

    # 2. agréger par dimension
    novelty_norm_weights = {f"{m}_norm": w for m, w in weights["novelty"].items() if f"{m}_norm" in out.columns}
    value_norm_weights = {f"{m}_norm": w for m, w in weights["value"].items() if f"{m}_norm" in out.columns}
    surprise_norm_weights = {f"{m}_norm": w for m, w in weights["surprise"].items() if f"{m}_norm" in out.columns}

    out["novelty_score"] = weighted_row_mean(out, novelty_norm_weights)
    out["value_score"] = weighted_row_mean(out, value_norm_weights)
    out["surprise_score"] = weighted_row_mean(out, surprise_norm_weights)

    # 3. score global provisoire
    out["creativity_index_provisional"] = weighted_row_mean(
        out,
        weights["dimensions"],
    )

    # 4. prédiction provisoire avec seuil 0.5
    out["creative_pred_provisional"] = out["creativity_index_provisional"] >= 0.5

    return out 
"""