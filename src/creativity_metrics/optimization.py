from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass
class OptimizationResult:
    feature_names: List[str]
    coefficients: Dict[str, float]
    intercept: float
    threshold: float
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


def get_normalized_metric_columns(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.endswith("_norm")])


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "creative",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = get_normalized_metric_columns(df)

    if target_col not in df.columns:
        raise ValueError(f"Colonne cible absente: {target_col}")

    model_df = df[feature_cols + [target_col]].copy()
    model_df = model_df.dropna()

    if model_df.empty:
        raise ValueError("Aucune donnée disponible après suppression des NaN.")

    X = model_df[feature_cols].copy()
    y = model_df[target_col].astype(int).copy()

    return X, y, feature_cols


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["spearman_with_target"] = pd.Series(y_prob).corr(pd.Series(y_true), method="spearman")
    else:
        metrics["roc_auc"] = np.nan
        metrics["spearman_with_target"] = np.nan

    return metrics


def train_creativity_logistic_regression(
    df: pd.DataFrame,
    target_col: str = "creative",
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
    class_weight: str | None = "balanced",
    max_iter: int = 1000,
) -> Tuple[pd.DataFrame, OptimizationResult]:
    X, y, feature_cols = prepare_training_data(df, target_col=target_col)

    stratify = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_predictions(y_train, train_prob, threshold=threshold)
    test_metrics = evaluate_predictions(y_test, test_prob, threshold=threshold)

    coefficients = {
        feature: float(coef)
        for feature, coef in zip(feature_cols, model.coef_[0])
    }

    result = OptimizationResult(
        feature_names=feature_cols,
        coefficients=coefficients,
        intercept=float(model.intercept_[0]),
        threshold=threshold,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    scored_df = df.copy()
    valid_mask = scored_df[feature_cols].notna().all(axis=1)
    scored_df["creativity_index_logreg"] = np.nan
    scored_df["creative_pred_logreg"] = pd.Series(
        pd.NA,
        index=scored_df.index,
        dtype="boolean",
    )

    if valid_mask.sum() > 0:
        full_prob = model.predict_proba(scored_df.loc[valid_mask, feature_cols])[:, 1]
        full_pred = pd.Series(
            full_prob >= threshold,
            index=scored_df.index[valid_mask],
            dtype="boolean",
        )
        scored_df.loc[valid_mask, "creativity_index_logreg"] = full_prob
        scored_df.loc[valid_mask, "creative_pred_logreg"] = full_pred

    return scored_df, result


def optimization_result_to_tables(result: OptimizationResult) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coef_df = pd.DataFrame(
        [{"feature": k, "coefficient": v} for k, v in result.coefficients.items()]
    ).sort_values("coefficient", ascending=False)

    metrics_rows = []
    for split_name, metrics in [("train", result.train_metrics), ("test", result.test_metrics)]:
        row = {"split": split_name, "threshold": result.threshold, "intercept": result.intercept}
        row.update(metrics)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    return coef_df, metrics_df
