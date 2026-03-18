from __future__ import annotations

from pathlib import Path
from typing import List, Sequence
import numpy as np
import pandas as pd


HUMAN_COLUMNS = [
    "creative",
    "useful",
    "complete",
    "incorrect",
    "superficial",
    "instructions_not_followed",
]


def metric_columns(df: pd.DataFrame) -> List[str]:
    prefixes = ("novelty_", "value_", "surprise_", "judge_")
    return [c for c in df.columns if c.startswith(prefixes)]


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cols = metric_columns(df)
    if not cols:
        return pd.DataFrame()
    summary = df[cols].describe().T
    summary["missing_rate"] = df[cols].isna().mean()
    return summary.sort_index()


def correlations_with_human_labels(df: pd.DataFrame) -> pd.DataFrame:
    metrics = metric_columns(df)
    available_human = [c for c in HUMAN_COLUMNS if c in df.columns]
    corr_rows = []
    for metric in metrics:
        row = {"metric": metric}
        for human_col in available_human:
            valid = df[[metric, human_col]].dropna()
            if len(valid) < 5:
                row[human_col] = np.nan
            else:
                row[human_col] = valid[metric].corr(valid[human_col].astype(float), method="spearman")
        corr_rows.append(row)
    return pd.DataFrame(corr_rows).set_index("metric").sort_index()


def compare_creative_vs_noncreative(df: pd.DataFrame) -> pd.DataFrame:
    metrics = metric_columns(df)
    if "creative" not in df.columns:
        return pd.DataFrame()
    rows = []
    for metric in metrics:
        grp = df[[metric, "creative"]].dropna()
        if grp.empty:
            continue
        row = {
            "metric": metric,
            "mean_creative_true": grp.loc[grp["creative"], metric].mean(),
            "mean_creative_false": grp.loc[~grp["creative"], metric].mean(),
            "difference": grp.loc[grp["creative"], metric].mean() - grp.loc[~grp["creative"], metric].mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric").sort_index()


def save_analysis_tables(
    output_dir: str | Path,
    summary: pd.DataFrame,
    corr: pd.DataFrame,
    comparison: pd.DataFrame,
    provisional_eval: pd.DataFrame | None = None,
    logreg_eval: pd.DataFrame | None = None,
    logreg_coef: pd.DataFrame | None = None,
    logreg_metrics: pd.DataFrame | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_dir / "metrics_summary.csv")
    corr.to_csv(output_dir / "correlations_with_human_labels.csv")
    comparison.to_csv(output_dir / "creative_vs_noncreative.csv")

    if provisional_eval is not None and not provisional_eval.empty:
        provisional_eval.to_csv(output_dir / "provisional_index_eval.csv", index=False)

    if logreg_eval is not None and not logreg_eval.empty:
        logreg_eval.to_csv(output_dir / "logreg_score_eval.csv", index=False)

    if logreg_coef is not None and not logreg_coef.empty:
        logreg_coef.to_csv(output_dir / "logreg_coefficients.csv", index=False)

    if logreg_metrics is not None and not logreg_metrics.empty:
        logreg_metrics.to_csv(output_dir / "logreg_train_test_metrics.csv", index=False)
        
def evaluate_provisional_index(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    if "creative" not in df.columns or "creativity_index_provisional" not in df.columns:
        return pd.DataFrame()

    eval_df = df[["creative", "creativity_index_provisional"]].dropna().copy()
    if eval_df.empty:
        return pd.DataFrame()

    eval_df["pred"] = eval_df["creativity_index_provisional"] >= threshold
    eval_df["creative"] = eval_df["creative"].astype(bool)

    tp = int(((eval_df["pred"] == True) & (eval_df["creative"] == True)).sum())
    tn = int(((eval_df["pred"] == False) & (eval_df["creative"] == False)).sum())
    fp = int(((eval_df["pred"] == True) & (eval_df["creative"] == False)).sum())
    fn = int(((eval_df["pred"] == False) & (eval_df["creative"] == True)).sum())

    accuracy = (tp + tn) / len(eval_df) if len(eval_df) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) else np.nan

    # corrélation score ↔ label humain
    spearman = eval_df["creativity_index_provisional"].corr(eval_df["creative"].astype(float), method="spearman")

    return pd.DataFrame([{
        "threshold": threshold,
        "n": len(eval_df),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "spearman_with_creative": spearman,
        "mean_score_creative_true": eval_df.loc[eval_df["creative"], "creativity_index_provisional"].mean(),
        "mean_score_creative_false": eval_df.loc[~eval_df["creative"], "creativity_index_provisional"].mean(),
    }])

def evaluate_score_against_creative(
    df: pd.DataFrame,
    score_col: str,
    target_col: str = "creative",
    threshold: float = 0.5,
) -> pd.DataFrame:
    if score_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    eval_df = df[[score_col, target_col]].dropna().copy()
    if eval_df.empty:
        return pd.DataFrame()

    eval_df[target_col] = eval_df[target_col].astype(int)
    eval_df["pred"] = (eval_df[score_col] >= threshold).astype(int)

    tp = int(((eval_df["pred"] == 1) & (eval_df[target_col] == 1)).sum())
    tn = int(((eval_df["pred"] == 0) & (eval_df[target_col] == 0)).sum())
    fp = int(((eval_df["pred"] == 1) & (eval_df[target_col] == 0)).sum())
    fn = int(((eval_df["pred"] == 0) & (eval_df[target_col] == 1)).sum())

    accuracy = (tp + tn) / len(eval_df) if len(eval_df) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) else np.nan
    spearman = eval_df[score_col].corr(eval_df[target_col], method="spearman")

    return pd.DataFrame([{
        "score_col": score_col,
        "threshold": threshold,
        "n": len(eval_df),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "spearman_with_creative": spearman,
        "mean_score_creative_true": eval_df.loc[eval_df[target_col] == 1, score_col].mean(),
        "mean_score_creative_false": eval_df.loc[eval_df[target_col] == 0, score_col].mean(),
    }])