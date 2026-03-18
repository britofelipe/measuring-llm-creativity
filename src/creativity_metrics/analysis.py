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


def save_analysis_tables(output_dir: str | Path, summary: pd.DataFrame, corr: pd.DataFrame, comparison: pd.DataFrame) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "metrics_summary.csv")
    corr.to_csv(output_dir / "correlations_with_human_labels.csv")
    comparison.to_csv(output_dir / "creative_vs_noncreative.csv")
