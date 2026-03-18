from __future__ import annotations

from dataclasses import asdict
from typing import Optional
import pandas as pd


def load_reactions_dataset(path: str, sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)
    return df


def validate_columns(df: pd.DataFrame, required_columns) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset: {missing}")


def prepare_reactions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Standardiser les colonnes textuelles
    for col in ["question_content", "response_content", "system_prompt"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    # Nettoyage minimal des annotations humaines
    bool_cols = [
        "creative", "useful", "complete", "incorrect",
        "superficial", "instructions_not_followed"
    ]
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)

    # Longueurs utiles pour le diagnostic
    out["response_char_len"] = out["response_content"].str.len()
    out["question_char_len"] = out["question_content"].str.len()
    return out
