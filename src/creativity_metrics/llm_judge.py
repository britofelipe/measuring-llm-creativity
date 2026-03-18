from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol
import numpy as np
import pandas as pd


class JudgeBackend(Protocol):
    def score_value(self, question: str, response: str, system_prompt: str = "") -> Dict[str, float]:
        ...

    def score_surprise(self, question: str, response: str, system_prompt: str = "") -> Dict[str, float]:
        ...


@dataclass
class NullJudge:
    """
    Backend par défaut : ne fait aucun appel API.
    Retourne NaN pour permettre au pipeline de tourner.
    """
    def score_value(self, question: str, response: str, system_prompt: str = "") -> Dict[str, float]:
        return {
            "judge_value_relevance": np.nan,
            "judge_value_global_coherence": np.nan,
            "judge_value_utility": np.nan,
        }

    def score_surprise(self, question: str, response: str, system_prompt: str = "") -> Dict[str, float]:
        return {
            "judge_surprise_unexpected_but_justified": np.nan,
        }


def add_judge_scores(df: pd.DataFrame, backend: Optional[JudgeBackend] = None) -> pd.DataFrame:
    backend = backend or NullJudge()
    out = df.copy()

    value_scores = [
        backend.score_value(q, r, s)
        for q, r, s in zip(out["question_content"], out["response_content"], out["system_prompt"])
    ]
    surprise_scores = [
        backend.score_surprise(q, r, s)
        for q, r, s in zip(out["question_content"], out["response_content"], out["system_prompt"])
    ]

    for key in ["judge_value_relevance", "judge_value_global_coherence", "judge_value_utility"]:
        out[key] = [row[key] for row in value_scores]

    out["judge_surprise_unexpected_but_justified"] = [
        row["judge_surprise_unexpected_but_justified"] for row in surprise_scores
    ]
    return out
