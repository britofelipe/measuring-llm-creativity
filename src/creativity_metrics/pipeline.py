from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from .config import PipelineConfig, EmbeddingConfig, MetricConfig
from .data import load_reactions_dataset, validate_columns, prepare_reactions_dataframe
from .embeddings import SentenceEmbedder
from .metrics_novelty import build_ngram_reference, add_novelty_metrics
from .metrics_value import add_value_metrics
from .metrics_surprise import add_surprise_metrics
from .llm_judge import add_judge_scores, JudgeBackend


def run_pipeline(
    pipeline_config: PipelineConfig,
    embedding_config: EmbeddingConfig,
    metric_config: MetricConfig,
    judge_backend: Optional[JudgeBackend] = None,
) -> pd.DataFrame:
    df = load_reactions_dataset(
        path=pipeline_config.dataset_path,
        sample_size=pipeline_config.sample_size,
        random_state=pipeline_config.random_state,
    )
    validate_columns(df, pipeline_config.required_columns)
    df = prepare_reactions_dataframe(df)

    embedder = SentenceEmbedder(
        model_name=embedding_config.model_name,
        batch_size=embedding_config.batch_size,
        normalize_embeddings=embedding_config.normalize_embeddings,
    )

    question_embeddings = embedder.encode(df["question_content"].tolist())
    response_embeddings = embedder.encode(df["response_content"].tolist())

    rarity_reference = build_ngram_reference(
        df["response_content"].tolist(),
        n=metric_config.rarity_ngram_n,
    )

    df = add_novelty_metrics(
        df=df,
        response_embeddings=response_embeddings,
        rarity_reference_counts=rarity_reference,
        mattr_window=metric_config.mattr_window,
        distinct_n_value=metric_config.distinct_n,
        rarity_ngram_n=metric_config.rarity_ngram_n,
    )

    df = add_value_metrics(
        df=df,
        sentence_embedder=embedder,
        bertscore_lang=metric_config.bertscore_lang,
    )

    df = add_surprise_metrics(
        df=df,
        question_embeddings=question_embeddings,
        response_embeddings=response_embeddings,
        sentence_embedder=embedder,
        neighbor_k=metric_config.neighbor_k,
    )

    df = add_judge_scores(df, backend=judge_backend)
    return df
