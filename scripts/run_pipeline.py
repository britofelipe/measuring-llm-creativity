from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from creativity_metrics.config import PipelineConfig, EmbeddingConfig, MetricConfig
from creativity_metrics.pipeline import run_pipeline
from creativity_metrics.analysis import (
    summarize_metrics,
    correlations_with_human_labels,
    compare_creative_vs_noncreative,
    evaluate_score_against_creative,
    evaluate_provisional_index,
    save_analysis_tables,
)
from creativity_metrics.optimization import (
    train_creativity_logistic_regression,
    optimization_result_to_tables,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Calcul des métriques de créativité sur comparia-reactions")
    parser.add_argument("--dataset", type=str, default="https://object.data.gouv.fr/ministere-culture/COMPARIA/reactions.parquet")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--mattr-window", type=int, default=50)
    parser.add_argument("--distinct-n", type=int, default=2)
    parser.add_argument("--rarity-ngram-n", type=int, default=2)
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--bertscore-lang", type=str, default="fr")
    parser.add_argument("--rarity-reference-path", type=str, default=None)
    parser.add_argument("--optimize-logreg", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    pipeline_config = PipelineConfig(
        dataset_path=args.dataset,
        sample_size=args.sample_size,
        random_state =args.seed,
    )
    embedding_config = EmbeddingConfig(
        model_name=args.embedding_model,
    )
    metric_config = MetricConfig(
        mattr_window=args.mattr_window,
        distinct_n=args.distinct_n,
        rarity_ngram_n=args.rarity_ngram_n,
        neighbor_k=args.neighbor_k,
        bertscore_lang=args.bertscore_lang,
        rarity_reference_path=args.rarity_reference_path,
    )

    df_scores = run_pipeline(
        pipeline_config=pipeline_config,
        embedding_config=embedding_config,
        metric_config=metric_config,
        judge_backend=None,  # remplacer plus tard par un vrai backend si besoin
    )

    provisional_eval = evaluate_score_against_creative(
        df_scores,
        score_col="creativity_index_provisional",
        threshold=args.threshold,
    )

    logreg_eval = None
    logreg_coef = None
    logreg_metrics = None

    if args.optimize_logreg:
        df_scores, opt_result = train_creativity_logistic_regression(
            df_scores,
            target_col="creative",
            test_size=args.test_size,
            threshold=args.threshold,
        )
        logreg_coef, logreg_metrics = optimization_result_to_tables(opt_result)

        logreg_eval = evaluate_score_against_creative(
            df_scores,
            score_col="creativity_index_logreg",
            threshold=args.threshold,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_scores.to_parquet(output_dir / "comparia_reactions_with_metrics.parquet", index=False)

    summary = summarize_metrics(df_scores)
    corr = correlations_with_human_labels(df_scores)
    comparison = compare_creative_vs_noncreative(df_scores)
    provisional_eval = evaluate_provisional_index(df_scores, threshold=0.5)

    save_analysis_tables(
        output_dir,
        summary,
        corr,
        comparison,
        provisional_eval=provisional_eval,
        logreg_eval=logreg_eval,
        logreg_coef=logreg_coef,
        logreg_metrics=logreg_metrics,
    )

    print("\\n=== Évaluation du score provisoire ===")
    print(provisional_eval.to_string(index=False))

    if logreg_coef is not None:
        print("\\n=== Coefficients de la régression logistique ===")
        print(logreg_coef.to_string(index=False))

    if logreg_metrics is not None:
        print("\\n=== Performance train/test de la régression logistique ===")
        print(logreg_metrics.to_string(index=False))

    if logreg_eval is not None:
        print("\\n=== Évaluation du score logistique sur l'ensemble scoré ===")
        print(logreg_eval.to_string(index=False))


if __name__ == "__main__":
    main()
