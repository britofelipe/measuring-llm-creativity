from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from creativity_metrics.config import PipelineConfig, EmbeddingConfig, MetricConfig
from creativity_metrics.pipeline import run_pipeline
from creativity_metrics.analysis import summarize_metrics, correlations_with_human_labels, compare_creative_vs_noncreative, save_analysis_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcul des métriques de créativité sur comparia-reactions")
    parser.add_argument("--dataset", type=str, default="https://object.data.gouv.fr/ministere-culture/COMPARIA/reactions.parquet")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--mattr-window", type=int, default=50)
    parser.add_argument("--distinct-n", type=int, default=2)
    parser.add_argument("--rarity-ngram-n", type=int, default=2)
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--bertscore-lang", type=str, default="fr")
    args = parser.parse_args()

    pipeline_config = PipelineConfig(
        dataset_path=args.dataset,
        sample_size=args.sample_size,
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
    )

    df_scores = run_pipeline(
        pipeline_config=pipeline_config,
        embedding_config=embedding_config,
        metric_config=metric_config,
        judge_backend=None,  # remplacer plus tard par un vrai backend si besoin
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_scores.to_parquet(output_dir / "comparia_reactions_with_metrics.parquet", index=False)

    summary = summarize_metrics(df_scores)
    corr = correlations_with_human_labels(df_scores)
    comparison = compare_creative_vs_noncreative(df_scores)

    save_analysis_tables(output_dir, summary, corr, comparison)

    print("\n=== Aperçu des métriques ===")
    print(summary.head(20).to_string())
    print("\n=== Corrélations avec annotations humaines ===")
    print(corr.to_string())
    print("\n=== Différences creative vs non-creative ===")
    print(comparison.to_string())
    print(f"\nFichiers écrits dans: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
