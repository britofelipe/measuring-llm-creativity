# measuring-llm-creativity

Projet Python modulaire pour :
- charger `comparia-reactions`
- calculer les métriques proposées pour **nouveauté**, **valeur** et **surprise**
- produire des tableaux de synthèse, corrélations simples et comparaisons avec les annotations humaines (`creative`, `useful`, etc.)

## Structure

- `src/creativity_metrics/config.py` : configuration globale
- `src/creativity_metrics/data.py` : chargement et préparation du dataset
- `src/creativity_metrics/text_utils.py` : tokenisation et segmentation en phrases
- `src/creativity_metrics/embeddings.py` : embeddings et similarités
- `src/creativity_metrics/metrics_novelty.py` : métriques de nouveauté
- `src/creativity_metrics/metrics_value.py` : métriques de valeur
- `src/creativity_metrics/metrics_surprise.py` : métriques de surprise
- `src/creativity_metrics/llm_judge.py` : interface facultative pour un LLM juge
- `src/creativity_metrics/pipeline.py` : pipeline principal de calcul
- `src/creativity_metrics/analysis.py` : résumés et corrélations
- `scripts/run_pipeline.py` : exécution de bout en bout

## Installation conseillée

```bash
pip install pandas pyarrow numpy scipy scikit-learn sentence-transformers bert-score rouge-score tqdm
```

## Exécution

```bash
python scripts/run_pipeline.py \
  --dataset "hf://datasets/ministere-culture/comparia-reactions/reactions.parquet" \
  --sample-size 5000 \
  --output-dir outputs
```

````
python scripts/build_ngram_reference.py \
  --dataset-name kaitchup/wikipedia-20220301-fr-sample-10k \
  --split train \
  --text-col text \
  --ngram-n 2 \
  --output-path resources/wikipedia_fr_sample_bigram_counts.pkl
```
```
python scripts/run_pipeline.py \
  --sample-size 50 \
  --rarity-reference-path resources/wikipedia_fr_sample_bigram_counts.pkl \
  --output-dir outputs_wiki_ref
```
```
python scripts/run_pipeline.py \
  --sample-size 200 \
  --rarity-reference-path resources/wikipedia_fr_sample_bigram_counts.pkl \
  --output-dir outputs_logreg \
  --optimize-logreg
```
## Remarques

- Les métriques utilisant des embeddings exigent un modèle `sentence-transformers`.
- Les métriques `LLM judge` sont optionnelles. Le projet inclut une interface/stub propre, mais **ne lance pas automatiquement d’appel API**.
- Le `N-gram Rarity Score` est calculé ici par défaut **sur le corpus compar:IA lui-même** (baseline empirique). Vous pourrez remplacer cette baseline plus tard par un corpus externe.

