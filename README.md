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

## Remarques

- Les métriques utilisant des embeddings exigent un modèle `sentence-transformers`.
- Les métriques `LLM judge` sont optionnelles. Le projet inclut une interface/stub propre, mais **ne lance pas automatiquement d’appel API**.
- Le `N-gram Rarity Score` est calculé ici par défaut **sur le corpus compar:IA lui-même** (baseline empirique). Vous pourrez remplacer cette baseline plus tard par un corpus externe.

## Question 2 : Modèle Bradley-Terry (compar:IA)

Pipeline pour classer les modèles selon les préférences humaines (`comparia-votes`) via l'estimation du modèle de Bradley-Terry.

### Structure (`src/bradley-terry/`)
- `preprocessing.py` : Nettoyage, filtres de N comparaisons, matrices de gains ($W[i,j]$) et d'empates.
- `exercise_2_1.py` : Entraînement MM du modèle global vs classement créativité (Spearman $\rho$).
- `exercise_2_2.py` : Transitivité stochastique, power analysis, et modèle de Davidson pour *ex-æquo*.
- `exercise_2_3.py` : Modèle à covariables (Régression Logistique GLMM) capturant l'effet du temps et de la longueur des réponses sur la préférence.

### Exécution
Assurez-vous d'être connecté à compte HuggingFace via `huggingface-cli login` pour télécharger `comparia-votes`.
```bash
python src/bradley-terry/exercise_2_1.py
python src/bradley-terry/exercise_2_2.py
python src/bradley-terry/exercise_2_3.py
```
