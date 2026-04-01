# measuring-llm-creativity

Projet Python modulaire pour :
- charger `comparia-reactions`
- calculer les métriques proposées pour **nouveauté**, **valeur** et **surprise**
- produire des tableaux de synthèse, corrélations simples et comparaisons avec les annotations humaines (`creative`, `useful`, etc.)

## État d'avancement

- **Pipeline v0 opérationnel** sur `comparia-reactions` avec calcul des métriques de nouveauté, valeur et surprise.
- **Creativity Index v0 implémenté** : normalisation robuste + agrégation pondérée (poids initiaux égaux).
- **Optimisation supervisée ajoutée** : régression logistique (`creative`) pour apprendre des poids empiriques.
- **Résultat Ex.3.1 (biais de longueur, `comparia-votes`)** : ρ Spearman = `0.1237` (p `< 2e-16`) ; coefficient BT `+0.1428`.
- **Résultat Ex.3.2 (biais de position, `comparia-reactions`)** : avantage position B pour `liked` (p `= 0.003`), `useful` (p `= 2.48e-06`) et `creative` (p `= 0.019`) ; non significatif pour `incorrect` (p `= 0.319`).


## Structure

- `src/creativity_metrics/config.py` : configuration globale
- `src/creativity_metrics/data.py` : chargement et préparation du dataset
- `src/creativity_metrics/text_utils.py` : tokenisation et segmentation en phrases
- `src/creativity_metrics/embeddings.py` : embeddings et similarités
- `src/creativity_metrics/metrics_novelty.py` : métriques de nouveauté
- `src/creativity_metrics/metrics_value.py` : métriques de valeur
- `src/creativity_metrics/metrics_surprise.py` : métriques de surprise
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

To generate the N-gram reference from wikipedia:
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

### Question 2 : Modèle Bradley-Terry (compar:IA)

Analyse complète du classement des LLM par préférences humaines via le modèle de Bradley-Terry.

#### Notebook Principal
- [exercise_2_combined.ipynb](file:///Users/felipebrito/Workspace/measuring-llm-creativity/src/bradley-terry/exercise_2_combined.ipynb) : Contient l'intégralité des analyses (Exercices 2.1, 2.2 et 2.3).
  - **2.1** : Modèle global vs créativité, intervalles de confiance par bootstrap.
  - **2.2** : Transitivité stochastique, analyse de puissance, extension Davidson pour les ex-æquo.
  - **2.3** : Modèles à covariables (Output length, turns, categories) et GLMM.

### Utilisation
1. Ouvrez le notebook `exercise_2_combined.ipynb` dans VS Code ou Jupyter.
2. Assurez-vous d'avoir les dépendances installées (`statsmodels`, `datasets`, etc.).
3. Les graphiques et résultats sont automatiquement sauvegardés dans le dossier `results/`.
