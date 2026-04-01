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
- `scripts/question1_fiabilite.py` : **Question 1 — Fiabilité des jugements humains**

## Installation conseillée

```bash
pip install -r requirements.txt
```

Ou manuellement :

```bash
pip install pandas pyarrow numpy scipy scikit-learn sentence-transformers bert-score rouge-score tqdm datasets krippendorff matplotlib seaborn huggingface_hub statsmodels
```

### Authentification Hugging Face

Les datasets compar:IA sont protégés (gated). Avant de lancer les scripts, il faut :

1. Accepter les conditions d'utilisation sur chaque page HF :
   - [comparia-reactions](https://huggingface.co/datasets/ministere-culture/comparia-reactions)
   - [comparia-votes](https://huggingface.co/datasets/ministere-culture/comparia-votes)
   - [comparia-conversations](https://huggingface.co/datasets/ministere-culture/comparia-conversations)

2. Se connecter via le terminal :
   ```bash
   huggingface-cli login
   ```

## Exécution

### Pipeline de métriques (Exercice 0)

```bash
python scripts/run_pipeline.py \
  --dataset "hf://datasets/ministere-culture/comparia-reactions/reactions.parquet" \
  --sample-size 5000 \
  --output-dir outputs
```

### Question 1 — Fiabilité des jugements humains

```bash
python scripts/question1_fiabilite.py --output-dir outputs/question1
```

## Remarques

- Les métriques utilisant des embeddings exigent un modèle `sentence-transformers`.
- Les métriques `LLM judge` sont optionnelles. Le projet inclut une interface/stub propre, mais **ne lance pas automatiquement d'appel API**.
- Le `N-gram Rarity Score` est calculé ici par défaut **sur le corpus compar:IA lui-même** (baseline empirique). Vous pourrez remplacer cette baseline plus tard par un corpus externe.

---

### Question 1 : Fiabilité des jugements humains (compar:IA)

Analyse de l'accord inter-annotateurs (IAA) sur les labels `creative`, `useful` et `incorrect` du dataset `comparia-reactions`.

#### Script principal
- [scripts/question1_fiabilite.py](scripts/question1_fiabilite.py) : Contient l'intégralité des analyses (Exercices 1.1, 1.2 et 1.3).
  - **1.1** : Accord inter-utilisateurs (taux d'accord brut + κ de Cohen) via deux approches : stricte (`response_content`) et proxy (`question_content`).
  - **1.2** : Biais de sélection des votants (Mann-Whitney, KS, χ²) sur 472 334 conversations.
  - **1.3** : α de Krippendorff sur les items multi-annotés (≥ 3 annotateurs distincts).

#### Résultats clés
- **κ de Cohen** (`creative`) : 0,25 (approche stricte, 57 items) / 0,05 (proxy, 778 items) → zone « slight » à « fair ».
- **Biais de sélection** : les conversations votées ont des réponses plus longues (médiane +342 tokens, r = −0,18), biais faible.
- **α de Krippendorff** (`creative`) : 0,185 (stricte, 14 items) / 0,114 (proxy, 176 items) → très en dessous du seuil de 0,67.

#### Analyse complète
Voir [Question 1.md](Question%201.md) pour la méthodologie détaillée, les tableaux de résultats complets et l'interprétation.

#### Utilisation
```bash
python scripts/question1_fiabilite.py --output-dir outputs/question1
```
Les graphiques et résultats sont sauvegardés dans `outputs/question1/`.

