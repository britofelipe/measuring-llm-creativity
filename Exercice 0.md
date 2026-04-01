# measuring-llm-creativity

# Exercice 0 — Justification théorique des métriques

## 1. Introduction

Il existe trois dimensions théoriques de la créativité computationnelle : **nouveauté**, **valeur** et **surprise**. La nouveauté renvoie à la distance par rapport aux conventions et au corpus d’entraînement ; la valeur à la pertinence, la cohérence et l’utilité ; la surprise au caractère inattendu mais rétrospectivement justifié de la réponse. Le **Creativity Index (CI)** doit ensuite agréger ces dimensions et être validé par corrélation avec le **compar:IA Creative Score** (`creative`, `conv_creative_a`, `conv_creative_b`).

Dans notre cas, le choix des métriques est contraint par les données effectivement disponibles dans compar:IA. Nous disposons du **texte de la question**, du **texte de la réponse**, parfois du **system prompt**, ainsi que des **annotations humaines**. En revanche, nous ne disposons ni des **logits** ni des **probabilités token par token** du modèle générateur. Nous avons donc retenu uniquement des métriques **calculables à partir du texte** ou à l’aide de ressources externes légères (modèle d’embeddings, corpus de référence), mais pas des métriques nécessitant un accès interne au modèle, comme l’entropie de prédiction ou la surprise moyenne probabiliste.

Le dataset le plus adapté pour un calcul **au niveau message** est **`comparia-reactions`**, car il contient directement les colonnes `question_content`, `response_content`, `system_prompt` et `creative`. Les datasets `comparia-votes` et `comparia-conversations` seront surtout utiles pour une validation complémentaire au niveau conversationnel.

---

## 2. Choix des métriques par dimension de créativité

### 2.1 Nouveauté

La **nouveauté** opérationnalise le fait qu’une réponse s’écarte des formulations banales, des patrons fréquents et des continuations trop attendues. Dans le document, cette dimension est liée à la rareté des n-grammes, à la diversité intra-modèle et à la distance sémantique.

Nous retenons les métriques suivantes :

#### a) MATTR (Moving-Average Type-Token Ratio)
- **Ce que la métrique mesure** : la diversité lexicale d’une réponse, en corrigeant la sensibilité du TTR classique à la longueur du texte.
- **Pourquoi elle est pertinente** : une réponse créative mobilise souvent un lexique moins répétitif et moins stéréotypé.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`
- **Niveau d’analyse** : message individuel.

#### b) Distinct-n (par ex. Distinct-2)
- **Ce que la métrique mesure** : la proportion de n-grammes distincts dans la réponse.
- **Pourquoi elle est pertinente** : elle capture la diversité de surface et complète MATTR par une vision plus combinatoire.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`

#### c) N-gram Rarity Score
- **Ce que la métrique mesure** : la rareté des bigrammes ou trigrammes produits, par comparaison à un corpus de référence.
- **Pourquoi elle est pertinente** : elle approxime la distance de la réponse aux formulations fréquentes du langage ordinaire ou des corpus standards.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`
- **Ressource externe requise** :
  - corpus de référence externe.
- **Implémentation retenue** :
  - dans une prémière version, nous utilisons une **référence externe Wikipedia FR** sous forme de table de fréquences de bigrammes pré-calculée et sérialisée.
- **Remarque** :
  - cette solution est plus stable que l’usage du seul échantillon compar:IA comme baseline empirique, qui rendait la métrique trop circulaire et peu discriminante.

#### d) Distance sémantique par embeddings
- **Ce que la métrique mesure** : l’écart sémantique entre la réponse et un point de référence.
- **Pourquoi elle est pertinente** : la créativité ne se réduit pas à la diversité lexicale ; une réponse peut être originale au niveau des idées tout en restant lexicalement simple.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`
  - éventuellement `question_content` ou `question_id` pour définir des voisinages de prompts comparables.
- **Ressource externe requise** :
  - modèle d’embeddings.
- **Implémentation retenue** :
  - dans une prémière version, nous calculons la **distance de la réponse au centroïde sémantique des réponses** du corpus analysé.

**Résumé pour la dimension nouveauté** :  
Nous utilisons **MATTR**, **Distinct-n**, **N-gram Rarity Score** et **distance sémantique au centroïde**. Ces métriques couvrent la diversité lexicale, la diversité combinatoire et une première approximation de l’originalité sémantique.

---

### 2.2 Valeur

La **valeur** renvoie à la qualité de la réponse en tant que solution au prompt : respect de la consigne, pertinence sémantique, cohérence interne et utilité. Le document associe cette dimension à la cohérence logique, au respect du prompt et à la qualité narrative.

Nous retenons les métriques suivantes :

#### a) BERTScore (prompt ↔ réponse)
- **Ce que la métrique mesure** : la proximité sémantique fine entre le prompt et la réponse.
- **Pourquoi elle est pertinente** : elle estime si la réponse traite réellement la demande, même sans répétition littérale des termes du prompt.
- **Colonnes utilisées** :
  - `comparia-reactions.question_content`
  - `comparia-reactions.response_content`

#### b) ROUGE-L (prompt ↔ réponse)
- **Ce que la métrique mesure** : le recouvrement séquentiel entre le prompt et la réponse.
- **Pourquoi elle est pertinente** : elle fournit un signal simple de respect de la consigne, utile comme mesure d’appoint.
- **Colonnes utilisées** :
  - `comparia-reactions.question_content`
  - `comparia-reactions.response_content`
- **Limite connue** :
  - elle peut confondre pertinence et simple reprise lexicale.

#### c) Cohérence locale
- **Ce que la métrique mesure** : la continuité sémantique entre phrases consécutives de la réponse.
- **Pourquoi elle est pertinente** : une réponse créative mais incohérente ne satisfait pas pleinement la dimension de valeur.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`
- **Ressource externe requise** :
  - segmentation en phrases + embeddings de phrases.
- **Implémentation retenue** :
  - nous calculons la **similarité cosinus moyenne entre phrases successives** ; une chute brutale signale une rupture discursive ou une instabilité narrative.


**Résumé pour la dimension valeur** :  
Les métriques calculées sont **BERTScore**, **ROUGE-L** et **cohérence locale**. 

---

### 2.3 Surprise

La **surprise** désigne le fait qu’une réponse soit inattendue, mais qu’elle paraisse néanmoins pertinente une fois lue. Le document propose pour cette famille des métriques comme l’entropie de prédiction et la surprise moyenne, mais celles-ci supposent un accès aux probabilités du modèle, ce que compar:IA ne fournit pas. Nous devons donc construire des **proxies textuels** de la surprise.

Nous retenons les métriques suivantes :

#### a) Distance sémantique contrôlée entre prompt et réponse
- **Ce que la métrique mesure** : le degré de déviation sémantique de la réponse par rapport au prompt.
- **Pourquoi elle est pertinente** : une réponse surprenante n’est pas seulement pertinente ; elle explore une direction moins triviale tout en restant reliée à la consigne.
- **Colonnes utilisées** :
  - `comparia-reactions.question_content`
  - `comparia-reactions.response_content`
- **Ressource externe requise** :
  - modèle d’embeddings.
- **Remarque** :
  - contrairement à BERTScore, qui mesure surtout l’alignement, cette métrique cherche un **écart contrôlé**.
- **Implémentation retenue** :
  - nous calculons la **distance cosinus entre embedding du prompt et embedding de la réponse**.

#### b) Distance aux réponses voisines dans le corpus
- **Ce que la métrique mesure** : à quel point une réponse s’écarte des réponses les plus proches pour des prompts identiques ou similaires.
- **Pourquoi elle est pertinente** : elle remplace, de manière empirique, l’idée d’« écart à une baseline stochastique » proposée dans le document.
- **Colonnes utilisées** :
  - `comparia-reactions.question_content`
  - `comparia-reactions.response_content`
  - `comparia-reactions.question_id`
  - éventuellement `msg_index` ou les identifiants de modèle
- **Ressource externe requise** :
  - embeddings + procédure de recherche de voisins.
- **Implémentation retenue** :
  - nous recherchons les **k plus proches voisins dans l’espace des questions**, puis nous calculons la distance moyenne entre la réponse courante et les réponses associées à ces voisins.


#### d) Divergent Thinking Score
- **Ce que la métrique mesure** : le nombre d’idées distinctes, de pistes de réponse différentes ou de métaphores non conventionnelles présentes dans une même réponse.
- **Pourquoi elle est pertinente** : dans la dimension de **surprise**, l’intérêt n’est pas seulement qu’une réponse soit éloignée du prompt, mais qu’elle produise des associations inattendues, variées et néanmoins interprétables.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`
  - éventuellement `comparia-reactions.question_content`
- **Opérationnalisation possible** :
  - segmenter la réponse en unités d’idées ou en phrases ;
  - détecter les idées distinctes par clustering sémantique ;
  - repérer des associations non conventionnelles ;
  - calculer en complément une **variance de l’inattendu**.
- **Implémentation retenue** :
  - segmentation de la réponse en phrases ;
  - embeddings de phrases ;
  - clustering agglomératif des phrases ;
  - calcul du **nombre de groupes sémantiques distincts** ;
  - calcul de la **variance des similarités entre phrases consécutives** ;
  - construction d’un **score composite provisoire** de divergence.

#### e) Variance de l’inattendu
- **Ce que la métrique mesure** : l’écart type de la similarité entre phrases consécutives ou segments d’idées.
- **Pourquoi elle est pertinente** : elle vise à détecter des sauts sémantiques modérés, compatibles avec une réponse inventive mais encore ancrée dans un fil global.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`
- **Implémentation retenue** :
  - cette quantité est calculée comme **métrique autonome** et également mobilisée dans le Divergent Thinking Score.

**Résumé pour la dimension surprise** :  
Les métriques calculées sont **distance prompt-réponse**, **distance aux voisins du corpus**, **Divergent Thinking Score** et **variance de l’inattendu**.

---

## 3. Colonnes du dataset mobilisées

### Dataset principal : `comparia-reactions`
Ce dataset constitue la base principale du calcul du Creativity Index, car il fournit directement le couple **question-réponse** au niveau message.

#### Colonnes utilisées pour les métriques
- `question_content` : texte de la question utilisateur
- `response_content` : texte de la réponse du modèle
- `system_prompt` : consigne système du modèle, utile pour le jugement automatique
- `question_id` : identifiant de la question, utile pour comparer des réponses à la même demande
- `msg_index` : utile si l’on veut tenir compte de la position du message dans la conversation

#### Colonnes utilisées pour la validation
- `creative` : ancre principale de validation humaine du CI
- `useful` : variable complémentaire pour la dimension de valeur
- `complete` : autre indicateur de qualité perçue
- `incorrect` : contrôle négatif
- `superficial` : contrôle négatif utile pour la profondeur créative
- `instructions_not_followed` : contrôle négatif de pertinence

### Datasets secondaires
- `comparia-votes` :
  - validation complémentaire au niveau conversation avec `conv_creative_a`, `conv_creative_b`
- `comparia-conversations` :
  - utile si l’on souhaite agréger les scores à l’échelle de la conversation complète.

---

## 4. Métriques calculées

### Nouveauté
- `novelty_mattr`
- `novelty_distinct_n`
- `novelty_ngram_rarity`
- `novelty_semantic_distance_centroid`

### Valeur
- `value_bertscore_f1`
- `value_rouge_l_prompt_response`
- `value_local_coherence`

### Surprise
- `surprise_prompt_response_distance`
- `surprise_distance_to_neighbors`
- `surprise_divergent_idea_count`
- `surprise_unexpected_variance`
- `surprise_divergent_score`


Le **Creativity Index** est donc calculé à partir de ces métriques.

---

## 5. Tableau synthétique

| Dimension | Métrique retenue | Ce qu’elle mesure | Colonnes compar:IA utilisées | Statut v0 |
|---|---|---|---|---|
| Nouveauté | MATTR | Diversité lexicale robuste à la longueur | `response_content` | Calculée |
| Nouveauté | Distinct-n | Diversité de n-grammes | `response_content` | Calculée |
| Nouveauté | N-gram Rarity Score | Rareté des formulations produites | `response_content` | Calculée |
| Nouveauté | Distance sémantique | Originalité conceptuelle / éloignement du centroïde | `response_content` | Calculée |
| Valeur | BERTScore | Pertinence sémantique fine par rapport au prompt | `question_content`, `response_content` | Calculée |
| Valeur | ROUGE-L | Respect lexical / structurel de la consigne | `question_content`, `response_content` | Calculée |
| Valeur | Cohérence locale | Continuité entre phrases | `response_content` | Calculée |
| Surprise | Distance prompt-réponse contrôlée | Déviation sémantique inattendue mais liée au prompt | `question_content`, `response_content` | Calculée |
| Surprise | Distance aux voisins du corpus | Écart à des réponses typiques pour prompts proches | `question_content`, `response_content`, `question_id` | Calculée |
| Surprise | Divergent Thinking Score | Multiplicité d’idées distinctes et variation sémantique | `response_content`, éventuellement `question_content` | Calculée |
| Surprise | Variance de l’inattendu | Variabilité des similarités entre phrases | `response_content` | Calculée |

---

## 6. Construction du Creativity Index

### 6.1 Normalisation des métriques

Comme les métriques brutes n’ont pas les mêmes échelles, nous appliquons une **normalisation robuste** à chaque métrique calculée.

Pour chaque métrique \(x\), nous calculons d’abord un **robust z-score** :

\[
z_{\text{robuste}} = \frac{x - \text{médiane}(x)}{\text{IQR}(x)}
\]

où l’IQR correspond à l’intervalle interquartile :

\[
\text{IQR}(x) = Q_{0.75}(x) - Q_{0.25}(x)
\]

Cette transformation est préférée à une standardisation classique par moyenne et écart-type, car elle est **moins sensible aux valeurs extrêmes**.

Ensuite, pour obtenir un score borné entre 0 et 1, nous appliquons une **fonction sigmoïde** :

\[
s(x) = \frac{1}{1 + e^{-z_{\text{robuste}}}}
\]

Chaque métrique brute \(x\) est ainsi transformée en une métrique normalisée \(s(x)\), notée dans le code par un suffixe `_norm`.

Exemples :
- `novelty_mattr_norm`
- `value_bertscore_f1_norm`
- `surprise_divergent_score_norm`

### 6.2 Pondération initiale

Nous avons choisi une stratégie simple :

- **un poids par métrique**
- **tous les poids initialement égaux**
- **pas de contrainte imposant un poids fixe par dimension théorique**

Autrement dit, nous ne supposons pas a priori que **nouveauté**, **valeur** et **surprise** doivent contribuer chacune pour un tiers. Le score provisoire est calculé comme une **somme pondérée directe de toutes les métriques disponibles**, après normalisation.

Si l’ensemble des métriques activées est noté \(M = \{m_1, \dots, m_K\}\), alors le poids initial est :

\[
w_k = \frac{1}{K}
\]

et le score provisoire est :

\[
CI_{\text{provisoire}} = \sum_{k=1}^{K} w_k \, m_k^{\text{norm}}
\]

avec renormalisation des poids disponibles si certaines métriques sont manquantes pour une observation donnée.

### 6.3 Seuil de décision provisoire

Une fois le score obtenu, nous définissons une prédiction binaire provisoire :

\[
\widehat{creative} =
\begin{cases}
1 & \text{si } CI_{\text{provisoire}} \geq 0.5 \\\\
0 & \text{sinon}
\end{cases}
\]

Ce seuil à `0.5` est uniquement une **convention initiale**. Il ne constitue pas encore un calibrage optimal.

---

## 7. Régression logistique pour optimiser les poids

### 7.1 Objectif

Après la définition du score provisoire, nous avons introduit une étape d’**optimisation supervisée** afin d’ajuster les poids des métriques de façon plus cohérente avec l’annotation humaine `creative`.

L’idée est de ne plus fixer les poids manuellement, mais de les **apprendre à partir des données**.

### 7.2 Variables explicatives

Les variables utilisées dans la régression sont les **métriques normalisées** calculées précédemment, c’est-à-dire les colonnes suffixées par `_norm`.

Exemples :
- `novelty_mattr_norm`
- `novelty_ngram_rarity_norm`
- `value_local_coherence_norm`
- `surprise_distance_to_neighbors_norm`
- `surprise_divergent_score_norm`

La variable cible est :

- `creative` \(\in \{0,1\}\)

### 7.3 Modèle

Nous utilisons une **régression logistique binaire** :

\[
P(creative = 1 \mid X) = \sigma \left(\beta_0 + \sum_{k=1}^{K} \beta_k X_k \right)
\]

où :
- \(X_k\) représente la \(k\)-ième métrique normalisée,
- \(\beta_k\) est le poids appris pour cette métrique,
- \(\beta_0\) est l’interception,
- \(\sigma\) est la fonction sigmoïde logistique.

La probabilité prédite par le modèle devient un second score de créativité :

\[
CI_{\text{logreg}} = P(creative = 1 \mid X)
\]

Ce score est noté dans le pipeline :

- `creativity_index_logreg`

et la prédiction binaire associée :

- `creative_pred_logreg`

### 7.4 Intérêt de cette étape

La régression logistique permet :

- d’obtenir une **pondération empirique** de chaque métrique ;
- d’identifier quelles métriques poussent le plus fortement vers `creative = 1` ou `creative = 0` ;
- de comparer un **score manuel provisoire** à un **score appris** ;
- de préparer une version ultérieure du CI davantage alignée sur les jugements humains.

En particulier :
- un coefficient \(\beta_k > 0\) signifie qu’une métrique élevée augmente la probabilité d’être jugé créatif ;
- un coefficient \(\beta_k < 0\) signifie qu’une métrique élevée est plutôt associée à `creative = 0`.

### 7.5 Évaluation

Pour cette première version, nous évaluons le score logistique au moyen de :
- l’**accuracy**
- la **precision**
- le **recall**
- le **F1-score**
- l’**AUC ROC**
- la **corrélation de Spearman** entre le score et `creative`

Nous conservons également les coefficients appris dans un tableau séparé, afin de pouvoir les interpréter et éventuellement les réinjecter ensuite dans une nouvelle version du **Creativity Index**.

---

## 8. Récapitulation

Cette première sélection de métriques vise à construire un **Creativity Index textuel, calculable et optimisable** à partir des données réellement disponibles dans compar:IA.

Le projet comprend :
- un ensemble de **métriques textuelles et sémantiques effectivement calculées** ;
- une **normalisation robuste** de ces métriques ;
- un **score provisoire** obtenu par somme pondérée à poids égaux ;
- une première étape de **régression logistique supervisée** pour apprendre des poids mieux alignés avec `creative`.

Les prochaines étapes consisteront à :
1. étudier les corrélations entre métriques ;
2. vérifier qualitativement les exemples extrêmes ;
3. analyser les effets de longueur et les biais éventuels ;
4. comparer le score provisoire et le score appris ;
5. valider ensuite le CI sur d’autres sous-ensembles, puis au niveau conversationnel avec `conv_creative_*`.

---

## 9. Famille D — Métriques agrégées et composées

### 9.1 Dimension théorique opérationnalisée

La famille des métriques agrégées opérationnalise la créativité comme une **propriété émergente multi-dimensionnelle** : une réponse est jugée créative si elle combine simultanément :
- de la nouveauté (distance aux formulations banales),
- de la valeur (pertinence/cohérence),
- de la surprise (inattendu interprétable).

Le rôle des métriques agrégées est donc d’**intégrer** ces signaux partiels dans un score unique. Puis il faut vérifier que ce score est bien aligné avec le signal humain compar:IA (`creative` au niveau message, `conv_creative_*` au niveau conversation).

### 9.2 Protocole expérimental implémenté

Le pipeline actuel implémente deux agrégats.

#### a) Agrégat manuel : `creativity_index_provisional`
- normalisation robuste de chaque métrique : robust z-score (médiane/IQR) puis sigmoïde ;
- agrégation par moyenne pondérée ligne à ligne ;
- pondération initiale uniforme sur toutes les métriques disponibles ;
- seuil binaire provisoire : `0.5`.

Ce protocole est codé dans `scoring.py` (`normalize_metric`, `weighted_row_mean`, `add_provisional_creativity_index`).

#### b) Agrégat appris : `creativity_index_logreg`
- variables explicatives : toutes les colonnes normalisées ;
- cible : `creative` (binaire) ;
- split `train/test` stratifié (`test_size = 0.2`) ;
- modèle : `LogisticRegression` avec `class_weight="balanced"` ;
- seuil de décision : **variable** (mode `train_f1`) ; le seuil retenu sur ce run est `0.565`.

Ce protocole est codé dans `optimization.py` (`prepare_training_data`, `train_creativity_logistic_regression`).

### 9.3 Résultats observés (`outputs_logreg2`)

Sur l’ensemble scoré (niveau message), on observe :

| Score agrégé | n | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | Spearman vs `creative` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `creativity_index_provisional` | 2000 | 99 | 821 | 1041 | 39 | 0.460 | 0.0868 | 0.717 | 0.155 | 0.077 |
| `creativity_index_logreg` | 1914 | 64 | 1359 | 423 | 68 | 0.743 | 0.131 | 0.485 | 0.207 | 0.162 |

Le passage à la régression logistique améliore les performances globales (accuracy `0.46 -> 0.74`, F1 `0.155 -> 0.207`, Spearman `0.077 -> 0.162`) ;
- le score provisoire sur-prédit fortement la classe créative (`1140` prédictions positives pour seulement `138` positifs réels) ;
- le score logistique réduit nettement les faux positifs (`1041 -> 423`), mais au prix d’une baisse du rappel (`0.717 -> 0.485`).
- l’écart `n=2000` vs `n=1914` vient des lignes incomplètes sur certaines métriques (notamment celles basées sur segmentation en phrases).

Résultats `train/test` (`logreg_train_test_metrics.csv`) :
- train : AUC `0.689`, F1 `0.221` ;
- test : AUC `0.664`, F1 `0.154` ;
- seuil appris : `0.565` (`threshold_mode = train_f1`).

Interprétation :
- le problème reste fortement déséquilibré (environ `6.9%` de positifs `creative`) ;
- l’optimisation du seuil améliore la calibration par rapport à un `0.5` arbitraire ;
- le modèle apporte un gain net vs score provisoire, mais la séparation créatif/non-créatif demeure modérée (AUC test ~ `0.66`).

### 9.4 Pondération justifiée pour le CI (à partir des coefficients appris)

Les coefficients logistiques les plus élevés sont :
- `novelty_ngram_rarity_norm` (`+1.364`),
- `novelty_distinct_n_norm` (`+1.233`),
- `surprise_prompt_response_distance_norm` (`+0.949`),
- `value_local_coherence_norm` (`+0.841`),
- `surprise_distance_to_neighbors_norm` (`+0.812`).

Les coefficients négatifs les plus marqués sont :
- `value_rouge_l_prompt_response_norm` (`-1.470`),
- `novelty_semantic_distance_centroid_norm` (`-1.082`),
- `surprise_unexpected_variance_norm` (`-0.135`).

En agrégeant la **magnitude** des coefficients (`|beta|`) par famille :
- nouveauté : `39.6%`,
- valeur : `27.4%`,
- surprise : `33.0%`.

Une pondération de famille cohérente avec ces résultats est donc :

\[
\alpha_{\text{nouveauté}} = 0.40,\quad
\beta_{\text{valeur}} = 0.27,\quad
\gamma_{\text{surprise}} = 0.33,\quad
\alpha+\beta+\gamma=1
\]

et, au niveau métrique, une pondération interne proportionnelle à `|beta_k|` :

\[
w_k = \frac{|\beta_k|}{\sum_j |\beta_j|}
\]

Cette règle garde la simplicité du CI tout en remplaçant les poids arbitraires par une estimation empirique.

### 9.5 Cas d’échec identifiés (faux positifs / faux négatifs)

Cas d’échec plausibles de la famille D dans ce pipeline :
- **Faux positifs (FP)** : réponses très divergentes lexicalement/sémantiquement, riches en idées distinctes, mais jugées peu créatives par l’humain (hors-sujet, verbeuses, gimmicks stylistiques).
- **Faux positifs (FP)** : réponses “surprenantes” qui ressemblent à de l’instabilité discursive ; la surprise est captée, mais la valeur perçue ne suit pas.
- **Faux négatifs (FN)** : réponses créatives mais concises/sobres, avec faible divergence de surface et fort recouvrement lexical du prompt (potentiellement pénalisées par la combinaison des poids appris).
- **Faux négatifs (FN)** : créativité contextuelle ou culturelle subtile (jeu de mots, référence implicite) insuffisamment visible via embeddings génériques.

Analyse rapide à partir de `failure_cases_report.md` :
- Sur les `1914` lignes éligibles, on observe `423` FP contre `68` FN ; l’erreur dominante reste donc la sur-prédiction de la créativité.
- Les FP les plus probables (ex. `id=308403`, `id=317475`) sont des réponses courtes ou de suivi conversationnel, avec forte rareté n-gramme et distance sémantique, mais sans signal humain de créativité.
- Cette forte rareté n-gramme peut provenir d’éléments de style ou de contexte (termes spécifiques, formulations rares), sans garantir une créativité perçue par l’annotateur.
- Les FN inspectés (ex. `id=264392`, `id=336716`) montrent l’effet inverse : réponses jugées créatives par l’humain mais pénalisées par de fortes contributions négatives de `value_rouge_l_prompt_response_norm` et `novelty_semantic_distance_centroid_norm`.
- Dans `id=336716` (menu Dragon Ball), des termes d’univers et jeux d’associations (“Capsules de Shenron”, “Ki”, etc.) augmentent la rareté n-gramme, mais la structure très guidée de menu/recette peut réduire d’autres signaux.
- Dans `id=264392` (naming entreprise écologique), la proposition de noms composés/néologiques (`EcoRénov`, `RénoVert`, `Rénov'Eco`) crée aussi de la rareté lexicale, mais cela n’est pas toujours converti en score créatif final selon les autres coefficients.
- Ces cas confirment que le modèle capte bien l’originalité formelle, mais sous-capte encore la créativité “utile et contextuelle”.

### 9.6 Validation compar:IA Creative Score et limites actuelles

Validation message-level effectuée :
- score provisoire : Spearman `0.077`.
- score logistique : Spearman `0.162`.

Validation encore incomplète au regard de la consigne famille D :
- corrélation conversationnelle avec `conv_creative_a` / `conv_creative_b` : à faire (nécessite agrégation au niveau `comparia-votes`).
- **Creativity-Coherence Frontier** : à produire (frontière Pareto `novelty_score` vs `value_score`) pour visualiser le compromis originalité/cohérence entre modèles.
