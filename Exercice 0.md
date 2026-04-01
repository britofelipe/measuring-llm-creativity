# measuring-llm-creativity

# Exercice 0 — Justification théorique des métriques

## 1. Positionnement général

Il existe trois dimensions théoriques de la créativité computationnelle : **nouveauté**, **valeur** et **surprise**. La nouveauté renvoie à la distance par rapport aux conventions et au corpus d’entraînement ; la valeur à la pertinence, la cohérence et l’utilité ; la surprise au caractère inattendu mais rétrospectivement justifié de la réponse. Le **Creativity Index (CI)** doit ensuite agréger ces dimensions et être validé par corrélation avec le **compar:IA Creative Score** (`creative`, `conv_creative_a`, `conv_creative_b`).

Dans notre cas, le choix des métriques est contraint par les données effectivement disponibles dans compar:IA. Nous disposons du **texte de la question**, du **texte de la réponse**, parfois du **system prompt**, ainsi que des **annotations humaines**. En revanche, nous ne disposons ni des **logits** ni des **probabilités token par token** du modèle générateur. Nous avons donc retenu uniquement des métriques **calculables à partir du texte** ou à l’aide de ressources externes légères (modèle d’embeddings, corpus de référence, LLM juge), mais pas des métriques nécessitant un accès interne au modèle, comme l’entropie de prédiction ou la surprise moyenne probabiliste.

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
  - dans la version 0, nous utilisons une **référence externe Wikipedia FR** sous forme de table de fréquences de bigrammes pré-calculée et sérialisée.
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
  - dans la version 0, nous calculons la **distance de la réponse au centroïde sémantique des réponses** du corpus analysé.

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

#### d) LLM-as-a-judge pour la pertinence et la cohérence globale
- **Ce que la métrique mesure** : une évaluation automatique, par un modèle juge, de la pertinence au prompt, de la cohérence globale, de l’utilité et éventuellement de la qualité rédactionnelle.
- **Pourquoi elle est pertinente** : certaines composantes de la valeur sont difficilement réductibles à une seule mesure purement lexicale.
- **Colonnes utilisées** :
  - `comparia-reactions.question_content`
  - `comparia-reactions.response_content`
  - `comparia-reactions.system_prompt` (facultatif)
- **Statut dans la version 0** :
  - l’interface logicielle a été préparée, mais le **LLM judge n’a pas encore été intégré** dans le calcul effectif du CI.
- **Conséquence** :
  - cette métrique n’entre pas encore dans l’agrégation numérique du score provisoire.

**Résumé pour la dimension valeur** :  
Dans la version 0, les métriques effectivement calculées sont **BERTScore**, **ROUGE-L** et **cohérence locale**. Le **LLM judge** reste prévu mais non activé à ce stade.

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

#### c) LLM-as-a-judge pour « inattendu mais justifié »
- **Ce que la métrique mesure** : une note donnée par un modèle juge à la dimension suivante : *la réponse propose-t-elle une idée inhabituelle, mais défendable a posteriori ?*
- **Pourquoi elle est pertinente** : cette formulation est très proche de la définition théorique de la surprise retenue dans le document.
- **Colonnes utilisées** :
  - `comparia-reactions.question_content`
  - `comparia-reactions.response_content`
  - `comparia-reactions.system_prompt` (facultatif)
- **Statut dans la version 0** :
  - prévu, mais **non encore implémenté** dans le score effectif.

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
- **Implémentation retenue dans la version 0** :
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
Dans la version 0, les métriques effectivement calculées sont **distance prompt-réponse**, **distance aux voisins du corpus**, **Divergent Thinking Score** et **variance de l’inattendu**. Le **LLM judge** de surprise reste prévu mais non activé.

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

## 4. Métriques effectivement calculées dans la version 0

À ce stade du projet, les métriques suivantes sont **effectivement calculées** dans le pipeline :

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

### Métriques préparées mais non activées
- `judge_value_relevance`
- `judge_value_global_coherence`
- `judge_value_utility`
- `judge_surprise_unexpected_but_justified`

Le **Creativity Index version 0** est donc calculé **sans les métriques LLM judge**.

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
| Valeur | LLM juge | Pertinence, cohérence globale, utilité | `question_content`, `response_content`, `system_prompt` | Préparée, non activée |
| Surprise | Distance prompt-réponse contrôlée | Déviation sémantique inattendue mais liée au prompt | `question_content`, `response_content` | Calculée |
| Surprise | Distance aux voisins du corpus | Écart à des réponses typiques pour prompts proches | `question_content`, `response_content`, `question_id` | Calculée |
| Surprise | LLM juge “inattendu mais justifié” | Surprise interprétable et rétrospectivement plausible | `question_content`, `response_content`, `system_prompt` | Préparée, non activée |
| Surprise | Divergent Thinking Score | Multiplicité d’idées distinctes et variation sémantique | `response_content`, éventuellement `question_content` | Calculée |
| Surprise | Variance de l’inattendu | Variabilité des similarités entre phrases | `response_content` | Calculée |

---

## 6. Construction du Creativity Index version 0

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

Dans la version 0, nous avons choisi une stratégie volontairement simple :

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

## 8. Conclusion provisoire

Cette première sélection de métriques vise à construire un **Creativity Index textuel, calculable et optimisable** à partir des données réellement disponibles dans compar:IA.

La version 0 du projet comprend désormais :
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