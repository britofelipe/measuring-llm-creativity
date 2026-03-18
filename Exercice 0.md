# measuring-llm-creativity


# Exercice 0 — Justification théorique des métriques

## 1. Positionnement général

Il existe trois dimensions théoriques de la créativité computationnelle : **nouveauté**, **valeur** et **surprise**. La nouveauté renvoie à la distance par rapport aux conventions et au corpus d’entraînement ; la valeur à la pertinence, la cohérence et l’utilité ; la surprise au caractère inattendu mais rétrospectivement justifié de la réponse. Le **Creativity Index (CI)** doit ensuite agréger ces dimensions et être validé par corrélation avec le **compar:IA Creative Score** (`creative`, `conv_creative_a`, `conv_creative_b`). 

Dans notre cas, le choix des métriques est contraint par les données effectivement disponibles dans compar:IA. Nous disposons du **texte de la question**, du **texte de la réponse**, parfois du **system prompt**, ainsi que des **annotations humaines**. En revanche, nous ne disposons ni des **logits** ni des **probabilités token par token** du modèle générateur. Nous avons décidé de retenir donc uniquement des métriques **calculables à partir du texte** ou à l’aide de ressources externes légères (modèle d’embeddings, corpus de référence, LLM juge), mais pas des métriques nécessitant un accès interne au modèle, comme l’entropie de prédiction ou la surprise moyenne probabiliste. 

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
- **Ce que la métrique mesure** : la rareté des bigrammes ou trigrammes produits, par comparaison à un corpus de référence (ex: Wikipedia FR ou The Pile).
- **Pourquoi elle est pertinente** : elle approxime la distance de la réponse aux formulations fréquentes du langage ordinaire ou des corpus standards.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`
- **Ressource externe requise** :
  - corpus de référence externe, ou bien corpus compar:IA utilisé comme baseline empirique.
- **Remarque** : cette métrique est cohérente avec la famille A du document, qui propose explicitement la rareté des n-grammes comme indicateur de nouveauté. 

#### d) Distance sémantique par embeddings
- **Ce que la métrique mesure** : l’écart sémantique entre la réponse et un point de référence (centroïde du corpus, réponses voisines, ou espace des réponses typiques).
- **Pourquoi elle est pertinente** : la créativité ne se réduit pas à la diversité lexicale ; une réponse peut être originale au niveau des idées tout en restant lexicalement simple.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`
  - éventuellement `question_content` ou `question_id` pour définir des voisinages de prompts comparables.
- **Ressource externe requise** :
  - modèle d’embeddings.

**Résumé pour la dimension nouveauté** :  
Nous utiliserons donc **MATTR**, **Distinct-n**, **N-gram Rarity** et **distance sémantique**. Ces métriques couvrent ensemble la diversité lexicale, la diversité combinatoire et l’originalité sémantique.

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
- **Limite connue** : elle peut confondre pertinence et simple reprise lexicale, ce qui est déjà souligné dans le document.  [oai_citation:1‡etude_cas_creativite_LLM_v2.docx](sediment://file_00000000e61c72438bc5eb48dc9182d0)

#### c) Cohérence locale
- **Ce que la métrique mesure** : la continuité sémantique entre phrases consécutives de la réponse. Calculer la similarité cosinus entre les vecteurs d'embeddings de phrases consécutives. Une chute brutale de similarité indique une rupture narrative ou une hallucination.
- **Pourquoi elle est pertinente** : une réponse créative mais incohérente ne satisfait pas pleinement la dimension de valeur.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`
- **Ressource externe requise** :
  - segmentation en phrases + embeddings de phrases.

#### d) LLM-as-a-judge pour la pertinence et la cohérence globale
- **Ce que la métrique mesure** : une évaluation automatique, par un modèle juge, de la pertinence au prompt, de la cohérence globale, de l’utilité et éventuellement de la qualité rédactionnelle.
- **Pourquoi elle est pertinente** : certaines composantes de la valeur sont difficilement réductibles à une seule mesure purement lexicale.
- **Colonnes utilisées** :
  - `comparia-reactions.question_content`
  - `comparia-reactions.response_content`
  - `comparia-reactions.system_prompt` (facultatif, mais utile pour contextualiser la consigne du modèle)
- **Remarque** : cette métrique n’est pas dans la liste initiale du document, mais elle est compatible avec la logique de la famille B, puisqu’elle vise explicitement la pertinence et la cohérence globale.

**Résumé pour la dimension valeur** :  
Nous utiliserons **BERTScore**, **ROUGE-L**, **cohérence locale** et un **LLM juge**. Ensemble, ces métriques couvrent le respect de la consigne, la cohérence textuelle et l’utilité perçue de la réponse.

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
- **Remarque** : contrairement à BERTScore, qui mesure surtout l’alignement, cette métrique cherche un **écart contrôlé**.

#### b) Distance aux réponses voisines dans le corpus
- **Ce que la métrique mesure** : à quel point une réponse s’écarte des réponses les plus proches pour des prompts identiques ou similaires.
- **Pourquoi elle est pertinente** : elle remplace, de manière empirique, l’idée d’« écart à une baseline stochastique » proposée dans le document.  [oai_citation:2‡etude_cas_creativite_LLM_v2.docx](sediment://file_00000000e61c72438bc5eb48dc9182d0)
- **Colonnes utilisées** :
  - `comparia-reactions.question_content`
  - `comparia-reactions.response_content`
  - `comparia-reactions.question_id` pour regrouper des questions identiques
  - éventuellement `model_a_name`, `model_b_name`, `refers_to_model`, `msg_index` pour structurer les comparaisons
- **Ressource externe requise** :
  - embeddings + procédure de recherche de voisins.

#### c) LLM-as-a-judge pour « inattendu mais justifié »
- **Ce que la métrique mesure** : une note donnée par un modèle juge à la dimension suivante : *la réponse propose-t-elle une idée inhabituelle, mais défendable a posteriori ?*
- **Pourquoi elle est pertinente** : cette formulation est très proche de la définition théorique de la surprise retenue dans le document.  [oai_citation:3‡etude_cas_creativite_LLM_v2.docx](sediment://file_00000000e61c72438bc5eb48dc9182d0)
- **Colonnes utilisées** :
  - `comparia-reactions.question_content`
  - `comparia-reactions.response_content`
  - `comparia-reactions.system_prompt` (facultatif)

#### d) Divergent Thinking Score
- **Ce que la métrique mesure** : le nombre d’idées distinctes, de pistes de réponse différentes ou de métaphores non conventionnelles présentes dans une même réponse. Cette métrique vise à capter une forme de pensée divergente : la réponse ne se contente pas d’une seule continuation attendue, mais explore plusieurs directions sémantiques, tout en restant liée au sujet.  
- **Pourquoi elle est pertinente** : dans la dimension de **surprise**, l’intérêt n’est pas seulement qu’une réponse soit éloignée du prompt, mais qu’elle produise des associations inattendues, variées et néanmoins interprétables. Le **Divergent Thinking Score** permet ainsi de compléter la simple distance sémantique en mesurant la multiplicité des idées originales à l’intérieur d’un même texte.
- **Colonnes utilisées** :
  - `comparia-reactions.response_content`
  - éventuellement `comparia-reactions.question_content` pour vérifier que les idées restent ancrées dans le sujet global
- **Opérationnalisation possible** :
  - segmenter la réponse en unités d’idées ou en phrases ;
  - détecter les idées distinctes par clustering sémantique ou par extraction de segments thématiques ;
  - repérer les métaphores ou analogies non conventionnelles via la distance sémantique entre termes source–cible ;
  - calculer en complément une **variance de l’inattendu**, c’est-à-dire l’écart type des similarités entre phrases consécutives ou entre segments d’idées, afin de détecter des sauts sémantiques modérés mais non arbitraires.
- **Ressource externe requise** :
  - modèle d’embeddings ;
  - éventuellement heuristiques d’extraction d’analogies ou de segmentation en idées.
- **Remarque** : cette métrique reste un proxy textuel de la surprise. Elle est particulièrement utile pour distinguer une réponse simplement différente d’une réponse réellement inventive, capable de proposer plusieurs associations inattendues mais encore cohérentes avec la consigne.

**Résumé pour la dimension surprise** :  
Nous utiliserons des proxys textuels de surprise : **distance sémantique prompt-réponse**, **distance aux réponses voisines du corpus**, et **LLM juge “inattendu mais justifié”**. Ce choix est nécessaire, car les métriques probabilistes proposées dans la famille C ne sont pas directement calculables avec les données disponibles.  [oai_citation:4‡etude_cas_creativite_LLM_v2.docx](sediment://file_00000000e61c72438bc5eb48dc9182d0)

---

## 3. Colonnes du dataset mobilisées

### Dataset principal : `comparia-reactions`
Ce dataset sera la base principale du calcul du Creativity Index, car il fournit directement le couple **question-réponse** au niveau message.

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

## 4. Tableau synthétique

| Dimension | Métrique retenue | Ce qu’elle mesure | Colonnes compar:IA utilisées |
|---|---|---|---|
| Nouveauté | MATTR | Diversité lexicale robuste à la longueur | `response_content` |
| Nouveauté | Distinct-n | Diversité de n-grammes | `response_content` |
| Nouveauté | N-gram Rarity Score | Rareté des formulations produites | `response_content` |
| Nouveauté | Distance sémantique | Originalité conceptuelle / éloignement du centre ou des réponses typiques | `response_content`, éventuellement `question_content`, `question_id` |
| Valeur | BERTScore | Pertinence sémantique fine par rapport au prompt | `question_content`, `response_content` |
| Valeur | ROUGE-L | Respect lexical / structurel de la consigne | `question_content`, `response_content` |
| Valeur | Cohérence locale | Continuité entre phrases | `response_content` |
| Valeur | LLM juge | Pertinence, cohérence globale, utilité | `question_content`, `response_content`, `system_prompt` |
| Surprise | Distance prompt-réponse contrôlée | Déviation sémantique inattendue mais liée au prompt | `question_content`, `response_content` |
| Surprise | Distance aux voisins du corpus | Écart à des réponses typiques pour prompts proches | `question_content`, `response_content`, `question_id` |
| Surprise | LLM juge “inattendu mais justifié” | Surprise interprétable et rétrospectivement plausible | `question_content`, `response_content`, `system_prompt` |
| Surprise | Divergent Thinking Score | Multiplicité d’idées distinctes, associations inattendues, métaphores non conventionnelles et variance de l’inattendu entre segments | `response_content`, éventuellement `question_content` |

---

## 5. Conclusion provisoire

Cette première sélection de métriques vise à construire un **Creativity Index textuel et calculable** à partir des données réellement disponibles dans compar:IA. Nous retenons :
- pour la **nouveauté** : des métriques de diversité et de distance sémantique ;
- pour la **valeur** : des métriques d’alignement au prompt, de cohérence et de jugement automatique ;
- pour la **surprise** : des proxys sémantiques et un jugement automatique orienté vers l’« inattendu justifié ».

Dans une étape ultérieure, il faudra :
1. normaliser ces métriques ;
2. étudier leurs corrélations entre elles ;
3. identifier celles qui sont réellement informatives ;
4. proposer une pondération pour le **Creativity Index** ;
5. valider ce CI par corrélation avec `creative` et `conv_creative_*`, conformément à l’énoncé. 