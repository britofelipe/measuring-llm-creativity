# Exercice 3 — Biais systématiques dans les données

## Exercice 3.1 — Quantification du biais de longueur

**Statistiques descriptives sur comparia-votes (152 842 matchs) :**
- **Corrélation de Spearman** entre la différence de longueur (`length_diff`) et la victoire du modèle A (`vote_A`) : **ρ = 0.1237** (p-value < 2e-16).
- **Coefficient de régression logistique (BT)** pour `length_diff_norm` : **+0.1428**.

**Interprétation :**
La corrélation positive et statistiquement très significative confirme puissamment le biais de verbosité (*Length Bias*) : plus la réponse d'un modèle est longue par rapport à l'autre, plus ses chances d'être votées comme la meilleure réponse augmentent.
Le coefficient positif dans la régression de Bradley-Terry indique que la différence de longueur explique à elle seule une part substantielle de la décision humaine de choisir A plutôt que B, indépendamment du modèle utilisé.

**Impact sur le classement Bradley-Terry (BT) :**
L'introduction de la longueur comme variable de contrôle modifie de façon évidente le classement intrinsèque des modèles. Nous constatons de grands "**Changements de rang majeurs**" une fois le score corrigé du biais de longueur :
- `glm-5` : chute du rang 39 au rang 52.
- `grok-3-mini-beta` : chute du rang 29 au rang 39.
- `trinity-large-preview` : remonte du rang 71 au rang 62.

Cela prouve que certains modèles (comme GLM-5 ou Grok-3) obtiennent une part significative de leurs bons scores publics non pas par une intelligence supérieure, mais par une simple astuce d'interface : ils génèrent beaucoup plus de texte. À l'inverse, des modèles plus concis comme Trinity sont pénalisés à tort par les utilisateurs dans un classement non-corrigé.

---

## Exercice 3.2 — Biais de position A/B

Tests d'indépendance du Chi-Deux sur la variable binaire `model_pos` (A ou B) dans le dataset `comparia-reactions`.

**Résultats des tests statistiques :**
- **`liked` (Réaction positive générale) :** Chi2 = 8.66, p-value = 0.003
  - Taux A = 67.27% | Taux B = 68.18% -> **Différence significative** (Avantage B)
- **`useful` (Utile) :** Chi2 = 22.18, p-value = 2.48e-06
  - Taux A = 23.14% | Taux B = 24.47% -> **Différence très significative** (Avantage B)
- **`creative` (Créatif) :** Chi2 = 5.49, p-value = 0.019
  - Taux A = 6.55% | Taux B = 6.94% -> **Différence significative** (Avantage B)
- **`incorrect` (Faux) :** Chi2 = 0.99, p-value = 0.319
  - Taux A = 11.13% | Taux B = 10.92% -> **Non significatif**

**Interprétation :**
Le biais de position est indéniablement mesurable et statistiquement significatif sur le corpus. Étonnamment (contrairement à beaucoup de "Chatbot arenas" où la position A, à gauche, est favorisée par le biais de primauté), dans l'interface ou le corpus *compar:IA*, la position **B** attire systématiquement un taux d'upvotes, d'annotations `useful` et `creative` légèrement plus important.

Toutefois, nous remarquons que pour les jugements négatifs et "durs" quant au factuel ("`incorrect`"), la p-value de 0.319 (qui est supérieure à 0.05) montre que l'effet de la position disparaît. Les annotateurs sont donc biaisés par la position quand il s'agit de jugements subjectifs et qualitatifs de préférence (créativité, utilité), mais restent rigoureux et objectifs face à des erreurs factuelles claires (indépendamment de si la réponse était A ou B).

---

## Exercice 3.3 — Protocole d’annotation amélioré

La nature ouverte et "crowdsourcée" de plateformes comme compar:IA ou la LMSYS Chatbot Arena les expose inévitablement à plusieurs catégories de biais cognitifs et statistiques. Notre objectif est de proposer des ajustements au protocole d'annotation pour mitiger le **biais de position**, le **biais de longueur**, et le **biais de sélection**, le tout en évitant d’alourdir la tâche cognitive de l'utilisateur (ce qui pourrait accroître le taux d'abandon au-delà de 15%).

Voici les propositions d'amélioration intégrables dans une future version :

### 1. Atténuation du Biais de Longueur (Length Bias)
Le *verbosity bias* ou biais de longueur fait qu'un utilisateur perçoit souvent un texte plus long comme plus exhaustif et plus créatif, même s’il est dilué (Singhal et al., 2023). 
* **Solution d'interface (UI) :** Implémenter une troncature visuelle symétrique. Si la différence de longueur entre le modèle A et B dépasse 15%, les deux textes sont affichés de base dans une boîte de hauteur maximale fixe (avec fondu au blanc) accompagnée d'un bouton « Lire la suite ». Cela supprime le choc visuel initial ("le modèle A a beaucoup plus travaillé que B"). L'utilisateur doit s'engager avec le texte pour juger la longueur, et non plus se contenter du coup d'œil.
* **Effet sur l'abandon :** Très faible. Le "Lire la suite" est un standard du web et demande un effort minime.

### 2. Atténuation du Biais de Sélection des prompts (Selection Bias)
La base de données souffre d'un biais de distribution car seule une fraction d'utilisateurs "kicks" le système avec des prompts véritablement créatifs, tandis que la majorité pose des questions superficielles. 
* **Solution fonctionnelle :** Introduire un bouton bien visible **« Surprenez-moi » (Surprise Me)** ou **« Mode Challenge »** au-dessus de la barre de saisie, qui préremplit le champ avec des prompts tirés d'une banque préétablie couvrant uniformément un spectre d'instructions créatives, de raisonnement logique mathématique et d'écriture contrainte. 
* **Effet sur l'abandon :** Négatif (c'est-à-dire positif pour la rétention). Cela soulage l'angoisse de la page blanche pour les utilisateurs "curieux mais sans idée", abaissant la charge mentale (frictionless UX).

### 3. Atténuation du Biais de Position A/B (Position / Primacy Bias)
La littérature montre (notamment Wang et al., 2023 dans *Large Language Model as a judge*) que les annotateurs humains et les LLM juges favorisent intrinsèquement la première option lue (Position A souvent affichée à gauche).
* **Solution :** 
  - Maintenir un **contrebalancement temporel strict** : l'ordre de présentation des modèles sur le serveur doit être rigoureusement aléatoire et suivre une distribution uniforme (50/50).
  - Activer un tracking d'eye-tracking logiciel (via le temps de hover/scroll sur mobile) pour s'assurer que l'utilisateur a physiquement fait défiler la réponse B avant que les boutons de vote ne deviennent cliquables (grisés pendant 2 secondes).
* **Effet sur l'abandon :** Modéré. Griser un bouton 2 secondes force la lecture mais peut causer une légère frustration. L'impact serait bien inférieur aux 15% tolérés.

### Références implicites à la littérature Chatbot Arena
- Zheng, L., et al. (2023). *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena*. (Met en évidence le "verbosity bias" et le "position bias" dans les évaluations pairwise).
- Singhal et al. (2023). *A long way to go: Length bias in LLM evaluations.*
