# Exercice 3 — Biais systématiques dans les données

## Exercice 3.1 — Quantification du biais de longueur

**Statistiques descriptives (152 842 matchs) :**

* **Corrélation de Spearman** entre `length_diff` et `vote_A` : **ρ = 0.1237** (p-value < 2e-16)
* **Coefficient de régression logistique (BT)** pour `length_diff_norm` : **+0.1428**

**Interprétation :**

La corrélation positive et significative indique l’existence d’un **biais de longueur** : une réponse plus longue tend à être choisie plus fréquemment. Toutefois, l’ampleur de cet effet reste **modérée**. Le coefficient de régression correspond à une augmentation d’environ **15 % des chances relatives** (odds) de victoire pour une augmentation d’un écart-type de la différence de longueur.

La grande taille de l’échantillon explique en partie la forte significativité statistique observée.

**Impact sur le classement Bradley-Terry :**

L’ajout de la longueur comme variable de contrôle entraîne des changements de rang mesurables, principalement dans les positions intermédiaires :

* `glm-5` : 39 → 52
* `grok-3-mini-beta` : 29 → 39
* `trinity-large-preview` : 71 → 62

Ces résultats suggèrent que la verbosité peut influencer partiellement les performances observées, sans remettre en cause la stabilité globale des modèles les mieux classés.

---

## Exercice 3.2 — Biais de position A/B

Tests du Chi-deux sur la variable `model_pos` (A ou B) dans `comparia-reactions`.

**Résultats principaux :**

| Variable  | p-value  | Taux A  | Taux B  | Conclusion                        |
| --------- | -------- | ------- | ------- | --------------------------------- |
| liked     | 0.003    | 67.27 % | 68.18 % | Différence significative (faible) |
| useful    | 2.48e-06 | 23.14 % | 24.47 % | Différence significative (faible) |
| creative  | 0.019    | 6.55 %  | 6.94 %  | Différence significative (faible) |
| incorrect | 0.319    | 11.13 % | 10.92 % | Non significatif                  |

**Interprétation :**

Un **biais de position** est statistiquement détectable : la position **B** reçoit légèrement plus de réactions positives. Cependant, les écarts observés restent **faibles** (environ 0.5 à 1.3 points de pourcentage).

L’absence d’effet significatif pour `incorrect` suggère que les jugements factuels sont moins sensibles à la position que les évaluations plus subjectives.

---

## Exercice 3.3 — Protocole d’annotation amélioré

Objectif : réduire les biais de longueur, de position et de sélection sans augmenter fortement la charge cognitive.

### 1. Réduction du biais de longueur

**Troncature visuelle symétrique** lorsque l’écart de longueur dépasse un seuil (ex. 15 %), avec bouton « Lire la suite ».
→ Limite l’effet visuel initial sans restreindre l’accès au contenu.

### 2. Réduction du biais de sélection

**Bouton « Surprenez-moi »** proposant automatiquement des prompts variés.
→ Augmente la diversité des tâches et réduit la dépendance aux initiatives des utilisateurs.

### 3. Réduction du biais de position

* **Randomisation stricte** de l’ordre A/B
* **Activation des boutons de vote après exposition minimale** aux deux réponses

→ Encourage une évaluation plus équilibrée avec un impact limité sur l’abandon.


