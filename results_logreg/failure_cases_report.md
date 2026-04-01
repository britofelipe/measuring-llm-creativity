# Failure Cases Analysis (No Re-Training)

scored_parquet: outputs_logreg2/comparia_reactions_with_metrics.parquet
coefficients_csv: outputs_logreg2/logreg_coefficients.csv
metrics_csv: outputs_logreg2/logreg_train_test_metrics.csv
threshold=0.5650 (mode=train_f1), intercept=-2.401897

## Summary
- total rows in parquet: 2000
- eligible rows (all required features present): 1914
- positive rate (creative=True) on eligible rows: 0.0690
- false positives: 423
- false negatives: 68

## Consistency Check
- max |recomputed_prob - creativity_index_logreg|: 0.00000000
- mean |recomputed_prob - creativity_index_logreg|: 0.00000000

## Top 2 False Positives
### FP #1 | id=308403 | question_id=057342097fda484384ab42cb34d5516f-dea127ce70d74677b711842968bb287c-4 | msg_index=9
creative=False | pred=True | prob=0.8485 | threshold=0.5650 | model=gpt-5-mini

Question:
le texte va être afficher. Il doit donner envie aux élèves de venir écouter en donnant quelques données théoriques

Response:
Viens écouter un médiateur : culture = savoirs, rites, valeurs. Crée, comprends, réussis !

Conversation excerpt:
[source: conversation_b, turns 7..9]
  - [7] assistant: Viens écouter un médiateur culturel : découvre, crée, partage, booste ton avenir pro !
  - [8] user: le texte va être afficher. Il doit donner envie aux élèves de venir écouter en donnant quelques données théoriques
  - [9] assistant: Viens écouter un médiateur : culture = savoirs, rites, valeurs. Crée, comprends, réussis !  <== msg_index

Top feature contributions (coef * feature):
- novelty_ngram_rarity_norm: coef=+1.3641, value=0.9282, contrib=+1.2661
- novelty_distinct_n_norm: coef=+1.2329, value=0.7272, contrib=+0.8965
- surprise_prompt_response_distance_norm: coef=+0.9494, value=0.8219, contrib=+0.7804
- value_local_coherence_norm: coef=+0.8413, value=0.6965, contrib=+0.5859
- surprise_distance_to_neighbors_norm: coef=+0.8120, value=0.6933, contrib=+0.5630

### FP #2 | id=317475 | question_id=51318419cec04cd5bc33a86a39850011-3c58357d48a94034b1a7c22879435b42-1 | msg_index=3
creative=False | pred=True | prob=0.8415 | threshold=0.5650 | model=aya-expanse-32b

Question:
Ottimo

Response:
Sono contenta che tu lo trovi utile! Questo piano di studi offre una struttura solida per uno studente di magistrale che desidera specializzarsi nel Natural Language Processing. Ricorda che la flessibilità è importante, quindi sentiti libero di adattarlo alle tue esigenze e interessi specifici. Se hai altre domande o desideri approfondire alcuni aspetti, non esitare a chiedere. Posso fornirti ulteriori dettagli sui corsi, suggerire risorse aggiuntive o aiutarti a esplorare potenziali percorsi di carriera in questo campo entusiasmante.

Conversation excerpt:
[source: conversation_b, turns 1..3]
  - [1] assistant: ## Corso di Studio per Magistrale in Natural Language Processing Questo piano di studi è progettato per uno studente di magistrale con una base di Python che desidera specializzarsi nel Natural Language Processing (NLP). **Obiettivi:** * Fornire una solida comprensione dei pri...
  - [2] user: Ottimo
  - [3] assistant: Sono contenta che tu lo trovi utile! Questo piano di studi offre una struttura solida per uno studente di magistrale che desidera specializzarsi nel Natural Language Processing. Ricorda che la flessibilità è importante, quindi sentiti libero di adattarlo alle tue esigenze e in...  <== msg_index

Top feature contributions (coef * feature):
- novelty_ngram_rarity_norm: coef=+1.3641, value=0.9217, contrib=+1.2573
- novelty_distinct_n_norm: coef=+1.2329, value=0.7272, contrib=+0.8965
- surprise_prompt_response_distance_norm: coef=+0.9494, value=0.8809, contrib=+0.8364
- surprise_distance_to_neighbors_norm: coef=+0.8120, value=0.7183, contrib=+0.5833
- value_local_coherence_norm: coef=+0.8413, value=0.4304, contrib=+0.3621

## Top 2 False Negatives
### FN #1 | id=264392 | question_id=d899fffeba754d4e8e8567def34a3aaa-b52e86816bed4db78843357d25981470-0 | msg_index=1
creative=True | pred=False | prob=0.2090 | threshold=0.5650 | model=llama-3.1-405b

Question:
proposer un nom pour la création d'une entreprise spécialisée : dans les travaux de rénovation, de conseils et d'audits, dans la proposition de solutions techniques écologiques

Response:
Voici quelques suggestions de noms pour votre entreprise spécialisée dans les travaux de rénovation, de conseils et d'audits, ainsi que dans la proposition de solutions techniques écologiques : 1. **EcoRénov** : un nom qui combine les notions de rénovation et d'écologie. 2. **Verdi**: un nom qui évoque la couleur verte et la notion de développement durable. 3. **RénoVert** : un nom qui associe la rénovation à l'écologie et au développement durable. 4. **Eco Conseil** : un nom simple et clair qui met en avant la mission de conseil et d'audit de l'entreprise. 5. **GreenBuild** : un nom qui évoque la construction et la rénovation écologique. 6. **Symbiose**: un nom qui suggère une relation harmonieuse entre l'homme et l'environnement. 7. **EcoSolutions**: un nom qui met en avant la proposition de solutions écologiques. 8. **Rénov'Eco** : un nom qui combine les notions de rénovation et d'écologie. 9. **Clima Conseil** : un nom qui met en avant la mission de conseil et d'audit en matière de climat et d'environnement. 10. **TerraVerde** : un nom qui évoque la Terre et la couleur verte, symbole de l'écologie. N'oubliez pas de vérifier la disponibilité du nom choisi et de vous assurer qu'il n'est pas déjà utilisé par une autre entreprise. Il est également important de choisir un nom qui reflète les valeurs et la mission de votre entreprise.

Conversation excerpt:
[source: conversation_a, turns 0..1]
  - [0] user: proposer un nom pour la création d'une entreprise spécialisée : dans les travaux de rénovation, de conseils et d'audits, dans la proposition de solutions techniques écologiques
  - [1] assistant: Voici quelques suggestions de noms pour votre entreprise spécialisée dans les travaux de rénovation, de conseils et d'audits, ainsi que dans la proposition de solutions techniques écologiques : 1. **EcoRénov** : un nom qui combine les notions de rénovation et d'écologie. 2. **...  <== msg_index

Top feature contributions (coef * feature):
- value_rouge_l_prompt_response_norm: coef=-1.4702, value=0.6508, contrib=-0.9568
- novelty_semantic_distance_centroid_norm: coef=-1.0824, value=0.4109, contrib=-0.4447
- novelty_ngram_rarity_norm: coef=+1.3641, value=0.2943, contrib=+0.4014
- surprise_prompt_response_distance_norm: coef=+0.9494, value=0.3942, contrib=+0.3743
- surprise_distance_to_neighbors_norm: coef=+0.8120, value=0.4274, contrib=+0.3471

### FN #2 | id=336716 | question_id=e96139ccb26a4c78b8a6f9c2f72cbf23-91ceb9013f4e4bfe8c6d61767f67c047-0 | msg_index=1
creative=True | pred=False | prob=0.2247 | threshold=0.5650 | model=llama-4-scout

Question:
redige un menu complet avec les recettes pour un fan de Dragon ball

Response:
Un menu complet pour un fan de Dragon Ball ! Voici une proposition de menu avec des recettes inspirées de l'univers de Dragon Ball : **Entrées** 1. **Capsules de Shenron** (raviolis chinois) * 12 raviolis de porc et légumes * Sauce de soja, vinaigre de riz et huile de sésame 2. **Kame House** (salade de fruits de mer) * Mélange de crevettes, moules, calamars et fruits (comme des pommes et des poires) * Sauce légère au citron et au gingembre **Plats principaux** 1. **Le plat du Super Saiyan** (Poulet au curry et au riz) * Poulet cuit au curry avec des épices et des herbes * Riz basmati et légumes (carottes, pois et oignons) 2. **Le Ki de Goku** (Boeuf grillé avec des légumes) * Boeuf grillé avec des légumes (poivrons, oignons, champignons) * Sauce teriyaki et riz **Desserts** 1. **Les boules de Ki** (boulettes de riz gluant) * Boulettes de riz gluant fourrées de pâte de sésame ou de haricots azuki 2. **Le gâteau d'anniversaire de Gohan** (gâteau au chocolat) * Gâteau au chocolat avec des décorations de chiffres (comme le nombre 10) **Boissons** 1. **La boisson de Senzu** (thé vert japonais) * Thé vert japonais servi avec des biscuits japonais 2. **Le ki de vie** (smoothie de fruits) * Mélange de fruits (comme des fraises, des bananes et des oranges) **Recettes détaillées** ### Capsules de Shenron (raviolis chinois) Ingrédients : * 200g de farine * 100g de porc haché * 1 oignon émincé * 1 cuillère à soupe de sauce de soja * 1 cuillère à soupe de vinaigre de riz * 1 cuillère à soupe d'huile de sésame Préparation : 1. Préparez la pâte à raviolis. 2. Mélangez le porc haché, l...

Conversation excerpt:
[source: conversation_a, turns 0..1]
  - [0] user: redige un menu complet avec les recettes pour un fan de Dragon ball
  - [1] assistant: Un menu complet pour un fan de Dragon Ball ! Voici une proposition de menu avec des recettes inspirées de l'univers de Dragon Ball : **Entrées** 1. **Capsules de Shenron** (raviolis chinois) * 12 raviolis de porc et légumes * Sauce de soja, vinaigre de riz et huile de sésame 2...  <== msg_index

Top feature contributions (coef * feature):
- value_rouge_l_prompt_response_norm: coef=-1.4702, value=0.6162, contrib=-0.9059
- novelty_ngram_rarity_norm: coef=+1.3641, value=0.5642, contrib=+0.7695
- novelty_semantic_distance_centroid_norm: coef=-1.0824, value=0.7095, contrib=-0.7680
- surprise_distance_to_neighbors_norm: coef=+0.8120, value=0.4866, contrib=+0.3951
- value_local_coherence_norm: coef=+0.8413, value=0.4326, contrib=+0.3639
