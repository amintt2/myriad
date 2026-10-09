# Phase 0, mode B : fusion de familles différentes (QCM, jeu test)

Exactitude en %. Répétitions : balanced option order (balanced_perm). **1 passage** : chaque passage noté séparément, moyenne (pire–meilleur passage). **Moyenne des passages** : les distributions des passages sont moyennées en une seule décision par question, pour chaque système, modèles seuls compris (cet ensemble gratuit est donc aussi donné aux modèles seuls). Les fusions « calibrées » apprennent une température par modèle sur le jeu **dev**, puis l'appliquent telle quelle (méthode figée). IC à 95 % : bootstrap apparié par question. Le meilleur expert est choisi séparément pour chaque protocole, sur ces mêmes données.

## ARC-Challenge (4 choix) : 300 questions × 5 passages

| système | 1 passage | moyenne des 5 passages |
| --- | --- | --- |
| Qwen/Qwen3-1.7B _gpu seul | 75.6 (73.0–77.7) | 79.3 |
| ibm-granite/granite-3.3-2b-instruct _gpu seul | 71.5 (70.7–72.3) | 73.3 |
| HuggingFaceTB/SmolLM3-3B _colab seul (meilleur, moyenne) | 79.3 (78.3–80.3) | 82.3 |
| google/gemma-4-E2B-it _gpu seul (meilleur, 1 passage) | 80.9 (79.0–83.0) | 82.3 |
| Qwen/Qwen3-4B _colab seul (référence locale) | 87.9 (86.3–89.3) | 91.0 |
| fusion : moyenne | 83.9 (83.3–84.3) | 86.0 |
| fusion : produit | 83.7 (82.3–84.7) | 86.3 |
| fusion : pondérée par la confiance | 84.0 (83.0–84.7) | 85.7 |
| fusion : vote | 83.4 (83.0–84.0) | 85.7 |
| fusion : produit calibré | 84.0 (83.0–85.0) | 86.0 |
| fusion : mélange calibré | 84.7 (83.3–85.3) | 86.0 |
| plafond top-1 (au moins un expert a raison) | 94.5 (94.0–95.0) | — |

Gain sur le meilleur expert de chaque protocole (1 passage : google/gemma-4-E2B-it _gpu ; moyenne : HuggingFaceTB/SmolLM3-3B _colab), en points, IC 95 % :

| fusion | 1 passage | moyenne des passages |
| --- | --- | --- |
| moyenne | +3.0 [+0.6 ; +5.4] | +3.7 [+0.3 ; +7.3] |
| produit | +2.7 [+0.1 ; +5.2] | +4.0 [+0.7 ; +8.0] |
| pondérée par la confiance | +3.1 [+0.7 ; +5.5] | +3.3 [-0.3 ; +7.0] |
| vote | +2.5 [+0.1 ; +4.8] | +3.3 [+0.3 ; +6.7] |
| produit calibré | +3.1 [+0.5 ; +5.5] | +3.7 [+0.3 ; +7.0] |
| mélange calibré | +3.7 [+1.2 ; +6.2] | +3.7 [+0.7 ; +7.3] |

Gain sur chaque expert (fusions calibrées), borne basse de l'IC à 95 % unilatéral (percentile 5 %) : « bat tous les experts » n'est soutenu que si toutes sont positives.

- produit calibré, 1 passage : Qwen3-1.7B _gpu +6.1, granite-3.3-2b-instruct _gpu +9.7, SmolLM3-3B _colab +2.5, gemma-4-E2B-it _gpu +0.9
- produit calibré, moyenne : Qwen3-1.7B _gpu +3.3, granite-3.3-2b-instruct _gpu +8.7, SmolLM3-3B _colab +1.0, gemma-4-E2B-it _gpu +0.3
- mélange calibré, 1 passage : Qwen3-1.7B _gpu +6.7, granite-3.3-2b-instruct _gpu +10.3, SmolLM3-3B _colab +3.1, gemma-4-E2B-it _gpu +1.5
- mélange calibré, moyenne : Qwen3-1.7B _gpu +3.3, granite-3.3-2b-instruct _gpu +9.0, SmolLM3-3B _colab +1.0, gemma-4-E2B-it _gpu +0.7

Gain sur la référence locale (Qwen/Qwen3-4B _colab) :
- produit calibré : 1 passage -3.9 [-6.9 ; -1.1], moyenne -5.0 [-9.0 ; -1.3]
- mélange calibré : 1 passage -3.2 [-6.3 ; -0.5], moyenne -5.0 [-8.7 ; -1.3]

Températures figées (apprises sur dev) : Qwen3-1.7B _gpu 5.66, granite-3.3-2b-instruct _gpu 3.90, SmolLM3-3B _colab 2.00, gemma-4-E2B-it _gpu 4.53

## MMLU-Pro (10 choix) : 300 questions × 5 passages

| système | 1 passage | moyenne des 5 passages |
| --- | --- | --- |
| Qwen/Qwen3-1.7B _gpu seul | 27.7 (25.0–29.0) | 33.3 |
| ibm-granite/granite-3.3-2b-instruct _gpu seul | 27.3 (26.0–29.0) | 31.7 |
| HuggingFaceTB/SmolLM3-3B _colab seul (meilleur, moyenne) | 29.2 (27.0–32.0) | 38.7 |
| google/gemma-4-E2B-it _gpu seul (meilleur, 1 passage) | 29.8 (28.0–32.0) | 33.3 |
| Qwen/Qwen3-4B _colab seul (référence locale) | 39.1 (38.0–40.0) | 44.3 |
| fusion : moyenne | 30.7 (29.0–32.0) | 36.7 |
| fusion : produit | 30.7 (29.7–31.7) | 36.0 |
| fusion : pondérée par la confiance | 30.9 (29.3–32.0) | 37.0 |
| fusion : vote | 31.5 (30.3–33.0) | 38.3 |
| fusion : produit calibré | 31.7 (30.3–33.0) | 38.0 |
| fusion : mélange calibré | 31.9 (31.3–32.7) | 38.7 |
| plafond top-1 (au moins un expert a raison) | 53.7 (51.0–54.7) | — |

Gain sur le meilleur expert de chaque protocole (1 passage : google/gemma-4-E2B-it _gpu ; moyenne : HuggingFaceTB/SmolLM3-3B _colab), en points, IC 95 % :

| fusion | 1 passage | moyenne des passages |
| --- | --- | --- |
| moyenne | +0.9 [-1.5 ; +3.3] | -2.0 [-7.0 ; +2.7] |
| produit | +0.9 [-1.9 ; +3.4] | -2.7 [-7.7 ; +2.0] |
| pondérée par la confiance | +1.1 [-1.7 ; +3.6] | -1.7 [-6.7 ; +3.0] |
| vote | +1.7 [-0.6 ; +4.1] | -0.3 [-5.0 ; +4.3] |
| produit calibré | +1.9 [-0.8 ; +4.4] | -0.7 [-5.3 ; +3.7] |
| mélange calibré | +2.1 [-0.5 ; +4.3] | +0.0 [-4.7 ; +4.3] |

Gain sur chaque expert (fusions calibrées), borne basse de l'IC à 95 % unilatéral (percentile 5 %) : « bat tous les experts » n'est soutenu que si toutes sont positives.

- produit calibré, 1 passage : Qwen3-1.7B _gpu +1.8, granite-3.3-2b-instruct _gpu +2.0, SmolLM3-3B _colab +0.5, gemma-4-E2B-it _gpu -0.4
- produit calibré, moyenne : Qwen3-1.7B _gpu +0.7, granite-3.3-2b-instruct _gpu +2.6, SmolLM3-3B _colab -4.7, gemma-4-E2B-it _gpu +0.7
- mélange calibré, 1 passage : Qwen3-1.7B _gpu +2.1, granite-3.3-2b-instruct _gpu +2.3, SmolLM3-3B _colab +0.4, gemma-4-E2B-it _gpu -0.2
- mélange calibré, moyenne : Qwen3-1.7B _gpu +1.3, granite-3.3-2b-instruct _gpu +3.0, SmolLM3-3B _colab -4.0, gemma-4-E2B-it _gpu +1.3

Gain sur la référence locale (Qwen/Qwen3-4B _colab) :
- produit calibré : 1 passage -7.3 [-10.9 ; -4.1], moyenne -6.3 [-11.3 ; -1.3]
- mélange calibré : 1 passage -7.2 [-10.5 ; -3.9], moyenne -5.7 [-10.3 ; -1.0]

Températures figées (apprises sur dev) : Qwen3-1.7B _gpu 8.20, granite-3.3-2b-instruct _gpu 4.53, SmolLM3-3B _colab 3.36, gemma-4-E2B-it _gpu 5.25
