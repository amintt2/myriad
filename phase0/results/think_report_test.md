# Phase 0, mode B : fusion de familles différentes (QCM, jeu test)

Exactitude en %. Répétitions : sampling seed (original option order). **1 passage** : chaque passage noté séparément, moyenne (pire–meilleur passage). **Moyenne des passages** : les distributions des passages sont moyennées en une seule décision par question, pour chaque système, modèles seuls compris (cet ensemble gratuit est donc aussi donné aux modèles seuls). Les fusions « calibrées » apprennent une température par modèle sur le jeu **dev**, puis l'appliquent telle quelle (méthode figée). IC à 95 % : bootstrap apparié par question. Le meilleur expert est choisi séparément pour chaque protocole, sur ces mêmes données.

## ARC-Challenge (4 choix) : 100 questions × 3 passages

| système | 1 passage | moyenne des 3 passages |
| --- | --- | --- |
| Qwen/Qwen3-1.7B _colab_think768 seul | 89.0 (88.0–90.0) | 88.0 |
| ibm-granite/granite-3.3-2b-instruct _colab_think768 seul | 81.0 (79.0–84.0) | 82.0 |
| HuggingFaceTB/SmolLM3-3B _colab_think768 seul | 88.3 (87.0–91.0) | 89.0 |
| google/gemma-4-E2B-it _colab_think768 seul (meilleur, 1 passage ; meilleur, moyenne) | 91.3 (90.0–92.0) | 93.0 |
| Qwen/Qwen3-4B _colab_think768 seul (référence locale) | 96.0 (95.0–97.0) | 98.0 |
| fusion : moyenne | 94.0 (93.0–95.0) | 95.0 |
| fusion : produit | 93.0 (92.0–94.0) | 93.0 |
| fusion : pondérée par la confiance | 94.0 (93.0–95.0) | 95.0 |
| fusion : vote | 94.0 (93.0–95.0) | 95.0 |
| fusion : produit calibré | 93.3 (92.0–95.0) | 96.0 |
| fusion : mélange calibré | 93.0 (91.0–95.0) | 94.0 |
| plafond top-1 (au moins un expert a raison) | 98.0 (98.0–98.0) | — |

Gain sur le meilleur expert de chaque protocole (1 passage : google/gemma-4-E2B-it _colab_think768 ; moyenne : google/gemma-4-E2B-it _colab_think768), en points, IC 95 % :

| fusion | 1 passage | moyenne des passages |
| --- | --- | --- |
| moyenne | +2.7 [-0.3 ; +6.3] | +2.0 [+0.0 ; +5.0] |
| produit | +1.7 [-0.0 ; +4.0] | +0.0 [-3.0 ; +3.0] |
| pondérée par la confiance | +2.7 [-0.3 ; +6.3] | +2.0 [+0.0 ; +5.0] |
| vote | +2.7 [-0.3 ; +6.3] | +2.0 [+0.0 ; +5.0] |
| produit calibré | +2.0 [-0.3 ; +4.7] | +3.0 [+0.0 ; +7.0] |
| mélange calibré | +1.7 [-0.7 ; +4.3] | +1.0 [-2.0 ; +4.0] |

Gain sur chaque expert (fusions calibrées), borne basse de l'IC à 95 % unilatéral (percentile 5 %) : « bat tous les experts » n'est soutenu que si toutes sont positives.

- produit calibré, 1 passage : Qwen3-1.7B _colab_think768 +0.3, granite-3.3-2b-instruct _colab_think768 +7.3, SmolLM3-3B _colab_think768 +2.0, gemma-4-E2B-it _colab_think768 +0.0
- produit calibré, moyenne : Qwen3-1.7B _colab_think768 +3.0, granite-3.3-2b-instruct _colab_think768 +9.0, SmolLM3-3B _colab_think768 +3.0, gemma-4-E2B-it _colab_think768 +0.9
- mélange calibré, 1 passage : Qwen3-1.7B _colab_think768 +0.3, granite-3.3-2b-instruct _colab_think768 +7.0, SmolLM3-3B _colab_think768 +2.0, gemma-4-E2B-it _colab_think768 -0.3
- mélange calibré, moyenne : Qwen3-1.7B _colab_think768 +2.0, granite-3.3-2b-instruct _colab_think768 +6.0, SmolLM3-3B _colab_think768 +1.0, gemma-4-E2B-it _colab_think768 -2.0

Gain sur la référence locale (Qwen/Qwen3-4B _colab_think768) :
- produit calibré : 1 passage -2.7 [-6.0 ; +0.3], moyenne -2.0 [-6.0 ; +1.0]
- mélange calibré : 1 passage -3.0 [-6.3 ; +0.0], moyenne -4.0 [-9.0 ; +0.0]

Températures figées (apprises sur dev) : Qwen3-1.7B _colab_think768 4.88, granite-3.3-2b-instruct _colab_think768 5.25, SmolLM3-3B _colab_think768 2.50, gemma-4-E2B-it _colab_think768 7.07

## MMLU-Pro (10 choix) : 100 questions × 3 passages

| système | 1 passage | moyenne des 3 passages |
| --- | --- | --- |
| Qwen/Qwen3-1.7B _colab_think768 seul | 47.0 (43.0–49.0) | 50.0 |
| ibm-granite/granite-3.3-2b-instruct _colab_think768 seul | 41.3 (37.0–47.0) | 46.0 |
| HuggingFaceTB/SmolLM3-3B _colab_think768 seul | 48.3 (47.0–50.0) | 51.0 |
| google/gemma-4-E2B-it _colab_think768 seul (meilleur, 1 passage ; meilleur, moyenne) | 51.3 (50.0–53.0) | 53.0 |
| Qwen/Qwen3-4B _colab_think768 seul (référence locale) | 53.3 (51.0–57.0) | 52.0 |
| fusion : moyenne | 51.3 (49.0–53.0) | 54.0 |
| fusion : produit | 52.0 (51.0–54.0) | 54.0 |
| fusion : pondérée par la confiance | 53.0 (51.0–56.0) | 57.0 |
| fusion : vote | 51.0 (49.0–53.0) | 55.0 |
| fusion : produit calibré | 52.0 (50.0–53.0) | 53.0 |
| fusion : mélange calibré | 51.7 (50.0–53.0) | 50.0 |
| plafond top-1 (au moins un expert a raison) | 74.7 (73.0–78.0) | — |

Gain sur le meilleur expert de chaque protocole (1 passage : google/gemma-4-E2B-it _colab_think768 ; moyenne : google/gemma-4-E2B-it _colab_think768), en points, IC 95 % :

| fusion | 1 passage | moyenne des passages |
| --- | --- | --- |
| moyenne | +0.0 [-5.3 ; +5.7] | +1.0 [-5.0 ; +8.0] |
| produit | +0.7 [-4.7 ; +6.0] | +1.0 [-5.0 ; +8.0] |
| pondérée par la confiance | +1.7 [-3.3 ; +7.0] | +4.0 [-2.0 ; +10.0] |
| vote | -0.3 [-6.0 ; +5.7] | +2.0 [-4.0 ; +8.0] |
| produit calibré | +0.7 [-5.0 ; +6.3] | +0.0 [-7.0 ; +8.0] |
| mélange calibré | +0.3 [-5.0 ; +6.0] | -3.0 [-10.0 ; +4.0] |

Gain sur chaque expert (fusions calibrées), borne basse de l'IC à 95 % unilatéral (percentile 5 %) : « bat tous les experts » n'est soutenu que si toutes sont positives.

- produit calibré, 1 passage : Qwen3-1.7B _colab_think768 -0.3, granite-3.3-2b-instruct _colab_think768 +5.3, SmolLM3-3B _colab_think768 -1.3, gemma-4-E2B-it _colab_think768 -4.0
- produit calibré, moyenne : Qwen3-1.7B _colab_think768 -4.0, granite-3.3-2b-instruct _colab_think768 -1.0, SmolLM3-3B _colab_think768 -4.0, gemma-4-E2B-it _colab_think768 -6.0
- mélange calibré, 1 passage : Qwen3-1.7B _colab_think768 -0.7, granite-3.3-2b-instruct _colab_think768 +5.0, SmolLM3-3B _colab_think768 -1.3, gemma-4-E2B-it _colab_think768 -4.0
- mélange calibré, moyenne : Qwen3-1.7B _colab_think768 -7.0, granite-3.3-2b-instruct _colab_think768 -3.0, SmolLM3-3B _colab_think768 -7.0, gemma-4-E2B-it _colab_think768 -8.0

Gain sur la référence locale (Qwen/Qwen3-4B _colab_think768) :
- produit calibré : 1 passage -1.3 [-9.3 ; +6.0], moyenne +1.0 [-8.0 ; +10.0]
- mélange calibré : 1 passage -1.7 [-9.7 ; +6.0], moyenne -2.0 [-11.0 ; +6.0]

Températures figées (apprises sur dev) : Qwen3-1.7B _colab_think768 5.25, granite-3.3-2b-instruct _colab_think768 5.66, SmolLM3-3B _colab_think768 3.12, gemma-4-E2B-it _colab_think768 7.07
