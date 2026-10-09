# Phase 0, mode B : fusion de familles différentes (QCM, jeu dev)

Exactitude en %. Répétitions : sampling seed (original option order). **1 passage** : chaque passage noté séparément, moyenne (pire–meilleur passage). **Moyenne des passages** : les distributions des passages sont moyennées en une seule décision par question, pour chaque système, modèles seuls compris (cet ensemble gratuit est donc aussi donné aux modèles seuls). Les fusions « calibrées » apprennent une température par modèle sur une moitié des questions et sont évaluées sur l'autre (validation croisée par question). IC à 95 % : bootstrap apparié par question, procédure entière réajustée à chaque tirage. Le meilleur expert est choisi séparément pour chaque protocole, sur ces mêmes données.

## ARC-Challenge (4 choix) : 100 questions × 3 passages

| système | 1 passage | moyenne des 3 passages |
| --- | --- | --- |
| Qwen/Qwen3-1.7B _colab_think768 seul | 85.0 (84.0–86.0) | 86.0 |
| ibm-granite/granite-3.3-2b-instruct _colab_think768 seul | 78.3 (77.0–79.0) | 80.0 |
| HuggingFaceTB/SmolLM3-3B _colab_think768 seul (meilleur, 1 passage ; meilleur, moyenne) | 91.0 (89.0–93.0) | 93.0 |
| google/gemma-4-E2B-it _colab_think768 seul | 91.0 (90.0–92.0) | 93.0 |
| Qwen/Qwen3-4B _colab_think768 seul (référence locale) | 91.7 (91.0–92.0) | 92.0 |
| fusion : moyenne | 91.0 (90.0–93.0) | 92.0 |
| fusion : produit | 92.7 (91.0–94.0) | 89.0 |
| fusion : pondérée par la confiance | 91.0 (90.0–93.0) | 92.0 |
| fusion : vote | 91.0 (90.0–93.0) | 92.0 |
| fusion : produit calibré | 92.0 (91.0–94.0) | 90.0 |
| fusion : mélange calibré | 91.3 (90.0–93.0) | 92.0 |
| plafond top-1 (au moins un expert a raison) | 98.3 (98.0–99.0) | — |

Gain sur le meilleur expert de chaque protocole (1 passage : HuggingFaceTB/SmolLM3-3B _colab_think768 ; moyenne : HuggingFaceTB/SmolLM3-3B _colab_think768), en points, IC 95 % :

| fusion | 1 passage | moyenne des passages |
| --- | --- | --- |
| moyenne | +0.0 [-2.7 ; +2.7] | -1.0 [-5.0 ; +2.0] |
| produit | +1.7 [-2.0 ; +5.0] | -4.0 [-9.0 ; +0.0] |
| pondérée par la confiance | +0.0 [-2.7 ; +2.7] | -1.0 [-5.0 ; +2.0] |
| vote | +0.0 [-2.7 ; +2.7] | -1.0 [-5.0 ; +2.0] |
| produit calibré | +1.0 [-1.7 ; +4.0] | -3.0 [-7.0 ; +1.0] |
| mélange calibré | +0.3 [-2.0 ; +3.3] | -1.0 [-4.0 ; +3.0] |

Gain sur chaque expert (fusions calibrées), borne basse de l'IC à 95 % unilatéral (percentile 5 %) : « bat tous les experts » n'est soutenu que si toutes sont positives.

- produit calibré, 1 passage : Qwen3-1.7B _colab_think768 +3.0, granite-3.3-2b-instruct _colab_think768 +8.7, SmolLM3-3B _colab_think768 -1.3, gemma-4-E2B-it _colab_think768 -2.0
- produit calibré, moyenne : Qwen3-1.7B _colab_think768 -1.0, granite-3.3-2b-instruct _colab_think768 +4.0, SmolLM3-3B _colab_think768 -6.0, gemma-4-E2B-it _colab_think768 -7.0
- mélange calibré, 1 passage : Qwen3-1.7B _colab_think768 +2.3, granite-3.3-2b-instruct _colab_think768 +8.3, SmolLM3-3B _colab_think768 -1.7, gemma-4-E2B-it _colab_think768 -2.7
- mélange calibré, moyenne : Qwen3-1.7B _colab_think768 +1.0, granite-3.3-2b-instruct _colab_think768 +6.0, SmolLM3-3B _colab_think768 -4.0, gemma-4-E2B-it _colab_think768 -6.0

Gain sur la référence locale (Qwen/Qwen3-4B _colab_think768) :
- produit calibré : 1 passage +0.3 [-2.7 ; +3.7], moyenne -2.0 [-6.0 ; +3.0]
- mélange calibré : 1 passage -0.3 [-3.3 ; +3.0], moyenne +0.0 [-4.0 ; +5.0]

## MMLU-Pro (10 choix) : 100 questions × 3 passages

| système | 1 passage | moyenne des 3 passages |
| --- | --- | --- |
| Qwen/Qwen3-1.7B _colab_think768 seul | 46.7 (45.0–49.0) | 49.0 |
| ibm-granite/granite-3.3-2b-instruct _colab_think768 seul | 36.7 (36.0–38.0) | 40.0 |
| HuggingFaceTB/SmolLM3-3B _colab_think768 seul | 47.0 (43.0–49.0) | 49.0 |
| google/gemma-4-E2B-it _colab_think768 seul (meilleur, 1 passage ; meilleur, moyenne) | 53.7 (51.0–57.0) | 57.0 |
| Qwen/Qwen3-4B _colab_think768 seul (référence locale) | 54.0 (53.0–55.0) | 55.0 |
| fusion : moyenne | 51.7 (49.0–55.0) | 55.0 |
| fusion : produit | 54.7 (52.0–56.0) | 56.0 |
| fusion : pondérée par la confiance | 53.7 (52.0–56.0) | 57.0 |
| fusion : vote | 51.7 (48.0–56.0) | 53.0 |
| fusion : produit calibré | 53.7 (51.0–57.0) | 54.0 |
| fusion : mélange calibré | 53.3 (51.0–55.0) | 55.0 |
| plafond top-1 (au moins un expert a raison) | 72.0 (70.0–74.0) | — |

Gain sur le meilleur expert de chaque protocole (1 passage : google/gemma-4-E2B-it _colab_think768 ; moyenne : google/gemma-4-E2B-it _colab_think768), en points, IC 95 % :

| fusion | 1 passage | moyenne des passages |
| --- | --- | --- |
| moyenne | -2.0 [-7.0 ; +3.7] | -2.0 [-9.0 ; +6.0] |
| produit | +1.0 [-2.7 ; +5.0] | -1.0 [-7.0 ; +5.0] |
| pondérée par la confiance | +0.0 [-4.7 ; +5.0] | +0.0 [-6.0 ; +6.0] |
| vote | -2.0 [-7.3 ; +4.0] | -4.0 [-12.0 ; +4.0] |
| produit calibré | +0.0 [-5.0 ; +6.0] | -3.0 [-10.0 ; +4.0] |
| mélange calibré | -0.3 [-5.0 ; +6.3] | -2.0 [-9.0 ; +5.0] |

Gain sur chaque expert (fusions calibrées), borne basse de l'IC à 95 % unilatéral (percentile 5 %) : « bat tous les experts » n'est soutenu que si toutes sont positives.

- produit calibré, 1 passage : Qwen3-1.7B _colab_think768 +1.0, granite-3.3-2b-instruct _colab_think768 +11.0, SmolLM3-3B _colab_think768 +1.3, gemma-4-E2B-it _colab_think768 -4.3
- produit calibré, moyenne : Qwen3-1.7B _colab_think768 -1.0, granite-3.3-2b-instruct _colab_think768 +7.0, SmolLM3-3B _colab_think768 -2.0, gemma-4-E2B-it _colab_think768 -9.0
- mélange calibré, 1 passage : Qwen3-1.7B _colab_think768 +1.0, granite-3.3-2b-instruct _colab_think768 +11.3, SmolLM3-3B _colab_think768 +1.3, gemma-4-E2B-it _colab_think768 -4.0
- mélange calibré, moyenne : Qwen3-1.7B _colab_think768 -1.0, granite-3.3-2b-instruct _colab_think768 +7.0, SmolLM3-3B _colab_think768 -1.0, gemma-4-E2B-it _colab_think768 -8.0

Gain sur la référence locale (Qwen/Qwen3-4B _colab_think768) :
- produit calibré : 1 passage -0.3 [-8.7 ; +8.7], moyenne -1.0 [-10.0 ; +8.0]
- mélange calibré : 1 passage -0.7 [-8.3 ; +9.3], moyenne +0.0 [-10.0 ; +10.0]
