# Phase 0, mode B : fusion de familles différentes (QCM, jeu dev)

Exactitude en %. Répétitions : balanced option order (balanced_perm). **1 passage** : chaque passage noté séparément, moyenne (pire–meilleur passage). **Moyenne des passages** : les distributions des passages sont moyennées en une seule décision par question, pour chaque système, modèles seuls compris (cet ensemble gratuit est donc aussi donné aux modèles seuls). Les fusions « calibrées » apprennent une température par modèle sur une moitié des questions et sont évaluées sur l'autre (validation croisée par question). IC à 95 % : bootstrap apparié par question, procédure entière réajustée à chaque tirage. Le meilleur expert est choisi séparément pour chaque protocole, sur ces mêmes données.

## ARC-Challenge (4 choix) : 300 questions × 5 passages

| système | 1 passage | moyenne des 5 passages |
| --- | --- | --- |
| Qwen/Qwen3-1.7B _gpu seul | 74.5 (70.0–77.3) | 75.7 |
| ibm-granite/granite-3.3-2b-instruct _gpu seul | 72.0 (70.3–74.0) | 73.7 |
| HuggingFaceTB/SmolLM3-3B _colab seul | 79.7 (77.7–81.7) | 82.7 |
| google/gemma-4-E2B-it _gpu seul (meilleur, 1 passage ; meilleur, moyenne) | 83.2 (82.0–84.0) | 85.7 |
| Qwen/Qwen3-4B _colab seul (référence locale) | 88.0 (87.0–89.0) | 90.0 |
| fusion : moyenne | 83.4 (81.0–85.3) | 85.7 |
| fusion : produit | 84.3 (82.7–86.7) | 87.7 |
| fusion : pondérée par la confiance | 83.7 (81.7–86.3) | 86.7 |
| fusion : vote | 83.3 (80.3–86.0) | 85.7 |
| fusion : produit calibré | 84.5 (83.0–85.0) | 88.0 |
| fusion : mélange calibré | 84.5 (83.7–85.3) | 87.7 |
| plafond top-1 (au moins un expert a raison) | 94.6 (93.3–95.7) | — |

Gain sur le meilleur expert de chaque protocole (1 passage : google/gemma-4-E2B-it _gpu ; moyenne : google/gemma-4-E2B-it _gpu), en points, IC 95 % :

| fusion | 1 passage | moyenne des passages |
| --- | --- | --- |
| moyenne | +0.2 [-2.5 ; +2.9] | +0.0 [-3.7 ; +3.7] |
| produit | +1.1 [-1.5 ; +3.7] | +2.0 [-1.0 ; +5.3] |
| pondérée par la confiance | +0.5 [-2.3 ; +3.1] | +1.0 [-2.3 ; +4.3] |
| vote | +0.1 [-2.6 ; +2.7] | +0.0 [-3.3 ; +3.7] |
| produit calibré | +1.3 [-0.9 ; +3.8] | +2.3 [-1.3 ; +5.3] |
| mélange calibré | +1.3 [-1.1 ; +3.7] | +2.0 [-1.7 ; +5.0] |

Gain sur chaque expert (fusions calibrées), borne basse de l'IC à 95 % unilatéral (percentile 5 %) : « bat tous les experts » n'est soutenu que si toutes sont positives.

- produit calibré, 1 passage : Qwen3-1.7B _gpu +7.6, granite-3.3-2b-instruct _gpu +10.1, SmolLM3-3B _colab +2.8, gemma-4-E2B-it _gpu -0.6
- produit calibré, moyenne : Qwen3-1.7B _gpu +8.3, granite-3.3-2b-instruct _gpu +10.3, SmolLM3-3B _colab +2.0, gemma-4-E2B-it _gpu -0.7
- mélange calibré, 1 passage : Qwen3-1.7B _gpu +7.6, granite-3.3-2b-instruct _gpu +10.1, SmolLM3-3B _colab +2.8, gemma-4-E2B-it _gpu -0.7
- mélange calibré, moyenne : Qwen3-1.7B _gpu +8.0, granite-3.3-2b-instruct _gpu +10.0, SmolLM3-3B _colab +1.7, gemma-4-E2B-it _gpu -1.0

Gain sur la référence locale (Qwen/Qwen3-4B _colab) :
- produit calibré : 1 passage -3.5 [-6.1 ; -0.6], moyenne -2.0 [-6.3 ; +1.7]
- mélange calibré : 1 passage -3.5 [-6.3 ; -0.5], moyenne -2.3 [-6.3 ; +1.3]

## MMLU-Pro (10 choix) : 300 questions × 5 passages

| système | 1 passage | moyenne des 5 passages |
| --- | --- | --- |
| Qwen/Qwen3-1.7B _gpu seul | 28.3 (26.0–31.7) | 33.0 |
| ibm-granite/granite-3.3-2b-instruct _gpu seul | 28.5 (27.3–31.7) | 30.3 |
| HuggingFaceTB/SmolLM3-3B _colab seul | 29.3 (25.7–34.0) | 35.3 |
| google/gemma-4-E2B-it _gpu seul (meilleur, 1 passage ; meilleur, moyenne) | 32.5 (29.0–35.0) | 36.7 |
| Qwen/Qwen3-4B _colab seul (référence locale) | 39.4 (37.7–41.7) | 43.7 |
| fusion : moyenne | 33.0 (29.3–38.3) | 37.3 |
| fusion : produit | 33.1 (31.0–37.0) | 38.3 |
| fusion : pondérée par la confiance | 32.9 (30.3–37.7) | 38.0 |
| fusion : vote | 32.9 (29.0–36.7) | 37.7 |
| fusion : produit calibré | 33.5 (30.7–37.0) | 39.3 |
| fusion : mélange calibré | 33.9 (30.7–38.7) | 40.0 |
| plafond top-1 (au moins un expert a raison) | 57.2 (54.3–60.3) | — |

Gain sur le meilleur expert de chaque protocole (1 passage : google/gemma-4-E2B-it _gpu ; moyenne : google/gemma-4-E2B-it _gpu), en points, IC 95 % :

| fusion | 1 passage | moyenne des passages |
| --- | --- | --- |
| moyenne | +0.5 [-2.2 ; +3.0] | +0.7 [-4.0 ; +5.3] |
| produit | +0.6 [-2.2 ; +3.5] | +1.7 [-2.7 ; +6.7] |
| pondérée par la confiance | +0.4 [-2.3 ; +3.0] | +1.3 [-3.0 ; +6.0] |
| vote | +0.4 [-2.2 ; +2.9] | +1.0 [-3.7 ; +5.3] |
| produit calibré | +0.9 [-1.9 ; +3.4] | +2.7 [-2.7 ; +7.7] |
| mélange calibré | +1.4 [-1.3 ; +3.9] | +3.3 [-2.0 ; +8.3] |

Gain sur chaque expert (fusions calibrées), borne basse de l'IC à 95 % unilatéral (percentile 5 %) : « bat tous les experts » n'est soutenu que si toutes sont positives.

- produit calibré, 1 passage : Qwen3-1.7B _gpu +2.7, granite-3.3-2b-instruct _gpu +2.5, SmolLM3-3B _colab +1.8, gemma-4-E2B-it _gpu -1.4
- produit calibré, moyenne : Qwen3-1.7B _gpu +1.3, granite-3.3-2b-instruct _gpu +5.0, SmolLM3-3B _colab -0.3, gemma-4-E2B-it _gpu -2.0
- mélange calibré, 1 passage : Qwen3-1.7B _gpu +3.1, granite-3.3-2b-instruct _gpu +2.9, SmolLM3-3B _colab +2.3, gemma-4-E2B-it _gpu -0.9
- mélange calibré, moyenne : Qwen3-1.7B _gpu +2.0, granite-3.3-2b-instruct _gpu +5.7, SmolLM3-3B _colab +0.7, gemma-4-E2B-it _gpu -1.0

Gain sur la référence locale (Qwen/Qwen3-4B _colab) :
- produit calibré : 1 passage -5.9 [-9.6 ; -2.5], moyenne -4.3 [-11.0 ; +1.7]
- mélange calibré : 1 passage -5.5 [-9.3 ; -1.9], moyenne -3.7 [-10.0 ; +2.0]
