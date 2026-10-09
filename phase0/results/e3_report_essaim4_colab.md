# E3 : vote pondéré et certificat d'arrêt (GSM8K, rejeu, essaim4_colab)

Poids appris sur **dev** (exactitude de chaque pair), appliqués tels quels sur **test** (200 questions). IC 95 % : bootstrap apparié par question.

Deux pairs faux donnent la même réponse fausse avec une probabilité c = 7/79 = 0.089 sur dev, soit K - 1 = 11 réponses fausses « équiprobables ».

| pair | exactitude dev | poids log-odds (2 classes) | poids log-odds (K classes) |
| --- | --- | --- | --- |
| Qwen/Qwen3-1.7B | 68.0 | 0.75 | 3.18 |
| ibm-granite/granite-3.3-2b-instruct | 69.0 | 0.80 | 3.22 |
| HuggingFaceTB/SmolLM3-3B | 82.0 | 1.52 | 3.94 |
| google/gemma-4-E2B-it | 70.0 | 0.85 | 3.27 |

| règle (test) | exactitude | gain sur le meilleur pair de dev | gain sur la référence |
| --- | --- | --- | --- |
| meilleur pair de dev seul (HuggingFaceTB/SmolLM3-3B) | 86.5 | — | -4.5 [-10.0 ; +1.0] |
| vote simple | 89.0 | +2.5 [-1.5 ; +6.5] | -2.0 [-6.5 ; +2.0] |
| vote pondéré log-odds (2 classes) | 89.0 | +2.5 [+0.5 ; +5.0] | -2.0 [-6.5 ; +2.5] |
| vote pondéré log-odds (K classes) | 89.0 | +2.5 [+0.0 ; +5.0] | -2.0 [-7.0 ; +3.0] |
| Qwen/Qwen3-4B seul (référence) | 91.0 | | |

## Certificat d'arrêt (rejeu des temps de calcul mesurés)

Les réponses arrivent dans l'ordre des temps de calcul mesurés de chaque pair. On décide dès que la réponse en tête ne peut plus être rattrapée par les pairs manquants. La décision est identique à celle de l'attente de tous (vérifié ci-dessous).

| règle | pairs attendus (moy.) | temps de décision moy. (s) | attente de tous moy. (s) | décisions identiques |
| --- | --- | --- | --- | --- |
| vote | 3.31 / 4 | 4.64 | 5.13 | 200/200 |
| logodds | 2.54 / 4 | 4.33 | 5.13 | 200/200 |
| logodds_K | 2.54 / 4 | 4.33 | 5.13 | 200/200 |
