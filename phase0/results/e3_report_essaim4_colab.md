# E3 : vote pondéré et certificat d'arrêt (GSM8K, rejeu, essaim4_colab)

Poids appris sur **dev** (exactitude de chaque pair parmi les questions où il a donné une réponse : une abstention ne vote pas), appliqués tels quels sur **test** (200 questions, disjointes de dev, mêmes pairs et même protocole : vérifié). IC 95 % : bootstrap apparié par question.

Deux pairs qui donnent chacun une réponse fausse donnent la même avec une probabilité c = 7/32 = 0.219 sur dev, soit K - 1 = 5 réponses fausses « équiprobables ».

| pair | exactitude dev | exactitude dev parmi ses réponses | poids log-odds (2 classes) | poids log-odds (K classes) |
| --- | --- | --- | --- | --- |
| Qwen/Qwen3-1.7B | 68.0 | 79.1 | 1.33 | 2.85 |
| ibm-granite/granite-3.3-2b-instruct | 69.0 | 79.3 | 1.34 | 2.86 |
| HuggingFaceTB/SmolLM3-3B | 82.0 | 82.8 | 1.57 | 3.09 |
| google/gemma-4-E2B-it | 70.0 | 92.1 | 2.46 | 3.98 |

| règle (test) | exactitude | gain sur le meilleur pair de dev | gain sur la référence |
| --- | --- | --- | --- |
| meilleur pair de dev seul (HuggingFaceTB/SmolLM3-3B) | 86.5 | — | -4.5 [-10.0 ; +1.0] |
| vote simple | 89.0 | +2.5 [-1.5 ; +6.5] | -2.0 [-6.5 ; +2.0] |
| vote pondéré log-odds (2 classes) | 92.0 | +5.5 [+2.5 ; +9.0] | +1.0 [-3.0 ; +5.5] |
| vote pondéré log-odds (K classes) | 92.0 | +5.5 [+2.0 ; +9.0] | +1.0 [-3.5 ; +5.5] |
| Qwen/Qwen3-4B seul (référence) | 91.0 | | |

## Certificat d'arrêt (rejeu des temps de calcul mesurés)

Les réponses arrivent dans l'ordre des temps de calcul mesurés de chaque pair. On décide dès que la réponse en tête ne peut plus être rattrapée par les pairs manquants. La décision est identique à celle de l'attente de tous (vérifié ci-dessous).

| règle | pairs attendus (moy.) | temps de décision moy. (s) | attente de tous moy. (s) | décisions identiques |
| --- | --- | --- | --- | --- |
| vote | 3.31 / 4 | 4.64 | 5.13 | 200/200 |
| logodds | 3.26 / 4 | 4.64 | 5.13 | 200/200 |
| logodds_K | 3.26 / 4 | 4.64 | 5.13 | 200/200 |
