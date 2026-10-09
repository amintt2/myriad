# E4 : un essaim de petits modèles de familles différentes contre des modèles bien plus gros

Chaque modèle répond seul, une fois (glouton), via llama-server comme dans l'app. Les décisions de l'essaim sont calculées hors ligne (un aller-retour par requête). Poids appris sur **dev**, résultats sur **test**. IC 95 % : intervalle du score de Tango pour la différence appariée. Verdicts : tests exacts non conditionnels (essaim/stats.py) ; équivalence (TOST) : marge ±2.0 points, deux tests unilatéraux à 5 % ; supériorité ou infériorité : test unilatéral à 2,5 %. Poids : exactitude parmi les réponses données (une abstention ne vote pas), collision entre réponses fausses données.

## gsm8k (300 questions test)

Collision des réponses fausses sur dev : c = 0.460.

| système | exactitude test (%) | poids (wvote) |
| --- | --- | --- |
| Qwen/Qwen3.5-4B seul (Qwen) | 91.7 | 2.96 |
| google/gemma-4-E4B-it seul (Google) | 94.3 | 3.21 |
| ibm-granite/granite-4.2-3b seul (IBM) | 88.0 | 3.48 |
| HuggingFaceTB/SmolLM3-3B seul (Hugging Face) | 86.0 | 2.35 |
| mistralai/Ministral-3-3B-Instruct-2512 seul (Mistral) | 90.7 | 2.66 |
| microsoft/Phi-4-mini-instruct seul (Microsoft) | 91.7 | 2.86 |
| allenai/OLMo-2-0425-1B-Instruct seul (AllenAI) | 59.7 | 1.72 |
| Qwen/Qwen3.5-2B seul (hors essaim) | 72.7 | |
| google/gemma-4-E2B-it seul (hors essaim) | 91.7 | |
| **essaim 7 familles, vote** | **95.3** | |
| **essaim 7 familles, vote pondéré** | **96.0** | |
| Qwen/Qwen3.5-9B seul (référence, 9.0 G) | 95.7 | |
| google/gemma-4-12B-it seul (référence, 12.0 G) | 97.3 | |
| mistralai/Ministral-3-14B-Instruct-2512 seul (référence, 14.0 G) | 94.0 | |
| Qwen/Qwen3.8-27B seul (référence, 27.0 G) | 97.7 | |
| oracle (au moins un pair juste) | 98.0 | |

| essaim (wvote) moins | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |
| --- | --- | --- | --- |
| meilleur pair de dev (google/gemma-4-E4B-it) | +1.7 [-0.4 ; +4.2] | 0.002 ; 0.444 | non inférieur (-2) |
| Qwen/Qwen3.5-9B | +0.3 [-1.9 ; +2.7] | 0.023 ; 0.074 | non inférieur (-2) |
| google/gemma-4-12B-it | -1.3 [-4.2 ; +1.3] | 0.328 ; 0.009 | indéterminé |
| mistralai/Ministral-3-14B-Instruct-2512 | +2.0 [-0.3 ; +4.7] | 0.002 ; 0.606 | non inférieur (-2) |
| Qwen/Qwen3.8-27B | -1.7 [-4.2 ; +0.4] | 0.444 ; 0.002 | indéterminé |

Passage à l'échelle (vote pondéré) : moyenne sur tous les sous-ensembles de k familles, et les k meilleures familles de dev.

| k | moyenne | k meilleures |
| --- | --- | --- |
| 1 | 86.0 | 94.3 |
| 2 | 92.4 | 94.3 |
| 3 | 94.3 | 94.7 |
| 4 | 95.0 | 95.3 |
| 5 | 95.3 | 96.3 |
| 6 | 95.9 | 96.0 |
| 7 | 96.0 | 96.0 |

Certificat d'arrêt (ordre d'arrivée = temps mesurés) : 4.72 pairs attendus sur 7 en moyenne.

## math500 (250 questions test)

Collision des réponses fausses sur dev : c = 0.168.

| système | exactitude test (%) | poids (wvote) |
| --- | --- | --- |
| Qwen/Qwen3.5-4B seul (Qwen) | 81.6 | 4.21 |
| google/gemma-4-E4B-it seul (Google) | 21.2 | 5.59 |
| ibm-granite/granite-4.2-3b seul (IBM) | 68.8 | 4.48 |
| HuggingFaceTB/SmolLM3-3B seul (Hugging Face) | 70.0 | 2.73 |
| mistralai/Ministral-3-3B-Instruct-2512 seul (Mistral) | 67.6 | 3.51 |
| microsoft/Phi-4-mini-instruct seul (Microsoft) | 68.0 | 2.57 |
| allenai/OLMo-2-0425-1B-Instruct seul (AllenAI) | 20.0 | 0.58 |
| Qwen/Qwen3.5-2B seul (hors essaim) | 64.4 | |
| google/gemma-4-E2B-it seul (hors essaim) | 23.2 | |
| **essaim 7 familles, vote** | **84.0** | |
| **essaim 7 familles, vote pondéré** | **84.0** | |
| Qwen/Qwen3.5-9B seul (référence, 9.0 G) | 84.4 | |
| google/gemma-4-12B-it seul (référence, 12.0 G) | 91.2 | |
| mistralai/Ministral-3-14B-Instruct-2512 seul (référence, 14.0 G) | 70.8 | |
| Qwen/Qwen3.8-27B seul (référence, 27.0 G) | 86.8 | |
| oracle (au moins un pair juste) | 87.2 | |

| essaim (wvote) moins | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |
| --- | --- | --- | --- |
| meilleur pair de dev (Qwen/Qwen3.5-4B) | +2.4 [+0.2 ; +5.3] | 0.001 ; 0.682 | supérieur |
| Qwen/Qwen3.5-9B | -0.4 [-3.8 ; +2.9] | 0.167 ; 0.072 | indéterminé |
| google/gemma-4-12B-it | -7.2 [-11.8 ; -3.1] | 0.995 ; 0.000 | inférieur |
| mistralai/Ministral-3-14B-Instruct-2512 | +13.2 [+8.9 ; +18.3] | 0.000 ; 1.000 | supérieur |
| Qwen/Qwen3.8-27B | -2.8 [-6.8 ; +0.9] | 0.684 ; 0.007 | indéterminé |

Passage à l'échelle (vote pondéré) : moyenne sur tous les sous-ensembles de k familles, et les k meilleures familles de dev.

| k | moyenne | k meilleures |
| --- | --- | --- |
| 1 | 56.7 | 81.6 |
| 2 | 72.8 | 83.2 |
| 3 | 78.2 | 83.2 |
| 4 | 80.7 | 83.6 |
| 5 | 82.3 | 83.6 |
| 6 | 83.4 | 83.6 |
| 7 | 84.0 | 84.0 |

Certificat d'arrêt (ordre d'arrivée = temps mesurés) : 5.57 pairs attendus sur 7 en moyenne.

## arc (300 questions test)

Collision des réponses fausses sur dev : c = 0.711.

| système | exactitude test (%) | poids (wvote) |
| --- | --- | --- |
| Qwen/Qwen3.5-4B seul (Qwen) | 96.7 | 3.61 |
| google/gemma-4-E4B-it seul (Google) | 95.3 | 3.44 |
| ibm-granite/granite-4.2-3b seul (IBM) | 88.0 | 2.61 |
| HuggingFaceTB/SmolLM3-3B seul (Hugging Face) | 84.0 | 2.09 |
| mistralai/Ministral-3-3B-Instruct-2512 seul (Mistral) | 90.3 | 2.47 |
| microsoft/Phi-4-mini-instruct seul (Microsoft) | 87.7 | 2.50 |
| allenai/OLMo-2-0425-1B-Instruct seul (AllenAI) | 28.0 | 0.61 |
| Qwen/Qwen3.5-2B seul (hors essaim) | 89.0 | |
| google/gemma-4-E2B-it seul (hors essaim) | 91.3 | |
| **essaim 7 familles, vote** | **95.0** | |
| **essaim 7 familles, vote pondéré** | **95.7** | |
| Qwen/Qwen3.5-9B seul (référence, 9.0 G) | 97.0 | |
| google/gemma-4-12B-it seul (référence, 12.0 G) | 97.7 | |
| mistralai/Ministral-3-14B-Instruct-2512 seul (référence, 14.0 G) | 94.7 | |
| Qwen/Qwen3.8-27B seul (référence, 27.0 G) | 97.7 | |
| oracle (au moins un pair juste) | 99.0 | |

| essaim (wvote) moins | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |
| --- | --- | --- | --- |
| meilleur pair de dev (Qwen/Qwen3.5-4B) | -1.0 [-3.4 ; +1.2] | 0.194 ; 0.006 | indéterminé |
| Qwen/Qwen3.5-9B | -1.3 [-4.0 ; +1.1] | 0.318 ; 0.006 | indéterminé |
| google/gemma-4-12B-it | -2.0 [-4.6 ; +0.1] | 0.606 ; 0.001 | indéterminé |
| mistralai/Ministral-3-14B-Instruct-2512 | +1.0 [-1.5 ; +3.7] | 0.013 ; 0.282 | non inférieur (-2) |
| Qwen/Qwen3.8-27B | -2.0 [-4.7 ; +0.3] | 0.606 ; 0.002 | indéterminé |

Passage à l'échelle (vote pondéré) : moyenne sur tous les sous-ensembles de k familles, et les k meilleures familles de dev.

| k | moyenne | k meilleures |
| --- | --- | --- |
| 1 | 81.4 | 96.7 |
| 2 | 92.6 | 96.7 |
| 3 | 93.8 | 96.3 |
| 4 | 94.9 | 97.3 |
| 5 | 95.3 | 96.7 |
| 6 | 95.3 | 95.7 |
| 7 | 95.7 | 95.7 |

Certificat d'arrêt (ordre d'arrivée = temps mesurés) : 5.11 pairs attendus sur 7 en moyenne.

## mmlupro (300 questions test)

Collision des réponses fausses sur dev : c = 0.390.

| système | exactitude test (%) | poids (wvote) |
| --- | --- | --- |
| Qwen/Qwen3.5-4B seul (Qwen) | 64.0 | 2.10 |
| google/gemma-4-E4B-it seul (Google) | 59.3 | 2.02 |
| ibm-granite/granite-4.2-3b seul (IBM) | 37.0 | 1.84 |
| HuggingFaceTB/SmolLM3-3B seul (Hugging Face) | 44.3 | 0.96 |
| mistralai/Ministral-3-3B-Instruct-2512 seul (Mistral) | 52.0 | 1.37 |
| microsoft/Phi-4-mini-instruct seul (Microsoft) | 54.0 | 1.34 |
| allenai/OLMo-2-0425-1B-Instruct seul (AllenAI) | 9.0 | 0.00 |
| Qwen/Qwen3.5-2B seul (hors essaim) | 44.3 | |
| google/gemma-4-E2B-it seul (hors essaim) | 45.7 | |
| **essaim 7 familles, vote** | **68.3** | |
| **essaim 7 familles, vote pondéré** | **70.7** | |
| Qwen/Qwen3.5-9B seul (référence, 9.0 G) | 67.3 | |
| google/gemma-4-12B-it seul (référence, 12.0 G) | 71.7 | |
| mistralai/Ministral-3-14B-Instruct-2512 seul (référence, 14.0 G) | 67.3 | |
| Qwen/Qwen3.8-27B seul (référence, 27.0 G) | 64.3 | |
| oracle (au moins un pair juste) | 82.0 | |

| essaim (wvote) moins | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |
| --- | --- | --- | --- |
| meilleur pair de dev (google/gemma-4-E4B-it) | +11.3 [+7.2 ; +16.0] | 0.000 ; 1.000 | supérieur |
| Qwen/Qwen3.5-9B | +3.3 [-1.8 ; +8.5] | 0.022 ; 0.745 | non inférieur (-2) |
| google/gemma-4-12B-it | -1.0 [-5.8 ; +3.8] | 0.358 ; 0.113 | indéterminé |
| mistralai/Ministral-3-14B-Instruct-2512 | +3.3 [-1.8 ; +8.5] | 0.022 ; 0.745 | non inférieur (-2) |
| Qwen/Qwen3.8-27B | +6.3 [+0.7 ; +12.0] | 0.002 ; 0.941 | supérieur |

Passage à l'échelle (vote pondéré) : moyenne sur tous les sous-ensembles de k familles, et les k meilleures familles de dev.

| k | moyenne | k meilleures |
| --- | --- | --- |
| 1 | 44.4 | 59.3 |
| 2 | 58.9 | 68.0 |
| 3 | 63.6 | 70.3 |
| 4 | 66.2 | 71.3 |
| 5 | 68.2 | 72.7 |
| 6 | 69.8 | 70.7 |
| 7 | 70.7 | 70.7 |

Certificat d'arrêt (ordre d'arrivée = temps mesurés) : 5.73 pairs attendus sur 7 en moyenne.

## Moyenne des benchmarks complets (gsm8k, math500, arc, mmlupro)

| système | exactitude moyenne (%) |
| --- | --- |
| essaim 7 familles, vote | 85.7 |
| essaim 7 familles, wvote | 86.6 |
| Qwen/Qwen3.5-4B | 83.5 |
| google/gemma-4-E4B-it | 67.5 |
| ibm-granite/granite-4.2-3b | 70.5 |
| HuggingFaceTB/SmolLM3-3B | 71.1 |
| mistralai/Ministral-3-3B-Instruct-2512 | 75.2 |
| microsoft/Phi-4-mini-instruct | 75.3 |
| allenai/OLMo-2-0425-1B-Instruct | 29.2 |
| Qwen/Qwen3.5-9B | 86.1 |
| google/gemma-4-12B-it | 89.5 |
| mistralai/Ministral-3-14B-Instruct-2512 | 81.7 |
| Qwen/Qwen3.8-27B | 86.6 |
