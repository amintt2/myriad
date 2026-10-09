# E10 : sections en parallèle — sot1 Colab

## Intégrité et méthode

Les 42 JSONL et 42 manifests (84 fichiers) sont identiques par SHA-256 aux copies du dépôt principal. Les empreintes sont archivées dans `sot_summary_colab.json`. Avant analyse : JSON strict jusqu’à la dernière ligne, clés exactes sans doublons, identité des données épinglées, configurations et provenance des modèles, points et affectation ordonnée, empreintes des squelettes et des réponses jugées contrôlés. Aucune donnée partielle utilisée. Les poids eux-mêmes ne sont pas présents ici : leurs empreintes sont celles des manifests.

Commande exécutée depuis phase0 :

```text
uv run python analyze_sot.py --tag sot1 --outline-model Qwen/Qwen3.5-4B --judge Qwen/Qwen3.8-27B --suffix _colab --peers Qwen/Qwen3.5-4B google/gemma-4-E4B-it ibm-granite/granite-4.2-3b mistralai/Ministral-3-3B-Instruct-2512 microsoft/Phi-4-mini-instruct HuggingFaceTB/SmolLM3-3B --baselines Qwen/Qwen3.5-4B google/gemma-4-E4B-it ibm-granite/granite-4.2-3b mistralai/Ministral-3-3B-Instruct-2512 microsoft/Phi-4-mini-instruct HuggingFaceTB/SmolLM3-3B --json results/sot_summary_colab.json
```

Les pairs et baselines suivent exactement cet ordre. Premiers tours MT-Bench à révision figée : writing, roleplay, stem, humanities (dev 16, test 24). Reasoning, math, coding, extraction (40 prompts) forment other, rapporté séparément.

Le meilleur solo est celui contre lequel le parallèle a le score moyen le plus faible sur dev, puis le score souple en cas d’égalité : **Qwen/Qwen3.5-4B**. Ce choix est figé pour test et other ; il coïncide avec le modèle du squelette. Ce critère ne classe pas indépendamment les solos.

## Qualité jugée

Qwen/Qwen3.8-27B compare chaque paire dans les deux ordres. Par ordre, score 1/0,5/0 pour victoire/égalité/défaite ; moyenne par prompt. Une inversion victoire/défaite vaut une égalité. Victoire/égalité et défaite/égalité valent 0,75 et 0,25 : les comptes V/N/D ne suffisent pas à reconstruire le score. Score souple : P(parallèle meilleur) + P(égalité)/2, probabilités renormalisées sur les lettres disponibles. Sur other, six comparaisons dans les deux ordres portent sur des réponses identiques (mtbench-131 contre Qwen, Gemma et Granite) : égalités automatiques, score souple 0,5, sans appel au juge ni probabilités. Les 954 appels restants terminent au plafond d’un jeton imposé au verdict ; ce plafond ne tronque pas une justification, puisque le protocole ne demande qu’une lettre. IC à 95 % par bootstrap de prompts (10 000 rééchantillonnages, graine 0), ordres gardés ensemble, sans correction des comparaisons multiples.

### Test — résultats principaux

| Solo | V/N/D | Score [IC 95 %] | Souple [IC 95 %] | Cohérence des ordres | Verdicts A |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-4B | 0/4/20 | 0.115 [0.042 ; 0.198] | 0.173 [0.128 ; 0.223] | 87.5 % | 39.6 % |
| google/gemma-4-E4B-it | 0/4/20 | 0.115 [0.042 ; 0.198] | 0.178 [0.125 ; 0.238] | 87.5 % | 35.4 % |
| ibm-granite/granite-4.2-3b | 1/4/19 | 0.167 [0.083 ; 0.260] | 0.220 [0.163 ; 0.287] | 75.0 % | 33.3 % |
| mistralai/Ministral-3-3B-Instruct-2512 | 2/6/16 | 0.260 [0.156 ; 0.375] | 0.294 [0.212 ; 0.383] | 70.8 % | 29.2 % |
| microsoft/Phi-4-mini-instruct | 8/13/3 | 0.562 [0.448 ; 0.677] | 0.575 [0.478 ; 0.669] | 79.2 % | 27.1 % |
| HuggingFaceTB/SmolLM3-3B | 8/11/5 | 0.562 [0.427 ; 0.698] | 0.551 [0.438 ; 0.666] | 91.7 % | 27.1 % |

### Dev — sélection

| Solo | V/N/D | Score [IC 95 %] | Souple [IC 95 %] | Cohérence des ordres | Verdicts A |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-4B | 0/3/13 | 0.141 [0.047 ; 0.250] | 0.204 [0.150 ; 0.257] | 81.2 % | 37.5 % |
| google/gemma-4-E4B-it | 0/3/13 | 0.156 [0.062 ; 0.250] | 0.226 [0.173 ; 0.288] | 68.8 % | 46.9 % |
| ibm-granite/granite-4.2-3b | 1/4/11 | 0.312 [0.203 ; 0.438] | 0.273 [0.210 ; 0.343] | 50.0 % | 28.1 % |
| mistralai/Ministral-3-3B-Instruct-2512 | 0/7/9 | 0.266 [0.156 ; 0.375] | 0.292 [0.226 ; 0.355] | 81.2 % | 21.9 % |
| microsoft/Phi-4-mini-instruct | 4/9/3 | 0.484 [0.375 ; 0.594] | 0.526 [0.427 ; 0.616] | 62.5 % | 21.9 % |
| HuggingFaceTB/SmolLM3-3B | 5/10/1 | 0.594 [0.469 ; 0.719] | 0.564 [0.458 ; 0.668] | 81.2 % | 21.9 % |

### Other — catégories exclues

| Solo | V/N/D | Score [IC 95 %] | Souple [IC 95 %] | Cohérence des ordres | Verdicts A |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-4B | 1/1/38 | 0.050 [0.006 ; 0.113] | 0.088 [0.046 ; 0.141] | 95.0 % | 47.5 % |
| google/gemma-4-E4B-it | 1/1/38 | 0.037 [0.000 ; 0.087] | 0.082 [0.043 ; 0.132] | 95.0 % | 48.8 % |
| ibm-granite/granite-4.2-3b | 2/7/31 | 0.150 [0.075 ; 0.237] | 0.174 [0.107 ; 0.248] | 80.0 % | 42.5 % |
| mistralai/Ministral-3-3B-Instruct-2512 | 2/5/33 | 0.131 [0.062 ; 0.212] | 0.160 [0.098 ; 0.230] | 82.5 % | 41.2 % |
| microsoft/Phi-4-mini-instruct | 5/7/28 | 0.231 [0.138 ; 0.338] | 0.226 [0.149 ; 0.308] | 77.5 % | 40.0 % |
| HuggingFaceTB/SmolLM3-3B | 6/6/28 | 0.250 [0.150 ; 0.356] | 0.261 [0.171 ; 0.359] | 77.5 % | 38.8 % |

Le parallèle perd nettement contre le solo choisi sur dev selon ce juge. Les IC contre Phi et SmolLM couvrent 0,5 : aucune équivalence démontrée. Un seul juge de la famille Qwen peut favoriser sa famille ; les deux ordres atténuent le biais de position sans éliminer tous les biais. Il ne s’agit ni de notes humaines ni d’une vérification factuelle.

## Latence MODÉLISÉE

Solo = RTT + jetons/débit. Parallèle = 2 RTT + squelette/débit + maximum du coût des points successifs de chaque pair. Le repli ajoute le squelette à la baseline Qwen. Prompt processing, files, contention et transport effectif omis. Les temps de génération sous service en lot ne sont pas employés. Aucun réseau parallèle réel chronométré.

La table `speeds_consumer.json` mesure seulement Granite (56 jetons/s) et SmolLM (37 jetons/s), mono-flux llama-bench tg128, Q8_0, RX 6650 XT. **Qwen, Gemma, Ministral et Phi utilisent 40 jetons/s par défaut : hypothèse, pas mesure.** Une variante suppose 50 jetons/s pour tous. Les RTT sont paramétrés.

| Partition | Scénario | RTT ms | Parallèle moyen s | Solo choisi moyen s | Ratio latence | Ratio médian par prompt | Ratio débit effectif |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | mono-flux + défauts | 50 | 10.01 | 16.97 | 1.69 | 1.90 | 3.14 |
| test | mono-flux + défauts | 100 | 10.11 | 17.02 | 1.68 | 1.88 | 3.12 |
| test | mono-flux + défauts | 150 | 10.21 | 17.07 | 1.67 | 1.87 | 3.09 |
| test | toutes à 50 tok/s | 50 | 7.96 | 13.58 | 1.71 | 1.91 | 3.16 |
| test | toutes à 50 tok/s | 100 | 8.06 | 13.63 | 1.69 | 1.90 | 3.13 |
| test | toutes à 50 tok/s | 150 | 8.16 | 13.68 | 1.68 | 1.88 | 3.10 |
| dev | mono-flux + défauts | 50 | 9.94 | 15.25 | 1.53 | 1.40 | 3.19 |
| dev | mono-flux + défauts | 100 | 10.04 | 15.30 | 1.52 | 1.39 | 3.16 |
| dev | mono-flux + défauts | 150 | 10.14 | 15.35 | 1.51 | 1.39 | 3.14 |
| dev | toutes à 50 tok/s | 50 | 7.99 | 12.21 | 1.53 | 1.40 | 3.17 |
| dev | toutes à 50 tok/s | 100 | 8.09 | 12.26 | 1.52 | 1.39 | 3.15 |
| dev | toutes à 50 tok/s | 150 | 8.19 | 12.31 | 1.50 | 1.38 | 3.12 |
| other | mono-flux + défauts | 50 | 7.57 | 10.36 | 1.37 | 1.23 | 2.63 |
| other | mono-flux + défauts | 100 | 7.67 | 10.41 | 1.36 | 1.21 | 2.61 |
| other | mono-flux + défauts | 150 | 7.77 | 10.46 | 1.35 | 1.20 | 2.59 |
| other | toutes à 50 tok/s | 50 | 6.10 | 8.30 | 1.36 | 1.26 | 2.62 |
| other | toutes à 50 tok/s | 100 | 6.20 | 8.35 | 1.35 | 1.24 | 2.59 |
| other | toutes à 50 tok/s | 150 | 6.30 | 8.40 | 1.33 | 1.22 | 2.56 |

Ratio latence = temps moyen solo / temps moyen parallèle. Débit effectif = somme des jetons / somme des temps modélisés. Tokenizers différents : les jetons ne sont pas une unité textuelle normalisée. Un ratio de débit accru ne signifie pas une qualité conservée.

## Longueur, troncatures et limites

Budgets : baseline 1024 jetons, squelette 256, chaque expansion 256. Squelette demandé de 3 à 8 points, 12 mots au plus par point. Développements concaténés sans révision ; budgets totaux différents.

| Partition | N | Statuts | Replis | Points >12 mots | Points moyens | Pairs moyens | Expansions au plafond | Jetons moyens parallèle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev | 16 | {'ok': 16} | 0 | 0 | 7.25 | 6.00 | 22/116 | 1262.9 |
| test | 24 | {'ok': 24} | 0 | 1 | 7.12 | 5.83 | 30/171 | 1252.5 |
| other | 40 | {'fallback': 1, 'ok': 39} | 1 | 7 | 5.67 | 5.10 | 13/221 | 793.1 |

| Partition | Solo | Jetons moyens | Réponses au plafond |
| --- | --- | --- | --- |
| dev | Qwen/Qwen3.5-4B | 607.9 | 5 |
| dev | google/gemma-4-E4B-it | 692.3 | 6 |
| dev | ibm-granite/granite-4.2-3b | 663.2 | 5 |
| dev | mistralai/Ministral-3-3B-Instruct-2512 | 691.9 | 5 |
| dev | microsoft/Phi-4-mini-instruct | 322.7 | 0 |
| dev | HuggingFaceTB/SmolLM3-3B | 441.1 | 1 |
| test | Qwen/Qwen3.5-4B | 676.6 | 8 |
| test | google/gemma-4-E4B-it | 750.2 | 10 |
| test | ibm-granite/granite-4.2-3b | 703.9 | 7 |
| test | mistralai/Ministral-3-3B-Instruct-2512 | 667.1 | 7 |
| test | microsoft/Phi-4-mini-instruct | 341.0 | 0 |
| test | HuggingFaceTB/SmolLM3-3B | 459.5 | 0 |
| other | Qwen/Qwen3.5-4B | 412.3 | 4 |
| other | google/gemma-4-E4B-it | 442.6 | 10 |
| other | ibm-granite/granite-4.2-3b | 543.3 | 11 |
| other | mistralai/Ministral-3-3B-Instruct-2512 | 377.2 | 3 |
| other | microsoft/Phi-4-mini-instruct | 223.4 | 0 |
| other | HuggingFaceTB/SmolLM3-3B | 305.3 | 2 |

Aucun squelette tronqué par le parseur ; un repli sur other. Les plafonds de génération et le point trop long sur test sont conservés sans exclusion. Les réponses parallèles sont plus longues, la comparaison confond méthode et budgets. Taille modeste, premiers tours d’un seul jeu et juge unique limitent la généralisation. Le gain de latence repose sur des hypothèses et accompagne une qualité jugée moindre contre le solo choisi. Ni équivalence de qualité ni accélération réseau réelle démontrée.

Références du protocole : [Skeleton-of-Thought](https://arxiv.org/abs/2307.15337), [MT-Bench et LLM-as-a-Judge](https://arxiv.org/abs/2306.05685).
