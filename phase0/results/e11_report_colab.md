# E11 : un essaim de petits modèles qui choisit un programme en l'EXÉCUTANT, contre des modèles plus gros

Candidats : la solution gloutonne de chaque modèle et ses solutions tirées (température 0.8). Règles et poids choisis sur **dev**, rapportés sur **test**. Exactitude = pass@1 du programme choisi sur les tests cachés EvalPlus. IC 95 % : score de Tango pour la différence appariée ; tests non conditionnels avec p maximisés numériquement (essaim/stats.py), TOST avec une marge de ±2.0 points, deux tests unilatéraux à 5 %.

Campagne canonique complète : empreintes SHA-256 des bruts, manifestes et modules dans e11_validation_colab.json ; validation offline des caches épinglés, des IDs/samples, de l'extraction et de la couverture exec exacte (y compris les exclusions).

| partition | problèmes bruts | générations | programmes + références | exclus | length |
| --- | --- | --- | --- | --- | --- |
| humanevalplus / dev | 82 | 5330 | 4664 | ['HumanEval/32'] | 113 |
| humanevalplus / test | 82 | 5330 | 4773 | [] | 188 |
| mbppplus / dev | 189 | 12285 | 9356 | [] | 274 |
| mbppplus / test | 189 | 12285 | 9618 | ['Mbpp/255'] | 246 |

Bac déclaré : sandbox-v3, namespace réseau unshare --net, audit hook Python, rlimits et délais par appel. **Landlock unavailable** dans les quatre manifestes ; aucune protection LocalLinuxEnv/seccomp n'est attribuée à cette campagne. L'audit hook n'est pas une barrière OS contre du code natif hostile. Aucun résultat n'a été renforcé ou réécrit rétroactivement.

Oracle historique tests-v4 : Mbpp/737, 787 et 794 ajoutent l'assertion exact_match omise dans l'export ; HumanEval/32 utilise le contrôle de racine prévu mais échoue aussi pour la référence et est exclu sur dev. Les résultats sont ceux de ce harnais, pas une exécution officielle EvalPlus inchangée. Les signatures sont des hashes typés tronqués (repli digest pour grands objets), arrondissent les flottants et peuvent aussi séparer des solutions justes hors préconditions.

Tests exploratoires sans correction de multiplicité ; pooling descriptif. Les poids sont régularisés (p borné à [0,02 ; 0,98], collision à [0,001 ; 0,999], poids négatifs annulés). Les mesures ne démontrent pas un passage à l'échelle distribué.

## humanevalplus (82 problèmes test)

Poids (dev) : Qwen 1.79, Google 1.79, IBM 1.70, Hugging Face 0.92, Mistral 1.62, Microsoft 1.03, AllenAI 0.00. Variante de regroupement choisie : **count** ; règle principale : **cluster-count|all** ; cascade : m = 3.

| système | pass@1 test (%) | appels au gros modèle |
| --- | --- | --- |
| Qwen/Qwen3.5-4B seul (Qwen), glouton | 82.9 |  |
| google/gemma-4-E4B-it seul (Google), glouton | 84.1 |  |
| ibm-granite/granite-4.2-3b seul (IBM), glouton | 73.2 |  |
| HuggingFaceTB/SmolLM3-3B seul (Hugging Face), glouton | 63.4 |  |
| mistralai/Ministral-3-3B-Instruct-2512 seul (Mistral), glouton | 74.4 |  |
| microsoft/Phi-4-mini-instruct seul (Microsoft), glouton | 70.7 |  |
| allenai/OLMo-2-0425-1B-Instruct seul (AllenAI), glouton | 13.4 |  |
| Qwen/Qwen3.5-2B seul (hors essaim), glouton | 48.8 |  |
| google/gemma-4-E2B-it seul (hors essaim), glouton | 72.0 |  |
| Qwen/Qwen3.5-9B seul (référence, 9.0 G), glouton | 79.3 |  |
| google/gemma-4-12B-it seul (référence, 12.0 G), glouton | 91.5 |  |
| mistralai/Ministral-3-14B-Instruct-2512 seul (référence, 14.0 G), glouton | 82.9 |  |
| Qwen/Qwen3.8-27B seul (référence, 27.0 G), glouton | 91.5 |  |
| (a) meilleur pair de dev (Qwen/Qwen3.5-4B), glouton | 82.9 |  |
| (a') meilleur pair, sélection sur ses 5 solutions | 87.8 |  |
| (b) vote sur le texte, gloutons | 82.9 |  |
| (c) tests visibles + poids, gloutons | 89.0 |  |
| (d) regroupement fonctionnel (count), gloutons | 90.2 |  |
| (d) regroupement fonctionnel (families), gloutons | 90.2 |  |
| (d) regroupement fonctionnel (wfamilies), gloutons | 90.2 |  |
| (b) vote sur le texte, gloutons + tirages | 85.4 |  |
| (c) tests visibles + poids, gloutons + tirages | 89.0 |  |
| (d) regroupement fonctionnel (count), gloutons + tirages | 89.0 |  |
| (d) regroupement fonctionnel (families), gloutons + tirages | 87.8 |  |
| (d) regroupement fonctionnel (wfamilies), gloutons + tirages | 89.0 |  |
| Qwen/Qwen3.5-9B + sélection sur ses propres solutions (count, choisi sur dev) | 85.4 |  |
| google/gemma-4-12B-it + sélection sur ses propres solutions (count, choisi sur dev) | 93.9 |  |
| mistralai/Ministral-3-14B-Instruct-2512 + sélection sur ses propres solutions (count, choisi sur dev) | 87.8 |  |
| Qwen/Qwen3.8-27B + sélection sur ses propres solutions (count, choisi sur dev) | 91.5 |  |
| (e) cascade vers Qwen/Qwen3.8-27B, m = 1 | 90.2 | 6 |
| (e) cascade vers Qwen/Qwen3.8-27B, m = 2 | 90.2 | 7 |
| (e) cascade vers Qwen/Qwen3.8-27B, m = 3 (choisi) | 90.2 | 8 |
| oracle (au moins un candidat de l'essaim juste) | 97.6 |  |

Comparaisons appariées (test) : différence en points [IC 95 %], p exacts des deux tests de marge, verdict.

| système | contre | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |
| --- | --- | --- | --- | --- |
| règle principale | meilleur pair (a) | +6.1 [-0.3 ; +14.1] | 0.011 ; 0.918 | non inférieur (-2) |
| règle principale | Qwen/Qwen3.5-9B | +9.8 [+0.9 ; +19.5] | 0.007 ; 0.975 | supérieur |
| règle principale | google/gemma-4-12B-it | -2.4 [-10.5 ; +5.1] | 0.593 ; 0.191 | indéterminé |
| règle principale | mistralai/Ministral-3-14B-Instruct-2512 | +6.1 [-2.8 ; +15.6] | 0.037 ; 0.849 | non inférieur (-2) |
| règle principale | Qwen/Qwen3.8-27B | -2.4 [-10.5 ; +5.1] | 0.593 ; 0.191 | indéterminé |
| règle principale | Qwen/Qwen3.5-9B + sélection | +3.7 [-4.1 ; +12.1] | 0.081 ; 0.774 | indéterminé |
| règle principale | google/gemma-4-12B-it + sélection | -4.9 [-12.5 ; +1.3] | 0.850 ; 0.020 | indéterminé |
| règle principale | mistralai/Ministral-3-14B-Instruct-2512 + sélection | +1.2 [-6.9 ; +9.5] | 0.219 ; 0.510 | indéterminé |
| règle principale | Qwen/Qwen3.8-27B + sélection | -2.4 [-10.5 ; +5.1] | 0.593 ; 0.191 | indéterminé |
| cascade (m choisi) | meilleur pair (a) | +7.3 [+0.7 ; +15.6] | 0.006 ; 0.975 | supérieur |
| cascade (m choisi) | Qwen/Qwen3.5-9B | +11.0 [+1.9 ; +20.9] | 0.004 ; 0.994 | supérieur |
| cascade (m choisi) | google/gemma-4-12B-it | -1.2 [-9.5 ; +6.9] | 0.510 ; 0.219 | indéterminé |
| cascade (m choisi) | mistralai/Ministral-3-14B-Instruct-2512 | +7.3 [-1.8 ; +17.1] | 0.024 ; 0.918 | non inférieur (-2) |
| cascade (m choisi) | Qwen/Qwen3.8-27B | -1.2 [-8.9 ; +6.2] | 0.510 ; 0.219 | indéterminé |
| cascade (m choisi) | Qwen/Qwen3.5-9B + sélection | +4.9 [-3.1 ; +13.6] | 0.053 ; 0.792 | indéterminé |
| cascade (m choisi) | google/gemma-4-12B-it + sélection | -3.7 [-11.5 ; +3.3] | 0.774 ; 0.053 | indéterminé |
| cascade (m choisi) | mistralai/Ministral-3-14B-Instruct-2512 + sélection | +2.4 [-5.8 ; +11.1] | 0.191 ; 0.585 | indéterminé |
| cascade (m choisi) | Qwen/Qwen3.8-27B + sélection | -1.2 [-8.9 ; +6.2] | 0.510 ; 0.219 | indéterminé |

Passage à l'échelle (gloutons + tirages) : moyenne sur tous les sous-ensembles de k familles.

| k | (b) texte | (c) visibles | (d) regroupement | oracle |
| --- | --- | --- | --- | --- |
| 1 | 66.7 | 72.1 | 72.5 | 77.9 |
| 2 | 78.6 | 84.3 | 84.7 | 92.0 |
| 3 | 81.4 | 86.9 | 87.2 | 95.3 |
| 4 | 83.2 | 88.2 | 88.3 | 96.5 |
| 5 | 84.5 | 88.9 | 89.0 | 97.1 |
| 6 | 85.2 | 89.2 | 89.0 | 97.4 |
| 7 | 85.4 | 89.0 | 89.0 | 97.6 |

Programmes exécutés par problème (essaim de 7 familles, gloutons + tirages, programmes distincts) : 32.6 sur les tests visibles, 24.1 sur les entrées supplémentaires (16 entrées dérivées + les entrées visibles).

Collisions fonctionnelles (paires de familles différentes) :

| pool | c (2 faux d'accord sur tout) | a (2 justes d'accord) | c texte | faux acceptés par les tests visibles (paires) |
| --- | --- | --- | --- | --- |
| greedy | 0.483 (58 paires) | 0.969 (804) | 0.000 | 0.262 |
| all | 0.546 (1252 paires) | 0.972 (18321) | 0.000 | 0.217 |

| modèle | glouton | moyenne des tirages et du glouton | passe les tests visibles | juste sachant visibles OK |
| --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-4B | 82.9 | 77.3 | 87.8 | 88.1 |
| google/gemma-4-E4B-it | 84.1 | 84.1 | 89.5 | 94.0 |
| ibm-granite/granite-4.2-3b | 73.2 | 74.4 | 85.4 | 87.1 |
| HuggingFaceTB/SmolLM3-3B | 63.4 | 56.1 | 71.5 | 78.5 |
| mistralai/Ministral-3-3B-Instruct-2512 | 74.4 | 72.0 | 84.9 | 84.8 |
| microsoft/Phi-4-mini-instruct | 70.7 | 62.2 | 75.4 | 82.5 |
| allenai/OLMo-2-0425-1B-Instruct | 13.4 | 13.4 | 24.9 | 53.9 |
| Qwen/Qwen3.5-2B | 48.8 | 44.4 | 55.6 | 79.8 |
| google/gemma-4-E2B-it | 72.0 | 73.7 | 84.1 | 87.5 |
| Qwen/Qwen3.5-9B | 79.3 | 81.7 | 89.5 | 91.3 |
| google/gemma-4-12B-it | 91.5 | 92.2 | 95.9 | 96.2 |
| mistralai/Ministral-3-14B-Instruct-2512 | 82.9 | 81.5 | 91.2 | 89.3 |
| Qwen/Qwen3.8-27B | 91.5 | 90.0 | 93.7 | 96.1 |

Tests visibles : 8 problèmes sans aucun test visible valide ; 6 tests visibles ignorés (la référence y échoue).


### Coûts observables de génération

Sommes des compteurs des appels nécessaires à chaque mode, par problème test retenu. Les compteurs de jetons utilisent les tokenizers des producteurs. La somme des ms est un proxy de durées d'appels, **pas le temps mur** : les requêtes de run_code.py sont parallèles (défaut 16), sur un GPU partagé. Même leur maximum serait une projection, sans WAN ni essaim réel. Une cascade est un replay offline : seuls les appels de référence déclenchés sont attribués. Le temps exec (visible + extra + HIDDEN) n'est pas un temps de sélection en ligne mesuré. Énergie, coût monétaire réel et coût total restent inconnus ; aucune comparaison à calcul égal.

| mode | jetons sortie/problème | jetons prompt/problème | somme durées appels (s)/problème, proxy | length/appels |
| --- | --- | --- | --- | --- |
| best | 273.65 | 201.18 | 13.83 | 1/82 |
| best_self | 1437.12 | 1005.91 | 74.27 | 12/410 |
| cluster-count / all | 10090.43 | 9727.93 | 405.01 | 99/2870 |
| cascade / 3 | 10128.40 | 9752.93 | 407.75 | 99/2878 |
| alone / Qwen/Qwen3.5-9B | 283.15 | 201.18 | 14.40 | 4/82 |
| refself / Qwen/Qwen3.5-9B | 1457.20 | 1005.91 | 75.46 | 16/410 |
| alone / google/gemma-4-12B-it | 285.79 | 207.12 | 18.03 | 2/82 |
| refself / google/gemma-4-12B-it | 1428.29 | 1035.61 | 91.56 | 10/410 |
| alone / mistralai/Ministral-3-14B-Instruct-2512 | 198.38 | 718.96 | 9.30 | 1/82 |
| refself / mistralai/Ministral-3-14B-Instruct-2512 | 1013.40 | 3594.82 | 47.81 | 4/410 |
| alone / Qwen/Qwen3.8-27B | 273.71 | 201.18 | 20.81 | 3/82 |
| refself / Qwen/Qwen3.8-27B | 1381.63 | 1005.91 | 107.27 | 18/410 |

Détails par problème/mode, compteurs connus et manquants, troncatures et observations par modèle/split : e11_summary_colab.json. Le coût des problèmes exclus figure dans generation_observations ; il ne disparaît pas des totaux de campagne.

## mbppplus (188 problèmes test, 1 exclus car la référence échoue : ['Mbpp/255'])

Poids (dev) : Qwen 1.07, Google 1.26, IBM 1.02, Hugging Face 0.74, Mistral 0.97, Microsoft 0.97, AllenAI 0.00. Variante de regroupement choisie : **count** ; règle principale : **cluster-count|all** ; cascade : m = 1.

| système | pass@1 test (%) | appels au gros modèle |
| --- | --- | --- |
| Qwen/Qwen3.5-4B seul (Qwen), glouton | 68.1 |  |
| google/gemma-4-E4B-it seul (Google), glouton | 73.9 |  |
| ibm-granite/granite-4.2-3b seul (IBM), glouton | 67.6 |  |
| HuggingFaceTB/SmolLM3-3B seul (Hugging Face), glouton | 62.2 |  |
| mistralai/Ministral-3-3B-Instruct-2512 seul (Mistral), glouton | 62.2 |  |
| microsoft/Phi-4-mini-instruct seul (Microsoft), glouton | 70.7 |  |
| allenai/OLMo-2-0425-1B-Instruct seul (AllenAI), glouton | 12.8 |  |
| Qwen/Qwen3.5-2B seul (hors essaim), glouton | 47.3 |  |
| google/gemma-4-E2B-it seul (hors essaim), glouton | 71.3 |  |
| Qwen/Qwen3.5-9B seul (référence, 9.0 G), glouton | 72.3 |  |
| google/gemma-4-12B-it seul (référence, 12.0 G), glouton | 75.5 |  |
| mistralai/Ministral-3-14B-Instruct-2512 seul (référence, 14.0 G), glouton | 72.9 |  |
| Qwen/Qwen3.8-27B seul (référence, 27.0 G), glouton | 78.2 |  |
| (a) meilleur pair de dev (google/gemma-4-E4B-it), glouton | 73.9 |  |
| (a') meilleur pair, sélection sur ses 5 solutions | 76.6 |  |
| (b) vote sur le texte, gloutons | 73.9 |  |
| (c) tests visibles + poids, gloutons | 78.2 |  |
| (d) regroupement fonctionnel (count), gloutons | 78.7 |  |
| (d) regroupement fonctionnel (families), gloutons | 78.7 |  |
| (d) regroupement fonctionnel (wfamilies), gloutons | 78.7 |  |
| (b) vote sur le texte, gloutons + tirages | 72.3 |  |
| (c) tests visibles + poids, gloutons + tirages | 80.3 |  |
| (d) regroupement fonctionnel (count), gloutons + tirages | 80.9 |  |
| (d) regroupement fonctionnel (families), gloutons + tirages | 80.9 |  |
| (d) regroupement fonctionnel (wfamilies), gloutons + tirages | 80.9 |  |
| Qwen/Qwen3.5-9B + sélection sur ses propres solutions (count, choisi sur dev) | 78.7 |  |
| google/gemma-4-12B-it + sélection sur ses propres solutions (count, choisi sur dev) | 76.1 |  |
| mistralai/Ministral-3-14B-Instruct-2512 + sélection sur ses propres solutions (count, choisi sur dev) | 75.0 |  |
| Qwen/Qwen3.8-27B + sélection sur ses propres solutions (count, choisi sur dev) | 81.4 |  |
| (e) cascade vers Qwen/Qwen3.8-27B, m = 1 (choisi) | 80.9 | 4 |
| (e) cascade vers Qwen/Qwen3.8-27B, m = 2 | 80.9 | 14 |
| (e) cascade vers Qwen/Qwen3.8-27B, m = 3 | 80.3 | 20 |
| oracle (au moins un candidat de l'essaim juste) | 87.2 |  |

Comparaisons appariées (test) : différence en points [IC 95 %], p exacts des deux tests de marge, verdict.

| système | contre | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |
| --- | --- | --- | --- | --- |
| règle principale | meilleur pair (a) | +6.9 [+4.1 ; +11.5] | 0.000 ; 1.000 | supérieur |
| règle principale | Qwen/Qwen3.5-9B | +8.5 [+3.5 ; +14.2] | 0.000 ; 0.998 | supérieur |
| règle principale | google/gemma-4-12B-it | +5.3 [+1.3 ; +10.1] | 0.001 ; 0.955 | supérieur |
| règle principale | mistralai/Ministral-3-14B-Instruct-2512 | +8.0 [+3.0 ; +13.6] | 0.000 ; 0.995 | supérieur |
| règle principale | Qwen/Qwen3.8-27B | +2.7 [-1.5 ; +7.3] | 0.018 ; 0.676 | non inférieur (-2) |
| règle principale | Qwen/Qwen3.5-9B + sélection | +2.1 [-2.3 ; +6.8] | 0.035 ; 0.560 | non inférieur (-2) |
| règle principale | google/gemma-4-12B-it + sélection | +4.8 [+0.9 ; +9.5] | 0.001 ; 0.925 | supérieur |
| règle principale | mistralai/Ministral-3-14B-Instruct-2512 + sélection | +5.9 [+0.9 ; +11.3] | 0.002 ; 0.941 | supérieur |
| règle principale | Qwen/Qwen3.8-27B + sélection | -0.5 [-4.5 ; +3.3] | 0.272 ; 0.108 | indéterminé |
| cascade (m choisi) | meilleur pair (a) | +6.9 [+4.1 ; +11.5] | 0.000 ; 1.000 | supérieur |
| cascade (m choisi) | Qwen/Qwen3.5-9B | +8.5 [+3.5 ; +14.2] | 0.000 ; 0.998 | supérieur |
| cascade (m choisi) | google/gemma-4-12B-it | +5.3 [+1.3 ; +10.1] | 0.001 ; 0.955 | supérieur |
| cascade (m choisi) | mistralai/Ministral-3-14B-Instruct-2512 | +8.0 [+3.0 ; +13.6] | 0.000 ; 0.995 | supérieur |
| cascade (m choisi) | Qwen/Qwen3.8-27B | +2.7 [-1.5 ; +7.3] | 0.018 ; 0.676 | non inférieur (-2) |
| cascade (m choisi) | Qwen/Qwen3.5-9B + sélection | +2.1 [-2.3 ; +6.8] | 0.035 ; 0.560 | non inférieur (-2) |
| cascade (m choisi) | google/gemma-4-12B-it + sélection | +4.8 [+0.9 ; +9.5] | 0.001 ; 0.925 | supérieur |
| cascade (m choisi) | mistralai/Ministral-3-14B-Instruct-2512 + sélection | +5.9 [+0.9 ; +11.3] | 0.002 ; 0.941 | supérieur |
| cascade (m choisi) | Qwen/Qwen3.8-27B + sélection | -0.5 [-4.5 ; +3.3] | 0.272 ; 0.108 | indéterminé |

Passage à l'échelle (gloutons + tirages) : moyenne sur tous les sous-ensembles de k familles.

| k | (b) texte | (c) visibles | (d) regroupement | oracle |
| --- | --- | --- | --- | --- |
| 1 | 60.0 | 66.4 | 66.7 | 68.8 |
| 2 | 69.4 | 76.6 | 76.8 | 80.5 |
| 3 | 71.1 | 78.3 | 78.9 | 83.4 |
| 4 | 71.8 | 79.0 | 79.7 | 84.9 |
| 5 | 72.0 | 79.5 | 80.1 | 85.9 |
| 6 | 72.1 | 79.9 | 80.5 | 86.6 |
| 7 | 72.3 | 80.3 | 80.9 | 87.2 |

Programmes exécutés par problème (essaim de 7 familles, gloutons + tirages, programmes distincts) : 27.5 sur les tests visibles, 16.8 sur les entrées supplémentaires (16 entrées dérivées + les entrées visibles).

Collisions fonctionnelles (paires de familles différentes) :

| pool | c (2 faux d'accord sur tout) | a (2 justes d'accord) | c texte | faux acceptés par les tests visibles (paires) |
| --- | --- | --- | --- | --- |
| greedy | 0.870 (185 paires) | 0.940 (1785) | 0.017 | 0.181 |
| all | 0.841 (4354 paires) | 0.945 (41860) | 0.010 | 0.157 |

| modèle | glouton | moyenne des tirages et du glouton | passe les tests visibles | juste sachant visibles OK |
| --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-4B | 68.1 | 65.5 | 78.6 | 82.8 |
| google/gemma-4-E4B-it | 73.9 | 73.3 | 86.5 | 84.5 |
| ibm-granite/granite-4.2-3b | 67.6 | 66.6 | 77.8 | 84.8 |
| HuggingFaceTB/SmolLM3-3B | 62.2 | 57.6 | 64.7 | 88.3 |
| mistralai/Ministral-3-3B-Instruct-2512 | 62.2 | 57.2 | 66.1 | 85.5 |
| microsoft/Phi-4-mini-instruct | 70.7 | 66.9 | 74.9 | 88.5 |
| allenai/OLMo-2-0425-1B-Instruct | 12.8 | 13.2 | 15.5 | 82.9 |
| Qwen/Qwen3.5-2B | 47.3 | 43.2 | 51.6 | 82.3 |
| google/gemma-4-E2B-it | 71.3 | 70.2 | 81.7 | 85.2 |
| Qwen/Qwen3.5-9B | 72.3 | 71.2 | 84.8 | 82.7 |
| google/gemma-4-12B-it | 75.5 | 75.0 | 92.0 | 81.2 |
| mistralai/Ministral-3-14B-Instruct-2512 | 72.9 | 70.4 | 81.9 | 85.8 |
| Qwen/Qwen3.8-27B | 78.2 | 78.0 | 92.3 | 83.9 |

Tests visibles : 0 problèmes sans aucun test visible valide ; 0 tests visibles ignorés (la référence y échoue).


### Coûts observables de génération

Sommes des compteurs des appels nécessaires à chaque mode, par problème test retenu. Les compteurs de jetons utilisent les tokenizers des producteurs. La somme des ms est un proxy de durées d'appels, **pas le temps mur** : les requêtes de run_code.py sont parallèles (défaut 16), sur un GPU partagé. Même leur maximum serait une projection, sans WAN ni essaim réel. Une cascade est un replay offline : seuls les appels de référence déclenchés sont attribués. Le temps exec (visible + extra + HIDDEN) n'est pas un temps de sélection en ligne mesuré. Énergie, coût monétaire réel et coût total restent inconnus ; aucune comparaison à calcul égal.

| mode | jetons sortie/problème | jetons prompt/problème | somme durées appels (s)/problème, proxy | length/appels |
| --- | --- | --- | --- | --- |
| best | 198.74 | 159.95 | 10.05 | 5/188 |
| best_self | 1071.43 | 799.76 | 54.62 | 32/940 |
| cluster-count / all | 4643.10 | 8250.64 | 215.82 | 139/6580 |
| cascade / 1 | 4652.03 | 8254.07 | 216.48 | 140/6584 |
| alone / Qwen/Qwen3.5-9B | 157.43 | 159.07 | 9.65 | 3/188 |
| refself / Qwen/Qwen3.5-9B | 817.77 | 795.35 | 49.79 | 18/940 |
| alone / google/gemma-4-12B-it | 124.81 | 163.95 | 9.23 | 2/188 |
| refself / google/gemma-4-12B-it | 684.46 | 819.76 | 51.56 | 20/940 |
| alone / mistralai/Ministral-3-14B-Instruct-2512 | 65.98 | 681.79 | 3.95 | 0/188 |
| refself / mistralai/Ministral-3-14B-Instruct-2512 | 382.78 | 3408.96 | 23.08 | 5/940 |
| alone / Qwen/Qwen3.8-27B | 112.85 | 159.07 | 10.74 | 1/188 |
| refself / Qwen/Qwen3.8-27B | 586.31 | 795.35 | 56.86 | 12/940 |

Détails par problème/mode, compteurs connus et manquants, troncatures et observations par modèle/split : e11_summary_colab.json. Le coût des problèmes exclus figure dans generation_observations ; il ne disparaît pas des totaux de campagne.

## Pooling descriptif (problèmes test mis bout à bout, MBPP pèse davantage)

| comparaison | différence [IC 95 %] | p exact (marge basse ; haute) | verdict |
| --- | --- | --- | --- |
| règle principale moins meilleur pair (a) | +6.7 [+3.9 ; +10.4] | 0.000 ; 1.000 | supérieur |
| règle principale moins Qwen/Qwen3.5-9B | +8.9 [+4.5 ; +13.7] | 0.000 ; 1.000 | supérieur |
| règle principale moins google/gemma-4-12B-it | +3.0 [-0.6 ; +6.9] | 0.005 ; 0.720 | non inférieur (-2) |
| règle principale moins mistralai/Ministral-3-14B-Instruct-2512 | +7.4 [+3.1 ; +12.1] | 0.000 ; 0.993 | supérieur |
| règle principale moins Qwen/Qwen3.8-27B | +1.1 [-2.5 ; +4.9] | 0.046 ; 0.372 | non inférieur (-2) |
| règle principale moins Qwen/Qwen3.5-9B + sélection | +2.6 [-1.1 ; +6.5] | 0.010 ; 0.703 | non inférieur (-2) |
| règle principale moins google/gemma-4-12B-it + sélection | +1.9 [-1.6 ; +5.5] | 0.016 ; 0.545 | non inférieur (-2) |
| règle principale moins mistralai/Ministral-3-14B-Instruct-2512 + sélection | +4.4 [+0.4 ; +8.8] | 0.002 ; 0.905 | supérieur |
| règle principale moins Qwen/Qwen3.8-27B + sélection | -1.1 [-4.6 ; +2.2] | 0.371 ; 0.034 | indéterminé |
| cascade (m choisi) moins meilleur pair (a) | +7.0 [+4.2 ; +10.8] | 0.000 ; 1.000 | supérieur |
| cascade (m choisi) moins Qwen/Qwen3.5-9B | +9.3 [+4.9 ; +14.1] | 0.000 ; 1.000 | supérieur |
| cascade (m choisi) moins google/gemma-4-12B-it | +3.3 [-0.3 ; +7.3] | 0.003 ; 0.824 | non inférieur (-2) |
| cascade (m choisi) moins mistralai/Ministral-3-14B-Instruct-2512 | +7.8 [+3.4 ; +12.5] | 0.000 ; 0.997 | supérieur |
| cascade (m choisi) moins Qwen/Qwen3.8-27B | +1.5 [-2.1 ; +5.2] | 0.028 ; 0.413 | non inférieur (-2) |
| cascade (m choisi) moins Qwen/Qwen3.5-9B + sélection | +3.0 [-0.8 ; +7.0] | 0.006 ; 0.714 | non inférieur (-2) |
| cascade (m choisi) moins google/gemma-4-12B-it + sélection | +2.2 [-1.3 ; +6.0] | 0.010 ; 0.579 | non inférieur (-2) |
| cascade (m choisi) moins mistralai/Ministral-3-14B-Instruct-2512 + sélection | +4.8 [+0.7 ; +9.3] | 0.001 ; 0.914 | supérieur |
| cascade (m choisi) moins Qwen/Qwen3.8-27B + sélection | -0.7 [-4.1 ; +2.6] | 0.233 ; 0.054 | indéterminé |

Pour comparaison, collision des réponses fausses en mathématiques et QCM (E4, dev) : gsm8k 0.460, math500 0.162, arc 0.711, mmlupro 0.390.
