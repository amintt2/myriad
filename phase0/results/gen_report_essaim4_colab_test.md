# Expérience 2 : écrire ensemble (GSM8K, jeu test)

200 questions, une réponse par question et par mode. Pairs : Qwen/Qwen3-1.7B, ibm-granite/granite-3.3-2b-instruct, HuggingFaceTB/SmolLM3-3B, google/gemma-4-E2B-it. Blocs de 16 jetons, au plus 40 tours, k = tous, seuil du mode croisé tau = -2.5, 320 jetons au plus pour une réponse seule. IC à 95 % : bootstrap apparié par question. Le meilleur pair seul est choisi sur ces mêmes données.

## Exactitude

| système | exactitude (%) |
| --- | --- |
| Qwen/Qwen3-1.7B seul | 69.5 |
| ibm-granite/granite-3.3-2b-instruct seul | 73.5 |
| HuggingFaceTB/SmolLM3-3B seul (meilleur) | 86.5 |
| google/gemma-4-E2B-it seul | 74.0 |
| **essaim : vote (majorité des réponses seules)** | **89.0** |
| **essaim : accord (préfixe commun, 1 aller-retour par tour)** | **87.0** |
| **essaim : croisé (brouillons notés par tous, 2 allers-retours par tour)** | **89.5** |
| Qwen/Qwen3-4B seul (référence locale) | 91.0 |
| plafond (au moins un pair seul a raison) | 95.0 |

Gain sur le meilleur pair seul (HuggingFaceTB/SmolLM3-3B), en points :

- vote : +2.5 [-1.5 ; +6.5]
- accord : +0.5 [-5.0 ; +5.5]
- croise : +3.0 [-1.5 ; +7.5]

Gain sur la référence (Qwen/Qwen3-4B) :

- vote : -2.0 [-6.0 ; +2.0]
- accord : -4.0 [-8.5 ; +0.5]
- croise : -1.5 [-5.5 ; +2.5]

## Coût

Temps de décision mesuré (pairs sur une seule machine), puis temps projeté sur un réseau étendu : temps mesuré + allers-retours × RTT.

| mode | allers-retours (moy. / max) | tours | temps mesuré (moy. / méd., s) | projeté RTT 50 ms (s) | projeté RTT 100 ms (s) | projeté RTT 150 ms (s) | appels en échec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| solo | 1.0 / 1 | 1.0 | 5.1 / 5.0 | 5.2 | 5.2 | 5.3 | 0 / 800 |
| vote | 1.0 / 1 | 1.0 | 5.1 / 5.0 | 5.2 | 5.2 | 5.3 | 0 / 800 |
| accord | 19.7 / 40 | 19.7 | 8.2 / 7.5 | 9.2 | 10.2 | 11.2 | 0 / 15742 |
| croise | 33.1 / 80 | 17.1 | 34.6 / 31.9 | 36.2 | 37.9 | 39.6 | 60 / 26498 |

Le vote ne coûte rien de plus que les réponses seules : il les réutilise (1 aller-retour au total).
