# Expérience 2 : écrire ensemble (GSM8K, jeu dev)

100 questions, une réponse par question et par mode. Pairs : Qwen/Qwen3-1.7B, ibm-granite/granite-3.3-2b-instruct, HuggingFaceTB/SmolLM3-3B, google/gemma-4-E2B-it. Blocs de 16 jetons, au plus 40 tours, k = tous, seuil du mode croisé tau = -2.5, 320 jetons au plus pour une réponse seule. IC à 95 % : bootstrap apparié par question. Le meilleur pair seul est choisi sur ces mêmes données.

## Exactitude

| système | exactitude (%) |
| --- | --- |
| Qwen/Qwen3-1.7B seul | 68.0 |
| ibm-granite/granite-3.3-2b-instruct seul | 69.0 |
| HuggingFaceTB/SmolLM3-3B seul (meilleur) | 82.0 |
| google/gemma-4-E2B-it seul | 70.0 |
| **essaim : vote (majorité des réponses seules)** | **86.0** |
| **essaim : accord (préfixe commun, 1 aller-retour par tour)** | **83.0** |
| **essaim : croisé (brouillons notés par tous, 2 allers-retours par tour)** | **87.0** |
| Qwen/Qwen3-4B seul (référence locale) | 84.0 |
| plafond (au moins un pair seul a raison) | 93.0 |

Gain sur le meilleur pair seul (HuggingFaceTB/SmolLM3-3B), en points :

- vote : +4.0 [-2.0 ; +10.0]
- accord : +1.0 [-7.0 ; +9.0]
- croise : +5.0 [-2.0 ; +12.0]

Gain sur la référence (Qwen/Qwen3-4B) :

- vote : +2.0 [-5.0 ; +9.0]
- accord : -1.0 [-9.0 ; +7.0]
- croise : +3.0 [-4.0 ; +11.0]

## Coût

Temps de décision mesuré (pairs sur une seule machine), puis temps projeté sur un réseau étendu : temps mesuré + allers-retours × RTT.

| mode | allers-retours (moy. / max) | tours | temps mesuré (moy. / méd., s) | projeté RTT 50 ms (s) | projeté RTT 100 ms (s) | projeté RTT 150 ms (s) | appels en échec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| solo | 1.0 / 1 | 1.0 | 5.8 / 5.3 | 5.8 | 5.9 | 5.9 | 0 / 400 |
| vote | 1.0 / 1 | 1.0 | 5.8 / 5.3 | 5.8 | 5.9 | 5.9 | 0 / 400 |
| accord | 19.9 / 40 | 19.9 | 8.9 / 8.1 | 9.9 | 10.8 | 11.8 | 0 / 7939 |
| croise | 34.1 / 80 | 17.5 | 35.4 / 32.3 | 37.1 | 38.8 | 40.5 | 51 / 13627 |

Le vote ne coûte rien de plus que les réponses seules : il les réutilise (1 aller-retour au total).
