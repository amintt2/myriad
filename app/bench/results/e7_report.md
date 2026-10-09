# E7 : passage à l'échelle du protocole et du traqueur

> Généré par `uv run python -m bench.bench_scale report` depuis les fichiers `e7_*.json` de ce dossier.
> Aucun chiffre n'est recopié à la main.

## Montage

- Machine : Windows-11-10.0.26200-SP0, 12 fils, Python 3.12.13. Tout tourne sur ce seul PC, sans GPU.
- **Traqueur réel** (code de production, réglages par défaut : contrôles aléatoires 5 %, balayage 0,25 s, jobs gardés 600 s, registre SQLite sur disque), dans **son propre processus** : son temps CPU est mesuré seul (`time.process_time`). Exception : crédit de départ 10⁹ pour que les demandeurs ne tombent jamais à court.
- **Nœuds simulés** : vrais `NodeClient` (WebSocket, signatures ed25519) avec `FakeEngine`, répartis sur 1 à 4 processus ; un seul job à la fois par nœud (`max_parallel = 1`, un GPU grand public).
- **Demandeurs** : vraies `Gateway` (sélection des pairs, vote pondéré, certificat d'arrêt, reçus), dans 1 à 3 processus, arrivées de Poisson (charge ouverte). Délai par requête 60 s, annuaire des pairs gardé 2 s (valeur de production) sauf mention.
- **Modèle de réponses** : le pair i a raison avec la probabilité p_i de sa famille ; s'il se trompe, il donne la mauvaise réponse « populaire » de la question avec la probabilité √c, sinon une réponse à lui. Deux pairs qui se trompent coïncident donc avec la probabilité c = 0.09 (valeur mesurée sur GSM8K). Les familles mesurées en phase 0 : smollm p = 0.865, gemma p = 0.74, qwen p = 0.535, granite p = 0.495. Pour k > 4, familles synthétiques plus faibles (p = 0.45).
- **Temps de calcul** : lognormal, médiane mesurée par famille (smollm 3.54 s, gemma 5.02 s, qwen 3.91 s, granite 3.91 s), σ = 0.36 (phase 0, GSM8K test, un GPU). **Facteur d'échelle** indiqué pour chaque série : 1 = temps mesurés ; 0,1 et 0,025 = temps divisés par 10 et 40 pour charger le traqueur en un temps raisonnable (les RTT, eux, ne sont pas réduits).
- **WAN émulé** dans le relais du traqueur (`myriad/netem.py`) : chaque trame reçue et chaque trame envoyée attend un retard aller simple lognormal (σ = 0.25), ordre conservé ; « RTT » = aller-retour médian nominal client–traqueur (2 × la médiane aller simple). Une requête traverse 4 sauts retardés (demandeur → traqueur → nœud → traqueur → demandeur), soit 2 RTT, plus l'annuaire HTTP (retardé aussi).
- Exactitude : réponse fusionnée comparée à la vraie réponse (oracle). « Attendue » : exactitude Monte-Carlo du vote complet pondéré avec les poids a priori, quand tous les pairs répondent.

## A. Latence de décision : certificat d'arrêt contre attente de tous les pairs

Temps de calcul mesurés (échelle 1), 256 nœuds, 2 requêtes/s (charge légère), mêmes questions dans les deux modes.

| scénario | RTT (ms) | k | mode | requêtes | p50 (s) | p95 (s) | p99 (s) | pairs attendus | arrêt anticipé | exactitude | attendue | pairs refusés (occupés) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lat_rtt0_k4 | 0 | 4 | certificat | 131 | 4.74 | 7.83 | 9.29 | 3.05 / 4.00 | 63.4 % | 95.4 % | 93.3 % | 1.7 % |
| lat_rtt0_k4 | 0 | 4 | attendre tous | 136 | 6.27 | 9.45 | 11.57 | 3.93 / 4.00 | 0.0 % | 94.1 % | 93.3 % | 1.7 % |
| lat_rtt50_k4 | 50 | 4 | certificat | 131 | 4.80 | 7.76 | 9.12 | 2.91 / 4.00 | 71.0 % | 93.9 % | 93.3 % | 1.1 % |
| lat_rtt50_k4 | 50 | 4 | attendre tous | 136 | 6.45 | 9.90 | 12.42 | 3.90 / 4.00 | 0.0 % | 94.8 % | 93.3 % | 2.6 % |
| lat_rtt100_k4 | 100 | 4 | certificat | 131 | 4.99 | 8.42 | 10.27 | 3.09 / 4.00 | 61.1 % | 90.8 % | 93.3 % | 1.0 % |
| lat_rtt100_k4 | 100 | 4 | attendre tous | 136 | 6.26 | 10.11 | 11.82 | 3.93 / 4.00 | 0.0 % | 91.9 % | 93.3 % | 1.8 % |
| lat_rtt150_k4 | 150 | 4 | certificat | 131 | 5.20 | 8.06 | 11.71 | 2.94 / 4.00 | 64.1 % | 92.4 % | 93.3 % | 2.1 % |
| lat_rtt150_k4 | 150 | 4 | attendre tous | 136 | 6.59 | 10.09 | 11.64 | 3.95 / 4.00 | 0.0 % | 93.4 % | 93.3 % | 1.3 % |
| lat_rtt100_k5 | 100 | 5 | certificat | 131 | 5.04 | 7.88 | 8.63 | 3.82 / 5.00 | 74.8 % | 97.7 % | 94.5 % | 2.1 % |
| lat_rtt100_k5 | 100 | 5 | attendre tous | 136 | 6.44 | 9.91 | 11.14 | 4.86 / 5.00 | 0.0 % | 94.1 % | 94.5 % | 2.8 % |
| lat_rtt100_k6 | 100 | 6 | certificat | 131 | 5.34 | 8.44 | 9.32 | 4.53 / 6.00 | 85.5 % | 95.4 % | 95.4 % | 2.3 % |
| lat_rtt100_k6 | 100 | 6 | attendre tous | 136 | 6.74 | 9.82 | 10.83 | 5.81 / 6.00 | 0.0 % | 95.6 % | 95.4 % | 3.2 % |
| lat_rtt100_k7 | 100 | 7 | certificat | 131 | 5.39 | 9.72 | 11.80 | 5.38 / 7.00 | 77.1 % | 97.7 % | 96.0 % | 2.9 % |
| lat_rtt100_k7 | 100 | 7 | attendre tous | 136 | 6.95 | 10.97 | 12.36 | 6.75 / 7.00 | 0.0 % | 98.5 % | 96.0 % | 3.6 % |
| lat_rtt100_k4_hetero | 100 | 4 | certificat | 131 | 5.57 | 13.36 | 17.78 | 3.02 / 4.00 | 63.4 % | 92.4 % | 93.3 % | 1.0 % |
| lat_rtt100_k4_hetero | 100 | 4 | attendre tous | 136 | 8.09 | 15.81 | 25.90 | 3.91 / 4.00 | 0.0 % | 92.7 % | 93.3 % | 2.2 % |

Les deux modes reçoivent les mêmes questions (numérotées depuis 0) ; seules les dates d'arrivée (Poisson) diffèrent, d'où des nombres de requêtes un peu différents.

Gain du certificat (médiane et p95, en secondes) :

| scénario | p50 attendre tous | p50 certificat | gain p50 | p95 attendre tous | p95 certificat | gain p95 |
| --- | --- | --- | --- | --- | --- | --- |
| lat_rtt0_k4 | 6.27 | 4.74 | 24.4 % | 9.45 | 7.83 | 17.1 % |
| lat_rtt50_k4 | 6.45 | 4.80 | 25.6 % | 9.90 | 7.76 | 21.6 % |
| lat_rtt100_k4 | 6.26 | 4.99 | 20.4 % | 10.11 | 8.42 | 16.7 % |
| lat_rtt150_k4 | 6.59 | 5.20 | 21.1 % | 10.09 | 8.06 | 20.1 % |
| lat_rtt100_k5 | 6.44 | 5.04 | 21.7 % | 9.91 | 7.88 | 20.5 % |
| lat_rtt100_k6 | 6.74 | 5.34 | 20.8 % | 9.82 | 8.44 | 14.1 % |
| lat_rtt100_k7 | 6.95 | 5.39 | 22.5 % | 10.97 | 9.72 | 11.3 % |
| lat_rtt100_k4_hetero | 8.09 | 5.57 | 31.2 % | 15.81 | 13.36 | 15.5 % |

## B. Nombre de nœuds × RTT

Temps de calcul / 10 (échelle 0,1), k = 4, charge fixée à 0,15 requête/s par nœud, plafonnée à 12 requêtes/s (sous la saturation du traqueur à 1024 nœuds, voir C).

| N | RTT (ms) | req/s offertes | servies | p50 (s) | p95 (s) | p99 (s) | pairs attendus | exactitude | CPU traqueur | ms CPU / requête | trames / requête | annuaire (ms / appel) | appels annuaire / s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0 | 0.5 | 0.5 | 0.49 | 0.77 | 0.99 | 2.94 | 92.0 % | 1.0 % | 19.1 | 19.0 | 0.2 | 0.3 |
| 4 | 50 | 0.5 | 0.4 | 0.71 | 1.10 | 1.27 | 2.77 | 88.0 % | 1.2 % | 23.8 | 18.2 | 0.2 | 0.4 |
| 4 | 100 | 0.5 | 0.4 | 0.90 | 1.23 | 1.45 | 2.79 | 80.0 % | 1.4 % | 28.1 | 17.7 | 0.2 | 0.4 |
| 4 | 150 | 0.5 | 0.4 | 1.12 | 1.44 | 1.61 | 2.75 | 74.0 % | 1.0 % | 20.9 | 17.4 | 0.2 | 0.4 |
| 16 | 0 | 2.0 | 2.0 | 0.45 | 0.75 | 0.96 | 2.74 | 98.0 % | 2.2 % | 11.2 | 19.0 | 0.7 | 0.8 |
| 16 | 50 | 2.0 | 2.0 | 0.65 | 1.00 | 1.19 | 2.68 | 90.0 % | 3.4 % | 18.1 | 19.9 | 0.7 | 0.9 |
| 16 | 100 | 2.0 | 2.0 | 0.81 | 1.14 | 1.20 | 2.49 | 88.0 % | 3.3 % | 17.5 | 19.3 | 0.7 | 1.0 |
| 16 | 150 | 2.0 | 1.9 | 0.98 | 1.38 | 1.46 | 2.47 | 94.0 % | 2.9 % | 15.0 | 18.7 | 0.7 | 1.1 |
| 64 | 0 | 8.9 | 8.8 | 0.48 | 0.79 | 0.96 | 2.75 | 89.9 % | 10.0 % | 11.9 | 18.7 | 2.6 | 3.0 |
| 64 | 50 | 8.9 | 8.7 | 0.66 | 1.06 | 1.24 | 2.56 | 88.3 % | 12.2 % | 14.8 | 18.7 | 2.6 | 3.1 |
| 64 | 100 | 8.9 | 8.6 | 0.77 | 1.13 | 1.23 | 2.43 | 88.8 % | 8.9 % | 11.0 | 18.5 | 2.6 | 3.1 |
| 64 | 150 | 8.9 | 8.4 | 0.94 | 1.39 | 1.56 | 2.49 | 85.5 % | 8.7 % | 10.7 | 18.6 | 2.6 | 3.4 |
| 256 | 0 | 11.3 | 11.0 | 0.47 | 0.77 | 1.16 | 2.86 | 88.9 % | 13.7 % | 13.1 | 20.3 | 10.8 | 3.2 |
| 256 | 50 | 11.3 | 10.7 | 0.63 | 0.99 | 1.23 | 2.85 | 93.8 % | 13.7 % | 13.4 | 20.8 | 10.8 | 3.2 |
| 256 | 100 | 11.3 | 10.8 | 0.74 | 1.12 | 1.27 | 2.85 | 92.9 % | 14.2 % | 13.9 | 21.4 | 10.8 | 3.5 |
| 256 | 150 | 11.3 | 10.8 | 0.91 | 1.48 | 1.63 | 2.81 | 91.6 % | 18.3 % | 17.9 | 21.2 | 10.8 | 3.9 |
| 1024 | 0 | 11.3 | 11.0 | 0.56 | 0.89 | 1.07 | 3.12 | 91.1 % | 31.8 % | 30.2 | 24.2 | 43.0 | 3.4 |
| 1024 | 50 | 11.3 | 10.8 | 0.73 | 1.40 | 1.68 | 2.88 | 92.0 % | 35.7 % | 34.7 | 25.2 | 43.0 | 4.2 |
| 1024 | 100 | 11.3 | 10.8 | 0.85 | 1.37 | 1.52 | 2.95 | 93.4 % | 32.0 % | 31.3 | 25.4 | 43.0 | 4.0 |
| 1024 | 150 | 11.3 | 10.7 | 1.08 | 1.66 | 2.00 | 3.08 | 91.1 % | 34.3 % | 33.7 | 25.6 | 43.0 | 4.5 |

« ms CPU / requête » = tout le CPU du traqueur divisé par le nombre de requêtes : il inclut le travail de fond (battements de cœur des nœuds toutes les 15 s, balayage, sonde de latence), qui domine aux faibles débits (N = 4, 16). Annuaire : section E.

## C. Débit : montée en charge jusqu'à saturation

Temps de calcul / 40 (échelle 0,025, médiane ≈ 0,1 s) pour que les nœuds ne soient pas le goulot ; 24 passerelles dans 3 processus ; paliers de 15 s ; arrêt au premier palier saturé (débit servi < 90 % de l'offert, p95 > 3 × celui du premier palier + 1 s, ou plus de 5 % d'échecs).

### tput_n64_rtt0 (N = 64, RTT 0 ms, annuaire gardé 2 s)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs occupés | CPU traqueur | ms CPU / requête | µs CPU / trame | appels annuaire / s | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.14 | 0.23 | 95.4 % | 1.9 % | 4.8 % | 11.8 | 560 | 1.6 | 21 | 2.5 % |
| 9.9 | 9.9 | 0 | 0.13 | 0.23 | 94.0 % | 2.9 % | 9.7 % | 10.5 | 518 | 2.9 | 20 | 4.1 % |
| 14.9 | 14.8 | 0 | 0.13 | 0.21 | 91.5 % | 8.0 % | 9.7 % | 7.0 | 358 | 3.4 | 20 | 3.8 % |
| 20.5 | 20.4 | 2 | 0.13 | 0.22 | 89.6 % | 9.6 % | 16.4 % | 8.5 | 446 | 4.5 | 23 | 5.2 % |
| 25.6 | 25.3 | 1 | 0.13 | 0.22 | 89.1 % | 12.3 % | 20.3 % | 8.5 | 455 | 5.4 | 21 | 6.6 % |
| 30.4 | 30.2 | 0 | 0.13 | 0.22 | 91.0 % | 14.5 % | 22.7 % | 8.0 | 433 | 5.8 | 23 | 7.0 % |
| 39.9 | 39.1 | 3 | 0.12 | 0.22 | 86.0 % | 18.1 % | 25.3 % | 6.9 | 382 | 7.0 | 24 | 8.9 % |
| 52.6 | 51.9 | 3 | 0.12 | 0.22 | 86.4 % | 22.8 % | 35.3 % | 7.2 | 417 | 8.0 | 22 | 11.2 % |
| 62.0 | 60.7 | 13 | 0.13 | 0.22 | 85.9 % | 26.0 % | 37.0 % | 6.4 | 379 | 8.8 | 22 | 11.4 % |
| 78.0 | 75.2 | 30 | 0.13 | 0.24 | 81.6 % | 32.1 % | 45.2 % | 6.2 | 386 | 10.5 | 23 | 12.6 % |
| 100.1 | 95.2 | 57 | 0.13 | 0.24 | 77.5 % | 36.7 % | 52.5 % | 5.7 | 368 | 11.0 | 24 | 14.4 % |
| 129.6 | 120.7 | 111 | 0.13 | 0.25 | 75.6 % | 43.4 % | 63.4 % | 5.3 | 363 | 12.0 | 22 | 17.8 % |

### tput_n256_rtt0 (N = 256, RTT 0 ms, annuaire gardé 2 s)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs occupés | CPU traqueur | ms CPU / requête | µs CPU / trame | appels annuaire / s | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.14 | 0.25 | 90.8 % | 0.0 % | 7.3 % | 18.0 | 741 | 1.7 | 16 | 2.9 % |
| 9.9 | 9.9 | 0 | 0.13 | 0.24 | 90.6 % | 1.5 % | 14.6 % | 15.8 | 724 | 2.7 | 21 | 4.7 % |
| 14.9 | 14.9 | 0 | 0.13 | 0.22 | 92.4 % | 2.2 % | 16.2 % | 11.6 | 553 | 3.3 | 23 | 4.5 % |
| 20.5 | 20.5 | 0 | 0.14 | 0.25 | 95.8 % | 3.7 % | 21.8 % | 11.3 | 548 | 4.6 | 21 | 6.2 % |
| 25.6 | 25.1 | 0 | 0.13 | 0.24 | 93.2 % | 3.3 % | 24.4 % | 10.4 | 500 | 5.3 | 24 | 6.9 % |
| 30.4 | 30.3 | 0 | 0.13 | 0.24 | 90.8 % | 3.7 % | 29.2 % | 10.3 | 511 | 5.9 | 22 | 7.3 % |
| 39.9 | 39.3 | 0 | 0.14 | 0.27 | 91.3 % | 6.0 % | 36.3 % | 9.9 | 496 | 7.7 | 25 | 10.0 % |
| 52.6 | 51.8 | 0 | 0.14 | 0.30 | 90.8 % | 8.5 % | 46.1 % | 9.5 | 485 | 8.9 | 29 | 13.0 % |
| 62.0 | 61.2 | 0 | 0.15 | 0.39 | 91.1 % | 10.2 % | 53.7 % | 9.4 | 481 | 10.3 | 49 | 13.2 % |
| 78.0 | 76.7 | 1 | 0.17 | 0.51 | 90.3 % | 13.6 % | 65.4 % | 9.1 | 478 | 12.4 | 70 | 16.7 % |
| 100.1 | 58.8 | 212 | 1.37 | 9.32 | 73.0 % | 27.9 % | 86.6 % | 13.2 | 859 | 31.5 | 629 | 28.9 % |

### tput_n1024_rtt0 (N = 1024, RTT 0 ms, annuaire gardé 2 s)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs occupés | CPU traqueur | ms CPU / requête | µs CPU / trame | appels annuaire / s | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.16 | 0.30 | 90.8 % | 0.4 % | 13.7 % | 33.6 | 940 | 1.7 | 40 | 3.2 % |
| 9.9 | 9.9 | 0 | 0.17 | 0.41 | 98.0 % | 0.2 % | 25.2 % | 27.2 | 997 | 3.2 | 52 | 4.6 % |
| 14.9 | 14.7 | 0 | 0.17 | 0.36 | 92.4 % | 0.9 % | 32.5 % | 23.7 | 958 | 3.8 | 89 | 5.9 % |
| 20.5 | 17.7 | 0 | 0.94 | 4.24 | 89.6 % | 3.7 % | 72.6 % | 43.5 | 1816 | 9.8 | 513 | 12.1 % |

### tput_n1024_rtt100 (N = 1024, RTT 100 ms, annuaire gardé 2 s)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs occupés | CPU traqueur | ms CPU / requête | µs CPU / trame | appels annuaire / s | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.63 | 1.26 | 86.2 % | 0.4 % | 19.4 % | 49.0 | 1317 | 2.4 | 79 | 4.4 % |
| 9.9 | 9.6 | 0 | 0.76 | 2.74 | 93.3 % | 3.0 % | 43.9 % | 49.2 | 1485 | 5.7 | 203 | 6.5 % |
| 14.9 | 14.2 | 0 | 0.69 | 1.32 | 94.2 % | 2.4 % | 50.5 % | 38.0 | 1467 | 6.6 | 108 | 9.4 % |
| 20.5 | 15.2 | 0 | 6.34 | 8.62 | 90.9 % | 4.7 % | 88.3 % | 61.3 | 2231 | 13.2 | 605 | 14.7 % |

### tput_n1024_rtt0_ttl30 (N = 1024, RTT 0 ms, annuaire gardé 30 s)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs occupés | CPU traqueur | ms CPU / requête | µs CPU / trame | appels annuaire / s | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.14 | 0.28 | 93.8 % | 0.0 % | 7.5 % | 18.5 | 513 | 0.4 | 25 | 1.9 % |
| 9.9 | 9.9 | 0 | 0.14 | 0.25 | 95.3 % | 0.2 % | 11.6 % | 12.5 | 465 | 0.6 | 41 | 2.8 % |
| 14.9 | 14.9 | 0 | 0.13 | 0.22 | 94.2 % | 0.7 % | 15.8 % | 11.3 | 462 | 0.4 | 24 | 3.6 % |
| 20.5 | 20.4 | 0 | 0.13 | 0.26 | 94.2 % | 1.0 % | 19.5 % | 10.2 | 437 | 0.8 | 46 | 4.7 % |
| 25.6 | 25.4 | 0 | 0.13 | 0.23 | 95.0 % | 1.3 % | 24.0 % | 10.1 | 440 | 0.9 | 41 | 5.8 % |
| 30.4 | 30.0 | 0 | 0.13 | 0.23 | 92.3 % | 0.9 % | 24.3 % | 8.6 | 392 | 0.6 | 41 | 6.4 % |
| 39.9 | 39.3 | 0 | 0.13 | 0.28 | 93.7 % | 1.9 % | 34.2 % | 9.3 | 423 | 1.0 | 51 | 9.3 % |
| 52.6 | 51.8 | 0 | 0.15 | 0.66 | 91.8 % | 2.8 % | 49.1 % | 10.1 | 476 | 1.8 | 116 | 12.1 % |
| 62.0 | 61.5 | 0 | 0.14 | 0.26 | 92.4 % | 2.4 % | 51.0 % | 8.8 | 421 | 0.7 | 55 | 12.5 % |
| 78.0 | 77.1 | 0 | 0.94 | 1.86 | 90.0 % | 11.5 % | 73.6 % | 10.2 | 498 | 3.8 | 488 | 17.9 % |

### Débit soutenu

| scénario | N | RTT (ms) | annuaire gardé (s) | débit soutenu (req/s) | CPU traqueur à ce débit | ms CPU / requête | coût d'un appel à l'annuaire (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tput_n64_rtt0 | 64 | 0 | 2 | 95.2 | 52.5 % | 5.7 | 2.6 |
| tput_n256_rtt0 | 256 | 0 | 2 | 76.7 | 65.4 % | 9.1 | 10.8 |
| tput_n1024_rtt0 | 1024 | 0 | 2 | 14.7 | 32.5 % | 23.7 | 43.0 |
| tput_n1024_rtt100 | 1024 | 100 | 2 | 14.2 | 50.5 % | 38.0 | 43.0 |
| tput_n1024_rtt0_ttl30 | 1024 | 0 | 30 | 61.5 | 51.0 % | 8.8 | 43.0 |

## D. Pannes : 25 % des nœuds tombent pendant la mesure

Temps de calcul mesurés (échelle 1), 256 nœuds, RTT 50 ms, 2,5 requêtes/s pendant 120 s, panne à 40 s. « crash » : connexion coupée net (TCP réinitialisé) ; « hang » : le nœud reste connecté mais ne répond plus. Délai par requête 30 s.

| mode | fenêtre | requêtes | servies | exactitude | p50 (s) | p95 (s) | max (s) | pairs attendus | erreurs des pairs | échecs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crash | before | 87 | 87 | 93.1 % | 5.06 | 7.73 | 12.48 | 3.05 | peer_busy 6, peer_disconnected 10 | – |
| crash | during_10s | 16 | 16 | 100.0 % | 4.14 | 7.36 | 7.50 | 3.00 | – | – |
| crash | after | 189 | 189 | 88.9 % | 4.97 | 8.04 | 14.62 | 2.94 | peer_busy 36 | – |
| hang | before | 87 | 87 | 92.0 % | 5.07 | 7.81 | 13.86 | 2.92 | peer_busy 10 | – |
| hang | during_10s | 16 | 16 | 100.0 % | 5.18 | 30.32 | 30.32 | 2.50 | peer_busy 1, deadline 6 | – |
| hang | after | 189 | 189 | 90.0 % | 5.51 | 30.29 | 30.33 | 2.61 | peer_busy 33, deadline 60 | – |

## E. Coût de l'annuaire des pairs selon N

Mesuré sur Windows-11-10.0.26200-SP0 (12 fils), machine sans autre charge. Temps de construction d'une réponse à `GET /v1/peers` (tous les pairs, comme FastAPI la produit : vue de chaque pair avec deux requêtes SQLite, puis `jsonable_encoder` et JSON), moyenne sur 20 appels : temps écoulé (la boucle du traqueur est bloquée pendant ce temps) et temps CPU du processus sur le lot (le compteur CPU de Windows avance par 15,6 ms : valeur fiable seulement aux grands N). Puis le temps vu par un client HTTP local (médiane de 20 appels, RTT 0). Les sections B et C reprennent le temps écoulé quand leurs mesures viennent de la même machine.

| N | vue des pairs (ms) | encodage (ms) | total écoulé (ms) | total CPU (ms) | HTTP, vu du client (ms) | taille (octets) | /v1/reliability (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.08 | 0.10 | 0.18 | 0.78 | 1.36 | 1316 | 0.13 |
| 16 | 0.28 | 0.38 | 0.67 | 0.78 | 1.88 | 5231 | 0.13 |
| 64 | 1.12 | 1.50 | 2.62 | 2.34 | 4.17 | 20891 | 0.14 |
| 256 | 4.68 | 6.09 | 10.77 | 10.94 | 12.78 | 83531 | 0.16 |
| 1024 | 18.63 | 24.33 | 42.96 | 42.97 | 45.43 | 334091 | 0.29 |

## v1.1 : sélection par le traqueur, nœuds figés détectés, pairs refusés remplacés

Protocole essaim/1.1 (compatible avec essaim/1). « Avant » : les fichiers `e7_*.json` d'E7 (essaim/1, mesurés le 2026-10-09) ; « après » : les fichiers `e7_v11_*.json` (mesurés le 2026-10-09, Windows-11-10.0.26200-SP0, 12 fils). Même montage, mêmes modèles de réponses et de temps de calcul, mêmes réglages de production (annuaire gardé 2 s par les passerelles essaim/1, délai 60 s par requête sauf mention).

Ce qui change :

- **Sélection côté traqueur** : la passerelle envoie ses k jobs sans cible ; le traqueur choisit chaque pair en O(k) dans des index par modèle (nœuds connectés, acceptant des jobs, non suspendus, avec un créneau libre), une famille différente par job, le modèle le plus fiable d'abord, le meilleur de deux tirages au hasard (moins chargé, moins de manquements). Une trame `assigned` nomme le pair avant tout autre message sur ce job. La passerelle ne télécharge plus l'annuaire ; `GET /v1/peers` est servi depuis un instantané reconstruit par morceaux en arrière-plan.
- **Nœuds figés** : ping applicatif toutes les ~10 s (pong attendu 5 s, après une sonde du moteur) ; un nœud muet, au moteur en panne, ou qui dépasse deux fois de suite le délai d'un job (délai ≥ 10 s) est suspendu 10 s × 2^niveau (300 s au plus), puis réadmis après un pong sain.
- **Pairs refusés** : un pair qui refuse (occupé, en pause) ou échoue vite est remplacé une fois, dans une famille pas encore utilisée, sinon la même, tant qu'il reste un quart du délai. Le certificat d'arrêt compte tous les jobs en attente, remplaçants compris, et n'est évalué qu'une fois leurs poids connus : la décision reste exactement celle du vote complet.
- Le balayage du traqueur n'examine plus tous les jobs gardés (600 s) à chaque passage : tas d'échéances et files.

### V1. Débit soutenu (montée en charge, temps de calcul / 40)

Même échelle de charge qu'en C (paliers de 15 s, 24 passerelles dans 3 processus, arrêt au premier palier saturé). Débit soutenu = débit servi au dernier palier non saturé ; p50 et p95 à ce palier.

| N | RTT (ms) | avant : req/s | avant : p50 (s) | avant : p95 (s) | après : req/s | après : p50 (s) | après : p95 (s) | CPU traqueur (après) | ms CPU / requête (après) | remarque |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 100 | – | – | – | 73.7 | 0.35 | 0.46 | 47.5 % | 6.7 | pas de mesure avant |
| 256 | 100 | – | – | – | 95.5 | 0.58 | 0.93 | 68.9 % | 7.7 | pas de mesure avant |
| 1024 | 100 | 14.2 | 0.69 | 1.32 | 92.3 | 0.68 | 1.08 | 70.4 % | 8.2 |  |
| 4096 | 100 | – | – | – | 75.2 | 0.51 | 0.76 | 71.6 % | 10.2 | pas de mesure avant |
| 64 | 0 | 95.2 | 0.13 | 0.24 | 126.0 | 0.18 | 0.28 | 73.9 % | 6.1 |  |
| 256 | 0 | 76.7 | 0.17 | 0.51 | 120.4 | 0.77 | 1.00 | 70.5 % | 6.1 |  |
| 1024 | 0 | 14.7 | 0.17 | 0.36 | 96.7 | 0.22 | 0.73 | 68.5 % | 7.6 |  |

« – » avant : configuration non mesurée en E7 (E7 n'a mesuré le débit à RTT 100 ms qu'à 1024 nœuds, et n'est pas allé au-delà de 1024 nœuds).

### V2. Détail des paliers (après)

#### tput_n64_rtt100 (N = 64, RTT 100 ms)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs interrogés | pairs refusés | remplacements / requête | CPU traqueur | ms CPU / requête | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.34 | 0.51 | 93.8 % | 4.00 | 0.0 % | 0.00 | 5.0 % | 12.5 | 21 | 2.1 % |
| 9.9 | 9.8 | 0 | 0.34 | 0.44 | 93.3 % | 4.00 | 0.0 % | 0.00 | 9.6 % | 10.5 | 21 | 2.3 % |
| 14.9 | 14.7 | 0 | 0.33 | 0.46 | 91.5 % | 4.00 | 0.0 % | 0.00 | 13.3 % | 9.8 | 19 | 2.5 % |
| 20.5 | 20.3 | 0 | 0.34 | 0.45 | 92.9 % | 4.00 | 0.0 % | 0.00 | 15.4 % | 8.2 | 23 | 5.0 % |
| 25.6 | 25.1 | 0 | 0.33 | 0.43 | 95.3 % | 4.00 | 0.0 % | 0.00 | 19.3 % | 8.3 | 18 | 5.1 % |
| 30.4 | 29.6 | 0 | 0.34 | 0.44 | 92.3 % | 4.00 | 0.0 % | 0.00 | 25.2 % | 9.2 | 22 | 5.4 % |
| 39.9 | 38.5 | 1 | 0.34 | 0.44 | 93.0 % | 3.95 | 0.0 % | 0.00 | 29.0 % | 8.1 | 22 | 6.8 % |
| 52.6 | 51.1 | 3 | 0.35 | 0.44 | 90.1 % | 3.82 | 0.0 % | 0.00 | 38.9 % | 8.1 | 24 | 7.7 % |
| 62.0 | 60.1 | 6 | 0.35 | 0.45 | 90.6 % | 3.65 | 0.0 % | 0.00 | 43.1 % | 7.7 | 23 | 8.9 % |
| 78.0 | 73.7 | 32 | 0.35 | 0.46 | 85.9 % | 3.25 | 0.0 % | 0.00 | 47.5 % | 6.7 | 24 | 11.0 % |
| 100.1 | 89.8 | 106 | 0.35 | 0.46 | 79.1 % | 2.83 | 0.0 % | 0.00 | 53.3 % | 5.9 | 26 | 11.5 % |

#### tput_n256_rtt100 (N = 256, RTT 100 ms)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs interrogés | pairs refusés | remplacements / requête | CPU traqueur | ms CPU / requête | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.35 | 0.49 | 90.8 % | 4.00 | 0.0 % | 0.00 | 4.5 % | 11.3 | 24 | 1.5 % |
| 9.9 | 9.7 | 0 | 0.33 | 0.45 | 94.0 % | 4.00 | 0.0 % | 0.00 | 10.0 % | 11.1 | 18 | 2.3 % |
| 14.9 | 14.7 | 0 | 0.34 | 0.45 | 95.5 % | 4.00 | 0.0 % | 0.00 | 12.1 % | 8.8 | 19 | 2.2 % |
| 20.5 | 20.2 | 0 | 0.34 | 0.42 | 94.8 % | 4.00 | 0.0 % | 0.00 | 17.4 % | 9.3 | 20 | 4.3 % |
| 25.6 | 25.0 | 0 | 0.34 | 0.44 | 93.2 % | 4.00 | 0.0 % | 0.00 | 19.2 % | 8.3 | 24 | 4.3 % |
| 30.4 | 29.7 | 0 | 0.34 | 0.43 | 93.4 % | 4.00 | 0.0 % | 0.00 | 22.9 % | 8.3 | 23 | 5.9 % |
| 39.9 | 38.8 | 0 | 0.35 | 0.45 | 93.5 % | 4.00 | 0.0 % | 0.00 | 33.1 % | 9.2 | 22 | 6.8 % |
| 52.6 | 51.2 | 0 | 0.35 | 0.45 | 92.9 % | 4.00 | 0.0 % | 0.00 | 39.2 % | 8.2 | 23 | 7.6 % |
| 62.0 | 60.5 | 0 | 0.36 | 0.46 | 92.5 % | 4.00 | 0.0 % | 0.00 | 45.4 % | 8.1 | 26 | 8.6 % |
| 78.0 | 76.1 | 0 | 0.39 | 0.50 | 92.7 % | 4.00 | 0.0 % | 0.00 | 56.5 % | 8.0 | 34 | 11.4 % |
| 100.1 | 95.5 | 0 | 0.58 | 0.93 | 94.1 % | 3.98 | 0.0 % | 0.00 | 68.9 % | 7.7 | 136 | 14.9 % |
| 129.6 | 115.6 | 95 | 0.92 | 1.24 | 87.3 % | 3.66 | 0.0 % | 0.00 | 70.0 % | 6.2 | 226 | 14.1 % |

#### tput_n1024_rtt100 (N = 1024, RTT 100 ms)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs interrogés | pairs refusés | remplacements / requête | CPU traqueur | ms CPU / requête | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.36 | 0.45 | 93.8 % | 4.00 | 0.0 % | 0.00 | 12.3 % | 30.8 | 21 | 2.4 % |
| 9.9 | 9.8 | 0 | 0.34 | 0.48 | 96.0 % | 4.00 | 0.0 % | 0.00 | 17.0 % | 18.7 | 24 | 3.6 % |
| 14.9 | 14.7 | 0 | 0.33 | 0.44 | 95.5 % | 4.00 | 0.0 % | 0.00 | 14.9 % | 10.9 | 23 | 2.4 % |
| 20.5 | 20.2 | 0 | 0.34 | 0.43 | 91.9 % | 4.00 | 0.0 % | 0.00 | 21.3 % | 11.3 | 24 | 4.9 % |
| 25.6 | 25.0 | 0 | 0.33 | 0.42 | 94.8 % | 4.00 | 0.0 % | 0.00 | 25.5 % | 10.9 | 20 | 4.9 % |
| 30.4 | 29.8 | 0 | 0.34 | 0.45 | 92.5 % | 4.00 | 0.0 % | 0.00 | 28.5 % | 10.3 | 20 | 5.6 % |
| 39.9 | 39.0 | 0 | 0.34 | 0.43 | 94.5 % | 4.00 | 0.0 % | 0.00 | 36.0 % | 9.9 | 20 | 6.8 % |
| 52.6 | 51.5 | 0 | 0.35 | 0.45 | 94.2 % | 4.00 | 0.0 % | 0.00 | 46.3 % | 9.7 | 25 | 9.6 % |
| 62.0 | 59.6 | 0 | 0.36 | 0.47 | 95.2 % | 4.00 | 0.0 % | 0.00 | 52.2 % | 9.4 | 27 | 10.5 % |
| 78.0 | 75.9 | 0 | 0.39 | 0.52 | 94.2 % | 4.00 | 0.0 % | 0.00 | 65.2 % | 9.2 | 41 | 11.9 % |
| 100.1 | 92.3 | 0 | 0.68 | 1.08 | 93.2 % | 4.00 | 0.0 % | 0.00 | 70.4 % | 8.2 | 193 | 11.8 % |
| 129.6 | 106.3 | 0 | 2.25 | 3.56 | 91.9 % | 4.00 | 0.0 % | 0.00 | 72.2 % | 7.4 | 673 | 13.9 % |

#### tput_n4096_rtt100 (N = 4096, RTT 100 ms)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs interrogés | pairs refusés | remplacements / requête | CPU traqueur | ms CPU / requête | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.35 | 0.48 | 92.3 % | 4.00 | 0.0 % | 0.00 | 18.2 % | 45.9 | 19 | 1.3 % |
| 9.9 | 9.8 | 0 | 0.35 | 0.46 | 96.6 % | 4.00 | 0.0 % | 0.00 | 20.3 % | 22.3 | 25 | 1.6 % |
| 14.9 | 14.6 | 0 | 0.34 | 0.44 | 93.3 % | 4.00 | 0.0 % | 0.00 | 24.9 % | 18.4 | 22 | 3.4 % |
| 20.5 | 20.3 | 0 | 0.34 | 0.45 | 90.3 % | 4.00 | 0.0 % | 0.00 | 27.4 % | 14.5 | 24 | 3.7 % |
| 25.6 | 25.1 | 0 | 0.34 | 0.52 | 93.0 % | 4.00 | 0.0 % | 0.00 | 34.1 % | 14.6 | 26 | 4.5 % |
| 30.4 | 29.9 | 0 | 0.34 | 0.44 | 93.9 % | 4.00 | 0.0 % | 0.00 | 35.5 % | 12.8 | 25 | 4.9 % |
| 39.9 | 38.8 | 0 | 0.35 | 0.55 | 92.6 % | 4.00 | 0.0 % | 0.00 | 47.9 % | 13.2 | 25 | 6.6 % |
| 52.6 | 51.4 | 0 | 0.35 | 0.45 | 94.5 % | 4.00 | 0.0 % | 0.00 | 57.1 % | 11.9 | 26 | 8.5 % |
| 62.0 | 60.9 | 0 | 0.37 | 0.47 | 93.5 % | 4.00 | 0.0 % | 0.00 | 65.0 % | 11.5 | 31 | 10.3 % |
| 78.0 | 75.2 | 0 | 0.51 | 0.76 | 93.8 % | 4.00 | 0.0 % | 0.00 | 71.6 % | 10.2 | 92 | 10.3 % |
| 100.1 | 87.6 | 0 | 1.64 | 2.64 | 93.3 % | 4.00 | 0.0 % | 0.00 | 73.7 % | 9.2 | 448 | 10.9 % |

#### tput_n64_rtt0 (N = 64, RTT 0 ms)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs interrogés | pairs refusés | remplacements / requête | CPU traqueur | ms CPU / requête | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.13 | 0.21 | 93.8 % | 4.00 | 0.0 % | 0.00 | 3.7 % | 9.1 | 19 | 1.6 % |
| 9.9 | 9.9 | 0 | 0.12 | 0.22 | 92.6 % | 4.00 | 0.0 % | 0.00 | 7.8 % | 8.4 | 19 | 1.6 % |
| 14.9 | 14.9 | 0 | 0.12 | 0.20 | 95.5 % | 4.00 | 0.0 % | 0.00 | 10.4 % | 7.5 | 23 | 2.5 % |
| 20.5 | 20.5 | 0 | 0.13 | 0.20 | 91.2 % | 4.00 | 0.0 % | 0.00 | 14.4 % | 7.5 | 22 | 2.5 % |
| 25.6 | 25.4 | 0 | 0.12 | 0.20 | 93.0 % | 4.00 | 0.0 % | 0.00 | 17.2 % | 7.2 | 25 | 3.6 % |
| 30.4 | 30.3 | 0 | 0.12 | 0.21 | 93.4 % | 4.00 | 0.0 % | 0.00 | 17.8 % | 6.3 | 22 | 3.7 % |
| 39.9 | 39.6 | 0 | 0.12 | 0.20 | 92.0 % | 4.00 | 0.0 % | 0.00 | 27.8 % | 7.5 | 25 | 5.6 % |
| 52.6 | 52.1 | 0 | 0.12 | 0.20 | 93.8 % | 4.00 | 0.0 % | 0.00 | 39.1 % | 8.0 | 24 | 8.4 % |
| 62.0 | 61.4 | 0 | 0.12 | 0.22 | 94.4 % | 4.00 | 0.0 % | 0.00 | 39.0 % | 6.8 | 22 | 8.5 % |
| 78.0 | 77.3 | 0 | 0.13 | 0.22 | 92.7 % | 3.96 | 0.0 % | 0.00 | 49.8 % | 6.9 | 21 | 11.4 % |
| 100.1 | 98.2 | 6 | 0.16 | 0.26 | 92.3 % | 3.70 | 0.0 % | 0.00 | 69.1 % | 7.5 | 37 | 15.2 % |
| 129.6 | 126.0 | 40 | 0.18 | 0.28 | 85.4 % | 3.23 | 0.0 % | 0.00 | 73.9 % | 6.1 | 46 | 14.3 % |
| 160.7 | 144.2 | 224 | 0.20 | 0.31 | 76.1 % | 2.74 | 0.0 % | 0.00 | 75.8 % | 5.1 | 60 | 16.3 % |

#### tput_n256_rtt0 (N = 256, RTT 0 ms)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs interrogés | pairs refusés | remplacements / requête | CPU traqueur | ms CPU / requête | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.12 | 0.22 | 95.4 % | 4.00 | 0.0 % | 0.00 | 5.3 % | 13.0 | 21 | 1.8 % |
| 9.9 | 9.9 | 0 | 0.12 | 0.21 | 93.3 % | 4.00 | 0.0 % | 0.00 | 8.6 % | 9.3 | 24 | 2.1 % |
| 14.9 | 14.7 | 0 | 0.12 | 0.22 | 93.7 % | 4.00 | 0.0 % | 0.00 | 12.6 % | 9.1 | 25 | 2.8 % |
| 20.5 | 20.5 | 0 | 0.13 | 0.22 | 91.6 % | 4.00 | 0.0 % | 0.00 | 14.8 % | 7.7 | 27 | 2.8 % |
| 25.6 | 25.4 | 0 | 0.12 | 0.22 | 92.5 % | 4.00 | 0.0 % | 0.00 | 17.5 % | 7.4 | 23 | 3.3 % |
| 30.4 | 30.2 | 0 | 0.13 | 0.23 | 92.5 % | 4.00 | 0.0 % | 0.00 | 19.8 % | 7.0 | 24 | 4.2 % |
| 39.9 | 39.6 | 0 | 0.13 | 0.23 | 92.1 % | 4.00 | 0.0 % | 0.00 | 27.3 % | 7.4 | 25 | 4.7 % |
| 52.6 | 52.1 | 0 | 0.12 | 0.22 | 93.3 % | 4.00 | 0.0 % | 0.00 | 36.7 % | 7.5 | 25 | 7.7 % |
| 62.0 | 61.5 | 0 | 0.12 | 0.22 | 93.1 % | 4.00 | 0.0 % | 0.00 | 44.5 % | 7.8 | 25 | 9.1 % |
| 78.0 | 77.2 | 0 | 0.13 | 0.22 | 93.2 % | 4.00 | 0.0 % | 0.00 | 47.6 % | 6.6 | 21 | 9.7 % |
| 100.1 | 98.8 | 0 | 0.15 | 0.27 | 92.6 % | 4.00 | 0.0 % | 0.00 | 60.4 % | 6.5 | 33 | 12.8 % |
| 129.6 | 120.4 | 57 | 0.77 | 1.00 | 91.0 % | 3.88 | 0.0 % | 0.00 | 70.5 % | 6.1 | 312 | 13.5 % |
| 160.7 | 125.5 | 410 | 1.04 | 1.42 | 74.5 % | 3.47 | 0.0 % | 0.00 | 69.2 % | 4.9 | 456 | 14.3 % |

#### tput_n1024_rtt0 (N = 1024, RTT 0 ms)

| offert (req/s) | servi (req/s) | échecs | p50 (s) | p95 (s) | exactitude | pairs interrogés | pairs refusés | remplacements / requête | CPU traqueur | ms CPU / requête | retard boucle p99 (ms) | CPU demandeurs (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3 | 4.3 | 0 | 0.12 | 0.22 | 95.4 % | 4.00 | 0.0 % | 0.00 | 7.1 % | 17.6 | 23 | 1.7 % |
| 9.9 | 9.9 | 0 | 0.13 | 0.24 | 89.9 % | 4.00 | 0.0 % | 0.00 | 10.4 % | 11.2 | 23 | 2.3 % |
| 14.9 | 14.9 | 0 | 0.13 | 0.22 | 93.3 % | 4.00 | 0.0 % | 0.00 | 11.4 % | 8.2 | 23 | 1.7 % |
| 20.5 | 20.5 | 0 | 0.13 | 0.20 | 93.2 % | 4.00 | 0.0 % | 0.00 | 14.7 % | 7.7 | 25 | 2.6 % |
| 25.6 | 25.4 | 0 | 0.12 | 0.20 | 94.5 % | 4.00 | 0.0 % | 0.00 | 19.7 % | 8.3 | 21 | 5.5 % |
| 30.4 | 30.3 | 0 | 0.12 | 0.20 | 94.3 % | 4.00 | 0.0 % | 0.00 | 23.3 % | 8.2 | 22 | 4.7 % |
| 39.9 | 39.5 | 0 | 0.12 | 0.21 | 93.0 % | 4.00 | 0.0 % | 0.00 | 27.5 % | 7.4 | 23 | 5.1 % |
| 52.6 | 52.0 | 0 | 0.13 | 0.23 | 91.9 % | 4.00 | 0.0 % | 0.00 | 40.0 % | 8.2 | 22 | 9.0 % |
| 62.0 | 61.6 | 0 | 0.13 | 0.21 | 94.3 % | 4.00 | 0.0 % | 0.00 | 42.8 % | 7.4 | 23 | 7.4 % |
| 78.0 | 76.9 | 0 | 0.14 | 0.26 | 92.9 % | 4.00 | 0.0 % | 0.00 | 54.9 % | 7.6 | 22 | 9.4 % |
| 100.1 | 96.7 | 0 | 0.22 | 0.73 | 93.1 % | 4.00 | 0.0 % | 0.00 | 68.5 % | 7.6 | 197 | 12.1 % |
| 129.6 | 116.0 | 0 | 1.40 | 1.85 | 93.8 % | 4.00 | 0.0 % | 0.00 | 69.9 % | 6.5 | 540 | 14.5 % |

### V3. Charge fixe, RTT 100 ms (temps de calcul / 10)

0,15 requête/s par nœud, plafonnée à 12 requêtes/s, k = 4 (comme en B).

| N | version | req/s offertes | servies | p50 (s) | p95 (s) | p99 (s) | pairs attendus | exactitude | CPU traqueur | ms CPU / requête | appels annuaire / s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | avant | 8.9 | 8.6 | 0.77 | 1.13 | 1.23 | 2.43 | 88.8 % | 8.9 % | 11.0 | 3.1 |
| 64 | après | 8.9 | 8.8 | 0.69 | 0.94 | 1.12 | 2.99 | 93.8 % | 8.1 % | 9.8 | 0.0 |
| 256 | avant | 11.3 | 10.8 | 0.74 | 1.12 | 1.27 | 2.85 | 92.9 % | 14.2 % | 13.9 | 3.5 |
| 256 | après | 11.3 | 10.9 | 0.67 | 1.03 | 1.36 | 3.02 | 94.2 % | 13.3 % | 12.8 | 0.0 |
| 1024 | avant | 11.3 | 10.8 | 0.85 | 1.37 | 1.52 | 2.95 | 93.4 % | 32.0 % | 31.3 | 4.0 |
| 1024 | après | 11.3 | 10.8 | 0.69 | 1.00 | 1.21 | 3.02 | 92.0 % | 12.3 % | 12.0 | 0.0 |
| 4096 | avant | – | – | – | – | – | – | – | – | – | – |
| 4096 | après | 11.3 | 10.9 | 0.69 | 0.99 | 1.22 | 3.03 | 93.8 % | 25.0 % | 24.3 | 0.0 |

### V4. Pannes de 25 % des nœuds (256 nœuds, RTT 50 ms, temps réels, délai 30 s)

« hang » : le moteur se fige (ses générations et sa sonde de santé ne répondent plus ; le nœud reste connecté). « hang_engine » (nouveau) : seules les générations se figent, la sonde répond normalement : seuls les dépassements de délai trahissent le nœud. Fenêtres par date de soumission : avant la panne, 10 s après, puis le reste.

| mode | fenêtre | version | requêtes | servies | exactitude | p50 (s) | p95 (s) | max (s) | pairs attendus | erreurs des pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crash | before | avant | 87 | 87 | 93.1 % | 5.06 | 7.73 | 12.48 | 3.05 | peer_busy 6, peer_disconnected 10 |
| crash | before | après | 87 | 87 | 90.8 % | 4.60 | 8.20 | 9.73 | 3.09 | peer_disconnected 7 |
| crash | during_10s | avant | 16 | 16 | 100.0 % | 4.14 | 7.36 | 7.50 | 3.00 | – |
| crash | during_10s | après | 16 | 16 | 100.0 % | 4.46 | 7.89 | 8.11 | 3.00 | – |
| crash | after | avant | 189 | 189 | 88.9 % | 4.97 | 8.04 | 14.62 | 2.94 | peer_busy 36 |
| crash | after | après | 189 | 189 | 94.7 % | 4.55 | 7.83 | 10.67 | 2.98 | – |
| hang | before | avant | 87 | 87 | 92.0 % | 5.07 | 7.81 | 13.86 | 2.92 | peer_busy 10 |
| hang | before | après | 87 | 87 | 90.8 % | 4.92 | 8.75 | 10.64 | 3.13 | – |
| hang | during_10s | avant | 16 | 16 | 100.0 % | 5.18 | 30.32 | 30.32 | 2.50 | peer_busy 1, deadline 6 |
| hang | during_10s | après | 16 | 16 | 81.2 % | 7.29 | 30.14 | 30.15 | 2.56 | deadline 11 |
| hang | after | avant | 189 | 189 | 90.0 % | 5.51 | 30.29 | 30.33 | 2.61 | peer_busy 33, deadline 60 |
| hang | after | après | 189 | 189 | 94.7 % | 4.67 | 8.08 | 30.14 | 3.02 | deadline 1 |
| hang_engine | before | avant | – | – | – | – | – | – | – | non mesuré en E7 |
| hang_engine | before | après | 87 | 87 | 92.0 % | 4.74 | 7.67 | 8.58 | 3.07 | – |
| hang_engine | during_10s | avant | – | – | – | – | – | – | – | non mesuré en E7 |
| hang_engine | during_10s | après | 16 | 16 | 93.8 % | 4.84 | 30.15 | 30.15 | 2.81 | deadline 8 |
| hang_engine | after | avant | – | – | – | – | – | – | – | non mesuré en E7 |
| hang_engine | after | après | 189 | 189 | 86.2 % | 5.43 | 30.14 | 30.18 | 2.81 | deadline 44 |

Santé vue par le traqueur pendant les mesures « après » (suspensions par motif, réadmissions ; un nœud toujours figé à la fin de sa suspension est suspendu de nouveau, plus longtemps), et remplacements faits par les passerelles sur toute la mesure :

| mode | événements | nœuds suspendus à la fin | remplacements / requête |
| --- | --- | --- | --- |
| crash | – | 0 | 0.024 |
| hang | engine_down 250, strike_timeout 13, suspend_engine_down 250 | 64 | 0.000 |
| hang_engine | readmitted 5, strike_timeout 68, suspend_timeout 10 | 5 | 0.000 |

Un pair figé ne renvoie pas d'erreur rapide : il n'est pas remplacé, il est écarté des requêtes suivantes par la suspension. Quand la sonde du moteur ne voit rien (hang_engine), seuls deux dépassements de délai consécutifs le trahissent, et la plupart de ses jobs sont annulés par le certificat avant leur délai : la détection reste lente, d'où un p95 encore au délai de la requête.

### V5. Pairs occupés

64 nœuds, temps / 40, RTT 0 (échelle de charge de V1) : part des pairs interrogés qui refusent le job, et pairs qui ont répondu, à chaque palier mesuré dans les deux versions.

| offert (req/s) | avant : refusés | avant : pairs interrogés | avant : exactitude | après : refusés | après : pairs interrogés | après : remplacements / requête | après : exactitude |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1.9 % | 4.00 | 95.4 % | 0.0 % | 4.00 | 0.00 | 93.8 % |
| 10 | 2.9 % | 4.00 | 94.0 % | 0.0 % | 4.00 | 0.00 | 92.6 % |
| 15 | 8.0 % | 4.00 | 91.5 % | 0.0 % | 4.00 | 0.00 | 95.5 % |
| 20 | 9.6 % | 4.00 | 89.6 % | 0.0 % | 4.00 | 0.00 | 91.2 % |
| 25 | 12.3 % | 4.00 | 89.1 % | 0.0 % | 4.00 | 0.00 | 93.0 % |
| 30 | 14.5 % | 4.00 | 91.0 % | 0.0 % | 4.00 | 0.00 | 93.4 % |
| 40 | 18.1 % | 4.00 | 86.0 % | 0.0 % | 4.00 | 0.00 | 92.0 % |
| 50 | 22.8 % | 4.00 | 86.4 % | 0.0 % | 4.00 | 0.00 | 93.8 % |
| 60 | 26.0 % | 4.00 | 85.9 % | 0.0 % | 4.00 | 0.00 | 94.4 % |
| 80 | 32.1 % | 4.00 | 81.6 % | 0.0 % | 3.96 | 0.00 | 92.7 % |
| 100 | 36.7 % | 4.00 | 77.5 % | 0.0 % | 3.70 | 0.00 | 92.3 % |
| 130 | 43.4 % | 4.00 | 75.6 % | 0.0 % | 3.23 | 0.00 | 85.4 % |

Le traqueur ne choisit que des nœuds qui ont un créneau libre : plus aucun pair n'est refusé. Quand une famille n'a plus de nœud libre, la requête part avec moins de pairs (colonne « pairs interrogés ») au lieu de perdre des pairs refusés, et il n'y a rien à remplacer. Le remplacement sert aux refus et pannes rapides vus par le nœud lui-même (coupure, moteur en erreur, refus local).

Charge proche de la saturation des nœuds (64 nœuds, temps / 10, médiane ≈ 0,4 s, un job à la fois par nœud, RTT 0), trois variantes mesurées dans la même session : v1.1 complète ; sans remplacement ; et passerelles en mode annuaire sans remplacement (le comportement d'essaim/1).

| variante | offert (req/s) | servi (req/s) | p50 (s) | p95 (s) | pairs interrogés | pairs ayant répondu | refusés | remplacements / requête | exactitude |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1.1 | 9.2 | 9.0 | 0.49 | 0.82 | 4.00 | 3.18 | 0.0 % | 0.00 | 96.2 % |
| v1.1 | 19.4 | 18.4 | 0.46 | 0.86 | 3.99 | 3.05 | 0.0 % | 0.00 | 91.8 % |
| v1.1 | 29.5 | 28.1 | 0.47 | 0.81 | 3.71 | 2.80 | 0.0 % | 0.00 | 91.5 % |
| v1.1 | 39.2 | 36.5 | 0.45 | 0.80 | 3.38 | 2.62 | 0.0 % | 0.00 | 85.7 % |
| v1.1 sans remplacement | 9.2 | 9.0 | 0.48 | 0.76 | 4.00 | 3.08 | 0.0 % | 0.00 | 94.6 % |
| v1.1 sans remplacement | 19.4 | 19.2 | 0.47 | 0.82 | 3.97 | 3.04 | 0.0 % | 0.00 | 90.8 % |
| v1.1 sans remplacement | 29.5 | 27.9 | 0.46 | 0.79 | 3.74 | 2.86 | 0.0 % | 0.00 | 91.9 % |
| v1.1 sans remplacement | 39.2 | 36.4 | 0.46 | 0.83 | 3.35 | 2.63 | 0.0 % | 0.00 | 87.4 % |
| annuaire (essaim/1) | 9.2 | 9.0 | 0.49 | 0.89 | 4.00 | 2.45 | 23.0 % | 0.00 | 90.8 % |
| annuaire (essaim/1) | 19.4 | 18.4 | 0.47 | 0.81 | 4.00 | 2.12 | 36.0 % | 0.00 | 77.9 % |
| annuaire (essaim/1) | 29.5 | 26.5 | 0.45 | 0.80 | 4.00 | 1.86 | 43.2 % | 0.00 | 73.9 % |
| annuaire (essaim/1) | 39.2 | 32.4 | 0.44 | 0.81 | 4.00 | 1.64 | 49.7 % | 0.00 | 67.1 % |

### V6. Coût de l'annuaire et de la sélection

Avant : chaque `GET /v1/peers` construisait la liste complète (boucle bloquée pendant ce temps). Après : la réponse vient d'un instantané ; sa reconstruction (au plus toutes les 3 s, seulement si quelqu'un lit l'annuaire) rend la main à la boucle tous les 256 pairs ; la sélection de 4 pairs ne dépend pas de N. Reconstruction : moyenne sur 20 ; plus long blocage : maximum sur ces 20 reconstructions (il inclut l'assemblage final de la réponse et les pauses du ramasse-miettes). Sélection : moyenne sur 2000 appels.

| N | avant : construction (ms, boucle bloquée) | avant : HTTP vu du client (ms) | après : HTTP vu du client (ms) | après : reconstruction (ms) | après : plus long blocage (ms) | après : sélection de 4 pairs (µs) |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 2.62 | 4.17 | 1.20 | 0.47 | 0.78 | 9.4 |
| 256 | 10.77 | 12.78 | 1.13 | 1.77 | 2.11 | 9.0 |
| 1024 | 42.96 | 45.43 | 1.55 | 8.13 | 2.44 | 11.4 |
| 4096 | – | – | 4.10 | 37.30 | 14.05 | 13.3 |

