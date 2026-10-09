# Détecteurs de données personnelles : résultats sur `test` (α = 0.01)

> Rapport historique de `pii-detector` (`e254079`, `06eba72`), conservé pour provenance.
> Les règles et la politique du garde ont changé : ces chiffres ne valident pas la calibration de
> l'intégration 5b. Le runtime demande prudemment confirmation lorsque le modèle fonctionne et ne
> revendique aucune garantie statistique. Rampart est rejeté et absent des catalogues exécutables ;
> ses résultats ci-dessous sont uniquement historiques. Voir [l'état actuel](../../../docs/09_detection_pii.md).

6897 documents ; découpage {'train': 2113, 'cal': 2053, 'test': 2731}.

Sources (positifs / négatifs) : gretel 1712/388, handmade 160/58, humaneval 0/164, nemotron 1172/28, oasst 0/1325, openpii 1837/53. gretel (étiquettes bruitées) est hors calibration et rapporté à part.

## 1. Politique « bloquer » : seuil conforme à α = 0.01 (calibré sur `cal`, mesuré sur `test`)

Ratés : documents avec données personnelles dont aucun jeton n'atteint λ (IC 95 % de Clopper-Pearson). FPR : documents sans donnée personnelle retenus au seuil λ (négatifs du banc : fait main, oasst, HumanEval) ; « oasst » : vraies premières questions d'utilisateurs, négatifs présumés.

| détecteur | λ | ratés test (IC 95 %) | ratés, moyenne de 500 re-tirages cal/test | ratés FR | ratés EN | FPR | FPR oasst | AUROC | ratés gretel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex+nym-small-edge+gte-small-probe-lr | 0.76 | 1.03 % (0.55 % – 1.75 %) | 0.97 % | 2.1 % | 0.7 % | 8.2 % | 5.4 % | 0.980 | 8.6 % |
| nym-small-edge | 0.445 | 1.50 % (0.91 % – 2.33 %) | 0.97 % | 4.9 % | 0.7 % | 9.0 % | 6.6 % | 0.982 | 9.6 % |
| regex+nym-small-edge+embeddinggemma2-probe-lr | 0.75 | 0.87 % (0.43 % – 1.55 %) | 0.95 % | 2.1 % | 0.5 % | 9.0 % | 5.8 % | 0.980 | 8.2 % |
| regex+nym-small-edge+e5-small-probe-lr | 0.738 | 0.63 % (0.27 % – 1.24 %) | 0.96 % | 1.4 % | 0.5 % | 9.3 % | 5.3 % | 0.980 | 8.5 % |
| regex+nym-small-edge | 0.445 | 0.79 % (0.38 % – 1.45 %) | 0.96 % | 2.1 % | 0.5 % | 10.3 % | 7.4 % | 0.979 | 9.0 % |
| regex+nym-small-edge+mminilm-probe-lr | 0.686 | 0.71 % (0.33 % – 1.35 %) | 0.97 % | 1.4 % | 0.7 % | 10.6 % | 6.8 % | 0.980 | 7.3 % |
| regex+embeddinggemma2-probe-lr | 0.259 | 0.55 % (0.22 % – 1.14 %) | 0.96 % | 2.1 % | 0.0 % | 11.7 % | 6.2 % | 0.985 | 1.0 % |
| embeddinggemma2-probe-lr | 0.237 | 0.71 % (0.33 % – 1.35 %) | 0.95 % | 0.7 % | 0.0 % | 11.8 % | 6.6 % | 0.990 | 0.6 % |
| e5-small-probe-lr | 0.339 | 1.26 % (0.72 % – 2.04 %) | 0.96 % | 3.5 % | 0.0 % | 12.3 % | 7.6 % | 0.987 | 2.2 % |
| embeddinggemma2-probe-mlp | 0.171 | 0.87 % (0.43 % – 1.55 %) | 0.94 % | 0.7 % | 0.0 % | 12.6 % | 8.0 % | 0.990 | 0.7 % |
| e5-small-probe-mlp | 0.353 | 1.18 % (0.66 % – 1.95 %) | 0.95 % | 2.8 % | 0.0 % | 12.8 % | 8.2 % | 0.986 | 2.5 % |
| regex+e5-small-probe-lr | 0.339 | 0.32 % (0.09 % – 0.81 %) | 0.94 % | 0.7 % | 0.0 % | 13.2 % | 8.2 % | 0.985 | 2.0 % |
| regex+embeddinggemma2-probe-mlp | 0.171 | 0.39 % (0.13 % – 0.92 %) | 0.95 % | 0.7 % | 0.0 % | 13.2 % | 8.4 % | 0.986 | 0.6 % |
| regex+rampart+mminilm-probe-lr | 0.653 | 0.47 % (0.17 % – 1.03 %) | 0.94 % | 0.7 % | 0.7 % | 13.5 % | 10.9 % | 0.989 | 8.0 % |
| regex+e5-small-probe-mlp | 0.353 | 0.32 % (0.09 % – 0.81 %) | 0.94 % | 0.7 % | 0.0 % | 13.7 % | 8.8 % | 0.985 | 2.2 % |
| regex+rampart+e5-small-probe-lr | 0.643 | 0.32 % (0.09 % – 0.81 %) | 0.95 % | 0.0 % | 0.5 % | 13.8 % | 9.9 % | 0.989 | 8.9 % |
| regex+rampart+embeddinggemma2-probe-lr | 0.579 | 0.24 % (0.05 % – 0.69 %) | 0.96 % | 0.0 % | 0.3 % | 15.6 % | 11.9 % | 0.989 | 6.3 % |
| regex+rampart+gte-small-probe-lr | 0.587 | 0.39 % (0.13 % – 0.92 %) | 0.96 % | 0.0 % | 0.7 % | 16.2 % | 12.5 % | 0.988 | 7.0 % |
| regex+rampart | 0.16 | 0.47 % (0.17 % – 1.03 %) | 0.97 % | 0.0 % | 0.7 % | 20.2 % | 17.9 % | 0.987 | 9.2 % |
| regex+bert-small-pii+embeddinggemma2-probe-lr | 0.432 | 0.71 % (0.33 % – 1.35 %) | 0.92 % | 1.4 % | 0.2 % | 21.5 % | 18.3 % | 0.973 | 1.0 % |
| regex+mminilm-probe-lr | 0.204 | 0.95 % (0.49 % – 1.65 %) | 0.95 % | 2.1 % | 0.3 % | 21.8 % | 18.7 % | 0.981 | 1.6 % |
| regex+bert-small-pii+e5-small-probe-lr | 0.432 | 0.71 % (0.33 % – 1.35 %) | 0.94 % | 0.7 % | 0.5 % | 22.2 % | 19.1 % | 0.972 | 1.3 % |
| mminilm-probe-lr | 0.185 | 1.11 % (0.61 % – 1.85 %) | 0.93 % | 2.1 % | 0.3 % | 22.6 % | 20.0 % | 0.979 | 1.6 % |
| regex+bunker-laya | 0.0164 | 0.63 % (0.27 % – 1.24 %) | 0.95 % | 0.7 % | 0.7 % | 22.6 % | 19.6 % | 0.982 | – |
| rampart | 0.0834 | 1.66 % (1.03 % – 2.52 %) | 0.98 % | 2.8 % | 1.2 % | 23.8 % | 22.0 % | 0.984 | 7.7 % |
| regex+bert-small-pii+gte-small-probe-lr | 0.474 | 1.03 % (0.55 % – 1.75 %) | 0.94 % | 0.7 % | 0.3 % | 24.3 % | 21.4 % | 0.970 | 1.9 % |
| regex+bert-small-pii+mminilm-probe-lr | 0.389 | 0.71 % (0.33 % – 1.35 %) | 0.95 % | 0.7 % | 0.3 % | 24.3 % | 21.8 % | 0.972 | 1.0 % |
| regex+mminilm-probe-mlp | 0.281 | 0.95 % (0.49 % – 1.65 %) | 0.96 % | 0.7 % | 0.3 % | 25.7 % | 23.0 % | 0.979 | 2.5 % |
| regex+gte-small-probe-lr | 0.351 | 0.71 % (0.33 % – 1.35 %) | 0.96 % | 0.7 % | 0.3 % | 25.8 % | 23.5 % | 0.980 | 3.9 % |
| mminilm-probe-mlp | 0.271 | 1.11 % (0.61 % – 1.85 %) | 0.94 % | 1.4 % | 0.3 % | 26.4 % | 23.9 % | 0.975 | 2.5 % |
| regex+gliner-pii-small | 0.503 | 0.87 % (0.43 % – 1.55 %) | 0.96 % | 0.0 % | 0.2 % | 27.4 % | 25.5 % | 0.980 | 2.2 % |
| regex+gte-small-probe-mlp | 0.346 | 0.95 % (0.49 % – 1.65 %) | 0.95 % | 0.7 % | 0.5 % | 27.4 % | 25.5 % | 0.978 | 4.8 % |
| gliner-pii-small | 0.492 | 1.82 % (1.16 % – 2.71 %) | 0.97 % | 3.5 % | 0.5 % | 28.9 % | 26.8 % | 0.976 | 1.6 % |
| regex+ettin-32m | 0.415 | 1.34 % (0.78 % – 2.14 %) | 0.96 % | 1.4 % | 0.3 % | 29.1 % | 28.0 % | 0.972 | 10.3 % |
| bunker-laya | 0.0161 | 0.79 % (0.38 % – 1.45 %) | 0.92 % | 0.7 % | 1.2 % | 29.4 % | 27.6 % | 0.979 | – |
| ettin-32m | 0.383 | 1.90 % (1.22 % – 2.81 %) | 0.96 % | 2.8 % | 0.3 % | 29.9 % | 29.4 % | 0.969 | 11.2 % |
| gte-small-probe-lr | 0.304 | 0.47 % (0.17 % – 1.03 %) | 0.96 % | 0.0 % | 0.2 % | 32.5 % | 31.1 % | 0.975 | 2.5 % |
| gte-small-probe-mlp | 0.28 | 1.11 % (0.61 % – 1.85 %) | 0.96 % | 0.0 % | 0.5 % | 34.1 % | 33.5 % | 0.970 | 3.6 % |
| regex+minilm-nemotron | 0.205 | 1.26 % (0.72 % – 2.04 %) | 0.94 % | 0.7 % | 0.0 % | 43.7 % | 46.3 % | 0.964 | 7.1 % |
| minilm-nemotron | 0.178 | 1.58 % (0.97 % – 2.43 %) | 0.94 % | 1.4 % | 0.0 % | 44.8 % | 48.2 % | 0.957 | 7.3 % |
| regex+bert-small-pii | 0.0162 | 0.55 % (0.22 % – 1.14 %) | 0.92 % | 0.0 % | 0.2 % | 57.2 % | 59.1 % | 0.958 | 0.1 % |
| bert-small-pii | 0.0091 | 1.34 % (0.78 % – 2.14 %) | 0.91 % | 0.0 % | 0.3 % | 63.9 % | 67.3 % | 0.945 | 0.3 % |
| regex | 0 | 0.00 % (0.00 % – 0.29 %) | 0.00 % | 0.0 % | 0.0 % | 100.0 % | 100.0 % | 0.850 | 0.0 % |

### 1 bis. Variante PAC : taux de ratés ≤ α avec probabilité ≥ 0.95 sur le tirage de `cal`

| détecteur | λ PAC | ratés test (IC 95 %) | FPR |
| --- | --- | --- | --- |
| bunker-laya | 0.016 | 0.71 % (0.33 % – 1.35 %) | 33.6 % |
| e5-small-probe-lr | 0.272 | 0.32 % (0.09 % – 0.81 %) | 17.9 % |
| e5-small-probe-mlp | 0.182 | 0.08 % (0.00 % – 0.44 %) | 23.6 % |
| embeddinggemma2-probe-lr | 0.102 | 0.08 % (0.00 % – 0.44 %) | 29.9 % |
| embeddinggemma2-probe-mlp | 0.0308 | 0.00 % (0.00 % – 0.29 %) | 27.2 % |
| ettin-32m | 0.327 | 1.34 % (0.78 % – 2.14 %) | 33.9 % |
| gliner-pii-small | 0.446 | 0.71 % (0.33 % – 1.35 %) | 38.1 % |
| gte-small-probe-lr | 0.281 | 0.39 % (0.13 % – 0.92 %) | 35.5 % |
| gte-small-probe-mlp | 0.264 | 1.11 % (0.61 % – 1.85 %) | 35.9 % |
| minilm-nemotron | 0.0914 | 1.11 % (0.61 % – 1.85 %) | 57.4 % |
| mminilm-probe-lr | 0.164 | 0.71 % (0.33 % – 1.35 %) | 24.7 % |
| mminilm-probe-mlp | 0.217 | 0.63 % (0.27 % – 1.24 %) | 32.3 % |
| nym-small-edge | 0.0205 | 0.95 % (0.49 % – 1.65 %) | 19.0 % |
| rampart | 0.0388 | 1.11 % (0.61 % – 1.85 %) | 28.9 % |
| regex+bert-small-pii | 0.0108 | 0.39 % (0.13 % – 0.92 %) | 62.5 % |
| regex+bunker-laya | 0.016 | 0.39 % (0.13 % – 0.92 %) | 31.7 % |
| regex+e5-small-probe-lr | 0.272 | 0.16 % (0.02 % – 0.57 %) | 18.8 % |
| regex+e5-small-probe-mlp | 0.182 | 0.08 % (0.00 % – 0.44 %) | 24.4 % |
| regex+embeddinggemma2-probe-lr | 0.102 | 0.08 % (0.00 % – 0.44 %) | 30.5 % |
| regex+embeddinggemma2-probe-mlp | 0.0308 | 0.00 % (0.00 % – 0.29 %) | 27.8 % |
| regex+ettin-32m | 0.332 | 0.95 % (0.49 % – 1.65 %) | 34.7 % |
| regex+gliner-pii-small | 0.446 | 0.32 % (0.09 % – 0.81 %) | 38.9 % |
| regex+gte-small-probe-lr | 0.304 | 0.16 % (0.02 % – 0.57 %) | 33.1 % |
| regex+gte-small-probe-mlp | 0.28 | 0.39 % (0.13 % – 0.92 %) | 34.8 % |
| regex+minilm-nemotron | 0.0914 | 0.79 % (0.38 % – 1.45 %) | 58.0 % |
| regex+mminilm-probe-lr | 0.164 | 0.63 % (0.27 % – 1.24 %) | 25.3 % |
| regex+mminilm-probe-mlp | 0.217 | 0.47 % (0.17 % – 1.03 %) | 32.8 % |
| regex+nym-small-edge | 0.0205 | 0.32 % (0.09 % – 0.81 %) | 20.1 % |
| regex+rampart | 0.0388 | 0.24 % (0.05 % – 0.69 %) | 29.5 % |

## 2. Compromis selon α (même protocole)

| détecteur | α = 1 % : λ / ratés / FPR | α = 2 % : λ / ratés / FPR | α = 5 % : λ / ratés / FPR | α = 10 % : λ / ratés / FPR |
| --- | --- | --- | --- | --- |
| nym-small-edge | 0.445 / 1.50 % / 9.0 % | 0.828 / 2.29 % / 6.4 % | 0.988 / 4.50 % / 3.6 % | 0.999 / 9.32 % / 2.2 % |
| rampart | 0.0834 / 1.66 % / 23.8 % | 0.24 / 2.37 % / 17.6 % | 0.852 / 4.66 % / 8.4 % | 0.99 / 8.14 % / 2.5 % |
| regex+bert-small-pii | 0.0162 / 0.55 % / 57.2 % | 0.0279 / 1.18 % / 47.4 % | 0.173 / 4.82 % / 24.7 % | 0.599 / 8.37 % / 14.0 % |
| regex+bunker-laya | 0.0164 / 0.63 % / 22.6 % | 0.017 / 1.26 % / 15.9 % | 0.0324 / 2.92 % / 7.3 % | 0.983 / 7.19 % / 3.1 % |
| regex+e5-small-probe-lr | 0.339 / 0.32 % / 13.2 % | 0.423 / 0.87 % / 9.2 % | 0.603 / 4.19 % / 4.8 % | 0.697 / 8.21 % / 3.3 % |
| regex+e5-small-probe-mlp | 0.353 / 0.32 % / 13.7 % | 0.502 / 0.87 % / 9.8 % | 0.788 / 4.19 % / 4.8 % | 0.885 / 8.61 % / 3.1 % |
| regex+embeddinggemma2-probe-lr | 0.259 / 0.55 % / 11.7 % | 0.346 / 1.18 % / 9.3 % | 0.537 / 3.63 % / 5.0 % | 0.716 / 9.64 % / 3.0 % |
| regex+embeddinggemma2-probe-mlp | 0.171 / 0.39 % / 13.2 % | 0.487 / 1.34 % / 8.2 % | 0.818 / 4.82 % / 4.7 % | 0.927 / 8.69 % / 3.1 % |
| regex+ettin-32m | 0.415 / 1.34 % / 29.1 % | 0.683 / 2.69 % / 18.7 % | 0.877 / 5.06 % / 10.4 % | 0.968 / 9.24 % / 6.2 % |
| regex+gliner-pii-small | 0.503 / 0.87 % / 27.4 % | 0.563 / 1.66 % / 17.9 % | 0.642 / 3.24 % / 9.6 % | 0.74 / 8.21 % / 4.2 % |
| regex+gte-small-probe-lr | 0.351 / 0.71 % / 25.8 % | 0.448 / 1.97 % / 14.5 % | 0.549 / 4.34 % / 7.6 % | 0.625 / 8.14 % / 5.0 % |
| regex+gte-small-probe-mlp | 0.346 / 0.95 % / 27.4 % | 0.464 / 2.29 % / 16.0 % | 0.585 / 4.34 % / 9.6 % | 0.701 / 8.85 % / 5.4 % |
| regex+minilm-nemotron | 0.205 / 1.26 % / 43.7 % | 0.38 / 1.58 % / 30.6 % | 0.829 / 4.74 % / 16.8 % | 0.949 / 8.85 % / 11.5 % |
| regex+mminilm-probe-lr | 0.204 / 0.95 % / 21.8 % | 0.315 / 1.74 % / 12.3 % | 0.545 / 4.19 % / 6.7 % | 0.695 / 7.11 % / 4.7 % |
| regex+mminilm-probe-mlp | 0.281 / 0.95 % / 25.7 % | 0.439 / 1.42 % / 15.4 % | 0.668 / 4.34 % / 7.8 % | 0.817 / 7.74 % / 5.0 % |
| regex+nym-small-edge | 0.445 / 0.79 % / 10.3 % | 0.903 / 1.50 % / 7.2 % | 0.995 / 3.40 % / 4.4 % | 1 / 7.98 % / 3.4 % |
| regex+rampart | 0.16 / 0.47 % / 20.2 % | 0.534 / 0.63 % / 13.4 % | 0.994 / 2.45 % / 3.6 % | 1 / 6.87 % / 2.5 % |

## 3. Au seuil fixe 0,5 (sans calibration)

| détecteur | ratés | FPR | ratés FR | ratés EN | FPR oasst | rappel des passages |
| --- | --- | --- | --- | --- | --- | --- |
| bert-small-pii | 11.45 % | 15.4 % | 13.4 % | 8.4 % | 15.2 % | 60.5 % |
| bunker-laya | 11.30 % | 4.2 % | 7.7 % | 17.4 % | 2.1 % | – |
| e5-small-probe-lr | 5.21 % | 5.6 % | 7.7 % | 2.5 % | 2.1 % | – |
| e5-small-probe-mlp | 2.69 % | 8.4 % | 3.5 % | 0.7 % | 4.3 % | – |
| embeddinggemma2-probe-lr | 5.29 % | 4.4 % | 6.3 % | 1.7 % | 1.6 % | – |
| embeddinggemma2-probe-mlp | 2.29 % | 6.8 % | 2.8 % | 1.0 % | 2.3 % | – |
| ettin-32m | 2.45 % | 24.4 % | 5.6 % | 0.3 % | 23.3 % | 80.0 % |
| gliner-pii-small | 1.82 % | 26.9 % | 3.5 % | 0.5 % | 25.5 % | 82.6 % |
| gte-small-probe-lr | 7.82 % | 8.6 % | 10.6 % | 5.7 % | 5.3 % | – |
| gte-small-probe-mlp | 6.08 % | 13.4 % | 7.0 % | 4.4 % | 10.7 % | – |
| minilm-nemotron | 3.00 % | 24.7 % | 4.2 % | 0.0 % | 24.3 % | 81.4 % |
| mminilm-probe-lr | 8.21 % | 6.2 % | 12.7 % | 5.2 % | 3.7 % | – |
| mminilm-probe-mlp | 3.95 % | 11.7 % | 7.0 % | 1.2 % | 8.4 % | – |
| nym-small-edge | 1.58 % | 8.6 % | 4.9 % | 0.7 % | 6.0 % | 92.7 % |
| rampart | 3.00 % | 12.3 % | 2.8 % | 2.9 % | 11.1 % | 91.3 % |
| regex | 28.36 % | 1.6 % | 23.2 % | 21.3 % | 1.2 % | 29.2 % |
| regex+bert-small-pii | 7.82 % | 16.3 % | 7.7 % | 5.9 % | 15.8 % | 65.2 % |
| regex+bunker-laya | 4.03 % | 5.6 % | 4.9 % | 4.7 % | 3.1 % | – |
| regex+e5-small-probe-lr | 1.50 % | 7.0 % | 2.1 % | 0.8 % | 3.3 % | – |
| regex+e5-small-probe-mlp | 0.87 % | 9.8 % | 0.7 % | 0.3 % | 5.4 % | – |
| regex+embeddinggemma2-probe-lr | 2.92 % | 5.6 % | 4.9 % | 1.2 % | 2.5 % | – |
| regex+embeddinggemma2-probe-mlp | 1.34 % | 7.9 % | 2.8 % | 0.5 % | 3.3 % | – |
| regex+ettin-32m | 1.66 % | 25.5 % | 2.8 % | 0.3 % | 23.9 % | 81.5 % |
| regex+gliner-pii-small | 0.87 % | 27.7 % | 0.0 % | 0.2 % | 25.9 % | 85.2 % |
| regex+gte-small-probe-lr | 3.55 % | 10.0 % | 4.2 % | 1.4 % | 6.4 % | – |
| regex+gte-small-probe-mlp | 3.00 % | 14.6 % | 2.8 % | 1.0 % | 11.7 % | – |
| regex+minilm-nemotron | 2.13 % | 25.8 % | 2.1 % | 0.0 % | 24.9 % | 82.5 % |
| regex+mminilm-probe-lr | 3.55 % | 7.2 % | 4.9 % | 2.0 % | 4.3 % | – |
| regex+mminilm-probe-mlp | 2.05 % | 12.6 % | 2.1 % | 0.8 % | 8.9 % | – |
| regex+nym-small-edge | 0.87 % | 9.8 % | 2.1 % | 0.5 % | 6.8 % | 93.4 % |
| regex+rampart | 0.63 % | 13.4 % | 0.0 % | 1.0 % | 11.7 % | 94.1 % |

## 4. Rappel des passages par type (seuil 0,5)

| détecteur | ADDRESS | DATE_OF_BIRTH | EMAIL | FINANCIAL | ID | IP | PERSON | PHONE | SECRET | USERNAME |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bert-small-pii | 9 % | 0 % | 100 % | 73 % | 26 % | 100 % | 80 % | 94 % | 27 % | 62 % |
| ettin-32m | 58 % | 94 % | 100 % | 97 % | 82 % | 99 % | 72 % | 85 % | 99 % | 89 % |
| gliner-pii-small | 83 % | 76 % | 90 % | 95 % | 96 % | 87 % | 64 % | 99 % | 93 % | 82 % |
| minilm-nemotron | 66 % | 95 % | 99 % | 96 % | 81 % | 100 % | 72 % | 94 % | 99 % | 93 % |
| nym-small-edge | 94 % | 95 % | 97 % | 100 % | 87 % | 87 % | 91 % | 100 % | 90 % | 88 % |
| rampart | 94 % | 5 % | 94 % | 87 % | 89 % | 60 % | 99 % | 99 % | 36 % | 86 % |
| regex | 17 % | 58 % | 99 % | 10 % | 6 % | 69 % | 13 % | 75 % | 42 % | 17 % |
| regex+bert-small-pii | 26 % | 58 % | 100 % | 76 % | 28 % | 100 % | 81 % | 98 % | 52 % | 72 % |
| regex+ettin-32m | 59 % | 99 % | 100 % | 98 % | 83 % | 100 % | 72 % | 96 % | 99 % | 98 % |
| regex+gliner-pii-small | 84 % | 82 % | 100 % | 95 % | 96 % | 97 % | 67 % | 100 % | 96 % | 93 % |
| regex+minilm-nemotron | 67 % | 100 % | 100 % | 97 % | 81 % | 100 % | 73 % | 98 % | 99 % | 100 % |
| regex+nym-small-edge | 94 % | 100 % | 100 % | 100 % | 87 % | 96 % | 92 % | 100 % | 93 % | 97 % |
| regex+rampart | 94 % | 62 % | 100 % | 88 % | 89 % | 79 % | 99 % | 100 % | 71 % | 95 % |

## 5. Politique « masquer » : couverture de tous les passages

Raté = au moins un passage personnel sans jeton ≥ λ (on masque tout ce qui dépasse λ).

| détecteur | α = 1 % : λ / fuites / FPR | α = 5 % : λ / fuites / FPR | α = 10 % : λ / fuites / FPR | α = 20 % : λ / fuites / FPR |
| --- | --- | --- | --- | --- |
| regex+bert-small-pii | 0 / 0.00 % / 100.0 % | 0 / 0.00 % / 100.0 % | 0 / 0.00 % / 100.0 % | 0 / 0.00 % / 100.0 % |
| regex+ettin-32m | 0 / 0.00 % / 100.0 % | 0.0058 / 4.90 % / 96.7 % | 0.0171 / 9.79 % / 86.9 % | 0.0701 / 19.19 % / 69.5 % |
| regex+gliner-pii-small | 0.0655 / 1.18 % / 99.5 % | 0.206 / 7.11 % / 85.1 % | 0.306 / 13.59 % / 67.2 % | 0.402 / 21.56 % / 47.3 % |
| regex+minilm-nemotron | 0 / 0.00 % / 100.0 % | 0 / 0.00 % / 100.0 % | 0.0115 / 9.64 % / 85.4 % | 0.0742 / 22.35 % / 60.5 % |
| regex+nym-small-edge | 0 / 0.00 % / 100.0 % | 0 / 0.00 % / 100.0 % | 0.0066 / 10.27 % / 25.3 % | 0.412 / 19.04 % / 10.4 % |
| regex+rampart | 0 / 0.00 % / 100.0 % | 0 / 0.00 % / 100.0 % | 0.0391 / 9.95 % / 29.4 % | 0.612 / 19.98 % / 11.8 % |

## 5 bis. Politique « masquer » retenue : règles masquées, modèle en porte sur le reste (α = 0.01)

Fuite = un passage personnel que les règles ne masquent pas entièrement, et aucun jeton du modèle hors des passages des règles n'atteint λ. « Retenu » : négatifs envoyés ni automatiquement ni masqués (demander ou local).

| combinaison | λ masque | positifs entièrement masqués par les règles (cal) | fuites test (IC 95 %) | négatifs retenus | λ PAC | fuites PAC | négatifs retenus PAC | négatifs masqués par les règles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regex+bert-small-pii | 0.0091 | 7.6 % | 0.71 % (0.33 % – 1.35 %) | 63.9 % | 0.0073 | 0.63 % (0.27 % – 1.24 %) | 66.4 % | 1.6 % |
| regex+ettin-32m | 0.166 | 7.6 % | 0.63 % (0.27 % – 1.24 %) | 51.8 % | 0.0914 | 0.24 % (0.05 % – 0.69 %) | 64.2 % | 1.6 % |
| regex+gliner-pii-small | 0.425 | 7.6 % | 0.79 % (0.38 % – 1.45 %) | 42.1 % | 0.347 | 0.16 % (0.02 % – 0.57 %) | 56.8 % | 1.6 % |
| regex+minilm-nemotron | 0.0633 | 7.6 % | 1.03 % (0.55 % – 1.75 %) | 61.6 % | 0.0299 | 0.55 % (0.22 % – 1.14 %) | 73.3 % | 1.6 % |
| regex+nym-small-edge | 0.0177 | 7.6 % | 0.87 % (0.43 % – 1.55 %) | 19.8 % | 0.0072 | 0.63 % (0.27 % – 1.24 %) | 24.0 % | 1.6 % |
| regex+rampart | 0.0165 | 7.6 % | 0.79 % (0.38 % – 1.45 %) | 38.3 % | 0.0073 | 0.63 % (0.27 % – 1.24 %) | 45.6 % | 1.6 % |

## 6. Décisions du scanneur sur `test` (règle de `privacy_model.decide`)

| combinaison | politique | λ bas | λ haut | positifs : envoyer / demander / masquer-local | négatifs : envoyer / demander / masquer-local | oasst : envoyer / demander / masquer-local |
| --- | --- | --- | --- | --- | --- | --- |
| regex+bert-small-pii | block | 0.0162 | 0.978 | 7 / 343 / 916 | 275 / 346 / 22 | 210 / 291 / 13 |
| regex+bert-small-pii | block-pac | 0.0108 | 0.978 | 5 / 346 / 915 | 241 / 381 / 21 | 178 / 324 / 12 |
| regex+bert-small-pii | mask | 0.0091 | 0.978 | 5 / 1201 / 60 | 230 / 400 / 13 | 167 / 340 / 7 |
| regex+bert-small-pii | mask-pac | 0.0073 | 0.978 | 4 / 1208 / 54 | 214 / 416 / 13 | 156 / 351 / 7 |
| regex+ettin-32m | block | 0.415 | 0.991 | 17 / 276 / 973 | 456 / 163 / 24 | 370 / 128 / 16 |
| regex+ettin-32m | block-pac | 0.332 | 0.991 | 12 / 286 / 968 | 420 / 200 / 23 | 340 / 159 / 15 |
| regex+ettin-32m | mask | 0.166 | 0.991 | 5 / 1009 / 252 | 304 / 321 / 18 | 236 / 267 / 11 |
| regex+ettin-32m | mask-pac | 0.0914 | 0.991 | 1 / 1081 / 184 | 229 / 403 / 11 | 173 / 334 / 7 |
| regex+gliner-pii-small | block | 0.503 | 0.692 | 11 / 257 / 998 | 467 / 142 / 34 | 383 / 114 / 17 |
| regex+gliner-pii-small | block-pac | 0.446 | 0.692 | 4 / 284 / 978 | 393 / 219 / 31 | 324 / 173 / 17 |
| regex+gliner-pii-small | mask | 0.425 | 0.692 | 2 / 997 / 267 | 367 / 249 / 27 | 302 / 199 / 13 |
| regex+gliner-pii-small | mask-pac | 0.347 | 0.692 | 0 / 1097 / 169 | 274 / 344 / 25 | 219 / 283 / 12 |
| regex+minilm-nemotron | block | 0.205 | 0.989 | 16 / 251 / 999 | 362 / 250 / 31 | 276 / 219 / 19 |
| regex+minilm-nemotron | block-pac | 0.0914 | 0.989 | 10 / 264 / 992 | 270 / 344 / 29 | 194 / 303 / 17 |
| regex+minilm-nemotron | mask | 0.0633 | 0.989 | 9 / 791 / 466 | 243 / 378 / 22 | 171 / 333 / 10 |
| regex+minilm-nemotron | mask-pac | 0.0299 | 0.989 | 4 / 821 / 441 | 169 / 459 / 15 | 113 / 393 / 8 |
| regex+nym-small-edge | block | 0.445 | 0.988 | 10 / 133 / 1123 | 577 / 37 / 29 | 476 / 24 / 14 |
| regex+nym-small-edge | block-pac | 0.0205 | 0.988 | 4 / 181 / 1081 | 514 / 100 / 29 | 422 / 78 / 14 |
| regex+nym-small-edge | mask | 0.0177 | 0.988 | 4 / 666 / 596 | 509 / 107 / 27 | 419 / 83 / 12 |
| regex+nym-small-edge | mask-pac | 0.0072 | 0.988 | 3 / 717 / 546 | 482 / 135 / 26 | 399 / 104 / 11 |
| regex+rampart | block | 0.16 | 0.967 | 6 / 84 / 1176 | 513 / 93 / 37 | 422 / 67 / 25 |
| regex+rampart | block-pac | 0.0388 | 0.967 | 3 / 119 / 1144 | 453 / 156 / 34 | 375 / 117 / 22 |
| regex+rampart | mask | 0.0165 | 0.967 | 3 / 513 / 750 | 393 / 223 / 27 | 329 / 169 / 16 |
| regex+rampart | mask-pac | 0.0073 | 0.967 | 2 / 579 / 685 | 347 / 270 / 26 | 296 / 203 / 15 |

## 7. Coût : mémoire et latence (i5-10400F, 6 cœurs, onnxruntime CPU)

RSS mesurée par le système dans un processus neuf : « exécution » = Python + numpy + onnxruntime + tokenizers importés ; « chargé » = après chargement du modèle ; « pic » = pic de l'ensemble de travail après les questions faites main et un texte long. Latence : questions faites main (`short`, ≤ 600 caractères) et tous les documents du banc.

| détecteur | licence | RSS exécution | RSS chargé | pic | p50 / p95 court (ms) | p50 / p95 tous (ms) | chargement (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bert-small-pii | apache-2.0 | 57.9 | 100.0 | 182.7 | 4.2 / 10.9 | 12.9 / 96.3 | 0.174 |
| bunker-laya | apache-2.0 (weights; trained partly on CC-BY-NC data) | 56.2 | 1500.6 | 1983.6 | 222.8 / 429.9 | 409.4 / 1645.0 | 2.462 |
| e5-small-probe | mit | 57.9 | 434.0 | 451.2 | 9.1 / 24.8 | 34.0 / 142.6 | 1.308 |
| embeddinggemma2-probe | apache-2.0 | 58.4 | 204.9 | 476.1 | 65.2 / 140.1 | 258.5 / 993.2 | 2.561 |
| ettin-32m | mit | 56.3 | 200.9 | 274.9 | 9.3 / 23.7 | 27.2 / 154.1 | 0.773 |
| gliner-pii-small | apache-2.0 | 57.6 | 167.7 | 362.3 | 61.3 / 172.4 | 182.9 / 951.8 | 1.385 |
| gte-small-probe | mit | 57.8 | 115.0 | 129.9 | 9.8 / 23.5 | 62.9 / 308.5 | 0.739 |
| minilm-nemotron | mit | 57.7 | 160.5 | 257.7 | 7.1 / 16.3 | 16.3 / 130.8 | 0.248 |
| mminilm-probe | apache-2.0 | 56.4 | 435.5 | 453.2 | 10.7 / 25.8 | 56.4 / 282.2 | 1.225 |
| nym-small-edge | mit | 57.9 | 223.5 | 463.3 | 12.7 / 29.9 | 73.9 / 478.9 | 2.394 |
| rampart | cc-by-4.0 | 56.4 | 85.0 | 186.4 | 5.7 / 11.8 | 16.6 / 116.0 | 0.13 |
| regex | – | 57.8 | 57.9 | 58.8 | 0.2 / 0.3 | 0.6 / 2.3 | 0.0 |
