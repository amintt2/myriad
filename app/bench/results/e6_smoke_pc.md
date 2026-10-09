# E6 : l'app de bout en bout avec de vrais modèles (smoke_pc)

> **Test de fumée** : quelques questions sur le PC du dépôt, pour vérifier le montage. Ces chiffres ne sont **pas** des résultats de l'article.

- Date : 2026-10-09 13:22:10. Machine : Windows-11-10.0.26200-SP0, GPU : –.
- llama-server : version: 0.6.0-dev (build 11505, commit ff5888f99) | built with Clang 20.1.8 for Windows x86_64.
- Nœuds : Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf, ibm-granite/granite-3.3-2b-instruct-GGUF:granite-3.3-2b-instruct-Q8_0.gguf.
- k = 2, 2 emplacement(s) par llama-server, 2 requête(s) simultanée(s), 512 jetons au plus, température 0, délai 180 s.
- Questions : phase0 GSM8K test (openai/gsm8k, seeded shuffle, seed 0), 10 à partir de 0. Correcteur : phase0/essaim/answers.py extract('gsm8k', ended = finish_reason == 'stop').
- Traqueur : in-process, starter credit 1e9. WAN émulé dans le relais : retard aller simple lognormal (σ = 0.25), « RTT » = aller-retour médian nominal client–traqueur.

| RTT (ms) | certificat | requêtes | servies | exactitude | p50 (s) | p95 (s) | p99 (s) | pairs attendus | arrêt anticipé | req/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | oui | 10 | 10 | 70.0 % | 5.98 | 7.87 | 8.00 | 1.40 | 60 % | 0.330 |
| 0 | non | 10 | 10 | 70.0 % | 5.86 | 6.70 | 6.74 | 2.00 | 0 % | 0.350 |
| 100 | oui | 10 | 10 | 80.0 % | 6.10 | 7.21 | 7.21 | 1.70 | 30 % | 0.326 |
| 100 | non | 10 | 10 | 60.0 % | 6.17 | 6.96 | 7.00 | 2.00 | 0 % | 0.324 |

Par modèle (réponses reçues avant la décision) :

| RTT (ms) | certificat | modèle | réponses | exactitude | calcul p50 (ms) | latence p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | oui | Qwen/Qwen3-1.7B-GGUF | 10 | 80.0 % | 5931.7 | 5960.0 |
| 0 | oui | ibm-granite/granite-3.3-2b-instruct-GGUF | 4 | 75.0 % | 5341.3 | 5359.8 |
| 0 | non | Qwen/Qwen3-1.7B-GGUF | 10 | 80.0 % | 5550.2 | 5561.3 |
| 0 | non | ibm-granite/granite-3.3-2b-instruct-GGUF | 10 | 80.0 % | 4804.9 | 4821.3 |
| 100 | oui | Qwen/Qwen3-1.7B-GGUF | 10 | 80.0 % | 5740.1 | 6087.6 |
| 100 | oui | ibm-granite/granite-3.3-2b-instruct-GGUF | 7 | 71.4 % | 4661.5 | 5094.0 |
| 100 | non | Qwen/Qwen3-1.7B-GGUF | 10 | 80.0 % | 5671.9 | 6040.9 |
| 100 | non | ibm-granite/granite-3.3-2b-instruct-GGUF | 10 | 80.0 % | 4759.7 | 5239.9 |
