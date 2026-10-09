# Tâche 5 : essai Terminal-Bench 4.0 avec Harbor

## Provenance des trois tâches, avant tout résultat

Présélection fixée avant exécution : trois tâches publiques sans GPU, avec des environnements Python slim.
Ce choix vise un essai d'intégration à faible encombrement ; ce n'est ni un échantillon aléatoire ni un choix
de tâches démontrées faciles pour nos modèles. Aucune réussite n'a été mesurée.

| tâche | domaine officiel | licence | commit officiel |
| --- | --- | --- | --- |
| `session-window-debug` | Software / Systems | Apache-2.0 | `452bf305c6daa62fc59061d22133a7cbc7c1572e` |
| `wal-recovery-ordering` | Software / Databases | Apache-2.0 | `452bf305c6daa62fc59061d22133a7cbc7c1572e` |
| `interleaved-vigenere` | Security / Cryptography | Apache-2.0 | `452bf305c6daa62fc59061d22133a7cbc7c1572e` |

Source : [tag officiel v4.0.0](https://github.com/harbor-framework/terminal-bench/tree/v4.0.0/tasks),
[licence à ce commit](https://github.com/harbor-framework/terminal-bench/blob/452bf305c6daa62fc59061d22133a7cbc7c1572e/LICENSE).
Le tag contient 66 tâches. `phase0/harbor/tasks.json` fige les tailles et SHA-256 des 71 fichiers sélectionnés,
licence comprise : **291 385 octets**, sans images Docker. Les solutions officielles restent sur l'hôte ;
Harbor ne les installe pas dans le conteneur de notre agent. Les tests sont ceux du dépôt, sans remplacement.
Le téléchargement ne lit que ces fichiers publics, par URL contenant le commit, et vérifie leurs empreintes.
Une tâche modifiée, incomplète ou accompagnée de fichiers supplémentaires est refusée avant l'essai.

Harbor **0.24.0**, Apache-2.0, est épinglé au commit
`d5ac1be17f575852eaf4fffc4072fd18481c209b` du
[dépôt officiel](https://github.com/harbor-framework/harbor/tree/d5ac1be17f575852eaf4fffc4072fd18481c209b).
`phase0/harbor/uv.lock` verrouille ses dépendances ; environnement séparé en Python 3.12.
La [documentation des agents externes](https://docs.harborframework.com/agents/custom-agents) et le code de ce
commit ont été lus : `BaseAgent.setup/run`, `BaseEnvironment.exec`, `Trial.create/run`, `Verifier.verify`.
Le lanceur utilise ces API réelles. Il ne réimplémente ni le cycle Docker ni le protocole des récompenses.

## Ce qui fonctionne et ce qui reste bloqué

`HarborEnv` dans `phase0/essaim/agent.py` adapte l'environnement asynchrone prêté par Harbor à la boucle
synchrone existante. Le travail synchrone tourne dans un fil ; la boucle Harbor reste disponible pour les
délais. Chaque commande est passée à `environment.exec` dans `bash -lc`, sous GNU `timeout --signal=KILL`.
Le délai agit ainsi sur les processus de la commande dans le conteneur, pas uniquement sur le client Docker.
L'annulation ferme le pont et annule ses appels en cours. `close()` ne détruit pas le conteneur : Harbor doit
d'abord collecter les artefacts et exécuter son vérificateur. `Trial` assure ensuite le nettoyage.

Les trois tâches utilisent un **vérificateur séparé**, déclaré par leur `task.toml`. L'agent reçoit uniquement
l'instruction officielle et ses observations ; il ne reçoit ni solution ni tests du vérificateur.
Un message de soumission n'est jamais un verdict. Seule la récompense officielle binaire `reward` vaut note.
Harbor intercepte les délais d'agent dépassés et les sorties non nulles de l'agent, puis exécute le vérificateur.
Une récompense officielle valide de 0 ou 1 reste donc la note après ces arrêts ; l'incident est conservé
séparément dans `exception_type` et dans le résultat Harbor. Un délai ou un échec du vérificateur, une erreur
d'infrastructure ou un verdict manquant rend le résumé incomplet, avec score **null**, pas zéro.

Vérifié localement : téléchargement et empreintes des trois tâches ; import des tâches et de l'agent avec
Harbor épinglé ; tests simulant environnement, transport authentifié, budgets, annulation et cycle de nettoyage,
ainsi que l'appel au véritable `Verifier.verify` avec un environnement simulé. **Ces tests ne sont pas une
exécution des tâches officielles et ne donnent aucun score de modèle.**

Blocages : le démarrage du backend Docker Desktop échoue pendant « initializing Ingest server », avec un
fichier `sailor-ingest.sock` inaccessible ; aucun reset n'a été tenté. E11 reste en préparation et le service
d'inférence avec son tunnel authentifié n'est pas prêt pour cet essai.
Aucune session ou processus Colab n'a été consulté, lancé, arrêté ou modifié. Aucun réglage Docker n'a changé,
aucune image n'a été téléchargée. Harbor et ses dépendances CPU sont installés dans le répertoire ignoré
`phase0/data/terminal-bench/venv`. Aucun modèle n'a été téléchargé.

Les Dockerfiles retenus sont petits et sans téléchargement de modèle. Cela **ne prouve pas** leur taille
finale : elle devra être contrôlée lorsque Docker sera disponible. Certaines images de base officielles
utilisent des tags mutables (`python:3.12-slim`, `python:3.13-slim-bookworm`), d'autres un digest ; les fichiers
de tâches sont figés mais un rebuild futur n'est donc pas garanti identique octet pour octet. Conserver les
journaux de construction Harbor et relever les digests des bases lors du vrai essai. Avec seulement 44 Go
libres sur C:, ne pas lancer le jeu complet ni tirer une image de plus de 5 Go sans besoin explicite.

## Commandes reproductibles dans WSL

Depuis la racine du dépôt dans WSL, sans toucher aux environnements phase0/app habituels :

```bash
cd phase0
export UV_PROJECT_ENVIRONMENT="$PWD/data/terminal-bench/venv"
uv sync --project harbor --frozen --python 3.12
uv run --project harbor --frozen python run_terminal_bench.py --prepare
uv run --project harbor --frozen python run_terminal_bench.py --check
uv run --project harbor --frozen python -m unittest discover -s tests -p test_terminal_bench.py -v
```

`--check` ne contacte ni Docker, ni modèle, ni Colab. Il vérifie les fichiers, la version **et le commit installé**
de Harbor, puis charge les tâches et la classe d'agent avec l'API officielle. Les tâches restent dans
`phase0/data/terminal-bench/selected`, ignoré par git. Aucun secret n'est nécessaire pour préparer ou vérifier.

La mise en place du service et du tunnel authentifiés reste à réaliser par l'orchestrateur, déjà autorisé à
le faire lorsque la VM sera libre. Elle ne nécessite pas de demander au propriétaire de fournir le service.
Après cette étape, faire pointer `MYRIAD_BENCH_URL` vers le point d'accès OpenAI compatible : HTTP en boucle
locale, HTTPS pour un point distant. Le jeton est obligatoire, fourni par environnement, jamais dans une URL
ou argument de commande. Ne pas activer de trace shell. Exemple pour un tunnel local déjà établi :

```bash
export MYRIAD_BENCH_URL=http://127.0.0.1:8400/v1
read -rs -p 'Jeton du service authentifié : ' MYRIAD_BENCH_TOKEN
export MYRIAD_BENCH_TOKEN
printf '\n'
docker info

# Replace with the exact IDs of the models actually served.
SINGLE='Qwen/Qwen3.5-4B'
PEER='google/gemma-4-E4B-it'
REFERENCE='Qwen/Qwen3.5-9B'

uv run --project harbor --frozen python run_terminal_bench.py \
  --single "$SINGLE" --output results/terminal_smoke_3

uv run --project harbor --frozen python run_terminal_bench.py \
  --compare --single "$SINGLE" --vote "$SINGLE" "$PEER" --reference "$REFERENCE" \
  --output results/terminal_compare_3

unset MYRIAD_BENCH_TOKEN
```

Ce sont des noms d'exemple, pas l'affirmation que ces modèles sont actuellement disponibles. Vérifier leurs
identifiants et poids/quantifications servis avant l'essai. Éviter les alias de famille si plusieurs pairs
peuvent répondre au même alias. Le nom renvoyé par le service et son usage, s'ils sont fournis, sont journalisés.
La référence doit être servie au même point d'accès que les pairs. Le lanceur n'alloue aucune VM et ne démarre
aucun service. **Tout futur accès Colab doit passer par `phase0/colab/colab_phase0.sh` dans WSL.** Le wrapper
actuel n'a pas de commande de service/tunnel ; son extension, si nécessaire, reste à tester par l'orchestrateur
après libération de la VM, sans interférer avec E11. Aucune commande Colab n'est exécutée pour cette correction.
Ne pas utiliser `up`, `down` ou une commande Colab directe pendant E11 pour préparer cet essai.

## Protocole, journaux et interprétation

Le premier appel exécute les trois tâches avec un seul modèle. Le second exécute neuf essais frais,
séquentiels : trois `single`, trois `vote`, trois `reference` (stratégie `single`). Aucun conteneur ou historique
de conversation ne passe d'un essai au suivant. Pas de reprise silencieuse, pas de nouvelle tentative :
un répertoire de sortie existant est refusé. Les mêmes tâches et ressources officielles sont utilisées.
Chaque invocation crée un `run_id` aléatoire de 32 caractères hexadécimaux, enregistré dans `manifest.json`.
Il préfixe tous les noms d'essai et reste distinct après la sanitisation des projets Docker, y compris ceux
du vérificateur. Deux invocations avec des répertoires de sortie différents ne partagent donc pas leurs noms
de conteneurs ; les tâches et les plafonds restent reproductibles indépendamment de cet identifiant.

Plafonds communs par défaut, modifiables explicitement pour **tous** les modes d'un appel :

| plafond | valeur | option |
| --- | --- | --- |
| tours, erreurs de format incluses | 100 | `--max-steps` |
| temps d'agent, génération et commandes comprises | 1 800 s | `--timeout` |
| temps par commande | 60 s | `--cmd-timeout` |
| jetons de complétion demandés par requête | 1 024 | `--max-tokens` |
| somme des plafonds de complétion demandés par épisode | 102 400 | `--total-tokens` |
| graine demandée au service | 0 | `--seed` |

Le budget de génération est **réservé avant chaque requête**, même si le modèle produit moins de jetons.
Le vote partage ce budget entre ses pairs : il ne reçoit pas k fois le plafond du solo. Une proposition finale
peut recevoir une allocation réduite ; un tour de vote incomplet n'exécute aucune commande. Les tokens de prompt,
le calcul GPU et le comportement effectif de la graine ne sont pas rendus identiques par cette règle : ce n'est
pas une comparaison à FLOPs égaux. Les usages du serveur sont conservés, ou `null` s'il ne les fournit pas.
Le vote demande une proposition par pair, en séquence ; les égalités suivent l'ordre déclaré de `--vote`.
Le solo doit figurer parmi les pairs ; aucun meilleur modèle de dev n'est prétendu sans sélection de dev réelle.
Une requête HTTP a un délai propre de 30 s ; son échec est une erreur, pas une réponse incorrecte du modèle.

Le temps du vérificateur et celui de construction restent ceux des tâches officielles, identiques entre modes
pour une même tâche. Le plafond réduit de l'agent, notre boucle et notre budget rendent cet essai **non comparable
à un classement AA ou au score complet Terminal-Bench 4.0**.

Le lanceur affiche la provenance avant toute opération d'essai et écrit `manifest.json` avant les essais,
avec empreintes du code et du verrou, provenance et configuration sans jeton.
Dans `trials/<run_id>__<mode>__<tâche>/`,
Harbor conserve `config.json`, `lock.json`, `result.json`, artefacts, journaux de construction et de vérificateur.
`agent/episode.jsonl` conserve chaque génération (modèle demandé/retourné, allocation, usage, réponse), chaque
commande (votes, résultat et sortie complète) et la raison d'arrêt. Seule l'observation renvoyée au modèle est
tronquée. Les erreurs de création d'essai sont indiquées par type dans `verdicts.json`, sans faux verdict.

`summary.json` ne calcule pass@1 et Wilson à 95 % qu'après trois verdicts complets pour un mode ; une erreur
d'infrastructure laisse son score à `null` et entraîne un retour non nul. Une comparaison complète ajoute les
tests appariés exacts de `essaim/stats.py` (marge préfixée à 2 points), dans le même ordre de tâches. Trois tâches
ne permettent aucune affirmation générale de supériorité ou d'équivalence. Le pilote sert à vérifier
l'intégration et à examiner les échecs ; ces résultats ne doivent pas être utilisés dans l'article.

Validation initiale de `feb57b2` : suite phase0 Windows **170 tests, 9 ignorés**, suite app **308 réussis, 1 ignoré** ;
les **16 tests** `test_terminal_bench.py` passent dans l'environnement WSL Harbor, sans omission. Les omissions
Windows comprennent les tests nécessitant Harbor et le shell Linux ; ils ont bien été exécutés dans WSL.
`--prepare` et `--check` ont également réussi sur les fichiers réels. L'audit Astra avant fusion reste à
l'orchestrateur ; cette préparation n'est pas un résultat de benchmark.

Après corrections de revue : tests ciblés Windows **18 tests Terminal-Bench, 7 ignorés** et **11 tests d'export,
1 ignoré** ; WSL Harbor **18/18 réussis** et export **11 tests, 1 ignoré**. Les tests couvrent deux invocations
avec des noms Docker isolés après sanitisation, l'export exact des trois ressources et du guide, les notes 0/1
après incidents d'agent et les essais incomplets sans récompense. L'export complet propre et le nouvel audit
restent à l'orchestrateur. Aucun essai Docker ou modèle réel n'a été exécuté.
