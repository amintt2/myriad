# E17 — adaptateur BigCodeBench DEV local

Cette livraison prépare la notation académique d'inférence distribuée sur un banc public, selon la
clarification explicite du propriétaire du 13 octobre. DEV local GPU/WSL Landlock ; Colab seulement
pour l'évaluation finale figée et E16. Ni serveur, ni modèle, ni smoke officiel, ni benchmark exécuté ici.
Les décisions scientifiques et les révisions du harnais restent au parent après DEV Aider ; aucun P02/P03,
prompt SYSTEM nouveau ou changement de budget d'agent. La CLI autonome est prête pour audit parent.

## Provenance et périmètre

Port explicite depuis `orchestration/bigcodebench`, HEAD `0f7110daef31097784cd3e708f771463523938b6`,
code `716d3fc9fb75a1fe72733d87bd40eadc750a673c`. Aucun merge de fichiers partagés anciens : runtime,
lock, licence et modules RPC historiques copiés à l'identique ; manifeste réduit aux deux DEV et tests
officiels correspondants sélectionnés sans réécriture. Provenance et anciens blobs dans les preuves privées.
App/E11/article/VERIF/Colab/lanceurs concurrents intacts ; le Journal courant est uniquement complété.

Hard Instruct **v0.1.4**, révision `298d2cc7b96612e15e47313c3603ee124cee0c1f`, Apache-2.0 selon
la fiche Full épinglée `b74c0d0bf70d2c0bc459be537895cca163007f1a`. Métadonnées et preuves de la
sélection historique conservées dans la provenance originale ; aucun nouveau choix de tâche.
DEV : `BigCodeBench_120`, `BigCodeBench_184`. Les tests originaux gardent leurs empreintes exactes :
**6 + 5 méthodes, 8 + 5 sites statiques d'assertion**, dont la liste des 365 dates et les cinq DataFrames.
Aucun allègement d'assertion, aucune correction des attendus, aucun retrait de diagnostic DEV.
La CLI rejette toute autre tâche avant ouverture de son fichier. 199 reste archivé ailleurs ; TEST 208/211
reste uniquement réservé par métadonnées. Aucune généralisation validée sur TEST ni promesse de score TEST.

## Décision de frontière de confiance pour le parent

Le verrou historique exigeait une attestation de l'origine des exceptions contre du Python arbitraire.
Son contournement réel est conservé : lecture de `fd` via frame Python, écriture d'une enveloppe
`phase=call, exception_kind=value_error`, puis `os._exit(73)` ; le contrôleur levait un ValueError
et donnait un faux 6/6. Le marqueur public et le pipe prouvent des octets et une terminaison, pas une
exception authentique. Les anciennes preuves parent/Sol et l'audit du 10 octobre restent valables.
Ce P1 historique **n'est pas corrigé cryptographiquement** et aucun nonce de remplacement n'est introduit.

La décision proposée à la revue parent est une portée standard de banc public **non adversarial** :
noter la conformité fonctionnelle aux assertions officielles, en supposant que le candidat ne modifie
pas volontairement unittest, les tests, le superviseur ou le canal de résultat. C'est une redéfinition
explicite du verrou incompatible avec cette portée, conformément à l'autorisation donnée ; aucune nouvelle
demande d'approbation propriétaire. Le parent Astra juge la pertinence scientifique de cette frontière.

Le contrôleur copie uniquement `solution.py`, les tests de confiance et le runner dans un dossier neuf.
`LocalLinuxEnv` inchangé impose Landlock ABI >= 3, namespaces, seccomp, limites mémoire/CPU/processus/
fichiers et temps. Aucun repli, backend strict WSL uniquement. Le superviseur n'importe pas la solution :
il observe la fin de l'enfant et recueille un résultat JSON borné sur pipe distinct du stdout/stderr.
L'enfant importe la solution et exécute **directement** tous les tests officiels, sans codec des valeurs,
sans transformation des exceptions, sans filtre des types retournés et sans fail-fast. Les effets de bord,
flags pandas, timestamps nanoseconde et vraies sous-classes d'exception gardent leur sémantique Python.

La collection doit avoir le nombre attendu ; le contrôleur exige les identités des méthodes, leur
complétion et des comptes entiers exacts. Échecs, erreurs, skips, expected failures et unexpected successes
interdisent le vert. `exit(0)` sans résultat, `exit(73)`, syntaxe/import candidat, délais, résultat invalide,
sortie excessive et valeurs fausses sont des échecs candidats. Absence de reçu d'isolement, runtime/pins
invalides, problème de copie/contrôleur ou collection officielle défaillante sont une erreur infrastructure :
`status=error, passed=null`, sans rapport ni reprise scorable. Une exception de test est un échec candidat ;
une exception d'import candidat ne peut plus satisfaire un assertRaises officiel.

**Ce que ce dispositif ne garantit pas** : le pipe est accessible au worker et Python peut modifier unittest
ou ses frames ; la copie des tests n'est pas un secret dans son processus. Un attaquant délibéré peut encore
fabriquer une réussite. Une fixture le démontre en neutralisant assertEqual : diagnostic vert avec
`adversarial_attestation=false`, exclu de toute mesure. Aucun grade ne doit être présenté comme attesté contre
ce candidat. `scorable=true` signifie seulement un verdict exploitable sous l'hypothèse non adversariale
annoncée ; les campagnes, lignes, rapports et grades portent `trust_scope=public-benchmark-nonadversarial`
et `adversarial_attestation=false`. La frontière protège l'hôte et sépare la collecte, pas l'authenticité hostile.

Différences amont : une solution complète dans un fichier importé, puis suite officielle intacte dans
un worker neuf par vérification ; pas processus neuf par appel de fonction. Le superviseur et le contrôleur
restent séparés. L'agent shell E17, ses tests intermédiaires et ses budgets divergent du pass@1 amont.
Les limites mémoire/CPU restent celles du bac historique, pas une promesse d'éligibilité de tout Hard.
Les fixtures synthétiques vérifient les autres entry points et objets, pas des tâches inédites.

## Compatibilité de l'agent

`run_bigcode_pilot.py` réutilise **sans modification** `aider_agent.run_episode`, son SYSTEM, BudgetChat strict,
la commande `MYRIAD_RUN_OFFICIAL_TESTS`, le départage et les plafonds existants : 20 étapes, 4096 jetons/réponse,
49152 plafonds réservés, 900 s/épisode, 30 s/commande, 120 s/vérification. Aucun serveur lancé ici.
Les sources doivent avoir un vert complet inchangé avant soumission ; vérification finale indépendante.
Les jetons inconnus restent null, plafonds et jetons observés distincts ; énergie/coût sous hypothèses.
Manifeste, transcriptions, reprise identique et grille DEV complète exigés avant rapport.

Cette CLI constitue une interface autonome ; `bigcode_pilot.run_one(args, task, mode, models, client, output,
campaign_hash)` accepte le même client `post` que code_pilot, donc aussi le pont stdio local existant.
Le lanceur local corrigé/audité `28cd9d9` est maintenant intégré dans ce seul worktree. `e17_local`
dispose d'un dispatch explicite `--benchmark bigcode`, Aider restant le défaut inchangé. Staging limité
à 120/184, runtime/lock dédiés et format de smoke `negative` avec reçu à la racine du grade sont sélectionnés
avant appel à run_one via **PipeClient/worker Windows**. Le chemin Aider garde `incorrect` et ses reçus par appel.
SYSTEM, boucle, budgets, pins Qwen/Granite et chargement Vulkan séquentiel sont conservés. Les grades finals,
runtime/registry/manifest gardent la portée non adversariale ; `test_frozen=false` et `ready_for_test=false`.
Une panne de runtime/staging précède le relais ; une panne de smoke précède toute requête modèle.
Le [guide local](16_e17_dev_local.md) donne aussi les commandes Aider compatibles de l'ancien premier bloc.

## Prérequis réels et commandes parent WSL après audit

WSL Linux x86_64 avec Landlock >= 3, unshare/namespaces et seccomp disponibles. Le runtime historique
`$HOME/.local/share/myriad-bigcode-pilot/venv` est présent et a passé check_runtime : Python 3.12.15,
binaire et 31 distributions/licences/composants conformes au lock. Ne pas faire uv sync dans cet ancien
runtime. Son cache contient les deux fichiers DEV nécessaires, vérifiés par hash en lecture seule.
La disponibilité réelle du GPU/serveur et des modèles n'a pas été validée ici. Le raccordement stdio
du lanceur est vérifié uniquement sur fixtures synthétiques ; le smoke officiel reste au parent après audit.
Les commandes suivantes ne téléchargent rien et n'ouvrent aucun autre fichier de tâche.
`MYRIAD_ROOT` désigne le worktree audité sous WSL ; DEV_CACHE, SMOKE_DIR et PRIVATE_CAMPAIGN_DIR sont neufs.

Pour le GPU Windows, utiliser le pont stdio du lanceur, jamais un loopback Windows supposé accessible WSL.
Depuis WSL, `MYRIAD_ASSETS` est le répertoire WSL des assets Windows déjà présents, WINDOWS_PYTHON un
python.exe existant avec httpx (chemin WSL), LOCAL_PROOFS un parent de sorties neuves. Après audit parent :

```bash
cd "$MYRIAD_ROOT/phase0"
export PYTHONDONTWRITEBYTECODE=1
BIGCODE_RUNTIME="$HOME/.local/share/myriad-bigcode-pilot/venv"
BIGCODE_CACHE="$HOME/.local/share/myriad-bigcode-pilot/data"
bigcode_args=(--benchmark bigcode --runtime "$BIGCODE_RUNTIME" --windows-python "$WINDOWS_PYTHON"
  --binary "$(wslpath -w "$MYRIAD_ASSETS/tools/llama.cpp-bin/llama-server.exe")"
  --archive "$(wslpath -w "$MYRIAD_ASSETS/tools/llama-b11505-bin-win-vulkan-x64.zip")"
  --models-dir "$(wslpath -w "$MYRIAD_ASSETS/models")" --tasks-dir "$BIGCODE_CACHE")
"$BIGCODE_RUNTIME/bin/python" e17_local.py preflight "${bigcode_args[@]}" --mode single \
  --output "$LOCAL_PROOFS/bigcode-preflight"
"$BIGCODE_RUNTIME/bin/python" e17_local.py smoke "${bigcode_args[@]}" --mode single \
  --output "$LOCAL_PROOFS/bigcode-smoke"
# Le smoke doit rendre 0 et confirmer 6/6 + 5/5 et tous les négatifs, sinon arrêter sans modèle.
"$BIGCODE_RUNTIME/bin/python" e17_local.py run "${bigcode_args[@]}" --mode single \
  --output "$LOCAL_PROOFS/bigcode-single"
"$BIGCODE_RUNTIME/bin/python" e17_local.py run "${bigcode_args[@]}" --mode vote \
  --output "$LOCAL_PROOFS/bigcode-vote"
```

Le runtime et les deux données DEV sont contrôlés avant démarrage du relais, preflight compris.
Le smoke de chaque run précède les modèles ; clôture worker/serveur confirmée et deux grades valides
exigés pour registry.complete, jamais un score article ou TEST. La provenance modèle conserve ses champs
exacts, liée par hash au manifeste du benchmark. Aucune sélection scientifique ni stratégie nouvelle.
Les commandes autonomes ci-dessous restent disponibles pour un endpoint HTTP distinct préparé par le parent ;
elles ne remplacent pas ce raccordement Windows stdio.

```bash
cd "$MYRIAD_ROOT/phase0"
export PYTHONDONTWRITEBYTECODE=1
PY="$HOME/.local/share/myriad-bigcode-pilot/venv/bin/python"
OLD_CACHE="$HOME/.local/share/myriad-bigcode-pilot/data"
"$PY" run_bigcode_pilot.py --prepare --source-cache "$OLD_CACHE" --tasks-dir "$DEV_CACHE" --sandbox-backend strict
"$PY" run_bigcode_pilot.py --check --tasks-dir "$DEV_CACHE" --sandbox-backend strict
# Seulement après audit parent : smoke des deux références et des contrôles négatifs, sans modèle.
"$PY" run_bigcode_pilot.py --smoke --tasks-dir "$DEV_CACHE" --sandbox-backend strict --output "$SMOKE_DIR"
```

Le smoke exige 6/6 et 5/5, tous les négatifs rejetés ; s'il échoue, arrêter sans retirer d'assertion,
remplacer de tâche ou démarrer un modèle. Les commandes shell doivent rendre 0 séparément, logs clos
et codes persistés par le parent. Les références ne sont pas un score de modèle.
Ensuite seulement, endpoint local préparé par le parent et provenance non secrète conforme à
`essaim/code_provenance.py`, budget R00 inchangé, tâche scientifique DEV décidée après Aider :
la CLI ci-dessous utilise un endpoint HTTP accessible depuis WSL via MYRIAD_BENCH_URL. Le GPU Windows
du lanceur e17_local passe par le raccordement stdio décrit ci-dessus.
Ne pas supposer que le loopback WSL rejoint un serveur Windows, ni que ces commandes lancent le GPU.

```bash
"$PY" run_bigcode_pilot.py --check --tasks-dir "$DEV_CACHE" --single "$SMALL_MODEL" --provenance "$PROVENANCE"
"$PY" run_bigcode_pilot.py --tasks-dir "$DEV_CACHE" --sandbox-backend strict \
  --single "$SMALL_MODEL" --provenance "$PROVENANCE" --output "$PRIVATE_CAMPAIGN_DIR" \
  --max-steps 20 --max-tokens 4096 --total-tokens 49152 --timeout 900 --cmd-timeout 30 \
  --verify-timeout 120 --seed 0 --watts 100 --eur-kwh 0.25
"$PY" run_bigcode_pilot.py --report --tasks-dir "$DEV_CACHE" --output "$PRIVATE_CAMPAIGN_DIR"
```

La mesure refait le smoke obligatoire avant la première requête et après reprise. Pour la comparaison
autorisée par le parent, ajouter `--vote "$SMALL_MODEL" "$SECOND_SMALL_MODEL" --reference "$LARGE_MODEL"`
à la même commande, avec provenance de tous les modèles et runtime serveur compatible. Ne pas déduire
la disponibilité locale de 27B : prérequis non validé, aucun chargement fait. L'ordre des modes reste
single/vote/cascade/reference. Aucune mesure TEST dans cette CLI DEV ; elle restera à préparer après gel.

## Vérification de cette livraison

Les tests nouveaux sont synthétiques, y compris les vraies exécutions WSL Landlock. Ils couvrent résultats
corrects/faux, exceptions/sous-classes/homonymes, import/syntaxe, skips/collection vide, timeout/sortie excessive,
source et retour prématuré, autre entry point, mutation d'entrée, pandas flags/nanosecondes, soumission E17
et reprise/rapport partiel refusé. Ils conservent deux démonstrations adversariales, sans les compter comme scores.
Les tests DEV officiels sont vérifiés par hash/AST sans exécuter de candidat officiel.
L'app est réutilisable seulement après vérification des 171 blobs et du log clos 449/3 de verif1-provenance.
Suite phase0 avec groupe paper et tests WSL hermétiques, scanner et export simulé sont consignés dans
le rapport privé de livraison. Aucun test réel affaibli ni SkipTest artificiel de données historiques.
Commit français puis STOP pour audit Astra parent ; aucun score expérimental revendiqué.
