# Pilote d'agent de code sans Docker

Jalon préparé les 2026-10-09/10 ; **premier pilote CPU réel terminé le 2026-10-10 : 0 réussite sur 3**.
La consommation électrique reste estimée, jamais mesurée ; les résultats et limites sont détaillés ci-dessous.
Ce premier banc de code prépare 5c ; il ne constitue pas l'évaluation d'un dépôt logiciel réel.

## Données et présélection

[Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark/tree/7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f)
contient 225 exercices Exercism dans six langages. Ce jalon prépare **les 34 exercices Python** seulement.
Le pilote est fixé avant toute génération : les trois premiers identifiants lexicographiques,
**affine-cipher, beer-song, book-store**. Aucune sélection selon la réussite, aucun remplacement d'exercice.

`phase0/code_pilot/tasks.json` épingle le commit `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`, le SHA-256
de l'archive publique et les tailles/SHA-256 des **282 fichiers** conservés (Python, README, licence).
La préparation télécharge une archive publique épinglée ; elle ne copie que les entrées régulières énumérées
dans le manifeste, sans extraction de chemins libres. Le cache est dans `phase0/data/`, ignoré par git.
Toute donnée manquante, supplémentaire, altérée ou liée symboliquement est refusée.

Le README Aider attribue les exercices à Exercism et renvoie aux licences des pistes ; il n'y a pas de
licence racine Aider Polyglot. Pour Python, la [licence MIT Exercism](https://github.com/exercism/python/blob/617408a174390ad45200d11b7ba06e068bd7a94b/LICENSE)
est épinglée à `617408a174390ad45200d11b7ba06e068bd7a94b`, vérifiée et conservée. Les métadonnées d'auteurs
restent dans le cache. La copie agent contient sources initiales, tests publics, instructions, attribution
et licence, **aucun exemple de solution**, aucun `.meta`, `.git` ou autre exercice.

## Préparation et vérification sans modèle

Depuis PowerShell, lancer WSL et se placer à la racine du worktree ; les commandes suivantes sont dans bash WSL.
Le runtime est **un nouvel environnement utilisateur dédié sur le disque Linux natif**. Il ne modifie ni
un environnement root, ni un environnement Linux partagé. Aucun modèle n'est téléchargé.

```bash
cd phase0/code_pilot
UV_PROJECT_ENVIRONMENT="$HOME/.local/share/myriad-code-pilot/venv" \
  ~/.local/bin/uv sync --frozen --python 3.12
cd ..
PY="$HOME/.local/share/myriad-code-pilot/venv/bin/python"
"$PY" run_code_pilot.py --prepare
"$PY" run_code_pilot.py --check
"$PY" run_code_pilot.py --smoke --output data/code-pilot-smoke
CODE_PILOT_LINUX_TESTS=1 "$PY" -m unittest discover -s tests -p test_code_pilot.py -v
```

Python 3.12.15 déjà installé par uv, uv 0.12.24, pytest 8.4.2, httpx 0.28.1 et matplotlib 3.10.8 ont été utilisés ;
`uv.lock` fige les dépendances transitives. Python système WSL est 3.14.4 et sert seulement au lanceur de
protection. Installer la venv sur `/mnt/c` est à éviter : le smoke a révélé un refus de lecture Landlock
sur ses fichiers DrvFS. L'exécution vérifie que l'interpréteur du contrôleur est celui du runtime déclaré.

Le smoke exécute les références `.meta/example.py` officielles dans des copies de vérification séparées :
**16 tests affine-cipher, 8 tests beer-song, 20 tests book-store** ; les trois références doivent réussir
et les trois solutions volontairement incorrectes doivent échouer. Il est **obligatoire avant chaque
invocation pouvant faire une inférence**, reprise incluse. Son résultat est un contrôle du harnais,
jamais une mesure de modèle. Une erreur du smoke interdit toute requête modèle.

## Frontière d'exécution

`essaim/local_linux.py:LocalLinuxEnv` respecte `Env.run(command, timeout_s)`. Chaque commande est lancée
dans des namespaces utilisateur, montage, réseau, PID, IPC et UTS neufs via `unshare --fork --kill-child=KILL`.
Le contrôleur reste hors du namespace PID ; `PR_SET_PDEATHSIG` lie le lanceur à sa vie. Au délai ou à la fermeture,
il tue le lanceur ; la destruction
du namespace PID tue aussi les sous-enfants ayant appelé `setsid`, contrairement au seul kill de groupe.
Un reçu émis avant exec confirme la mise en place des protections ; son absence est une erreur fermée.

Landlock ABI ≥ 3 est obligatoire : lecture dans la tâche, les binaires/bibliothèques système, la venv et
l'installation Python uv exacte ; écriture seulement dans la copie tâche et `/dev/null`. Aucun accès
au dépôt Myriad, aux données/références, `.ssh`, caches, `/proc`, `/sys` ou aux autres dossiers utilisateur.
Les symlinks ne permettent pas de lire/écrire leur cible externe. Le filtre seccomp x86_64 interdit tous
les sockets, y compris Unix, les opérations mount/unshare/setns, ptrace, process_vm, io_uring, BPF,
les appareils, trousseaux noyau et les modifications de métadonnées (`chmod`, `chown`, dates, attributs étendus).
Les variantes `futimesat`, `setxattrat`, `removexattrat` et `file_setattr` sont également refusées.
Les numéros x86_64 ont été vérifiés contre les en-têtes WSL et l'implémentation
[Linux 6.18 de file_setattr](https://github.com/torvalds/linux/blob/v6.18/fs/file_attr.c).
Les `ioctl` sont interdits sauf `FIOCLEX`, nécessaire à Python pour fermer ses descripteurs à exec ;
cette exception ne modifie aucun inode. Un descripteur ouvert en lecture peut sinon modifier les
drapeaux d'un inode du runtime via `FS_IOC_SETFLAGS`.
L'environnement ne contient que PATH/runtime, HOME/TMPDIR/tâche, locale, graine Python et paramètres
de threads/plugins. Aucun secret, proxy, PYTHONPATH ou environnement du propriétaire n'est transmis.

Limites par processus : mémoire virtuelle 512 MiB, CPU borné par le délai commande, fichier 16 MiB,
32 processus, 128 descripteurs, aucun core dump ; sortie collectée limitée à 1 MiB. Les rlimits sont
hérités, avec plafonds durs. Ce n'est pas une VM : on ne prétend pas résister à une faille du noyau,
ni disposer d'une limite agrégée de mémoire/disque par cgroup. Linux x86_64 et les protections vérifiées
sont exigés ; Windows orchestre seulement. Aucun code modèle n'est exécuté directement sur Windows.

L'audit hook Python E11 refuse fork/subprocess et ne convient pas à bash. Il demeure **inchangé** :
ce nouveau bac réutilise le principe Landlock, avec une adaptation explicite permettant fork/exec,
compensée par les namespaces et le filtre système hérité par tous les programmes.

## Protocole arrêté avant le pilote

Toutes les stratégies : mêmes trois tâches, ordre fixe, température 0, seed 0, 20 étapes maximum,
1 024 jetons de completion au plus par requête, **12 288 plafonds de requêtes cumulés par épisode**,
900 s par épisode, 30 s par commande, 120 s pour la vérification finale. Chaque appel pair ou référence,
même en échec, réserve son plafond avant de partir ; aucun remboursement selon l'usage ou la réussite.
Cette comptabilité de plafonds est distincte des jetons effectivement générés (comme le pilote Harbor).
Le libellé « plafonds facturés » du rapport signifie **réservations du budget de requêtes en jetons**,
pas facturation monétaire, dépense API ou consommation réelle de jetons.

- `single` : un modèle présélectionné explicitement par le parent, une proposition par étape.
- `vote` : une proposition par pair nommé ; pluralité sur `agent.normalise`, égalité tranchée par l'ordre
  CLI des pairs, fixé avant la campagne. Aucun poids appris sur ces trois exercices.
- `cascade` : mêmes propositions ; référence appelée si au moins une réponse est mal formée **ou** si
  les commandes normalisées ne sont pas unanimes. Elle reçoit seulement les instructions et le même
  historique de tests visibles. Sa commande est choisie ; réponse de référence mal formée = étape sans
  commande exécutée. Aucun correcteur caché ou résultat final n'intervient dans le choix.
- `reference` : si `--reference` est donné, trois épisodes supplémentaires avec la référence seule,
  stratégie `single`, strictement les mêmes budgets. La grille complète exige aussi ces trois verdicts.

Si le délai global expire pendant la génération, le plafond de requête reste réservé, l'usage devient
inconnu et l'épisode se termine par `stopped=timeout`. Les sources déjà produites sont vérifiées et la
tâche reste dans le dénominateur. Une panne endpoint avant cette échéance est une erreur d'infrastructure.

La boucle reste `essaim/agent.py`, une commande dans un bloc bash par tour, fin par sa sentinelle habituelle.
La cascade est réalisée côté boucle avec des appels Myriad ciblés, sans exiger une cascade native.
Chaque requête demande `myriad.k=1`, sans arrêt anticipé et sans reformulation d'instructions par la passerelle.
Pour un serveur compatible standard, ce champ additionnel peut être ignoré ; vérifier sa compatibilité
avant le pilote. Employer des noms exacts ou `myriad:family=...`, jamais un nom d'essaim non ciblé.

La vérification finale ignore tous les tests/artifacts/notes produits par l'agent : seules les sources
autorisées sont relues (fichiers réguliers bornés, sans liens), puis réappliquées dans des copies fraîches.
**Les tests officiels originaux s'exécutent dans le contrôleur de confiance** ; les fonctions du candidat
sont remplacées par des appels isolés qui transmettent seulement entrées et résultats JSON ordinaires.
Le code candidat ne voit ni les assertions, ni les valeurs attendues du contrôleur, ni l'objet de résultat.
Chaque appel repart d'une copie source neuve. Les tests visibles pendant l'épisode ne sont pas secrets.
Cette adaptation est limitée aux API de fonctions pures des trois exercices : elle ne couvre pas les
31 autres exercices préparés, notamment leurs API à objets/état. Ce n'est pas le harnais officiel Aider.

Une erreur de noyau/lancement/protection, de préparation ou du harnais donne `status=error, passed=null`.
Après mise en place des protections, un arrêt candidat (`os._exit(0)` compris), une réponse non JSON,
une syntaxe invalide ou un délai candidat donnent un **échec noté**, jamais une
exclusion de la tâche. La limite par appel est 10 s, incluse dans les 120 s de vérification par exercice.
Le retour JSON utilise un fichier dédié borné, régulier et sans liens, séparé des diagnostics stdout/stderr
conservés dans le journal. Des `print` n'altèrent pas la note : seules les valeurs/exceptions retournées
sont comparées aux assertions officielles dans le contrôleur. Un score imprimé par le candidat est ignoré ;
un arrêt sans retour valide échoue. Une source supprimée, illisible, non UTF-8 ou contraire aux gardes
de fichiers après l'épisode donne aussi un échec candidat, avec zéro test exécuté et couverture incomplète.
La même anomalie dans les sources initiales reste une erreur de préparation, avant toute requête modèle.
Le vérificateur est fail-fast dès le premier échec ; `tests`, `tests_expected`, `tests_complete`, `fail_fast`
et la raison d'échec distinguent explicitement tests exécutés et tests restants. On ne prétend pas avoir
exécuté tous les tests d'une solution infinie. Une campagne interrompue ou avec erreur infra ne peut pas
produire de rapport final. Les tests de référence du smoke doivent, eux, tous réussir.

## Lancements déclarés par le parent

Le parent fournit un service compatible `/v1/chat/completions` en loopback HTTP, ou HTTPS authentifié.
La clé provient uniquement de `MYRIAD_BENCH_TOKEN`, jamais d'un argument/fichier. Le client refuse les
redirections, ignore les proxys d'environnement, borne le temps mur de chaque requête et ferme ses clients.
`MYRIAD_BENCH_URL` est l'URL de base terminée par `/v1`, **sans** `/chat/completions`.

Le premier pilote CPU a utilisé **le serveur Linux direct llama.cpp préparé par le parent**, loopback
WSL natif ; il ne mesure ni la passerelle ni un réseau Myriad. Le parent a arrêté le serveur après la mesure
(port 18490 fermé). Le loopback Windows–WSL étant indisponible ici, il ne faut pas élargir l'écoute pour
le contourner. Aucun tracker ou nœud Myriad n'a été lancé. Le parent seul fixe le protocole des services futurs.

**Avant toute inférence**, `--provenance` exige un JSON nonsecret (UTF-8, BOM accepté), copié dans le
manifeste et lié à la reprise. Le fichier solo fourni par le parent contient : `declared_at`, `endpoint_kind`,
`model`, `alias`, `licence`, `repository`, `revision`, `file` (nom seul), `quantization`, `bytes`, `sha256`,
`server`, `hardware`, `energy`. Champs imbriqués obligatoires :

- `server` : `version`, `commit`, `artifact`, `sha256`, `context`, `parallel`, `gpu_layers`,
  `reasoning_budget`, `jinja` ; valeurs explicites, aucune ligne de commande avec credentials.
- `hardware` : `cpu`, `host_memory_gib`, `execution`, `gpu`, `cpu_threads`.
- `energy` : `watts`, `eur_kwh`, `measured=false`, `scope`, mêmes hypothèses que les options CLI.

Pour plusieurs modèles, fournir `{"models": {"NOM_DE_REQUETE": DECLARATION_COMPLETE, ...}}`, couvrant
exactement tous les pairs/références demandés. Champs supplémentaires/credentials refusés. Cette provenance
est **déclarée par le parent**, pas une attestation distante des poids. Les métadonnées effectivement retournées
restent conservées séparément. Le parent a déclaré Qwen3.5-2B Q8_0, llama.cpp b11505 / `ff5888f99`, contexte
32 768, un slot, CPU seul, reasoning-budget 0 ; 100 W et 0,25 €/kWh sont des hypothèses.

Depuis WSL natif, `MYRIAD_BENCH_URL` et `MYRIAD_BENCH_TOKEN` sont définis par le parent,
`PROVENANCE` désignant son fichier JSON (aucune clé à copier dans les journaux). La commande CPU ci-dessous
décrit l'exécution terminée : **ne pas la relancer dans ce dossier**. Toute campagne future reçoit un nom
distinct conforme à `code_pilot_{cpu,compare}_AAAAMMJJ_NN`, choisi avant la mesure.

```bash
"$PY" run_code_pilot.py --check --single 'Qwen3.5-2B' --provenance "$PROVENANCE"
"$PY" run_code_pilot.py --output results/code_pilot_cpu_20261010_01 \
  --single 'Qwen3.5-2B' --provenance "$PROVENANCE"
# Plus tard : même pilote, comparaisons déclarées, limites par défaut identiques.
"$PY" run_code_pilot.py --output results/code_pilot_compare_20261010_01 \
  --single 'myriad:family=FAMILLE_SOLO' \
  --vote 'myriad:family=FAMILLE_A' 'myriad:family=FAMILLE_B' \
  --reference 'MODELE_REFERENCE_EXACT' --provenance "$PROVENANCE_COMPARAISON"
"$PY" run_code_pilot.py --report --output results/code_pilot_compare_20261010_01
```

Le serveur Linux direct est accepté avec un modèle/alias exact et `--host 127.0.0.1 -ngl 0`.
Aucun GPU local utilisable n'est supposé, aucune configuration réseau n'est modifiée.
Le parent seul autorise et supervise toute campagne Colab ou tout service futur.
Une courbe de référence n'apparaît que si ses trois épisodes seuls ont effectivement été exécutés et notés.

## Résultat CPU du 2026-10-10

Campagne `code_pilot_cpu_20261010_01`, lancée par le parent à **00:50:43 CEST**, terminée à **01:07:45 CEST**
le 10 octobre 2026, processus terminé avec code 0. Qwen3.5-2B Q8_0, CPU i5-10400F/32 Gio sous WSL2,
llama.cpp b11505 (`ff5888f99`), contexte 32 768, un slot, zéro couche GPU, `reasoning_budget=0`.
Les réponses ne contiennent pas de champ de raisonnement séparé. Les limites ci-dessus étaient fixées
avant la mesure ; aucune tâche ni aucun budget n'a été changé selon la réussite.

Le [rapport original](../phase0/results/code_pilot_cpu_20261010_01/report.md) et le
[résumé JSON](../phase0/results/code_pilot_cpu_20261010_01/summary.json) donnent **0/3**, mode `single` seul.
Ni vote, ni cascade, ni référence seule n'ont été mesurés. Ce résultat n'est ni un score officiel Aider
sur 225 exercices/six langages, ni une évaluation de dépôt logiciel 5c, ni une mesure de passerelle/essaim Myriad.

| tâche | arrêt | temps mur avec vérification | jetons réels | plafonds réservés |
| --- | --- | --- | --- | --- |
| affine-cipher | soumis | 76,141 s | 673 | 3 072 |
| beer-song | délai global | 900,283 s | 7 168 connus + un appel inconnu | 8 192 |
| book-store | soumis au premier appel | 22,311 s | 149 | 1 024 |

Sur [affine-cipher](../phase0/results/code_pilot_cpu_20261010_01/single__affine-cipher.jsonl), la première
réponse contient deux blocs bash : elle est rejetée pour format, sans exécution. Le modèle lance ensuite
`python3 affine_cipher.py`, qui exécute seulement le squelette sans appeler ses fonctions ni les tests.
Il interprète le code retour 0 comme des tests réussis et soumet. Les fonctions `encode`/`decode` sont
restées `pass` ; le premier test officiel de décodage échoue sur le retour `None` (1 test exécuté sur 16).

Sur [beer-song](../phase0/results/code_pilot_cpu_20261010_01/single__beer-song.jsonl), les sept réponses
achevées énumèrent les couplets et atteignent chacune 1 024 jetons (`finish_reason=length`) avant la fermeture
du bloc bash. Toutes sont rejetées pour format ; aucune commande n'est exécutée. Le huitième appel dépasse
le délai global pendant la génération : son usage est inconnu. La tâche reste notée, avec `recite` inchangé
(`pass`) ; le premier test officiel, couvrant tous les couplets, échoue (1 test exécuté sur 8).
Les 900,283 s incluent la vérification et les frais de fin d'épisode après le plafond de génération de 900 s.

Sur [book-store](../phase0/results/code_pilot_cpu_20261010_01/single__book-store.jsonl), la première réponse
expose une idée de regroupement puis émet directement la sentinelle de soumission, sans écrire ni tester
de code. `total` reste `pass` ; le premier test officiel de regroupement par quatre échoue sur `None`
(1 test exécuté sur 20). Dans les trois épisodes, les sources finales sont identiques aux sources initiales.
Les verdicts sont des échecs candidats valides, sans erreur du harnais ; la campagne est complète,
mais les vérifications sont fail-fast et ne prétendent pas avoir exécuté les 44 tests sur ces candidats.

Le total de jetons générés reste **inconnu** : 7 990 sont connus, plus le travail non retourné du dernier
appel beer-song. Les 12 288 « plafonds facturés » du rapport sont les réservations cumulées de cette campagne,
distinctes du plafond de 12 288 par épisode. Temps moyen observé : 332,912 s/tâche. À **100 W supposés** et
**0,25 €/kWh supposé**, l'énergie/coût moyens sont estimés à 9,247545 Wh et 0,002311886 €/tâche ;
ce ne sont ni des mesures électriques ni une dépense monétaire observée.

Figures originales : [exactitude/temps, PNG](../phase0/results/code_pilot_cpu_20261010_01/accuracy_seconds_per_task.png)
([SVG](../phase0/results/code_pilot_cpu_20261010_01/accuracy_seconds_per_task.svg),
[PDF](../phase0/results/code_pilot_cpu_20261010_01/accuracy_seconds_per_task.pdf)) et
[exactitude/coût estimé, PNG](../phase0/results/code_pilot_cpu_20261010_01/accuracy_estimated_eur_per_task.png)
([SVG](../phase0/results/code_pilot_cpu_20261010_01/accuracy_estimated_eur_per_task.svg),
[PDF](../phase0/results/code_pilot_cpu_20261010_01/accuracy_estimated_eur_per_task.pdf)).
Wilson 95 % ≈ [0 ; 0,561] est **descriptif pour n=3** : cette présélection lexicographique n'est pas
un échantillon représentatif. L'intervalle ne justifie aucune généralisation sur le modèle ou les autres bancs.

Le [manifeste](../phase0/results/code_pilot_cpu_20261010_01/manifest.json) et les
[verdicts](../phase0/results/code_pilot_cpu_20261010_01/verdicts.json) ont été vérifiés avec les trois
transcriptions : grille complète, paramètres, usage, empreintes des sources, cinq modules, lock et données.
SHA-256 manifeste : `7f51409dacf240388ffa0b7e850b4ceb11adc3c4ff2b9efdd0e7e8069695b1ee` ;
verdicts : `ff580168ee83972b8b13c5ca689ac9b3715262289eabab58b2c82ec1ec140551`.
Le mode `--report` a été rejoué **sur une copie temporaire**, sans service ni inférence : résumé/rapport
et deux PNG identiques octet pour octet ; SVG/PDF identiques après neutralisation des dates de création
et identifiants SVG générés. Les 13 artefacts originaux sont conservés sans modification.
Pour reproduire cette vérification hors inférence, copier uniquement les cinq entrées dans un dossier neuf :

```bash
REPRO="$(mktemp -d)"
cp results/code_pilot_cpu_20261010_01/{manifest.json,verdicts.json,single__*.jsonl} "$REPRO/"
"$PY" run_code_pilot.py --report --output "$REPRO"
cmp results/code_pilot_cpu_20261010_01/summary.json "$REPRO/summary.json"
cmp results/code_pilot_cpu_20261010_01/report.md "$REPRO/report.md"
```

## Journaux, reprise et rapport

`manifest.json` contient identifiants, empreinte des 34 tâches, sources du pilote, hash des cinq modules,
lock uv, versions Python/paquets et provenance structurée serveur/modèle déclarée, endpoint sans credentials, instructions,
limites et hypothèses énergétiques. Chaque `STRATEGIE__TACHE.jsonl` conserve requêtes/paramètres,
modèle servi, métadonnées Myriad, message complet (raisonnement retourné inclus), `finish_reason`, réponses
de tous les pairs/référence, votes, cascade, commandes,
observations, reçu d'isolation, sources avant/après et verdict. Aucune clé d'authentification n'est journalisée.
Les durées mur sont conservées par requête, épisode et vérification ; les appels de vérification sont détaillés.

Les jetons réels incluent le raisonnement retourné dans le comptage de completion du serveur, sans retrait.
Ils utilisent `usage.completion_tokens` pour un serveur standard et
`myriad.total_completion_tokens` pour la passerelle (tous les pairs ayant répondu). Sans usage ou avec
des pairs remplacés/échoués non comptabilisés, le total est **inconnu** ; la somme partielle connue et le
nombre de requêtes à usage inconnu sont conservés. Les plafonds réservés sont rapportés séparément.
La passerelle peut faire des remplacements internes : leur travail non retourné ne devient pas une mesure.

Un verrou exclusif refuse deux contrôleurs simultanés dans le même dossier, rapport compris.
Un arrêt brutal peut laisser `.campaign.lock` : conserver le dossier et en utiliser un nouveau, ou faire
examiner par le parent le PID enregistré avant de retirer explicitement un verrou devenu orphelin.
Écriture atomique des manifestes/verdicts ; `--resume` exige une provenance exactement identique et les
empreintes des transcriptions. Seuls les épisodes intégralement notés sont repris ; une erreur ou une
transcription orpheline interdit une relance silencieuse : conserver la campagne, en créer une nouvelle
avec un nom distinct et documenter la cause. L'ordre et la présélection ne changent pas selon les scores.

Le rapport JSON+Markdown et deux figures matplotlib (chacune SVG, PNG et PDF) ne sont produits que pour une
grille complète de verdicts valides. Le tracé est importé à la demande, hors des tests indépendants.
Ils donnent exactitude/Wilson 95 %, coût énergétique **ESTIMÉ** et temps par tâche, uniquement pour les
stratégies effectivement exécutées. Les figures contiennent les valeurs exactes en métadonnées.
Par défaut : **100 W supposés**, **0,25 €/kWh supposé** ; `Wh = W × secondes / 3600`,
`€ = Wh × €/kWh / 1000`. Le temps inclut agent et vérification, hors préparation/smoke.
Adapter `--watts` à une puissance agrégée explicite pour des pairs distants ; ce n'est pas une mesure
électrique, ni une estimation du prix API. Trois exercices donnent des intervalles très larges et une
description, pas une comparaison scientifique définitive. Wilson avec n=3 ne transforme pas cette
présélection en échantillon représentatif. Aucun résultat simulé n'est livré comme mesure.

Les tests par défaut sont sans réseau et sans données téléchargées. Les tests WSL réels sont activés
explicitement ci-dessus ; les fixtures de faux chat sont des contrôles du logiciel uniquement.
Suites obligatoires : `cd app && uv run pytest -q` ; dans `phase0`,
`uv run python -m unittest discover -s tests -q`. L'audit Astra est lancé par le parent après ce commit.

L'export public autorise uniquement les dossiers `results/code_pilot_{cpu,compare}_AAAAMMJJ_NN` et leurs
22 noms d'artefacts fixes : manifest/verdicts/summary JSON, report Markdown, les douze transcriptions
stratégie × tâche et les six figures prévues. Aucun cache, smoke, lock, fichier partiel, archive, fichier
voisin ou sous-dossier n'est exportable par cette règle. Chaque artefact reste soumis au scan bloquant
des secrets et chemins personnels. La campagne CPU terminée livre les 13 artefacts autorisés correspondant
au seul mode mesuré : quatre fichiers manifest/verdicts/summary/report, trois JSONL et six figures.
