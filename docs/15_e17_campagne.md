# E17 â€” campagne DEV R00 sur VM, prÃ©paration sans allocation

Ce jalon fournit `e17-campaign`, indÃ©pendant de `colab_jobs.py`. Il ne contient aucun score,
aucune donnÃ©e inventÃ©e dans le registre et aucune ouverture de TEST. P01 conserve R00 (`067cf41`) :
les modifications de transport, de chargement et de supervision ne prouvent aucun gain du harnais.
Le backend final `15ecb9f`, fusionnÃ© en `5677052` puis intÃ©grÃ© depuis `main 1926e05`, est raccordÃ©.
La sonde autorisÃ©e du parent est compatible et sa VM a Ã©tÃ© libÃ©rÃ©e ; ce dÃ©veloppeur ne lance aucun
Colab rÃ©el. L'autorisation propriÃ©taire du parent existe dÃ©jÃ  : aucune nouvelle confirmation n'est
requise. L'audit Astra du lanceur et son intÃ©gration restent Ã  terminer avant son exÃ©cution parent.

## Plans fixÃ©s avant score

`e17-dev-single` : L4, affine-cipher / beer-song / book-store, Qwen3.5-2B Q8_0 seul.
`e17-dev-compare` : A100, les mÃªmes trois DEV, dans l'ordre single, vote, cascade, rÃ©fÃ©rence.
Les pairs sont Qwen3.5-2B Q8_0 puis Granite4.2-3B Q8_0 ; cascade vers Qwen3.8-27B Q8_0,
rÃ©fÃ©rence 27B seul. Aucun remplacement ni changement de quantification aprÃ¨s rÃ©sultat.
Le registre distingue `initial-single-L4` et `comparison-A100`, avec identitÃ©s de campagnes sÃ©parÃ©es.
Le single A100 est une nouvelle tentative DEV dÃ©clarÃ©e ; il ne remplace ni n'efface le single L4,
et les deux ne sont pas interchangeables pour attribuer une diffÃ©rence de temps au harnais.

Les deux petits modÃ¨les reprennent exactement rÃ©visions, tailles et SHA-256 du catalogue app.
Les deux archives CUDA 12.8 officielles b11505 reprennent les Ã©pingles de `app/myriad/llamacpp.py` ;
le binaire exÃ©cutÃ©, les fichiers extraits, la version et le commit `ff5888f99` sont relevÃ©s sur VM.
Le 27B est fixÃ© avant score Ã  la rÃ©vision GGUF `4ca720788d1e01f1bff70c033e0d0028fd02e502`,
29 047 086 048 octets et SHA-256 `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`,
licence Apache-2.0, selon `reference-metadata.json` fourni par le parent. Ce n'est pas la rÃ©vision du
tokenizer. `--reference-revision` peut rÃ©pÃ©ter cette Ã©pingle mais refuse une substitution ; il n'est
plus un prÃ©requis technique ouvert. Aucun accÃ¨s rÃ©seau n'a Ã©tÃ© fait par le dÃ©veloppeur pour la dÃ©terminer.

TempÃ©rature 0, seed 0, 4 096 / 49 152 jetons rÃ©servÃ©s, 20 Ã©tapes, 900 s d'Ã©pisode,
30 s de commande, 120 s de vÃ©rification restent identiques dans tous les modes. Le diagnostic final
garde son budget sÃ©parÃ©. Douze appels maximaux : six tours de vote, quatre tours si chaque cascade
appelle aussi la rÃ©fÃ©rence. Aucun plafond ne change automatiquement selon les verdicts.

Tout calcul se dÃ©roule sur VM : tÃ©lÃ©chargement, validation des poids/donnÃ©es, installation du runtime,
serveur, contrÃ´leur, agent et vÃ©rification. Le PC transmet au plus 8 Mio de sources explicitement
sÃ©lectionnÃ©es et rÃ©cupÃ¨re au plus 64 Mio d'artefacts. Aucun modÃ¨le ni jeu de donnÃ©es ne part du PC.
Le manifeste de donnÃ©es est une vue des Ã©pingles existantes limitÃ©e aux fichiers des trois DEV et
Ã  leur licence et au `README.md` d'attribution requis par `copy_task`. Les mÃ©tadonnÃ©es du catalogue
de 34 tÃ¢ches restent intactes pour le contrÃ´le historique
de sÃ©lection ; aucun contenu des autres exercices n'est extrait, exÃ©cutÃ© ou lu par le vÃ©rificateur.
Le processus VM dÃ©diÃ© sÃ©lectionne cette vue via `aider_python.MANIFEST`, sans modifier la source
de l'adaptateur, le prompt, la boucle, les assertions ou les budgets R00.

Un seul modÃ¨le occupe le GPU Ã  la fois. Le client transmis Ã  `code_pilot.run_one` choisit le modÃ¨le
du payload, arrÃªte son propre serveur, charge le suivant et utilise son endpoint loopback. Les temps
de chargement consomment le budget de l'appel/Ã©pisode : ils ne sont ni cachÃ©s ni retirÃ©s des mesures.
Les Ã©vÃ©nements `loaded` / `load-failed` enregistrent leur durÃ©e en secondes, arrÃªt du modÃ¨le prÃ©cÃ©dent
compris ; les Ã©vÃ©nements `request` mesurent sÃ©parÃ©ment l'appel HTTP. Les tentatives du registre portent
les sommes `loading_seconds` / `request_seconds`. Le champ historique de temps d'appel dans les verdicts
reste inchangÃ© et inclut le client complet, donc son chargement Ã©ventuel : il ne faut pas le sommer
une seconde fois avec les durÃ©es sÃ©parÃ©es. Cette distinction conserve les stratÃ©gies et budgets R00.
Un slot, contexte 32 768, `--jinja`, 999 couches GPU demandÃ©es, raisonnement 0, aucun fallback ni retry
de serveur. La L4 exige au moins 22 000 Mio observÃ©s ; l'A100 au moins 39 000 Mio, pour la rÃ©servation
historique du 27B et la marge. Ces gardes ne garantissent pas qu'un contexte tient : OOM reste une erreur
d'infrastructure, sans rÃ©duction silencieuse. C'est une **simulation d'essaim centralisÃ©e sur une VM**,
sans app, tracker, nÅ“ud, routage serveur nouveau ou preuve de rÃ©seau distribuÃ©.
`LD_LIBRARY_PATH` inclut les rÃ©pertoires des bibliothÃ¨ques extraites et `/usr/lib64-nvidia`, comme le
bootstrap historique. Cet environnement est transmis Ã  `--version`, `--list-devices` et chaque serveur.
Le contrÃ´leur exige un pÃ©riphÃ©rique `CUDA0` avant infÃ©rence et demande `--device CUDA0` : absence de
CUDA, erreur de bibliothÃ¨que ou de chargement bloque la campagne, sans fallback CPU. La liste observÃ©e
et le chemin de bibliothÃ¨ques figurent dans `runtime.json`.

## Contrat minimal avec le backend distinct

Aucun fichier du worktree `C:/wt-e17-ephemeral` n'a Ã©tÃ© Ã©crit. Aucun fichier `code_pilot`,
`aider_python`, `local_linux` ou `e17_probe*.py` n'est modifiÃ© par ce jalon.
Le lanceur refuse `--execute` avant le moindre CLI si le code transfÃ©rÃ© ne dÃ©clare pas
`--sandbox-backend`. Ce garde de dÃ©pendance n'est pas une attestation ; la VM vÃ©rifie ensuite :

- `code_pilot.parser()` accepte `--sandbox-backend ephemeral-vm`, avec strict par dÃ©faut ;
- `code_pilot.execute(args)` pour `--prepare` et `--smoke`, puis `code_pilot.run_one(args, task, mode,
  models, client, output, campaign_hash)` propagent ce choix Ã  l'environnement d'agent et Ã  **toutes**
  les vÃ©rifications, y compris le diagnostic final ;
- le contrÃ´leur appelle `essaim.ephemeral_vm.require_colab()`, sans `find_spec("google.colab")` dans
  le venv : Linux x86_64 hors WSL, fichiers `/var/colab/hostname` et `/opt/google/drive/drive` requis ;
- les reÃ§us exacts sont `isolation_ready="ephemeral-vm-v1"`, `landlock_abi=null`, `seccomp="on"`,
  `namespaces=["user","mount","pid","net","ipc","uts"]`. Aucun champ `backend` ou
  `colab_detected` n'est inventÃ© dans ces reÃ§us. Les grades/appels dÃ©clarent sÃ©parÃ©ment
  `sandbox_backend="ephemeral-vm"` et les appels candidats `python_audit_hook=true` ;
- le smoke conserve le format historique et les 16 / 8 / 20 tests attendus, rÃ©fÃ©rences vertes complÃ¨tes
  et incorrects rejetÃ©s. Chaque `calls[].isolation` porte le reÃ§u ci-dessus ; l'Ã©vÃ©nement `isolation`
  de la transcription le porte lorsque le shell a effectivement Ã©tÃ© lancÃ©. Les actions test officiel,
  soumission ou erreur de format peuvent ne lancer aucun shell : son reÃ§u reste alors `null`, observÃ©,
  avec aucun appel shell acceptÃ© sans reÃ§u. Le prÃ©flight de campagne et chaque appel de vÃ©rification
  exigent leur reÃ§u rÃ©el, y compris dans `step.note.official_test` et le diagnostic final.

Ce sont les seules dÃ©pendances nouvelles. Aucun changement du serveur n'est demandÃ© :
`run_one` possÃ¨de dÃ©jÃ  l'argument `client`, utilisÃ© par le nouveau `SequentialClient`.
Le manifeste porte le reÃ§u de prÃ©flight de campagne et `colab_image` (marqueurs, systÃ¨me, machine,
noyau rÃ©ellement observÃ©s) ; le runtime conserve cette mÃªme preuve. Le smoke prÃ©cÃ¨de tout
tÃ©lÃ©chargement de poids ou dÃ©marrage de serveur. Les contraintes du backend restent celles de son audit.

Le runtime dÃ©diÃ© est crÃ©Ã© sur VM par uv 0.11.21, `uv sync --frozen --python 3.12.15` sur le lock existant
du pilote. Installation bornÃ©e par Ã©tape (600 s) et par l'enveloppe totale ; pas de retry automatique.
Les archives serveur ont chacune 600 s, chaque poids 1 200 s, lectures rÃ©seau 30 s et hashes/tailles
obligatoires avant utilisation. Le Python, uv, les packages installÃ©s, le lock, les sources transfÃ©rÃ©es,
les fichiers serveur et le GPU rÃ©ellement observÃ© restent dans les artefacts. Les hashes des exÃ©cutables
sont des observations, pas des Ã©pingles prÃ©alables de la distribution Python/uv. Le noyau, le pilote CUDA
et les bibliothÃ¨ques systÃ¨me de Colab ne sont pas reproductibles bit Ã  bit par ce plan.

## Cycle de vie et bornes

`Campaign` Ã©tend `e17_probe.Probe` : CLI assainie/bornÃ©e, verrou de cycle de vie commun,
autoritÃ© en mÃ©moire aprÃ¨s `new` confirmÃ©, contrÃ´le du reÃ§u, compte avant/pendant/aprÃ¨s et cleanup auditÃ©.
L'extension fixe le GPU du plan et ne change pas les helpers historiques. Un reÃ§u/session prÃ©existant
interdit `new`. Une seule tentative ; aucun retry de `new` ambigu, aucune adoption automatique ou arrÃªt
d'une identitÃ© Ã©trangÃ¨re. L'accusÃ© distant lie campagne, allocation, session, nonce, plan et hashes sources.
Un dÃ©marrage sans accusÃ© reste un Ã©chec, mÃªme si ses artefacts sont rÃ©cupÃ©rÃ©s ensuite.

Le travail VM possÃ¨de son propre watchdog de groupe de processus. Le superviseur garde 1 800 s de
rÃ©cupÃ©ration dans ses 5 400 s par dÃ©faut, puis rÃ©serve sÃ©parÃ©ment 1 800 s de cleanup. Le compte est
relu toutes les 30 s pendant les RPC ; toute lecture inconnue, tarif dÃ©passÃ© ou rÃ©serve atteinte ferme
le travail. Ce dÃ©faut reste mÃ©morisÃ© : une nouvelle rÃ©cupÃ©ration ne reprend pas des opÃ©rations ordinaires
aprÃ¨s une erreur de compte. Les signaux TERM/INT sont diffÃ©rÃ©s pendant `new` jusqu'Ã  confirmation
mÃ©moire ; rÃ©pÃ©tÃ©s pendant la clÃ´ture, ils sont ignorÃ©s. Le cleanup borne ses retries de `stop`, confirme
l'absence et Ã©crit l'usage final durable avant d'effacer le reÃ§u.

`--spent-units` est obligatoire : le parent y inscrit l'enveloppe E17 dÃ©jÃ  consommÃ©e/engagÃ©e, sonde
comprise selon sa comptabilitÃ© prudente. `spent + budget <= 20` et plancher au moins 15 sont imposÃ©s.
Il n'y a pas de registre global multi-campagne automatique : le parent fixe chaque sous-campagne,
une seule Ã  la fois sous le reÃ§u/verrou partagÃ©. Le solde rÃ©el est Ã  lire de nouveau au lancement.
Une diffÃ©rence de compte n'est pas une facture scientifique ; les interruptions gardent leurs coÃ»ts inconnus.

La marge comptable par dÃ©faut est 161 s (lecture 120 + deux grÃ¢ces 5 + 31). Au plafond 5,3 unitÃ©s/h :

| Enveloppe | Borne conservatrice |
| --- | ---: |
| single, 5 400 s + cleanup 1 800 s + marge 161 s | 10,836 unitÃ©s, donc budget 11 |
| trois Ã©pisodes, 3 Ã— (900 + 120) s, hors prÃ©paration/cleanup | 3 060 s |
| douze Ã©pisodes, 12 Ã— (900 + 120) s, hors prÃ©paration/cleanup | 12 240 s |
| plafond E17 total 20 au mÃªme tarif, cleanup/marge dÃ©duits | 11 623,9 s de travail au plus |

Le plafond logiciel est 11 500 s, et le parent dÃ©duit les blocs prÃ©cÃ©dents. La grille complÃ¨te n'est
**pas garantie**, mÃªme avant installation et chargements. La comparaison reste une option conditionnÃ©e
par une projection prudente aprÃ¨s les trois single ; le tarif L4 historique 1,54/h n'est pas une
observation nouvelle ni le plafond par dÃ©faut. Une enveloppe insuffisante arrÃªte avant l'Ã©pisode suivant,
sans hausse de budget, remplacement de tÃ¢che ni reprise opportuniste. L'installation peut Ã©puiser
l'enveloppe avant toute infÃ©rence. La rÃ©serve de cleanup est conservÃ©e dans tous les cas.

## PrÃ©paration et lancement exact

Depuis Git Bash, uniquement la premiÃ¨re commande est une prÃ©paration sans CLI :

```bash
MSYS_NO_PATHCONV=1 wsl.exe -e bash -lc '
  repo=$1; evidence=$2; spent=$3
  mkdir -p "$evidence"
  bash "$repo/phase0/colab/colab_phase0.sh" e17-campaign \
    --plan e17-dev-single --output "$evidence/single-plan.json" \
    --budget-units 11 --spent-units "$spent"
' bash "$REPO_WSL" "$EVIDENCE_WSL" "$E17_SPENT_UNITS"
```

**Commande rÃ©servÃ©e au parent aprÃ¨s audit/intÃ©gration du lanceur, selon l'autorisation dÃ©jÃ  donnÃ©e**
(jamais exÃ©cutÃ©e pour ce jalon ; utiliser un dossier neuf et le mÃªme reÃ§u/verrou partagÃ©) :

```bash
MSYS_NO_PATHCONV=1 wsl.exe -e bash -lc '
  repo=$1; evidence=$2; spent=$3
  mkdir -p "$evidence"
  bash "$repo/phase0/colab/colab_phase0.sh" e17-campaign --execute \
    --plan e17-dev-single --output "$evidence/single-run.json" \
    --seconds 5400 --budget-units 11 --spent-units "$spent" \
    --min-balance-units 15 --cleanup-seconds 1800 --recover-seconds 1800 \
    --max-rate-units-hour 5.3
' bash "$REPO_WSL" "$EVIDENCE_WSL" "$E17_SPENT_UNITS"
```

Pour l'option A100 : `--plan e17-dev-compare` (rÃ©vision 27B dÃ©jÃ  fixÃ©e),
avec `--seconds`, `--budget-units`, `--spent-units` fixÃ©s par le parent **avant score** dans le reliquat.
Pas de boucle d'allocations, pas de TEST, P02 ni gel implicite Ã  ce jalon.

## RÃ©cupÃ©ration, registre et limites

Le superviseur annule le seul worker Ã©pinglÃ© par PID **et** instant de naissance avant rÃ©cupÃ©ration
de clÃ´ture, sauf s'il a dÃ©jÃ  fini. Un statut final inventoriÃ© est requis : jamais de hash d'un flux ouvert.
Les fichiers listÃ©s sont rapatriÃ©s par blocs de 256 Kio vers un dossier neuf, avec couverture/tailles/SHA-256.
Les manifestes, registre R00/P01, smoke, provenance, runtime, traces de serveurs, transcriptions et verdicts
doivent tous Ãªtre prÃ©sents et cohÃ©rents avant `dev-complete`. Un retour CLI zÃ©ro ou un champ `complete`
ne suffit pas : toutes les cellules dÃ©clarÃ©es et leurs fins de transcription sont contrÃ´lÃ©es.
Le journal contrÃ´leur est fermÃ© avant inventaire, et l'inventaire lui-mÃªme n'est pas inclus dans ses hashes.

Sur Ã©chec, rÃ©cupÃ©rer les fichiers existants tant que le compte et la propriÃ©tÃ© restent lisibles ;
les garder comme preuves partielles, avec les hashes vÃ©rifiÃ©s, sans les dÃ©clarer complets. En cas de panne
rÃ©seau persistante, compte inconnu ou budget de rÃ©cupÃ©ration Ã©puisÃ©, la libÃ©ration prime : la rÃ©cupÃ©ration
peut rester incomplÃ¨te. `closed`, `absence_confirmed` et `retrieved_complete` restent distincts.
Le reÃ§u demeure si absence/compte final ne sont pas confirmÃ©s. Le parent doit alors rÃ©concilier la
propriÃ©tÃ© retenue ; une allocation ambiguÃ« n'est jamais adoptÃ©e par ce lanceur.

Le registre conserve une tentative commencÃ©e avant chaque cellule, sa fin/hash lorsqu'il existe, et
la proposition sans modification. Un 0/3 complet est recevable ; une panne d'infrastructure n'est pas
un Ã©chec candidat. Aucun rÃ©sumÃ© partiel ne peut entrer dans l'article. Ce lanceur ne gÃ©nÃ©ralise pas
l'adaptateur aux exercices TEST et ne prÃ©pare pas BigCodeBench. Ni attestation inviolable de Python
arbitraire, ni sÃ©curitÃ© contre un candidat adversarial sur la VM jetable ne sont revendiquÃ©es.
SIGKILL/panne du PC ou du stockage peuvent interrompre la clÃ´ture : le reÃ§u durable est une responsabilitÃ©,
pas une preuve de libÃ©ration. Les tests locaux sont uniquement synthÃ©tiques, avec CLI/runtime/donnÃ©es
simulÃ©s ; ils ne prouvent pas la compatibilitÃ© d'une VM Colab ou les capacitÃ©s d'un modÃ¨le.


Transport corrigé après audit Astra : blocs de 4 Mio, enveloppe JSON limitée à 8 Mio,
trames base64 de 60 000 caractères (ligne complète < 65 536 octets). Tous les ACK et inventaires
passent par ce format ; empreinte et indices détectent pertes, duplications et altérations.
Le sanitizer reste inchangé. Au plafond 64 Mio, la fragmentation par fichier demande au plus
16 + 22 appels de données en comparaison, plus cancel/status. Simulation à 30 s par RPC
(propriété/compte inclus) : 1 200 s, avec 600 s de marge dans les 1 800 s réservées.
Cette hypothèse n'est pas une mesure Colab ni une garantie réseau : au délai global, cleanup prime.
La réserve est déduite des 5 400 s totales, sans ajouter de temps ni relever le budget 11/20.
L'ordre explicite single/vote/cascade/reference est vérifié dans plan, manifeste, registre et verdicts.
La réservation de métadonnées pré-mesure est transférée et vérifiée par SHA-256 avant travail et rapatriement.

## Corrections du réaudit intégré

Les seules métadonnées canoniques sont `docs/e17_proposition_p01.md` et
`docs/e17_selection_pre_mesure.json`, autorisées individuellement dans la liste publique.
P01 conserve ses octets ; la réservation conserve identifiants, horodatage, règles et provenance,
avec chemin du protocole actualisé et empreinte de son original privé dans `reservation_origin`.
Aucune nouvelle sélection ni lecture de contenu TEST. Les deux fichiers sont transférés comme sources
légères et leurs empreintes sont liées au plan, vérifiées sur VM et au rapatriement.

Le worker fixe une seule échéance monotonic absolue avant installation et la transmet au contrôleur
sur la même VM. Installation, préparation, chargements et réserve entière de 1 020 s par épisode
consomment cette enveloppe initiale ; aucun redémarrage du budget après installation.

Un appel candidat jamais lancé pour `verification_budget` est accepté seulement avec verdict
`graded`, `passed=false`, appel complet, isolation explicitement nulle, durée exactement zéro,
backend et crochet connus, sans diagnostics d'exécution. Tous les appels lancés exigent le reçu
exact. Cela conserve l'échec candidat sans le transformer en succès ou en erreur d'infrastructure.

## Correctif technique après la première tentative DEV

La première tentative réelle L4 du 13 octobre 2026 a échoué avant dataset, smoke officiel,
serveur et modèle : le Python système lançant `venv` a reçu un retour 1 d'`ensurepip`.
Sa cause interne reste inconnue. Les deux artefacts clos ont été récupérés et hashés ;
la VM a été libérée avec absence et clôture confirmées. Aucun score E17 n'en découle.

Le worker évite désormais cette dépendance initiale : archive autonome officielle UV 0.11.21,
`uv-x86_64-unknown-linux-gnu.tar.gz`, 24 437 894 octets,
SHA-256 `8c88519b0ef0af9801fcdee419bbb12116bd9e6b18e162ae093c932d8b264050`, depuis
`https://github.com/astral-sh/uv/releases/download/0.11.21/uv-x86_64-unknown-linux-gnu.tar.gz`.
Seul le membre régulier `uv-x86_64-unknown-linux-gnu/uv` est écrit dans une cible neuve
`installer/bin/uv`, après validation de tous les membres : chemins sortants, liens,
doublons et fichiers spéciaux refusés, tailles bornées, aucun remplacement de cible.
La version est contrôlée avant `uv sync --frozen --python 3.12.15` sur le lock inchangé.
L'archive UV a donc une épingle préalable ; l'empreinte de son exécutable extrait reste une observation,
comme celle du Python installé, dont la distribution n'est pas épinglée par empreinte.

Téléchargement, extraction et version consomment une seule étape de 600 s au plus,
dans la même échéance absolue fixée avant préparation ; lectures réseau limitées à 30 s.
Aucun fallback ni retry automatique. Le runtime partiel conserve l'épingle, les phases,
les observations et une trace d'erreur bornée ; le statut conserve phase et retour.
En cas de réussite, ces métadonnées UV restent dans `runtime.json` et dans le runtime du manifeste.
`provenance.json` conserve l'enveloppe exacte `models` exigée par le validateur commun.
En cas d'échec, les fichiers déjà clos restent inventoriés comme incomplets.

Cette correction est vérifiée uniquement avec des archives et machines synthétiques,
sans téléchargement du binaire réel sur PC. Elle ne constitue pas une nouvelle observation Colab.
R00, modèles, lock, budgets, transport et isolement sont conservés. Pour les admissions futures,
le parent retient une réserve prudente de 0,40 unité pour les tentatives précédentes ;
ce montant n'est pas un coût exact. Un nouvel essai réel reste à faire par le parent après audit et fusion.
