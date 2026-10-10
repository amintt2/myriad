# Phase 0, mode B : des modèles de familles différentes qui répondent ensemble

La question à trancher : est-ce que plusieurs petits modèles **de familles différentes**, gelés et sans aucun
entraînement, font mieux ensemble que le meilleur d'entre eux, et que le modèle plus gros qui tient seul sur
le PC ? Le contexte est résumé dans `../RESEARCH.md`.

| rôle | modèle | famille | machine |
| --- | --- | --- | --- |
| expert | Qwen/Qwen3-1.7B | Alibaba | PC |
| expert | HuggingFaceTB/SmolLM3-3B | Hugging Face | PC |
| expert | ibm-granite/granite-3.3-2b-instruct | IBM | PC |
| expert | google/gemma-4-E2B-it | Google | Mac |
| référence locale | Qwen/Qwen3-4B | Alibaba | PC |

Tous sont sous licence Apache-2.0 et non restreints. Ils tournent en Q8 sur le GPU via llama.cpp (Vulkan
sur le PC, Metal sur le Mac). Les GGUF viennent des dépôts officiels (`fastdl.py`, SHA-256 vérifié) ou sont
convertis localement (`make_gguf.py`).

## Méthode

- **Deux jeux disjoints** par benchmark (`essaim/data.py`) :
  - **dev** (300 questions) sert à choisir la méthode de fusion ;
  - **test** (300 autres) n'est utilisé qu'une fois la méthode figée.
- **5 passages par question**, avec un ordre des réponses différent et équilibré à chaque passage
  (`balanced_perm`) : chaque option passe par plusieurs positions.
- **Lecture exacte** de la probabilité de chaque lettre. Toutes ses graphies d'un seul jeton (« A » et
  « A » précédé d'une espace) sont lues, sans aucune valeur estimée.
- **Fichiers de résultats sûrs** (`essaim/results.py`). Un manifeste enregistre :
  - la révision du tokenizer, l'empreinte du GGUF et l'identité des données ;
  - le protocole, la version du prompt et la date figée des modèles de chat.

  Une reprise avec une autre configuration est refusée, un verrou empêche deux écritures simultanées, et les
  doublons sont détectés.
- **Analyse** (`analyze_mc.py`). Deux protocoles :
  - « 1 passage » ;
  - « moyenne des passages » : un ensemble gratuit, donné aussi aux modèles seuls.

  Fusions comparées :
  - **sans apprentissage** : moyenne, produit, pondérée par la confiance, vote ;
  - **calibrées** : une température par modèle.

  Intervalles de confiance par bootstrap apparié par question. Gain sur chaque expert et sur la référence
  locale. Sur le jeu test, les températures sont **apprises sur dev puis figées** (`--fit-split dev`).

## Expérience 1 : QCM (ARC-Challenge, MMLU-Pro)

```
uv run python run_mc.py --model Qwen/Qwen3-1.7B --gguf ../models/Qwen3-1.7B-Q8_0.gguf --suffix _gpu --split dev
uv run python run_mc.py --model Qwen/Qwen3-1.7B --gguf ../models/Qwen3-1.7B-Q8_0.gguf --suffix _gpu --split test
uv run python analyze_mc.py --split dev --experts Qwen/Qwen3-1.7B@_gpu HuggingFaceTB/SmolLM3-3B@_gpu ibm-granite/granite-3.3-2b-instruct@_gpu google/gemma-4-E2B-it@_gpu --reference Qwen/Qwen3-4B@_gpu
uv run python analyze_mc.py --split test --fit-split dev --experts ... --reference ...
```

Le volet **avec réflexion** (`run_think.py`) utilise le mode réflexion de chaque famille, avec un budget de
768 jetons ; on lit ensuite la lettre exactement de la même façon.

## Expérience 2 : générer ensemble (GSM8K) sur le réseau

Chaque modèle tourne comme un **pair** HTTP (`essaim.peer`), et `run_gen.py` les coordonne :

| mode | ce qui se passe | allers-retours |
| --- | --- | --- |
| `solo` | chaque pair répond seul | 1 |
| `vote` | majorité des réponses finales des pairs (calculé à partir de `solo`) | 1 au total |
| `accord` | chaque pair écrit un brouillon de 16 jetons ; on garde le plus long début commun à la majorité, sinon le brouillon le plus confiant | 1 par tour |
| `croise` | brouillons, puis chaque pair note tous les brouillons, mot par mot ; on garde le meilleur jusqu'au premier mot jugé improbable par le groupe | 2 par tour |

```
uv run python -m essaim.peer --model Qwen/Qwen3-1.7B --gguf ../models/Qwen3-1.7B-Q8_0.gguf --port 8101
ESSAIM_TOKEN=<secret> uv run python run_gen.py --peers http://127.0.0.1:8101 ... http://IP_DU_MAC:8104 --tag essaim4
```

Ce que garantit le coordinateur (vérifié par une revue de code avant l'expérience) :
- **Arrêt « réponse finale » nommé.** Il ignore ce qui est écrit dans la réflexion, y compris une réflexion
  déjà ouverte dans le préfixe.
- **k réponses valides exactement**, prises dans leur ordre d'arrivée :
  - un seul envoi en cours par pair ;
  - un pair en erreur n'arrête pas le tour ;
  - le k effectivement obtenu et les erreurs sont journalisés pour chaque phase.
- **Coûts complets.** Le temps de décision est séparé du coût total, et les appels tardifs ou en échec sont
  comptés dans leur mode.
- **Pairs protégés.** Prompt et candidats sont bornés en jetons, la génération a une échéance, et le pair
  exige un jeton d'accès hors de la machine locale.

## E10 : sections en parallèle (Skeleton-of-Thought entre pairs hétérogènes)

La question : plusieurs pairs peuvent-ils écrire **une seule** réponse longue en même temps, sans perdre en
qualité face à un pair qui répond seul ? Un pair écrit le squelette (3 à 8 points numérotés, 12 mots au plus
chacun), puis le point i est développé par le pair i mod P d'une liste fixe de P familles, tous les points à
la fois (Ning et al., 2023, « Skeleton-of-Thought »).

- **Données** : premiers tours de MT-Bench (`HuggingFaceH4/mt_bench_prompts`, Apache-2.0, commit fixé).
  On garde les catégories à réponse longue faite de parties séparables : writing, roleplay, stem,
  humanities (40 prompts, dev 16 et test 24, tirage fixe). Les autres (reasoning, math, coding, extraction)
  forment la partition `other`, rapportée à part : une seule chaîne de calcul ou une réponse courte ne se
  découpe pas en points.
- **Génération** (`run_sot.py`, un modèle à la fois sur le GPU) : `baseline` (chaque modèle répond seul,
  1024 jetons au plus), `outline` (le modèle désigné écrit les squelettes, lecture robuste, repli enregistré
  quand il est illisible : le modèle du squelette répond alors seul), `expand` (chaque pair développe ses
  points, 256 jetons au plus). Chaque appel enregistre ses jetons, son temps mesuré **sous service en lot**
  et sa raison d'arrêt.
- **Juge** (`judge_sot.py`) : un modèle plus gros compare la réponse parallèle à chaque réponse seule, dans
  **les deux ordres**, avec un verdict d'une lettre (A, B ou C pour l'égalité) imposé par une grammaire ; les
  probabilités des trois lettres sont lues aussi.
- **Analyse** (`analyze_sot.py`) : victoires, égalités et défaites après débiaisage de l'ordre (un verdict
  qui change avec l'ordre compte comme une égalité), IC bootstrap appariés sur les prompts. Comparaisons
  déclarées d'avance : le modèle du squelette seul, et le « meilleur modèle seul », celui contre lequel la
  réponse parallèle fait le moins bien sur dev. La **vitesse est modélisée** : jetons mesurés, vitesses
  mono-flux mesurées sur une RX 6650 XT (`speeds_consumer.json`), RTT de 50, 100 et 150 ms, et une variante
  où tous les pairs vont à 50 jetons/s.

```
bash colab/colab_phase0.sh up sot-1 A100      # étapes : réponses seules + squelettes, développements, juge
uv run python analyze_sot.py --tag sot1 --outline-model Qwen/Qwen3.5-4B --judge Qwen/Qwen3.8-27B --suffix _colab \
    --peers Qwen/Qwen3.5-4B google/gemma-4-E4B-it ibm-granite/granite-4.2-3b mistralai/Ministral-3-3B-Instruct-2512 \
            microsoft/Phi-4-mini-instruct HuggingFaceTB/SmolLM3-3B --baselines <les mêmes six modèles>
uv run python -m unittest discover -s tests   # tests sans modèle
```

## E11 : du code choisi en l'exécutant

La question : sur la génération de code, un essaim de petits modèles de familles différentes, qui choisit un
programme en l'**exécutant** (et non par un vote sur le texte), rivalise-t-il avec des modèles bien plus gros ?
Et l'accord fonctionnel (des programmes qui donnent les mêmes sorties sur des entrées générées) joue-t-il le
rôle de la dispersion des réponses en mathématiques ?

- **Données** : HumanEval+ (`evalplus/humanevalplus`, 164 problèmes, 82 dev et 82 test) et MBPP+
  (`evalplus/mbppplus`, 378 problèmes, 189 et 189), Apache-2.0 (HumanEval : MIT, MBPP : CC-BY-4.0), commits
  fixés, licence enregistrée dans l'identité des données, tirage fixe. Les **tests visibles** sont ceux du
  prompt : les exemples de la docstring pour HumanEval (plusieurs formats lus ; 12 problèmes sur 164 n'en ont
  aucun de lisible), les `assert` de MBPP. Les **tests cachés** (entrées de base + « plus » d'EvalPlus) ne
  servent qu'à noter. LiveCodeBench est écarté : licence « cc » sans variante, problèmes copiés de LeetCode,
  AtCoder et Codeforces, chargement par un script distant.
- **Génération** (`run_code.py`, chaque modèle via son llama-server sur le GPU, comme E4) : une solution gloutonne
  et 4 tirages à température 0,8 (graines fixes) par problème, 1024 jetons au plus, réflexion coupée. Le code
  est extrait du bloc ```` ```python ```` qui définit la fonction, puis nettoyé (exemples d'usage, `print`,
  `assert` et blocs `__main__` retirés ; une initialisation dont la solution se sert, `solver = Solver()`,
  est gardée).
- **Exécution** (`exec_code.py`, `essaim/sandbox.py`) : chaque programme distinct tourne dans un processus
  Python séparé, dans un répertoire temporaire neuf, trois fois : arguments des tests visibles, entrées
  supplémentaires (signatures des sorties), entrées des tests cachés. **Le processus du programme ne reçoit
  que des entrées** et renvoie ses sorties (encodage typé, ou empreinte au-delà d'un million de nœuds) ;
  les valeurs attendues, `assertion`, `ref_func` et les tests eux-mêmes restent dans un **processus noteur
  de confiance**, qui rejoue les tests du jeu de données sur ces sorties. Barrières : crochet d'audit (pas
  d'écriture hors du répertoire temporaire, aucune variante `dir_fd`, pas de descripteur de dossier hors de
  ce répertoire, lectures limitées à l'installation de Python, jamais `data/`, `results/` ni le cache
  Hugging Face, pas de processus, pas de socket, pas de `ctypes`), limites de temps, sortie bornée ; sous
  Linux aussi les rlimits (mémoire, CPU, taille de fichier), un délai par appel, `unshare --net` quand il
  marche, **Landlock** quand le noyau l'a (les mêmes règles de fichiers, imposées par le noyau) et
  `PR_SET_PDEATHSIG` (un enfant meurt avec son parent) ; sous Windows un objet job (mémoire, arbre tué d'un
  coup, aussi à la mort du parent) mais **pas** de délai par appel, d'espace réseau ni de Landlock : Windows ne
  sert qu'à l'essai de fumée. Sur SIGTERM ou Ctrl-C, `exec_code.py` tue tous les enfants en cours. La solution
  de référence passe par le même chemin : les tests visibles qu'elle échoue sont ignorés, et un problème
  qu'elle échoue est exclu (HumanEval/32 : l'oracle exporté est cassé).
- **Entrées supplémentaires** : les entrées visibles, puis 16 entrées tirées d'elles par une ou deux petites
  mutations qui gardent le type et le domaine apparent (signe des nombres, alphabet des chaînes, jamais de
  séquence vide), graine = identifiant du problème. Une entrée hors des préconditions sépare des programmes
  justes ; le taux a (deux programmes justes d'accord sur tout) le mesure.
- **Analyse** (`analyze_code.py`) : règles choisies sur dev, rapportées sur test : (a) meilleur modèle seul ;
  (b) vote sur le texte normalisé ; (c) tests visibles puis poids de la famille ; (d) regroupement fonctionnel
  (CodeT, AlphaCode) des candidats qui passent les tests visibles, plus gros groupe (nombre de programmes, de
  familles, ou somme des poids des familles) ; (e) cascade vers la plus grosse référence quand aucun candidat
  ne passe les tests visibles ou que le groupe gagnant a moins de m familles. Un accord exige au moins une
  sortie valide en commun : des programmes qui échouent sur toutes les entrées ne forment jamais un groupe.
  pass@1 sur les tests cachés, intervalle du score de Tango et tests d'équivalence non conditionnels
  avec p maximisés numériquement
  (TOST, ±2 points, `essaim/stats.py`) contre le meilleur modèle et chaque référence (seule, et avec la
  sélection fonctionnelle sur ses propres solutions, variante choisie sur dev), évolution offline k = 1..7 familles,
  programmes exécutés par
  problème, et le **taux de collision fonctionnelle** c (deux programmes faux de familles différentes, qui
  passent les tests visibles, d'accord sur toutes les entrées), l'analogue du c de la théorie du vote.

```
bash colab/colab_phase0.sh up code-1 A100     # étape 0 : 13 modèles écrivent le code ; étape 1 : exécution (CPU)
bash colab/colab_phase0.sh pull
uv run python analyze_code.py --suffix _colab
uv run python -m unittest tests.test_code     # tests sans modèle (bac à sable compris)
```

Essai de fumée sur le PC (5 problèmes, deux modèles, poids choisis sur les mêmes problèmes, fichiers `*_smoke*`
ignorés par git) : `run_code.py ... --suffix _smoke --benches humanevalplus --splits test --n 5`, puis
`exec_code.py --suffix _smoke ...` et `analyze_code.py --suffix _smoke --swarm <les deux modèles> --refs
--fit-split test`. `exec_code.py --reference-only` vérifie le banc seul (aucun modèle).

Les lectures WSL `colab/colab_phase0.sh status` et `diagnose` ont une borne murale externe de 90 secondes :
Un superviseur Python envoie `TERM` au groupe local, puis `KILL` au plus tard 5 secondes après. Le délai interne
`colab exec --timeout 60` concerne l'exécution distante et ne suffit pas à borner une connexion bloquée.
Les codes d'erreur remontent au surveillant ; la campagne distante reste active. Cette borne ne concerne pas `up`.
Le superviseur reste hors du groupe du CLI : il termine aussi les enfants qui ignorent `TERM`, sans dépendre
du comportement de l'utilitaire système `timeout` (uutils sous WSL sur ce PC).

### Campagne E11 finale et reproduction offline

La campagne finale est terminée. Les bruts et manifestes sont conservés sans réécriture ;
`e11_validation_colab.json` recense leurs SHA-256, les identités des treize modèles, les versions du harnais,
les partitions et chaque exclusion. `validate_code.py` exige tous les modèles attendus, références et
frères hors essaim compris, les clés (id, sample) exactes, les extractions et leurs hashes, toutes les références
de dataset et la couverture exec exacte, y compris sur les problèmes exclus. Les comptes cachés sont vérifiés
par lecture de l'AST des listes stockées, sans exécuter les sources dataset. Les caches épinglés
`data/{humanevalplus,mbppplus}_all_s0_v2.{jsonl,meta.json}` restent ignorés par Git. L'analyse canonique refuse
un cache absent ou modifié avant tout chargement : aucun téléchargement implicite, modèle ou code candidat exécuté.
Sur un nouveau clone, il faut fournir ces caches ou les préparer explicitement depuis les parquets épinglés
avec les chargeurs `essaim.data.humanevalplus` et `mbppplus`.

```powershell
cd phase0
uv sync --group paper
uv run python analyze_code.py --suffix _colab
uv run python -m unittest tests.test_e11_analysis -q
cd ..
uv run --project phase0 python paper/make_numbers.py
```

Le rapport est `results/e11_report_colab.md`, son résumé `e11_summary_colab.json`. Les choix de dev
(variantes, poids, meilleur pair, seuil de cascade et variantes propres des références) y sont enregistrés
avec leurs scores de dev ; aucun choix n'est optimisé sur test.
Le score `count` compte les candidats générés, y compris un programme identique proposé plusieurs fois ;
`families` et `wfamilies` comptent chaque famille une seule fois dans un groupe. La déduplication concerne
l'exécution, pas le vote des candidats.
Les modes de fumée et les sous-ensembles explicites restent disponibles et ne peuvent pas alimenter
l'article canonique. `paper/make_numbers.py`
exige les empreintes de toutes les sources et des modules locaux qui produisent le résumé ; les hashes de ces
modules attestent la reproduction locale, pas le checkout distant de Colab, dont le hash de source n'est pas
enregistré dans les anciens manifestes. Les identités GGUF, données et versions déclarées sont conservées.

Le bac déclaré par Colab est **sandbox-v3, unshare --net, audit hook Python, Landlock unavailable** :
pas les protections du nouveau LocalLinuxEnv. L'audit hook seul ne garantit pas le confinement de code natif
hostile. L'oracle historique tests-v4 ajoute l'assertion `exact_match` absente de l'export pour Mbpp/737,
787 et 794 ; son contrôle de racine HumanEval/32 échoue pour la référence (exclusion dev). Mbpp/255 est
exclu sur test après un délai de la référence (88/112 cas cachés réussis). Aucune réparation rétroactive des notes.
Tous les tests visibles échoués par la référence sont recensés séparément. Ces scores concernent le harnais
versionné décrit ici, pas un passage inchangé de la suite officielle EvalPlus.

Les compteurs `n_tokens`, `prompt_tokens`, `ms`, `finish` alimentent des détails par problème/mode,
des observations par modèle/split et des totaux connus/manquants. Pour une cascade rejouée offline, on attribue
la génération de référence seulement sur les problèmes qui l'appellent ; la campagne a néanmoins généré
toutes les réponses de référence. Les problèmes exclus restent dans les observations de génération.
Les figures matplotlib `e11_accuracy_{n_tokens,ms}_colab.{png,svg,pdf}` confrontent l'exactitude test aux jetons
et à la **somme des durées d'appels de génération par problème (proxy)**. Les requêtes sont parallèles
(`run_code.py`, défaut 16), avec contention GPU : cette somme n'est ni le temps mur de campagne ni la latence
d'un essaim/WAN. Même un maximum de durées serait une projection.
La concurrence ne figure pas dans les anciens manifestes ; le lanceur local demande 16 slots (8 pour le 27B)
et autorise jusqu'à trois jobs de modèles simultanés selon les ressources, sans journal de temps mur par problème.
Le ms d'exec inclut visible, extra et HIDDEN,
et n'est pas un coût mesuré de sélection en ligne. Énergie, coût réel total et temps mur restent inconnus ;
aucun scénario énergétique ni comparaison à calcul égal n'est inféré. Les jetons emploient les tokenizers des
producteurs. Les sorties sont déterministes, y compris les figures sans dates/IDs aléatoires.

Les tests statistiques sont exploratoires, sans correction de multiplicité ; pooling descriptif, dominé
en nombre par MBPP. Les courbes moyennes sur tous les sous-ensembles de familles décrivent des candidats
pré-générés ; elles ne constituent pas une mesure du passage à l'échelle distribué. Les bruts des résultats
négatifs, les collisions et les divergences entre programmes justes sont conservés.

## E12 : des bancs plus réels (GPQA Diamond, SciCode ; essai Terminal-Bench préparé, DeepSWE en conception)

La question du propriétaire : l'essaim d'E4 (7 petits modèles de 7 familles) obtient-il un score supérieur à 0 sur les
bancs qu'utilise Artificial Analysis, face aux mêmes références seules ? Une section par banc (source et version,
licence, harnais, infrastructure, Colab, rôle de l'essaim, coût, attentes réalistes) est dans le rapport de
faisabilité du dépôt privé.

- **GPQA Diamond** (198 QCM, protégé sur Hugging Face) : `essaim/gpqa.py`, `run_aa.py`. Aucune question dans le
  dépôt, aucun fichier `*gpqa*` dans l'export public ; rien n'est ajusté sur GPQA (poids et meilleur pair viennent
  du dev de MMLU-Pro d'E4).
- **SciCode** (65 problèmes de test, 288 sous-problèmes, chaînes de code, tests officiels) : `essaim/scicode.py`,
  `run_aa.py`, `exec_scicode.py` (bac à sable E11 ; le code du modèle et les cibles vivent dans un seul processus :
  VM jetable). `exec_scicode.py --oracle` vérifie uniquement les références de dev et arrête le plan si
  le banc est cassé. Pas de décision d'essaim sur du code : chaque pair seul, meilleur pair de dev, plafond
  « au moins un pair juste ».
- **Analyse** : `analyze_e12.py`, porte scientifique `validate_e12.py`, statistiques par grappes et comptabilité
  `e12_costs.py`. Les helpers GPQA historiques gardent Wilson et le test exact contre le hasard ; SciCode
  utilise un bootstrap déterministe de problèmes entiers, avec différences appariées exploratoires.
- **Terminal-Bench 4.0** : adaptateur Harbor réel et lanceur sur trois tâches publiques épinglées,
  puis comparaison single/vote/référence aux mêmes plafonds. Aucun score mesuré : Docker WSL et tunnel
  authentifié d'inférence indisponibles. Commandes, provenance et limites : [guide](../docs/11_terminal_bench.md).
  **Terminal-Bench et DeepSWE sont gelés par décision du propriétaire : aucun Docker.**
- **Pilote code sans Docker** : 34 exercices Python Aider Polyglot préparés et épinglés, trois premiers
  lexicographiques présélectionnés (affine-cipher, beer-song, book-store), boucle single/vote/cascade/référence seule,
  bac Linux WSL Landlock/namespaces/seccomp, vérificateur séparé et comptabilité jetons/temps/coût estimé.
  Provenance modèle/serveur structurée obligatoire ; solo direct Linux accepté. Aucune inférence ni mesure modèle
  dans ce jalon ; [reproduction et limites](../docs/12_code_pilot.md).
- **5c, dépôt Python réel** : trois printers SymPy historiques présélectionnés (11400, 11897, 12171),
  runtime uv Python 3.9.25 et mpmath 0.19 BSD, vrais dépôts temporaires sans Git, assertions de chaînes
  dans contrôleur séparé et mêmes boucle/budgets/comptabilité. Smoke WSL : trois bases rejetées,
  références 7 + 6 + 5 cas réussis ; aucun modèle mesuré, aucun score SWE-bench officiel.
  Diagnostic Requests LGPL conservé distinctement. [Commandes et limites](../docs/13_repo_pilot.md).

Deux fichiers viennent du propriétaire et ne sont jamais commités (`phase0/data/` est ignoré) : `gpqa_diamond.csv`
(accepter les conditions de GPQA sur Hugging Face) et `scicode_test_data.h5` (dossier Drive du README de SciCode ;
`uv run python -m essaim.scicode` affiche son SHA-256, à reporter dans `H5_SHA256`).

Dans `aa-1`, seul le HDF5 est obligatoire : sans CSV GPQA, l'archive WSL et les commandes de génération
exécutent SciCode seul. Un CSV présent mais invalide reste une erreur, même si un cache GPQA existe.
L'analyse choisit également SciCode seul par défaut si le CSV manque (`--benches` permet un choix explicite).
Le split test SciCode n'a aucune référence : ses 288 étapes notées (291 brutes moins 3 ignorées) passent les tests officiels,
sans exclusion par oracle. Les échecs de référence peuvent être exclus de dev pour choisir le meilleur pair.
Les trois étapes ignorées utilisent les fichiers officiels épinglés de `essaim/scicode_skipped/`.
Le cache SciCode, le prompt (`scicode-ours-v2`) et le harnais (`scicode-exec-v2`) changent de version :
les anciens résultats ne sont pas repris silencieusement et doivent être régénérés sous ce protocole.

```
bash colab/colab_phase0.sh up aa-1 A100       # oracle SciCode (CPU), 13 modèles (GPU), tests de SciCode (CPU)
bash colab/colab_phase0.sh pull
# Analyse seulement après livraison complète explicite du parent : voir le contrat ci-dessous.
uv run python -m unittest tests.test_aa tests.test_agent   # tests sans modèle ni données protégées
```

## Préparation E12 du 10 octobre 2026 : supervision et mesures

### Préparation de l'analyse scientifique (priorité 2, sans résultats)

La campagne réelle est supervisée exclusivement par le parent. Cette préparation teste du code sur des fixtures
synthétiques : elle ne lit aucun score partiel, n'exécute aucun modèle ni code de dataset et ne clôt pas E12.
Le succès de `verify_e12_pull.py` est une preuve de récupération, pas une validation scientifique.

La CLI refuse toute écriture sans une livraison complète du parent et son SHA-256 communiqué indépendamment.
Les chemins d'entrée désignent des archives immuables complètes, jamais le répertoire d'une campagne active.
La validation lit uniquement les deux JSONL officiels déjà disponibles, vérifie leurs blobs Git épinglés, puis
reconstruit l'identité du cache par parsing. Aucun téléchargement n'est déclenché. Elle assemble les programmes
pour vérifier leurs hashes, sans les exécuter. Les treize modèles, leurs dev/test exacts, l'oracle dev et tous les
compagnons sont obligatoires. Les identités des poids/moteurs doivent être attestées par le parent dans la livraison.
Les sources de génération/notation uploadées et récupérées sont liées à l'inventaire ; leurs modules critiques
doivent correspondre aux sources locales inchangées. Les analyseurs préparés ici ne modifient pas ces sources.
Ce contrôle inclut `essaim/llamacpp.py`, `essaim/common.py` et `pyproject.toml` : absence ou divergence refusée.

Contrat `delivery.json` (schema 1, document fourni après campagne ; aucun exemple de brut réel fabriqué) :

- `complete: true`, `plan: "aa-1"`, `benches: ["scicode"]` pour cette campagne sans GPQA.
- `files` : objet nom relatif → SHA-256 des 53 JSONL scientifiques, leurs 53 manifestes, les 26 compagnons
  timing, `e12_campaign_sources.json` et la copie uploadée `e12_campaign_sources.local.json`. Les preuves annexes
  peuvent aussi figurer dans l'inventaire ; chaque fichier inventorié doit être présent et identique.
- `identities` : exactement les treize IDs de `FAMILIES + EXTRA + REFS`, chacun avec `model`, `revision`
  (commit de 40 caractères), `gguf`, `weights` (`name`, `size`, `sha256`), `engine`, `backend`. Le parent vérifie
  ces identités contre les poids réellement préparés et les preuves de campagne ; une déclaration seule n'atteste
  pas les poids distants. La CLI impose aussi les noms GGUF du catalogue et le protocole canonique.
- `execution_environment` : `isolation`, `python`, `numpy`, `scipy`, identiques dans les 27 manifestes de notation.
  Ce sont les déclarations historiques du harnais, pas une attestation de sécurité ; SciCode partage candidat
  et cibles dans un processus académique. Aucun durcissement ni Docker ajouté.
- `attempts` : liste ordonnée de toutes les tentatives, objets `campaign_id` (32 caractères hexadécimaux),
  `interrupted` et `resumed` (booléens), `evidence` (identifiant relatif de preuve immuable → SHA-256).
  Le parent conserve et vérifie ces originaux ; seuls les identifiants relatifs et empreintes entrent au résumé,
  jamais des chemins personnels. Inclure la tentative `4a85834264f14e0d8f3333b368013476` et la campagne reprise
  `2b437e9608224949abdffdbe1b6b6f6a`, ainsi que les autres interruptions pertinentes. Ne pas inventer les preuves.

Commande prévue après livraison (substituer les chemins et le SHA fournis par le parent) :

```bash
uv run python analyze_e12.py --suffix _colab --benches scicode \
  --results-dir <archive-complete> --dataset-dir <sources-officielles-offline> \
  --delivery <delivery.json> --delivery-sha256 <sha256-parent> --output-dir <analyse-complete>
```

SciCode conserve dev 15 problèmes/50 étapes et test 65/288 étapes notées, avec les trois étapes fournies
officielles. Le harnais dev doit passer au moins 90 % avant toute exclusion dev ; aucune exclusion test nouvelle.
Le meilleur pair est choisi uniquement sur dev, départage en ordre `FAMILIES`. Les quatre références restent
séparées. Les scores sont les ratios exacts de sommes de réussites/étapes ; les IC percentiles à 95 % rééchantillonnent
les problèmes entiers (10 000 réplications, graine 12), avec la même sélection de grappes dans les différences
appariées. Ces IC descriptifs restent exploratoires, sans correction de multiplicité ; aucun verdict
d'équivalence ou de supériorité générale. Réutilisation historique et contamination restent des limites.
La couverture par étape peut combiner des chaînes incompatibles ; elle est séparée du compte de problèmes
où un même pair réussit sa chaîne entière. Aucun essaim SciCode ni sélecteur réalisable n'est mesuré.

La comptabilité conserve sommes observées, appels, compteurs `null`, erreurs, troncatures, durées de worker et
de batch séparément, par problème, modèle et split. Les sommes d'appels ne sont pas du temps mur. La concurrence
des workers est explicite ; la concurrence de modèles sur GPU n'est pas attribuée. Toute reprise/interruption
laisse les totaux inconnus, même si les JSON et réponses sont complets. Les relevés de solde Colab ne sont pas
une facture campagne. Grading, démarrage du serveur, téléchargement et préflight sont hors durées de génération.
Pour une tentative propre, le worker doit couvrir ses appels séquentiels ; le batch doit couvrir le plus long
worker, sans additionner les workers concurrents. Une contradiction conserve les observations
mais rend durée totale, scénario et point de figure inexploitables. Aucun ancien worker n'est comparé au nouveau
batch d'une reprise ; les totaux de reprise restent inconnus sans invalider les réponses scientifiques complètes.

Les figures Matplotlib SVG/PNG/PDF sont déterministes et limitées aux métriques interprétables : exactitude
contre durée de batch amortie par problème (débit, pas latence individuelle). Sur la campagne reprise, cette durée
totale est inconnue : aucun point n'est fabriqué. Un scénario facultatif `--scenario-watts W --scenario-eur-kwh T`
ajoute une estimation `W × secondes / 3 600 000 × T`, explicitement hypothétique : puissance d'un appareil entier
pendant le batch, sans attribution du GPU partagé. Ce n'est ni une mesure physique, ni une extrapolation A100
vers PC/WAN, ni un tarif API. Aucune figure de modèle réel n'est créée pendant cette préparation.

Les produits sont préparés dans un répertoire temporaire avant remplacement ; un refus scientifique conserve
les rapports, résumés et figures antérieurs. Après construction réussie de tous les produits, seules les anciennes
figures gérées `e12<suffix>_{time,cost}.{svg,png,pdf}` non reproduites sont retirées. Les autres fichiers et suffixes
sont préservés. Les helpers synthétiques existants gardent leur API, mais ne sont pas une voie de validation.

GPQA reste absent de la campagne réelle arrêtée. La disponibilité attestée par `delivery.benches` et les deux
manifestes de sources doit être identique et canonique : `["scicode"]` ou `["gpqa", "scicode"]`, jamais GPQA seul.
`--benches` filtre uniquement le rapport : sous-ensemble non vide des bancs disponibles, sans doublons.
Une même livraison combinée complète accepte `--benches gpqa`, `--benches scicode` et
`--benches gpqa scicode`, sans réécrire les manifestes, empreintes ou inventaires, ni retirer des artefacts d'entrée.
Sans `--benches`, un CSV local présent sélectionne les deux bancs,
sinon SciCode seul. Le CSV est contrôlé contre son blob et sa taille épinglés puis parsé directement, sans
téléchargement ni cache implicite. Un CSV invalide est refusé ; aucun CSV gated n'est téléchargé ici.
La validation scientifique porte toujours sur **toute** la livraison déclarée complète. Une référence absente,
un protocole altéré ou une provenance divergente du banc non affiché reste un refus et conserve les produits.
Un banc demandé absent est refusé. Le résumé distingue `provenance.available_benches` et `report_benches`.
GPQA exige les réponses, manifestes et timings liés des treize modèles, toutes références comprises, les sources
`essaim/gpqa.py`, `essaim/answers.py`, `analyze_e4.py`, et `e4_summary<suffix>.json` dans l'inventaire épinglé.
Les poids et probabilités par famille et le meilleur pair sont repris de MMLU-Pro dev E4 ; le meilleur respecte
le départage en ordre `FAMILIES`. Le parent atteste la provenance dev de ce résumé immuable, aucun réglage GPQA.
`--results-dir` s'applique aussi aux réponses GPQA et au résumé E4 ; `--output-dir` reçoit seulement les agrégats.
Pour valider une livraison combinée, même avec `--benches gpqa` seul, fournir `--dataset-dir` vers les deux JSONL
SciCode officiels hors ligne, `--results-dir` vers l'archive complète et `--delivery` / `--delivery-sha256` du parent.
Même avec `--benches scicode` seul, le CSV GPQA local admis reste nécessaire (`GPQA_DIAMOND_CSV` ou emplacement
local habituel), ainsi que le résumé E4 dev épinglé. Le filtre ne permet pas de contourner ces contrôles.
Aucune question, sortie de modèle ou trace brute GPQA n'est publiée. Les modes facultatifs sont testés uniquement
sur fixtures synthétiques et ne permettent pas d'analyser une livraison SciCode incomplète.
Article, macros, rapport scientifique et revue des données complètes attendent le parent.

```bash
uv run python -m unittest tests.test_e12_analysis tests.test_aa.Analysis tests.test_aa.EndToEnd -q
```

E11 est terminé ; cette préparation ne lance aucune VM et ne produit aucun résultat E12. Le parent doit encore
relire et réauditer le changement avant fusion, puis livrer les données et preuves complètes pour contrôle
scientifique. Cette préparation hors ligne ne rouvre pas la file arrêtée. GPQA reste facultatif ; aucun script
candidat ni test numérique SciCode ne doit tourner sur le PC.
Le protocole, les prompts, les modèles, les splits et l'oracle dev avant inférence GPU restent identiques.

Entrée versionnée, depuis la racine du dépôt fusionné sous WSL (Python 3, bash et CLI Colab déjà configurés) :

```bash
bash phase0/colab/chain4.sh --hours 12 >> tools/wsl/chain4.log 2>&1
```

Le parent crée auparavant `tools/wsl/chain4.log`. `tools/wsl/` reste ignoré : aucun code de supervision ne dépend
de ce dossier. Ne pas lancer la commande historique `up` en parallèle. Le verrou par défaut
`/tmp/myriad-phase0-campaign.lock` protège les chaînes de ce superviseur dans une même distribution WSL ; il ne
coordonne pas d'autres machines ni des commandes manuelles. Ne pas changer `--lock` pour contourner ce verrou.

Pour un lancement Windows durable, garder un processus Windows propriétaire de WSL. Le premier lancement
avec `nohup` détaché a disparu lorsque WSL n'avait plus de processus Windows. Après revue/audit/fusion par le
parent, depuis la racine du dépôt retenu dans PowerShell (commande documentée, pas exécutée ici) :

```powershell
$arguments = @('--cd', ('"' + (Get-Location).Path + '"'), '-e', 'bash', 'phase0/colab/chain4.sh', '--hours', '12')
Start-Process -FilePath wsl.exe -ArgumentList $arguments -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput C:/tmp/myriad-e12-transfer-launch.stdout.log `
    -RedirectStandardError C:/tmp/myriad-e12-transfer-launch.stderr.log
```

Le parent crée le dossier de journaux et choisit des noms ne recouvrant pas une preuve antérieure. Le PID rendu
est celui de `wsl.exe`, pas un accusé de succès de campagne ; vérifier les états finaux du superviseur.

### Transfert E12 après l'échec HTTP 500 du 10 octobre

La taille de l'archive est une cause plausible, non démontrée par le seul HTTP 500. Le wrapper conserve la
préparation scientifique existante (HDF5, sources, checkpoints et leurs empreintes), puis envoie l'archive
en fichiers de **8 Mio maximum** via la CLI officielle. Même une petite archive suit cette vérification.
La mémoire de découpage/reconstruction utilise des blocs de 1 Mio ; la CLI ne reçoit qu'un petit fichier à
la fois, car son upload charge tout le fichier et sa représentation base64. Un seul morceau local est conservé
à la fois en plus de l'archive et du staging existants. La limite est 8 192 morceaux, soit 64 Gio d'archive.
Aucune nouvelle authentification ni appel HTTP direct. Les erreurs DNS/connexion/délai/HTTP 408, 429 et 5xx
permettent trois tentatives au maximum, avec attente exponentielle (1, 2 s, plafonnée à 8 s).

Le préfixe `/content/dllm-transfer-<identité>` vient du SHA-256 de l'archive (fichiers plats). Une reprise du
même fichier, y compris dans un nouveau processus, retrouve donc les morceaux. Avant chaque upload, un script
borné vérifie la taille et le SHA-256 du fichier distant : un morceau déjà vérifié n'est pas retransmis.
Après une réponse d'upload perdue, cette lecture précède toute nouvelle écriture. Un morceau absent/corrompu
est remplacé puis revérifié ; une lecture incohérente ou sans accusé échoue fermement.
Même le troisième upload ambigu reçoit une dernière vérification avec défi frais, sans quatrième upload.
Le wrapper conserve aussi l'archive exacte et le bootstrap épinglé à côté du reçu, avec leurs SHA-256.
Après un envoi manuel interrompu, `resume-up <plan> <gpu initial>` reprend ces mêmes octets sans `new`,
uniquement si la session est encore la nôtre et qu'aucune campagne n'a démarré. Un cache modifié est refusé.
La libération confirmée supprime ce cache ; la chaîne supervisée garde son nettoyage final après épuisement
des tentatives. Le transfert ne change donc pas rétroactivement les sources d'une tentative reprise.
Le manifeste contient indices, noms exacts, tailles et SHA-256 des morceaux et de l'archive. Le petit script
versionné `colab/reconstruct.py`, lancé par le wrapper avec `colab exec --timeout 180`, exige aussi l'empreinte
du manifeste transmise dans son code. Il refuse les chemins inattendus, liens symboliques, indices désordonnés,
données manquantes, mélangées ou corrompues. Il recompose en streaming dans un fichier temporaire puis remplace
atomiquement `/content/dllm.tgz` seulement après validation complète. Il nettoie uniquement le manifeste et
les morceaux connus de cette tentative après validation ; modèles et résultats existants ne sont pas effacés.
Une tentative interrompue peut laisser des morceaux distants jusqu'à la libération de la VM ; aucun nettoyage
par glob des tentatives antérieures. Les temporaires locaux sont supprimés à la sortie normale ou signalée ;
SIGKILL/panne peut les laisser en place.

Un code CLI 0 ne prouve pas la réussite distante. Avant bootstrap, le wrapper exige exactement un accusé
`DLLM_TRANSFER_ACK` JSON avec statut, tentative, taille, SHA-256 d'archive et de manifeste et défi frais identiques ; accusé
absent, multiple, périmé ou incorrect ⇒ refus. Le verrou, les reçus d'allocation et les délais mur du superviseur
restent obligatoires. Après un accusé perdu, la reconstruction peut revérifier l'archive publiée sans renvoyer
les morceaux supprimés. Le défi frais évite un accusé périmé ; ce n'est pas une attestation contre un candidat hostile.
Une publication interrompue pendant la suppression des morceaux est aussi reprise : taille et SHA de l'archive
finale sont revérifiés, puis le nettoyage du manifeste épinglé et de ses seuls morceaux devient idempotent.

Les sorties CLI sont capturées dans des fichiers temporaires anonymes, assainies avant stdout/stderr, puis
supprimées. Toutes les requêtes d'URL sont masquées, ainsi que les paramètres d'accès courants et Bearer hors URL.
Opération, HTTP et code de retour restent disponibles ; les lignes dépassant 64 Kio sont omises entièrement.
Aucun argument ni environnement n'est journalisé. Le journal brut du premier essai reste ignoré et ne doit pas
être lu/copié. Smoke synthétique hors ligne sous WSL, sans CLI réelle, modèle ni exécution de dataset :

```bash
cd phase0
uv run python -m unittest tests.test_colab_transfer tests.test_campaign tests.test_colab_timeout tests.test_colab_reliability -q
```

Toutes les opérations passent par `bash phase0/colab/colab_phase0.sh`. La chaîne refuse une session `phase0`
existante et une campagne déjà présente, sans l'arrêter, l'écraser ni adopter son PID. Elle n'appelle `up aa-1 <gpu>`
qu'une fois, sans réessayer une allocation ambiguë. Les lectures `sessions-json` et `snapshot` conservent les erreurs ;
le format texte de la CLI est adapté strictement, toute sortie inconnue ou vide est une erreur. Une évolution de
ce format nécessite une adaptation revue, jamais un repli par recherche de texte. Les snapshots exposent des états filtrés
et la vie du lanceur, sans messages/logs bruts ni URL signées ; les fichiers originaux restent intacts.

Le délai de campagne inclut l'allocation/bootstrap et vaut 12 h par défaut (`--hours`). Chaque invocation a sa borne
mur locale et un groupe de processus TERM/KILL, même en cas de connexion bloquée : `--up-seconds 2400`,
`--read-seconds 120`, `--transfer-seconds 900`, `--grace 5`. Trois lectures successives au maximum sont tolérées
(`--read-failures`), espacées de `--poll-seconds 60`. Les récupérations périodiques ont lieu toutes les 600 s
(`--pull-seconds`), après initialisation des jobs ; les téléchargements initiaux ne déclenchent pas de pull.
Un échec du pull périodique déclenche au plus `--read-failures` tentatives, puis laisse la campagne saine continuer
avec une récupération en attente. Les observations conservent la deadline globale et l'appartenance ; le pull
final reste obligatoire et est réessayé dans son budget réservé. Chaque appel officiel, manuel ou supervisé,
a sa propre borne via `safe_cli.py` : 120 s par défaut, ou exécution distante + 30 s si plus longue.
`DLLM_CLI_SECONDS` et `DLLM_CLI_GRACE` permettent une borne explicite. Les lectures avant allocation et le snapshot
avant upload sont donc bornés séparément du budget `up`. Les superviseurs transmettent TERM avant KILL ; sur
interruption, le wrapper termine immédiatement son groupe CLI, y compris ses enfants résistants à TERM.
Un parent CLI sorti sans son enfant ne laisse pas cet enfant vivant. SIGINT/SIGTERM déclenchent le nettoyage ; les signaux répétés
sont ignorés pendant ce nettoyage borné à 1800 s (`--cleanup-seconds`), dont une réserve permet de tenter `down`
même après expiration du transfert. SIGKILL, panne du PC ou disparition de WSL ne permettent pas de garantir le nettoyage.

Le reçu persiste dans `~/.local/state/myriad-colab/ownership.json`, hors magasin CLI ; chemin configurable avec
`DLLM_CAMPAIGN_RECEIPT` ou `--receipt`. Publication atomique/fsync, empreinte opaque de session, matériel,
campagne, identité d'allocation, plan, budget et relevé initial, aucune clé ni URL. Le verrou commun du reçu couvre
`up` et `resume-up` jusqu'à la fin du bootstrap, ainsi que `down`. Les helpers héritent de ce même verrou sans
le réacquérir sur un autre descripteur ; la supervision conserve son verrou distinct, sans attente circulaire.
Les concurrents sont refusés (73). L'identité initiale est attachée à l'opération : un reçu réécrit ne permet pas
à une ancienne reprise d'adopter une nouvelle allocation. La chaîne vérifie statut, campagne et empreinte avant
récupération ; `down` relit l'empreinte avant chaque arrêt.
Les appels de transfert/bootstrap vérifient aussi cette appartenance avant chaque lecture/écriture distante.
Une campagne étrangère reste bloquée dans le reçu au lieu de supprimer la preuve.
Le superviseur refuse aussi son nettoyage si le reçu appartient à une autre campagne, même avant sa première
épingle ; `down` vérifie lui-même la campagne attendue dès l'entrée, avant toute lecture CLI. La reprise volontaire
`--cleanup-only` adopte explicitement le reçu choisi sous le verrou commun, puis transmet ce même verrou et
l'identité épinglée au wrapper jusqu'à la fin du nettoyage. Un reçu remplacé entre adoption et `down` est refusé.
Ce refus interdit aussi tout pull si le snapshot du nettoyage est illisible. Les métadonnées E12 téléchargées
ne sont publiées localement qu'après vérification de leur identifiant de campagne. Les checkpoints envoyés et
récupérés sont filtrés par les familles du plan enregistré, jamais par le chemin du reçu : `code-1` conserve
`code_*` et `codeexec_*` sans provenance E12 ; `aa-1` conserve provenance et vérification finale complète.

`down` réessaie lecture et arrêt jusqu'à absence explicitement confirmée, même après erreur DNS/connexion,
arrêt ambigu ou lecture perdue après un arrêt réussi. Budget total : 1800 s par défaut (`DLLM_CLEANUP_SECONDS`,
`--cleanup-seconds` sous supervision), avec borne de chaque appel et attentes exponentielles. À expiration,
retour 124 et **CLEANUP PENDING, VM peut encore être facturée**, reçu conservé, nouvelle allocation interdite.
Reprendre sans allocation :

```bash
bash colab/chain4.sh --cleanup-only --receipt /chemin/ownership.json
# ou, avec le même DLLM_CAMPAIGN_RECEIPT :
bash colab/colab_phase0.sh down
```

Une session remplacée n'est jamais arrêtée. Avant `new`, un reçu `allocation-unconfirmed` est écrit ; un résultat
ambigu garde ce reçu. Après un `new` confirmé, seule la lecture d'identité est réessayée trois fois au maximum,
avec bornes CLI et backoff (1 puis 2 s par défaut) ; une panne durable conserve `allocation_returned=true` et
la preuve non confirmée. Le CLI installé ne fournit
aucun endpoint dans l'accusé de `new` : une session apparue ultérieurement ne prouve pas la propriété.
Dans ce cas, le nettoyage peut confirmer une absence, mais ne peut pas arrêter une session présente :
réconciliation indépendante indispensable par l'opérateur, sans adoption automatique ni nouvelle allocation.
SIGKILL, panne du PC ou disparition de WSL peuvent empêcher l'arrêt ; la preuve persistante permet la reprise.
L'absence explicitement confirmée d'une session possédée est distincte d'une identité distante remplacée.
Si la session expire après `up` ou pendant l'observation, la campagne reste en échec et la récupération n'est
pas présentée comme complète, mais `down` peut confirmer l'absence et terminer la comptabilité sans arrêt
superflu. Une identité différente conserve le refus des récupérations et du nettoyage ; aucune adoption implicite.

Avant allocation, `usage`/`usage-json` expose via le wrapper un JSON strict du solde, débit horaire du compte
et nombre d'allocations. Un solde vide/inconnu/insuffisant ou une autre allocation facturée interdit `new`.
Budget obligatoire : `--budget-units` pour la chaîne ou `DLLM_BUDGET_UNITS` pour le mode manuel ; l'opérateur
doit le dimensionner pour la durée et le tarif prévus. Aucun tarif futur n'est inventé. Les relevés avant
allocation et après absence confirmée sont inscrits dans `<reçu>.usage.jsonl` et dans le journal.
La diminution du solde du compte exige deux relevés lisibles ; recharge/autres usages peuvent la modifier.
`campaign_consumption_units` reste `null`, sans prétendre une facture de campagne. Un relevé final impossible
est enregistré explicitement avec valeurs inconnues, sans remettre en cause l'absence confirmée.
Le reçu reste alors `released/accounting-pending`, la commande retourne 124 et aucune nouvelle allocation n'est
permise. Le relevé final est réessayé trois fois au maximum dans le budget de nettoyage. `down` ou
`--cleanup-only` reprend uniquement `usage`, sans sessions ni stop supplémentaires ; le reçu et le cache ne sont
supprimés qu'après un relevé lisible. Le journal distingue cette comptabilité en attente d'une VM encore facturable.
Exemple de préparation, qui n'autorise pas une relance E12 :

```bash
bash colab/chain4.sh --gpu auto --hours 10 --budget-units 60
```

`--gpu auto` et `up <plan> auto` préfèrent L4 si les réservations du plan tiennent, puis A100/H100.
Le choix peut être explicite (`L4`, `A100`, `H100`, `T4`). La plus grande réservation modèle/contexte/slots,
avec la marge existante de 2 Gio, est vérifiée avant allocation, puis la mémoire libre réelle est mesurée par
`nvidia-smi` avant upload/bootstrap. `aa-1` garde Qwen 27B et sa concurrence : 37 + 2 = **39 Gio libres**,
donc L4 est refusé. `smoke` exige 6 Gio et peut choisir L4. Le scheduler garde ses réservations pour limiter
la concurrence ; aucun changement de modèle, quantification, contexte ou protocole scientifique.

Succès exige les quinze jobs exacts réussis, le téléchargement final complet de cette invocation (générations
SciCode et notes dev/test des treize modèles, oracle dev, métadonnées, observations de temps et provenance), puis
`down` réussi et absence explicitement confirmée. Un JSON absent/incomplet, une erreur, une annulation ou un simple
retour 0 du bootstrap ne constitue jamais un succès. Si la récupération finale échoue, le journal indique que les
résultats ne sont pas confirmés récupérés et le code est non nul ; l'erreur initiale est conservée si elle existe.
Ce contrôle de récupération ne remplace pas l'analyse scientifique et ses contrôles de provenance/dev/test.

Avant archivage/upload, `e12_campaign_sources.json` empreinte les fichiers sources effectivement présents dans le
répertoire de préparation (chemins relatifs, tailles, SHA-256 ; données/checkpoints/résultats exclus).
La copie locale `results/e12_campaign_sources.local.json` doit être identique au manifeste récupéré sur la VM.
Les seize sources figées d'E11 et ses bruts ne changent pas.

`run_aa.py` inscrit `wall_s` par appel dans les réponses et un journal compagnon `*.jsonl.timing.jsonl` : appels
(jetons retournés ou `null`, type d'erreur), problèmes SciCode (temps observé, achèvement, étapes reprises), batch
(temps global observé et nombre de nouvelles réponses). Le compagnon est initialisé avant la première inférence,
avec une empreinte du manifeste et le nom du résultat.
La version `e12-wall-v2` du manifeste refuse la reprise
silencieuse d'anciens fichiers non instrumentés. Les observations de reprise ne reconstituent pas le temps passé
avant une interruption ; les checkpoints transmettent les journaux de temps disponibles. Le temps d'un problème
est mesuré dans son worker, après l'attente dans la file, et comprend construction du prompt, appels séquentiels,
extraction et écriture. Le batch mesure le temps global de génération, hors démarrage du serveur et chargement
des données ; **il ne faut pas sommer les durées des problèmes/appels concurrents pour le remplacer**.
`ms` reste la mesure interne historique du client ; pour un appel échoué elle vaut `null`, ainsi que les compteurs
de jetons inconnus. Un batch interrompu ne fournit pas de durée globale complète ; les observations déjà écrites
restent partielles. Ces temps locaux VM incluent l'attente/ordonnancement serveur, pas une mesure distribuée WAN.
Le pull manuel récupère aussi les compagnons des résultats instrumentés, sans reçu de campagne. Les résultats
historiques sans champ `measurements` restent récupérables sans compagnon ; aucun total de temps n’en découle.
Une dernière ligne JSON complète sans retour ligne est normalisée. Un fragment final ou une corruption au milieu
refuse la reprise avant inférence et la publication : preuve brute `*.rejected-<sha256>` conservée, mesures
incomplètes et jetons inconnus. La publication est atomique et refuse un préfixe ancien/divergent. Les essais
répétés restent des observations distinctes ; le journal est vérifié une fois par reprise, pas à chaque ajout.
Une interruption sans trace finale de problème/batch interdit de prétendre à un temps global complet.
Énergie et coût restent inconnus sans hypothèses explicites de puissance et de tarif.

Une interruption peut consommer des jetons avant que la première observation de l'appel soit écrite. Après une
interruption ou une reprise, la validité JSON et la couverture des réponses ne suffisent donc pas à établir un
coût total complet : conserver l'historique des campagnes et rapporter le total comme inconnu, sauf preuve
complète de toutes les tentatives. La somme observée reste distincte de ce total ; les compteurs `null` ne valent
jamais zéro. Ces limites s'appliquent aussi si les réponses scientifiques sont finalement toutes récupérées.

## Sur le Mac

Avec llama.cpp de Homebrew : `LLAMA_SERVER=/opt/homebrew/bin/llama-server uv run python run_mc.py --model
google/gemma-4-E2B-it --gguf ../models/gemma-4-E2B-it-Q8_0.gguf --suffix _gpu --split dev`, puis `--split
test`. Pour l'expérience 2, le pair écoute sur le réseau local seulement avec un jeton :
`ESSAIM_TOKEN=<secret> ... -m essaim.peer ... --host 0.0.0.0 --port 8104`.

## Audits

Chaque modification de code a été relue par une revue automatique (Codex), chaque remarque vérifiée à la
main. Les résultats de la première série (avant corrections) ne sont pas publiés.
