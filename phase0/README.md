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
- **Analyse** : `analyze_e12.py` (Wilson, test exact contre le hasard, tests appariés exacts comme E4).
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
uv run python analyze_e12.py --suffix _colab
uv run python -m unittest tests.test_aa tests.test_agent   # tests sans modèle ni données protégées
```

## Préparation E12 du 10 octobre 2026 : supervision et mesures

E11 est terminé ; cette préparation ne lance aucune VM et ne produit aucun résultat E12. Le parent doit encore
relire, auditer et fusionner le changement, fournir le HDF5 épinglé dans `phase0/data/` et vérifier la disponibilité
de la session. GPQA reste facultatif ; aucun script candidat ni test numérique SciCode ne doit tourner sur le PC.
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
Aucune nouvelle authentification, aucun appel HTTP direct, aucune nouvelle tentative automatique d'upload.

Chaque tentative utilise un préfixe aléatoire sûr sous `/content/dllm-transfer-<tentative>` (fichiers plats).
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
`DLLM_TRANSFER_ACK` JSON avec statut, tentative, taille, SHA-256 d'archive et de manifeste identiques ; accusé
absent, multiple, périmé ou incorrect ⇒ refus. Le verrou, les reçus d'allocation et les délais mur du superviseur
restent inchangés ; upload/reconstruction partagent son groupe de processus, sans groupe imbriqué.

Les sorties CLI sont capturées dans des fichiers temporaires anonymes, assainies avant stdout/stderr, puis
supprimées. Toutes les requêtes d'URL sont masquées, ainsi que les paramètres d'accès courants et Bearer hors URL.
Opération, HTTP et code de retour restent disponibles ; les lignes dépassant 64 Kio sont omises entièrement.
Aucun argument ni environnement n'est journalisé. Le journal brut du premier essai reste ignoré et ne doit pas
être lu/copié. Smoke synthétique hors ligne sous WSL, sans CLI réelle, modèle ni exécution de dataset :

```bash
cd phase0
uv run python -m unittest tests.test_colab_transfer tests.test_campaign tests.test_colab_timeout -q
```

Toutes les opérations passent par `bash phase0/colab/colab_phase0.sh`. La chaîne refuse une session `phase0`
existante et une campagne déjà présente, sans l'arrêter, l'écraser ni adopter son PID. Elle n'appelle `up aa-1 A100`
qu'une fois. Les lectures structurées `sessions-json` et `snapshot` conservent les erreurs de connexion et de JSON ;
le format texte de la CLI est adapté strictement, toute sortie inconnue ou vide est une erreur. Une évolution de
ce format nécessite une adaptation revue, jamais un repli par recherche de texte. Les snapshots exposent des états filtrés
et la vie du lanceur, sans messages/logs bruts ni URL signées ; les fichiers originaux restent intacts.

Le délai de campagne inclut l'allocation/bootstrap et vaut 12 h par défaut (`--hours`). Chaque invocation a sa borne
mur locale et un groupe de processus TERM/KILL, même en cas de connexion bloquée : `--up-seconds 2400`,
`--read-seconds 120`, `--transfer-seconds 900`, `--grace 5`. Trois lectures successives au maximum sont tolérées
(`--read-failures`), espacées de `--poll-seconds 60`. Les récupérations périodiques ont lieu toutes les 600 s
(`--pull-seconds`), après initialisation des jobs ; les téléchargements initiaux ne déclenchent pas de pull.
Leurs erreurs arrêtent la campagne. Sous supervision, le snapshot partage le groupe borné extérieur ; les snapshots
manuels conservent leur propre borne TERM/KILL. SIGINT/SIGTERM déclenchent le nettoyage ; les signaux répétés
sont ignorés pendant ce nettoyage borné à 1800 s (`--cleanup-seconds`), dont une réserve permet de tenter `down`
même après expiration du transfert. SIGKILL, panne du PC ou disparition de WSL ne permettent pas de garantir le nettoyage.

Le reçu d'allocation contient seulement une empreinte opaque de session et le matériel ; aucune clé ni URL n'est
enregistrée. La chaîne vérifie cette empreinte avant récupération et libération. Une allocation refusée ou un
bootstrap refusé n'autorise aucun `down` ; un identifiant aléatoire de campagne transmis dans l'archive et inscrit
par le bootstrap empêche aussi d'adopter l'état réussi d'une autre campagne. Si la confirmation de propriété échoue
après `new`, ou si la session
disparaît/change, la chaîne échoue et ne risque pas d'arrêter une session étrangère : vérification manuelle nécessaire.
Une panne réseau durable peut donc empêcher de confirmer la libération ; le journal le dit explicitement.

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
