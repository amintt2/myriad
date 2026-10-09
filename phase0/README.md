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
- **Génération** (`run_code.py`, un modèle à la fois sur le GPU, llama-server comme E4) : une solution gloutonne
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
  pass@1 sur les tests cachés, intervalle du score de Tango et tests d'équivalence exacts non conditionnels
  (TOST, ±2 points, `essaim/stats.py`) contre le meilleur modèle et chaque référence (seule, et avec la
  même sélection sur ses propres solutions), passage à l'échelle k = 1..7 familles, programmes exécutés par
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

## Sur le Mac

Avec llama.cpp de Homebrew : `LLAMA_SERVER=/opt/homebrew/bin/llama-server uv run python run_mc.py --model
google/gemma-4-E2B-it --gguf ../models/gemma-4-E2B-it-Q8_0.gguf --suffix _gpu --split dev`, puis `--split
test`. Pour l'expérience 2, le pair écoute sur le réseau local seulement avec un jeton :
`ESSAIM_TOKEN=<secret> ... -m essaim.peer ... --host 0.0.0.0 --port 8104`.

## Audits

Chaque modification de code a été relue par une revue automatique (Codex), chaque remarque vérifiée à la
main. Les résultats de la première série (avant corrections) ne sont pas publiés.
