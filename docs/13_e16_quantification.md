# E16 : un plus gros modèle quantifié par pair, ou un petit modèle en Q8_0 ?

Préparé le 2026-10-13 (branche `e16-quant`). Rien n'a été lancé sur Colab : ce document est le protocole
**préenregistré**, le plan Colab (`e16-1`) et le script d'analyse (`phase0/analyze_e16.py`) sont prêts.

## 1. Question

Un PC de grand public a environ 8 Go de VRAM (le PC du projet : RX 6650 XT, 8 Go). E4 donne à chaque pair un petit
modèle en Q8_0 (1,5 à 7,5 Gio de poids ; Qwen3.5-4B : 4,2 Gio). À mémoire par pair à peu près égale, un modèle de 9 à
14 milliards de paramètres en 3 à 4,5 bits par poids rentre dans le même budget. Trois questions :

1. **Un pair seul** : le meilleur gros modèle quantifié du budget fait-il mieux que le meilleur petit modèle en Q8_0 ?
2. **Un essaim** : un essaim de k gros pairs quantifiés fait-il mieux qu'un essaim de k (ou de 6 à 7) petits pairs Q8_0 ?
   Le gain du vote (E4, E11) vient de la diversité des familles ; des pairs plus forts mais moins nombreux la
   conservent-ils ?
3. **Prix** : que coûtent la quantification (précision perdue contre le Q8_0 du même modèle), la vitesse (jetons/s) et
   la mémoire ?

Le budget de comparaison est un fichier de poids d'au plus **6,5 Gio**. Il laisse au moins 1,5 Gio d'une carte de 8 Go
pour le cache KV et les tampons de calcul. Le cache KV en f16, calculé à partir des `config.json` (lus le 2026-10-13) :
Ministral-3-14B (40 couches, 8 têtes KV, dimension 128) : 160 Kio par jeton, soit 0,63 Gio à 4 k et 1,25 Gio à 8 k ;
Qwen3.5-9B (seulement 8 couches d'attention complète sur 32, 4 têtes KV, dimension 256) : 32 Kio par jeton, 0,25 Gio à
8 k ; Gemma-4-12B (fenêtre glissante de 1 024 sur 40 couches, 8 couches globales à une tête KV) : environ 0,45 Gio à 8 k.
Ministral-3-14B en Q3_K_M (6,22 Gio) est donc le cas limite : il tient à 4 k de contexte mais pas à 8 k sur une carte qui
sert aussi à l'affichage. La mémoire réellement utilisée est mesurée par llama.cpp au moment du lancement sur le PC (à faire,
hors de cette tâche).

## 2. État de la littérature (recherche du 2026-10-13)

Je n'ai pas lu tous ces articles en entier ; ce qui suit vient des résumés et pages consultés, avec les limites dites.

- **Dettmers et Zettlemoyer, « The case for 4-bit precision: k-bit Inference Scaling Laws »**, ICML 2023
  ([arXiv 2212.09720](https://arxiv.org/abs/2212.09720)). Plus de 35 000 expériences (BLOOM, OPT, Pythia, GPT-2, de 19 M à
  176 G de paramètres, 3 à 8 bits) : à nombre total de bits fixé, 4 bits donne presque toujours la meilleure précision
  zéro-coup ; à 3 bits la tendance s'inverse. Les gains viennent d'un petit bloc de quantification et du type de
  donnée (flottant ou quantile plutôt qu'entier). *Limite* : modèles de 2022, tâches zéro-coup, pas de raisonnement en chaîne.
  C'est l'argument pour « plus gros, moins de bits » jusqu'à 4 bits, et la raison de tester aussi des fichiers à 4 à 4,6 bits
  (IQ4_XS, Q4_0 QAT) à côté de ceux à 3 bits.
- **Liu et coll., « Quantization Hurts Reasoning? »** ([arXiv 2504.04823](https://arxiv.org/abs/2504.04823), 2025). Modèles de
  raisonnement de 1,5 G à 70 G : W8A8 et W4A16 sont sans perte mesurable ; en dessous, le risque est réel et dépend de la
  taille du modèle, de son origine (les modèles entraînés par apprentissage par renforcement souffrent plus) et de la
  difficulté de la tâche (AIME, GPQA, LiveCodeBench plus touchés que MATH-500). *Limite* : ce sont des quantifications
  GPTQ/AWQ, pas des quantifications llama.cpp, et des modèles qui réfléchissent longuement ; nos modèles répondent en
  mode instruct sans réflexion. Raison de mesurer le code (E11), où la perte attendue est la plus forte.
- **Banc communautaire `yrougy/llm-quant-bench`** (GitHub, `LEGACY.md` ; lm-evaluation-harness avec llama.cpp,
  tailles d'échantillon non indiquées, barres d'erreur de ±3 à ±4 points sur HumanEval) : sur Qwen3.5-4B, IQ3_XXS perd
  environ 11 points sur HumanEval contre Q4_K_M (48,8 contre 59,8) ; sur Qwen3.5-9B environ 5,5 points (60,4 contre 65,9)
  et rien de net sur GSM8K (86,7 contre 87,1) ; sur le 27B, 3 points. Ordre de grandeur utile, **pas une source
  citable** : non évalué par les pairs, un seul passage.
- **llama.cpp**. Les types `IQ*` utilisent une matrice d'importance (imatrix) calculée sur un corpus de calibration ; les
  noms et les tailles nominales (IQ3_XXS 3,06 ; IQ4_XS 4,25 bits par poids) sont dans la documentation de `llama-quantize`.
  Toutes les quantifications retenues ici sont des fichiers imatrix de bartowski (corpus de calibration publié dans
  chaque fiche), sauf le fichier QAT de Google. Le type `Q4_0` de Google est obtenu par **entraînement conscient de la
  quantification** (QAT), pas par quantification après coup : c'est le seul fichier de la liste dont l'éditeur annonce une
  qualité proche de bfloat16 ([fiche](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)).
- **SLQ, « Statistically-Lossless Quantization of LLMs »** (Helcig, Kurtic, Alistarh,
  [arXiv 2605.02404](https://arxiv.org/abs/2605.02404), code [IST-DASLab/SLQ](https://github.com/IST-DASLab/SLQ)). Quantification par
  couche, non uniforme, asymétrique, avec recherche de largeur de bits ; elle annonce une compression « sans perte sur les
  tâches » dès 3,3 bits par paramètre (selon le modèle), et des accélérations de 1,7 à 3,7 fois par rapport à FP16 avec ses
  propres noyaux. **Elle n'est pas disponible dans llama.cpp** : la page de l'article ne mentionne ni llama.cpp ni GGUF, et je
  n'ai trouvé aucun portage. Elle n'est donc pas candidate pour ce prototype (llama.cpp b11505 épinglé sur la VM et sur le
  PC). À réexaminer si un format GGUF ou un noyau Vulkan existe un jour.

Ce que la littérature prédit, sans le garantir : la perte due à la quantification baisse quand le modèle grossit ; 4 bits
est le meilleur compromis à nombre de bits fixé, 3 bits est déjà du mauvais côté ; la perte est plus forte sur le code et
le raisonnement long que sur les QCM. E16 le mesure sur nos modèles, nos benchmarks et nos règles de vote.

## 3. Candidats

Tous en poids ouverts, **licence Apache-2.0** (Qwen3.5-9B, Gemma-4-12B, Ministral-3-14B-Instruct-2512 : fiches des dépôts
GGUF et des modèles d'origine, relevées par l'API Hugging Face le 2026-10-13), au format safetensors à l'origine, aucun
`trust_remote_code`. Les fichiers sont identifiés par leur SHA-256 publié (`phase0/e16_quant.py` les épingle ;
`fastdl.py` refuse un fichier dont l'empreinte diffère, `prepare_e16` revérifie, l'analyse revérifie dans le manifeste).
« bits/poids » = taille du fichier × 8 / paramètres du modèle (`essaim/models.py`) : une mesure du fichier, plongements
et tenseurs gardés en haute précision compris, non le nombre nominal du type de quantification.

| modèle | type | dépôt | révision du dépôt | fichier | taille | bits/poids | réservation GPU (Gio) | SHA-256 (début) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | IQ3_XXS | bartowski/Qwen_Qwen3.5-9B-GGUF | `182be2fd6c7b` | Qwen_Qwen3.5-9B-IQ3_XXS.gguf | 4,53 Go / 4,22 Gio | 4,03 | 14 | `478d619ab6f2` |
| Qwen3.5-9B | IQ3_M | idem | idem | Qwen_Qwen3.5-9B-IQ3_M.gguf | 4,98 Go / 4,64 Gio | 4,43 | 14 | `cbeadcf49dea` |
| Qwen3.5-9B | Q3_K_M | idem | idem | Qwen_Qwen3.5-9B-Q3_K_M.gguf | 5,18 Go / 4,82 Gio | 4,60 | 15 | `24fccd8416fc` |
| Qwen3.5-9B | IQ4_XS | idem | idem | Qwen_Qwen3.5-9B-IQ4_XS.gguf | 5,50 Go / 5,12 Gio | 4,89 | 15 | `7d977cc96c2e` |
| Gemma-4-12B-it | IQ3_XXS | bartowski/gemma-4-12B-it-GGUF | `2ae7d41be21c` | gemma-4-12B-it-IQ3_XXS.gguf | 5,15 Go / 4,79 Gio | 3,43 | 16 | `3a85fa01d72d` |
| Gemma-4-12B-it | IQ3_M | idem | idem | gemma-4-12B-it-IQ3_M.gguf | 5,97 Go / 5,56 Gio | 3,98 | 17 | `def80efba2f9` |
| Gemma-4-12B-it | Q3_K_M | idem | idem | gemma-4-12B-it-Q3_K_M.gguf | 6,30 Go / 5,87 Gio | 4,20 | 18 | `b3a5c0834941` |
| Gemma-4-12B-it | IQ4_XS | idem | idem | gemma-4-12B-it-IQ4_XS.gguf | 6,78 Go / 6,32 Gio | 4,52 | 18 | `60e817ec84aa` |
| Gemma-4-12B-it | QAT Q4_0 | google/gemma-4-12B-it-qat-q4_0-gguf | `29d097773436` | gemma-4-12b-it-qat-q4_0.gguf | 6,98 Go / 6,50 Gio | 4,65 | 18 | `93567e57a8fe` |
| Ministral-3-14B | IQ3_XXS | bartowski/mistralai_Ministral-3-14B-Instruct-2512-GGUF | `9289ba93995b` | mistralai_Ministral-3-14B-Instruct-2512-IQ3_XXS.gguf | 5,43 Go / 5,05 Gio | 3,10 | 17 | `c048183d069e` |
| Ministral-3-14B | IQ3_M | idem | idem | …-IQ3_M.gguf | 6,27 Go / 5,84 Gio | 3,58 | 18 | `0784b672b6a6` |
| Ministral-3-14B | Q3_K_M | idem | idem | …-Q3_K_M.gguf | 6,68 Go / 6,22 Gio | 3,82 | 18 | `b440f656c6a3` |

**Écartés** : Ministral-3-14B IQ4_XS (6,90 Gio, hors budget ; listé, jamais lancé) ; tous les types à 2 bits (IQ2_*),
trop dégradés pour être utiles d'après la littérature ci-dessus ; les variantes « UD » d'Unsloth (recette propriétaire
différente, qui compliquerait l'interprétation : une seule recette par famille, celle de bartowski) ; les modèles de
raisonnement fins-tunés de la communauté.

**Références Q8_0 (E4, déjà mesurées, suffixe `_colab`)** : petits pairs Qwen3.5-4B (4,17 Gio), Granite-4.2-3B (3,63), Phi-4-mini
(3,80), Ministral-3-3B (3,40), SmolLM3-3B (3,05), OLMo-2-1B (1,47), Gemma-4-E4B (7,48 Gio, **hors budget** : ses plongements
par couche gonflent le fichier ; il reste dans l'essaim E4 à sept familles) ; et les mêmes trois gros modèles en Q8_0
(Qwen3.5-9B 8,87 Gio, Gemma-4-12B 11,8 Gio, Ministral-3-14B 13,4 Gio), qui mesurent le coût de la quantification seule.

Une seule recette par famille ne sépare pas l'effet du type de quantification de celui de l'auteur du fichier : c'est une
limite assumée (§8). Les modèles gros sont les trois seuls modèles de 9 à 14 G déjà dans E4 ; aucune quatrième famille n'est
ajoutée, de sorte que les essaims de gros pairs ont **3 membres au plus** (contre 6 à 7 petits).

## 4. Protocole préenregistré

Écrit avant tout résultat. Toute modification ultérieure est inscrite avec sa date et sa raison.

**Données et exécution.** Les mêmes que E4 et E11 : GSM8K (300 questions dev, 300 test), MATH-500 (250, 250), ARC (300, 300),
MMLU-Pro (300, 300) ; HumanEval+ (82, 82) et MBPP+ (189, 189). Mêmes invites (`solo-v1`, `code-v1`), glouton, 16 emplacements de
3 072 jetons, llama-server `--jinja`, réflexion désactivée, même version de llama.cpp (b11505), via `run_solo.py` et
`run_code.py` sans modification ; seul `--gguf` et `--suffix` changent. Code : un programme glouton et 4 tirages par
problème (température 0,8), exécutés dans le bac à sable d'E11 (`exec_code.py`, `essaim/sandbox.py`). Le manifeste enregistre
l'empreinte du fichier de poids, la révision du modèle d'origine et la version du moteur ; l'analyse vérifie que le
protocole est identique entre tous les systèmes (comme `analyze_e4.check_manifests`) et que chaque fichier quantifié est
bien celui épinglé.

**Garde contre les fichiers défectueux.** Avant toute lecture de résultats, le rapport affiche pour chaque système le taux
d'abstention (aucune réponse extractible) et de troncature (pas de fin naturelle). Une quantification dont l'un des deux est
nettement supérieur à celui du Q8_0 du même modèle est signalée comme défectueuse (gabarit de conversation ou fichier), et
non interprétée comme une perte de précision ; on la relance ou on l'écarte avec une note, jamais en silence.

**Choix sur dev (règles figées ici).**
- Meilleur petit pair du budget : le pair Q8_0 de E4 (fichier ≤ 6,5 Gio) à la meilleure exactitude macro-moyenne dev
  (moyenne des quatre benchmarks) ; égalité : fichier plus petit, puis ordre alphabétique.
- Meilleure quantification d'un gros modèle : même règle, parmi ses fichiers du budget. Meilleur gros pair : le meilleur de ces
  trois. Pour le code : la même règle sur le pass@1 glouton dev moyen, parmi les quantifications dont le code a été généré.
- Poids de vote (vote pondéré de Nitzan-Paroush, `analyze_e4.fit_weights`) : appris sur dev, **refaits pour chaque essaim sur
  ses propres membres**. Règle de sélection du code : celle choisie par E11 sur son propre dev (regroupement fonctionnel,
  score « count », tous les candidats), lue dans `e11_summary_colab.json`, jamais re-choisie ici.

**Rapport sur test, une seule fois.** Le test n'est lu que pour les choix figés ; tous les chiffres test calculés sont
rapportés (aucune recherche sur le test, aucune sélection a posteriori). Les tests exacts sont ceux d'E4 et d'E11
(`essaim/stats.py`) : intervalle de score de Tango à 95 % pour la différence appariée, supériorité à 2,5 % unilatéral,
équivalence par deux tests unilatéraux (TOST) à 5 % avec une marge de ±2 points (`DELTA`), non-infériorité.

**Comparaison principale (la seule confirmatoire).**
Meilleur gros pair quantifié du budget **contre** meilleur petit pair Q8_0 du budget, sur les 1 150 questions test des quatre
benchmarks de QCM et de mathématiques mises bout à bout, à mémoire à peu près égale. Tout le reste est exploratoire, sans correction de
multiplicité, et décrit comme tel : par benchmark, par modèle, par type de quantification, tous les essaims, le code, les
paliers de mémoire (≤ 5,5 Gio et ≤ 6,5 Gio).

**Essaims comparés** (vote pondéré) : 7 familles Q8_0 (E4) ; petits pairs du budget (6) ; les 3 meilleurs petits par dev ;
3 gros (une quantification par modèle, choisie sur dev) ; 3 gros tous au même type ; essaim mixte (petits du budget + 3 gros).
Courbes de passage à l'échelle comme en E4 : moyenne sur tous les sous-ensembles de k familles, et les k meilleures par dev.
Mémoire totale et plus gros fichier sont rapportés à côté de chaque essaim, parce que « à mémoire égale » vaut par pair, non
pour la somme.

**Ce que l'analyse ne conclut pas.** Elle ne dit pas « le gros quantifié est meilleur » à partir d'un seul modèle ou d'un
seul type : la phrase est permise seulement si la comparaison principale est « supérieur » *et* que les gardes ci-dessus
sont sans alerte. Un essaim de 3 gros et un essaim de 6 petits diffèrent par le nombre de pairs, la mémoire totale et la
diversité : ce sont des comparaisons de configurations, non d'effets isolés. Les résultats code portent sur six fichiers
seulement (les bornes) : une quantification intermédiaire n'est pas inférée.

## 5. Mise en œuvre

| fichier | rôle |
| --- | --- |
| `phase0/e16_quant.py` | table `QUANT` (fichiers, tailles, SHA-256), budget, réservations GPU, construction des plans, estimation de vitesse grand public |
| `phase0/colab_jobs.py` (fin du fichier) | plans `e16-1`, `e16-mc`, `e16-code`, `e16-2` ; types de jobs `solo-q`, `code-q`, `solo-speed` ; téléchargement vérifié par SHA-256 (`prepare_e16`). `SOLO` n'est pas modifié |
| `phase0/colab/session_json.py` | `capacity()` connaît les types de jobs d'E16 (choix du GPU) |
| `phase0/exec_e16.py` | lance `exec_code.py` une fois par fichier quantifié (résultats `codeexec_e16<modèle>_<type>_colab_*`, jamais partagés entre modèles : `exec_code.py` refuse de reprendre si la liste des modèles change, et `e16-2` ajoute des fichiers après `e16-1`), l'un après l'autre |
| `phase0/speed_quant.py` | `llama-bench` (pp512, tg128, 3 répétitions) de chaque fichier, seul sur le GPU ; le nom du GPU est dans le manifeste, un point de reprise restauré sur un autre GPU est refusé |
| `phase0/analyze_e16.py` | rapport `results/e16_report.md` et `results/e16_summary.json`, réutilise `analyze_e4`, `analyze_code` et `essaim/stats.py` |
| `phase0/tests/test_e16.py` | tests (table, plans, commandes, analyse sur les résultats Q8_0 réels relus comme des quantifications) |

Les fichiers de résultats portent le suffixe `_<type>_colab` (ex. `solo_Qwen__Qwen3.5-9B_iq3xxs_colab_gsm8k_dev.jsonl`), qui
contient `_colab_` : le `pull` de `colab_phase0.sh` les récupère sans modification.

Les plans, dans l'ordre d'une exécution fragmentée si le solde le demande :

| plan | contenu | jobs |
| --- | --- | --- |
| `e16-1` | tout le cœur : E4 sur les 12 fichiers, code sur les 6 bornes, exécution, vitesse | 12 + 6 + 1 + 1 |
| `e16-mc` | E4 sur les 12 fichiers + vitesse (moitié peu coûteuse) | 12 + 1 |
| `e16-code` | code sur les 6 bornes + exécution (l'autre moitié) | 6 + 1 |
| `e16-2` | code sur les 6 autres fichiers + exécution de toutes les quantifications | 6 + 1 |

Bornes pour le code (fixées avant résultat) : IQ3_XXS et IQ4_XS pour Qwen3.5-9B, IQ3_XXS et QAT Q4_0 pour Gemma-4-12B, IQ3_XXS et
Q3_K_M pour Ministral-3-14B. `e16-1` = `e16-mc` puis `e16-code` ; `e16-2` n'est lancé que si les bornes montrent que le code vaut la peine.

**Mémoire GPU honnête.** Chaque job réserve (`MemoryBudget`) la taille de son fichier plus le surcoût fixe de son modèle
(cache KV de 16 emplacements de 3 072 jetons, tampons, marge), égal à la réservation de la ligne Q8_0 de `SOLO` moins son fichier
(un test le vérifie) : 14 à 18 Gio par job. Sur un A100 de 40 Go (≈ 37 Gio utilisables), **deux jobs** tournent à la fois, pas trois.

**Vitesse.** Le débit en jetons/s sous charge (E4, E11) dépend des voisins qui partagent le GPU : il n'est comparable qu'entre
jobs lancés ensemble. Le job `solo-speed` mesure donc, en dernière étape, seul sur le GPU, la génération sur une requête
(`tg128`) et le traitement du prompt (`pp512`) de chaque fichier quantifié **et** du Q8_0 de 12 modèles de comparaison
(les 9 petits de E4 et les 3 gros ; ~50 Go à télécharger en plus). Le A100 n'est pas le GPU visé : un décodage à une requête y est limité par le
lancement des noyaux autant que par la bande passante, donc les rapports entre fichiers y sont plus faibles que sur la carte de
8 Go. Le résultat sert à rapporter la mémoire et à comparer les fichiers entre eux ; il ne prédit pas la vitesse grand public.

**Estimation de la vitesse grand public, sans GPU, sans Vulkan.** Le décodage à une requête lit chaque poids une fois par jeton :
jetons/s ≈ B / taille du fichier, où B est une bande passante effective. `speeds_consumer.json` donne des mesures `llama-bench tg128`
en Q8_0 sur la RX 6650 XT (bande passante nominale 280 Go/s). Pour les quatre modèles denses (Qwen3-1.7B, Qwen3-4B, Granite-4.2-3B,
SmolLM3-3B) la bande passante effective va de 121 à 218 Go/s (médiane 184) ; `analyze_e16.py` en déduit, pour chaque fichier, une
estimation (médiane) et un intervalle (minimum et maximum observés), par exemple 41 (27 à 48) jetons/s pour Qwen3.5-9B IQ3_XXS,
36 (24 à 42) pour Gemma-4-12B IQ3_XXS, 27 (18 à 33) pour Ministral-3-14B Q3_K_M. **Limites** : (i) Gemma E2B et Qwen3.5-2B sont écartés
de l'étalonnage (plongements par couche, architecture hybride) ; Qwen3.5-2B atteint seulement 72 Go/s, ce qui donnerait environ
16 jetons/s pour Qwen3.5-9B IQ3_XXS : l'estimation est donc optimiste pour la famille Qwen ; (ii) le coût de déquantification des
types `IQ*` sur Vulkan est inconnu et n'est pas dans le modèle ; (iii) l'étalonnage est fait sur des fichiers de 1,8 à 4,3 Go et
extrapole à 5 à 7 Go. **La mesure réelle** (`llama-bench -m <fichier> -p 512 -n 128 -ngl 99` avec la build Vulkan sur le PC, fichiers déjà
téléchargés par ailleurs) reste à faire par le propriétaire ou la session PC ; elle n'a pas été lancée ici, comme demandé.

## 6. Coût attendu sur le A100

Durées des jobs d'E4 et d'E11 déjà mesurées sur le même A100 (somme des temps de requête / 16 emplacements, trois jobs en
concurrence) : MC 0,85 h pour Qwen3.5-9B, 0,83 h pour Gemma-4-12B, 0,64 h pour Ministral-3-14B ; code 0,55 h, 0,61 h et 0,28 h.
Une quantification de 3 à 4,5 bits lit moins d'octets que le Q8_0 mais les noyaux des types `IQ*` déquantifient plus : on suppose un
temps comparable (à ±30 % près).

| étape | travail (heures de job, mesure E4/E11) | heures murales (2 à 3 jobs en parallèle) |
| --- | --- | --- |
| E4 sur 12 fichiers | 4 × 0,85 + 5 × 0,83 + 3 × 0,64 = 9,5 | 3,2 à 4,8 |
| code, 6 bornes | 2 × 0,55 + 2 × 0,61 + 2 × 0,28 = 2,9 | 1,0 à 1,5 |
| exécution (CPU), 6 fichiers | — | 0,5 (estimation ; E11 a exécuté 13 modèles) |
| vitesse (24 fichiers, 12 Q8 à télécharger) | — | 0,5 |
| téléchargement (64 Go de quantifications), vérifications SHA-256, installation | — | 0,3 |
| **`e16-1`** | | **5,5 à 7,6 h** |

Avec 5,3 unités par heure : **≈ 29 à 40 unités pour `e16-1`**. Solde ≈ 59 unités, plancher 15 : le plan tient avec 19 unités de reste
dans le pire cas. `e16-mc` seul : ≈ 4,0 à 5,6 h, **21 à 30 unités** ; `e16-code` ensuite : ≈ 1,8 à 2,3 h, **10 à 12 unités** ; `e16-2` :
≈ 2 à 2,5 h, **11 à 13 unités** (à lancer seulement après lecture des bornes). Ces estimations ne sont pas des mesures : le
lanceur écrit le temps de chaque job dans `colab_status.json`, à comparer ensuite. Si le solde descend sous 45 unités avant le
lancement, commencer par `e16-mc`. Vérifier `colab usage` avant et après, et libérer la VM (`down`) dans tous les cas.

Définir `MYRIAD_ROOT` dans l’environnement WSL vers le clone du projet.

Commande exacte (depuis WSL, après fusion de la branche dans `main`, une seule campagne à la fois) :

```
wsl.exe -e bash -lc 'cd "$MYRIAD_ROOT" && bash phase0/colab/colab_phase0.sh up e16-1 A100'
wsl.exe -e bash -lc 'cd "$MYRIAD_ROOT" && bash phase0/colab/colab_phase0.sh status'
wsl.exe -e bash -lc 'cd "$MYRIAD_ROOT" && bash phase0/colab/colab_phase0.sh pull'
wsl.exe -e bash -lc 'cd "$MYRIAD_ROOT" && bash phase0/colab/colab_phase0.sh down'
cd phase0 && uv run python analyze_e16.py
```

Il faut passer `A100` explicitement : sans GPU indiqué, `select_gpu` choisirait un L4 (20 Gio requis ≤ 22), ce qui allongerait la campagne.

## 7. Ce qu'on saura, et ce qu'on ne saura pas

On saura : si, au budget de 6,5 Gio et sur ces six benchmarks, un 9 à 14 G à 3 à 4,6 bits bat le meilleur petit Q8_0
(seul, puis en essaim de trois) ; combien de points coûte chaque type de quantification par rapport au Q8_0 du même modèle ; la taille
et le débit de chaque fichier.

On ne saura pas : l'effet de la quantification du cache KV ; la qualité sur des contextes longs (tout tient en 3 072 jetons) ; la vitesse
sur la carte grand public (estimation seulement) ; la qualité en raisonnement long (E13) ; ce que ferait un autre quantificateur
(un seul auteur par famille).

## 8. Limites connues, à écrire dans l'article

- Une recette par famille : l'effet du type de quantification et celui de l'auteur du fichier ne sont pas séparés.
- Trois gros modèles seulement, trois familles : les essaims de gros pairs ont 3 membres, et les modèles diffèrent par le
  nombre de paramètres, la famille et le Q8_0 de départ en même temps.
- Les modèles de comparaison à 4 à 5 Go sont ceux d'E4, choisis pour d'autres raisons : aucun petit modèle n'a été ajusté à
  la taille du budget (Qwen3.5-4B occupe 4,2 Gio, donc moins que les gros fichiers : l'égalité de mémoire est un plafond
  commun, non une égalité de consommation).
- Le A100 mesure des débits sans rapport direct avec ceux d'une carte à 280 Go/s.
- Comparaisons exploratoires sans correction de multiplicité, hors la comparaison principale.
- Je n'ai pas relu en entier les articles cités (voir §2) ; les chiffres de `yrougy/llm-quant-bench` viennent d'un
  banc communautaire.
