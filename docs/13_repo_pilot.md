# Pilote 5c sur dépôt Python réel, sans Docker

Premier pilote CPU mesuré le 2026-10-10 : **0 réussite sur les trois réparations historiques SymPy**,
avec Qwen3.5-2B Q8_0 et llama.cpp direct en loopback WSL. Les trois sources restent inchangées.
Smoke WSL : références **7 + 6 + 5 comparaisons de chaînes réussies**, chaque base échoue sur un cas
pertinent. Ce contrôle du harnais reste distinct de la mesure agent décrite ci-dessous.
Le pilote Aider CPU 0/3 reste distinct : exercices, pas réparation d'un dépôt logiciel réel.

## Population, filtres et sélection avant inférence

Population : les **77 SymPy** du split test SWE-bench Lite à
`6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`. Le parquet de 300 lignes est épinglé
(1 119 540 octets, SHA-256 `7a21f37b8bc179c7db5beeb14e88ac538ba283455c776e6b2535bbfb6e3551b4`).
Les lignes du viewer public déjà téléchargées ont fourni l'inventaire ; les **77 lignes SymPy** et
les six Requests ont été comparées au parquet épinglé. Le manifeste conserve les IDs, bases et chemins.

Critères cumulatifs : vrai commit Python avant réparation ; changement de **printer à sorties chaînes** ;
code et dépendances nécessaires permissifs ; expressions fixes et tests sans réseau ; assertions dans
un contrôleur séparé ; runtime compatible verrouillé ; bac Linux fermé et smoke négatif/positif.
Règle : trier les IDs éligibles lexicographiquement et retenir au plus trois premiers, avant inférence.
Le début lexical complet est :

| instance | critère technique | décision |
| --- | --- | --- |
| sympy__sympy-11400 | printer C, relations et sinc | retenue |
| sympy__sympy-11870 | réécriture trigonométrique d'expressions symboliques, hors printing | exclue |
| sympy__sympy-11897 | printer LaTeX, Piecewise dans un produit | retenue |
| sympy__sympy-12171 | printer Mathematica, Derivative | retenue |

Les 73 autres IDs viennent après cette limite et sont marqués **non examinés après la limite**,
pas déclarés inéligibles. Aucun filtre basé sur une réussite de modèle, aucun remplacement selon un score.
La sélection et tous les fichiers autorisés sont dans le [manifeste](../phase0/repo_pilot/tasks.json).

| tâche | commit initial exact | PR et référence |
| --- | --- | --- |
| 11400 | `8dcb12a6cf500e8738d6729ab954a261758f49ca` | [PR 11400](https://github.com/sympy/sympy/pull/11400), `63215172cb3ec279411432f541d7d5dcf3bbb939` |
| 11897 | `e2918c1205c47345eb73c9be68b14c0f15fdeb17` | [PR 11897](https://github.com/sympy/sympy/pull/11897), `779634e85281c1d36c623e497222bf53a5d97c38` |
| 12171 | `ca6ef27272be31c9dc3753ede9232c39df9a75d8` | [PR 12171](https://github.com/sympy/sympy/pull/12171), `278c050255dc4f5942338de97b986560cb6794b8` |

Les trois fichiers de référence aux commits de fusion correspondent **exactement** aux patches code
Lite appliqués aux bases, vérifié hors exécution. Ils ne remplacent que le module corrigé ; les autres
sources restent à la base. Les archives font 4 766 489 / 4 844 543 / 4 873 086 octets ; tailles et SHA-256
des archives et des **4 248 fichiers** conservés sont épinglés. Préparation par copie d'entrées régulières
explicitement listées, aucune extraction libre de tar. Sources/cache/références restent hors Git.

## Licences et données

Les `LICENSE` de ces **commits historiques**, pas les licences actuelles, sont BSD-3-Clause et conservés :
[11400](../phase0/repo_pilot/licenses/sympy-11400.txt), [11897](../phase0/repo_pilot/licenses/sympy-11897.txt),
[12171](../phase0/repo_pilot/licenses/sympy-12171.txt). Le `setup.py` de chaque base demande mpmath ≥ 0.19.
Le pilote verrouille **mpmath 0.19**, dont l'archive PyPI et la
[licence BSD historique](../phase0/repo_pilot/licenses/mpmath-0.19.txt) sont épinglées.
Aucune mention GPL n'a été trouvée dans les modules Python des trois archives inspectées.
Le pilote n'active aucune dépendance optionnelle SymPy pour ces expressions/printers.
Les dépendances du contrôleur HTTP/figures sont distinctes de la dépendance nécessaire aux printers ;
leurs versions et métadonnées de licence observées sont enregistrées, sans prétendre qu'elles sont
toutes BSD/MIT/Apache (par exemple certifi annonce MPL-2.0).

Le [README SWE-bench épinglé](https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/README.md)
présente **code et données** et annonce MIT ; le
[LICENSE racine](https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/LICENSE)
est MIT. La [fiche HF Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/blob/6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2/README.md)
n'a pas de tag licence. Cette absence n'annule pas l'annonce des auteurs, mais la portée sur chaque texte
d'issue issu de tiers n'est pas établie ici. **Aucun texte d'issue/hint HF n'est republié** : consignes
originales décrivant les réparations historiques et citant les instances/PR, sans problème inventé.
Les cas reproduits viennent des tests de code SymPy BSD ; leurs sources officielles, empreintes,
fonctions et nombres de cas sont épinglés. Patches code/test Lite conservés seulement par empreinte,
fichiers de référence et tests complets téléchargés hors Git.

## Frontière de notation

L'agent explore une copie réelle complète du dépôt initial : **1 401 / 1 419 / 1 422 fichiers**, sans `.git`
ni historique, avec tests visibles au commit initial et licence. `_submission.json` énumère exactement
les **541 / 544 / 545 modules Python existants autorisés**, hors tests et benchmarks. Les autres modifications,
nouveaux fichiers, tests et scores produits par le candidat sont ignorés. Fichiers sources UTF-8 réguliers,
sans liens, ≤ 1 MiB chacun et ≤ 16 MiB au total ; anomalie après épisode = échec candidat noté.

Le contrôleur lit les sources, puis chaque cas repart d'une copie source neuve, sans tests ni références.
Le worker protégé importe le SymPy **candidat** et construit l'expression par un fragment Python fixe de
confiance issu du test officiel. Ce fragment est exécuté seulement **dans le bac**, jamais dans le contrôleur.
Il ne contient ni assertion ni chaîne attendue. Il appelle le printer et retourne une **chaîne JSON**
bornée à 64 KiB, ou un nom d'exception ; aucun eval/pickle sur une réponse candidate.
Le contrôleur compare la chaîne à la valeur attendue ; aucune importation de SymPy candidat côté contrôleur.

Les assertions reproduisent les fonctions concernées **entières** :

| tâche | fonctions officielles | comparaisons |
| --- | --- | --- |
| 11400 | `test_ccode_Relational`, `test_ccode_sinc` | 6 relations + 1 sinc = 7 |
| 11897 | `test_latex_Piecewise` | 6, dont standalone, itex et produits non commutatifs des deux côtés |
| 12171 | `test_Derivative` dans `test_mathematica.py` | 5, variables ordonnées et répétées |

**PASS_TO_PASS n'est pas couvert**, ni les autres fonctions des fichiers de tests. Les nombres `tests`
et `tests_expected` comptent des comparaisons de chaînes, pas des fonctions pytest. Le vérificateur est
fail-fast ; `tests_complete` indique si tous les cas ont été exécutés. Le smoke exige chaque référence
positive complète et au moins un échec pertinent de chaque base. Un ancien problème n'est pas décontaminé
pour les modèles 2026. Cette adaptation non officielle n'est pas un score SWE-bench Lite ou un banc général.

Le bac est `LocalLinuxEnv` inchangé : Landlock ABI ≥ 3, namespaces, seccomp, rlimits, environnement réduit,
runtime protégé en lecture, sans réseau ni accès au dépôt Myriad/cache de contrôle. Le bac E11 est inchangé.
stdout, faux score ou pytest retournant zéro ne décident pas la note. Après reçu d'isolation, sortie zéro
prématurée, syntaxe, exception, réponse non JSON/non chaîne, symlink et délai sont des échecs candidats.
Une protection absente, un mauvais runtime ou une panne du harnais donne une erreur infra, `passed=null`.

## Tests visibles compatibles avec les bases historiques

Les consignes utilisent le runner natif SymPy. `pytest 8.4.2` ne reconnaît pas les anciennes exceptions
XFail/Skipped : le lancer directement sur ces tests donne des échecs trompeurs. Depuis la copie agent,
dans `LocalLinuxEnv` et avec le runtime verrouillé, les commandes sont :

```bash
# sympy__sympy-11400 : fichier C complet
python bin/test --no-subprocess --no-colors --seed 0 sympy/printing/tests/test_ccode.py
# sympy__sympy-11897 : fonction Piecewise entière du fichier historique
python bin/test --no-subprocess --no-colors --seed 0 sympy/printing/tests/test_latex.py -k test_latex_Piecewise
# sympy__sympy-12171 : fichier Mathematica complet
python bin/test --no-subprocess --no-colors --seed 0 sympy/printing/tests/test_mathematica.py
```

Contrôle réel sur les six copies complètes, tests initiaux inchangés et seul fichier source corrigé par
la référence : C **30 tests réussis** sur base et référence ; Mathematica **9 réussis** sur chacune ;
LaTeX **1 réussi** sur base, **1 échec d'assertion** sur référence. Ce dernier échec est réel : l'assertion
initiale de `test_latex_Piecewise` attend l'ancien rendu du produit non commutatif, remplacé dans les tests
finaux de la PR. Il ne signifie pas que le runner est incompatible ni que la réparation est incorrecte.
Le contrôleur indépendant valide toujours les **6 chaînes corrigées** LaTeX, et **18/18** au total.
Les tests visibles initiaux C/Mathematica ne couvrent pas les nouveaux cas ajoutés par leurs PR.

Le fichier LaTeX entier n'est pas conseillé : l'essai réel a donné cinq exceptions hors réparation sur
les deux copies (quatre avertissements `collections` transformés en exceptions et une incompatibilité
AST Python 3.9 dans `test_issue_8470`). Le runner natif traite correctement ses deux XFAIL historiques ;
aucune dépendance optionnelle n'est nécessaire aux commandes retenues. Les warnings d'import anciens
restent visibles ; aucune assertion, source de base, dépendance ou protection n'est neutralisée.

Le smoke contrôle les commandes sur des copies neuves séparées et conserve leurs sorties, codes et reçus
dans `visible_tests`, avec `final_score=false`. Il exige les bilans historiques connus, y compris le
conflit LaTeX précis, pour détecter un défaut de runner. Ces diagnostics ne notent jamais une soumission
candidate ; ni stdout ni leur code retour ne remplacent les comparaisons RPC du contrôleur.

## Préparation et smoke sans modèle

Depuis bash WSL, à la racine du worktree. Runtime dédié sur disque Linux natif, aucune venv partagée
synchronisée en parallèle. Python **3.9.25** évite de modifier les imports `collections` des bases anciennes.
Son binaire (18 598 712 octets, SHA-256 `84c920f304e265c53096e42d83309c6f7b829988f937a7dd15c24eebcb63f59a`)
est vérifié ; `uv.lock` verrouille le runtime de paquets. Aucun modèle téléchargé.

```bash
cd phase0/repo_pilot
UV_PROJECT_ENVIRONMENT="$HOME/.local/share/myriad-repo-pilot/venv" \
  ~/.local/bin/uv sync --frozen --python 3.9.25
cd ..
PY="$HOME/.local/share/myriad-repo-pilot/venv/bin/python"
TASKS="$HOME/.local/share/myriad-repo-pilot/tasks"
PROOFS="$(wslpath 'C:/tmp/myriad-repo-smoke')"
"$PY" run_repo_pilot.py --prepare --tasks-dir "$TASKS"
"$PY" run_repo_pilot.py --check --tasks-dir "$TASKS"
"$PY" run_repo_pilot.py --smoke --tasks-dir "$TASKS" --output "$PROOFS"
REPO_PILOT_LINUX_TESTS=1 "$PY" -m unittest discover -s tests -p test_repo_pilot.py -v
```

Un cache Linux natif est conseillé : préparation initiale dans `phase0/data/repo-pilot` sur DrvFS vérifiée,
mais les milliers de contrôles de chemins y sont plus lents. Les copies d'exécution temporaires sont sur
Linux natif. `--prepare`/`--check` n'exécutent aucun code du dépôt et peuvent être orchestrés depuis Windows ;
`--smoke`/épisodes exigent Linux et l'interpréteur dédié. Toute invocation susceptible de faire de l'inférence,
reprise incluse, exige à nouveau un smoke complet réussi avant le premier appel.

## Commandes pour un endpoint futur déclaré par le parent

**Ces commandes n'ont pas été exécutées avec un modèle.** Aucun serveur/tracker/nœud/identité à créer.
Le parent fournit `MYRIAD_BENCH_URL`, `MYRIAD_BENCH_TOKEN` et un fichier de provenance nonsecret suivant
le [protocole commun](12_code_pilot.md). Même endpoint HTTP loopback ou HTTPS authentifié, client sans
proxy d'environnement ni redirection, budgets et journaux de tous les appels.

```bash
"$PY" run_repo_pilot.py --check --tasks-dir "$TASKS" \
  --single 'myriad:family=FAMILLE_SOLO' --provenance "$PROVENANCE"
"$PY" run_repo_pilot.py --tasks-dir "$TASKS" --output results/repo_pilot_cpu_20261010_01 \
  --single 'MODELE_SOLO_EXACT' --provenance "$PROVENANCE"
"$PY" run_repo_pilot.py --tasks-dir "$TASKS" --output results/repo_pilot_compare_20261010_01 \
  --single 'myriad:family=FAMILLE_SOLO' \
  --vote 'myriad:family=FAMILLE_A' 'myriad:family=FAMILLE_B' \
  --reference 'MODELE_REFERENCE_EXACT' --provenance "$PROVENANCE_COMPARAISON"
"$PY" run_repo_pilot.py --report --tasks-dir "$TASKS" --output results/repo_pilot_compare_20261010_01
```

Boucle `agent.py` et pilote commun `code_pilot.py`, avec petit paramètre backend, sans copie de lanceur.
Single/vote/cascade/reference utilisent les mêmes plafonds : 20 étapes, 1 024 jetons par requête,
12 288 plafonds réservés cumulés par épisode, 900 s d'épisode, 30 s de commande, 120 s de vérification,
10 s par RPC incluses dans ces 120 s ; température 0, seed 0. La référence seule fait ses propres trois
épisodes. La cascade consulte la référence sur désaccord/format, jamais les assertions finales.
Jetons connus/usage inconnu distincts des réservations ; temps mur agent + vérification ; énergie et
coût seulement **estimés**, par défaut 100 W et 0,25 €/kWh, hypothèses agrégées à adapter aux pairs distants.

Manifestes, provenance structurée modèle/serveur, limites, hashes des modules et lock, transcriptions,
sources initiales/finales, appels, votes, reçus, réponses et verdicts restent liés à la reprise stricte.
Rapport JSON/Markdown et figures SVG/PNG/PDF seulement après grille complète sans erreur infra.
Même verrou et refus des transcriptions orphelines que le pilote Aider. Tous les artefacts restent
soumis au scan de secrets/chemins privés : une sortie contenant de tels chemins peut bloquer son export.
Whitelist exacte : onze ressources, **22 noms d'artefacts** par dossier générique
`results/repo_pilot_{cpu,compare}_AAAAMMJJ_NN`. Exception explicite : le dossier brut de cette campagne
`repo_pilot_cpu_20261010_01` est entièrement exclu, et son dérivé public exact admet seulement
treize artefacts et `publication.json`. Caches, smoke, locks, patches, voisins et sous-dossiers exclus.

## Diagnostic Requests conservé

Le [manifeste historique](../phase0/repo_pilot/requests_diagnostic.json) conserve les six bases Requests,
archives et 25 preuves vérifiées : 1963, 2148, 2317, 2674, 3362, 863, dans cet ordre lexical.
Les six embarquent Chardet LGPL-2.1-or-later, notamment `chardet2` pour Python 3 de 863.
Sources représentatives : [NOTICE 1963](https://github.com/psf/requests/blob/110048f9837f8441ea536804115e80b69f400277/NOTICE),
[Chardet 3362](https://github.com/psf/requests/blob/36453b95b13079296776d11b09cab2567ea3e703/requests/packages/chardet/__init__.py).
La licence principale est Apache-2.0 sauf [863, ISC](https://github.com/psf/requests/blob/a0df2cbb10419037d11d04352b3175405ab52941/LICENSE).
2317/2674 ajoutent des tests httpbin ; les autres scénarios ajoutés sont locaux. Compatibilité runtime
Requests non mesurée. Le refus LGPL applique le critère de ce pilote, pas une interdiction générale d'évaluer
localement du LGPL. Aucun retrait/substitution de dépendance ni rebase moderne n'a été effectué.
Ce diagnostic ne bloque pas l'alternative SymPy désormais préparée.

## Validation et travail restant

Smoke réel WSL, Landlock ABI 7, seccomp/namespaces reçus : trois bases rejetées, références complètes
7/6/5. Probes synthétiques du transport : faux score, exit zéro, syntaxe, timeout, symlink, assertion
inaccessible, tests modifiés ignorés et erreur infra distincte. Les validations finales des suites sont
rapportées ci-dessous ; aucun test existant affaibli.

Validation finale après correction P2 du runner : app `uv run pytest -q`, **426 réussis, 3 ignorés**
(97,43 s) ; phase0 `uv run python -m unittest discover -s tests -q`, **265 tests, 38 ignorés** (52,226 s).
Sous WSL, suite réelle du pilote dépôt avec protections : **17/17 tests** (2,814 s), sans tests ignorés.
Validation Aider précédente : **54/54 tests** (8,649 s) ; modules Aider et bac inchangés dans ce correctif.
Le contrôle `--check` précédent valide 4 248 fichiers ; manifestes et cinq ressources épinglées inchangés.

Smoke précédent conservé hors Git : `myriad-repo-pilot-smoke-final-20261010/smoke.json`, **45 656 octets**,
SHA-256 `c03ed2dddc2feb2521d8e1fd03c75835fdb1ed3b5ed8ec31ae4b641eedf40e07`.
Smoke final avec les six diagnostics natifs : `myriad-repo-pilot-runner-smoke-20261010/smoke.json`,
**59 359 octets**, SHA-256 `8105b01c6262aefde9f05a72b35a70aea28cd7837d21e45572b9ff05304fd02b`,
conservé sous `C:/tmp` pour survivre à la fermeture WSL. Code retour **0**, trois bases négatives,
références **18/18** et bilans natifs documentés ci-dessus. Exploration du fichier LaTeX entier et
première vérification ciblée conservées séparément hors Git dans `myriad-repo-runner-probe.json` et
`myriad-repo-runner-targeted.json`. Aucun audit autonome, inférence ou export effectif de ce correctif.
Bases : échecs `wrong_string` après 1/7, 5/6 et 1/5 cas ; références : 7/7, 6/6 et 5/5.
Les références positives couvrent tous les cas déclarés ; les bases restent fail-fast.

Le pilote solo CPU ci-dessous complète ce jalon. Une comparaison vote/cascade/référence reste à autoriser
et mesurer séparément, à provenance et budgets déclarés ; aucune preuve de réseau Myriad n'est acquise.
E11/chain5, ses résultats, Colab et les réglages du propriétaire n'ont pas été touchés.
Audit, fusion, push et export effectif sont laissés au parent.

## Mesure solo CPU du 2026-10-10

Campagne unique `repo_pilot_cpu_20261010_01`, engagée à **12:08:16 CEST** et terminée à **12:17:17 CEST**,
code retour **0**, incluant le smoke obligatoire avant les épisodes et la production du rapport.
Les durées par tâche ci-dessous excluent ce smoke et incluent agent et vérification. Présélection,
ordre 11400/11897/12171, instructions et budgets par défaut sont restés fixes : 20 étapes, 1 024 jetons
par appel, 12 288 réservés et 900 s par épisode, température 0, seed 0. Aucun épisode rejoué.

La provenance nonsecrète a été déclarée avant la mesure et copiée dans le
[manifeste](../phase0/results/repo_pilot_public_cpu_20261010_01/manifest.json) : modèle Qwen3.5-2B Q8_0,
révision `f6d5376be1edb4d416d56da11e5397a961aca8ae`, **2 012 012 800 octets**, SHA-256
`1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1`.
Archive llama.cpp b11505 vérifiée : SHA-256
`9e23bb8c48e0abbd03a558e8c341ec768fa32d56146a09e5fd13db5b0461e60c` ; version exécutée `ff5888f99`.
CPU i5-10400F, hôte 32 Gio, WSL2 Linux x86_64, contexte 32 768, un slot, zéro couche GPU,
reasoning-budget 0 et jinja ; six threads CPU selon le journal du serveur. **L'hôte était partagé** :
tar/gzip de préparation Colab vers 12:08–12:09, puis uploads CLI pendant l'intervalle de mesure
12:08–12:17 sur le même PC. Cette compression et ces transferts ajoutaient une charge CPU/E/S locale,
au-delà de la supervision seule ; leur effet sur la latence n'est pas quantifié. Les temps sont donc
observés sur hôte partagé, pas un banc CPU isolé. La déclaration brute antérieure « CPU-light supervision »
est conservée honnêtement ; cette précision est rétrospective, sans correction artificielle des durées.
Les 100 W sont une hypothèse globale non mesurée, sans addition d'un coût Colab à ce scénario local.
Aucune requête externe ou référence modèle n'a été effectuée par le pilote.
Il s'agit du **serveur direct WSL**, sans passerelle, essaim ou cascade Myriad.

Le [rapport](../phase0/results/repo_pilot_public_cpu_20261010_01/report.md), le
[résumé JSON](../phase0/results/repo_pilot_public_cpu_20261010_01/summary.json) et les
[verdicts](../phase0/results/repo_pilot_public_cpu_20261010_01/verdicts.json) donnent **0/3**, single seul,
trois échecs candidats valides, aucune erreur d'infrastructure pendant les épisodes.

| tâche | arrêt | temps mur agent + vérification | jetons retournés | plafonds réservés | cas exécutés |
| --- | --- | --- | --- | --- | --- |
| 11400 | budget de requêtes épuisé | 138,266325 s | 676 | 12 288 | 1/7 |
| 11897 | budget de requêtes épuisé | 209,437870 s | 1 772 | 12 288 | 5/6 |
| 12171 | budget de requêtes épuisé | 139,642720 s | 531 | 12 288 | 1/5 |

Les trois épisodes s'arrêtent après douze appels, avant le plafond de vingt étapes. Sur
[11400](../phase0/results/repo_pilot_public_cpu_20261010_01/single__sympy__sympy-11400.jsonl), le modèle lit le
printer C mais cherche surtout un historique Git absent et des dossiers supposés. Sur
[11897](../phase0/results/repo_pilot_public_cpu_20261010_01/single__sympy__sympy-11897.jsonl), le premier message
atteint 1 024 jetons et est rejeté pour format ; les onze commandes suivantes répètent une recherche
Git dans un dossier supposé inexistant. Sur
[12171](../phase0/results/repo_pilot_public_cpu_20261010_01/single__sympy__sympy-12171.jsonl), le modèle lit le
printer Mathematica et la liste des modules autorisés, puis continue à chercher Git et des dossiers
supposés. Aucune source soumise n'est modifiée. Les sept cas exécutés au total reproduisent les
échecs des bases (`wrong_string`) ; le vérificateur reste fail-fast et ne prétend pas avoir exécuté
les dix-huit cas sur ces candidats. Aucune sentinelle de soumission finale n'a été produite.

Les **2 979 jetons retournés sont tous connus**, sur 36 appels. Les **36 864 plafonds réservés**
sont une comptabilité de budget, pas des jetons consommés ni une facture. Moyenne mesurée :
**162,448972 s/tâche**. À **100 W supposés** et **0,25 €/kWh supposé**, énergie et coût moyens
**estimés**, non mesurés : **4,512471435 Wh** et **0,001128117859 €/tâche**.
Wilson 95 % descriptif : [0 ; 0,561] pour n=3 ; la sélection lexicographique ne constitue pas
un échantillon représentatif. Seulement dix-huit assertions de chaînes, aucun PASS_TO_PASS,
problèmes anciens sans décontamination, ni score SWE-bench officiel ni vraie comparaison réseau.
Aucun chiffre de ce petit pilote n'est ajouté à l'article.

Figures publiques, PNG identiques aux originaux inspectés : [exactitude/temps PNG](../phase0/results/repo_pilot_public_cpu_20261010_01/accuracy_seconds_per_task.png)
([SVG](../phase0/results/repo_pilot_public_cpu_20261010_01/accuracy_seconds_per_task.svg),
[PDF](../phase0/results/repo_pilot_public_cpu_20261010_01/accuracy_seconds_per_task.pdf)) et
[exactitude/coût estimé PNG](../phase0/results/repo_pilot_public_cpu_20261010_01/accuracy_estimated_eur_per_task.png)
([SVG](../phase0/results/repo_pilot_public_cpu_20261010_01/accuracy_estimated_eur_per_task.svg),
[PDF](../phase0/results/repo_pilot_public_cpu_20261010_01/accuracy_estimated_eur_per_task.pdf)).

### Preuves et reproduction sans inférence

Avant la première requête, `--check` réel : **4 248 fichiers**, retour 0. Smoke explicite conservé
sous `C:/tmp/myriad-repo-cpu-smoke-20261010/smoke.json`, SHA-256
`a4afa2ca7d8441d290e804bd19eb75fb04cca9a300316724f66b6a9813714046` : trois bases négatives,
références **18/18**, six diagnostics natifs conformes, Landlock ABI 7/seccomp/namespaces.
Le CLI a refait ce smoke sans contournement avant les épisodes ; sa preuve est également conservée
hors des artefacts publiables sous `C:/tmp/myriad-repo-cpu-mandatory-smoke-20261010.json`, SHA-256
`e954f832bfb61f135b836499c5baaf708961bf796607c957d40d12c1cac3d253`.
Provenance déclarée : `C:/tmp/myriad-repo-cpu-provenance-20261010.json`, SHA-256
`e9bcc6ecef84da8076a9959e37695fa486abd6b37a67f5c8823dfbc1b0730fd3`.

Un premier refus de lancement est conservé séparément : variable client obligatoire omise,
retour 1 **avant manifeste et avant toute inférence**, dossier de sortie vide, serveur arrêté.
L'invocation corrigée fournit uniquement une valeur client factice nonsecrète par environnement,
requise par le harnais pour ce serveur local sans authentification ; aucun identifiant existant modifié.
Ce refus préparatoire n'est ni un épisode interrompu ni une relance de tâche mesurée.

Manifeste d'exécution inchangé SHA-256 : `d6c1c1fef11b092525ee496da13e02d9fcd72336f9e5aa11aa84be112c926911` ;
verdicts privés originaux SHA-256 : `64d5fe16fa5472fd1ad0fe0c8b7a734124cb52fc506f5369e0b1ac741f6ce2ff`.
Les hashes de campagne, trois transcriptions, sources initiales/finales, sept modules et lock,
paramètres et compteurs ont été vérifiés ensemble. Inventaire original et contrôle d'intégrité :
`C:/tmp/myriad-repo-cpu-integrity-20261010.json`.

`--report` a été exécuté sur une **copie des cinq entrées brutes**, serveur arrêté et sans inférence :
JSON, Markdown et deux PNG identiques octet pour octet. Les SVG diffèrent uniquement par date et
identifiants générés ; les PDF uniquement par date de création, comparaison normalisée réussie.
Les originaux n'ont pas été modifiés. Preuves persistantes sous
`C:/tmp/myriad-repo-cpu-reproduction-20261010/` et
`C:/tmp/myriad-repo-cpu-reproduction-comparison-20261010.json`.

```bash
# Dans un dossier neuf, copier uniquement manifest.json, verdicts.json et les trois single__*.jsonl.
"$PY" run_repo_pilot.py --report --tasks-dir "$TASKS" --output "$COPIE_DES_BRUTS"
```

Journaux nonsecrets et commandes : `C:/tmp/myriad-repo-cpu-*.log`. Serveurs temporaires WSL PID 765
(refus préparatoire), puis 320 (mesure), tous deux arrêtés, retours 0 et port 18490 fermé.
Aucun service persistant, tunnel ou élargissement d'écoute ; aucun appel Colab, arrêt/reprise E12,
Docker, audit imbriqué, export effectif, fusion, push, tag, release ou déploiement.

Suites finales obligatoires, code retour **0** chacune : app `uv run pytest -q`, **426 réussis,
3 ignorés**, 98,61 s ; phase0 Windows `uv run python -m unittest discover -s tests -q`,
**305 tests, 48 ignorés**, 86,418 s. Venvs Windows dédiées existantes via `UV_PROJECT_ENVIRONMENT`,
sans partage avec Linux. Les quatre caches E11 ignorés nécessaires à la validation ont été copiés
et comparés par hash depuis la racine, sans modification des originaux ni des sorties E11.
Les seize sources figées E11, les treize corrections de l'article, la boucle agent et le bac sont inchangés.

### Dérivation publique distincte, sans rejeu

Les **treize artefacts de `7f95fe9` restent privés et immuables**, ancien rapport compris. Leur scan
historique donnait 258 occurrences bloquantes : diagnostics contenant des chemins home réels,
chemins supposés par le modèle et adresses des instantanés de sources. Ils ne sont pas réécrits et
le dossier brut est maintenant entièrement exclu de la sélection publique, sans remplacement silencieux.
Tous les liens de la mesure ci-dessus pointent vers `repo_pilot_public_cpu_20261010_01`, un dossier distinct.

Le [helper hors inférence](../phase0/publish_repo_pilot.py) lit uniquement JSON et SHA, jamais du code
candidat par import, exec ou pickle. L'[inventaire gelé](../phase0/repo_pilot/cpu_20261010_private_hashes.json)
verrouille les tailles/hashes des treize originaux. Il impose cette campagne et ses trois tâches,
les accords end/verdicts/transcriptions, les notes, compteurs, paramètres et durées. Les sources
initiales **et** finales sont vérifiées contre les modules des bases épinglées et entre elles :
**541/544/545 modules inchangés**. Toute source réparée, manquante ou différente fait refuser cette
dérivation limitée ; elle ne peut donc pas faire disparaître un vrai patch.

Les trois records `changes`, contenant chacun les gros instantanés initiaux/finals, sont remplacés
par un type distinct `pinned_source_references` : dépôt, commit initial, archive, chemins, tailles et SHA.
Les hashes ne sont pas présentés comme des chaînes de code. Les sources BSD se récupèrent dans les
archives épinglées du [manifeste des tâches](../phase0/repo_pilot/tasks.json), licences conservées.
Les préfixes home POSIX/Windows des messages, commandes, réponses et diagnostics sont remplacés par
`<HOME>` ; les champs assainis et occurrences sont comptés sans recopier les chaînes originales.
**Les commandes dérivées ne constituent pas un rejeu textuel exact des commandes privées.** Aucun
champ scientifique, code de retour, statut, temps ou hash de source n'est masqué ; un tel changement
est refusé. Seuls les `transcript_sha256` des verdicts sont recalculés pour lier les nouvelles transcriptions.
Les end publics s'accordent avec les verdicts publics ; `campaign_hash`, notes et mesures sont conservés.

[publication.json](../phase0/results/repo_pilot_public_cpu_20261010_01/publication.json) relie les
tailles/SHA des treize originaux privés et treize dérivés, le schéma, les règles, les comptes de masquage
et les hashes des générateurs de publication et de rapport. Il n'inclut pas son propre hash : aucun cycle.
Le manifeste d'exécution est **identique au brut**, y compris la provenance antérieure et les sept
hashes du code réellement exécuté. Le nouveau hash de `code_pilot.py`, dont seul le rapport a évolué,
est une provenance de **publication** séparée ; il n'est pas substitué au hash historique d'exécution.
Le rapport autonome dit désormais « plafonds réservés » et précise : ni consommation de jetons ni facture.
L'ancien rapport privé et les résultats Aider sont conservés sans modification.

La dérivation produit le même summary et les mêmes PNG octet pour octet. Le Markdown diffère seulement
par ce libellé, sa précision et la mention de dérivation. SVG/PDF gardent les mêmes figures et métriques ;
leurs dates/identifiants sont fixés pour rendre deux dérivations identiques. La date PDF en UTC raccourcit
aussi le bloc de métadonnées : son espacement et l'offset `startxref` associé changent, sans changement
du contenu graphique. La comparaison normalisée vérifie ces seules différences. Un rapport peut être reproduit
sur une copie **publique**, sans disposer des instantanés privés et sans serveur :

```bash
# Création hors inférence, sortie neuve obligatoire ; jamais lancer run_repo_pilot sur les bruts immuables.
"$PY" publish_repo_pilot.py
# Après copie des quatorze fichiers publics dans un dossier neuf :
"$PY" publish_repo_pilot.py --report --output "$COPIE_PUBLIQUE"
```

Les filtres globaux du scan restent inchangés, sans nouvelle exception. Un secret ou chemin privé
injecté ensuite dans un dérivé reste bloquant. Aucun export effectif ou publication n'est réalisé ici ;
le parent effectue l'audit Astra puis examine la publication.

Validation finale du correctif hors inférence : app **426 réussis / 3 ignorés**, 106,37 s ; phase0 Windows
**315 tests / 50 ignorés**, 107,369 s, commandes complètes obligatoires ci-dessus, retours **0**.
Ciblé WSL Python 3.9.25 : **9/9**, 11,128 s, retour **0**, incluant les sources traitées comme texte,
refus de corruption et deux dérivations réelles. Les tests de sélection/scan vérifient l'exclusion complète
du brut, les quatorze noms publics exacts, les voisins refusés et le blocage d'un secret ou home injecté.
Les masquages documentés dans `publication.json` comptent **763 occurrences**, dont les répétitions
dans les historiques, et les trois records d'instantanés remplacés. Aucun autre champ n'est changé.

Preuves persistantes : `C:/tmp/myriad-repo-cpu-public-commit-integrity-20261010.json` (treize originaux
identiques au commit), `C:/tmp/myriad-repo-cpu-public-reproduction-20261010/proof.json` (deux dérivations
et copie publique reproduite octet pour octet), `C:/tmp/myriad-repo-cpu-public-figure-comparison-20261010.json`
(différences autorisées des figures), journaux `C:/tmp/myriad-repo-cpu-public-*.log`.
Scan **en lecture seule de la sélection publique finale : 859 fichiers, zéro blocage, retour 0** ;
preuve `C:/tmp/myriad-repo-cpu-public-scan-20261010.json`. Les avertissements de références restent
non bloquants ; aucun filtre global ou exception n'a été relâché et aucun export n'a été effectué.
