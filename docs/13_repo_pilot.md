# Pilote 5c sur dépôt Python réel, sans Docker

Préparé le 2026-10-10 : **trois réparations historiques de printers SymPy réellement exécutables**.
Smoke WSL : références **7 + 6 + 5 comparaisons de chaînes réussies**, chaque base échoue sur un cas
pertinent. Aucune requête modèle faite. Ce contrôle du harnais ne mesure ni agent ni essaim.
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
Whitelist exacte : dix ressources et **22 noms d'artefacts** par dossier
`results/repo_pilot_{cpu,compare}_AAAAMMJJ_NN` ; caches, smoke, locks, patches, voisins et sous-dossiers exclus.

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

À mesurer par le parent : un pilote agent solo, puis comparaison vote/cascade/référence, à provenance et
budgets déclarés. Aucun résultat modèle, score officiel ou preuve de réseau Myriad à ce jalon.
E11/chain5, ses résultats, Colab et les réglages du propriétaire n'ont pas été touchés.
Audit, fusion, push et export effectif sont laissés au parent.
