# Supervision bornée E16

Cette entrée prépare uniquement `e16-1` ou `e16-mc`, toujours sur A100. L'ancien `up` supervisé
reste réservé à `aa-1`. `campaign.state` accepte explicitement ces trois plans et exige leurs
jobs exacts ; le cycle allocation/clôture E16 utilise `Probe`, son verrou partagé, les helpers
de compte, le transfert audité et les délais de groupes de processus existants.

Le parent lance après son audit, la fusion et ses tests. Depuis Git Bash, définir `REPO_WSL`,
`PREFLIGHT_JSON` et `EXECUTION_JSON` avec les chemins WSL du dépôt et de deux nouveaux rapports :

```bash
export MSYS_NO_PATHCONV=1
wsl.exe -e bash -lc 'cd "$1" && bash phase0/colab/colab_phase0.sh e16-supervise --output "$2"' e16 "$REPO_WSL" "$PREFLIGHT_JSON"
wsl.exe -e bash -lc 'cd "$1" && bash phase0/colab/colab_phase0.sh e16-supervise --execute --output "$2"' e16 "$REPO_WSL" "$EXECUTION_JSON"
```

Pour l'admission autorisée depuis le solde brut **59,62**, conserver la réserve antérieure
non réconciliée **1,15 unité** dans les deux commandes, après audit parent :

```bash
wsl.exe -e bash -lc 'cd "$1" && shift && bash phase0/colab/colab_phase0.sh e16-supervise "$@"' \
  e16 "$REPO_WSL" --prior-reserve-units 1.15 --output "$PREFLIGHT_JSON"
wsl.exe -e bash -lc 'cd "$1" && shift && bash phase0/colab/colab_phase0.sh e16-supervise "$@"' \
  e16 "$REPO_WSL" --execute --prior-reserve-units 1.15 --output "$EXECUTION_JSON"
```

`--prior-reserve-units` accepte un nombre fini non négatif, défaut 0 compatible. Il ne
change pas le seuil de choix de plan, calculé sur le solde brut : ici `e16-1`. Le budget
plafonné devient **43,47**, le minimum protégé **16,15 = 15 + 1,15**. La politique du reçu
garde cette réserve pendant l'exécution et les reprises du compte. Les relevés restent
bruts ; la réserve n'est ni soustraite du solde enregistré ni ajoutée à la consommation
de campagne. Ce n'est pas un coût exact. Garder le paramètre tant que les charges différées
ne sont pas réconciliées ; aucune réconciliation automatique n'est déduite d'un compteur.

La première commande lit uniquement sessions/usage côté cloud, imprime le choix et le coût,
vérifie les références existantes puis prépare temporairement le paquet et ses empreintes.
Elle ne télécharge aucun modèle/jeu et n'alloue rien. La seconde refait l'admission avant son
unique `new` ; elle n'utilise pas un ancien relevé pour décider. Aucun autre plan n'est lancé.
Chaque rapport et ses compagnons doivent avoir un nom neuf. Avant tout appel CLI, leurs
chemins résolus sont comparés au reçu, verrou, archives et journal de compte ; les dossiers
`e16-<identité>` de récupération sont aussi réservés, y compris pour des chemins relatifs. Le reçu par défaut reste celui
partagé avec les autres campagnes ; `--receipt` sert surtout aux tests hermétiques.

La règle de choix est figée : solde ≥ 45 → `e16-1`, sinon `e16-mc`, avant allocation.
L’admission borne le budget au minimum de l’enveloppe haute prudente et du solde moins
15 unités. Elle réserve 1 800 s de clôture, 161 s de marge CLI/compte et 900 s d’analyse et
récupération, au tarif plafond inchangé de 5,3 unités/h. Elle refuse un compte inconnu ou
invalide et toute absence de fenêtre scientifique positive après ces réserves.

| Plan choisi | Estimation scientifique | Enveloppe haute prudente |
| --- | --- | --- |
| `e16-1` | 5,5–7,6 h ; environ 29–40 unités | 44,50 unités |
| `e16-mc` | 4–5,6 h ; environ 21–30 unités | 33,90 unités |

Avant allocation, le rapport imprimé expose les estimations basse/haute, le budget, la
fenêtre scientifique disponible et le risque explicite d’incomplétude lorsque la projection
haute ne tient pas. À 45 unités, `e16-1` dispose d’un budget de 30 unités ; à 44,9, `e16-mc`
de 29,9 ; à 59,62, `e16-1` conserve l’enveloppe haute de 44,50. Ces valeurs sont des exemples
synthétiques, pas de nouveaux relevés. Tous les jobs du plan sont conservés, sans substitution
ni raccourcissement silencieux. Aucune estimation ne promet une campagne finie : expiration,
panne ou récupération trop lente donnent une tentative incomplète avec récupération des
partiels et clôture, jamais un succès scientifique. Pendant toute opération longue, usage
est contrôlé toutes les 30 s ; débit > 5,3, compte inconnu ou réserve atteinte déclenchent
la clôture bornée. Le plancher de 15 unités reste réservé avant, pendant et après.

## Références et analyse

Le préflight exige les JSONL et leurs manifestes Q8 `_colab` des 12 modèles de `SPEED_Q8`,
sur GSM8K, MATH-500, ARC et MMLU-Pro, dev/test. Pour `e16-1`, il exige en plus les générations
E11 des sept `analyze_e4.FAMILIES` sur HumanEval+/MBPP+, dev/test, les quatre fichiers
`codeexec_e11_colab_*`, leurs manifestes, `e11_summary_colab.json` et `e11_validation_colab.json`.
Les noms viennent des tables existantes. Sur le PC, ces octets existants sont seulement copiés
et empreintés ; aucune analyse E16 ni lecture scientifique de TEST n'y est ajoutée.

Le manifeste `e16_sources.json` lie la campagne, le plan, HEAD, chaque source envoyée et chaque
référence. La VM revérifie ces empreintes avant collecte, puis vérifie les identifiants exacts
des partitions via les loaders existants, les cinq échantillons code, tous les programmes
distincts et références exécutés, les pins de poids et les 24 entrées vitesse sur A100.
L'analyse existante tourne exclusivement sur la VM via `uv --group paper`, avec `--no-code`
pour `e16-mc`. Sa sortie doit contenir les choix et, pour `e16-1`, le rapport code.
Le protocole, les quantifications, les jobs, les adaptateurs et les résultats E11 restent ceux
du dépôt. Aucune mise à jour de provenance E11 n'est nécessaire ici : `colab_jobs.py` est intact.

## Récupération et clôture

Une récupération périodique toutes les 600 s et une récupération finale/à l'échec copient côté
VM les artefacts dans un répertoire scellé par nonce. Bornes : 64 Mio/fichier, 512 Mio au total,
512 entrées, accusé décompressé ≤ 256 Kio. L'accusé compressé évite la limite stdout 65 536
octets de `safe_cli` ; les données passent par `download` vers des fichiers, puis taille et
SHA-256 sont vérifiés. Le contrôleur vérifie l'identité de session avant chaque mutation/lecture
distante. Aucun contenu généré n'est imprimé pour la supervision locale.

Chaque récupération crée un dossier distinct `e16-<campagne>/partial-<nonce>` ou
`final-<nonce>`, près du rapport. `seal.json` décrit les octets ; `verified.json` existe uniquement
après validation de tous les fichiers et de la provenance. `recovery_attempts` distingue les
dossiers incomplets des `recovered` vérifiés. Aucun résultat historique n'est remplacé.
Les journaux et snapshots sont des observations prises à cet instant ; un journal du lanceur
peut encore être ouvert au moment du scellement. Leur empreinte atteste la copie, pas une
promesse de clôture de tous les flux distants. Les résultats scientifiques doivent être complets.

Succès exige tous les jobs exacts, les artefacts complets, l'analyse VM, la récupération vérifiée,
l'absence de notre VM et le compte final lisible sans affectation/débit. Le contrôleur conserve
l'autorité mémoire après `new` confirmé, même si les marqueurs returned/owned échouent ou si
TERM/INT arrive ; il ne rejoue jamais `new`. Un échec bootstrap historique avec notre `campaign_id` et sans `plan` reste une initialisation
incomplète : sa récupération peut échouer, mais elle ne retire pas l’autorité de clôture. Un
`plan` explicitement étranger ou un `campaign_id` étranger interdisent toujours l’arrêt.
Allocation ambiguë, session/remplacement étrangers
et campagne préexistante interdisent l'arrêt. Une clôture non confirmée garde le reçu.

Pour reprendre exclusivement une clôture avec reçu E16 owned durable, utiliser un nouveau
rapport et la même identité de reçu, sans allocation :

```bash
wsl.exe -e bash -lc 'cd "$1" && bash phase0/colab/colab_phase0.sh e16-supervise --cleanup-only --output "$2"' e16 "$REPO_WSL" "$CLEANUP_JSON"
```

Après absence confirmée, le reçu `released/accounting-pending` permet seulement la reprise
comptable ; aucun nouveau stop ni sessions n'est nécessaire. Un reçu pending/ambigu n'est pas
adopté : il faut examiner les preuves, sans arrêt automatique d'une VM potentiellement étrangère.
Les pannes de stockage et SIGKILL ne garantissent pas une clôture autonome après mort du
contrôleur ; le reçu et les preuves restent nécessaires au parent. Aucun repli du bac E11 n'est
introduit par cette supervision.

## Lecture du compte : décision du 13 octobre à 12 h 40

Les commandes ci-dessus utilisent automatiquement la garde commune E16/E17 et `chain4`.
Seul un timeout de lecture (`usage`, retour 124) autorise la reprise : pauses de 1, 2, 4, 8,
16 puis 30 s au maximum (base `retry-seconds` pour Probe/E17). Les lectures, pauses et délai
de terminaison des processus appartiennent à une seule fenêtre de **600 s maximum** par
lecture interrompue. Cette fenêtre se termine plus tôt à la borne globale ou à la réserve
budgétaire, **pas à la courte borne du RPC englobant**. La première lecture peut être coupée
par cette borne ; les suivantes disposent encore de la fenêtre compte. Le RPC expiré est
terminé/récolté pendant cette attente et sa réponse tardive n'autorise aucun succès.
La surveillance partagée reste active dans chaque lecture et pause : elle consulte la borne
RPC au plus toutes les secondes, avec une attente raccourcie jusqu'à son échéance exacte.
TERM puis KILL/récolte conservent la grâce bornée ; cette grâce consomme la fenêtre compte,
sans repousser les échéances ni rendre admissible un résultat RPC tardif. Aucun thread n'est ajouté.
Après compte frais et budget/propriété revalidés, E17 peut réémettre une seule fois `status`,
lecture idempotente ; ni `new`, ni `start`, ni épisode ne sont rejoués. Une interruption TERM
pendant l'attente bloque la récupération et ne permet que la clôture propriétaire.
Le dernier solde connu est diminué prudemment au tarif plafond pendant l'attente ; les
réserves de cleanup et de marge restent soustraites, avec un plancher absolu de **15 unités**
plus toute réserve antérieure explicitement configurée. Le temps d'attente reste dans
l'enveloppe globale ; les réserves cleanup et marge ne sont pas prolongées.

Aucun nouveau RPC scientifique n'est lancé pendant le compte inconnu. Solde insuffisant,
forme invalide, affectation étrangère ou erreur CLI autre que 124 déclenchent immédiatement
la clôture ; ils ne bénéficient pas de cette attente. Une fenêtre épuisée interdit aussi la
récupération scientifique E16, même dans sa réserve de récupération. Les vérifications de
propriété nécessaires à l'arrêt restent autorisées. `new` n'est jamais réessayé.

Après épuisement, arrêt répété uniquement de notre session, absence puis compte final valide
sont requis avant suppression du reçu. La clôture garde sa borne de 1 800 s par défaut :
une panne durable conserve le reçu et un état explicite (`cleanup_blocked=true` ou
`released/accounting-pending` après absence), avec un retour d'échec ; ce n'est
pas une garantie de libération en cas de réseau indisponible ou de mort du contrôleur.
Reprendre uniquement avec la commande `e16-supervise --cleanup-only` ci-dessus, vers un
rapport neuf. Pour un reçu E17 ou `chain4` identifié et examiné, la reprise commune est :

```bash
wsl.exe -e bash -lc 'cd "$1" && bash phase0/colab/chain4.sh --cleanup-only --receipt "$2" --cleanup-seconds 1800' cleanup "$REPO_WSL" "$RECEIPT_WSL"
```

Ne pas supprimer un reçu bloqué, adopter une session étrangère ou allouer de nouveau pour
contourner la panne. Ces chemins sont validés avec horloge/CLI synthétiques seulement ;
aucune stabilité réseau Colab ni nouvelle mesure scientifique n'est attestée.

Le minimum E16 est calculé une fois (`15 + prior_reserve_units`) puis le disponible est
`balance_units - minimum`, identique au contrôle strict du préflight. Il n'y a aucune epsilon
abaissant le plancher. À brut 59,62 et réserve 2,3, le minimum vaut 17,3 et le plafond environ
42,32 ; la fenêtre scientifique reste arrondie vers le bas à la microseconde.
