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
