# E17 — correctif ciblé du harnais Aider

Jalon du 2026-10-13 : corriger le défaut de boucle du pilote, avant toute nouvelle inférence.
Les trois exercices déjà vus (affine-cipher, beer-song, book-store) servent uniquement au développement.
Aucune tâche TEST n'est consultée pour ce correctif ; aucun nouveau résultat de modèle n'est produit.
Le registre de révisions DEV et le lanceur entièrement sur VM seront préparés avec le parent après audit.

## Boucle et commande officielle

`run_code_pilot.py` passe maintenant par `essaim/aider_agent.py`, sans modifier `essaim/agent.py`.
Une réponse contient exactement un bloc bash. Le modèle peut lire les sources, les modifier et exécuter
des commandes dans le bac existant. Pour tester une soumission, il utilise ce bloc seul :

```bash
MYRIAD_RUN_OFFICIAL_TESTS
```

Ce nom est une action du contrôleur, pas un programme installé dans le bac. Une commande composée,
un `echo` de ce nom, un lancement du squelette ou un pytest candidat ne constitue pas ce test.
Le contrôleur lit un instantané des seuls fichiers solution autorisés, puis appelle `aider_python.verify`.
Les assertions officielles originales restent dans le contrôleur ; chaque appel candidat repart d'une
copie source neuve dans `LocalLinuxEnv`. Les tests visibles modifiés et les scores imprimés sont ignorés.
Le feedback indique couverture, échecs, identifiant de test, traceback borné et raisons d'échec candidat.
Les verdicts détaillés du vérificateur et les empreintes des sources testées restent dans le journal.

Après un échec, le modèle reçoit ce retour, corrige les sources et redemande un test. Une soumission
immédiate est refusée avec la commande à utiliser ; elle consomme une étape, sans arrêter la boucle.
La sentinelle historique `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` reste la commande de soumission.
Elle n'est acceptée qu'après un test officiel vert complet des sources actuelles : nombre de tests positif,
tous les tests exécutés, appels candidats complets, aucune panne ni erreur du vérificateur.
Le contrôleur compare le contenu des fichiers solution après chaque commande suivant un vert et lors
de la soumission. Une modification observée invalide ce vert, même si une commande ultérieure restaure
le contenu précédent. Un changement pendant le test empêche aussi l'enregistrement d'un vert.

La vérification finale est conservée, indépendamment des tests de la boucle. Une réussite d'épisode
exige à la fois une soumission autorisée, les mêmes sources et une nouvelle vérification complète verte.
Sans soumission autorisée, `passed=false` même si ce dernier diagnostic réussit ; `verifier.passed` reste
le diagnostic brut. Une erreur d'infrastructure garde `status=error, passed=null`, sans rapport final.
Des sources finales invalides restent un échec candidat ; des sources initiales invalides restent une
erreur de préparation avant le premier appel modèle.

## Budgets et sorties incomplètes

Les nouveaux défauts Aider sont 4 096 jetons par réponse et 49 152 plafonds cumulés par épisode.
`--max-tokens` et `--total-tokens` permettent de les fixer explicitement avant la campagne ; aucune
augmentation automatique selon les scores. Le plafond d'une requête est le minimum du plafond choisi
et du budget restant. Chaque appel, pair ou référence, réussi ou échoué, réserve ce plafond sans remboursement.
L'usage observé et les réservations restent distincts ; le raisonnement retourné compte selon l'usage serveur.
Le serveur doit fournir `finish_reason=stop` pour qu'une réponse puisse devenir une commande.
Une réponse `length`, filtrée ou sans raison de fin est conservée et comptabilisée, mais jamais exécutée,
même si elle contient un bloc complet. Le modèle est invité à proposer une modification plus courte.
En cascade, une réponse incomplète est traitée comme une proposition mal formée ; la référence est vérifiée aussi.

Les autres défauts restent 20 étapes, 900 s d'épisode, 30 s de commande et 120 s par vérification.
Les tests dans la boucle consomment le temps de l'épisode et une étape ; leur délai est borné par le
minimum du budget de vérification et du temps restant. Un vert arrivé après ce délai n'est pas accepté.
Le dernier diagnostic possède son budget séparé de vérification, comme historiquement ; il ne transforme
pas un arrêt par délai ou budget en réussite. Un timeout de commande (`124`) ou une sortie dépassant le
plafond de collecte (`125`) arrête l'épisode sans réussite. La réduction du feedback à 4 000 caractères
garde un marqueur de coupure ; elle ne crée aucun verdict. Les limites d'isolation ne sont pas assouplies.

`single`, `vote`, `cascade` et `reference` partagent ces règles et ces budgets. La pluralité et le départage
restent ceux d'`agent.py`. Le manifeste des nouvelles campagnes porte `code-pilot-e17-v1`, le nouveau
prompt et l'empreinte d'`aider_agent.py` ; une reprise historique n'est donc pas silencieusement réinterprétée.
Les rapports hors inférence de campagnes anciennes restent lisibles. Les autres backends, notamment SymPy,
gardent leur boucle historique et leurs défauts 1 024 / 12 288 ; aucun protocole historique n'est réécrit.

## Limites réelles et suite VM

Cette adaptation note seulement les trois API pures déjà prises en charge, avec les tests publics originaux.
Elle n'est ni le score officiel Aider sur 225 exercices ni une preuve sur dépôt logiciel général.
Les tests sont publics et les valeurs attendues peuvent apparaître dans le feedback DEV. Le transport JSON
des résultats candidats reste celui du vérificateur existant : il ne constitue pas une attestation inviolable
contre un programme qui connaît le worker et tente de fabriquer ses résultats. La séparation des assertions,
les copies neuves et le bac restent utiles pour noter ce banc académique, sans garantie générale inventée.
La comparaison des sources porte sur leur contenu aux points de contrôle, pas sur chaque écriture intermédiaire
d'une commande. Les limites noyau, mémoire/disque non agrégés et runtime du guide historique restent applicables.

Les tests de régression utilisent chat, environnement et verdicts simulés ; ils ne mesurent aucun modèle.
L'audit gpt-6-astra est confié au parent après le seul commit de ce jalon. Avant une exécution réelle,
il faudra vérifier les protections sur la VM et refaire le smoke officiel, sans assouplissement si Colab
ne permet pas les namespaces ou Landlock requis. L'infrastructure actuelle `colab_jobs.py` ne possède
pas encore de plan E17 et le superviseur de campagne vise `aa-1` : ce correctif ne lance rien.

La suite préparera un registre conservant chaque révision, ses sources/prompts/budgets, tâches DEV et
traces, puis figera le harnais avant une seule mesure sur TEST jamais consulté. Le serveur, le contrôleur,
la boucle agent et les tests seront tous sur la VM ; le PC transmettra seulement le code et récupérera
les résultats. Allocation, contrôle du solde, récupération et libération resteront au parent, après audit.

## Sonde Colab avant modèles

Le chemin `colab_phase0.sh e17-probe` prépare **une seule tentative L4**, sans `colab_jobs.py`,
modèle, serveur, jeu de données ou installation de paquet. Le développeur ne lance que la CLI factice ;
le lancement réel reste au parent après audit. Aucun résultat WSL ne vaut observation de compatibilité Colab.
Le wrapper appelle un superviseur dédié, sous le verrou de cycle de vie partagé avec les autres opérations.
Un reçu déjà présent, une session existante, un compte illisible/alloué ou un budget incompatible
interdisent `new`. Un `new` ambigu n'est jamais réessayé et n'autorise aucun arrêt d'une identité inconnue.

Défauts configurables : préparation/sonde au plus **600 s**, budget **4 unités**, plancher **15**,
nettoyage **1 800 s**, tarif plafond **5,3 unités/h**, lecture **120 s**, grâce **5 s**, relevé pendant
les appels distants toutes les **30 s**. La marge comptable vaut lecture + deux grâces + 31 s, soit **161 s**.
L'enveloppe conservatrice est donc `5,3 × (600 + 1800 + 161) / 3600 = 3,770361…`, inférieure à 4.
Réserver 4 exige au moins 19 unités au préflight. Le relevé historique 59,62 / zéro allocation est une
information passée, **à revérifier au lancement parent** ; il n'est pas inclus comme observation nouvelle.
Un tarif supérieur au plafond, un solde devenu inconnu ou une réserve atteinte entraîne la clôture.

Le petit programme distant autonome mesure noyau/Linux/x86_64, ABI Landlock et les namespaces
user/mount/net/pid/ipc/uts séparément puis ensemble, avec les drapeaux exacts de `LocalLinuxEnv`.
Il installe aussi le filtre seccomp inchangé dans un processus jetable et vérifie le refus de socket.
Enfin, il importe la source exacte de `local_linux.py` dans un répertoire temporaire, crée un runtime
dédié dont `bin/python` pointe vers `/usr/bin/python3`, puis exécute dans `LocalLinuxEnv` uniquement
`assert 2 + 2 == 4` et une sentinelle. Aucun assouplissement du bac, aucune source scientifique transmise.
Les fichiers temporaires sont supprimés par le contrôleur distant ; la VM est libérée ensuite.

Le JSON récupéré par `exec -f` doit avoir un accusé unique, la campagne, l'identité de session/allocation,
un nonce frais et les empreintes des sources exécutées (UTF-8 avec fins de lignes normalisées à la lecture).
CLI 0 seul, accusé manquant/périmé, faux verdict compatible ou isolement incomplet sont refusés.
`compatible` exige toutes les observations positives et le reçu complet du vrai `LocalLinuxEnv`.
`incompatible` décrit un refus observé des exigences ; `infrastructure-unknown` décrit une observation
impossible, un timeout, une panne réseau ou une clôture non confirmée. Un smoke retournant 124 reste
`infrastructure-unknown`, même avec un reçu d'isolation établi. Une lecture périodique pendant un exec
garde sa borne propre sans réduire durablement le délai global, sauf réduction liée au budget du compte.
Les valeurs machine et noyau restent dans le JSON. Cette sonde ne prouve ni sécurité générale contre un candidat hostile ni qualité d'un modèle.

Une erreur/TERM après `new` confirmé, même pendant une sauvegarde avant ou après remplacement atomique,
conserve l'autorité en mémoire : réconciliation bornée, arrêt de la seule identité épinglée, absence explicite,
puis relevé final obligatoire. La réception de TERM/INT est différée pendant `new` jusqu'à l'enregistrement
en mémoire de son retour. Ce masque différé reste dans le contrôleur : TERM/INT sont débloqués dans
l'enfant avant l'exécution de safe_cli, afin que son gestionnaire de signal puisse arrêter le groupe du CLI officiel
sur timeout externe. Les signaux répétés sont ignorés pendant le nettoyage borné. Le snapshot impossible
n'empêche pas la clôture. Une identité ou un reçu remplacé, ou une campagne préexistante détectée, interdit
tout arrêt étranger. Les échecs d'arrêt sont réessayés ; après absence, seules les lectures du compte reprennent.
Le reçu n'est supprimé qu'après relevé final durable. Le JSON final distingue observation brute, erreur,
absence et clôture ; seul `compatible` avec clôture complète rend 0, tout autre résultat rend 1.

Commande portable **réservée au parent**, depuis Git Bash, chemins WSL fournis par variables :

```bash
MSYS_NO_PATHCONV=1 wsl.exe -e bash -lc '
  repo=$1; evidence=$2
  mkdir -p "$evidence"
  bash "$repo/phase0/colab/colab_phase0.sh" e17-probe \
    --output "$evidence/probe.json" \
    --seconds 600 --budget-units 4 --min-balance-units 15 \
    --cleanup-seconds 1800 --max-rate-units-hour 5.3
' bash "$REPO_WSL" "$PROBE_WSL"
```

Le reçu partagé par défaut reste sous `$HOME/.local/state/myriad-colab/ownership.json` ; `--receipt`
permet un chemin explicite, à partager avec `DLLM_CAMPAIGN_RECEIPT` pour les autres opérations.
Ne pas créer plusieurs chemins de verrou pour allouer en parallèle. Un résultat négatif suffit : arrêter
après nettoyage, sans boucle d'allocations. Si la clôture échoue, conserver reçu/JSON et faire vérifier le
parent ; `DLLM_CAMPAIGN_RECEIPT=… bash colab_phase0.sh down` peut reprendre une propriété complète ou
la comptabilité après absence. Une allocation ambiguë exige une réconciliation explicite par le parent,
jamais une adoption automatique. SIGKILL, panne du PC ou du stockage peuvent interrompre le superviseur :
le reçu durable reste alors une preuve de responsabilité, pas une preuve de libération.

Les tests locaux emploient seulement la CLI factice et des observations synthétiques : compatible,
incompatible, faux succès/accusé absent, sessions/campagnes étrangères, budget/solde, timeout/TERM,
persistance des transitions après `new`, répétition de `stop`, absence et compte final. Aucun Colab réel
ni processus du bac distant n'est exécuté par ces tests.
