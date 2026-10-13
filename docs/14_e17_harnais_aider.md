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
