# Pilote sur dépôt Python réel

Préparé le 2026-10-10, sans modèle : trois réparations historiques de printers SymPy,
`sympy__sympy-11400`, `sympy__sympy-11897`, `sympy__sympy-12171`.

`tasks.json` épingle les dépôts, fichiers, références de PR, critères et exclusions. `final_cases.json`
contient les 18 assertions de chaînes du contrôleur, avec les licences BSD historiques dans `licenses/`.
Ces ressources ne sont jamais copiées dans l'espace agent. Les références téléchargées et les tests
finaux restent dans un cache séparé, hors Git. Le candidat reçoit le vrai dépôt au commit de base,
ses tests historiques et une liste explicite de sources soumises ; aucun `.git` ni historique.

`requests_diagnostic.json` conserve l'examen précédent : six tâches Requests exclues pour Chardet LGPL.
L'annonce MIT code et données de SWE-bench reste citée ; aucune issue HF n'est republiée.
Les consignes sont originales et décrivent les réparations historiques, sans retirer du code.

Le [guide](../../docs/13_repo_pilot.md) donne préparation, smoke, commandes de passerelle et limites.
Ce protocole local ne produit pas un score SWE-bench officiel ni une couverture PASS_TO_PASS complète.
Les trois références ont passé 7 + 6 + 5 cas sous WSL ; les trois bases ont échoué. Aucune inférence.

Tests visibles : runner natif `python bin/test --no-subprocess --no-colors --seed 0`, sur le fichier
C/Mathematica concerné, et `test_latex.py -k test_latex_Piecewise` pour LaTeX ; commandes exactes dans le
guide. Le smoke vérifie les six copies complètes : 30/30 et 9/9 sur base/référence, 1 réussite LaTeX sur
base et 1 assertion historique incompatible avec le rendu corrigé sur référence. Ce diagnostic reste
distinct de la note finale (18 chaînes attendues dans le contrôleur) ; pytest moderne n'est pas conseillé.
