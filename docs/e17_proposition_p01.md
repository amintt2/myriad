## Proposition P01

**Décision : aucune révision supplémentaire du harnais avant le premier DEV sous R00 (`067cf41`).** Enregistrer P01 comme proposition sans modification, avec cette justification ; ce n’est ni un audit favorable ni une validation expérimentale.

Les traces historiques établissent trois échecs, avec sources finales inchangées :

- **affine-cipher** : première réponse mal formée, puis lancement du squelette avec retour zéro, interprété comme une réussite avant soumission ;
- **beer-song** : sept réponses achevées tronquées à 1 024 jetons, aucune commande exécutée, puis expiration pendant le huitième appel ;
- **book-store** : soumission immédiate sans modification ni test.

**R00 traite déjà ces mécanismes** : plafonds 4 096/49 152, rejet des réponses incomplètes, action officielle de test, feedback pour correction, refus de soumettre sans vert complet des sources actuelles, invalidation après modification et vérification finale indépendante.

Il reste à mesurer si le modèle utilise effectivement cette boucle, produit des modifications complètes et atteint une soumission valide dans les budgets. Augmenter encore les plafonds, enrichir le prompt ou automatiser davantage la boucle maintenant brouillerait cette première mesure sans preuve DEV supplémentaire.

## Critères DEV d’acceptation

Avant inférence, sur VM : vérifier les protections requises et réussir le smoke officiel avec références correctes et candidats incorrects. Un échec technique bloque la mesure ; aucune protection n’est assouplie pour obtenir un score.

Premier bloc : **les trois DEV historiques × quatre modes**, une exécution par cellule, température 0, seed 0, budgets R00 conservés : 20 étapes, 900 secondes d’épisode, 30 secondes par commande, 120 secondes par vérification, diagnostic final séparé.

Distinguer deux conclusions :

- **Validité mécanique** : aucune réponse tronquée exécutée ; aucune soumission acceptée sans vert complet actuel ; aucun vert conservé après modification observée ; erreurs d’infrastructure séparées des échecs candidats.
- **Utilité empirique** : publier réussites sur trois, épisodes ayant modifié les sources, tests officiels demandés, corrections après échec, soumissions refusées, troncatures, arrêts, appels, jetons observés/réservés et temps.

Un nouveau 0/3 reste publiable : il ne justifie ni remplacement des tâches ni hausse automatique du budget. Une réussite supplémentaire serait un signal descriptif, pas une preuve de généralisation. La comparaison historique CPU contre nouvelle VM ne permet pas d’attribuer les différences de vitesse à R00.

Pour une éventuelle révision ultérieure : comparaison appariée avec la version précédente sur le même DEV et la même configuration. Acceptation préfixée : invariants conservés, aucune réussite perdue, au moins une réussite gagnée, dans les plafonds inchangés ; sinon conserver la version précédente. Ce seuil sert à décider, pas à démontrer une significativité.

## Protocole/gel/test

**Plafond fixé maintenant : trois propositions au total, P01 comprise**, donc au maximum deux candidates modificatrices après R00 : P02 sur DEV Aider, puis P03 sur DEV BigCodeBench. Une proposition invalide ou rejetée consomme son emplacement. Aucun prolongement après lecture des scores.

Le registre conserve intégralement : proposition et motivation, acceptation/rejet/invalidation, sources consultées, parent et empreintes, différences de code/prompt, modèles, budgets, identifiants DEV, toutes les tentatives, erreurs, coûts et décision. Les correctifs techniques ne constituent pas une voie de révision illimitée ; une modification pertinente pour le comportement doit être enregistrée.

Configuration proposée :

- `single` : Qwen3.5-2B Q8_0 historique, sous réserve de correspondance au catalogue existant ;
- `vote` : ce modèle et un second petit modèle distinct du catalogue ;
- `cascade` : les mêmes pairs, avec Qwen3.8-27B ;
- référence : **Qwen3.8-27B seul**.

Le parent épingle avant score le second modèle, les artefacts, quantifications, ordre des pairs et paramètres serveur. Le catalogue n’étant pas dans les lectures autorisées, je ne certifie pas ici ses entrées. Aucun modèle nouveau ni substitution après résultats.

Deux pairs limitent la dépense : 49 152/4 096 donnent **douze appels maximaux**, donc six tours de vote ou quatre tours si chaque cascade appelle aussi la référence. Mesurer cette contrainte avant de proposer sa modification. Conserver pluralité, départage et déclenchement de cascade existants.

Après Aider : **deux tâches BigCodeBench DEV**, choisies par identifiants selon une règle déterministe avant ouverture, chacune dans les quatre modes. Leur adaptateur devra avoir été rendu opérable ; sa disponibilité n’est pas établie ici.

**L’adaptateur actuel ne couvre que les trois API historiques.** L’étendre est indispensable pour TEST inédit : généralisation validée sur DEV et fixtures synthétiques, sans adaptation aux contenus TEST. Ne jamais rebaptiser les trois exercices actuels « TEST ».

Avant nouveaux scores, réserver par métadonnées **deux tâches Aider inédites et deux BigCodeBench inédites**, disjointes des DEV. Après développement, geler code, adaptateurs, prompts, modèles, budgets, sélection et analyse. Exécuter chaque cellule TEST une seule fois : seize épisodes. Aucun retour TEST au proposeur pour révision ; interruption conservée comme incomplétude, sans rejeu opportuniste.

## Ressources et limites

Répartition proposée des **59,62 unités** :

| Affectation | Unités |
|---|---:|
| Plancher absolu | 15 |
| E17, préparation et clôture comprises | 20 |
| verif-1 | 10 |
| E16 | 8 |
| E12 | 5 |
| Marge non engagée | 1,62 |

À **5,3 unités/heure, tarif historique à revérifier**, E17 représente environ 3 h 46, préparation et libération incluses. La grille sans révision compte 36 épisodes ; elle n’est **pas garantie** dans cette enveloppe.

Commencer par les trois DEV `single`, mesurer durée et coût, puis n’admettre les blocs suivants que si leur projection prudente tient dans le reliquat, avec réserve explicite pour TEST et clôture. Sinon déclarer l’exploration incomplète et préserver les priorités suivantes. Solde contrôlé avant, pendant et après ; compte inconnu : arrêt et clôture ; libération confirmée.

Serveurs, modèles, agent, téléchargements et tests tournent sur VM. Le PC transmet uniquement le code léger et récupère les résultats. Plusieurs modèles sur un GPU constituent une **simulation centralisée d’essaim**, sans preuve de réseau distribué.

L’étude s’inspire de *Harness Learning Enables Generalizable Test-Time Adaptation*, arXiv 2609.35738v1 ; Astra reste figé, sans RL : aucune reproduction de leur entraînement revendiquée. Le bac note des bancs publics, sans attestation cryptographique de Python arbitraire. Ce pilote ne prouve pas un agent sur dépôt réel.

**Lectures effectuées :** `CLAUDE.md`, journal du plan dont la dernière entrée du 13 octobre ; `docs/12_code_pilot.md`, `docs/14_e17_harnais_aider.md` ; rapport et trois JSONL historiques autorisés ; `aider_agent.py` et passages pertinents de `agent.py`, `aider_python.py`, `code_pilot.py`, `local_linux.py`, sous `phase0/essaim/`. Aucun contenu TEST, `data/`, réseau, test exécuté, sous-agent ou fichier écrit.