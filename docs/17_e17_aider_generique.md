# E17 — adaptateur Aider Python générique

Cette préparation technique ne lit aucun exercice TEST. Elle ne change ni R00, ni l'agent, ses prompts,
ses budgets, ses modèles, sa sélection ou ses lanceurs. L'audit Astra parent précède toute évaluation.
Les trois DEV historiques conservent exactement leur notation RPC par défaut ; `generic=True` dans
`aider_python.verify` sert seulement à une comparaison explicite des chemins sur DEV autorisé.
Les autres exercices passent automatiquement par le nouveau chemin, sans table de noms d'API.

La configuration publique `.meta/config.json` fournit les fichiers `solution` et `test` au moment
de l'évaluation. Le contrôleur vérifie les chemins, les tailles, UTF-8, les fichiers réguliers sans liens,
la séparation sources/tests et un inventaire statique non nul des méthodes unittest. Il copie uniquement
les fichiers déclarés et les deux workers de confiance dans un dossier jetable. Les tests originaux
sont copiés octet pour octet ; leurs empreintes et celles des sources soumises accompagnent le résultat.
Aucun fichier de référence n'est requis par ce chemin.

Le worker importe les modules déclarés, vérifie leur origine puis charge les unittest publics originaux.
Les modules officiels sont enregistrés dans `sys.modules` avant exécution et collection, afin que
`setUpModule` et `tearDownModule` soient appliqués, y compris leurs erreurs qui interdisent le vert.
Fonctions, classes, état des objets, exceptions natives, imports entre plusieurs sources et packages
restent dans ce processus Python. La collection réelle doit correspondre exactement à l'inventaire
attendu : noms de module de test, classe et méthode. Les classes statiques dérivées de `unittest.TestCase`,
ses alias d'import et les héritages locaux sont pris en charge. Une collection dynamique, un helper
importé avec des méthodes de test ou une autre convention non reconnue bloque la notation sans repli.
Les solutions doivent être des modules Python importables, sans collision avec les deux workers réservés.

Le superviseur séparé et le validateur d'enveloppe réutilisent les mécanismes BigCode audités du commit
exact `4c86c0d`. Seul un paramètre optionnel de chemin du worker est ajouté au superviseur ; sa valeur
par défaut et le banc BigCode sont conservés. La sortie imprimée par le candidat n'est jamais un verdict.
Le pipe borné, le retour réel du worker, les compteurs et les méthodes terminées sont contrôlés.
La notation complète n'utilise pas fail-fast ; un skip, une erreur, un échec attendu ou une réussite
inattendue interdit le vert complet. Le résultat reste compatible avec `aider_agent.complete_green`.

Une assertion fausse, une exception du candidat, un import de solution échoué, une sortie prématurée,
un délai ou une sortie excessive produisent `graded/false`. Configuration, inventaire ou collection
invalide, runtime absent, démarrage du superviseur défaillant ou protection indisponible produisent
`error/null`. Un fichier de test importé qui manque une exportation d'un module de solution déclaré
est un échec candidat ; les autres erreurs de collection sont des erreurs d'infrastructure.
Le budget global de vérification, les bornes de sortie et le nettoyage restent stricts.

Ce chemin exige Linux x86-64, Landlock ABI ≥ 3, les namespaces et seccomp via `LocalLinuxEnv`, sans repli
vers le backend à audit Python. Il note des bancs publics non adversariaux :
`trust_scope=public-benchmark-nonadversarial`, `adversarial_attestation=false`.
Le candidat partage le processus des unittest et pourrait délibérément altérer leur fonctionnement
ou inspecter le pipe. Le contre-exemple historique BigCode reste conservé. Aucune attestation
cryptographique de Python arbitraire, ni score scientifique, n'est revendiqué.

Validation hermétique, sans installation ou réseau, dans `phase0` :

```bash
python -m unittest tests.test_aider_generic tests.test_aider_agent tests.test_code_pilot -q
```

Sous Windows, les nouveaux tests d'exécution stricts ont seulement le skip naturel de plateforme.
Sous WSL, utiliser le runtime code-pilot existant ; aucune protection absente n'est remplacée par un skip.
Les fixtures sont synthétiques : fonction, classe à état, plusieurs sources/tests, package, erreurs,
faux succès imprimé, délais, sortie excessive, nettoyage des enfants et copies synthétiques des trois DEV.
Les caches et tests DEV réels restent inchangés. Les suites, codes et preuves closes exactes sont dans
la livraison privée destinée au parent. L'export public est uniquement simulé sur ces sources autorisées.
