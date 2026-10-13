# VERIF-1 — amendement local A2–A5

Enregistré le 10 octobre 2026 AVANT toute nouvelle génération ou simulation. Registre parent complet
`docs/12_percees.md` : A2 a352038, A3 cb07e83, A4/A5. Aucun choix motivé par les observations CPU post-A3.
V01/V02 inchangés, V07 indépendant dans un autre worktree. Aucun cloud/GPU réel ni réponse test ouverte.

## Choix fixés avant génération

DEV : snapshot E11 56 fichiers, 81 HumanEval+ et 189 MBPP+ après exclusion historique HumanEval/32.
35 candidats existants, sept modèles × cinq échantillons, sans régénération. Préparation CPU lit uniquement
82/189 lignes dev des caches mixtes, jamais le suffixe. Projection publique stricte : contrat, imports,
signature/docstring HumanEval avec corps supprimé, exemples publics MBPP. Ni solution, référence, labels
ni tests cachés ne passent dans le paquet ou les prompts auteurs.

Ordre catalogue Qwen, Gemma, Granite, SmolLM, Mistral, Phi, OLMo. Révisions/GGUF/SHA E11 inchangés.
Deux appels auteurs de 512 tokens, contrat seul : paramètres natifs `temperature=0.7, top_p=0.95`, seeds
20261010/20261011, `enable_thinking=false`, contexte8192, un slot. Juges `temperature=0.0, top_p=1.0`,
seed20261010, 256 tokens. Templates natifs GGUF, llama.cpp b11505/ff5888f99, GPU intégral attesté.
Précision native avant GPU : `top_k=0, min_p=0.0, repeat_penalty=1.0` pour les deux rôles ; aucun autre
réglage de sampler envoyé, ordre des samplers du build épinglé. Requête complète conservée par empreinte.
Dédupliquer cas/propriétés identiques au sein d'une famille avant score ; facturer les deux appels entiers.
Aucun retry de format/troncature. Licences Apache-2.0 sauf Phi MIT ; cartes officielles dans le catalogue,
notamment [Phi](https://huggingface.co/microsoft/Phi-4-mini-instruct) et
[OLMo](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct), vérifiées par le parent le 10 octobre.
Pas de trust_remote_code.

Données JSON bornées, max quatre cas/quatre propriétés par appel. Relations déclaratives `equal`,
`length_equal`, `permutation`, deux entrées explicites et justification du domaine ; aucune expression Python.
Comparaison des cas et `equal` : JSON canonique strict (1 et 1.0 distincts). `length_equal` exige le même
type liste/chaîne/dictionnaire et même longueur ; `permutation` compare les multiensembles de listes,
doublons conservés. Limites structurelles : profondeur12,1024 noeuds,chaînes4096,int256bits,32 arguments.
Domaine incertain exclu des propriétés scorées. Mutations publiques E11 `_mutate`, SHA256
`(id + ':verif1:20261010')`, 40×64 tentatives au plus, préfixes16/64, déficit de couverture publié ; domaine
incertain. Matrice complète programmes×entrées dans le processus enfant Linux E11 seulement. Sortie
non JSON (tuple/set/objet) = perte de couverture explicite, jamais échec sémantique ; sortie brute conservée.
Un digest seul ou une erreur d'encodage E11 préfixée Output constitue aussi une perte de couverture,
distincte d'une erreur pendant l'appel au programme ; le codec E11 et ses bornes restent inchangés.

V03 deux seuils1/2 : soutien par familles visibles autres que auteur ET candidat noté, une voix/famille.
Exclure cas de sa propre famille. Moyenne des fractions passées par auteur présent puis entre auteurs.
V04 QUATRE configurations seuil1/2 × taille16/64 ; moyenne signaux propriétés et différentiel présents.
Représentant différentiel premier visible par famille dans ordre catalogue/échantillon ; autres familles,
erreurs ne sont jamais accords, non JSON exclu comme couverture manquante ; majorité n'est pas oracle.
V05 shortlist publique trois groupes signatures E11, nombre de familles distinctes décroissant puis ordre
catalogue/échantillon fixe, représentants même ordre ; sans K appris. Signature invalide = singleton.
Deux juges premières familles catalogue autres que candidat ; code intégral+contrat+résultats publics
bornés sans métadonnées auteur/modèle. Moyenne JSON valides, absence si aucun ; six appels maximum.
Identité aveugle aux métadonnées, le style du code peut rester reconnaissable.
V06 SEIZE configurations : seuil commun cas/propriétés1/2 × taille16/64 × poids (1,1,0),(1,0,1),
(0,1,1),(1,1,1). Moyenne normalisée des signaux pondérés présents ; absent si aucun. Les `selected` du
runner sont provisoires et ne remplacent jamais la sélection scientifique.

## Analyse dev, diagnostics et gel

Seul `verif1_analyze.py` charge les labels du snapshot dev, via l'analyseur historique E11 inchangé.
Il valide bundle/plan/manifeste/ledger/final full réel et les empreintes snapshot/programmes ; simulation
refusée comme source scientifique. Cinq plis externes/trois internes SHA256('percees-v1:'+id)%n.
Dans chaque train interne, K historique réappris ; toutes configurations évaluées sur validations internes
cumulées, moyenne macro bancs ; maximum macro, départage ordre grille. Réapprendre sur train externe,
évaluer son pli une fois. Aucun poids/choix ne voit labels du pli évalué.

Sélection : visibles prioritaires, sinon chargés, sinon pool complet historique ; maximum signal public,
départage K train (poids modèle, sample, catalogue). Aucun signal présent dans pool éligible : exactement
baseline historique visible+K. Un non-visible à score élevé ne bat jamais un candidat visible.
Comparateurs tous recalculés avec historique inchangé : visible+K5, familles pondérées5, P06 interne,
meilleur petit glouton train, tous 13 solos gloutons/multi internes. Tous essais internes et décisions au
registre, résultats banc/macro, prédictions, bootstrap apparié question2000 seed20261010 vers chaque
comparateur. IC descriptifs, aucune p-value dev, incertitude des entraînements recouvrants non capturée.
Réparations/cassures face à visible+K et Qwen27 sélectionné, oracle fixe35 et fraction de l'écart récupérée
par banc/macro, absence de signal éligible et temps d'analyse CPU publiés, jamais utilisés par le sélecteur.
Gel automatique : nouvelle recherche trois plis sur tout dev, toutes configs, mêmes choix/départages,
K final appris tout dev ; scores externes descriptifs ne servent pas au gel. Fichier par V03–V06 avec config,
poids et SHA code/plan/données/rapport. Pas de sélection manuelle sur scores externes.

Diagnostic distinct de sélection : accepte si TOUS cas soutenus de famille passés ; absent si aucun cas
soutenu ou couverture non JSON manquante. Publier seuils1/2, par famille puis paire sur population commune :
faux rejets sur candidats EvalPlus justes, faux accords sur faux, nombres de candidats ET questions,
marges et corrélation des acceptations sur faux, bootstrap par question. Paire accepte si deux acceptent.
Ne jamais substituer corrélations générateurs E11 à corrélations vérificateurs. EvalPlus proxy de correction,
entrées potentiellement hors contrat, diagnostic fallible, aucune règle de sélection supplémentaire.
Corrélation absente si marge dégénérée ; identité FA du parent sur même population conditionnelle.

Confirmation future helper seulement : registre partagé, réservation exclusive `VERIF-1-Vxx.opened.json`
AVANT lecture, SHA fichier gel dev obligatoire et bindings code/plan/data vérifiés. Même lecture échouée
consomme réservation ; seconde ouverture refusée. Sept variantes, alpha .05/28 (7×2 bancs×2 comparateurs).
Test consommé0/1 partout, aucun appel maintenant ; V07 gelé par son propre pipeline.

## Isolation et lancement après intégration

Landlock indisponible sur Colab E11 : pas nouvellement exigé. Sandbox Linux E11 inchangée, rlimits,
namespaces quand disponibles, audithook et bornes d'origine ; `sandbox.isolation()` réellement obtenu
journalisé, niveau historiquement accepté, aucune prétention de nouvelle sécurité. Smoke public synthétique
OBLIGATOIRE chaque invocation avant inférence, échec réel bloque GPU. WSL smoke sans GPU possible.
Certains fichiers système autorisés par E11 restent accessibles. Une exigence future Landlock dur rendrait
Colab connu incompatible et demanderait nouvel amendement/mode matrice WSL. Tests simulés ne valident pas Colab.

## Pilote DEV supervisé après revue du parent

Intégration sur main f7b7b8c : E12 a consommé son unique allocation autorisée, échouée puis libérée.
Dernier relevé indépendant fourni : 59,62 unités, débit nul, aucune VM à 20 h 18. BigCodeBench reste bloqué
P1 et ne fait pas partie de ce pilote. Le parent audite, fusionne puis lance ; aucun cloud réel par le développeur.

Reprise du 13 octobre : ordre parent E17 actif, puis E16, puis VERIF-1/E12 selon le budget restant.
Aucun lancement depuis ce worktree ; allocations et lectures du compte restent au parent après E16.
Le relevé ci-dessus est historique : il ne prouve ni VM libre ni budget disponible aujourd'hui.
Après audit/fusion, le parent doit rafraîchir le paquet et ses sources depuis le checkout audité,
dans un nouveau dossier ; l'ancien paquet ne correspond plus au code intégré corrigé.
Le bundle DEV autorisé reste inchangé, SHA-256
`bb78f63a9dde171008a0d05475c71616ef820888df2a7ffa9861420dfba56931` ; aucune lecture scientifique TEST.
Définir `VERIF_WORK` vers ce nouveau dossier préparé et son archive sœur, sans reprendre un reçu existant.
Le read_timeout et les mécanismes de propriété/nettoyage main sont conservés, E12 inchangé.

Une seule commande autorisable, scope `pilot` : 64 questions fixées, budget 25 unités, durée 4 h incluant
allocation/transfert/bootstrap/calcul/récupération, minimum 15, tarif plafond 5,3/h, nettoyage réservé 1 800 s.
Marge de compte 86 s (30 s + lecture 45 s + TERM/KILL et contrôle). Coût temporel worst-case
`5,3 × (4 + (1800 + 86)/3600) = 23,9766` unités, inférieur au budget 25 et aux 44,62 disponibles au-dessus
du plancher au dernier relevé. Cette enveloppe peut produire un pilote incomplet si les téléchargements ou
calculs durent davantage ; elle n'autorise aucune nouvelle allocation ni extension full.
Le relevé réel immédiatement avant `new` décide de la compatibilité effective, jamais l'ancien solde seul.
L'enveloppe inchangée de 25 impose au moins 40 unités disponibles sur le compte (25 + plancher 15).
Sous ce seuil après E16, le préflight refuse ; aucun ajustement automatique de durée ou budget n'est autorisé.

Depuis WSL bash de connexion, définir `MYRIAD_ROOT` vers le checkout audité et `VERIF_WORK` vers le dossier
ci-dessus, tous deux en chemins Linux. Exemple portable :

```bash
cd "$MYRIAD_ROOT"
bash phase0/colab/colab_phase0.sh verif-1 supervised \
  --archive "$VERIF_WORK/prepared.tgz" --prepared "$VERIF_WORK/prepared" \
  --receipt "$VERIF_WORK/pilot-cloud.json" --output "$VERIF_WORK/pilot-recovered" \
  --scope pilot --gpu A100 --hours 4 --budget-units 25 --min-balance-units 15 \
  --max-rate-units-hour 5.3 --cleanup-seconds 1800 --up-seconds 2400 \
  --read-seconds 45 --poll-seconds 30
```

Ajouter `--dry-run` pour valider le paquet/enveloppe hors ligne : `network=false`, aucune allocation,
aucun reçu cloud. `up` réel est désactivé ; il reste seulement comme contrôle offline de compatibilité.
Le lanceur durable Windows masqué proposé au parent est conservé hors dépôt avec la commande WSL exacte,
les chemins des logs et le code retour. Il ne faut pas interrompre seulement le terminal de suivi.
TERM au superviseur déclenche finally, puis ignore les TERM répétés pendant le nettoyage borné.
La commande reste sous responsabilité du parent jusqu'au retour final et à l'absence confirmée.

Toutes opérations passent par le routage officiel et `safe_cli.py`, sans changer CLI ni identifiants.
Le verrou lifecycle par défaut du main est tenu, avec verrou pilote vivant et mutex cleanup/pull/down partagé.
Un lanceur main configuré vers un autre reçu ne partage pas nécessairement ce verrou : le parent doit garder
les campagnes séquentielles. Toute session préexistante ou tout reçu main par défaut non réconcilié bloque le pilote.
Compte inconnu, débit non fini, allocations incohérentes, durée/budget/plancher incompatibles : refus fermé.
L'usage réel est lu avant allocation, puis toutes les 30 s pendant les appels locaux longs aussi.
Le solde courant, la baisse depuis le relevé initial et le maximum des débits observés raccourcissent la
borne, jamais ne l'allongent. Une lecture usage imbriquée partage la deadline de l'appel surveillé.

Tentative avec identifiant de campagne/allocation durable AVANT `new`. Aucun retry de `new`.
Après succès, lecture d'identité bornée trois fois maximum ; propriété session/hardware sauvegardée avant
transfert et compte alloué vérifié. Le succès de new ou son identité attestée reste une preuve volatile
séparée du reçu durable, si OSError/TERM interrompt la sauvegarde avant ou après remplacement.
La clôture n'accepte que le reçu exact avant/après cette transition pour les mêmes identifiants de campagne
et d'allocation ; elle réconcilie cette preuve, refuse toute identité différente et ne rejoue jamais new.
Après retour nonzero, adoption uniquement si la sortie propre de ce
même `new` fournit une identité stricte qui correspond à la lecture sessions ; sinon reçu incertain conservé,
aucun stop autorisé. Une apparition sans identité ne suffit jamais. Chaque mutation vérifie à nouveau la
propriété épinglée via le mécanisme main. Session remplacée/étrangère : aucun arrêt.

Le writer accuse son PID dans le marqueur de lancement. Status lit marqueur/présence PID et queue bootstrap
bornée (8 Kio, 20 lignes), jamais scores. Bootstrap échoué, marqueur absent/tronqué ou writer disparu sans fin :
arrêt de la supervision, aucune boucle infinie. Les lectures locales sont bornées à 45 s pendant le suivi,
allocation 180 s, transfert global 1 800 s dans la fenêtre up de 2 400 s ; toutes restent sous deadline globale.
Timeout/TERM arrête les descendants locaux, y compris TERM résistants, puis finally libère seulement la VM possédée.

Writer terminé (succès ou échec) : pull AVANT down dans un nouveau dossier immuable, puis contrôle technique
sans labels si succès. Writer incomplet : aucun archivage d'un ledger en mutation, nettoyage final quand même.
Après erreur/TERM, finally sonde encore la fin et récupère un writer terminé tardivement, dans au plus la
moitié de la réserve cleanup et sous la deadline du pilote ; l'autre moitié reste disponible pour down.
Usage contrôlé toutes les 30 s pendant ce pull : baisse du solde, hausse du débit, budget/plancher atteint
ou compte illisible interrompent la récupération, puis down reste autorisé pour la seule VM possédée.
Les lectures imbriquées partagent cette borne ; la récupération ne rallonge jamais la durée autorisée.
Ce pull ne relance jamais le writer.
Le dossier/réservation pull conserve toute récupération partielle ; pas de retry au même chemin.
Le superviseur n'appelle jamais extend. Aucun sampling, ordre, contrat, science, bundle ou sandbox changé.

Down réutilise le nettoyage main : stop réessayé de façon bornée jusqu'à absence explicitement lue, puis
usage final. Reçu conservé si stop/identité/absence/compte incertains.
Le compte est aussi observé toutes les 30 s pendant les appels longs de down ; un compte devenu inconnu
interdit le calcul, mais n'interdit pas de libérer la VM dont la propriété est attestée.
En cas de clôture incomplète, reprendre seulement la libération, même après fermeture du superviseur et
sans nouvelle campagne :

```bash
bash phase0/colab/colab_phase0.sh verif-1 down \
  --receipt "$VERIF_WORK/pilot-cloud.json" --cleanup-seconds 1800
```

Un reçu historique sans identité durable doit être réconcilié par le parent ; la CLI refuse son adoption
implicite. Le journal technique append-only `pilot-cloud.json.technical.jsonl` fournit usages, tentative,
propriété, états writer et clôture, sans scores. `*.usage.jsonl` conserve les relevés du compte ; la baisse
du compte n'est pas une facture scientifique de cette campagne. Le reçu cloud supprimé après absence et
comptabilité confirmées reste attesté par ces journaux, archive préparée et récupérations immuables.

Contrôle indépendant du parent, CPU offline, après pull :

```bash
uv run --project phase0 python phase0/verif1_analyze.py check \
  --prepared "$VERIF_WORK/prepared" --run "$VERIF_WORK/pilot-recovered/verif-1/results" --scope pilot
```

Ce contrôle en lecture seule valide plan/bundle/sources, manifeste réel, chaîne ledger, artefacts finaux,
64 questions × 35 candidats × 23 vecteurs et diagnostics techniques des juges, sans labels ni classement.
Un check réussi ne prouve pas la correction scientifique. Full/analyse/gel demandent une décision séparée
après revue de ce pilote ; aucune commande d'extension n'est préparée ici.

Les bruts/ledgers/rapports DEV historiques `percees_*` et `verif1_*` restent privés : exclusions ciblées
avant les globs jsonl/report de l'exporteur, y compris futurs dossiers. Code et ce guide sont publics utiles.
Scanner inchangé, fixtures adversariales toujours refusées ; aucun export réel ni scan global TEST ici.

Limites : aucun Colab/GPU réel validé ; panne réseau peut empêcher cleanup malgré sa réserve et ses retries.
SIGKILL, panne hôte ou perte WSL n'exécutent pas finally : le parent doit alors reprendre down avec le reçu,
sans réallocation. Absence Landlock historique Colab acceptée, protections E11 inchangées ; smoke synthétique
Linux avant inférence toujours obligatoire. Aucun nouveau score, candidat ni confirmation TEST ouvert.

### Provenance opérationnelle et export

`verif1_prepare.SOURCE_FILES` impose les empreintes du code, du verrou, du README et du guide
opératoire présents dans l'export. Les notes de recherche privées restent hors de cet inventaire et
ne sont pas publiées. Aucune source opérationnelle absente ou empreinte supplémentaire, manquante
ou altérée n'est acceptée. La préparation utilise séparément les entrées DEV et leur registre privé.

Un plan doit être préparé dans l'arbre qui sera réellement utilisé : le README peut être adapté par
l'exporteur. Les anciens plans, paquets et preuves restent intacts ; reconstruire dans un dossier neuf
après intégration et audit, puis valider et geler contre les empreintes de ce même arbre. Un gel
synthétique de test logiciel ne constitue pas une analyse scientifique DEV ni une autorisation de TEST.
