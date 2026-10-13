# E17 finale : gel exact et exécution unique supervisée

Le chemin final est implémenté et se teste sur fixtures synthétiques. **Aucun gel scientifique ni lancement
réel effectué ici.** P02 invalidée sans rejeu, R00 conservé ; P03 reste à clore par le parent.
Les correctifs techniques Aider, BigCode et délai local sont intégrés par SHAs exacts audités.
Le parent doit encore approuver la révision et le gel. Le DEV R00 historique garde son plan et sa garde R00.

## Préparation locale, sans données TEST

Depuis `phase0`, dans l'environnement Python existant, avec chemins de preuve neufs :

```powershell
python e17_final_freeze.py prepare --output "$E17_PREPARATION" --spent-units 1.15 --budget-units 18.85 `
  --max-rate-units-hour 5.3 --seconds 10000 --recover-seconds 3000 --cleanup-seconds 1800
python e17_final_freeze.py validate --preparation "$E17_PREPARATION"
python e17_final_freeze.py decision-template --output "$E17_DECISION"
```

Ces commandes lisent seulement code et métadonnées, jamais archives scientifiques, caches, instructions,
assertions ou références TEST. Le template indique `approved=false`, révision/sélection/décisions et hashes
d'audit/smokes non arrêtés. Il ne choisit pas R00/R01 ni P02/P03 et ne donne aucun droit de lancement.
Le parent complète le registre après clôture DEV : trois propositions maximum, P01 comprise ; acceptation,
rejet ou invalidation de chacune, proposition sélectionnée et révision scientifique explicites.

Le parent doit renseigner `code_commit` exact propre, `adapter_fix_commit` exact intégré et audité,
`audit_sha256`, `dev_smoke_sha256`, `synthetic_smoke_sha256`, et confirmer `system_sha256` et toutes les
empreintes `sources`. `actor=parent` et `approved=true` sont des déclarations d'autorisation explicites,
**pas une signature cryptographique**. Les propositions doivent être présentes sous les noms exacts
`docs/e17_proposition_p01.md`, puis éventuellement `p02.md` et `p03.md`, sans trou dans le registre.

Le codec historique est confirmé par `parent-bigcode-record-codec-proof.json` et son extrait exact
(lignes 5841-5873, SHA du log dans la preuve), archivés dans les preuves intégrées.
`bigcode_record_encoding=json-sort-ascii-default-v1` est fixé : projection des six champs `task_id`,
`instruct_prompt`, `code_prompt`, `canonical_solution`, `test`, `entry_point`, puis
`json.dumps(projection, sort_keys=True).encode()`, ASCII échappé et séparateurs Python par défaut,
sans saut final. Les colonnes supplémentaires sont exclues. Le canonique sert uniquement au hash de
provenance, jamais au staging, prompt ou smoke. Contrat vérifié par le parent sur DEV120/184, aucun TEST lu.

## Gel local, uniquement par le parent après décision et audit

```powershell
python e17_final_freeze.py freeze --preparation "$E17_PREPARATION" --decision "$E17_DECISION" `
  --output "$E17_FREEZE" --parent-approved
python e17_final_freeze.py validate-freeze --freeze "$E17_FREEZE"
```

Le gel lie décision/registre, code, prompt, adaptateurs, analyse, runtime, budgets, modèles, serveur,
sélection et ordre des seize cellules. Les empreintes exactes de la décision doivent correspondre au code
courant ; le commit doit être propre et le correctif P2 être un ancêtre. Le gel crée `freeze.json` et
`code.tar` dans un dossier neuf. Préparation interrompue ou existante : refus, aucune écriture par-dessus.
Toute altération, fichier absent/ajouté, archive étrangère ou changement de commit refuse avant allocation.
Une validation conforme autorise une tentative sous préconditions VM ; elle ne certifie pas le support TEST.

Liste blanche du bundle : Python directement sous `phase0`, `phase0/essaim`, `phase0/colab`, `phase0/tests` ;
scripts `.sh` directement sous `phase0/colab` ; `pyproject.toml` et `uv.lock` de `code_pilot` et `bigcode_pilot` ;
`bigcode_pilot/runtime.json` ; les **manifestes de seules métadonnées** `code_pilot/tasks.json` et
`bigcode_pilot/tasks.json` ; réservation, propositions présentes et ce guide. Ces manifestes portent noms,
versions et empreintes : aucune instruction, assertion, référence ou ligne de dataset TEST incluse.
Somme des sources bornée à 8 Mio. Aucun `official_tests.json`, cache, résultat, dataset ou poids transféré.

## Supervision explicite, uniquement sur WSL après validation parent du gel

Le parent prépare `MYRIAD_ROOT`, `E17_FREEZE`, `E17_OUTPUT` et `E17_FREEZE_SHA256`, puis, une seule fois :

```bash
bash "$MYRIAD_ROOT/phase0/colab/colab_phase0.sh" e17-final \
  --freeze "$E17_FREEZE" --output "$E17_OUTPUT"
# Après validation explicite de ce gel exact par le parent :
bash "$MYRIAD_ROOT/phase0/colab/colab_phase0.sh" e17-final \
  --freeze "$E17_FREEZE" --output "$E17_OUTPUT" \
  --execute --parent-validated-freeze "$E17_FREEZE_SHA256"
```

La première commande vérifie localement, sans CLI Colab. La seconde exige le gel intact et le SHA confirmé.
Elle écrit durablement un marqueur exclusif fixe sous le home du compte WSL,
`.local/state/myriad-colab/e17-final-test-consumed.json`, **avant toute opération distante**. Il ne dépend
ni du dossier de résultats, ni d'un autre reçu, ni de `DLLM_CAMPAIGN_RECEIPT`, ni d'une nouvelle révision.
Interruption, erreur, marqueur incomplet/étranger ou nouvel output : pas de deuxième tentative. Ne jamais
supprimer ce marqueur pour rejouer. Aucun `resume`, nouvelle allocation ou dossier de contournement.

`FinalCampaign` réutilise Campaign/Probe, les mêmes ownership/reçus/verrous, garde compte, watchdog, transport
multiparts, récupération et cleanup. Les seuls points d'extension sont source RPC, validation d'inventaire/
récupération et libellé final. Autorité d'arrêt, contrôle de PID/session étrangers et signatures RPC conservés.
RPC final explicite vers `e17_final_remote.py` ; les autres routes restent DEV. Aucun shell distant hors wrapper.
Un ACK `start` perdu ne rejoue pas `start` ; seule récupération de notre tentative, puis clôture.

## VM : préparation et notation figées

Avant bootstrap, deux marqueurs Colab réels et toutes les sources sont vérifiés. Le worker réutilise UV
autonome épinglé 0.12.22, Python 3.12.15 GNU x86_64 et `uv sync --frozen` sur `bigcode_pilot/uv.lock`.
Le binaire Python, 31 packages, lock, licences et composants compilés sont contrôlés par `check_runtime`.
Licence Python téléchargée selon son pin ; observations des licences/versions/hashes dans la provenance.
Le pin binaire WSL est conservé : sa correspondance sur Colab reste à établir au runtime ; mismatch = erreur
d'infrastructure, **pas une autre version/package/wheel installée après ouverture TEST**.

Préflight namespaces/seccomp/reçu `ephemeral-vm-v1` et smokes **synthétiques préfixés** des deux adaptateurs
avant ouverture réservée. Fixture Aider de fonctions/classes et hooks de module ; fixture BigCode correcte/
incorrecte. Le nom DEV historique utilisé comme seam de catalogue pour la fixture Aider ne change pas ses
octets synthétiques. Aucun smoke TEST avec solution correcte, aucune référence/oracle TEST exécutée.

Archive Aider/révision/hashes existants : uniquement configs, instructions, stubs et tests déclarés des deux
identifiants réservés, ainsi que licences/attribution. Pas de `.meta/example` ou solution exemple ouverte.
Parquet BigCode Hard Instruct v0.1.4/révision/bytes/SHA existants : filtrage par IDs réservés avant conversion
Python. Record intégral utilisé seulement pour checksum historique ; `canonical_solution` n'est ni stagé,
ni donné au prompt/adaptateur, ni exécuté. Tests originaux copiés verbatim, collection statique/compteurs
publics conservés. Support/configuration/import/runtime/collection non pris en charge = incomplet, sans
substitution de tâche ou adaptation. Les anciens manifestes et données restent immuables.

Le runner appelle le `run_one` et BudgetChat existants avec adaptateur explicite : même SYSTEM choisi par
parent, boucle Aider, vote/cascade et budgets. L'adaptateur générique utilise EphemeralVMEnv explicitement
sur VM, avec le crochet Python E11 avant import candidat et les reçus existants. Défaut strict local et
Landlock PC inchangés, aucun fallback. En-tête loopback public fixe pour éviter un Bearer vide, aucun secret.

Ordre : single, vote, cascade, reference ; dans chaque mode bottle-song, bowling, BigCodeBench/208, /211.
Seize cellules exactes. Registre durable avant chaque préparation d'adaptateur/épisode ; erreur ou interruption
conserve la cellule consommée et arrête la campagne. Aucun résultat incomplet assimilé à échec candidat/score.
Assertions originales dans la boucle normale et notation finale indépendante une fois par cellule ; diagnostics
TEST ne retournent jamais au proposeur. Stdout vert ne suffit pas : compteurs complets, provenance, snapshots
inchangés et soumission autorisée sont requis.

Qwen3.5-2B Q8_0 seul ; vote Qwen puis Granite4.2-3B Q8_0 ; cascade mêmes pairs vers Qwen3.8-27B Q8_0 ;
27B seul comme référence. Pins de `e17_campaign`, CUDA b11505/ff5888f99, A100, un slot séquentiel,
contexte32768, reasoning0, température0, seed0 ; 4096/49152,20 étapes,900s,30s commande,120s vérification.
Aucun 9B ajouté. Simulation centralisée, pas de réseau distribué ou mesure WAN revendiquée.

Budget TOTAL E17 20 incluant coûts antérieurs ; au moins1,15 chargé jusqu'à réconciliation et réserve de
compte prudente1,15 au-dessus du plancher absolu15. Plafond tarif déclaré, formule admission existante,
marge161s, attente usage600s, récupération au moins2850s réservée dans la fenêtre (27 artefacts/64Mio/
45RPC au plus,50s d'exécution distante/RPC+600s de marge), exemple3000s ; clôture1800s distincte.
Les lectures locales bornées120s et attentes compte peuvent épuiser cette réserve : hypothèse de récupération,
pas garantie réseau. Compte inconnu mène à clôture ; reçu conservé si absence/compte final non confirmés.
Seize épisodes ne sont pas garantis : 16×1020s dépasse déjà la fenêtre avant installation/chargements.
Budget insuffisant = incomplet, sans nouvelle allocation ou réduction silencieuse du protocole.

## Récupération et analyse

Tous les hashes de l'inventaire clos sont vérifiés même après échec ; partiels conservés comme incomplets,
sans score final. Complétude exige exact16 cellules, registre consommé, ordre, accords end/verdict/trace,
paramètres/jetons, backend/hook/reçus, Python/package/licences/lock, modèles/serveur, smoke et gel exacts.
Erreurs d'infrastructure/provenance distinctes des échecs candidats. Réutilisation des trames bornées 4Mio,
60k caractères et enveloppe8Mio ; total artefacts64Mio, pas d'inventaire libre ni chemin arbitraire.

Analyse à partir de livraison complète seulement : quatre tâches appariées descriptives, par banc, sans
score officiel ni preuve de généralisation. Jetons connus/réservés, temps appels/chargements/épisodes/
préparation/clôture distincts ; variation de compte différente d'une facture. Hypothèses historiques100W et
0,25EUR/kWh, non mesurées, inscrites ici avant gel. Aucun chiffre article manuel ni feedback au proposeur.
Ce pilote note des bancs publics non adversariaux, **sans attestation cryptographique de Python arbitraire**.

Export public éventuel : code léger et guides selon liste blanche précise, scanner inchangé ; gels,
décisions opérationnelles, journaux et artefacts TEST restent dans les preuves privées. L'export simulé de
sources ne certifie ni Colab, ni CUDA, ni support TEST.
