# Spécification de l'app Myriad (v1)

> Code et commentaires en anglais, textes visibles par l'utilisateur et documentation en français
> (convention du dépôt). Python ≥ 3.11, projet uv dans `app/`, package `myriad` (anciennement
> `essaim`). Dépendances raisonnables seulement : FastAPI, uvicorn, httpx, websockets, pydantic,
> cryptography. Pas de `transformers` dans l'app : le modèle est servi par llama-server (`--jinja`).

## Rôles

```
 PC A (utilisateur)                       Traqueur (une VM publique)                PC B, C, … (pairs)
 ┌───────────────────────┐   HTTPS        ┌─────────────────────────────┐   WSS (sortant)  ┌──────────────┐
 │ passerelle OpenAI     │ ─────────────▶ │ registre des nœuds          │ ◀─────────────── │ nœud B       │
 │ localhost:8400        │   jobs         │ relais (jobs → nœud par WS) │    jobs/résultats │ llama-server │
 │ nœud A + llama-server │ ◀───────────── │ registre des crédits        │ ───────────────▶ │              │
 │ interface localhost:8401                │ fiabilité par modèle        │                  └──────────────┘
 └───────────────────────┘                └─────────────────────────────┘
```

- **Nœud** (`myriad node`) : lance llama-server avec le modèle choisi (GGUF téléchargé depuis Hugging
  Face). Il ouvre une connexion WebSocket **sortante** vers le traqueur, ce qui permet de passer derrière
  une box (NAT) sans ouvrir de port. Il exécute les jobs reçus en respectant les limites données par
  l'utilisateur (jobs simultanés, part du GPU, plages horaires, pause/reprise) et renvoie des résultats
  signés.
- **Passerelle** (dans le même processus que le nœud) : API compatible OpenAI sur `127.0.0.1:8400`
  (`GET /v1/models`, `POST /v1/chat/completions`, flux SSE émulé). Elle choisit k pairs de familles
  différentes, leur envoie la requête en **un seul aller-retour** par le relais, puis fusionne les
  réponses et débite les crédits.
- **Traqueur** (`myriad tracker`) : FastAPI. Il gère le registre, le relais WebSocket, le registre des
  crédits (SQLite) et les statistiques de fiabilité. Il est sans état côté modèle et se déploie partout.
- **Interface** : page web locale `127.0.0.1:8401`. Elle montre l'état du nœud, les ressources allouées
  (modifiables), les crédits, les pairs du réseau, et un bac à sable de discussion.

## Identité et sécurité

- Chaque nœud a une paire de clés **ed25519**, créée par `myriad init` dans le dossier de configuration
  utilisateur. L'identifiant du nœud est le hash de sa clé publique.
- Connexion au traqueur : le traqueur envoie un défi, le nœud le signe. Les messages de résultat sont
  signés (JSON canonique : clés triées, UTF-8, sans espaces).
- La passerelle n'écoute que sur 127.0.0.1. Le traqueur n'accepte que des messages signés par des nœuds
  enregistrés. Limites de taille sur tous les champs, et délais sur tous les jobs.
- Aucun secret dans le dépôt. Les clés restent dans le dossier utilisateur.

### Chiffrement de bout en bout et protections (essaim/1.3, `e2e.py`, `privacy.py`, `security.py`)

Modèle de menace, garanties et arguments : `docs/08_securite.md`. En résumé :

- Chaque nœud qui sert publie une clé **X25519** signée par son identité ed25519 (`KxCert`, renouvelée
  chaque jour, l'ancienne gardée jusqu'à son expiration). Le demandeur vérifie la signature et que
  l'identifiant est le hash de la clé publique : le traqueur ne peut pas substituer sa clé.
- Par job : clé éphémère X25519, HKDF-SHA256 (contexte : protocole, job, pseudonyme, pair, les deux clés
  publiques), ChaCha20-Poly1305, une clé par sens ; données associées = l'en-tête en clair dont le
  traqueur a besoin (job, cible, limite de jetons, échéance, expiration) ; réponse en morceaux à nonce
  compteur avec marqueur final authentifié ; rejeu refusé (identifiant + expiration) ; tailles
  arrondies à une puissance de deux. Le job est signé par un **pseudonyme jetable** : le pair ne sait
  pas qui demande, le traqueur facture la connexion authentifiée qui l'a envoyé.
- Sélection par le traqueur : `Reserve` → `Assigned` (avec la clé du pair) → le demandeur vérifie le
  pair (liste de blocage, nœuds de confiance, essaim privé, fiabilité minimale, quarantaine, clé) et
  n'envoie qu'ensuite le job chiffré.
- Réglage « Exiger le chiffrement de bout en bout » activé par défaut : un pair sans clé n'est jamais
  choisi, et un traqueur sans la fonction `e2e` ne reçoit rien.
- Contrôles : les jobs chiffrés sont audités par des **canaris** du traqueur, indiscernables au niveau
  des trames (même forme, même taille arrondie, pseudonyme neuf, délai aléatoire), qui peuvent porter des
  jetons-pièges ; les jobs en clair (anciens pairs) gardent les contrôles par duplication.
- Garde de confidentialité locale : secrets et données personnelles remplacés par des marqueurs
  (`CLE_1`, `EMAIL_1`…) rétablis localement dans la réponse ; confirmation la première fois qu'un type
  sensible partirait non masqué ; règles « toujours local » ; mode local ; historique plafonné.
- Côté serveur : nœuds bloqués refusés, limites par demandeur (débit, simultanéité) appliquées par le
  traqueur (le pair ne voit que des pseudonymes), taille maximale des questions et des réponses ; aucun
  texte de job dans les journaux, cache du moteur effacé après chaque job.

## Protocole (`myriad/protocol.py`, modèles pydantic, version `essaim/1`)

L'identifiant du protocole garde l'ancien nom du paquet : `essaim/1` préfixe chaque message signé et
c'est la version annoncée par chaque nœud. Le changer casserait la compatibilité avec les nœuds déjà
déployés ; il ne suit donc pas le renommage du paquet.

- `NodeInfo` : node_id, pubkey, model (id HF), family, gguf, params_b (milliards), ctx, max_parallel,
  version, accepting (bool).
- `Job` : job_id, requester_id, messages (format OpenAI), max_tokens (≤ 2048), temperature, seed,
  deadline_ms, task_hint (`"math"`, `"mc"`, `"free"`, ou null).
- `JobResult` : job_id, node_id, model, text, finish_reason, completion_tokens, mean_logprob (ou null),
  compute_ms, signature.
- `Receipt` : job_id, requester_id, node_id, completion_tokens, model, signé par le demandeur. Le traqueur
  crédite le nœud et débite le demandeur.

Extension **`essaim/1.1`** (compatible : signatures et trames essaim/1 inchangées, nouveautés utilisées
seulement entre pairs qui les annoncent ; `GET /v1/health` donne `protocol_version` et `features`) :

- `JobFrame.route` (group, model, exclude, replaces) : job sans cible, le traqueur choisit le pair (une
  famille par job d'un même groupe, le modèle le plus fiable d'abord, nœuds libres et sans manquement
  récent) et l'annonce par une trame `assigned` (`PeerCard` : identité, modèle, famille, fiabilité),
  envoyée avant tout autre message sur ce job. `GET /v1/select?k=&model=&exclude=` donne le même choix.
- `ping` / `pong` : vérification applicative de vie (le nœud sonde son moteur avant de répondre) ; le pong
  porte aussi les limites du nœud et remplace le battement `Status`.
- Santé : un nœud muet, au moteur en panne, ou qui dépasse deux fois de suite le délai d'un job, est
  suspendu (recul exponentiel) et n'est plus choisi ; il est réadmis après un pong sain.

Fonction **`update`** (mises à jour de l'app, ajout compatible, même `protocol_version` `essaim/1.2`) :

- Le nœud la demande dans l'URL du WebSocket (`/v1/ws?features=update`) ; un traqueur plus ancien ignore
  la chaîne de requête. Seul un nœud qui l'a demandée reçoit `Welcome.latest_version` (champ omis sinon)
  et la trame `update` (`{"t": "update", "version": "X.Y.Z"}`) quand une nouvelle version paraît : un
  nœud plus ancien refuserait un champ ou une trame inconnus.
- Le traqueur interroge `GET https://api.github.com/repos/{dépôt}/releases/latest` toutes les 10 minutes
  (ETag et `If-None-Match` : une version inchangée coûte une réponse 304), ignore brouillons et
  pré-versions, lit `SHA256SUMS.txt`, recule en cas d'erreur ou de limite de débit et ne s'arrête jamais
  pour autant. `GET /v1/version` rend `{repo, latest: {version, tag, published_at, html_url, assets,
  sha256}, checked_at}` avec `Cache-Control: public, max-age=300`. Désactivé dans les tests et avec
  `MYRIAD_RELEASE_REPO=off`.
- Le client (`myriad/updater.py`) ne croit le traqueur que pour un numéro de version : regex stricte
  `X.Y.Z`, strictement supérieur à la version installée (jamais de retour en arrière). Il construit
  l'URL à partir du dépôt épinglé, choisit le fichier de sa plateforme (`Myriad-Setup-X.Y.Z.exe`,
  `Myriad-X.Y.Z-arm64.dmg` / `-x86_64.dmg`, `Myriad-X.Y.Z-x86_64.AppImage`), le télécharge (reprise par
  `Range`) et vérifie son SHA-256 contre le `SHA256SUMS.txt` de la version sur GitHub. Repli sur l'API
  GitHub seulement si le traqueur ne peut pas répondre. Crochet prévu pour une signature ed25519 de
  `SHA256SUMS.txt` (clé publique épinglée, pas encore de clé).
- Installation, après l'arrêt propre du nœud, de llama-server et de la passerelle : installateur Inno
  Setup silencieux (`/SILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS`, relance par
  `/MYRIADRELAUNCH=1`), échange du paquet `.app` sur macOS (`hdiutil attach`, `ditto`, renommages),
  remplacement atomique de l'AppImage. Les autres installations (zip portable, `.deb`, `.tar.gz`,
  sources, pip) sont seulement prévenues. La nouvelle instance, lancée avec `--after-update`, attend que
  l'ancienne libère son verrou.

## Fusion (`myriad/fusion.py`, fonctions pures, très testées)

- `extract_answer(text, task_hint)` : nombre final (style GSM8K « The answer is N », \\boxed{}, dernier
  nombre en repli) ou lettre de QCM ; None sinon.
- `weighted_vote(answers, weights)` : poids = log-odds de la fiabilité du modèle, ln(p/(1−p)), bornés.
  C'est le vote optimal de Nitzan–Paroush sous indépendance. Les égalités sont départagées par le
  mean_logprob.
- `medoid(texts, weights)` : pour le texte libre, la réponse la plus proche des autres (similarité
  ROUGE-L ou Jaccard de mots, pondérée), sans modèle supplémentaire.
- `stop_certificate(partial_votes, remaining_weight)` : renvoie la réponse dès qu'aucun ensemble de
  réponses manquantes ne peut plus la renverser (docs/03_idees_codex.md, idée 2). La passerelle rend la
  réponse dès que le certificat est atteint, sans attendre les retardataires. En v1.1, un pair refusé ou
  en échec rapide est remplacé une fois ; le certificat compte tous les jobs en attente, remplaçants
  compris, et n'est évalué qu'une fois leurs poids connus.
- Fiabilité : un a priori par famille (fichier livré, issu des mesures de phase 0), mis à jour par le
  traqueur à partir de l'accord avec le consensus (moyenne bêta). Les données sont exposées en lecture.

## Crédits (v1, centralisés au traqueur, vérifiables)

- Un nouveau nœud reçoit un crédit de départ.
- Servir un job rapporte completion_tokens × facteur(params_b).
- Consommer coûte la somme sur les pairs interrogés.
- Un solde négatif empêche de lancer de nouvelles requêtes.
- Contrôles aléatoires : une petite fraction des jobs en clair est dupliquée sur un autre nœud du même
  modèle ; pour les jobs chiffrés, le traqueur envoie ses propres jobs canaris (voir plus haut). Un
  désaccord répété (hors échantillonnage, hors réponses tronquées) fait baisser la réputation.
- Les résultats livrés mais pas encore réglés engagent leur coût exact sur le compte du demandeur
  jusqu'au règlement : un demandeur ne peut plus recycler son quota de jobs en vol sans payer.
- Tout est journalisé avec des reçus signés, pour qu'une v2 puisse remplacer le registre central par une
  vérification entre pairs.

### Usage gratuit (quota quotidien et récompense de disponibilité)

**Problème de départ.** Avec le seul crédit de départ (1000), un compte neuf ne payait qu'un millier de
jetons d'un modèle de 2 G : un run de sous-agents (plusieurs sous-tâches, k candidats chacune) commençait
puis s'arrêtait en route, faute de crédits. Trois causes, toutes reproduites dans `tests/test_free_tier.py` :

1. **Crédits épuisés en cours de route** (`insufficient_credits`) : le coût réel (jetons × facteur) dépasse
   le solde, et le traqueur ne refuse un job que lorsque le solde est déjà négatif.
2. **Plafond de jobs en vol d'un compte neuf** (`NEW_ACCOUNT_INFLIGHT` = 8) : 6 sous-tâches × k = 2 font
   12 jobs d'un coup ; les 4 derniers recevaient `too_many_jobs`, que la passerelle ne remplace pas, et leurs
   sous-tâches échouaient.
3. **Pairs occupés plus longtemps que la fenêtre de reprise** (7,5 s fixes) : la sous-tâche abandonnait
   avec `no_peers` alors que les pairs étaient seulement occupés.

**Correctifs côté passerelle et sous-agents.**

- La passerelle connaît le plafond de jobs en vol de son compte (`GET /v1/account/{id}`) et fait **attendre
  dans une file FIFO** les requêtes au-delà (`_Slots`) : `too_many_jobs` ne tue plus une sous-tâche (si le
  traqueur et la file divergent un instant, la sous-tâche attend et redemande). Une requête de chat qui
  attend plus que son délai reçoit 429.
- Une sous-tâche dont les pairs sont tous occupés attend **jusqu'à la date limite du run** (pauses de 0,5 à
  2 s) ; elle n'échoue tout de suite que si personne ne sert ce modèle ou cette famille.
- **On ne lance pas ce qui ne peut pas finir.** Avant le départ du premier job, `AgentRun.check_funds`
  estime le coût (somme des `max_tokens × k` des sous-tâches, plus la fusion, bornée par le budget de jetons
  du run, × le facteur moyen des pairs atteignables, × 0,5 de remplissage typique) et le compare à
  *solde + quota gratuit restant*. Sinon : refus HTTP 402 (`insufficient_credits`) avec l'estimation et les
  options (moins de candidats k, moins de jetons par sous-tâche avec une valeur suggérée, moins de
  sous-tâches, mode local, attendre ou servir le réseau). Un plan automatique est vérifié une fois écrit
  (le run se termine alors en statut `refused`). L'interface affiche le même message.
- Si les crédits manquent malgré tout en cours de route (l'estimation est typique, pas un maximum), le run
  ne démarre plus rien, **garde les résultats obtenus** (statut `partial`, concaténation sans fusion) et le
  dit.

**Quota gratuit par jour** (`free_daily_tokens`, 40 000 crédits par défaut ; 1 crédit = 1 jeton d'un modèle
de 2 G). Un job est d'abord payé sur ce quota, puis sur le solde ; le réseau crée la part gratuite (colonne
`ledger.free`, table `free_use`), donc la création monétaire est bornée par le quota de chaque compte et par
jour (UTC). Le quota dépend de la **réputation du compte**, pour résister aux Sybil :

    quota = 40 000 × (0,05 + 0,95 × âge × disponibilité)

- âge = min(1, âge du compte / 3 jours) ;
- disponibilité = min(1, canaris de disponibilité réussis sur 3 jours / 24), et 0 si le nœud a échoué
  des canaris (réponse qui ne s'ouvre pas : un jour ; réponses fausses ni rares ni minoritaires) ;
- un compte neuf, ou dont personne n'a vérifié la disponibilité, reçoit donc le plancher de 5 % (2 000
  crédits par jour) ; un compte banni, 0. Les protections existantes restent : plafond de jobs en vol
  d'un compte neuf, découvert borné à une rafale (`max_overdraft`).

**Récompense de disponibilité.** Un nœud qui sert un modèle, est en ligne, accepte des jobs et passe les
**canaris de disponibilité** gagne des crédits même sans aucune demande d'utilisateur, ce qui permet
d'amorcer un petit réseau.

- Le traqueur envoie à chaque nœud servant un canari (même trame chiffrée, même format qu'un vrai job :
  `canary.py` ; l'en-tête `max_tokens` est tiré parmi 128, 192 ou 256) toutes les ~15 min
  (`avail_interval_s` = 900, ±25 %), en plus des canaris qui suivent les vrais jobs. Un nœud occupé, en
  pause ou suspendu est reporté puis ignoré : pas de récompense.
- **La récompense est le paiement du canari, et seulement s'il est réussi** : la réponse s'ouvre et égale la
  réponse connue (sans second pair). Elle est réglée comme un job ordinaire (jetons × facteur, soit
  quelques dizaines de crédits pour un modèle de 2 G), créée par le réseau, et la ligne du registre est un
  reçu comme un autre : un pair ne peut pas y lire quels jobs étaient des canaris. Une réponse fausse est
  réglée à zéro et compte contre le nœud. Le total est **plafonné par nœud et par jour** à 9 600 × facteur
  du modèle (plafonné à 4) × rampe d'âge (0,25 le premier jour, 1 à 3 jours) : une fois atteint, le
  traqueur n'envoie plus de canari de disponibilité ce jour-là.
- Rien pour un nœud banni ni pour un nœud bloqué (aucun canari ne lui est envoyé) : une réponse qui ne
  s'ouvre pas, ou mal signée, bloque un jour ; des réponses fausses bloquent quand elles sont au moins 3 sur
  2 jours et au moins 1 sur 4. Un canari tronqué est payé comme une réponse mais ne compte pas comme réussi.
- Les chiffres de disponibilité que le propriétaire voit (facteur de disponibilité du quota,
  `availability.earned_24h`) ne comptent que les **heures révolues** : un compteur qui bougerait juste après
  un canari apprendrait au pair, qui interroge son propre compte, quel job vient d'être un canari.
- Tout est visible par le propriétaire : `GET /v1/account/{id}` (signé comme `/v1/balance`) rend le solde, le
  quota restant aujourd'hui, le plafond de jobs en vol et les gains de disponibilité (24 h révolues,
  estimation par jour, maximum par jour) ; la passerelle les relaie dans `/v1/myriad/status` et
  `/api/status`, affichés sous « Crédits gagnés ».

Réglages du traqueur : `--free-daily-tokens`, `--avail-interval`, `--avail-daily-cap` (0 désactive le
quota, ou les canaris de disponibilité).

**Cohérence avec la condition d'incitation de l'article** (`paper/main.tex`, théorème d'incitation :
servir honnêtement est une meilleure réponse ssi q(β−α)(P+R) ≥ g_H − g_d, le paiement ne dépendant pas de
la stratégie). La récompense de disponibilité ne paie pas un nœud qui triche :

1. Elle est versée seulement sur un canari réussi, c'est-à-dire sur une réponse honnête (de coût g_H),
   et les canaris sont indiscernables des vrais jobs (même trame, même taille, en-têtes plausibles, même
   ligne de registre). Avec λ canaris de disponibilité par job réel et une récompense ρ, l'utilité honnête gagne
   λρ(1 − α) et celle d'une déviation d gagne λρ(1 − β_d) : l'écart U_H − U_d devient
   q(β_d − α)(P+R) + λρ(β_d − α) − (g_H − g_d). Le terme ajouté est positif dès que β_d ≥ α, c'est-à-dire
   la condition que le théorème exige déjà. La récompense **renforce** donc l'incitation à l'honnêteté
   au lieu de l'affaiblir : tricher fait perdre ρ à chaque canari détecté, et un nœud bloqué ne touche plus rien.
2. Elle ne dépend pas du paiement des vrais jobs : l'hypothèse « paiement indépendant de la stratégie » reste
   vraie pour la partie vrai job. Elle rémunère la disponibilité vérifiée (coût d'opportunité), non un calcul
   non fait : un nœud qui ne sert plus ses jobs est suspendu et ne reçoit plus de canaris.
3. Limites assumées : un nœud ne perd la récompense qu'avec la probabilité qu'un canari voie sa
   déviation ; si la déviation ne s'applique que hors de la distribution des canaris (corollaire sur
   l'ensemble U), l'argument vaut comme dans l'article, pas davantage. Un nœud qui réussit les canaris avec un
   petit modèle ne gagne qu'au facteur de ce modèle (plafonné) ; faire tourner N nœuds en Sybil rapporte au
   plus N plafonds quotidiens, au facteur du modèle, avec la rampe d'âge : borné, mais pas nul.

## CLI

- `myriad init [--model Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf] [--tracker URL]` : clés, configuration,
  téléchargement du modèle.
- `myriad node` : nœud, passerelle et interface.
- `myriad tracker [--host --port --db]` : le traqueur.
- `myriad chat "question"` : client minimal de la passerelle.
- `myriad status` : état du nœud.
- `myriad update [--check]` : nouvelle version de Myriad ? (le traqueur répond, GitHub en repli).

## Tests (obligatoires)

- Tests unitaires de fusion (vote, médoïde, certificat, extraction), de crypto et des protocoles.
- **Test d'intégration sans GPU** : un traqueur, 4 nœuds avec un moteur simulé (réponses scriptées),
  une passerelle. On y vérifie une requête de bout en bout, la fusion, le certificat d'arrêt, la panne
  d'un nœud, les crédits débités et crédités, et le refus d'une signature invalide.
- **Test réel optionnel** (marqué) avec llama-server et un petit GGUF.
