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
- Contrôles aléatoires : une petite fraction des jobs est dupliquée sur un autre nœud du même modèle. Un
  désaccord répété (hors échantillonnage) fait baisser la réputation.
- Tout est journalisé avec des reçus signés, pour qu'une v2 puisse remplacer le registre central par une
  vérification entre pairs.

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
