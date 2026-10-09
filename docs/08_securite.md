# Sécurité et confidentialité de Myriad (essaim/1.3)

Ce document décrit ce que Myriad protège, ce qu'il ne protège pas, et pourquoi. Il couvre le
chiffrement de bout en bout entre le demandeur et le pair qui calcule, les pseudonymes, les jobs
canaris du traqueur, les outils de protection locaux et les limites qui restent. Le code correspondant :
`app/myriad/e2e.py` (chiffrement), `privacy.py` (garde de confidentialité), `security.py` (politique
des pairs, quarantaine, limites de service), `canary.py` (canaris et jetons-pièges), `tracker.py`,
`gateway.py` et `node.py` (protocole).

## 1. En une page

| Qui | Voit la question ? | Sait qui demande ? | Ce qu'il voit d'autre |
| --- | --- | --- | --- |
| Le traqueur (honnête mais curieux) | non | oui (le compte qui paie) | quel pair, quand, tailles arrondies, limites et nombre de jetons, délais |
| Le pair qui calcule | **oui, en clair** (après masquage local) | non (un pseudonyme par job) | la question, les paramètres d'échantillonnage, la réponse qu'il produit |
| Un autre pair | non | non | rien |
| Un observateur du réseau (avec `wss://`) | non | l'adresse IP de chaque machine, les volumes, les instants | |
| Un traqueur allié à un pair | oui | oui | tout ce que voient les deux |

Ce qui est garanti, sous les hypothèses cryptographiques usuelles :

- le traqueur ne peut ni lire une question chiffrée ni sa réponse, ni les modifier, tronquer, réordonner
  ou rejouer sans que cela soit détecté, ni substituer sa propre clé à celle d'un pair ;
- le pair ne connaît pas l'identité du demandeur ;
- les canaris du traqueur sont indiscernables des vrais jobs au niveau des trames.

Ce qui n'est pas garanti, et ne peut pas l'être avec la technologie disponible :

- **le pair qui exécute le modèle lit la question en clair** et rien n'empêche techniquement un pair
  malveillant de la copier. Le calcul homomorphe complet est de plusieurs ordres de grandeur trop lent
  pour un LLM, et les GPU grand public n'offrent pas de calcul confidentiel (enclaves attestées) ;
- un traqueur malveillant peut **choisir** les pairs : s'il fait tourner ses propres pairs (Sybil), il
  lit les questions qu'il leur attribue. Le mode « nœuds de confiance » et l'essaim privé y répondent ;
- les métadonnées (section 6) restent visibles du traqueur.

L'application compense ce qu'elle ne peut pas chiffrer : elle détecte localement les secrets et les
données personnelles et les remplace par des marqueurs avant l'envoi, garde une question en local si
une règle le demande, change de pairs à chaque question, envoie le moins de contexte possible, et
dissuade les fuites par des jetons-pièges (une fuite est détectée après coup, pas empêchée).

## 2. Modèle de menace

**Acteurs.**

- *Traqueur honnête mais curieux* : il suit le protocole, mais lit et garde tout ce qui passe.
- *Traqueur malveillant* : il peut modifier, supprimer, retarder, rejouer ou forger des trames, mentir
  dans l'annuaire, choisir les pairs, faire tourner ses propres nœuds.
- *Pair malveillant* : il sert des jobs et peut copier ce qu'il déchiffre, renvoyer des réponses bâclées,
  tenter de reconnaître les canaris pour ne tricher que sur les vrais jobs, mentir sur sa famille.
- *Demandeur malveillant* : il peut tenter de ne pas payer, d'épuiser la mémoire du traqueur, de
  saturer un pair, de signaler abusivement.
- *Observateur du réseau* : il voit les connexions (chiffrées par TLS si le traqueur est en `https://`).

**Hypothèses.** X25519 : le problème Diffie-Hellman calculatoire (gap-CDH) est difficile sur
Curve25519. HKDF-SHA256 : extracteur puis PRF (modèle de l'oracle aléatoire pour l'extraction).
ChaCha20-Poly1305 : AEAD sûr (IND-CPA et intégrité des chiffrés, INT-CTXT) tant qu'un couple
(clé, nonce) n'est jamais réutilisé. Ed25519 : signatures infalsifiables (EUF-CMA). SHA-256 : résistance
aux collisions (l'identifiant d'un nœud est le hash de sa clé publique, tronqué à 128 bits). Les
horloges des machines sont justes à quelques minutes près (`CLOCK_SKEW_S` = 120 s).

## 3. Le protocole

**Clés des pairs.** Chaque nœud qui sert tire une clé X25519 et publie un certificat `KxCert`
{node_id, kx, created, expires} signé par son identité ed25519 (type de signature `kx`, séparé des
autres par le préfixe `essaim/1\x00kx\x00`). Le certificat vit 26 h, une nouvelle clé est tirée toutes
les 24 h, l'ancienne est gardée jusqu'à son expiration pour les jobs en vol. Le nœud l'annonce dans son
`Hello` (champ optionnel `kx`) puis par une trame `KxFrame` à chaque renouvellement. Le traqueur
vérifie le certificat avant de l'accepter, mais **le demandeur le revérifie toujours** : node_id =
SHA-256(clé publique ed25519) tronqué, signature valide par cette clé, durée de vie ≤ 48 h, non expiré.

**Un job.** Le demandeur tire une clé éphémère X25519 `e` et une identité ed25519 jetable (le
*pseudonyme*, `requester_id` = son hash). Il calcule `s = X25519(e, kx)`, puis
`HKDF-SHA256(sel = "myriad-e2e-v1", ikm = s, info = signing_bytes("e2e-ctx", {job_id, pseudonyme,
pair, eph, kx}))` → 64 octets : `k_req` (demandeur → pair) et `k_resp` (pair → demandeur). L'en-tête en
clair `{job_id, requester_id, target, max_tokens, deadline_ms, expires_at, kx, eph}` sert de données
associées ; le contenu `{messages, temperature, seed, task_hint, thinking}` est mis en JSON canonique,
préfixé de sa longueur, complété de zéros jusqu'à une puissance de deux (1 Kio à 512 Kio), puis chiffré
par ChaCha20-Poly1305 avec `k_req` et le nonce 0. Le pseudonyme signe l'enveloppe (`SealedJob`,
type `sjob`).

**Le pair** vérifie la signature du pseudonyme, que la cible est bien lui, que `expires_at` est dans la
fenêtre [maintenant − 120 s, maintenant + 600 s + 120 s], retrouve sa clé privée par `kx`, dérive les
mêmes clés, déchiffre, et seulement ensuite inscrit le job dans son cache anti-rejeu (jusqu'à
l'expiration ; plein, il refuse : il échoue fermé).

**La réponse.** Le pair met `{text, finish_reason, mean_logprob, compute_ms, completion_tokens,
attest}` en JSON, le complète à une puissance de deux et le coupe en morceaux de 64 Kio. Le morceau i
est chiffré avec `k_resp`, le nonce i et les données associées `{empreinte de l'en-tête du job, i,
final}` ; seul le dernier porte `final = vrai`. `attest` est la signature du pair (type `rdigest`) sur
`{job, pair, modèle, jetons, pseudonyme, SHA-256 du texte}` : une preuve, à l'intérieur du chiffrement,
de ce que le pair a répondu, que le demandeur peut montrer à un tiers. L'enveloppe `SealedResult`
{job_id, node_id, model, completion_tokens, chunks} est signée par le pair (type `sresult`) : le
traqueur règle les crédits sur ces champs en clair, bornés par `max_tokens` et la taille du chiffré.

**Le routage.** Avec un traqueur qui annonce `e2e`, un job routé commence par une trame `Reserve`
(rien sur le contenu) : le traqueur choisit un pair en appliquant la politique du demandeur
(`Route.deny`, `deny_families`, `only`, `swarm`, `e2e`, `min_rel`, `soft_avoid`) et les politiques de
service des pairs, réserve sa place, et renvoie `Assigned` avec le certificat du pair. La passerelle
revérifie ce pair contre la politique locale ; s'il est refusé, elle annule la réservation et demande
un remplaçant : **le pair refusé ne reçoit rien**. Sinon elle scelle le job pour lui. En mode annuaire,
l'annuaire est filtré par la même politique avant le choix.

**Reçus et crédits.** Le reçu reste signé par l'identité réelle du demandeur et n'est vu que du
traqueur. Les vues publiques du registre (`/v1/ledger`, `/v1/evidence`) ne nomment plus le demandeur et
ne publient qu'une empreinte du résultat.

**Compatibilité.** Un traqueur antérieur refuse un `Hello` portant `kx` : le nœud se reconnecte sans
(puis sans étiquettes). Une passerelle face à un traqueur sans `e2e` refuse d'envoyer quoi que ce soit
si « Exiger le chiffrement » est activé (par défaut) ; sinon elle retombe sur l'ancien chemin en clair.
Un pair sans clé n'est jamais choisi tant que le réglage est activé.

## 4. Arguments

### 4.1 Confidentialité face au traqueur

Le traqueur voit, pour un job : l'en-tête, la clé éphémère `eph`, la clé du pair `kx`, le chiffré, puis
l'enveloppe de la réponse (nombre de jetons, nombre de morceaux, tailles) et tous les instants. Notons
`L(job)` cette **fonction de fuite** : le compte payeur, le pair, l'en-tête, les classes de taille de la
question et de la réponse, le nombre de jetons déclaré, la fin (`finish_reason` n'en fait pas partie :
il est chiffré) et les instants d'envoi, de réponse et de reçu.

*Énoncé visé.* Pour deux exécutions du protocole dont les fuites `L` sont identiques, la vue du
traqueur (honnête mais curieux) est indistinguable, sous les hypothèses de la section 2.

*Esquisse.* La clé `e` est neuve à chaque job et indépendante de tout le reste. Sous gap-CDH, `s =
X25519(e, kx)` est imprévisible pour qui ne connaît ni `e` ni la clé privée du pair ; HKDF, modélisé
comme un extracteur (oracle aléatoire) puis une PRF, en fait des clés `k_req`, `k_resp`
indistinguables de clés aléatoires indépendantes. On remplace ces clés par des clés aléatoires (perte
négligeable), puis la sécurité IND-CPA de ChaCha20-Poly1305 rend chaque chiffré indistinguable du
chiffré d'un autre message de même longueur complétée ; les longueurs complétées font partie de `L`.
Aucun couple (clé, nonce) n'est réutilisé : une clé par sens, un nonce par morceau, des clés neuves par
job. Ce n'est qu'une esquisse, pas une preuve mécanisée ; elle ne dit rien de ce que `L` révèle (section
6), et l'argument « même longueur complétée » seul serait insuffisant sans le reste de `L`.

*Pas de confidentialité persistante face au pair.* La clé X25519 du pair vit jusqu'à 26 h : qui la
vole (en compromettant le pair) peut déchiffrer tous les jobs enregistrés qui l'ont utilisée. La clé
éphémère du demandeur ne protège pas contre la compromission du destinataire (limite classique de ce
type de construction, cf. HPKE, RFC 9180 §9.7.4). Un nœud qui redémarre tire de nouvelles clés et
oublie les anciennes.

### 4.2 Intégrité face à un traqueur actif

- *Modifier l'en-tête* (cible, limite de jetons, échéance) : il est signé par le pseudonyme et sert de
  données associées. Changer un champ casse la signature ; re-signer avec un autre pseudonyme change
  `requester_id`, donc le contexte HKDF et les données associées : le déchiffrement échoue (INT-CTXT).
- *Modifier, tronquer, réordonner la réponse* : chaque morceau est authentifié avec son rang et le
  drapeau `final` ; un morceau déplacé n'a pas le bon nonce, une réponse tronquée n'a pas de morceau
  final authentifié, un morceau d'un autre job a d'autres clés et une autre empreinte d'en-tête.
- *Rejouer* un job : même `job_id` → refusé par le cache tant que l'en-tête n'a pas expiré (cache gardé
  jusqu'à `expires_at` + 120 s) ; après → refusé par la fenêtre `expires_at` (échéance de 10 min au plus,
  plus 2 min de tolérance d'horloge). Un nœud qui redémarre perd son cache, mais aussi ses clés : un ancien
  job chiffré ne s'ouvre plus. Les jobs **en clair** (anciens pairs) n'ont pas d'expiration : le nœud
  retient leurs identifiants une heure.
- *Faire payer une réponse illisible* : un pair peut renvoyer une enveloppe bien signée mais
  indéchiffrable. Le demandeur conteste (trame `Dispute`) en révélant la clé éphémère **de ce job**
  (le traqueur peut alors lire ce job-là, et seulement lui) ; le traqueur vérifie qu'elle correspond à
  `eph`, dérive lui-même les clés et essaie d'ouvrir la réponse : si elle ne s'ouvre pas, elle n'est
  pas payée et compte contre le pair ; si elle s'ouvre, le demandeur paie. Une contestation n'est donc
  pas un simple refus déclaratif. Les canaris sont déchiffrés avant d'être payés.
- *Relabelliser un pair* : le certificat de clé engage aussi le modèle servi (champ `model` signé) ; une
  carte `Assigned` qui annonce un autre modèle est refusée avant tout envoi. La passerelle refuse aussi
  qu'un même pair (ou, pour l'essaim, une même famille encore active) soit assigné deux fois dans une
  requête : une réponse ne compte qu'une voix. Cela n'empêche pas un pair malveillant de servir un autre
  modèle que celui qu'il signe.
- *Substituer sa clé* : pour faire chiffrer vers une clé qu'il contrôle sous l'identité du pair `P`, le
  traqueur doit produire un `KxCert` valide pour `P`, donc une signature ed25519 de `P` (EUF-CMA), ou
  une clé publique ed25519 dont le hash tronqué vaut l'identifiant de `P` (seconde préimage sur 128
  bits). Les tests couvrent les trois variantes (sa clé et sa signature ; sa clé relabellisée ; la
  signature de `P` sur une autre clé).
- *Forger une réponse* : l'enveloppe est signée par le pair et l'attestation intérieure aussi.

Ce qu'un traqueur malveillant **peut** faire : refuser de relayer, retarder, mentir sur la fiabilité ou
l'annuaire, et surtout **choisir les pairs**, y compris des nœuds à lui (le chiffrement protège alors
la question jusqu'à… lui). Parades : « Seulement mes nœuds de confiance » (identifiants ou codes
d'invitation), essaim privé (seuls les nœuds qui prouvent la connaissance du secret, preuve
`HMAC(K, "member" ‖ node_id)` vérifiée localement, non transférable à un autre identifiant), liste de
blocage, et la revérification locale de chaque pair assigné.

### 4.3 Non-associabilité (« qui » séparé de « quoi »)

Le pair ne voit qu'un pseudonyme neuf par job ; le traqueur sait qui paie mais pas ce qui est demandé.
Comme pour Oblivious HTTP, il faut la **collusion** d'un traqueur et d'un pair pour réunir les deux,
**à condition que le traqueur ne publie rien qui relie un job à son payeur** : les soldes par compte
(`/v1/balance/{id}`, les statistiques par compte de `/v1/stats`) ne sont plus publics (requête signée
par le titulaire), `/v1/balances` ne donne plus que des totaux, le registre public ne nomme plus le
demandeur et les preuves publiques ne contiennent plus qu'une empreinte. Sans cela, un pair pouvait
retrouver le payeur en comparant les soldes avant et après le règlement du job qu'il a servi (relevé par
la revue Codex). Limites : le traqueur et le pair peuvent rapprocher leurs journaux par l'instant et l'identifiant du
job ; les k pairs d'une même requête reçoivent la même question au même moment (deux pairs complices
savent qu'ils servent la même personne, sans savoir qui) ; une conversation longue envoyée en entier à
chaque tour permet à un pair de relier les tours (d'où le plafond d'historique, 8 tours par défaut, et la
rotation des pairs) ; le chemin en clair des anciens pairs, quand l'utilisateur l'autorise, révèle
l'identité réelle ; les preuves d'essaim privé, publiques, révèlent qui appartient au même essaim.
Les limites de service par demandeur (débit, simultanéité, liste de blocage) ne peuvent plus être
appliquées par le pair, qui ne voit que des pseudonymes : il les confie au traqueur (`ServePolicy`), qui
les applique par compte payeur.

### 4.4 Ce qu'on fait contre un pair qui lit

1. *Moins à lire* : masquage local réversible (`CLE_1`, `EMAIL_1`, `PERSONNE_1`, `TERME_1`…) des clés
   d'API, clés privées, cartes (Luhn), IBAN (mod 97), e-mails, téléphones, chemins et noms d'utilisateur,
   noms (heuristique), termes définis par l'utilisateur ; la table de correspondance ne quitte jamais
   la machine et la réponse est rétablie localement. Un second détecteur (modèle local, module
   optionnel `privacy_model`) peut s'y brancher : union des passages, décision la plus prudente, et une
   erreur du détecteur compte comme sensible (échec fermé). Plafond d'historique ; sous-agents : chaque
   pair ne reçoit que sa sous-tâche et ses extraits, l'orchestrateur seulement des étiquettes et de
   courts extraits (`planner_context_chars`, 600 par défaut).
2. *Rien à lire* : mode local, règles « toujours local » par type de contenu (la question est traitée
   par le modèle de la machine, ou refusée), confirmation la première fois qu'un type sensible partirait
   non masqué, indicateur de sensibilité avant l'envoi (ce qui est détecté, masqué, et qui lira).
3. *Moins de pairs qui lisent la même personne* : rotation (les pairs récents sont évités quand c'est
   possible), pseudonymes.
4. *Les pairs honnêtes ne gardent rien* : le contenu n'existe qu'en mémoire ; le moteur reçoit
   `cache_prompt: false` (pas de réutilisation du cache KV d'un autre demandeur), llama-server tourne
   avec `-cram 0` (pas de cache de prompts en mémoire hôte partagé) et `-lv 2` (journal limité aux
   avertissements et erreurs) ; après chaque job, les emplacements inactifs sont effacés
   (`POST /slots/{id}?action=erase`) ; `-lm mmap+mlock` garde le modèle en RAM quand elle suffit
   (réglage `mlock`, automatique). Les réponses gardées en local passent par le même effacement. Ces
   mesures sont « au mieux » : un llama-server qui ne connaît pas ces options est relancé sans elles
   (visible dans l'état du moteur : `hardened: false`), un effacement qui échoue n'arrête pas le service.
   Les erreurs renvoyées pour un job chiffré sont des codes, jamais un message du moteur. Aucun journal ne contient de texte de job (test : un marqueur unique n'apparaît ni
   dans les journaux ni dans aucun fichier écrit, base du traqueur comprise). Python ne permet pas
   d'effacer sûrement la mémoire : un vidage mémoire ou un fichier d'échange peut encore contenir un
   texte récent.
5. *Dissuader* : les jetons-pièges (section 4.6).

Aucune de ces mesures n'empêche un pair malveillant de garder ce qu'il a déchiffré.

### 4.5 Canaris indiscernables et taux d'audit effectif

Le traqueur ne peut plus dupliquer les jobs chiffrés pour les contrôler. Il envoie ses propres jobs :
après un vrai job chiffré relayé à un pair, avec probabilité `q` (`spot_rate`), il programme, après un
délai aléatoire (0,5 à 8 s), une question à réponse vérifiable (calcul) au même pair et à un pair servant
le même fichier de modèle, chiffrée exactement comme un vrai job : même trame, même en-tête (limite de
jetons et échéance recopiées du vrai job), même classe de taille (complétée jusqu'à celle du vrai job),
pseudonyme neuf (comme tout vrai job : l'identifiant du demandeur ne porte aucune information),
réglée dans le registre public avec le même type (`receipt`) qu'un vrai job. Il déchiffre les deux
réponses et les compare comme les anciens contrôles (réponses extraites ; une réponse tronquée,
`finish_reason = length`, est non concluante) ; les compteurs alimentent la même réputation (taux de
désaccord honnête α mesuré par modèle).

Le théorème d'incitation de l'article (`paper/main.tex`, théorème *Honest serving*) dit que servir
honnêtement est une meilleure réponse si et seulement si `q(β_d − α)(P + R) ≥ g_H − g_d` pour toute
déviation d, pour les utilités écrites dans l'article. Ce `q` est la probabilité qu'un job **sur lequel
le pair a choisi de tricher** soit audité, et le reste de la condition suppose une rémunération
comparable entre stratégies, un audit pertinent et une sanction effective.

*Ce que l'indiscernabilité apporte, exactement.* Notons `π` la proportion de canaris parmi les jobs que
voit un pair, et D son meilleur test « ce job est un canari », de taux de vrais positifs `t` et de faux
positifs `u`. S'il ne triche que sur les jobs que D juge réels, la probabilité qu'un job sur lequel il
triche soit un canari vaut
`Pr[canari | D = réel] = π(1 − t) / (π(1 − t) + (1 − π)(1 − u))`.
Elle vaut `π` quand D ne fait pas mieux que le hasard (`t = u`), et tend vers 0 quand `t → 1` : un pair
qui reconnaît tous les canaris n'est jamais puni, quels que soient `q` et `P`. (Le taux effectif n'est
donc pas simplement `q(1 − TPR)`, comme le disait une version précédente de ce document.) Si les
canaris sont indiscernables de la vue du pair, et si les audits sont indépendants, un pair qui triche sur
exactement m jobs est pris au moins une fois avec probabilité `1 − (1 − π β_d)^m` ; la forme
`1 − (1 − q f β_d)^n` sur n jobs dont une fraction f triche en est un cas particulier sous les mêmes
hypothèses. Il faut aussi distinguer les **tentatives** d'audit (`spot_rate`), les audits **livrés**
et **concluants**, et les **sanctions** : un audit reporté, sans contrôleur ou non concluant ne compte
pas. Les compteurs `canary_stats` du traqueur les séparent (envoyés, reportés, abandonnés, sans
contrôleur, concluants, non concluants, illisibles).

*Ce qui est indiscernable, et ce qui ne l'est pas.* Par construction (et vérifié par les tests en
comparant les trames) : la forme de la trame, les champs en clair, la taille, l'identifiant de
demandeur (un pseudonyme neuf, comme pour tout vrai job), la façon d'être réglé. **Le contenu ne l'est
pas** : les questions viennent d'une douzaine de gabarits publics (`canary.py`), qu'un pair peut
reconnaître exactement et auxquels il peut répondre juste (même sans modèle) tout en trichant ailleurs ;
les jetons-pièges (`/h/<24 hex>`, `myr_live_…`) sont eux-mêmes des indices. **Aujourd'hui, les canaris
ne dissuadent donc qu'un tricheur non adaptatif** ; l'efficacité contre un pair qui cible les gabarits
reste à établir (mesurer des classifieurs adverses sur la vue complète du pair, générer des canaris plus
variés). Ensuite, les canaris n'auditent que la classe de jobs auditable (gloutons, réponse extractible :
théorème *Duplicate-and-compare fails*) : **un pair qui ne triche que sur les jobs à température non
nulle ou à texte libre n'est jamais pris par un canari** ; pour ceux-là, seuls la fusion des réponses,
la quarantaine locale et les signalements protègent.

*Contournements fermés.* Un pair occupé au moment du canari ne l'évite plus : l'audit est reporté (4
fois au plus) au lieu d'être abandonné. Un pair sans contrôleur (fichier de modèle unique, ou `gguf`
déclaré exprès) est jugé sur la réponse connue de la question. Une fin « tronquée » (`length`)
n'est crue que pour une réponse au moins aussi longue que sa limite de jetons le permet (un caractère par
jeton, plafonné à 256), et tous les trois audits rendus non concluants par troncature, l'un compte comme
un désaccord pour ce pair. Une réponse de canari illisible n'est pas payée.

*Limites qui restent.* Deux pairs complices peuvent répondre la même constante fausse et se valider
mutuellement (la réponse connue n'est utilisée que sans contrôleur, pour ne pas confondre l'exactitude
d'un petit modèle avec son honnêteté) ; des identités complices peuvent fausser la mesure de α ; la
perte de réputation se contourne en changeant d'identité. La proportion de canaris parmi les jobs d'un
pair est d'environ `q/(1+q)` pour son rôle de cible, plus son rôle de contrôleur : `π` est de l'ordre de
`spot_rate`, sous réserve des audits reportés ou abandonnés.

### 4.6 Jetons-pièges

Une partie des canaris (la moitié) porte des détails faux, uniques et propres à chaque pair : une
adresse e-mail et une URL sur le domaine du traqueur (`/h/<jeton>`), parfois une chaîne qui ressemble à
une clé d'API. Le traqueur note quel pair a reçu quel jeton. Si un jeton réapparaît ailleurs (quelqu'un
le signale par `POST /v1/honeytoken`, ou visite l'URL `/h/<jeton>`), le pair qui l'a reçu est
identifié et banni. C'est une **détection après coup**, pas une prévention, et elle ne vaut que pour ce
qui est publié ou utilisé. Elle a un coût : un détail trop voyant trahit le canari. Les e-mails et URL
ne sont pas masqués par défaut dans les vraies questions, donc on en trouve aussi dans de vrais jobs ;
une vraie clé d'API, elle, est masquée par défaut, si bien qu'une clé en clair signalerait un canari :
d'où leur rareté (5 %). Seuls le traqueur et le pair connaissent un jeton : un traqueur malhonnête
pourrait accuser un pair à tort ; la sanction est un bannissement par ce traqueur, levable par
l'administrateur (`myriad tracker-bans --db tracker.sqlite --remove <nœud>`). Un pair averti peut aussi
retirer de ce qu'il publie tout ce qui ressemble à un jeton-piège.

## 5. Outils de protection

- **Liste de blocage** (identifiants, avec raison et date ; familles de modèles) : jamais choisi comme
  pair, ni par le traqueur (politique transmise) ni localement ; refusé comme demandeur (par le nœud
  pour les jobs en clair, par le traqueur pour les jobs chiffrés). Bouton « Bloquer ce nœud » sur les
  cartes de pair.
- **Nœuds de confiance** (identifiants ou codes d'invitation `myr1-…` avec somme de contrôle) et
  **essaim privé** (le secret partagé ne quitte jamais la machine : seule `K = scrypt(secret)` est
  gardée, le traqueur ne voit que `HMAC(K, "group")`, scrypt rendant coûteuse la recherche d'un secret
  court à partir de ce tag public).
- **Fiabilité minimale** des modèles ; **quarantaine locale** des pairs qui contredisent souvent une
  majorité certifiée (au moins 4 désaccords pondérés et 60 % de désaccords), envoient des réponses
  indéchiffrables ou mal signées, ou ont une réputation basse (canaris ratés) ; compteurs à demi-vie de
  6 h, quarantaine de 30 min doublée à chaque récidive (24 h au plus), jamais définitive sans action de
  l'utilisateur.
- **Mode local**, **garde de confidentialité** (section 4.4), **limites de service** (débit et jobs
  simultanés par demandeur, taille maximale des questions, jetons de sortie plafonnés).
- **Signalements** signés (`POST /v1/report`) : le poids d'un signaleur vaut
  `min(1, âge du compte / 7 j) × min(1, crédits dépensés chez les autres / 50)`, un compte neuf pèse 0 ;
  un désaccord ne se signale que sur un job payé à ce nœud ; au plus 20 signalements par heure et par
  compte ; un nœud n'est **dépriorisé** qu'à partir d'un score de 3 venant d'au moins 3 signaleurs
  pondérés, et **jamais banni** par des signalements. Les bannissements viennent du fichier de
  l'administrateur (`--ban-file`) et des jetons-pièges.

## 6. Ce qui reste visible

Pour le traqueur : le compte qui paie chaque job, le pair choisi, les instants (envoi, réponse,
reçu), les classes de taille du job et de la réponse, la limite et le nombre de jetons, l'échéance, le
nombre de pairs par requête (même groupe), les accords et désaccords déclarés dans les reçus, les
signalements. Pour le pair : la question (masquée), les paramètres d'échantillonnage, l'instant. Pour un
observateur réseau : les adresses IP et les volumes. Les canaris sont visibles du traqueur seul.

## 7. Coût mesuré

`app/bench/bench_crypto.py` (Windows, i5-10400F, Python 3.12), résultats dans
`app/bench/results/crypto.json` :

| question / réponse (caractères) | chiffré (base64) | CPU par job (sceller + ouvrir, aller et retour) |
| --- | --- | --- |
| 300 / 600 | 1,4 ko | 2,1 ms |
| 4 000 / 2 000 | 10,9 ko | 2,6 ms |
| 30 000 / 8 000 | 43,7 ko | 4,4 ms |
| 120 000 / 60 000 | 175 ko | 12,7 ms |

Latence d'une requête dans un essaim local (traqueur, 4 pairs à moteur instantané et passerelle dans un
même processus), médiane :

| | k = 1 | k = 4 |
| --- | --- | --- |
| en clair (protocole essaim/1.2) | 6,9 ms | 25,7 ms |
| chiffré (essaim/1.3) | 10,5 ms | 41,7 ms |
| en clair, WAN émulé 25 ms aller | 126 ms | 154 ms |
| chiffré, WAN émulé 25 ms aller | 190 ms | 218 ms |

Le surcoût vient surtout de la réservation (un aller-retour demandeur–traqueur de plus, ≈ 50 ms ici)
et, en local, du travail cryptographique fait par les cinq rôles dans un seul processus. Face à une
génération de plusieurs secondes, il est négligeable.

## 8. Limites et suites

- Le pair lit la question ; la collusion traqueur–pair réunit qui et quoi ; le traqueur choisit les
  pairs (parades : confiance, essaim privé, blocage).
- Les canaris ne couvrent que les jobs gloutons à réponse extractible ; leur contenu n'est
  indiscernable qu'heuristiquement.
- La détection des données sensibles est heuristique (les noms surtout) ; un marqueur peut être perdu
  ou déformé par le modèle ; masquer ne cache pas ce que dit le reste du texte.
- Les limites de service par demandeur reposent sur le traqueur pour les jobs chiffrés.
- **TLS est nécessaire** (`https://`) pour un traqueur hors de la machine ou du réseau local : le
  chiffrement de bout en bout protège le contenu, pas la session ; sur un lien `ws://`, un attaquant
  actif pourrait détourner la connexion authentifiée et dépenser le compte (le nœud l'écrit dans son
  journal).
- Crédits : un résultat livré engage son coût exact jusqu'au règlement, mais les jobs **en vol** ne
  réservent rien ; ce qu'un compte peut devoir est borné par une rafale de `max_inflight` jobs à leur
  coût maximal. Réserver le coût maximal à l'admission refuserait les requêtes ordinaires d'un compte
  neuf (k = 4 jobs de 512 jetons dépassent déjà le crédit de départ) ; le choix (crédit de départ,
  `max_inflight` plus bas pour les comptes neufs) est laissé au propriétaire.
- Les reçus (`agreed`) et les signalements restent manipulables par des comptes complices qui se paient
  entre eux : la fiabilité publiée mesure une satisfaction déclarée, pas un calcul vérifié.
- L'article (`paper/main.tex`) décrit encore les contrôles par duplication de la v1 ; le chemin chiffré
  essaim/1.3 les remplace par des canaris.
- Le cache anti-rejeu est en mémoire (fenêtre de rejeu bornée par l'échéance après un redémarrage).
- Python ne garantit pas l'effacement de la mémoire ; `mlock` dépend des droits du système.
- Suites possibles : relais à plusieurs sauts (le traqueur ne verrait plus le couple payeur–pair),
  jetons de paiement aveugles (le traqueur ne saurait plus qui paie quoi), calcul confidentiel quand
  les GPU l'offriront, génération de canaris par un modèle pour mieux imiter les vrais jobs.

## 9. Revue Codex du protocole et de ces arguments

Rapport brut : `audits/2026-10-09_audit_securite_protocole.md` (18 points). Chaque point a été vérifié ;
voici ce qui en a été fait.

| # | Point | Suite |
| --- | --- | --- |
| 1 | Les soldes publics relient un job à son payeur | corrigé : soldes et statistiques par compte sur requête signée, `/v1/balances` agrégé |
| 2 | Une réponse illisible est payée | corrigé : contestation vérifiable (`Dispute`), canaris déchiffrés avant paiement |
| 3 | Pas de réservation du coût maximal à l'admission | rejeté pour l'instant (section 8) : le coût exact des résultats livrés est engagé ; la rafale en vol reste bornée par `max_inflight` |
| 4 | Un chevauchement de détections masque une règle plus stricte | corrigé : décision sur toutes les détections, la règle la plus stricte gagne |
| 5 | L'échec du détecteur externe contournable sur plusieurs messages | corrigé : un échec sur n'importe quel message demande une confirmation |
| 6 | Canaris reconnaissables par leur contenu | accepté comme limite : l'affirmation est restreinte au niveau des trames (4.5) |
| 7 | Un pair occupé évite ses canaris ; `gguf` unique | corrigé : audit reporté ; réponse connue sans contrôleur |
| 8 | `length` neutralise un audit | corrigé : troncature crue seulement si crédible, non-concluants comptés |
| 9 | Réponse connue ignorée ; α empoisonnable | partiellement : réponse connue sans contrôleur ; collusion et α documentés (4.5) |
| 10 | Le traqueur peut relabelliser le modèle ; un pair compté deux fois | corrigé : modèle signé dans le certificat ; unicité des pairs et des familles |
| 11 | Le chemin en clair contourne la politique ; doublons de contrôle | corrigé : sélection locale si la politique ne peut pas s'appliquer avant l'envoi ; pas de duplication pour un demandeur qui annonce `e2e` |
| 12 | Reçus et signalements manipulables par des complices | accepté comme limite (section 8) |
| 13 | Corps HTTP du traqueur lus en entier ; historique des pseudonymes | corrigé : lecture bornée ; limites par pseudonyme supprimées |
| 14 | Durcissement du moteur présenté trop fortement | corrigé en partie : effacement aussi en local ; « au mieux » documenté, état visible |
| 15 | Détournement de session sans TLS | documenté (section 8) et signalé dans le journal du nœud |
| 16 | Formules d'incitation et de détection | corrigé dans ce document (4.5) |
| 17 | Fonction de fuite ; pas de confidentialité persistante | corrigé dans ce document (4.1) |
| 18 | Validation des certificats au traqueur ; bornes de rejeu ; rejeu en clair | corrigé : même validation partout ; bornes précisées ; rejeu en clair refusé par le nœud |
